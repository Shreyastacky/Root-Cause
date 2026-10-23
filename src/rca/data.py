"""Read the observed RCAEval Parquet layout; keep labels outside model inputs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/raw"


@dataclass(frozen=True)
class Case:
    case_id: str
    directory: Path
    inject_time: int
    root_cause_service: str
    fault_type: str
    system: str


def list_cases(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[Case]:
    """Select RE1-OB from the official index, never from a guessed directory list."""
    data_dir = Path(data_dir)
    index_file = data_dir / "cases.parquet"
    if not index_file.is_file():
        raise FileNotFoundError("Run python -m rca.download to obtain cases.parquet first.")
    index = pd.read_parquet(index_file)
    selected = index.loc[index["dataset"].eq("RE1-OB")].sort_values("case")
    if selected.empty or selected["case"].duplicated().any():
        raise ValueError("No unambiguous RE1-OB cases in the official index.")
    cases = []
    for row in selected.itertuples(index=False):
        directory = data_dir / row.case
        for filename in ("metrics.parquet", "inject_time.txt"):
            if not (directory / filename).is_file():
                raise FileNotFoundError(f"Missing {directory / filename}; download is incomplete.")
        cases.append(Case(str(row.case), directory, int(row.inject_time),
                          str(row.root_cause_service), str(row.fault), str(row.system_name)))
    return cases


def parse_metric_column(column: str) -> tuple[str, str]:
    """Observed examples: adservice_cpu, frontend_latency-90, main_mem."""
    if "_" not in column:
        raise ValueError(f"Unexpected metric column without service separator: {column!r}")
    service, metric_name = column.rsplit("_", 1)
    if not service or not metric_name:
        raise ValueError(f"Invalid metric column: {column!r}")
    return service, metric_name


def prepare_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Create a numeric timestamp index and (service, metric) column index.

    NaN is retained; +/- infinity becomes NaN. The baseline computes each
    window's statistics from observed values, without cross-window imputation.
    """
    if raw.columns.duplicated().any():
        raise ValueError("Duplicate raw metric columns need explicit inspection.")
    if "time" not in raw.columns:
        raise ValueError("The inspected dataset requires an explicit 'time' column.")
    timestamps = pd.to_numeric(raw["time"], errors="raise")
    if not np.isfinite(timestamps).all() or timestamps.duplicated().any():
        raise ValueError("Timestamps must be finite and unique.")
    values = raw.drop(columns="time").apply(pd.to_numeric, errors="raise").astype(float)
    pairs = [parse_metric_column(str(name)) for name in values.columns]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Metric parsing produced duplicate (service, metric) pairs.")
    values.columns = pd.MultiIndex.from_tuples(pairs, names=["service", "metric"])
    values.index = pd.Index(timestamps.to_numpy(), name="time")
    inf_count = int(np.isinf(values.to_numpy()).sum())
    missing_count = int(values.isna().sum().sum())
    values = values.replace([np.inf, -np.inf], np.nan).sort_index()
    values.attrs["cleaning"] = {"raw_missing_values": missing_count,
                                "nonfinite_values_changed_to_nan": inf_count,
                                "imputation": "none; per-window available observations"}
    return values


def load_case(case: Case | str, data_dir: Path | str = DEFAULT_DATA_DIR
              ) -> tuple[pd.DataFrame, int, str, str, str]:
    """Return metrics, injection time, root service, fault type, and system name.

    The metrics frame contains no label, case identifier, or fault metadata.
    Use frame.xs(service, axis=1, level='service') for one service's metrics.
    """
    if isinstance(case, str):
        matches = [item for item in list_cases(data_dir) if item.case_id == case]
        if not matches:
            raise KeyError(f"Case {case!r} is not in the selected RE1-OB subset.")
        case = matches[0]
    inject_time = int((case.directory / "inject_time.txt").read_text().strip())
    if inject_time != case.inject_time:
        raise ValueError(f"Index/file injection time mismatch for {case.case_id}.")
    metrics = prepare_metrics(pd.read_parquet(case.directory / "metrics.parquet"))
    return metrics, inject_time, case.root_cause_service, case.fault_type, case.system


def service_names(metrics_df: pd.DataFrame) -> list[str]:
    """Keep every observed prefix, including infrastructure-like candidates."""
    return sorted(metrics_df.columns.get_level_values("service").unique().tolist())
