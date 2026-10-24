"""Fixed per-service change features, without incident or service identities."""

import numpy as np
import pandas as pd

from rca.baselines import EPS, InsufficientWindowError
from rca.data import prepare_metrics, service_names


METRIC_TYPES = ("cpu", "mem", "load", "workload", "error", "latency", "latency-50", "latency-90")
STATISTICS = ("available", "signed_log_shift", "relative_shift", "log_std_ratio",
              "normal_constant", "zero_to_nonzero")
FEATURE_COLUMNS = [f"{metric}__{stat}" for metric in METRIC_TYPES for stat in STATISTICS]


def extract_service_features(metrics_df: pd.DataFrame, inject_time: int) -> pd.DataFrame:
    """Return one 48-feature row per service; the service name is only an index.

    Each incident is transformed independently. There is no dataset-wide fitting,
    imputation, label access, or file/metadata access in this function.
    """
    frame = metrics_df if isinstance(metrics_df.columns, pd.MultiIndex) else prepare_metrics(metrics_df)
    before_mask = frame.index < inject_time
    if before_mask.sum() < 2 or (~before_mask).sum() < 1:
        raise InsufficientWindowError("Feature extraction needs >=2 normal and >=1 anomalous rows.")
    unknown = set(frame.columns.get_level_values("metric")) - set(METRIC_TYPES)
    if unknown:
        raise ValueError(f"Inspect new metric types before extending the fixed schema: {sorted(unknown)}")
    output = pd.DataFrame(0.0, index=pd.Index(service_names(frame), name="service"),
                          columns=FEATURE_COLUMNS)
    for (service, metric), series in frame.items():
        before = series.loc[before_mask].to_numpy(dtype=float)
        after = series.loc[~before_mask].to_numpy(dtype=float)
        before, after = before[np.isfinite(before)], after[np.isfinite(after)]
        if len(before) < 2 or len(after) < 1:
            continue
        pre_mean, post_mean = float(before.mean()), float(after.mean())
        pre_std = float(before.std(ddof=1))
        post_std = float(after.std(ddof=1)) if len(after) > 1 else 0.0
        difference = post_mean - pre_mean
        standardized = min(abs(difference) / (pre_std + EPS), 1e12)
        values = (1.0, float(np.sign(difference) * np.log1p(standardized)),
                  difference / (abs(pre_mean) + abs(post_mean) + EPS),
                  float(np.clip(np.log((post_std + EPS) / (pre_std + EPS)), -20, 20)),
                  float(pre_std == 0), float(np.all(before == 0) and np.any(after != 0)))
        for stat, value in zip(STATISTICS, values):
            output.loc[service, f"{metric}__{stat}"] = value
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError("Nonfinite feature encountered; inspect the recording.")
    return output
