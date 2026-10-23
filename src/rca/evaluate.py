"""Evaluate every selected case and save reproducible service-level results."""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from rca.baselines import EPS, METHODS, InsufficientWindowError
from rca.data import DEFAULT_DATA_DIR, PROJECT_ROOT, list_cases, load_case, service_names
from rca.metrics import rank_of_root, summarize


def run_evaluation(data_dir: Path | str = DEFAULT_DATA_DIR,
                   results_dir: Path | str = PROJECT_ROOT / "results",
                   method: str = "zscore") -> tuple[pd.DataFrame, pd.DataFrame]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHODS)}.")
    ranker, description = METHODS[method]
    cases = list_cases(data_dir)
    rows = []
    for case in cases:
        metrics, injection, root, fault, system = load_case(case)
        # Labels only enter AFTER prediction, for scoring; never pass Case to the ranker.
        try:
            ranking = ranker(metrics, injection)
            if set(ranking) != set(service_names(metrics)):
                raise ValueError(f"Incomplete candidate ranking for {case.case_id}.")
            status, error = "ok", ""
            rank = rank_of_root(ranking, root)
        except InsufficientWindowError as exc:
            # Explicitly record invalid-window cases instead of inventing a ranking.
            ranking, rank, status, error = [], np.inf, "invalid_window", str(exc)
        rows.append({"case_id": case.case_id, "system": system, "fault_type": fault,
                     "true_root": root, "predicted_ranking": json.dumps(ranking),
                     "rank_of_true_root": rank, "n_candidates": len(service_names(metrics)),
                     "status": status, "error": error})
    # Other loading/prediction errors stop the run; no silently omitted cases.
    per_case = pd.DataFrame(rows)
    summary = summarize(per_case)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(results_dir / f"{method}_per_case.csv", index=False)
    summary.to_csv(results_dir / f"{method}_summary.csv", index=False)
    protocol = {"method": method, "n_cases": len(cases), "subset": "RE1-OB",
                "prediction_inputs": ["metrics", "known injection timestamp"],
                "normal_window": "all available time < inject_time",
                "anomalous_window": "all available time >= inject_time",
                "std_ddof": 1, "eps": EPS,
                "missing_values": "per-window observed values; no imputation",
                "rule": description,
                "constant_normal_metrics": "ignored" if method == "zscore_nonconst" else "kept (std + eps)",
                "service_aggregation": "maximum metric z-score",
                "candidate_policy": "all observed metric prefixes; no root-label filtering",
                "ties": "alphabetical service name", "training": "none",
                "n_evaluable": int(per_case["status"].eq("ok").sum()),
                "n_invalid_window": int(per_case["status"].ne("ok").sum()),
                "invalid_window_policy": "empty ranking; infinity rank; miss in all_cases; separately report evaluable_only",
                "note": "Full-record offline diagnosis, not online detection."}
    (results_dir / f"{method}_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return per_case, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), default="zscore")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    run_evaluation(args.data_dir, args.results_dir, args.method)


if __name__ == "__main__":
    main()
