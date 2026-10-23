"""Service-level localization metrics, using one ground-truth service per case."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def rank_of_root(ranking: Sequence[str], true_root: str) -> float:
    """Return the one-based rank; a missing root receives infinity (a miss)."""
    if len(ranking) != len(set(ranking)):
        raise ValueError("A service ranking must not contain duplicate services.")
    return float(ranking.index(true_root) + 1) if true_root in ranking else np.inf


def ac_at_k(ranks: Sequence[float], k: int) -> float:
    """Fraction of cases whose annotated root is among the first k services."""
    values = np.asarray(ranks, dtype=float)
    if k < 1 or values.size == 0:
        raise ValueError("k must be positive and at least one case is required.")
    if np.isnan(values).any() or (values < 1).any():
        raise ValueError("Ranks must be one-based numbers, or infinity for misses.")
    return float(np.mean(values <= k))


def avg_at_k(ranks: Sequence[float], k: int = 5) -> float:
    """Mean of AC@1 through AC@k; NOT mean reciprocal rank or average precision."""
    if k < 1:
        raise ValueError("k must be positive.")
    return float(np.mean([ac_at_k(ranks, cutoff) for cutoff in range(1, k + 1)]))


def summarize(per_case: pd.DataFrame) -> pd.DataFrame:
    """Compute an overall row and rows grouped by fault type from the same cases."""
    if per_case.empty:
        raise ValueError("Cannot summarize an empty evaluation.")
    rows = []
    sets = [("all_cases", per_case)]
    if "status" in per_case:
        sets.append(("evaluable_only", per_case.loc[per_case["status"].eq("ok")]))
    for evaluation_set, subset in sets:
        if subset.empty:
            continue
        groups = [("overall", "all", subset)]
        groups += [("fault_type", str(fault), frame)
                   for fault, frame in subset.groupby("fault_type", sort=True)]
        for scope, fault, frame in groups:
            ranks = frame["rank_of_true_root"].to_numpy(dtype=float)
            n_evaluable = int(frame["status"].eq("ok").sum()) if "status" in frame else len(frame)
            row = {"evaluation_set": evaluation_set, "scope": scope, "fault_type": fault,
                   "n_cases": len(frame), "n_evaluable": n_evaluable}
            row.update({f"AC@{k}": ac_at_k(ranks, k) for k in range(1, 6)})
            row["Avg@5"] = avg_at_k(ranks)
            rows.append(row)
    return pd.DataFrame(rows)
