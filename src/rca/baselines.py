"""A deliberately simple, deterministic mean-shift baseline with no labels."""

import numpy as np
import pandas as pd

from rca.data import prepare_metrics


EPS = 1e-8


class InsufficientWindowError(ValueError):
    """The supplied recording has no usable before/after comparison."""


def metric_anomaly_scores(metrics_df: pd.DataFrame, inject_time: int,
                          eps: float = EPS) -> pd.DataFrame:
    """Compute |post mean - pre mean| / (pre sample std + eps) per metric.

    pre: time < injection; post: time >= injection. No training or root label
    is required. Missing observations are ignored separately in each window.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    frame = metrics_df if isinstance(metrics_df.columns, pd.MultiIndex) else prepare_metrics(metrics_df)
    values = frame.replace([np.inf, -np.inf], np.nan)
    before = values.loc[values.index < inject_time]
    after = values.loc[values.index >= inject_time]
    if len(before) < 2 or after.empty:
        raise InsufficientWindowError(
            f"Need >=2 pre and >=1 post rows; observed pre={len(before)}, post={len(after)}.")
    stats = pd.DataFrame({"normal_mean": before.mean(), "anomalous_mean": after.mean(),
                          "normal_std": before.std(ddof=1),
                          "normal_count": before.count(), "anomalous_count": after.count()})
    valid = (stats["normal_count"] >= 2) & (stats["anomalous_count"] >= 1)
    stats["z"] = ((stats["anomalous_mean"] - stats["normal_mean"]).abs()
                  / (stats["normal_std"] + eps)).where(valid)
    stats["valid"] = valid
    return stats


def zscore_service_scores(metrics_df: pd.DataFrame, inject_time: int,
                          skip_constant: bool = False) -> pd.DataFrame:
    """Maximum metric score per service; alphabetical ordering breaks exact ties.

    skip_constant=True ignores metrics whose normal window has zero standard
    deviation. Their z-score is a change divided by eps (1e-8), so even a tiny
    blip, such as 0 -> 0.002 errors, would otherwise outrank a real CPU spike.
    """
    stats = metric_anomaly_scores(metrics_df, inject_time).reset_index()
    rows = []
    for service, group in stats.groupby("service", sort=True):
        observed = group.loc[group["valid"] & group["z"].notna()]
        if skip_constant:
            observed = observed.loc[observed["normal_std"] > 0]
        if observed.empty:
            score, winner = -np.inf, ""
        else:
            best = observed.sort_values(["z", "metric"], ascending=[False, True]).iloc[0]
            score, winner = float(best["z"]), str(best["metric"])
        rows.append({"service": service, "score": score, "winning_metric": winner,
                     "n_metrics": len(group), "n_valid_metrics": len(observed)})
    return pd.DataFrame(rows).sort_values(["score", "service"], ascending=[False, True]).reset_index(drop=True)


def zscore_rank(metrics_df: pd.DataFrame, inject_time: int) -> list[str]:
    """Rank all observed services without reading root cause or fault labels."""
    return zscore_service_scores(metrics_df, inject_time)["service"].tolist()


def zscore_nonconst_rank(metrics_df: pd.DataFrame, inject_time: int) -> list[str]:
    """Same rule, but metrics that were constant before injection are ignored."""
    return zscore_service_scores(metrics_df, inject_time, skip_constant=True)["service"].tolist()


# Method name -> (ranker, short description); used by evaluation and reports.
METHODS = {
    "zscore": (zscore_rank, "max |mean shift| / (pre std + 1e-8) over all metrics"),
    "zscore_nonconst": (zscore_nonconst_rank,
                        "same, ignoring metrics with zero pre-injection std"),
}
