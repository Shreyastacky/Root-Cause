"""Small, reusable EDA and plotting helpers used by both notebooks."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rca.baselines import InsufficientWindowError, metric_anomaly_scores, zscore_service_scores
from rca.data import DEFAULT_DATA_DIR, PROJECT_ROOT, list_cases, load_case, service_names


ROOT_COLOR = "#6a3f9b"
VICTIM_COLOR = "#cf671c"
RESULTS_DIR = PROJECT_ROOT / "results"


def configure_plots() -> None:
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.grid": True,
                         "grid.alpha": 0.18, "figure.facecolor": "white"})


def audit_dataset(data_dir: Path = DEFAULT_DATA_DIR,
                  results_dir: Path = RESULTS_DIR) -> tuple[pd.DataFrame, dict]:
    """Inspect all actual files, not just index statistics."""
    records, all_services, metric_names = [], set(), set()
    for case in list_cases(data_dir):
        metrics, injection, root, fault, _ = load_case(case)
        raw = pd.read_parquet(case.directory / "metrics.parquet")
        names = service_names(metrics)
        all_services.update(names)
        metric_names.update(metrics.columns.get_level_values("metric"))
        normal_rows = int((metrics.index < injection).sum())
        anomalous_rows = int((metrics.index >= injection).sum())
        records.append({"case_id": case.case_id, "fault_type": fault, "true_root": root,
                        "inject_time": injection, "time_start": int(metrics.index.min()),
                        "time_end": int(metrics.index.max()), "n_services": len(names),
                        "n_metrics": metrics.shape[1], "n_timesteps": len(metrics),
                        "normal_rows": normal_rows, "anomalous_rows": anomalous_rows,
                        "duration_minutes": float((metrics.index.max() - metrics.index.min()) / 60),
                        "missing_values": int(raw.isna().sum().sum()),
                        "infinite_values": int(np.isinf(raw.to_numpy()).sum()),
                        "duplicate_time_rows": int(raw["time"].duplicated().sum()),
                        "median_step_seconds": float(np.median(np.diff(metrics.index))),
                        "root_in_candidates": root in names,
                        "evaluable": normal_rows >= 2 and anomalous_rows >= 1})
    overview = pd.DataFrame(records)
    info = {"n_cases": len(overview), "n_evaluable": int(overview["evaluable"].sum()),
            "invalid_window_cases": overview.loc[~overview["evaluable"], "case_id"].tolist(),
            "fault_counts": overview["fault_type"].value_counts().sort_index().to_dict(),
            "root_services": sorted(overview["true_root"].unique().tolist()),
            "candidate_services": sorted(all_services), "metric_names": sorted(metric_names)}
    for field, key in [("n_metrics", "metric_count_range"), ("n_services", "service_count_range"),
                       ("n_timesteps", "row_count_range")]:
        info[key] = [int(overview[field].min()), int(overview[field].max())]
    info.update({"missing_values": int(overview["missing_values"].sum()),
                 "cases_with_missing_values": int(overview["missing_values"].gt(0).sum()),
                 "infinite_values": int(overview["infinite_values"].sum()),
                 "duplicate_timestamps": int(overview["duplicate_time_rows"].sum()),
                 "all_roots_present": bool(overview["root_in_candidates"].all())})
    results_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(results_dir / "dataset_overview.csv", index=False)
    (results_dir / "dataset_audit.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return overview, info


def plot_case_metrics(case, output: Path = RESULTS_DIR / "fig_case_metrics.png"):
    """Plot the annotated root and four comparison services in existing metrics."""
    metrics, injection, root, _, _ = load_case(case)
    scores = zscore_service_scores(metrics, injection)
    others = [name for name in scores["service"] if name != root][:4]
    selected = [root] + others
    existing = set(metrics.columns.get_level_values("metric"))
    latency = next((name for name in ("latency-90", "latency", "latency-50") if name in existing), None)
    kinds = [name for name in ["cpu", "mem", latency] if name is not None and name in existing]
    fig, axes = plt.subplots(len(kinds), 1, figsize=(12, 8.2), sharex=True, layout="constrained")
    minutes = (metrics.index.to_numpy() - injection) / 60
    palette = [ROOT_COLOR, "#118f97", "#d48a20", "#548044", "#586b9b"]
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        for i, service in enumerate(selected):
            if (service, kind) in metrics:
                ax.plot(minutes, metrics[(service, kind)], color=palette[i],
                        linewidth=2.1 if service == root else 1.0, alpha=0.95 if service == root else 0.7,
                        label=service + (" (annotated root)" if service == root else ""))
        ax.axvline(0, color="#222222", linestyle="--", linewidth=1.1, label="fault injection")
        ax.set_ylabel(f"{kind}\n(raw recorded units)")
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    np.atleast_1d(axes)[-1].set_xlabel("Minutes relative to the supplied injection timestamp")
    fig.suptitle(f"One incident, multiple services: {case.case_id}", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    return fig


def find_louder_cases(data_dir: Path = DEFAULT_DATA_DIR, results_dir: Path = RESULTS_DIR,
                      skip_constant: bool = False) -> pd.DataFrame:
    """Search every evaluable case; labels are used only AFTER anomaly scoring."""
    rows = []
    for case in list_cases(data_dir):
        metrics, injection, root, fault, _ = load_case(case)
        try:
            scores = zscore_service_scores(metrics, injection, skip_constant)
        except InsufficientWindowError:
            continue
        root_row = scores.loc[scores["service"].eq(root)].iloc[0]
        top = scores.iloc[0]
        if top["service"] != root and top["score"] > root_row["score"]:
            rows.append({"case_id": case.case_id, "fault_type": fault, "true_root": root,
                         "root_rank": int(scores.index[scores["service"].eq(root)][0] + 1),
                         "root_score": float(root_row["score"]),
                         "louder_nonroot": top["service"], "nonroot_score": float(top["score"]),
                         "nonroot_winning_metric": top["winning_metric"],
                         "root_winning_metric": root_row["winning_metric"]})
    hits = pd.DataFrame(rows, columns=["case_id", "fault_type", "true_root", "root_rank",
                                     "root_score", "root_winning_metric", "louder_nonroot",
                                     "nonroot_score", "nonroot_winning_metric"])
    hits["ratio"] = hits["nonroot_score"] / hits["root_score"]
    method = "zscore_nonconst" if skip_constant else "zscore"
    hits.to_csv(results_dir / f"victim_search_{method}.csv", index=False)
    return hits


def plot_louder_case(case, output: Path = RESULTS_DIR / "fig_victim_louder_than_culprit.png",
                     skip_constant: bool = True, title: str = "", caption: str = "",
                     selection_rule: str = ""):
    """Show the actual scores and winning metric traces without mixing their units."""
    metrics, injection, root, _, _ = load_case(case)
    scores = zscore_service_scores(metrics, injection, skip_constant)
    stats = metric_anomaly_scores(metrics, injection)
    victim = str(scores.iloc[0]["service"])
    if victim == root:
        raise ValueError("This case does not have a higher-scoring non-root candidate.")
    fig = plt.figure(figsize=(14, 8.3), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3])
    bar = fig.add_subplot(grid[:, 0])
    colors = [ROOT_COLOR if name == root else VICTIM_COLOR if name == victim else "#bac4cc"
              for name in scores["service"]]
    positions = np.arange(len(scores))
    bar.barh(positions, np.log10(1 + scores["score"].clip(lower=0)), color=colors)
    bar.set_yticks(positions, [name + (" [ROOT]" if name == root else " [NON-ROOT]" if name == victim else "")
                              for name in scores["service"]], fontsize=9)
    bar.invert_yaxis()
    bar.set_xlabel("log10(1 + service z-score)" + (", constant metrics ignored" if skip_constant else ""))
    bar.set_title("The largest anomaly is not the injected root", fontsize=11)
    minutes = (metrics.index.to_numpy() - injection) / 60
    details = []
    for row_index, (service, color, role) in enumerate([(root, ROOT_COLOR, "Annotated root"),
                                                       (victim, VICTIM_COLOR, "Higher-scoring non-root")]):
        record = scores.loc[scores["service"].eq(service)].iloc[0]
        metric = record["winning_metric"]
        observed = stats.loc[(service, metric)]
        ax = fig.add_subplot(grid[row_index, 1])
        ax.plot(minutes, metrics[(service, metric)], color=color, linewidth=1.1)
        ax.axvline(0, color="#222222", linestyle="--", label="fault injection")
        ax.set_title(f"{role}: {service} / {metric}\nz = {record['score']:,.2f}", fontsize=11)
        ax.set_xlabel("Minutes relative to injection")
        ax.set_ylabel("Raw recorded units")
        ax.legend(fontsize=8)
        ax.text(0.02, 0.04, f"Pre mean: {observed['normal_mean']:.5g}   Pre std: {observed['normal_std']:.5g}\n"
                f"Post mean: {observed['anomalous_mean']:.5g}", transform=ax.transAxes,
                fontsize=8, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"})
        details.append({"service": service, "role": role, "winning_metric": metric,
                        "score": float(record["score"]),
                        **{name: float(observed[name]) for name in ["normal_mean", "normal_std", "anomalous_mean"]}})
    fig.suptitle(title or f"Why anomaly size alone can mislead: {case.case_id}", fontsize=14)
    fig.supxlabel(caption or "Purple = annotated root; orange = higher-scoring non-root.", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    metadata = {"case_id": case.case_id, "method": "zscore_nonconst" if skip_constant else "zscore",
                "selection_rule": selection_rule, "inject_time": injection, "details": details}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return fig


def plot_baseline_comparison(summaries: dict[str, pd.DataFrame],
                             output: Path = RESULTS_DIR / "fig_baseline_comparison.png"):
    """AC@1 and AC@3 per fault type for each baseline, on all 125 supplied cases."""
    labels = {"zscore": "Z-score (all metrics)", "zscore_nonconst": "Z-score, constant metrics ignored"}
    colors = {"zscore": "#bac4cc", "zscore_nonconst": "#6a3f9b"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, layout="constrained")
    width = 0.38
    for ax, metric in zip(axes, ["AC@1", "AC@3"]):
        for i, (method, summary) in enumerate(summaries.items()):
            table = summary.loc[summary["evaluation_set"].eq("all_cases")]
            names = ["ALL"] + [name.upper() for name in table.loc[table["scope"].eq("fault_type"), "fault_type"]]
            x = np.arange(len(table))
            bars = ax.bar(x + (i - 0.5) * width, table[metric], width, label=labels[method], color=colors[method])
            ax.bar_label(bars, labels=[f"{value:.0%}" for value in table[metric]], padding=2, fontsize=8)
        ax.set_xticks(x, names)
        ax.set_title(f"{metric}: root cause ranked in the top {metric[-1]}")
        ax.set_ylim(0, 1.3)
        ax.set_yticks(np.arange(0, 1.01, .2), [f"{value:.0%}" for value in np.arange(0, 1.01, .2)])
    axes[0].set_ylabel("Share of cases")
    axes[0].legend(loc="upper left", ncol=2, fontsize=9)
    fig.suptitle("Ignoring constant metrics fixes most top-1 errors; packet loss stays hardest", fontsize=13)
    fig.supxlabel("All 125 supplied cases; the 2 invalid-window cases count as misses. Offline diagnosis with known injection time.", fontsize=9)
    fig.savefig(output)
    return fig


def worst_case_explanation(case, skip_constant: bool = True) -> str:
    """Generate a Markdown explanation from measured scores; causes stay hypotheses."""
    metrics, injection, root, _, _ = load_case(case)
    scores = zscore_service_scores(metrics, injection, skip_constant)
    root_position = int(scores.index[scores["service"].eq(root)][0])
    above = scores.iloc[:root_position]
    shown = scores.iloc[:root_position + 1].copy()
    stats = metric_anomaly_scores(metrics, injection)
    shown["pre_std_of_winner"] = [float(stats.loc[(r.service, r.winning_metric), "normal_std"])
                                   for r in shown.itertuples()]
    names = ", ".join(f"`{name}`" for name in above["service"])
    near_constant = shown.iloc[:-1].loc[shown.iloc[:-1]["pre_std_of_winner"] < 1e-6, "service"].tolist()
    explanation = (f"### {case.case_id}\n\nThe annotated root `{root}` ranks **{root_position + 1}**. "
                   f"The services/candidates above it are {names}.\n\n"
                   + shown[["service", "winning_metric", "score", "pre_std_of_winner"]].to_markdown(index=False, floatfmt=".5g")
                   + "\n\n**Observed:** the maximum standardized mean shift is larger for these candidates. ")
    if near_constant:
        explanation += ("Near-zero normal-window standard deviation occurs for "
                        + ", ".join(f"`{name}`" for name in near_constant)
                        + "; dividing by std + 1e-8 amplifies even small changes. ")
    explanation += ("**Possible explanation:** packet loss can affect callers and request/error metrics, while unrelated resource drift can also score highly. "
                    "The heuristic does not model dependencies or temporal propagation. "
                    "These measurements alone do not establish which non-root changes were caused by the injected fault.\n")
    return explanation
