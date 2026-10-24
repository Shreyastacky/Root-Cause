"""Figures and student-facing reports from the fixed supervised experiment."""

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rca.analysis import configure_plots
from rca.data import PROJECT_ROOT
from rca.supervised import OUTPUT_DIR


LABELS = {"zscore": "Original z-score", "zscore_nonconst": "Nonconstant z-score", "random_forest": "Random Forest (OOF)"}
COLORS = {"zscore": "#b6c0c9", "zscore_nonconst": "#178e94", "random_forest": "#6a3f9b"}


def overall_table(summary):
    table = summary.query("scope == 'overall' and evaluation_set == 'evaluable_only'").copy()
    table["method"] = table["method"].map(LABELS)
    table = table[["method", "n_cases", "AC@1", "AC@3", "AC@5", "Avg@5"]]
    for column in ["AC@1", "AC@3", "AC@5", "Avg@5"]:
        table[column] = table[column].map(lambda value: f"{value:.2%}")
    return table


def plot_grouped_comparison(summary, fold_summary, output_dir=OUTPUT_DIR):
    configure_plots()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout="constrained")
    table = summary.query("scope == 'overall' and evaluation_set == 'evaluable_only'").set_index("method")
    width = .24
    for offset, method in enumerate(LABELS):
        values = [table.loc[method, "AC@1"], table.loc[method, "AC@3"]]
        bars = axes[0].bar(np.arange(2) + (offset - 1) * width, values, width,
                           label=LABELS[method], color=COLORS[method])
        axes[0].bar_label(bars, labels=[f"{value:.1%}" for value in values], fontsize=8, padding=3)
        folds = fold_summary.loc[fold_summary["method"].eq(method)].sort_values("fold")
        axes[1].plot(folds["fold"], folds["AC@1"], marker="o", color=COLORS[method], label=LABELS[method])
    axes[0].set_xticks([0, 1], ["Root ranked first (AC@1)", "Root in first three (AC@3)"])
    axes[0].set_title("Same 123 evaluable incidents")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set_title("AC@1 varies across held-out scenario groups")
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xlabel("Grouped fold (25 / 25 / 25 / 24 / 24 incidents)")
    for ax in axes:
        ax.set_ylim(0, 1.22)
        ax.set_yticks(np.arange(0, 1.01, .2), [f"{value:.0%}" for value in np.arange(0, 1.01, .2)])
    fig.suptitle("First supervised baseline: fixed Random Forest, five grouped folds", fontsize=14)
    fig.supxlabel("Exploratory within Online Boutique; repeated root-service / fault scenarios never cross train and test in a fold.", fontsize=9)
    fig.savefig(Path(output_dir) / "fig_grouped_cv.png")
    return fig


def build_report(output_dir=OUTPUT_DIR, docs_dir=PROJECT_ROOT / "docs"):
    output_dir, docs_dir = Path(output_dir), Path(docs_dir)
    summary = pd.read_csv(output_dir / "summary.csv")
    folds = pd.read_csv(output_dir / "fold_summary.csv")
    pairs = pd.read_csv(output_dir / "paired_case_comparison.csv")
    manifest = json.loads((output_dir / "split_manifest.json").read_text())
    rf = summary.query("method == 'random_forest' and scope == 'overall'").set_index("evaluation_set")
    valid = rf.loc["evaluable_only"]
    conservative = rf.loc["all_cases"]
    rf_folds = folds.loc[folds["method"].eq("random_forest")].sort_values("fold")
    gained = int((pairs["rf_top1"] & ~pairs["heuristic_top1"]).sum())
    lost = int((~pairs["rf_top1"] & pairs["heuristic_top1"]).sum())
    per_case = pd.read_csv(output_dir / "per_case.csv")
    failures = per_case.query("method == 'random_forest' and status == 'ok' and rank_of_true_root > 1")[
        ["case_id", "fault_type", "fold", "true_root", "rank_of_true_root"]].sort_values(["rank_of_true_root", "case_id"], ascending=[False, True])
    failures.to_csv(output_dir / "random_forest_failures.csv", index=False)
    figure = plot_grouped_comparison(summary, folds, output_dir)
    plt.close(figure)
    fold_display = rf_folds[["fold", "n_cases", "AC@1", "AC@3", "Avg@5"]].copy()
    for name in ["AC@1", "AC@3", "Avg@5"]:
        fold_display[name] = fold_display[name].map(lambda value: f"{value:.2%}")
    fault_display = summary.query("method == 'random_forest' and scope == 'fault_type' and evaluation_set == 'evaluable_only'")[["fault_type", "n_cases", "AC@1", "AC@3", "AC@5", "Avg@5"]].copy()
    for name in ["AC@1", "AC@3", "AC@5", "Avg@5"]:
        fault_display[name] = fault_display[name].map(lambda value: f"{value:.2%}")
    report = f"""# First supervised baseline: Random Forest

## What changed

The project now trains a supervised Random Forest to score a candidate service
as root versus non-root, then ranks each incident's candidates. This is the first
trained model; the two existing methods remain statistical references.

The [fixed protocol](ML_PROTOCOL.md) specifies features, model parameters, and
evaluation before the first fit. There was no parameter search or seed selection.

## Data and features

- {manifest['n_incidents']} supplied incidents; {manifest['n_evaluable']} have valid windows.
- {manifest['n_scenario_groups']} root-service / fault-type scenario groups, keeping repetitions together.
- {manifest['n_service_rows']} candidate-service rows; these are not independent incidents.
- {len(manifest['feature_columns'])} numeric features: six changes/availability indicators for each of eight metric types.
- Service names, fault labels, filenames, timestamps, case IDs, and root labels are not model features.
- Class imbalance is handled using balanced_subsample class weights; no synthetic resampling.

Each feature row is computed independently from its own incident's pre/post
windows. Available metric types can still act as proxies for service roles.
The full post-injection recording and supplied injection time remain assumptions.

## Split and training

Five-fold GroupKFold holds each evaluable incident out exactly once. Each fold
trains on 98 or 99 incidents and tests on 24 or 25. A root/fault scenario and all
its repetitions stay wholly on one side of that fold's boundary. Group metadata
is used for splitting only, never as an input feature.

Each forest has 300 trees, maximum depth 6, minimum leaf size 3, sqrt feature
sampling, balanced_subsample weights, seed 42, and one worker. Trees are trained
on numeric features from training incidents only. Class-1 predict_proba values
are uncalibrated ranking scores, not verified confidence probabilities.

The [split manifest](../results/supervised/split_manifest.json) was saved before
fitting; it records every training/test incident, group, configuration, and source
hash. No final all-data model is used for the reported scores.

## Primary comparison: 123 evaluable incidents

{overall_table(summary).to_markdown(index=False)}

Random Forest places the root first in **{int(round(valid['AC@1']*valid['n_cases']))}/{int(valid['n_cases'])} incidents**.
It gets {gained} previously missed top-one cases right and loses {lost} that the
nonconstant heuristic got right. Its AC@5 is slightly below that heuristic;
the learned model does not improve every metric.

![Grouped comparison and fold variability](../results/supervised/fig_grouped_cv.png)

With all 125 supplied incidents in the denominator and invalid recordings counted
as misses, Random Forest achieves AC@1 **{conservative['AC@1']:.2%}**, AC@3
**{conservative['AC@3']:.2%}**, AC@5 **{conservative['AC@5']:.2%}**, and Avg@5
**{conservative['Avg@5']:.2%}**. The invalid cases were neither trained on nor predicted.

## Fold variability

{fold_display.to_markdown(index=False)}

The lowest fold AC@1 is **{rf_folds['AC@1'].min():.2%}**, versus a highest of
**{rf_folds['AC@1'].max():.2%}**. The pooled score hides meaningful scenario variation.
Folds are different scenario collections, not interchangeable repeated trials.

## By fault type, evaluable incidents only

{fault_display.to_markdown(index=False)}

## Remaining Random Forest failures

{failures.to_markdown(index=False)}

These are held-out prediction failures, not training-set errors. Refer to the
per-candidate scores and full rankings before proposing a mechanism. Impurity
importance in `feature_importance.csv` describes the fitted forests; it does not
establish causation or prove generalization.

## What this does and does not establish

The fixed learned model improved top-one ranking on this grouped experiment.
However, RE1-OB was previously explored while constructing the heuristics.
This is exploratory cross-validation within one application, not a pristine
untouched test or cross-system result. There are only 25 scenario groups.
Schema/availability patterns and recording regimes may act as shortcuts even
without service names. No causal, online-detection, production, or GNN claim is made.

Next: evaluate sensitivity to features and recording regimes, add a robust
statistical baseline, and reserve genuinely new data before further model
selection. Verify dependency topology before introducing a graph model.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m rca.supervised
.\\.venv\\Scripts\\python.exe scripts/execute_notebooks.py 03_supervised.ipynb
.\\.venv\\Scripts\\python.exe scripts/build_reports.py
.\\.venv\\Scripts\\python.exe scripts/verify_project.py
```

The executed [third notebook](../notebooks/03_supervised.ipynb) explains features,
fold isolation, training, results, importance, and failures. It reruns the complete
fixed experiment. See `results/supervised/` for the CSVs and provenance.
"""
    (docs_dir / "SUPERVISED_BASELINE.md").write_text(report, encoding="utf-8")
    return report


def update_project_reports(readme: str, slides: str, execution: list) -> tuple[str, str]:
    """Integrate the new stage without discarding the original baseline evidence."""
    summary = pd.read_csv(OUTPUT_DIR / "summary.csv")
    manifest = json.loads((OUTPUT_DIR / "split_manifest.json").read_text())
    row = summary.query("method == 'random_forest' and scope == 'overall' and evaluation_set == 'evaluable_only'").iloc[0]
    conservative = summary.query("method == 'random_forest' and scope == 'overall' and evaluation_set == 'all_cases'").iloc[0]
    reference = summary.query("method == 'zscore_nonconst' and scope == 'overall' and evaluation_set == 'evaluable_only'").iloc[0]
    folds = pd.read_csv(OUTPUT_DIR / "fold_summary.csv")
    rf_folds = folds.loc[folds["method"].eq("random_forest")]
    low, high = rf_folds["AC@1"].min(), rf_folds["AC@1"].max()
    readme = readme.replace("baselines**. A learned model and a graph neural network are proposed future work.",
                            "baselines**, plus a **supervised Random Forest evaluated with scenario-grouped cross-validation**. A temporal encoder and graph neural network remain future work.")
    readme = readme.replace("- Both notebooks executed top to bottom:", "- The original two notebooks executed top to bottom:")
    readme = readme.replace("There is no trained model, train/test split, cross-system\ngeneralization result, or deployment claim in this week's work.",
                            "The original heuristic experiment had no training split. The supervised stage above adds grouped train/test folds, but no cross-system generalization or deployment result.")
    readme = readme.replace("## Results\n", f"""## Supervised stage: first trained model

Five-fold grouped cross-validation keeps all repetitions of each root-service /
fault-type scenario together. The fixed Random Forest receives 48 numeric
service-change features, excluding names and labels. Its scores are out-of-fold.

{overall_table(summary).to_markdown(index=False)}

Primary denominator: **123 evaluable incidents**. With invalid cases included as
misses, Random Forest AC@1 is **{conservative['AC@1']:.2%}** over 125 cases.
This is exploratory within-application CV on already-explored data, not a pristine
final test. Fold AC@1 ranges from {low:.0%} to {high:.0%}; the model does not improve every
ranking metric. See the [full supervised report](docs/SUPERVISED_BASELINE.md),
[fixed protocol](docs/ML_PROTOCOL.md), and [executed third notebook](notebooks/03_supervised.ipynb).

```powershell
.\\.venv\\Scripts\\python.exe -m rca.supervised
```

![Grouped supervised comparison](results/supervised/fig_grouped_cv.png)

## Original heuristic results
""")
    readme = readme.replace("- [Eight-slide outline]", "- [03_supervised.ipynb](notebooks/03_supervised.ipynb): 48 features, grouped splits, trained Random Forest, comparisons and failures. [Rendered copy](results/03_supervised.html).\n- [Eight-slide outline]")
    readme = readme.replace("Both notebooks", "The original two notebooks")
    readme = readme.replace("Seven focused tests check", "Thirteen focused tests check")
    readme = readme.replace("`zscore_nonconst` ignores constant metrics.", "`zscore_nonconst` ignores constant metrics. New tests also cover feature semantics, identity independence, and disjoint scenario folds.")
    readme = readme.replace("and GNN training\nwere deliberately", "and GNN training\nremain deliberately")
    readme = readme.replace("2. Fix evaluation splits by whole incident; group repeated scenarios to limit\n   leakage. No row-level random split of the same incident.",
                            "2. Completed: scenario-grouped five-fold splits and a fixed Random Forest baseline. Reserve genuinely new data for the next evaluation; do not retune to these CV results.")
    readme = readme.replace("1. Try robust scaling", "1. Try robust scaling")
    readme += (f"\n## Supervised verification\n\nAll {len(execution)} notebooks have saved executed outputs. "
               f"The supervised split records {manifest['n_scenario_groups']} scenario groups and {manifest['n_service_rows']} candidate rows. "
               "The verification script checks fold isolation, source/data hashes, candidate scores and saved rankings, and reproduces the forests.\n")
    slides = re.sub(r"## 3\..*?(?=## 4\.)", """## 3. ML method — learn combinations of service changes

- Completed: a Random Forest classifier scores root versus non-root candidates.
- Inputs: 48 fixed features describing per-metric changes, variability, availability and onset.
- Service names, fault labels and case IDs are excluded from the feature matrix.
- Rank each incident's candidates by its forest's class-1 score; no calibrated-confidence claim.
- A temporal encoder and GNN remain later experiments, not implemented methods.

**Visual:** feature extraction → forest trained on other scenario groups → held-out service ranking.

""", slides, flags=re.DOTALL)
    slides = re.sub(r"## 6\..*?(?=## 7\.)", f"""## 6. Grouped evaluation — first trained baseline

- Fixed five-fold GroupKFold: all repetitions of each root/fault scenario stay together; {manifest['n_scenario_groups']} groups.
- Primary comparison on {int(row['n_cases'])} evaluable incidents: Random Forest AC@1={row['AC@1']:.2%}, AC@3={row['AC@3']:.2%}, Avg@5={row['Avg@5']:.2%}; stronger heuristic AC@1={reference['AC@1']:.2%}.
- All 125 supplied incidents, invalid cases counted as misses: forest AC@1={conservative['AC@1']:.2%}.
- Fold AC@1 varies from {low:.0%} to {high:.0%}; packet-loss failures remain. The forest's AC@5 is slightly worse than the stronger heuristic.
- This is exploratory within-application CV on previously explored data, not cross-system or pristine final-test performance.

**Figure:** `results/supervised/fig_grouped_cv.png`.
**Tables:** `results/supervised/summary.csv`, `results/supervised/fold_summary.csv`.
**Notes:** [SUPERVISED_BASELINE.md](SUPERVISED_BASELINE.md).

""", slides, flags=re.DOTALL)
    slides = slides.replace("- Weeks 2–3: resolve data-quality questions with upstream evidence; try robust scaling (median/MAD) and aggregation; freeze incident-level splits.",
                            "- Next stage, completed: fixed Random Forest, scenario-grouped five-fold evaluation, supervised notebook and failure analysis; later reserve genuinely new data.")
    build_report()
    return readme, slides
