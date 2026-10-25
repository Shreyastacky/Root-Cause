"""Render the robustness audit from its measured results, including regressions."""

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rca.data import PROJECT_ROOT
from rca.robustness import LABELS, OUTPUT_DIR, REFERENCE


def overall_table(summary, evaluation_set="evaluable_only"):
    rows = summary.query("scope == 'overall' and evaluation_set == @evaluation_set").set_index("method").loc[list(LABELS)]
    output = pd.DataFrame({"Experiment": [LABELS[name] for name in rows.index],
                           "Incidents": rows["n_cases"].to_numpy()})
    for column in ["AC@1", "AC@3", "AC@5", "Avg@5"]:
        output[column] = [f"{value:.2%}" for value in rows[column]]
    return output


def plot_audit(summary, folds):
    rows = summary.query("scope == 'overall' and evaluation_set == 'evaluable_only'").set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    names = [REFERENCE, "availability_only", "behavior_only"]
    values = rows.loc[names, "AC@1"].to_numpy() * 100
    bars = axes[0].bar(["Full (48)", "Availability (8)", "Behavior (24)"], values,
                       color=["#315c91", "#bd7045", "#388473"])
    axes[0].bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
    axes[0].set_title("Feature ablations: identical scenario folds")
    held = folds.loc[folds["method"].eq("held_out_root")].sort_values("held_out_root")
    labels = held["held_out_root"].str.replace("service", "", regex=False)
    values = held["AC@1"].to_numpy() * 100
    bars = axes[1].bar(labels, values, color="#8264a1")
    axes[1].bar_label(bars, labels=[f"{value:.1f}%\nn={n}" for value, n in zip(values, held["n_cases"])], padding=3)
    axes[1].axhline(rows.loc["held_out_root", "AC@1"] * 100, color="#593d74", linestyle="--",
                    label=f"Pooled held-out root: {rows.loc['held_out_root', 'AC@1']:.1%}")
    axes[1].set_title("Different task: hold out the root service")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.set_ylim(0, 110)
        ax.set_ylabel("Top-one root localization (%)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(OUTPUT_DIR / "fig_robustness.png", dpi=160)
    return fig


def build_report():
    summary = pd.read_csv(OUTPUT_DIR / "summary.csv")
    folds = pd.read_csv(OUTPUT_DIR / "fold_summary.csv")
    cases = pd.read_csv(OUTPUT_DIR / "per_case.csv")
    manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text())
    overall = summary.query("scope == 'overall' and evaluation_set == 'evaluable_only'").set_index("method")
    held = folds.loc[folds["method"].eq("held_out_root"), ["held_out_root", "n_cases", "AC@1", "AC@3", "AC@5"]].copy()
    for column in ["AC@1", "AC@3", "AC@5"]:
        held[column] = held[column].map(lambda value: f"{value:.2%}")
    paired = pd.read_csv(OUTPUT_DIR / "paired_ranks.csv")
    comparisons = []
    for method in ["availability_only", "behavior_only", "held_out_root"]:
        comparisons.append({"Experiment": LABELS[method],
                            "Top-one gains vs reference": int((paired[method].eq(1) & paired[REFERENCE].ne(1)).sum()),
                            "Top-one losses vs reference": int((paired[method].ne(1) & paired[REFERENCE].eq(1)).sum())})
    uniform_expected = np.mean(1 / cases.query("method == @REFERENCE and status == 'ok'")["n_candidates"])
    held_failures = cases.query("method == 'held_out_root' and status == 'ok' and rank_of_true_root > 1")
    fault_failures = held_failures.groupby("fault_type").size().to_dict()
    successes = {method: int(frame['rank_of_true_root'].eq(1).sum())
                 for method, frame in cases.loc[cases['status'].eq('ok')].groupby('method')}
    figure = plot_audit(summary, folds)
    plt.close(figure)
    report = f"""# Feature and root-service robustness audit

The existing scenario-CV result is **{overall.loc[REFERENCE, 'AC@1']:.2%}** top-one.
With availability alone it is **{overall.loc['availability_only', 'AC@1']:.2%}**;
with behavior features alone, **{overall.loc['behavior_only', 'AC@1']:.2%}**.
The separate held-out-root-service task gives
**{overall.loc['held_out_root', 'AC@1']:.2%}** top-one. All use the same fixed Random
Forest settings; no tuning or winner selection was performed.

## Why this experiment is necessary

A forest can learn relationships in measured changes, but it can also exploit
which measurements are available for a service. The previous folds held out
root/fault combinations while allowing the same root service in other training
scenarios. This audit probes both issues before increasing model complexity.
See the [pre-run protocol](ROBUSTNESS_PROTOCOL.md).

## Measured results

Primary denominator: **123 evaluable incidents**. The first three rows share the
exact original scenario folds. The last row changes the training partition and
answers a different generalization question.

{overall_table(summary).to_markdown(index=False)}

A uniformly random first-ranked candidate would have expected AC@1
**{uniform_expected:.2%}**, calculated as the average of 1 / candidate count over
these incidents. This is an analytical reference, not a trained or sampled model.

![Feature ablations and held-out service results](../results/robustness/fig_robustness.png)

All **125 supplied incidents**, counting the two invalid windows as misses:

{overall_table(summary, 'all_cases').to_markdown(index=False)}

### Every held-out root service

{held.to_markdown(index=False)}

All fault families and repetitions rooted at the named service are absent from
that fold's training incidents. The service can still appear as a negative
candidate in those training incidents. This is not a wholly unseen service or
application test. Pool incident ranks for the main score; do not silently average
the five percentages with their slightly different denominators.

### Paired changes from the original forest

{pd.DataFrame(comparisons).to_markdown(index=False)}

The held-out-root comparison changes the train/test task; its gains and losses
are descriptive and do not isolate a feature effect. Full failures and every
candidate score are saved in `results/robustness/`.

## Interpretation and limits

Availability patterns contain useful information, but availability alone does
not reproduce the full forest's performance. Removing the explicit flags changes
top-one success by {successes['behavior_only'] - successes[REFERENCE]} incident
({successes['behavior_only']} versus {successes[REFERENCE]}), while AC@3 and Avg@5 fall;
there is no consistent improvement across ranking cutoffs. We retain the original
baseline rather than select a new winner from these diagnostics.

The held-out-root experiment has {len(held_failures)} top-one misses, distributed
by fault as `{fault_failures}`. Productcatalogservice is weakest at
{folds.query("method == 'held_out_root' and held_out_root == 'productcatalogservice'").iloc[0]['AC@1']:.2%}
versus {overall.loc['held_out_root', 'AC@1']:.2%} pooled across all services.
Inspect the individual failures before interpreting the
high pooled score. Neither split is guaranteed to be empirically harder: each
holds out a different type of information and changes which training cases remain.

- Availability-only performance measures predictive information in availability
  patterns, including finite-observation requirements. It does not prove direct
  label leakage, service-name access, or causality.
- Removing explicit flags does not remove every schema clue. Behavior features
  still have named metric types and use zero for missing metrics.
- The held-out-root result measures transfer to a new fault location within
  Online Boutique. It does not test a new workload, fault family, or system.
- These data were already explored. These are diagnostic results, not a new
  untouched final test. No confidence intervals or statistical-significance
  claims are made; the incidents are dependent repetitions of only 25 scenarios.
- All experiments use complete before/after recordings and supplied injection
  times. Early detection and performance under short observation windows remain
  untested. Both invalid recordings remain unmodified.

## Evidence and reproduction

The manifest records {len(manifest['variants'])} fixed new variants and 15 new
forests, explicit case splits, full feature lists, input/source hashes and
scikit-learn {manifest['sklearn_version']}. Text hashes normalize CRLF to LF so
Windows notebook outputs and Git's LF files verify identically after a clone.
The reference scores are copied from
the original experiment without replacement. Verification independently refits
the new forests and recalculates the saved scores, rankings and summaries.

```powershell
.\\.venv\\Scripts\\python.exe scripts/execute_notebooks.py 04_robustness.ipynb
.\\.venv\\Scripts\\python.exe scripts/build_reports.py
.\\.venv\\Scripts\\python.exe scripts/verify_project.py
```

The [executed fourth notebook](../notebooks/04_robustness.ipynb) reruns the complete
audit. Its [rendered HTML](../results/04_robustness.html) retains outputs.
Alternatively run `python -m rca.robustness` followed by
`python -c "from rca.robustness_report import build_report; build_report()"`.

Next: reserve new data and specify its evaluation protocol before further model
selection. Add a robust statistical baseline and a fixed observation-window
study before interpreting gains from a temporal or graph model.
"""
    (PROJECT_ROOT / "docs/ROBUSTNESS_AUDIT.md").write_text(report, encoding="utf-8")
    return report


def update_project_reports(readme, slides):
    summary = pd.read_csv(OUTPUT_DIR / "summary.csv")
    rows = summary.query("scope == 'overall' and evaluation_set == 'evaluable_only'").set_index("method")
    section = f"""## Robustness audit: what does the forest learn?

{overall_table(summary).to_markdown(index=False)}

The first three rows reuse identical scenario folds. The fourth holds out all
incidents rooted at one service at a time and answers a different generalization question.
The service may still appear as a non-root training candidate. This is diagnostic
within-application evaluation, not an untouched or cross-system test.
See the [full audit](docs/ROBUSTNESS_AUDIT.md), [fixed protocol](docs/ROBUSTNESS_PROTOCOL.md),
and [executed notebook](notebooks/04_robustness.ipynb) ([HTML](results/04_robustness.html)).

![Robustness audit](results/robustness/fig_robustness.png)

"""
    readme = readme.replace("## Original heuristic results", section + "## Original heuristic results")
    readme = readme.replace("Thirteen focused tests check", "Focused unit tests check")
    readme = readme.replace("2. Completed: scenario-grouped five-fold splits and a fixed Random Forest baseline. Reserve genuinely new data for the next evaluation; do not retune to these CV results.",
                            "2. Completed: fixed Random Forest, scenario-grouped CV, feature ablations and root-service holdout. Reserve genuinely new data before further model selection; do not retune to these diagnostic results.")
    readme = readme.replace("- [Eight-slide outline]", "- [04_robustness.ipynb](notebooks/04_robustness.ipynb): fixed feature ablations, root-service holdout, all scores and failures. [Rendered copy](results/04_robustness.html).\n- [Eight-slide outline]")
    readme = readme.replace("## Supervised verification", "## Supervised and robustness verification")
    readme += ("\nThe robustness verifier independently refits all 15 additional forests, checks "
               "feature subsets and root-service isolation, and reproduces candidate scores and metrics.\n")
    slides = re.sub(r"## 7\..*?(?=## 8\.)", f"""## 7. Robustness audit — feature shortcuts and new fault locations

- Availability only: AC@1={rows.loc['availability_only', 'AC@1']:.2%}; behavior only: {rows.loc['behavior_only', 'AC@1']:.2%}, using the original scenario folds and fixed forest parameters.
- Full features, with every incident rooted at the test service excluded from training: AC@1={rows.loc['held_out_root', 'AC@1']:.2%} on 123 evaluable incidents.
- A held-out root may still appear as a negative candidate in training; this is a different task from scenario CV, not cross-system transfer.
- Keep every failure and both invalid cases visible; missing-feature zeros can still reveal schema after explicit flags are removed.
- Next: reserve new data, test fixed observation windows and robust statistics; verify topology before a temporal/GNN extension.

**Figure:** `results/robustness/fig_robustness.png`.
**Notes:** [ROBUSTNESS_AUDIT.md](ROBUSTNESS_AUDIT.md).

""", slides, flags=re.DOTALL)
    build_report()
    return readme, slides
