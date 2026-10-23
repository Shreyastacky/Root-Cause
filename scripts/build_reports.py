"""Build README and slide outline from saved measurements, never placeholder scores."""

import json
from pathlib import Path
import textwrap

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT / 'results'
    summary = pd.read_csv(results / 'zscore_summary.csv')
    strong = pd.read_csv(results / 'zscore_nonconst_summary.csv')
    audit = json.loads((results / 'dataset_audit.json').read_text())
    manifest = json.loads((results / 'dataset_manifest.json').read_text())
    hit_table = pd.read_csv(results / 'victim_search_zscore_nonconst.csv')
    artifact_hits = pd.read_csv(results / 'victim_search_zscore.csv')
    victim = json.loads((results / 'fig_victim_louder_than_culprit.json').read_text())
    artifact = json.loads((results / 'fig_zero_variance_artifact.json').read_text())
    worst = pd.read_csv(results / 'zscore_nonconst_worst_cases.csv')
    execution = json.loads((results / 'notebook_execution.json').read_text())
    assert len(execution) == 2 and all(row['errors'] == 0 for row in execution)
    overall = summary.query("evaluation_set == 'all_cases' and scope == 'overall'").iloc[0]
    valid = summary.query("evaluation_set == 'evaluable_only' and scope == 'overall'").iloc[0]
    by_fault = summary.query("evaluation_set == 'all_cases' and scope == 'fault_type'")
    strong_overall = strong.query("evaluation_set == 'all_cases' and scope == 'overall'").iloc[0]
    strong_valid = strong.query("evaluation_set == 'evaluable_only' and scope == 'overall'").iloc[0]
    strong_by_fault = strong.query("evaluation_set == 'all_cases' and scope == 'fault_type'")
    strong_loss = strong_by_fault.loc[strong_by_fault['fault_type'].eq('loss'), 'AC@1'].iloc[0]
    root_detail, nonroot_detail = victim['details']
    artifact_root, artifact_nonroot = artifact['details']
    ratio = nonroot_detail['score'] / root_detail['score']
    percent = lambda value: f'{value:.2%}'
    columns = ['method', 'evaluation_set', 'fault_type', 'n_cases', 'n_evaluable', 'AC@1', 'AC@3', 'AC@5', 'Avg@5']

    def table(frame):
        display = frame[columns].copy()
        for column in ['AC@1', 'AC@3', 'AC@5', 'Avg@5']:
            display[column] = display[column].map(percent)
        return display.to_markdown(index=False)

    both = pd.concat([summary.assign(method='zscore'), strong.assign(method='zscore_nonconst')])
    overall_table = table(both.loc[both['scope'].eq('overall')])
    fault_table = table(both.query("evaluation_set == 'all_cases' and scope == 'fault_type'")
                        .sort_values(['fault_type', 'method']))

    readme = f"""# Root cause localization for microservice failures

Given measurements collected during an incident, rank the service/candidate
most likely to be the annotated root cause. This university progress-review
project establishes a reproducible dataset audit and **two evaluated heuristic
baselines**. A learned model and a graph neural network are proposed future work.

## Current progress

- Downloaded **{audit['n_cases']} RE1-OB (Online Boutique) cases**, with 25 cases per
  fault type. Only this subset's metrics and injection times were downloaded.
- Downloaded data/index files: **{manifest['total_bytes']:,} bytes**
  ({manifest['total_bytes']/1_000_000:.2f} MB), excluding downloader cache metadata.
- **{audit['n_evaluable']} cases are evaluable**; two supplied recordings have
  invalid before/after windows. Their timestamps and raw files remain unchanged.
- Both notebooks executed top to bottom: **{execution[0]['code_cells']} code cells in EDA and
  {execution[1]['code_cells']} in baseline, zero errors**, with saved outputs.
- Baseline 1 (`zscore`): AC@1 **{percent(overall['AC@1'])}**. Baseline 2
  (`zscore_nonconst`, ignores metrics that were constant before injection):
  AC@1 **{percent(strong_overall['AC@1'])}**, on all {audit['n_cases']} cases.
- Even with the stronger baseline, **{len(hit_table)}** evaluable cases have a
  non-root candidate scoring strictly higher than the annotated root
  ({len(artifact_hits)} with constant metrics kept).

## Setup on Windows (Python 3.10+)

The completed run used Python 3.11.9. From this project folder in PowerShell:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m rca.download
.\\.venv\\Scripts\\python.exe -m rca.evaluate --method zscore
.\\.venv\\Scripts\\python.exe -m rca.evaluate --method zscore_nonconst
.\\.venv\\Scripts\\python.exe scripts/execute_notebooks.py
```

Activation is optional: using the environment's Python directly avoids
PowerShell activation-policy problems. If activated, the requested evaluation
command is simply `python -m rca.evaluate --method zscore`.

Direct dependencies are pinned in `requirements.txt`; `requirements-lock.txt`
also records all installed dependency versions for the verified environment.
No GPU is needed for this week's baseline.

## Download and provenance

The [official RCAEval README](https://github.com/phamquiluan/RCAEval) documents
the Hugging Face `snapshot_download` method. `python -m rca.download` downloads
the index from [phamquiluan/RCAEval](https://huggingface.co/datasets/phamquiluan/RCAEval),
selects `dataset == 'RE1-OB'`, and requests only each selected case's
`metrics.parquet` and `inject_time.txt` in `data/raw/`.

Revision: `{manifest['revision']}`. Every request explicitly uses `token=False`;
the cached machine token is neither changed nor deleted. The whole-benchmark
index is metadata, not a download of other cases' telemetry. Raw data is ignored
by Git. File hashes and byte counts are in [the manifest](results/dataset_manifest.json).

See [DATA_FORMAT.md](docs/DATA_FORMAT.md) for the actual schema, labels, missing
values, candidate names, fault counts, and invalid timestamps. If a future
download fails, stop and report the error; do not substitute invented data.

## Results

All metrics are **service-level**, between 0 and 1, shown here as percentages.
For `all_cases`, unscorable recordings count as misses. The second row shows the
separate valid-recording denominator; no cases disappear silently.
`zscore` keeps all metrics; `zscore_nonconst` ignores constant-before-injection metrics.

{overall_table}

By fault type, using all supplied cases (both methods):

{fault_table}

Source tables: per-case rankings ([zscore](results/zscore_per_case.csv),
[zscore_nonconst](results/zscore_nonconst_per_case.csv)) and summaries
([zscore](results/zscore_summary.csv), [zscore_nonconst](results/zscore_nonconst_summary.csv)).
These are our full-record z-score results,
not a reproduction of published BARO scores or the paper's exact window protocol.

![Both baselines by fault type](results/fig_baseline_comparison.png)

## What the baselines do

For each metric, compare all observations before injection with all observations
at or after injection:

```text
metric score = abs(mean_after - mean_before) / (sample_std_before + 1e-8)
service score = maximum metric score for that service
```

Order by descending score; resolve exact ties alphabetically.
`zscore_nonconst` applies the same rule but ignores any metric whose
pre-injection standard deviation is exactly zero, because there the denominator
is only 1e-8 and a tiny blip gets an enormous score. Every observed
prefix is a candidate, including infrastructure-like names. The candidate list
is not restricted to the five services known to appear as root labels.

Missing cells remain NaN; the mean and sample standard deviation use the
available observations within their own window. Metrics need at least two
normal and one anomalous observation. A service with no usable metrics stays
at the bottom. In `zscore`, constant normal metrics are retained, so the tiny
denominator amplifies small changes; `zscore_nonconst` exists to remove that.

**Known injection time is allowed**, matching the supplied RCA setting.
Root labels enter only after prediction for evaluation and example selection.
The ranker receives no fault label, case identifier, or root label.

**This is offline localization after an incident**, using the full available
post-injection recording. It does not yet detect an incident online or establish
causal propagation. There is no trained model, train/test split, cross-system
generalization result, or deployment claim in this week's work. The
constant-metric rule was chosen after looking at failures on these same cases,
so its score is an analysis result, not a held-out test.

## Motivating example and failures

In `{victim['case_id']}`, packet loss was injected into
`{root_detail['service']}`. With constant metrics ignored, its winning
`{root_detail['winning_metric']}` score is {root_detail['score']:,.2f}, while its
caller `{nonroot_detail['service']}` scores {nonroot_detail['score']:,.2f} on
`{nonroot_detail['winning_metric']}`, about {ratio:.0f}x higher. In Online
Boutique, `checkoutservice` calls `productcatalogservice`, so propagation is a
plausible explanation; the plot does not prove it. This example was hand-picked
from the {len(hit_table)} non-root wins as the clearest dependency to explain.

![Caller louder than the injected root](results/fig_victim_louder_than_culprit.png)

Why baseline 1 needed fixing: in `{artifact['case_id']}`, `{artifact_nonroot['service']}`
`{artifact_nonroot['winning_metric']}` went from exactly 0 to a mean of
{artifact_nonroot['anomalous_mean']:.4f} and scored {artifact_nonroot['score']:,.0f},
beating the injected root's `{artifact_root['winning_metric']}` jump from
{artifact_root['normal_mean']:.1f} to {artifact_root['anomalous_mean']:.1f}
(score {artifact_root['score']:,.0f}).

![Zero-variance artifact](results/fig_zero_variance_artifact.png)

Under `zscore_nonconst`, the first two worst ranked evaluable cases are `{worst.iloc[0]['case_id']}`
(root rank {int(worst.iloc[0]['rank_of_true_root'])}) and `{worst.iloc[1]['case_id']}`
(rank {int(worst.iloc[1]['rank_of_true_root'])}). Read the measured score tables and
qualified interpretations in [failure_analysis.md](docs/failure_analysis.md).

Two supplied cases cannot be ranked: `re1ob_currencyservice_loss_1` has zero
pre-injection rows, and `re1ob_productcatalogservice_cpu_3` has zero
post-injection rows. They have `status=invalid_window`, empty predictions, and
infinity root rank in the CSV. That infinity is a reporting sentinel, not a
model-produced rank. No timestamp correction was guessed.

## Notebooks and review material

- [01_eda.ipynb](notebooks/01_eda.ipynb): actual data audit, a root/comparison
  metric plot, search for a louder non-root, and the zero-variance artifact.
- [02_baseline.ipynb](notebooks/02_baseline.ipynb): evaluates both baselines,
  displays both denominator policies, compares them, and explains two failures.
- Read-only rendered copies: [EDA](results/01_eda.html) and
  [baseline](results/02_baseline.html). These include executed outputs.
- [Eight-slide outline](docs/slide_outline.md), [related work](docs/related_work.md),
  and [execution record](results/notebook_execution.json).

To edit interactively using the project environment:

```powershell
.\\.venv\\Scripts\\python.exe -m ipykernel install --sys-prefix --name rca-project --display-name "Python (RCA project)"
.\\.venv\\Scripts\\python.exe -m jupyter lab
```

Select **Python (RCA project)**. The execution script instead creates a local
kernel specification inside `.jupyter/` and uses the same Python that launches
the script. It preserves executed notebooks and exports HTML automatically.
Notebook execution requires permission for the kernel's local communication
sockets. No application server or distributed infrastructure is started.

After editing analysis code, rerun evaluation and the notebooks. If changing
the pinned baseline's failure explanations, refresh `docs/failure_analysis.md`
from `rca.analysis.worst_case_explanation`, then regenerate notebooks with
`scripts/create_notebooks.py`. This clears prior notebook outputs; execute
them again. Finally run `scripts/build_reports.py` to refresh the README and
slide numbers from CSVs.

## Checks and what was skipped

```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe -m pip check
```

Seven focused tests check the score formula, injection boundary, missing-value
handling, deterministic ties, invalid windows, metric denominators, and that
`zscore_nonconst` ignores constant metrics.
Constructed test fixtures are not presented as benchmark recordings.

The first download attempt failed with an expired cached OAuth token. Retrying
with `token=False` succeeded; **Zenodo fallback was not needed**. An initial
notebook launch stalled under restricted local-kernel permissions; execution
with local-kernel permission succeeded for both notebooks.

**BARO was skipped before installation.** Official RCAEval 1.8.1 default-extra
metadata requires NumPy <2 on Python 3.11 and ipykernel 6.23.1, conflicting with
this pinned environment. We did not downgrade it or claim a BARO result. See
[recorded package constraints](results/optional_baro_check.json).

Docker, Kubernetes, new fault injection, a dependency graph, and GNN training
were deliberately not set up. Existing raw files were not modified.

## Next steps

1. Try robust scaling (median/MAD) and other aggregations as stronger statistical baselines.
2. Fix evaluation splits by whole incident; group repeated scenarios to limit
   leakage. No row-level random split of the same incident.
3. Learn an anomaly encoder and compare it with a simple learned service ranker.
4. Verify application dependency topology, then test a GNN against an otherwise
   identical model without graph edges. Do not assume graph learning is causal.
5. Later collect controlled fault-injection telemetry and reserve another whole
   application for cross-system evaluation. These steps are proposed, not run.

## Citation

Luan Pham, Hongyu Zhang, Huong Ha, Flora Salim, and Xiuzhen Zhang. 2025.
[RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data](https://doi.org/10.1145/3701716.3715290).
Companion Proceedings of the ACM Web Conference 2025. The upstream dataset
is MIT licensed; our experiment is independently implemented.
"""
    (ROOT / 'README.md').write_text(textwrap.dedent(readme), encoding='utf-8')

    slides = f"""# Progress review: eight-slide outline

All measured numbers below come from `results/`. Architecture and timeline
items are proposals. This is an outline, not a generated PowerPoint deck.

## 1. Problem — which service actually started the incident?

- A distributed application can show errors in several services at once.
- Task: rank service/candidate names from incident metrics.
- Ground truth is the service where the benchmark fault was injected.
- Current setting: offline localization with a supplied injection timestamp.

**Visual:** `results/fig_case_metrics.png`; explain the dashed injection line.

## 2. Motivation — the loudest service can be a victim

- Example `{victim['case_id']}`: packet loss injected into `{root_detail['service']}`; its score is only {root_detail['score']:,.2f}.
- Its caller `{nonroot_detail['service']}` scores {nonroot_detail['score']:,.2f} on {nonroot_detail['winning_metric']}, about {ratio:.0f}x louder.
- Constant metrics are ignored, so this is not a scaling artifact; {len(hit_table)} of {audit['n_evaluable']} cases still have a louder non-root.
- Propagation through the caller is the hypothesis this project tests; the plot alone does not prove it.

**Figure:** `results/fig_victim_louder_than_culprit.png`.

## 3. Why ML — learn which anomaly patterns identify roots

- Current heuristic: maximum standardized before/after mean shift per service.
- It ignores dependencies: even after the constant-metric fix, packet-loss roots are often outranked by their callers.
- Proposed learned component: encode temporal anomaly patterns, then predict service relevance.
- Compare a learned independent ranker with a graph model to test the value of dependencies.
- A GNN is not automatically novel or better; benchmark results must justify it.

**Visual/table:** show the baseline formula next to the encoder → dependency GNN → ranking proposal.

## 4. Related work — place the proposal among existing approaches

- MicroRCA: attributed-graph reasoning about performance symptoms and resource measurements.
- RCD / CIRCA: localized causal discovery / conditional-distribution intervention recognition.
- BARO: robust change-point-based RCA; optional baseline skipped in this environment.
- Eadro: joint detection and localization from multiple telemetry sources; RCAEval: shared datasets and evaluation.

**Table:** use all six rows from [related_work.md](related_work.md), retaining its primary-source links.
Do not copy published accuracy numbers into our results table.

## 5. Dataset — actual RE1-OB recordings

- {audit['n_cases']} Online Boutique cases; five fault types, 25 cases each; five distinct annotated roots.
- {audit['metric_count_range'][0]}–{audit['metric_count_range'][1]} metric columns and {audit['service_count_range'][0]}–{audit['service_count_range'][1]} candidate prefixes per case; some prefixes represent infrastructure or unexpected source names.
- {audit['row_count_range'][0]}–{audit['row_count_range'][1]:,} rows per recording; varying pre/post recording lengths.
- {audit['missing_values']:,} missing cells in {audit['cases_with_missing_values']} recordings; no cross-window filling.
- {audit['n_evaluable']} usable window pairs; two supplied recordings remain uncorrected and explicitly flagged.

**Table:** `results/dataset_overview.csv` grouped by fault; schema/examples in `docs/DATA_FORMAT.md`.

## 6. Baseline results — two baselines and their failures

- Baseline 1, all metrics (all {audit['n_cases']} cases): AC@1={percent(overall['AC@1'])}, AC@3={percent(overall['AC@3'])}, Avg@5={percent(overall['Avg@5'])}.
- Why it failed: metrics constant before injection get divided by 1e-8, giving {len(artifact_hits)} non-root wins (e.g. two error blips beat a CPU jump to 97%).
- Baseline 2, constant metrics ignored: AC@1={percent(strong_overall['AC@1'])}, AC@3={percent(strong_overall['AC@3'])}, AC@5={percent(strong_overall['AC@5'])}, Avg@5={percent(strong_overall['Avg@5'])} (evaluable-only AC@1={percent(strong_valid['AC@1'])}).
- Packet loss stays hardest (AC@1={percent(strong_loss)}); worst case `{worst.iloc[0]['case_id']}` has the root at rank {int(worst.iloc[0]['rank_of_true_root'])}.
- The fix was chosen after seeing these failures, so it is an analysis result, not a held-out test; learned-model accuracy is **not yet run**.

**Figures:** `results/fig_baseline_comparison.png`, `results/fig_zero_variance_artifact.png`.
**Tables:** `results/zscore_summary.csv`, `results/zscore_nonconst_summary.csv`, `results/zscore_nonconst_worst_cases.csv`.
**Speaker notes:** [failure_analysis.md](failure_analysis.md).

## 7. Proposed method — learned encoding, graph reasoning, ranking

- Encode each service's normalized metric sequence into an anomaly representation.
- Pass representations through a GNN over a separately verified service dependency graph, then output service ranks.
- Train later on synthetic/controlled fault-injection scenarios; telemetry should be collected from running applications. Purely simulated series must be identified separately.
- Split by whole incidents/scenarios; compare with the same encoder without graph edges and stronger statistical baselines.
- Cross-system test: train on one application and reserve a different application; graph acquisition, training, and that test are **not yet run**.

**Diagram:** metrics → learned anomaly encoder → dependency GNN → service ranking.
**Status:** RE1-OB does not supply traces or an audited dependency graph. No new fault injection this week.

## 8. Semester timeline — staged experiments

- Week 1, completed: official subset, data audit, two z-score baselines, executed notebooks, review outline.
- Weeks 2–3: resolve data-quality questions with upstream evidence; try robust scaling (median/MAD) and aggregation; freeze incident-level splits.
- Weeks 4–6: train a simple temporal encoder/ranker; compare against statistical methods and report uncertainty across runs.
- Weeks 7–9: verify topology, add GNN, and run graph/no-graph ablations; schedule controlled fault data collection only at that stage.
- Weeks 10–12: evaluate held-out scenarios, attempt a reserved-system test if data is available, analyze failures, and package the report; broader deployment is capstone work.

**Visual:** a simple week-by-week timeline built from these five stages; no invented completion dates or performance targets.
"""
    (ROOT / 'docs/slide_outline.md').write_text(textwrap.dedent(slides), encoding='utf-8')
    print('Updated README.md and docs/slide_outline.md from saved results.')


if __name__ == '__main__':
    main()
