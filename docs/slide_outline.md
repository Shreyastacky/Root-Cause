# Progress review: eight-slide outline

All measured numbers below come from `results/`. Architecture and timeline
items are proposals. This is an outline, not a generated PowerPoint deck.

## 1. Problem — which service actually started the incident?

- A distributed application can show errors in several services at once.
- Task: rank service/candidate names from incident metrics.
- Ground truth is the service where the benchmark fault was injected.
- Current setting: offline localization with a supplied injection timestamp.

**Visual:** `results/fig_case_metrics.png`; explain the dashed injection line.

## 2. Motivation — the loudest service can be a victim

- Example `re1ob_productcatalogservice_loss_3`: packet loss injected into `productcatalogservice`; its score is only 14.99.
- Its caller `checkoutservice` scores 171.58 on latency-90, about 11x louder.
- Constant metrics are ignored, so this is not a scaling artifact; 28 of 123 cases still have a louder non-root.
- Propagation through the caller is the hypothesis this project tests; the plot alone does not prove it.

**Figure:** `results/fig_victim_louder_than_culprit.png`.

## 3. ML method — learn combinations of service changes

- Completed: a Random Forest classifier scores root versus non-root candidates.
- Inputs: 48 fixed features describing per-metric changes, variability, availability and onset.
- Service names, fault labels and case IDs are excluded from the feature matrix.
- Rank each incident's candidates by its forest's class-1 score; no calibrated-confidence claim.
- A temporal encoder and GNN remain later experiments, not implemented methods.

**Visual:** feature extraction → forest trained on other scenario groups → held-out service ranking.

## 4. Related work — place the proposal among existing approaches

- MicroRCA: attributed-graph reasoning about performance symptoms and resource measurements.
- RCD / CIRCA: localized causal discovery / conditional-distribution intervention recognition.
- BARO: robust change-point-based RCA; optional baseline skipped in this environment.
- Eadro: joint detection and localization from multiple telemetry sources; RCAEval: shared datasets and evaluation.

**Table:** use all six rows from [related_work.md](related_work.md), retaining its primary-source links.
Do not copy published accuracy numbers into our results table.

## 5. Dataset — actual RE1-OB recordings

- 125 Online Boutique cases; five fault types, 25 cases each; five distinct annotated roots.
- 49–59 metric columns and 13–19 candidate prefixes per case; some prefixes represent infrastructure or unexpected source names.
- 63–4,201 rows per recording; varying pre/post recording lengths.
- 5,243 missing cells in 10 recordings; no cross-window filling.
- 123 usable window pairs; two supplied recordings remain uncorrected and explicitly flagged.

**Table:** `results/dataset_overview.csv` grouped by fault; schema/examples in `docs/DATA_FORMAT.md`.

## 6. Grouped evaluation — first trained baseline

- Fixed five-fold GroupKFold: all repetitions of each root/fault scenario stay together; 25 groups.
- Primary comparison on 123 evaluable incidents: Random Forest AC@1=91.87%, AC@3=95.93%, Avg@5=96.10%; stronger heuristic AC@1=77.24%.
- All 125 supplied incidents, invalid cases counted as misses: forest AC@1=90.40%.
- Fold AC@1 varies from 68% to 100%; packet-loss failures remain. The forest's AC@5 is slightly worse than the stronger heuristic.
- This is exploratory within-application CV on previously explored data, not cross-system or pristine final-test performance.

**Figure:** `results/supervised/fig_grouped_cv.png`.
**Tables:** `results/supervised/summary.csv`, `results/supervised/fold_summary.csv`.
**Notes:** [SUPERVISED_BASELINE.md](SUPERVISED_BASELINE.md).

## 7. Robustness audit — feature shortcuts and new fault locations

- Availability only: AC@1=27.64%; behavior only: 92.68%, using the original scenario folds and fixed forest parameters.
- Full features, with every incident rooted at the test service excluded from training: AC@1=93.50% on 123 evaluable incidents.
- A held-out root may still appear as a negative candidate in training; this is a different task from scenario CV, not cross-system transfer.
- Keep every failure and both invalid cases visible; missing-feature zeros can still reveal schema after explicit flags are removed.
- Next: reserve new data, test fixed observation windows and robust statistics; verify topology before a temporal/GNN extension.

**Figure:** `results/robustness/fig_robustness.png`.
**Notes:** [ROBUSTNESS_AUDIT.md](ROBUSTNESS_AUDIT.md).

## 8. Semester timeline — staged experiments

- Week 1, completed: official subset, data audit, two z-score baselines, executed notebooks, review outline.
- Next stage, completed: fixed Random Forest, scenario-grouped five-fold evaluation, supervised notebook and failure analysis; later reserve genuinely new data.
- Weeks 4–6: train a simple temporal encoder/ranker; compare against statistical methods and report uncertainty across runs.
- Weeks 7–9: verify topology, add GNN, and run graph/no-graph ablations; schedule controlled fault data collection only at that stage.
- Weeks 10–12: evaluate held-out scenarios, attempt a reserved-system test if data is available, analyze failures, and package the report; broader deployment is capstone work.

**Visual:** a simple week-by-week timeline built from these five stages; no invented completion dates or performance targets.
