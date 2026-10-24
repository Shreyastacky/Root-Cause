# First supervised experiment: protocol fixed before training

## Question

Does a Random Forest trained on service-level change features rank injected root
services more accurately than the two existing statistical baselines on held-out
fault/service scenarios in RE1-OB?

## Scope and limitations

Use the existing 125 recordings only. The two invalid-window recordings remain
unchanged and unscorable. No new dataset, dependency graph, neural network,
Docker, Kubernetes, or fault injection is introduced.

This dataset was already explored while developing the statistical baselines.
The following split provides out-of-fold predictions for a fixed learned model;
it does not create a pristine, previously untouched final test set. Results are
exploratory within one application, not cross-system or causal evidence.

## Prediction unit and labels

- One feature row represents one candidate service in one incident.
- Label 1 means that candidate is the annotated injected root; all other
  candidates in the incident have label 0.
- Retain every observed candidate prefix, including infrastructure-like names.
- No service name, root name, fault label, timestamp, filename, repetition,
  case ID, or scenario group is a model input.
- Metadata and labels live in a separate table used for splitting/evaluation.
- Available measurement types can indirectly identify service roles or recording
  regimes; omitting names does not eliminate every possible proxy.

## Fixed features

Use the eight observed metric suffixes: cpu, mem, load, workload, error, latency,
latency-50, latency-90. For each suffix, extract six features, giving 48 columns:

1. available: at least two finite normal and one finite anomalous observations;
2. signed_log_shift: sign of mean change times log1p of its absolute standardized
   change; cap the standardized magnitude at 1e12;
3. relative_shift: mean change / (absolute normal mean + absolute anomalous mean
   + 1e-8), a bounded signed change;
4. log_std_ratio: log((anomalous sample std + 1e-8)/(normal sample std + 1e-8)),
   clipped to [-20, 20]; a single anomalous observation has std 0 by convention;
5. normal_constant: normal sample standard deviation is exactly zero;
6. zero_to_nonzero: all normal observations were zero and some anomalous
   observations are nonzero.

Unavailable metric features are all zero, with available=0. NaN observations
are omitted within their own window; no cross-window filling. These are fixed,
per-incident transforms: no global imputer, scaler, or selector is fitted to the
complete dataset. The forest needs no feature standardization. Features retain
constant/onset information rather than simply dropping it.

## Split fixed before inspecting model performance

- Five-fold `GroupKFold`, applied to evaluable **incidents**.
- Scenario group = (root service, fault type); all repetitions stay together.
- Expand train/test incident sets to their candidate rows only after splitting.
- Assert disjoint case IDs and disjoint scenario groups in every fold.
- Each evaluable incident receives exactly one held-out prediction.
- Save the full split manifest and configuration before fitting the first fold.
- No hyperparameter search or selection of the best seed/fold after results.

Grouping uses annotations to prevent scenario overlap, never as predictive
features. This tests unseen combinations of known services and fault types; it
does not hold out an entire service identity, fault family, or application.

## Fixed model

`RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
max_features='sqrt', class_weight='balanced_subsample', random_state=42,
n_jobs=1)` from scikit-learn 1.5.2. All remaining parameters use that version's
defaults. Fit one separate forest per fold. Class weighting handles the one-root
versus many-non-root imbalance; no resampling/SMOTE is used.

Rank each incident's candidates by the class-1 `predict_proba` score, breaking
exact ties alphabetically. These are uncalibrated ranking scores, not reliable
incident-level confidence probabilities. Do not fit a final all-data model and
score it on its training incidents for the reported experiment.

## Comparison and reporting

Compare Random Forest, zscore, and zscore_nonconst on the same test incidents in
each fold. Primary table: pooled out-of-fold results on 123 evaluable incidents.
Also report all 125 supplied incidents with invalid cases counted as misses.
Report AC@1..5, Avg@5, fold results, fault breakdowns, and wins/losses versus the
stronger statistical baseline. Feature importance is descriptive impurity-based
importance averaged across folds, not causal explanation.

Regardless of the outcome, keep this fixed experiment and report it. A result
below the heuristic baseline is useful evidence, not a reason to change the
split or hide the model.

## References

- [Random Forest API, scikit-learn 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [GroupKFold API, scikit-learn 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GroupKFold.html)
- [Dataset and original study](https://github.com/phamquiluan/RCAEval)
