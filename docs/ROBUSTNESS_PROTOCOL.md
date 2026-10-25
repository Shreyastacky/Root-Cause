# Feature and service generalization audit

This protocol is written before running this audit. The existing RE1-OB results
have already been inspected. This is a diagnostic study, not a new untouched
test set, model-selection exercise, or claim of production generalization.

## Fixed questions and experiments

Reuse the original 48 numeric features and Random Forest settings from
`docs/ML_PROTOCOL.md`. No hyperparameter, seed, feature-subset, or fold search.
Keep every observed candidate, alphabetical score ties, and both denominator
policies: 123 valid incidents, and all 125 with invalid recordings as misses.

1. **Reference:** reuse the existing 48-feature scenario-grouped out-of-fold
   predictions unchanged. Do not refit or replace the original experiment.
2. **Availability only:** train on the eight `__available` features using those
   exact same scenario folds. There are no measured change values. This checks
   whether measurement availability alone can help discriminate root candidates.
   Availability includes finite-observation requirements, so it is not purely
   a static inventory of metric names.
3. **Behavior only:** train on the 24 signed log shift, relative shift, and log
   standard-deviation ratio features, using the same scenario folds. Remove the
   explicit availability, constant-window and zero-onset flags. Missing features
   still contain zero and metric types retain their names; schema information
   can remain implicit. A result here cannot prove absence of shortcuts.
4. **Hold out root service:** train the original 48-feature forest five times,
   each time excluding every incident whose annotated root is one selected
   service, and test on all valid incidents rooted at that service. Every valid
   incident is tested once. No training incident has the held-out root label.
   That service may still appear as a **negative candidate** in other incidents.
   This tests a new fault location within Online Boutique, not a previously
   unseen service, application, workload, or fault family.

The two ablations isolate feature-set changes with identical folds. The last
experiment changes the training partition as well as its difficulty; its score
is a separate task result, not an apples-to-apples feature improvement.

## Evidence and interpretation

Save the full feature lists, model parameters, explicit train/test case IDs,
source/protocol/input hashes, and software versions **before fitting**. Save all
candidate scores, rankings, fold results, failures, and paired top-one changes.
Use raw per-incident AC@1/3/5 and Avg@5, not binary candidate accuracy. Report
pooled scores and every held-out service. Never pick a winner and overwrite the
original baseline. Do not treat class-1 scores as calibrated confidence.

Independently refit all 15 new forests from the saved manifest, verify every
candidate score and ranking, verify case/scenario or root-service isolation,
and recalculate the summary denominators. Preserve both invalid cases. Execute
the explanatory fourth notebook and retain its outputs and rendered HTML.

Any repeated use of these data for later design decisions is exploratory.
Reserve genuinely new data before final model selection. These experiments do
not test early detection, fixed observation budgets, or cross-system transfer.
