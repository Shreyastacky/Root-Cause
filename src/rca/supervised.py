"""Fixed Random Forest experiment with scenario-grouped out-of-fold rankings."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from rca.baselines import InsufficientWindowError, METHODS
from rca.data import DEFAULT_DATA_DIR, PROJECT_ROOT, list_cases, load_case, service_names
from rca.features import FEATURE_COLUMNS, extract_service_features
from rca.metrics import rank_of_root, summarize


RF_PARAMETERS = dict(n_estimators=300, max_depth=6, min_samples_leaf=3,
                     max_features="sqrt", class_weight="balanced_subsample",
                     random_state=42, n_jobs=1)
N_FOLDS = 5
OUTPUT_DIR = PROJECT_ROOT / "results/supervised"


def build_feature_dataset(data_dir=DEFAULT_DATA_DIR):
    """Keep the model matrix separate from labels and identifiers."""
    matrices, labels, incidents = [], [], []
    for case in list_cases(data_dir):
        metrics, injection, root, fault, system = load_case(case)
        incident = {"case_id": case.case_id, "system": system, "fault_type": fault,
                    "true_root": root, "scenario_group": f"{root}|{fault}",
                    "n_candidates": len(service_names(metrics)), "status": "ok", "error": ""}
        try:
            matrix = extract_service_features(metrics, injection)
        except InsufficientWindowError as exc:
            incident.update(status="invalid_window", error=str(exc))
            incidents.append(incident)
            continue
        metadata = pd.DataFrame({"case_id": case.case_id, "service": matrix.index,
                                 "target": (matrix.index == root).astype(int)})
        if metadata["target"].sum() != 1:
            raise ValueError(f"Expected exactly one root among candidates: {case.case_id}")
        matrices.append(matrix.reset_index(drop=True))
        labels.append(metadata.reset_index(drop=True))
        incidents.append(incident)
    return (pd.concat(matrices, ignore_index=True), pd.concat(labels, ignore_index=True),
            pd.DataFrame(incidents))


def make_splits(incidents: pd.DataFrame, n_splits: int = N_FOLDS) -> list[dict]:
    """Split incidents, then expand to service rows; repetitions never cross folds."""
    valid = incidents.loc[incidents["status"].eq("ok")].sort_values("case_id").reset_index(drop=True)
    if valid["case_id"].duplicated().any():
        raise ValueError("Duplicate incident IDs.")
    if valid["scenario_group"].nunique() < n_splits:
        raise ValueError("Not enough independent scenario groups for the requested folds.")
    folds = []
    for number, (train, test) in enumerate(GroupKFold(n_splits).split(valid, groups=valid["scenario_group"]), 1):
        train_cases, test_cases = valid.iloc[train], valid.iloc[test]
        fold = {"fold": number, "train_case_ids": sorted(train_cases["case_id"].tolist()),
                "test_case_ids": sorted(test_cases["case_id"].tolist()),
                "train_groups": sorted(train_cases["scenario_group"].unique().tolist()),
                "test_groups": sorted(test_cases["scenario_group"].unique().tolist())}
        if set(fold["train_case_ids"]) & set(fold["test_case_ids"]):
            raise ValueError("Incident leakage in fold.")
        if set(fold["train_groups"]) & set(fold["test_groups"]):
            raise ValueError("Scenario leakage in fold.")
        folds.append(fold)
    tested = [case_id for fold in folds for case_id in fold["test_case_ids"]]
    if len(tested) != len(set(tested)) or set(tested) != set(valid["case_id"]):
        raise ValueError("Every valid incident must be held out exactly once.")
    return folds


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMETERS)


def rank_candidate_scores(services, scores) -> list[str]:
    """Deterministic per-incident ranking from uncalibrated class-1 scores."""
    if len(services) != len(scores) or len(set(services)) != len(services):
        raise ValueError("Scores must correspond to unique candidates.")
    if not np.isfinite(scores).all():
        raise ValueError("Candidate scores must be finite.")
    return [service for service, _ in sorted(zip(services, scores), key=lambda item: (-item[1], item[0]))]


def run_experiment(data_dir=DEFAULT_DATA_DIR, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features, metadata, incidents = build_feature_dataset(data_dir)
    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("The fixed feature schema changed.")
    folds = make_splits(incidents)
    protocol_file = PROJECT_ROOT / "docs/ML_PROTOCOL.md"
    # Freeze the split and model configuration BEFORE fitting any model.
    manifest = {"experiment": "random_forest_grouped_cv", "n_folds": N_FOLDS,
                "group_definition": "annotated root service + fault type; all repetitions together",
                "model_parameters": RF_PARAMETERS, "feature_columns": FEATURE_COLUMNS,
                "n_incidents": len(incidents), "n_evaluable": int(incidents["status"].eq("ok").sum()),
                "n_scenario_groups": incidents.loc[incidents["status"].eq("ok"), "scenario_group"].nunique(),
                "n_service_rows": len(features), "sklearn_version": sklearn.__version__,
                "protocol_sha256": hashlib.sha256(protocol_file.read_bytes()).hexdigest(),
                "folds": folds, "selection": "fixed model; no hyperparameter search",
                "limitation": "exploratory grouped CV on already-explored RE1-OB; no untouched or cross-system test"}
    for name, filename in [("features.py", "feature_source_sha256"), ("supervised.py", "experiment_source_sha256")]:
        manifest[filename] = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    features.to_csv(output_dir / "service_features.csv", index=False)
    metadata.to_csv(output_dir / "service_metadata.csv", index=False)
    incidents.to_csv(output_dir / "incidents.csv", index=False)

    score_rows, importance_rows = [], []
    fold_by_case = {}
    for fold in folds:
        train_mask = metadata["case_id"].isin(fold["train_case_ids"])
        test_mask = metadata["case_id"].isin(fold["test_case_ids"])
        model = make_model()
        # Only these numeric columns enter sklearn: index/metadata cannot become inputs.
        model.fit(features.loc[train_mask, FEATURE_COLUMNS].to_numpy(), metadata.loc[train_mask, "target"].to_numpy())
        positive_column = list(model.classes_).index(1)
        scores = model.predict_proba(features.loc[test_mask, FEATURE_COLUMNS].to_numpy())[:, positive_column]
        held_out = metadata.loc[test_mask, ["case_id", "service"]].copy()
        held_out["score"] = scores
        held_out["fold"] = fold["fold"]
        score_rows.extend(held_out.to_dict("records"))
        importance_rows.extend({"fold": fold["fold"], "feature": feature, "importance": float(importance)}
                               for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_))
        fold_by_case.update({case_id: fold["fold"] for case_id in fold["test_case_ids"]})
        print(f"Fold {fold['fold']}: trained on {len(fold['train_case_ids'])} incidents; "
              f"predicted {len(fold['test_case_ids'])} held-out incidents.", flush=True)
    predictions = pd.DataFrame(score_rows).sort_values(["case_id", "service"]).reset_index(drop=True)
    predictions.to_csv(output_dir / "random_forest_oof_scores.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(output_dir / "feature_importance_per_fold.csv", index=False)
    importance.groupby("feature")["importance"].agg(["mean", "std"]).sort_values("mean", ascending=False).to_csv(
        output_dir / "feature_importance.csv")

    cases = {case.case_id: case for case in list_cases(data_dir)}
    scores_by_case = {case_id: frame for case_id, frame in predictions.groupby("case_id")}
    rows = []
    for incident in incidents.to_dict("records"):
        for method in ["zscore", "zscore_nonconst", "random_forest"]:
            error, status = incident["error"], incident["status"]
            ranking, rank = [], np.inf
            if status == "ok":
                if method == "random_forest":
                    frame = scores_by_case[incident["case_id"]]
                    ranking = rank_candidate_scores(frame["service"].tolist(), frame["score"].to_numpy())
                else:
                    measurements, injection, _, _, _ = load_case(cases[incident["case_id"]])
                    ranking = METHODS[method][0](measurements, injection)
                rank = rank_of_root(ranking, incident["true_root"])
                if len(ranking) != incident["n_candidates"]:
                    raise ValueError("Ranking omitted a candidate.")
            rows.append({"method": method, "case_id": incident["case_id"], "system": incident["system"],
                         "fault_type": incident["fault_type"], "true_root": incident["true_root"],
                         "scenario_group": incident["scenario_group"], "fold": fold_by_case.get(incident["case_id"], -1),
                         "n_candidates": incident["n_candidates"], "predicted_ranking": json.dumps(ranking),
                         "rank_of_true_root": rank, "status": status, "error": error})
    per_case = pd.DataFrame(rows)
    summary = pd.concat([summarize(frame).assign(method=method)
                         for method, frame in per_case.groupby("method", sort=True)], ignore_index=True)
    fold_summary = pd.concat([summarize(frame).query("scope == 'overall' and evaluation_set == 'evaluable_only'").assign(method=method, fold=fold)
                             for (method, fold), frame in per_case.loc[per_case["status"].eq("ok")].groupby(["method", "fold"])], ignore_index=True)
    per_case.to_csv(output_dir / "per_case.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False)
    paired = per_case.loc[per_case["status"].eq("ok")].pivot(index="case_id", columns="method", values="rank_of_true_root")
    paired["rf_top1"] = paired["random_forest"].eq(1)
    paired["heuristic_top1"] = paired["zscore_nonconst"].eq(1)
    paired.to_csv(output_dir / "paired_case_comparison.csv")
    print(summary.query("scope == 'overall' and evaluation_set == 'evaluable_only'")[
        ["method", "n_cases", "AC@1", "AC@3", "AC@5", "Avg@5"]].to_string(index=False))
    return per_case, summary, fold_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_experiment(args.data_dir, args.results_dir)


if __name__ == "__main__":
    main()
