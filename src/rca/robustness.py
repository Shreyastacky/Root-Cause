"""Fixed feature ablations and leave-one-root-service-out diagnostic evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from rca.data import PROJECT_ROOT
from rca.features import FEATURE_COLUMNS
from rca.metrics import rank_of_root, summarize
from rca.supervised import RF_PARAMETERS, rank_candidate_scores


OUTPUT_DIR = PROJECT_ROOT / "results/robustness"
SOURCE_DIR = PROJECT_ROOT / "results/supervised"
REFERENCE = "full_features_scenario_cv"
AVAILABILITY = [name for name in FEATURE_COLUMNS if name.endswith("__available")]
BEHAVIOR = [name for name in FEATURE_COLUMNS if name.rsplit("__", 1)[1] in
            {"signed_log_shift", "relative_shift", "log_std_ratio"}]
LABELS = {REFERENCE: "Full features / scenario CV (reference)",
          "availability_only": "Availability only / same scenario CV",
          "behavior_only": "Behavior only / same scenario CV",
          "held_out_root": "Full features / held-out root service"}


def sha256(path):
    # Git stores these text files with LF; pandas writes CRLF on Windows.
    # Canonical text hashes must survive a clone or notebook rerun on either OS.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def make_root_splits(incidents):
    """Hold out all fault families and repetitions rooted at one service."""
    valid = incidents.loc[incidents["status"].eq("ok")].sort_values("case_id")
    if valid["case_id"].duplicated().any() or valid["true_root"].nunique() < 2:
        raise ValueError("Need unique cases and at least two annotated root services.")
    folds = []
    for number, root in enumerate(sorted(valid["true_root"].unique()), 1):
        mask = valid["true_root"].eq(root)
        folds.append({"fold": number, "held_out_root": root,
                      "train_case_ids": valid.loc[~mask, "case_id"].tolist(),
                      "test_case_ids": valid.loc[mask, "case_id"].tolist()})
    return folds


def run_experiment(output_dir=OUTPUT_DIR):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(SOURCE_DIR / "service_features.csv")
    metadata = pd.read_csv(SOURCE_DIR / "service_metadata.csv")
    incidents = pd.read_csv(SOURCE_DIR / "incidents.csv").fillna({"error": ""})
    original = json.loads((SOURCE_DIR / "split_manifest.json").read_text())
    if features.columns.tolist() != FEATURE_COLUMNS or original["model_parameters"] != RF_PARAMETERS:
        raise ValueError("Original feature schema/model parameters changed; inspect before running.")
    if len(features) != len(metadata) or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature rows must align with metadata and be finite.")
    # Validate that the frozen upstream experiment still matches its own sources.
    for key, path in [("protocol_sha256", PROJECT_ROOT / "docs/ML_PROTOCOL.md"),
                      ("feature_source_sha256", PROJECT_ROOT / "src/rca/features.py"),
                      ("experiment_source_sha256", PROJECT_ROOT / "src/rca/supervised.py")]:
        if original[key] != sha256(path):
            raise ValueError(f"Upstream provenance mismatch: {key}")
    variants = [
        {"method": "availability_only", "columns": AVAILABILITY, "folds": original["folds"]},
        {"method": "behavior_only", "columns": BEHAVIOR, "folds": original["folds"]},
        {"method": "held_out_root", "columns": FEATURE_COLUMNS, "folds": make_root_splits(incidents)},
    ]
    inputs = ["service_features.csv", "service_metadata.csv", "incidents.csv",
              "split_manifest.json", "random_forest_oof_scores.csv", "per_case.csv"]
    paths = [SOURCE_DIR / name for name in inputs] + [
        PROJECT_ROOT / "docs/ROBUSTNESS_PROTOCOL.md", Path(__file__),
        PROJECT_ROOT / "src/rca/features.py", PROJECT_ROOT / "src/rca/supervised.py"]
    manifest = {"experiment": "fixed_robustness_audit", "model_parameters": RF_PARAMETERS,
                "sklearn_version": sklearn.__version__, "variants": variants,
                "hash_encoding": "SHA256 of UTF-8 text bytes with CRLF normalized to LF",
                "reference": "unchanged original scenario-CV out-of-fold predictions",
                "selection": "diagnostics only; no tuning or winning-subset selection",
                "sha256": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in paths}}
    # All splits, features and parameters are on disk before any new model fit.
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    reference_scores = pd.read_csv(SOURCE_DIR / "random_forest_oof_scores.csv").assign(method=REFERENCE)
    all_scores = [reference_scores]
    per_case = pd.read_csv(SOURCE_DIR / "per_case.csv").query("method == 'random_forest'").copy()
    per_case["method"] = REFERENCE
    per_case["held_out_root"] = ""
    all_cases, rows = [per_case], []
    for variant in variants:
        method, columns = variant["method"], variant["columns"]
        score_frames, assignments = [], {}
        for fold in variant["folds"]:
            train = metadata["case_id"].isin(fold["train_case_ids"])
            test = metadata["case_id"].isin(fold["test_case_ids"])
            if set(fold["train_case_ids"]) & set(fold["test_case_ids"]):
                raise ValueError("Train/test incident overlap.")
            forest = RandomForestClassifier(**RF_PARAMETERS)
            forest.fit(features.loc[train, columns].to_numpy(), metadata.loc[train, "target"].to_numpy())
            scored = metadata.loc[test, ["case_id", "service"]].copy()
            scored["score"] = forest.predict_proba(features.loc[test, columns].to_numpy())[:, list(forest.classes_).index(1)]
            scored["fold"], scored["method"] = fold["fold"], method
            score_frames.append(scored)
            assignments.update({case_id: fold for case_id in fold["test_case_ids"]})
            print(f"{method}, fold {fold['fold']}: train {len(fold['train_case_ids'])}, "
                  f"test {len(fold['test_case_ids'])}, features {len(columns)}", flush=True)
        predictions = pd.concat(score_frames, ignore_index=True)
        all_scores.append(predictions)
        scores_by_case = dict(tuple(predictions.groupby("case_id")))
        for incident in incidents.to_dict("records"):
            case_id = incident["case_id"]
            fold = assignments.get(case_id, {})
            ranking = []
            if incident["status"] == "ok":
                frame = scores_by_case[case_id]
                ranking = rank_candidate_scores(frame["service"].tolist(), frame["score"].to_numpy())
                if len(ranking) != incident["n_candidates"]:
                    raise ValueError("Candidate omitted from ranking.")
            rows.append({**incident, "method": method, "fold": fold.get("fold", -1),
                         "held_out_root": fold.get("held_out_root", ""),
                         "predicted_ranking": json.dumps(ranking),
                         "rank_of_true_root": rank_of_root(ranking, incident["true_root"])})
    all_cases.append(pd.DataFrame(rows))
    cases = pd.concat(all_cases, ignore_index=True).fillna({"error": "", "held_out_root": ""})
    scores = pd.concat(all_scores, ignore_index=True).sort_values(["method", "case_id", "service"])
    summary = pd.concat([summarize(frame).assign(method=method) for method, frame in cases.groupby("method")], ignore_index=True)
    folds = pd.concat([summarize(frame).query("scope == 'overall' and evaluation_set == 'evaluable_only'").assign(
        method=method, fold=fold, held_out_root=frame["held_out_root"].iloc[0])
        for (method, fold), frame in cases.loc[cases["status"].eq("ok")].groupby(["method", "fold"])], ignore_index=True)
    scores.to_csv(directory / "candidate_scores.csv", index=False)
    cases.to_csv(directory / "per_case.csv", index=False)
    summary.to_csv(directory / "summary.csv", index=False)
    folds.to_csv(directory / "fold_summary.csv", index=False)
    cases.loc[cases["status"].eq("ok") & cases["rank_of_true_root"].gt(1)].to_csv(directory / "failures.csv", index=False)
    paired = cases.loc[cases["status"].eq("ok")].pivot(index="case_id", columns="method", values="rank_of_true_root")
    paired.to_csv(directory / "paired_ranks.csv")
    print(summary.query("scope == 'overall' and evaluation_set == 'evaluable_only'")[
        ["method", "n_cases", "AC@1", "AC@3", "AC@5", "Avg@5"]].to_string(index=False))
    return cases, summary, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=OUTPUT_DIR)
    run_experiment(parser.parse_args().results_dir)


if __name__ == "__main__":
    main()
