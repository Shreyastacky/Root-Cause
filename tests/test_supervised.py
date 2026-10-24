"""Constructed fixtures test feature meaning and scenario isolation, not accuracy."""

import unittest

import numpy as np
import pandas as pd

from rca.baselines import InsufficientWindowError
from rca.features import FEATURE_COLUMNS, extract_service_features
from rca.supervised import RF_PARAMETERS, make_model, make_splits, rank_candidate_scores


class FeatureTests(unittest.TestCase):
    def test_fixed_numeric_features_have_expected_meaning(self):
        frame = pd.DataFrame({"time": [0, 1, 2, 3], "alpha_cpu": [0, 2, 4, 6]})
        features = extract_service_features(frame, 2)
        self.assertEqual(features.columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(features.shape, (1, 48))
        self.assertAlmostEqual(features.loc["alpha", "cpu__signed_log_shift"],
                               np.log1p(4 / (np.sqrt(2) + 1e-8)))
        self.assertAlmostEqual(features.loc["alpha", "cpu__relative_shift"], 4 / (6 + 1e-8))
        self.assertEqual(features.loc["alpha", "mem__available"], 0)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_constant_onset_and_missing_normal_window(self):
        frame = pd.DataFrame({"time": [0, 1, 2, 3], "a_error": [0, 0, 0, 1],
                              "b_mem": [np.nan, np.nan, 2, 3]})
        features = extract_service_features(frame, 2)
        self.assertEqual(features.loc["a", "error__normal_constant"], 1)
        self.assertEqual(features.loc["a", "error__zero_to_nonzero"], 1)
        self.assertEqual(features.loc["b", "mem__available"], 0)
        self.assertTrue(features.loc["b"].eq(0).all())

    def test_service_names_and_attrs_do_not_change_feature_values(self):
        raw = pd.DataFrame({"time": [0, 1, 2, 3], "alpha_cpu": [1, 2, 3, 4]})
        raw.attrs["root_label"] = "alpha"
        first = extract_service_features(raw, 2).to_numpy()
        renamed = raw.rename(columns={"alpha_cpu": "other_cpu"})
        renamed.attrs["root_label"] = "unrelated"
        np.testing.assert_array_equal(first, extract_service_features(renamed, 2).to_numpy())
        self.assertFalse(any(token in FEATURE_COLUMNS for token in ["target", "case_id", "service", "fault_type"]))

    def test_no_feature_extrapolation_for_invalid_injection(self):
        frame = pd.DataFrame({"time": [0, 1, 2], "a_cpu": [1, 2, 3]})
        for injection in [-10, 10]:
            with self.assertRaises(InsufficientWindowError):
                extract_service_features(frame, injection)


class SplitTests(unittest.TestCase):
    def test_repeated_scenarios_and_incidents_never_cross_folds(self):
        rows = [{"case_id": f"case_{group}_{repeat}", "scenario_group": f"scenario_{group}", "status": "ok"}
                for group in range(6) for repeat in range(3)]
        rows.append({"case_id": "invalid", "scenario_group": "invalid", "status": "invalid_window"})
        incidents = pd.DataFrame(rows)
        folds = make_splits(incidents, n_splits=3)
        tested = []
        for fold in folds:
            self.assertFalse(set(fold["train_case_ids"]) & set(fold["test_case_ids"]))
            self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
            self.assertNotIn("invalid", fold["test_case_ids"])
            tested.extend(fold["test_case_ids"])
        self.assertEqual(len(tested), 18)
        self.assertEqual(len(set(tested)), 18)
        self.assertEqual(folds, make_splits(incidents.sample(frac=1, random_state=7), n_splits=3))

    def test_fixed_classifier_and_deterministic_ranking(self):
        model = make_model()
        for parameter, value in RF_PARAMETERS.items():
            self.assertEqual(model.get_params()[parameter], value)
        self.assertEqual(rank_candidate_scores(["b", "a", "c"], [.5, .5, .8]), ["c", "a", "b"])
        with self.assertRaises(ValueError):
            rank_candidate_scores(["a", "a"], [.1, .2])
        with self.assertRaises(ValueError):
            rank_candidate_scores(["a"], [np.nan])


if __name__ == "__main__":
    unittest.main()
