"""Check the scientific contracts of the diagnostic splits and feature ablations."""

import unittest

import numpy as np
import pandas as pd

from rca.features import extract_service_features
from rca.robustness import AVAILABILITY, BEHAVIOR, make_root_splits


class RobustnessTests(unittest.TestCase):
    def test_all_faults_and_repetitions_of_root_are_held_out(self):
        incidents = pd.DataFrame([
            {"case_id": f"{root}_{fault}_{repeat}", "true_root": root,
             "fault_type": fault, "status": "ok"}
            for root in ["alpha", "beta", "gamma"] for fault in ["cpu", "loss"]
            for repeat in range(2)] + [
                {"case_id": "invalid", "true_root": "alpha", "fault_type": "cpu", "status": "invalid_window"}])
        tested = []
        for fold in make_root_splits(incidents):
            train = incidents.loc[incidents.case_id.isin(fold["train_case_ids"])]
            test = incidents.loc[incidents.case_id.isin(fold["test_case_ids"])]
            self.assertNotIn(fold["held_out_root"], train.true_root.tolist())
            self.assertEqual(set(test.true_root), {fold["held_out_root"]})
            self.assertEqual(set(test.fault_type), {"cpu", "loss"})
            self.assertFalse(set(train.case_id) & set(test.case_id))
            self.assertNotIn("invalid", fold["test_case_ids"])
            tested.extend(test.case_id)
        self.assertEqual(len(tested), len(set(tested)))
        self.assertEqual(len(tested), 12)
        self.assertEqual(make_root_splits(incidents), make_root_splits(incidents.sample(frac=1, random_state=9)))

    def test_availability_does_not_see_changed_metric_magnitudes(self):
        raw = pd.DataFrame({"time": [0, 1, 2, 3], "alpha_cpu": [1, 2, 3, 4]})
        altered = raw.copy()
        altered.loc[2:, "alpha_cpu"] = [100, 200]
        before = extract_service_features(raw, 2)
        after = extract_service_features(altered, 2)
        np.testing.assert_array_equal(before[AVAILABILITY], after[AVAILABILITY])
        self.assertFalse(np.array_equal(before[BEHAVIOR], after[BEHAVIOR]))
        self.assertEqual(before.loc["alpha", "mem__available"], 0)
        # Removing explicit flags still leaves zero as an implicit absence signal.
        self.assertTrue(before.loc["alpha", [c for c in BEHAVIOR if c.startswith("mem__")]].eq(0).all())

    def test_unavailable_metric_requires_observations_in_both_windows(self):
        raw = pd.DataFrame({"time": [0, 1, 2, 3], "alpha_cpu": [np.nan, 2, 3, 4]})
        self.assertEqual(extract_service_features(raw, 2).loc["alpha", "cpu__available"], 0)

    def test_degenerate_root_splits_are_rejected(self):
        incidents = pd.DataFrame([{"case_id": "one", "true_root": "alpha", "status": "ok"}])
        with self.assertRaises(ValueError):
            make_root_splits(incidents)


if __name__ == "__main__":
    unittest.main()
