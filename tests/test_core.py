"""Tiny constructed examples test mathematics; these are NOT benchmark data."""

import unittest

import numpy as np
import pandas as pd

from rca.baselines import (EPS, InsufficientWindowError, metric_anomaly_scores,
                           zscore_nonconst_rank, zscore_rank)
from rca.data import prepare_metrics
from rca.metrics import ac_at_k, avg_at_k, rank_of_root, summarize


class BaselineTests(unittest.TestCase):
    def test_formula_boundary_and_max_aggregation(self):
        # Injection at t=2: two observations per window. pre sample std=sqrt(2).
        frame = pd.DataFrame({"time": [0, 1, 2, 3], "a_cpu": [0, 2, 4, 6],
                              "a_mem": [5, 5, 5, 5], "b_cpu": [0, 2, 1, 1]})
        stats = metric_anomaly_scores(frame, 2)
        self.assertAlmostEqual(stats.loc[("a", "cpu"), "z"], 4 / (np.sqrt(2) + EPS))
        self.assertEqual(zscore_rank(frame, 2), ["a", "b"])
        self.assertEqual(frame.columns.tolist()[0], "time")  # input is not mutated

    def test_missing_values_are_not_filled_across_injection(self):
        frame = pd.DataFrame({"time": [0, 1, 2, 3], "a_cpu": [np.nan, np.nan, 10, 10],
                              "b_cpu": [0, 2, 3, np.nan], "c_cpu": [0, 2, np.nan, np.nan]})
        stats = metric_anomaly_scores(frame, 2)
        self.assertFalse(stats.loc[("a", "cpu"), "valid"])
        self.assertFalse(stats.loc[("c", "cpu"), "valid"])
        self.assertEqual(stats.loc[("b", "cpu"), "anomalous_count"], 1)
        self.assertEqual(zscore_rank(frame, 2), ["b", "a", "c"])

    def test_constant_metrics_ties_and_no_label_dependence(self):
        raw = pd.DataFrame({"time": [0, 1, 2, 3], "b_cpu": [0, 0, 1, 1],
                            "a_cpu": [0, 0, 1, 1]})
        frame = prepare_metrics(raw)
        frame.attrs["fake_root_label_for_test"] = "b"
        first = zscore_rank(frame, 2)
        frame.attrs["fake_root_label_for_test"] = "a"
        self.assertEqual(first, zscore_rank(frame, 2))
        self.assertEqual(first, ["a", "b"])
        self.assertTrue(np.isfinite(metric_anomaly_scores(frame, 2)["z"]).all())

    def test_nonconst_ignores_metrics_constant_before_injection(self):
        # a_err is 0 before injection, so plain zscore divides a tiny blip by eps.
        frame = pd.DataFrame({"time": [0, 1, 2, 3, 4, 5], "a_err": [0, 0, 0, 0, 0, .01],
                              "b_cpu": [1, 2, 1, 50, 50, 50]})
        self.assertEqual(zscore_rank(frame, 3), ["a", "b"])
        self.assertEqual(zscore_nonconst_rank(frame, 3), ["b", "a"])

    def test_invalid_windows_fail_instead_of_inventing_scores(self):
        frame = pd.DataFrame({"time": [0, 1, 2], "a_cpu": [1, 2, 3]})
        for injection in (-10, 10):
            with self.assertRaises(InsufficientWindowError):
                zscore_rank(frame, injection)


class EvaluationTests(unittest.TestCase):
    def test_known_ranks_and_avg_definition(self):
        ranks = [1, 2, 4, np.inf]
        self.assertEqual(ac_at_k(ranks, 1), .25)
        self.assertEqual(ac_at_k(ranks, 3), .5)
        self.assertEqual(ac_at_k(ranks, 5), .75)
        self.assertEqual(avg_at_k(ranks), (.25 + .5 + .5 + .75 + .75) / 5)
        self.assertTrue(np.isinf(rank_of_root(["a"], "b")))
        with self.assertRaises(ValueError):
            rank_of_root(["a", "a"], "a")

    def test_invalid_cases_have_explicit_denominators(self):
        cases = pd.DataFrame({"fault_type": ["cpu", "cpu"],
                              "rank_of_true_root": [1, np.inf],
                              "status": ["ok", "invalid_window"]})
        summary = summarize(cases)
        all_row = summary.query("evaluation_set == 'all_cases' and scope == 'overall'").iloc[0]
        valid_row = summary.query("evaluation_set == 'evaluable_only' and scope == 'overall'").iloc[0]
        self.assertEqual(all_row["AC@1"], .5)
        self.assertEqual(valid_row["AC@1"], 1)
        self.assertEqual(all_row["n_cases"], 2)
        self.assertEqual(valid_row["n_cases"], 1)


if __name__ == "__main__":
    unittest.main()
