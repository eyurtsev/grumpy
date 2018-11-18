import unittest

import numpy as np

from grumpy.metrics import make_classification_report, make_regression_summary_stats


class TestModelInspection(unittest.TestCase):
    def test_make_generic_report(self):
        # Test using empty array
        y = np.array([])
        metrics = make_regression_summary_stats(y, y)
        self.assertEqual(metrics["num_samples"], len(y))

        # Test using valid data.
        y = np.array([0, 1])
        yhat = y

        metrics = make_regression_summary_stats(y, yhat)
        output_keys = set(metrics.keys())

        required_keys = {
            "r2",  # Check that at least one default metric exists.
            "hat_mean",  # Check that at least one summary stat exists.
            "num_samples",  # Should always be present.
        }

        for required_key in required_keys:
            self.assertIn(required_key, output_keys)

        self.assertEqual(metrics["num_samples"], len(y))

    def test_make_classification_report(self):
        y = np.array([0, 1])
        proba = np.array([0.2, 0.9])
        yhat = (proba > 0.5).astype(int)
        report = make_classification_report(y, yhat, proba)
        expected_result = {
            "accuracy": 1.0,
            "fscore": 1.0,
            "hat_mean": 0.5,
            "hat_std": 0.70710678118654757,
            "matthews_corrcoef": 1.0,
            "num_samples": 2,
            "precision": 1.0,
            "recall": 1.0,
            "target_mean": 0.5,
            "target_std": 0.70710678118654757,
        }

        for key, expected_value in expected_result.items():
            self.assertAlmostEqual(expected_value, report[key], places=4)
