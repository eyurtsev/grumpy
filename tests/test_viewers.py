import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from grumpy import cross_validation, typedefs
from grumpy.model_wrappers import Fold
from grumpy.core import CrossValidator
from grumpy.viewers import CVViewer, AbstractFoldViewer


class TestCrossValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = pd.DataFrame(
            [{"x": 1, "y": False}, {"x": 2, "y": False}, {"x": 3, "y": True}, {"x": 4, "y": True}],
            index=["a", "b", "c", "d"],
        )

        df["x2"] = 0

        model_structure_iterator = cross_validation.ModelStructureIterator.from_multiple_params(
            typedefs.EstimatorType.REGRESSOR,
            LinearRegression,
            {"no_intercept": {"fit_intercept": True}, "with_intercept": {"fit_intercept": False}},
        )
        feature_iterator = cross_validation.FeatureIterator("y", {"f0": ["x"], "f1": ["x", "x2"]})
        model_specifier = cross_validation.generate_model_specifier(
            feature_iterator, model_structure_iterator
        )

        sample_splitter = cross_validation.SimpleSplit(train_fraction=0.7, seed=1)

        cls.cv_viewer = CrossValidator(model_specifier, sample_splitter).cross_validate(df, seed=1)

    def test_models_df(self):
        self.maxDiff = None
        models_df = self.cv_viewer.models_df
        self.assertEqual(
            models_df.reset_index().to_dict(orient="records"),
            [
                {
                    "feature_slice_id": "f0",
                    "fit_intercept": True,
                    "model_name": "LinearRegression",
                    "sample_slice_id": "1",
                    "structure_id": "no_intercept",
                },
                {
                    "feature_slice_id": "f0",
                    "fit_intercept": False,
                    "model_name": "LinearRegression",
                    "sample_slice_id": "1",
                    "structure_id": "with_intercept",
                },
                {
                    "feature_slice_id": "f1",
                    "fit_intercept": True,
                    "model_name": "LinearRegression",
                    "sample_slice_id": "1",
                    "structure_id": "no_intercept",
                },
                {
                    "feature_slice_id": "f1",
                    "fit_intercept": False,
                    "model_name": "LinearRegression",
                    "sample_slice_id": "1",
                    "structure_id": "with_intercept",
                },
            ],
        )

    def test_coefs_df(self):
        self.maxDiff = None
        coefs_df = self.cv_viewer.coefs_df

        self.assertEqual(
            list(coefs_df.index),
            [
                ("1", "f0", "no_intercept"),
                ("1", "f0", "with_intercept"),
                ("1", "f1", "no_intercept"),
                ("1", "f1", "with_intercept"),
            ],
        )

        coefs = coefs_df.iloc[0].to_dict()

        self.assertSetEqual(set(coefs.keys()), {"intercept", "x", "x2"})
        self.assertAlmostEqual(coefs["intercept"], 1.0)
        self.assertAlmostEqual(coefs["x"], 0.0)
        self.assertTrue(np.isnan(coefs["x2"]))

    def test_metrics_df(self):
        self.maxDiff = None
        metrics_df = self.cv_viewer.metrics_df

        self.assertEqual(
            list(metrics_df.index),
            [
                ("in_sample", "1", "f0", "no_intercept"),
                ("out_of_sample", "1", "f0", "no_intercept"),
                ("in_sample", "1", "f0", "with_intercept"),
                ("out_of_sample", "1", "f0", "with_intercept"),
                ("in_sample", "1", "f1", "no_intercept"),
                ("out_of_sample", "1", "f1", "no_intercept"),
                ("in_sample", "1", "f1", "with_intercept"),
                ("out_of_sample", "1", "f1", "with_intercept"),
            ],
        )

    def test_info_df(self):
        self.maxDiff = None
        info_df = self.cv_viewer.info_df

        self.assertEqual(
            list(info_df.index),
            [
                ("1", "f0", "no_intercept"),
                ("1", "f0", "with_intercept"),
                ("1", "f1", "no_intercept"),
                ("1", "f1", "with_intercept"),
            ],
        )

    def test_dunder_score_methods(self):
        self.assertIsInstance(repr(self.cv_viewer), str)
        self.assertIsInstance(str(self.cv_viewer), str)

    def test_resort_by_metric(self):
        self.maxDiff = None
        # Attempt invocations
        self.cv_viewer.resort_by_metric("r2", kind="in_sample")
        self.cv_viewer.resort_by_metric("r2", kind="out_of_sample")
        self.cv_viewer.resort_by_metric("mse", kind="in_sample")

        # Hard-code one response
        cv_viewer2 = self.cv_viewer.resort_by_metric("r2", kind="out_of_sample")
        self.assertIsInstance(cv_viewer2, CVViewer)

        folds = cv_viewer2.folds
        self.assertIsInstance(folds, list)
        self.assertEqual(len(folds), 4)
        self.assertIsInstance(folds[0], Fold)

        self.assertEqual(
            dict(folds[0].fold_id._asdict()),
            {"feature_slice_id": "f0", "sample_slice_id": "1", "structure_id": "no_intercept"},
        )

    def test_view_best_fold(self):
        self.maxDiff = None
        best_fold = self.cv_viewer.view_best_fold("r2")
        self.assertIsInstance(best_fold, AbstractFoldViewer)
