import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from grumpy.model_wrappers import ClassifierWrapper, RegressionModelWrapper, build_sklearn_regressor
from grumpy.typedefs import EstimatorType


class TestRegressionModelWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(
            [{"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": 3, "y": 3}, {"x": 4, "y": 4}],
            index=["a", "b", "c", "d"],
        )

    def test_regression_model_wrapper(self):
        linear_regression = LinearRegression()
        model = linear_regression.fit(self.df[["x"]], self.df["y"])
        model_wrapper = RegressionModelWrapper(model, ["x"], "y")

        prediction_ts = model_wrapper.predict(self.df)
        self.assertIsInstance(prediction_ts, pd.Series)
        self.assertEqual(prediction_ts.shape, (4,))
        self.assertEqual(prediction_ts.to_dict(), {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})

        score = model_wrapper.score(self.df)

        self.assertEqual(
            set(score.keys()),
            {"mae", "hat_std", "target_mean", "target_std", "num_samples", "hat_mean", "r2", "mse"},
        )

        self.assertEqual(model_wrapper.estimator_type, EstimatorType.REGRESSOR)


class TestClassificationModelWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(
            [{"x": 1, "y": False}, {"x": 2, "y": False}, {"x": 3, "y": True}, {"x": 4, "y": True}],
            index=["a", "b", "c", "d"],
        )

    def test_classifier_model_wrapper(self):
        self.maxDiff = None
        logistic_regressor = LogisticRegression(solver="lbfgs")
        model = logistic_regressor.fit(self.df[["x"]], self.df["y"])

        model_wrapper = ClassifierWrapper(model, ["x"], "y")

        # Hard decisions
        predictions_df = model_wrapper.predict(self.df)
        self.assertIsInstance(predictions_df, pd.Series)
        self.assertEqual(predictions_df.shape, (4,))
        self.assertEqual(predictions_df.to_dict(), {"a": False, "b": False, "c": True, "d": True})

        # Soft decisions
        probas_df = model_wrapper.predict_proba(self.df)
        self.assertIsInstance(probas_df, pd.DataFrame)
        self.assertEqual(probas_df.shape, (4, 2))

        first_obs_probas = probas_df.to_dict(orient="records")[0]

        # Probability of positive class for the first observation
        self.assertAlmostEqual(first_obs_probas[1], 0.192, places=3)

        # Evaluate scores
        score = model_wrapper.score(self.df)

        self.assertEqual(
            set(score.keys()),
            {
                "log_loss",
                "fscore",
                "hat_mean",
                "target_std",
                "matthews_corrcoef",
                "num_samples",
                "accuracy",
                "target_mean",
                "precision",
                "hat_std",
                "recall",
            },
        )


class TestModelWrappers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([0, 1, 2, 3, 4, 5])
        y = np.array(x)
        cls.df = pd.DataFrame([x, y], index=["x", "y"]).T

    def test_build_sklearn_style_regressor(self):
        """Attempt to build model for sklearn linear regression and stats models."""
        test_cases = (
            (LinearRegression, {"fit_intercept": False}, 1.0),
            (
                RandomForestRegressor,
                {"max_depth": 3, "random_state": 1, "n_estimators": 10},
                0.9617,
            ),
        )

        for model_cls, model_params, expected_r2 in test_cases:
            model = build_sklearn_regressor(model_cls, self.df, "y", model_params=model_params)
            score = model.score(self.df)
            self.assertIsInstance(model.coefs, dict)
            self.assertAlmostEqual(score["r2"], expected_r2, 3)
