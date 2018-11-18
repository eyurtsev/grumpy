from __future__ import division

import funcy
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


def _unbiased_std(x):
    """Calculate unbiased standard deviation"""
    return np.std(x, ddof=1)


def _get_num_samples(x):
    """Return the number of samples."""
    return np.shape(x)[0]


DEFAULT_STATS_MAPPING = {
    # functions accept a 1d metric and produce a summary statistic
    "mean": np.mean,
    "std": _unbiased_std,
}

DEFAULT_METRIC_MAPPING = {
    # functions map (y, y_hat) -> metric
    "r2": r2_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
}

ADDITIONAL_CLASSIFICATION_METRICS = {
    # Please note that precision, recall, and fscore are added automatically right now.
    # Which is why this contains **additional** metrics
    # Keys are metric names, values map (y, y_hat) -> metric
    "matthews_corrcoef": matthews_corrcoef,
    "accuracy": accuracy_score,
}


def _make_generic_report(y, y_hat, stats_mapping, metric_mapping):
    """Generate metrics for continuous estimation.

    Args:
        y: numpy array | pd.Series
            Target (continuous variable)
        y_hat: numpy array | pd.Series
            Prediction (continuous variable)
        stats_mapping: dict
            keys are used as base names of stats in the report
            values are functions that map 1D vectors to summary statistics.
            The number of samples is always added by default.
        metric_mapping: dict
            keys are used as base names of metrics in the report
            values are functions that map accept (y, y_hat) and produce a metric.

    Returns:
        dict (keys are name of corresponding metrics and statistics.)
    """
    # Use is None for checks. Empty dict should not result in default statistics.
    if "num_samples" in stats_mapping.keys():
        raise ValueError(
            u'The "samples" key is reserved and is always added by default. Received '
            u'the following stats_mapping: "{}"'.format(stats_mapping.keys())
        )

    if len(y) == 0:
        return {"num_samples": 0}

    # Calculate stats
    target_stats = {("target_" + name): f(y) for name, f in stats_mapping.items()}
    target_stats["num_samples"] = _get_num_samples(y)

    hat_stats = {("hat_" + name): f(y_hat) for name, f in stats_mapping.items()}

    # Calculate metrics
    metrics = {name: f(y, y_hat) for name, f in metric_mapping.items()}

    # Merge stats and metrics into a single report.
    report = funcy.merge(metrics, target_stats, hat_stats)
    return report


############
# PUBLIC API
############


def make_regression_summary_stats(y, yhat):
    """Make summary statistics for a regression task."""
    return _make_generic_report(y, yhat, DEFAULT_STATS_MAPPING, DEFAULT_METRIC_MAPPING)


def make_classification_report(y, yhat, yhat_probabilities=None):
    """Make a report with optional support for including classification results.

    Args:
        y: ndarray | pd.Series
            Categorical target
        yhat: ndarray | pd.Series
            Model prediction. This is a hard-prediction.
        yhat_probabilities: None | ndarray | pd.Series
            Model probability estimates.

    Returns:
        dict (keys = names of metrics)
    """
    # Make report for (continuous) estimation results
    estimation_report = _make_generic_report(y, yhat, DEFAULT_STATS_MAPPING, {})

    # Make report for classification results
    if y.any() and yhat.any():
        # If discretized predictions are available
        precision, recall, fscore, _ = precision_recall_fscore_support(y, yhat, average="micro")
    else:
        precision, recall, fscore = 0.0, 0.0, 0.0

    classification_report = {
        name: f(y, yhat) for name, f in ADDITIONAL_CLASSIFICATION_METRICS.items()
    }

    if yhat_probabilities is not None:
        classification_report.update({"log_loss": log_loss(y, yhat_probabilities)})

    classification_report.update({"precision": precision, "recall": recall, "fscore": fscore})

    # Add prefix to keys
    return funcy.merge(estimation_report, classification_report)
