"""Model wrappers that can be used to adapt code from other 3rd party packages."""
import abc
from typing import Any, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn import metrics

from .metrics import make_classification_report, make_regression_summary_stats
from .model_inspection import get_sklearn_model_coefficients
from .typedefs import EstimatorType, FoldId, FoldSpec


def _get_features(xy_data: pd.DataFrame, target: str, features=Optional[List[str]]) -> List[str]:
    """Get a list of the features to use.

    Args:
        xy_data: pd.DataFrame instance
        target: str
        features: Optional[List[str]], will use these features if provided

    Returns:
        list of features to use
    """
    known_columns = xy_data.columns.tolist()

    if target not in known_columns:
        raise ValueError('"{}" does not exist in data set.'.format(target))

    known_columns.remove(target)

    if features:
        missing_cols = set(features) - set(known_columns)
        if missing_cols:
            raise ValueError(u'Data is missing the following features: "{}"'.format(missing_cols))
        features_to_use = list(features)
    else:
        features_to_use = features or known_columns
    return features_to_use


def _validate_dataframe(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """Validate that the dataframe contains the expected columns."""
    existing_columns = set(df.columns)
    missing_columns = set(expected_columns) - existing_columns
    if missing_columns:
        raise ValueError(u'dataframe was missing required columns: "{}"'.format(missing_columns))


class AbstractModelWrapper(abc.ABC):
    def __init__(
        self, model: Any, features: List[str], target: str, coefs: Optional[dict] = None
    ) -> None:
        """A model wrapper interface."""
        self._model = model
        self._features = features
        self._target = target
        self._coefs = coefs or {}

    @abc.abstractmethod
    def estimator_type(cls) -> EstimatorType:
        """Returns information about what the estimator is."""
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame):
        """A predict method that takes in data and returns prediction results."""
        raise NotImplementedError()

    @abc.abstractmethod
    def score(self, df) -> dict:
        """Calculate summary statistics for the given dataframe."""
        raise NotImplementedError()

    @property
    def coefs(self) -> dict:
        """Get model coefficients if known."""
        return self._coefs


class RegressionModelWrapper(AbstractModelWrapper):
    def __init__(
        self, model: Any, features: List[str], target: str, coefs: Optional[dict] = None
    ) -> None:
        """A model wrapper interface."""
        super(RegressionModelWrapper, self).__init__(model, features, target, coefs=coefs)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict for the given dataframe.

        Args:
            df: pd.DataFrame instance

        Returns:
            pd.Series instance
        """
        _validate_dataframe(df, self._features)
        data = df[self._features]
        predictions_ts = pd.Series(self._model.predict(data), index=df.index)
        return predictions_ts

    def score(self, df: pd.DataFrame) -> dict:
        """Evaluate model performance on the given data."""
        _validate_dataframe(df, self._features + [self._target])
        y_hat = self.predict(df).values
        y_data = df[self._target].values
        return make_regression_summary_stats(y_data, y_hat)

    @property
    def estimator_type(self) -> EstimatorType:
        """Get the estimator type."""
        return EstimatorType.REGRESSOR


class ClassifierWrapper(AbstractModelWrapper):
    def __init__(
        self, model: Any, features: List[str], target: str, coefs: Optional[dict] = None
    ) -> None:
        """A model wrapper interface."""
        super(ClassifierWrapper, self).__init__(model, features, target, coefs=coefs)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict the class identity for the given class (hard decision).

        Args:
            df: pd.DataFrame instance

        Returns:
            pd.Series instance, predicts the class identity (hard decision)
        """
        _validate_dataframe(df, self._features)
        data = df[self._features]
        predictions_ts = pd.Series(self._model.predict(data), index=df.index)
        return predictions_ts

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities for all classes (soft decisions).

        Args:
            df: pd.DataFrame

        Returns:
            pd.DataFrame instance
        """
        _validate_dataframe(df, self._features)
        data = df[self._features]
        probas_df = pd.DataFrame(self._model.predict_proba(data), index=df.index)
        probas_df.columns = self.classes
        return probas_df

    def score(self, df: pd.DataFrame) -> dict:
        """Evaluate model performance on the given data."""
        _validate_dataframe(df, self._features + [self._target])
        y_hat = self.predict(df).values
        y_data = df[self._target].values
        probas_df = self.predict_proba(df)
        return make_classification_report(y_data, y_hat, probas_df)

    def get_precision_recall_curve(self, xy_data, target_class):
        """Generate a precision recall curve for the given data."""
        y_label = xy_data[self._target]
        y_pred = self.predict_proba(xy_data)[target_class]
        return metrics.precision_recall_curve(y_label, y_pred)

    @property
    def estimator_type(cls) -> EstimatorType:
        """Get the estimator type."""
        return EstimatorType.CLASSIFIER

    @property
    def classes(self) -> List[str]:
        """Get the classes"""
        return list(self._model.classes_)


def _build_sklearn_style_model(
    estimator_type: EstimatorType,
    model_cls: Any,
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    model_params=None,
) -> Union[ClassifierWrapper, RegressionModelWrapper]:
    """Build sklearn style model"""
    features = _get_features(df, target, features=features)
    model_params = model_params or {}
    model_instance = model_cls(**model_params)
    model_instance.fit(df[features], df[target])
    coefs = get_sklearn_model_coefficients(model_instance, features)
    if estimator_type == EstimatorType.REGRESSOR:
        model_wrapper = RegressionModelWrapper(model_instance, features, target, coefs=coefs)
    elif estimator_type == EstimatorType.CLASSIFIER:
        model_wrapper = ClassifierWrapper(model_instance, features, target, coefs=coefs)
    else:
        raise ValueError(u'Unsupported estimator type: "{}"'.format(estimator_type))
    return model_wrapper


############
# PUBLIC API
############


class Fold(NamedTuple):
    """A fold is a placeholder to store a model realization."""

    fold_id: FoldId
    fold_spec: FoldSpec
    model: AbstractModelWrapper
    model_coefficients: dict
    in_sample_metrics: dict
    out_of_sample_metrics: dict


def build_sklearn_regressor(
    model_cls: Any,
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    model_params=None,
) -> RegressionModelWrapper:
    """Build an sklearn model."""
    return _build_sklearn_style_model(
        EstimatorType.REGRESSOR, model_cls, df, target, features, model_params=model_params
    )


def build_sklearn_classifier(
    model_cls: Any,
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    model_params=None,
) -> ClassifierWrapper:
    """Build an sklearn style classifier."""
    return _build_sklearn_style_model(
        EstimatorType.CLASSIFIER, model_cls, df, target, features, model_params=model_params
    )


def build_model_realization(df: pd.DataFrame, fold_spec: FoldSpec, seed=None) -> Fold:
    """Build a model realization for the given data and the given fold spec."""
    train_data = df.loc[fold_spec.sample_slice.train_index]
    valid_data = df.loc[fold_spec.sample_slice.test_index]
    feature_cols = fold_spec.model_slice.feature_slice.features
    target_col = fold_spec.model_slice.feature_slice.target
    model_slice = fold_spec.model_slice

    if seed:
        np.random.seed(seed)

    model = _build_sklearn_style_model(
        fold_spec.model_slice.model_structure.estimator_type,
        model_slice.model_structure.model_cls,
        train_data,
        target_col,
        features=feature_cols,
        model_params=model_slice.model_structure.model_params,
    )

    in_sample_metrics = model.score(train_data)
    out_of_sample_metrics = model.score(valid_data)

    return Fold(
        fold_id=FoldId(
            sample_slice_id=fold_spec.sample_slice.slice_id,
            feature_slice_id=fold_spec.model_slice.feature_slice.name,
            structure_id=fold_spec.model_slice.model_structure.structure_id,
        ),
        fold_spec=fold_spec,
        model=model,
        model_coefficients=model.coefs,
        in_sample_metrics=in_sample_metrics,
        out_of_sample_metrics=out_of_sample_metrics,
    )
