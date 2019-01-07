"""Viewers for cross validation results."""
import abc
from typing import List, Union

import pandas as pd

from .model_wrappers import AbstractModelWrapper, ClassifierWrapper, Fold, RegressionModelWrapper
from .typedefs import FoldId


def _make_fold_view(fold: Fold, df: pd.DataFrame) -> Union["ClassificationView", "RegressionView"]:
    """Make a fold viewer based on the model structure."""
    if isinstance(fold.model, ClassifierWrapper):
        return ClassificationView(fold, df)
    elif isinstance(fold.model, RegressionModelWrapper):
        return RegressionView(fold, df)
    else:
        raise TypeError(u'Unsupported type: "{}"'.format(type(fold.model)))


class CVViewer(object):
    def __init__(self, folds: List[Fold], df: pd.DataFrame) -> None:
        """A viewer for cross-validation results.

        Args:
            folds: list of Fold, containing model realizations
            df: pd.DataFrame instance, the original data used for generating the models
        """
        self._folds = folds
        self._folds_mapping = {fold.fold_id: fold for fold in folds}
        self._df = df

    @property
    def folds(self) -> List[Fold]:
        """Get list of folds underlying the viewer."""
        return list(self._folds)

    def resort_by_metric(self, metric, kind="out_of_sample") -> "CVViewer":
        """Generate a new viewer with the folds resorted by the given metric."""
        index = self.metrics_df[metric][kind].sort_values().index
        folds = [self._folds_mapping[idx] for idx in index]
        return CVViewer(folds, self._df)

    def view_best_fold(self, metric, kind="out_of_sample") -> "AbstractFoldViewer":
        """Get the best fold as judged by the given metric."""
        idx_max = self.metrics_df[metric][kind].idxmax()
        fold = self._folds_mapping[idx_max]
        return _make_fold_view(fold, self._df)

    def get_fold(self, fold_id) -> "AbstractFoldViewer":
        """Get fold by fold id."""
        return _make_fold_view(self._folds_mapping[fold_id], self._df)

    @property
    def models_df(self) -> pd.DataFrame:
        """Get a dataframe summarizing model structure information."""
        models = {
            fold.fold_id: _make_fold_view(fold, self._df).model_structure for fold in self._folds
        }
        models_df = pd.DataFrame.from_dict(models, orient="index")
        models_df.index.names = FoldId._fields
        return models_df

    @property
    def coefs_df(self) -> pd.DataFrame:
        """Summary table showing coefficients.

        WARNING: Include warning about the fact there's an entire posterior distribution.
        The MLE may be insufficient in many cases. (At least marginalized 95% CI is useful.)
        """
        coefs = {fold.fold_id: fold.model_coefficients for fold in self._folds}
        coefs_df = pd.DataFrame.from_dict(coefs, orient="index")
        coefs_df.index.names = FoldId._fields
        return coefs_df

    @property
    def metrics_df(self) -> pd.DataFrame:
        """Get metrics as a dataframe."""
        df = pd.concat((_make_fold_view(fold, self._df).metrics for fold in self._folds), axis=0)
        return df

    @property
    def info_df(self) -> pd.DataFrame:
        """Get info across all folds as a dataframe."""
        infos = [_make_fold_view(fold, self._df).info for fold in self._folds]
        fields = list(FoldId._fields)
        return pd.DataFrame(infos).set_index(fields)


class AbstractFoldViewer(abc.ABC):
    def __init__(self, fold: Fold, data: pd.DataFrame):
        """A viewer for a fold instance.

        Args:
            fold: instance of Fold
            data: pd.DataFrame
                Must be the same dataframe on which the fold was defined.
        """
        self._fold = fold
        self._data = data

    @property
    def info(self) -> dict:
        """Collect basic information about the fold."""
        info = dict(self._fold.fold_id._asdict())
        info.update(
            {
                "num_train": len(self._fold.fold_spec.sample_slice.train_index),
                "num_test": len(self._fold.fold_spec.sample_slice.test_index),
            }
        )
        return info

    @property
    def coefs(self) -> dict:
        """Get model coefficients."""
        return self._fold.model_coefficients

    @property
    def features(self) -> List[str]:
        """Get the names of the features that were used in this fold."""
        return self._fold.fold_spec.model_slice.feature_slice.features

    @property
    def target(self):
        """Get the name of the target that was used in this fold."""
        return self._fold.fold_spec.model_slice.feature_slice.target

    @property
    def model_structure(self) -> dict:
        """Get information about the model structure."""
        model_slice = self._fold.fold_spec.model_slice
        name = str(model_slice.model_structure.model_cls.__name__)
        params = model_slice.model_structure.model_params
        model_structure_info = dict(params)
        model_structure_info.update({"model_name": name})
        return model_structure_info

    @property
    def metrics(self):
        """Get metrics for this fold."""
        fold_id = self._fold.fold_id

        metrics = {
            (("in_sample",) + fold_id): self._fold.in_sample_metrics,
            (("out_of_sample",) + fold_id): self._fold.out_of_sample_metrics,
        }

        df = pd.DataFrame.from_dict(metrics, orient="index")
        df.index.names = ["kind"] + list(FoldId._fields)
        return df

    def _get_within_slice_data(self, kind, include_all_features=False):
        """Get data within the slice."""
        if include_all_features:
            raise NotImplementedError()
        if kind == "in_sample":
            idx = self._fold.fold_spec.sample_slice.train_index
        elif kind == "out_of_sample":
            idx = self._fold.fold_spec.sample_slice.test_index
        else:
            raise ValueError(u'Unsupported value for kind ("{}").'.format(kind))
        df = self._data.loc[idx].copy()
        df.index.name = self._data.index.name  # Why is this necessary? looks like a bug
        return df

    @property
    def in_sample_prediction(self):
        """Get an in-sample prediction."""
        xy_data = self._get_within_slice_data("in_sample")
        return self.predict(xy_data)

    @property
    def in_sample_data(self):
        """Materialize the in-sample dataset associated with this fold."""
        return self._get_within_slice_data("in_sample")

    @property
    def out_of_sample_prediction(self):
        """Get an out-of-sample prediction."""
        xy_data = self._get_within_slice_data("out_of_sample")
        return self.predict(xy_data)

    @property
    def out_of_sample_data(self):
        """Materialize the out-of-sample dataset associated with this fold."""
        return self._get_within_slice_data("out_of_sample")

    @property
    def full_prediction(self):
        in_yhat = self.in_sample_prediction.assign(kind="in")
        out_yhat = self.out_of_sample_prediction.assign(kind="out")
        return pd.concat([in_yhat, out_yhat])

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Use the model of this fold to make a prediction for the given dataframe."""
        return self.model.predict(df)

    @property
    def model(self) -> AbstractModelWrapper:
        """Get the model associated with this fold."""
        return self._fold.model

    def _repr_html_(self) -> str:
        """Make an HTML report containing information about the fold."""
        html_template = """
            <div><h3>Fold (HTML view)</h3></div>
            <div>
               <div>{fold_info}</div>
               <div><h4>Coefficients<h4></div>
               <div>{coefs_table}</div>
               <div><h4>Metrics<h4></div> 
               <div>{metrics_table}</div>
            </div>
        """
        info_table = pd.Series(self.info).to_frame("spec").to_html()
        coefs = pd.Series(self.model.coefs).sort_index().to_frame("coefs")
        html = html_template.format(
            fold_info=info_table, coefs_table=coefs.to_html(), metrics_table=self.metrics.to_html()
        )
        return html

    def __repr__(self) -> str:
        """Overload repr"""
        return "<Fold {}>".format(self._fold.fold_id)


class ClassificationView(AbstractFoldViewer):
    def __init__(self, fold: Fold, df: pd.DataFrame):
        """Initializer a viewer that is specialized for a classifier."""
        super(ClassificationView, self).__init__(fold, df)
        if not isinstance(fold.model, ClassifierWrapper):
            raise TypeError(
                u'The model was supposed to be a classifier found: "{}"'.format(type(fold.model))
            )

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use the model of this fold to make a prediction for the given dataframe."""
        # TODO(eyurtsev): Fix this behavior.
        model: ClassifierWrapper = self.model
        return model.predict_proba(df)


class RegressionView(AbstractFoldViewer):
    def __init__(self, fold: Fold, df: pd.DataFrame):
        """Initializer a viewer that is specialized for a classifier."""
        super(RegressionView, self).__init__(fold, df)
        if not isinstance(fold.model, RegressionModelWrapper):
            raise TypeError(
                u'The model was supposed to be a classifier found: "{}"'.format(type(fold.model))
            )
