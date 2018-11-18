import abc
from typing import Iterator

import pandas as pd

from .cross_validation import AbstractModelSpecifier, AbstractSampleSplitter
from .model_wrappers import build_model_realization
from .typedefs import FoldSpec
from .viewers import CVViewer


class AbstractCrossValidator(abc.ABC):
    @abc.abstractmethod
    def split(self, xy_data: pd.DataFrame) -> Iterator[FoldSpec]:
        """Split a dataframe into appropriate slices."""
        raise NotImplementedError()


def _validate_dataframe(df):
    """Make sure that the dataframe is valid."""
    if df.index.duplicated().any():
        raise ValueError(
            u"Found duplicated entries in the dataframe. " u"Make sure the index is unique"
        )


############
# PUBLIC API
############


class CrossValidator(AbstractCrossValidator):
    def __init__(
        self, model_specifier: AbstractModelSpecifier, sample_splitter: AbstractSampleSplitter
    ) -> None:
        """Simple cross validator."""
        self._model_specifier = model_specifier
        self._sample_splitter = sample_splitter

    def split(self, xy_data: pd.DataFrame) -> Iterator[FoldSpec]:
        """Generate fold specs that can be used to fit the given data."""
        sample_slices = self._sample_splitter.get_sample_slices(xy_data)
        model_slices = self._model_specifier.get_model_slices()

        for sample_slice in sample_slices:
            for model_slice in model_slices:
                yield FoldSpec(sample_slice=sample_slice, model_slice=model_slice)

    def cross_validate(self, df: pd.DataFrame, seed=None) -> CVViewer:
        """Cross validate"""
        # TODO(eyurtsev): Add n_jobs parameter.
        _validate_dataframe(df)

        folds = []

        # The cross-validator can return a generalized iterator which accepts
        # history as an input. This allows one to implement recursive feature selection.
        for fold_spec in self.split(df):
            # Can add a dispatching interface for _build_fold.
            # The trivial speedup is multi-core processing
            # The non-trivial speedup is distributed computation.
            fold = build_model_realization(df, fold_spec, seed=seed)
            folds.append(fold)

        return CVViewer(folds, df)
