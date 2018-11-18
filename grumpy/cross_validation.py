"""Provide iteration objects that can materialize slices of dataframes"""
import abc
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from grumpy.typedefs import EstimatorType, FeatureSlice, ModelSlice, ModelStructure, SampleSlice


class AbstractSampleSplitter(abc.ABC):
    def get_sample_slices(self, xy_data: pd.DataFrame) -> Iterator[SampleSlice]:
        raise NotImplementedError()


class AbstractModelSpecifier(abc.ABC):
    """Interface for specifying a model configuration."""

    @abc.abstractmethod
    def get_model_slices(self) -> Iterator[ModelSlice]:
        """Get an iterator over possible model slices"""
        raise NotImplementedError()


#
# PUBLIC API
#


class SimpleSplit(AbstractSampleSplitter):
    def __init__(
        self, train_fraction: float = 0.7, randomize: bool = True, seed: Optional[int] = None
    ):
        """Generates a single split using a simple parameterization.

        Args:
            train_fraction: float, number between 0 and 1
            randomize: bool, if True will make a randomized version of the data
            seed: Optional[int], if provided will set a randomization seed.
        """
        super(AbstractSampleSplitter, self).__init__()
        if train_fraction > 1 or train_fraction < 0:
            raise ValueError(u"Expected a number between [0, 1] got: {}".format(train_fraction))
        self._train_fraction = train_fraction
        self._randomize = randomize
        self._seed = seed

    def get_sample_slices(self, xy_data: pd.DataFrame) -> Iterator[SampleSlice]:
        """Get the different sample slices."""
        index = list(xy_data.index)

        if self._randomize:
            if self._seed:
                np.random.seed(self._seed)
            final_index = pd.Index(np.random.permutation(index))
        else:
            final_index = pd.Index(index)

        num_samples = len(final_index)

        train_idx_end = int(num_samples * self._train_fraction)

        yield SampleSlice(
            slice_id="1",
            train_index=final_index[:train_idx_end],
            test_index=final_index[train_idx_end:],
        )


class SimpleModelSpecifier(AbstractModelSpecifier):
    def __init__(self, model_slices: List[ModelSlice]) -> None:
        """Interface for specifying a model configuration."""
        super(AbstractModelSpecifier, self).__init__()
        self._model_slices = model_slices

    def get_model_slices(self) -> Iterator[ModelSlice]:
        """Get model slices."""
        for model_slice in self._model_slices:
            yield model_slice


class FeatureIterator(object):
    def __init__(self, target: str, feature_set_name_to_features: Dict[str, List[str]]) -> None:
        """An iterator over feature space."""
        self._target = target
        self._feature_set_name_to_features = feature_set_name_to_features

    def get_feature_slices(self) -> [FeatureSlice]:
        """Get a list of feature slices."""
        for feature_set_name, features in self._feature_set_name_to_features.items():
            yield FeatureSlice(name=feature_set_name, features=features, target=self._target)


class ModelStructureIterator(object):
    def __init__(self, model_structures: List[ModelStructure]):
        """An interface to generate an iterator over model structures."""
        self._model_structures = model_structures

    def get_model_structures(self) -> Iterator[ModelStructure]:
        """Get an iterator over model structures."""
        for model_structure in self._model_structures:
            yield model_structure

    @classmethod
    def from_multiple_params(
        cls, estimator_type: EstimatorType, model_cls: Any, name_to_model_params: Dict[str, dict]
    ):
        """Make a model structure iterator from multiple parameter configurations."""
        model_structures = [
            ModelStructure(
                structure_id=name,
                estimator_type=estimator_type,
                model_cls=model_cls,
                model_params=model_params,
            )
            for name, model_params in name_to_model_params.items()
        ]
        return cls(model_structures)


def generate_model_specifier(
    feature_iterator: FeatureIterator, model_structure_iterator: ModelStructureIterator
) -> SimpleModelSpecifier:
    """Generate a model specifier"""
    model_slices = []

    for feature_slice in feature_iterator.get_feature_slices():
        for model_structure in model_structure_iterator.get_model_structures():
            model_slices.append(
                ModelSlice(feature_slice=feature_slice, model_structure=model_structure)
            )

    return SimpleModelSpecifier(model_slices=model_slices)
