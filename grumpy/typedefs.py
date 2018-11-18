import enum
from typing import Any, List, NamedTuple

import pandas as pd


class EstimatorType(enum.Enum):
    REGRESSOR = "regressor"
    CLASSIFIER = "classifier"


class SampleSlice(NamedTuple):
    """A slice of the data in sample space."""

    slice_id: str  # An identifier for the given sample slice
    train_index: pd.Index  # train indexes
    test_index: pd.Index  # test indexes


class FoldId(NamedTuple):
    sample_slice_id: str
    feature_slice_id: str
    structure_id: str


class FeatureSlice(NamedTuple):
    """A specification of which feature and target will be used"""

    name: str
    target: str
    features: List[str]


class ModelStructure(NamedTuple):
    structure_id: str
    estimator_type: EstimatorType
    model_cls: Any
    model_params: dict


class ModelSlice(NamedTuple):
    """A slice that can be used to define a model."""

    feature_slice: FeatureSlice
    model_structure: ModelStructure


class FoldSpec(NamedTuple):
    """A slice of both the sample and the model that yields a model realization."""

    sample_slice: SampleSlice
    model_slice: ModelSlice
