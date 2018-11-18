import types
import unittest

import pandas as pd
from sklearn.linear_model import LinearRegression

from grumpy import cross_validation, typedefs
from grumpy.core import CrossValidator, _validate_dataframe
from grumpy.typedefs import EstimatorType
from grumpy.viewers import CVViewer


class TestSampleSplitters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_simple_split(self):
        self.maxDiff = None
        simple_split = cross_validation.SimpleSplit(train_fraction=0.7, randomize=True, seed=1)
        sample_slices = simple_split.get_sample_slices(self.df)
        self.assertIsInstance(sample_slices, types.GeneratorType)
        sample_slices = list(sample_slices)
        self.assertEqual(len(sample_slices), 1)
        sample_slice = sample_slices[0]
        self.assertEqual(list(sample_slice.train_index), [2, 9, 6, 4, 0, 3, 1])
        self.assertEqual(list(sample_slice.test_index), [7, 8, 5])

    def test_simple_model_specifier(self):
        model_slice = typedefs.ModelSlice(
            feature_slice=typedefs.FeatureSlice(target="y", features=["x"], name="1"),
            model_structure=typedefs.ModelStructure(
                structure_id="1",
                estimator_type=EstimatorType.REGRESSOR,
                model_cls=LinearRegression,
                model_params={},
            ),
        )

        simple_model_specifier = cross_validation.SimpleModelSpecifier([model_slice])
        model_slices = simple_model_specifier.get_model_slices()

        self.assertIsInstance(model_slices, types.GeneratorType)
        model_slices = list(model_slices)
        self.assertEqual(len(model_slices), 1)
        self.assertEqual(model_slices[0], model_slice)

    def test_feature_iterator(self):
        self.maxDiff = None
        feature_iterator = cross_validation.FeatureIterator(
            target="y", feature_set_name_to_features={"f1": ["x"], "f2": ["x", "bias"]}
        )
        feature_slices = feature_iterator.get_feature_slices()
        self.assertIsInstance(feature_slices, types.GeneratorType)
        feature_slices = list(feature_slices)
        self.assertEqual(len(feature_slices), 2)
        first_feature_slice = feature_slices[0]
        self.assertIsInstance(first_feature_slice, typedefs.FeatureSlice)
        self.assertEqual(
            dict(first_feature_slice._asdict()), {"features": ["x"], "name": "f1", "target": "y"}
        )

    def test_model_structure_iterator(self):
        self.maxDiff = None
        model_structure = typedefs.ModelStructure(
            structure_id="1",
            estimator_type=EstimatorType.REGRESSOR,
            model_cls=LinearRegression,
            model_params={},
        )
        model_structure_iterator = cross_validation.ModelStructureIterator([model_structure])
        model_structures = model_structure_iterator.get_model_structures()

        self.assertIsInstance(model_structures, types.GeneratorType)
        model_structures = list(model_structures)
        self.assertEqual(len(model_structures), 1)
        self.assertEqual(model_structures[0].structure_id, "1")

        model_structure_iterator = cross_validation.ModelStructureIterator.from_multiple_params(
            estimator_type=EstimatorType.REGRESSOR,
            model_cls=LinearRegression,
            name_to_model_params={"1": {"fit_intercept": False}, "2": {"fit_intercept": True}},
        )
        model_structures = model_structure_iterator.get_model_structures()
        self.assertIsInstance(model_structures, types.GeneratorType)
        model_structures = list(model_structures)

        self.assertEqual(model_structures[0].structure_id, "1")
        self.assertEqual(model_structures[1].structure_id, "2")

    def test_generate_model_specifier(self):
        self.maxDiff = None
        feature_iterator = cross_validation.FeatureIterator(
            target="y", feature_set_name_to_features={"f1": ["x"], "f2": ["x", "bias"]}
        )

        model_structure = typedefs.ModelStructure(
            structure_id="1",
            estimator_type=EstimatorType.REGRESSOR,
            model_cls=LinearRegression,
            model_params={},
        )
        model_structure_iterator = cross_validation.ModelStructureIterator([model_structure])

        model_specifier = cross_validation.generate_model_specifier(
            feature_iterator, model_structure_iterator
        )
        self.assertIsInstance(model_specifier, cross_validation.AbstractModelSpecifier)

        model_slices = model_specifier.get_model_slices()
        self.assertIsInstance(model_slices, types.GeneratorType)
        model_slices = list(model_slices)
        model_slice = model_slices[0]
        self.assertIsInstance(model_slice, typedefs.ModelSlice)


class TestCrossValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xy_df = pd.Series(range(10)).to_frame("x")
        xy_df["y"] = xy_df["x"] + 1
        xy_df["x2"] = 0
        cls.df = xy_df

    def test_cross_validator(self):
        self.maxDiff = None
        model_structure_iterator = cross_validation.ModelStructureIterator.from_multiple_params(
            typedefs.EstimatorType.REGRESSOR,
            LinearRegression,
            {"no_intercept": {"fit_intercept": True}, "with_intercept": {"fit_intercept": False}},
        )
        feature_iterator = cross_validation.FeatureIterator("y", {"f0": ["x"], "f1": ["x", "x2"]})
        model_specifier = cross_validation.generate_model_specifier(
            feature_iterator, model_structure_iterator
        )

        sample_splitter = cross_validation.SimpleSplit(train_fraction=0.7)
        cv_viewer = CrossValidator(model_specifier, sample_splitter).cross_validate(self.df, seed=1)
        self.assertIsInstance(cv_viewer, CVViewer)
        self.assertEqual(
            cv_viewer.info_df.reset_index().to_dict(orient="records"),
            [
                {
                    "feature_slice_id": "f0",
                    "num_test": 3,
                    "num_train": 7,
                    "sample_slice_id": "1",
                    "structure_id": "no_intercept",
                },
                {
                    "feature_slice_id": "f0",
                    "num_test": 3,
                    "num_train": 7,
                    "sample_slice_id": "1",
                    "structure_id": "with_intercept",
                },
                {
                    "feature_slice_id": "f1",
                    "num_test": 3,
                    "num_train": 7,
                    "sample_slice_id": "1",
                    "structure_id": "no_intercept",
                },
                {
                    "feature_slice_id": "f1",
                    "num_test": 3,
                    "num_train": 7,
                    "sample_slice_id": "1",
                    "structure_id": "with_intercept",
                },
            ],
        )

    def test_index_validation(self):
        valid_dfs = [pd.DataFrame([]), pd.DataFrame([1, 2, 3], [1, 2, 3])]

        for valid_df in valid_dfs:
            _validate_dataframe(valid_df)

        invalid_dfs = [
            pd.DataFrame([1, 2], index=["a", "a"]),
            pd.DataFrame([1, 2, 3], index=["a", "b", "a"]),
        ]

        for invalid_df in invalid_dfs:
            with self.assertRaises(ValueError):
                _validate_dataframe(invalid_df)
