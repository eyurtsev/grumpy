from typing import Any, List

import numpy as np


def _unpack_nesting(attr_value: np.array) -> Any:
    """Unpack a nested structure."""
    if len(np.shape(attr_value)) == 2 and np.shape(attr_value)[0] == 1:
        # Some sklearn models have strange nesting structure.
        # TODO: Verify
        attr_value = attr_value[0]
    elif len(np.shape(attr_value)) == 0:
        return [attr_value]
    else:
        pass  # This is OK
    return attr_value


def _get_coefs(model, features, attr_name):
    """Get coefficient"""
    attr_value = getattr(model, attr_name)
    return dict(zip(features, _unpack_nesting(attr_value)))


#
# PUBLIC API
#


def get_sklearn_model_coefficients(model: Any, features: List[str]):
    """Extract dict of coefficients from sklearn model"""
    coefs_dict = {}

    if hasattr(model, "coef_"):
        coefs_dict.update(_get_coefs(model, features, "coef_"))

    if hasattr(model, "feature_importances_"):
        coefs_dict.update(_get_coefs(model, features, "feature_importances_"))

    if hasattr(model, "intercept_"):
        coefs_dict.update({"intercept": np.squeeze(model.intercept_)})

    return coefs_dict
