"""
Cross-validator building logic.
Author: JiaWei Jiang

This file contains the basic logic of building cross-validator.
"""
from typing import Any

from sklearn.model_selection import GroupKFold, KFold
from sklearn.model_selection._split import BaseCrossValidator


def build_cv(**cv_cfg: Any) -> BaseCrossValidator:
    """Build and return cross-validator.

    Args:
        cv_cfg: hyperparameters of cross-validator

    Returns:
        cv: cross-validator
    """
    cv_scheme = cv_cfg["scheme"]
    n_folds = cv_cfg["n_folds"]
    # random_state = cv_cfg["random_state"]

    if cv_scheme == "kf":
        cv = KFold(n_folds, shuffle=True, random_state=cv_cfg["random_state"])
    elif cv_scheme == "gpkf":
        cv = GroupKFold(n_folds)

    return cv
