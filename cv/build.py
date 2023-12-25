"""
Cross-validator building logic.
Author: JiaWei Jiang

This file contains the basic logic of building cross-validator.
"""
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
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
    elif cv_scheme == "tscv":
        cv = TSCV(n_folds, cv_cfg["val_size"], cv_cfg["n_gap_months"], cv_cfg["max_train_size"])

    return cv


class TSCV:
    """Time series cross-validator

    Args:
        n_splits: number of splits
        val_size: number of validation months per fold
        n_gap_months: number of gap months between train and val sets
        max_train_size: maximal number of training months per fold
    """

    def __init__(
        self, n_splits: int = 5, val_size: int = 3, n_gap_months: int = 0, max_train_size: Optional[int] = None
    ) -> None:
        self.n_splits = n_splits
        self.val_size = val_size
        self.n_gap_months = n_gap_months
        self.max_train_size = max_train_size

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        years = groups.dt.year
        months = groups.dt.month
        dates = years * 12 + months
        uniq_dates = sorted(dates.unique().tolist())

        val_months_end = [uniq_dates[-(i * self.val_size + 1)] for i in range(self.n_splits)]
        for val_month_end in val_months_end:
            val_month_start = val_month_end - self.val_size
            val_mask = (dates > val_month_start) & (dates <= val_month_end)
            val_idx = X[val_mask].index.to_numpy()

            tr_month_end = val_month_start - self.n_gap_months
            if self.max_train_size is not None:
                tr_month_start = tr_month_end - self.max_train_size
                tr_mask = (dates > tr_month_start) & (dates <= tr_month_end)
            else:
                tr_mask = dates <= tr_month_end
            tr_idx = X[tr_mask].index.to_numpy()

            yield tr_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits
