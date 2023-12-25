"""
Data processor.
Author: JiaWei Jiang

This file contains the definition of data processor generating datasets
ready for modeling phase.
"""
import logging
import pickle
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

from metadata import UNIT_ID_COL


class DataProcessor(object):
    """Data processor generating datasets ready for modeling phase.

    Args:
        dp_cfg: hyperparameters of data processor

    Attributes:
        _data_cv: training set
            *Note: Fold columns are expected if pre-splitting is done,
                providing mapping from sample index to fold number.
        _data_test: test set
    """

    # https://stackoverflow.com/questions/59173744
    _data_cv: Union[pd.DataFrame, np.ndarray]
    _data_test: Union[pd.DataFrame, np.ndarray]

    def __init__(self, data_path: Path, **dp_cfg: Any) -> None:
        # Setup data processor
        self.data_path = data_path
        self.dp_cfg = dp_cfg
        self._setup()

        # Load raw data
        self._load_data()

    def _setup(self) -> None:
        """Retrieve hyperparameters for data processing."""
        # Before data splitting
        self.feats_ver = self.dp_cfg["feats_ver"]
        self.tgt_col = self.dp_cfg["tgt_col"]
        self.tgt_aux_cols = self.dp_cfg["tgt_aux_cols"]
        self.drop_rows_with_null_tgt = self.dp_cfg["drop_rows_with_null_tgt"]
        self._load_feats()
        self.tgt_cols = [self.tgt_col] + self.tgt_aux_cols

        # After data splitting...
        self.scaling = None

    def _load_feats(self) -> None:
        """Load features."""
        with open(self.data_path / "feats" / f"v{self.feats_ver}.pkl", "rb") as f:
            feats = pickle.load(f)
        self._num_feats, self._cat_feats = feats["num"], feats["cat"]
        self._feats = self._num_feats + self._cat_feats

    def _load_data(self) -> None:
        """Load raw data."""
        cols_to_load = (
            # Features
            self.feats
            # Target
            + [self.tgt_col]
            # CV columns
            + [UNIT_ID_COL, "datetime"]
            # Separate modeling
            + ["is_consumption"]
        )
        for col in self.tgt_aux_cols:
            if col not in cols_to_load:
                cols_to_load.append(col)
        self._data_cv = pl.read_parquet(self.data_path / "data_eager.parquet", columns=cols_to_load)
        self._data_test = None

    @property
    def feats(self) -> List[str]:
        """Return all features."""
        return self._feats

    @property
    def num_feats(self) -> List[str]:
        """Returnn numeric features."""
        return self._num_feats

    @property
    def cat_feats(self) -> List[str]:
        """Return categorical features."""
        return self._cat_feats

    def run_before_splitting(self) -> None:
        """Clean and process data before data splitting (i.e., on raw
        static DataFrame).
        """
        logging.info("Run data cleaning and processing before data splitting...")

        # Put processing logic below...
        if self.drop_rows_with_null_tgt:
            logging.info(f"\t>> Drop rows with null target {self.tgt_col}...")
            self._data_cv = self._data_cv.filter(pl.col(self.tgt_col).is_not_null())

        logging.info("\t>> Convert pl.DataFrame to pd.DataFrame...")
        self._data_cv = self._data_cv.to_pandas()

        logging.info(f"\t>> Specify categorical features {self.cat_feats}")
        if len(self.cat_feats) != 0:
            self._data_cv[self.cat_feats] = self._data_cv[self.cat_feats].astype("category")

        logging.info("Done.")

    def run_after_splitting(
        self,
        data_tr: Union[pd.DataFrame, np.ndarray],
        data_val: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Any]:
        """Clean and process data after data splitting to avoid data
        leakage issue.

        Note that data processing is prone to data leakage, such as
        fitting the scaler with the whole dataset.

        Args:
            data_tr: training data
            data_val: validation data

        Returns:
            data_tr: processed training data
            data_val: processed validation data
            scaler: scaling object
        """
        logging.info("Run data cleaning and processing after data splitting...")

        # Scale data
        scaler = None
        if self.scaling is not None:
            data_tr, data_val, scaler = self._scale(data_tr, data_val)

        # Put more processing logic below...

        logging.info("Done.")

        return data_tr, data_val, scaler

    def get_data_cv(self) -> Union[pd.DataFrame, np.ndarray]:
        """Return data for CV iteration."""
        return self._data_cv

    def get_data_test(self) -> Union[pd.DataFrame, np.ndarray]:
        """Return unseen test set for final evaluation."""
        return self._data_test

    def _scale(
        self,
        data_tr: Union[pd.DataFrame, np.ndarray],
        data_val: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Any]:
        """Scale the data.

        Args:
            data_tr: training data
            data_val: validation data

        Returns:
            data_tr: scaled training data
            data_val: scaled validation data
            scaler: scaling object
        """
        logging.info(f"\t>> Scale data using {self.scaling} scaler...")
        # feats_to_scale = [c for c in data_tr.columns if c != TARGET_COL]
        feats_to_scale: List[str] = []
        if self.scaling == "standard":
            scaler = StandardScaler()

        # Scale cv data
        data_tr.loc[:, feats_to_scale] = scaler.fit_transform(data_tr[feats_to_scale].values)
        data_val.loc[:, feats_to_scale] = scaler.transform(data_val[feats_to_scale].values)

        # Scale test data...

        return data_tr, data_val, scaler
