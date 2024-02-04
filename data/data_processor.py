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

from metadata import TGT_PK_COLS


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
        # Load data
        self.data_file = self.dp_cfg["data_file"]
        self.reduce_mem = self.dp_cfg["reduce_mem"]

        # Before data splitting
        self.feats_ver = self.dp_cfg["feats_ver"]
        self.cat_cols = self.dp_cfg["cat_cols"]
        self._load_feats()

        self.tgt_col = self.dp_cfg["tgt_col"]
        self.tgt_aux_cols = self.dp_cfg["tgt_aux_cols"]
        self.drop_rows_with_null_tgt = self.dp_cfg["drop_rows_with_null_tgt"]
        self.tgt_cols = [self.tgt_col] + self.tgt_aux_cols

        # After data splitting...
        self.scaling = None

    def _load_feats(self) -> None:
        """Load features."""
        with open(self.data_path / "feats" / f"v{self.feats_ver}.pkl", "rb") as f:
            self._feats = pickle.load(f)
        assert isinstance(self._feats, list), "Please provide feature list."
        self._num_feats = [feat for feat in self._feats if feat not in self.cat_cols]
        self._cat_feats = self.cat_cols

    def _load_data(self) -> None:
        """Load raw data."""
        # cols_to_load = TGT_PK_COLS + [UNIT_ID_COL, "datetime", "is_consumption"]
        cols_to_load = TGT_PK_COLS + ["datetime", "is_consumption"]
        for feat in self.feats:
            if feat not in cols_to_load:
                cols_to_load.append(feat)
        for tgt_col in self.tgt_cols:
            if tgt_col not in cols_to_load:
                cols_to_load.append(tgt_col)
        self._data_cv = pl.read_parquet(self.data_path / self.data_file, columns=cols_to_load)
        self._data_test = None

        if self.reduce_mem:
            logging.info("Reduce memory footprint of loaded DataFrame...")
            self._data_cv = self._reduce_memory_usage(self._data_cv)

    def _reduce_memory_usage(self, df: pl.DataFrame, cols_to_skip: List[str] = []) -> pl.DataFrame:
        """Reduce memory usage by dtype casting.

        Args:
            df: raw DataFrame
            cols_to_skip: columns to skip dtype casting

        Returns:
            df: DataFrame with reduced memory footprint
        """
        start_mem = df.estimated_size("mb")
        logging.info(f"\t>> Memory usage of DataFrame is {start_mem:.2f} MB.")

        NUM_INT_TYPES = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32]
        NUM_FLOAT_TYPES = [pl.Float32, pl.Float64]
        for col in df.columns:
            if col in cols_to_skip:
                continue

            col_type = df[col].dtype
            c_min = df[col].min()
            c_max = df[col].max()
            if col_type in NUM_INT_TYPES:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(df[col].cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(df[col].cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(df[col].cast(pl.Int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df = df.with_columns(df[col].cast(pl.Int64))
            elif col_type in NUM_FLOAT_TYPES:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(df[col].cast(pl.Float32))
                else:
                    pass
            elif col_type == pl.Utf8:
                df = df.with_columns(df[col].cast(pl.Categorical))
            else:
                pass
        end_mem = df.estimated_size("mb")
        logging.info(f"\t>> Memory usage became: {end_mem:.2f} MB.")
        logging.info(f"\t>> Total {(start_mem-end_mem) / start_mem * 100:.2f}% reduced.")

        return df

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

        # if self.tgt_col == "target_f64" and self.drop_outliers:
        #     logging.info(f"\t>> Drop outliers...")
        #     n_samps_s = len(self._data_cv)
        #     self._data_cv = self._data_cv.filter(~outlier_mask)
        #     n_samps_e = len(self._data_cv)
        #     logging.info(f"\t>> -> #Drop outliers: {n_samps_s - n_samps_e}")

        logging.info("\t>> Add `seg` column for mimicing unseen units...")
        self._data_cv = self._data_cv.with_columns(pl.concat_str(*TGT_PK_COLS, separator="_").alias("seg"))

        logging.info("\t>> Convert pl.DataFrame to pd.DataFrame...")
        self._data_cv = self._data_cv.to_pandas()

        if len(self.cat_feats) != 0:
            logging.info(f"\t>> Specify categorical features {self.cat_feats}")
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
