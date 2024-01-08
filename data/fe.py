"""
Feature engineer.

* [ ] Switch to lazy mode.

Author: JiaWei Jiang
"""
from typing import List

import polars as pl

from data.preparation.dummy_data_storage import DataStorage


class FeatureEngineer(object):
    """Feature engineer."""

    _feats: List[str]

    # Define common join keys

    def __init__(self, ds: DataStorage) -> None:
        self.ds = ds

        # Add options for data version control...

        self._feats = []

    @property
    def feats(self) -> List[str]:
        return self._feats

    def gen_feats(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        """Generate default and specified features."""
        # Add `date` column (day resolution)
        df_feats = df_feats.with_columns(pl.col("datetime").cast(pl.Date).alias("date"))

        for fe_func in [
            self._gen_tid_feats,
            # self._gen_pk_encs,
        ]:
            df_feats = fe_func(df_feats)

        df_feats = self._to_pandas(df_feats)

        return df_feats

    def _gen_tid_feats(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        """Generate time stamp identifiers."""
        tid_feats = [
            pl.col("datetime").dt.quarter().alias("quarter"),
            pl.col("datetime").dt.month().alias("month"),
            pl.col("datetime").dt.day().alias("day"),
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.col("datetime").dt.hour().alias("hour"),
            # ===
            # Can convert to other encoding (e.g., sin/cos)
            # Don't think it'll be effective...
            pl.col("datetime").dt.ordinal_day().alias("dayofyear"),
            # ===
        ]
        df_feats = df_feats.with_columns(tid_feats)

        return df_feats

    def _gen_pk_enc(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        """Generate primary key encoding (i.e., combination of county
        and is_business).
        """
        # df_feats = df_feats.with_columns(...)
        return df_feats

    def _gen_cli_feats(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        """Generate client features."""

        return df_feats

    def _to_pandas(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        # Convert to pandas and specify cat feats...

        return df_feats
