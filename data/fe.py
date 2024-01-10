"""
Feature engineer.

* [ ] Switch to lazy mode.

Author: JiaWei Jiang
"""
import re
from typing import Any, List

import polars as pl

from data.preparation.dummy_data_storage import DataStorage
from metadata import CAST_COORDS, COORD_COL2ABBR, DBI, LOC_COLS, TGT_PK_COLS

BASE_FEATS: List[str] = []


class FeatureEngineer(object):
    """Feature engineer.

    Args:
        tgt_feats_xpc: if True, prod/cons targets are considered as
            features for cons/prod
    """

    _feats: List[str]

    # Define common join keys
    CLI_JOIN_KEYS: List[str] = TGT_PK_COLS + ["date"]
    REVEALED_TGT_JOIN_KEYS = TGT_PK_COLS + ["datetime"]

    # Define groupby keys
    FWTH_LGP_KEYS = ["county", DBI, "datetime"]  # gp as join
    FWTH_GGP_KEYS = [DBI, "datetime"]  # gp as join
    HWTH_LGP_KEYS = ["county", "datetime"]  # gp as join
    HWTH_GGP_KEYS = ["datetime"]  # gp as join

    def __init__(self, ds: DataStorage, **fe_cfg: Any) -> None:
        self.ds = ds

        # Add options for data version control...
        self.fill_wth_null = fe_cfg["fill_wth_null"]
        self.tgt_feats_xpc = fe_cfg["tgt_feats_xpc"]

    @property
    def feats(self) -> List[str]:
        return self._feats

    def gen_feats(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        """Generate default and specified features.

        Args:
            df_feats: base DataFrame for fe
                *Note: Pass test chunk for inference and complete
                    training set for training
        """
        # Add `date` column (day resolution)
        df_feats = df_feats.with_columns(pl.col("datetime").cast(pl.Date).alias("date"))

        self._feats = []
        for fe_func in [
            self._gen_tid_feats,
            # self._gen_pk_encs,
            self._gen_cli_feats,
            self._gen_fwth_feats,
            self._gen_hwth_feats,
            self._gen_tgt_feats,
        ]:
            df_feats = fe_func(df_feats)

        if self.ds.mode == "online":
            df_feats = self._to_pandas(df_feats)

        return df_feats

    def _gen_tid_feats(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        """Generate time stamp identifiers."""
        print(">> Generate time stamp identifiers...")
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
        print(">> Generate client features...")
        base_client = self.ds.base_client

        feats = ["installed_capacity", "eic_count"]
        feats_map = {feat: f"{feat}_lag2d" for feat in feats}
        base_client = base_client.with_columns((pl.col("date") + pl.duration(days=2)).cast(pl.Date))
        df_feats = df_feats.join(base_client, on=self.CLI_JOIN_KEYS, how="left").rename(feats_map)

        # Extend feature list
        new_feats = list(feats_map.values())
        self._feats.extend(new_feats)

        return df_feats

    def _gen_fwth_feats(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        """Generate forecast weather features."""
        print(">> Generate forecast weather features...")
        base_fwth = self.ds.base_fwth
        wstn_loc2county = self.ds.wstn_loc2county

        # Preprocess forecast weather
        base_fwth = (
            base_fwth.rename({"forecast_datetime": "datetime"})
            .drop(["origin_datetime", "hours_ahead"])
            .rename(COORD_COL2ABBR)
            .with_columns(CAST_COORDS)
            .join(wstn_loc2county, on=LOC_COLS, how="left")
        )
        base_fwth = base_fwth.unique(LOC_COLS + ["datetime", DBI])

        # Specify base features
        cols_to_skip = LOC_COLS + ["county", "datetime", DBI]
        fwth_feats = [c for c in base_fwth.columns if c not in cols_to_skip]

        # Generate forecast weather features
        agg_stats = [
            *[pl.col(feat).mean().alias(f"{feat}_local_mean") for feat in fwth_feats],
        ]
        fwth_stats_by_county = (
            base_fwth.filter(pl.col("county").is_not_null()).group_by(self.FWTH_LGP_KEYS).agg(agg_stats)
        )
        self._feats.extend([f"{feat}_local_mean" for feat in fwth_feats])

        agg_stats = [
            *[pl.col(feat).mean().alias(f"{feat}_global_mean") for feat in fwth_feats],
        ]
        fwth_stats = base_fwth.group_by(self.FWTH_GGP_KEYS).agg(agg_stats)
        self._feats.extend([f"{feat}_global_mean" for feat in fwth_feats])

        if self.fill_wth_null:
            pass

        df_feats = df_feats.join(fwth_stats_by_county, on=self.FWTH_LGP_KEYS, how="left").join(
            fwth_stats, on=self.FWTH_GGP_KEYS, how="left"
        )

        return df_feats

    def _gen_hwth_feats(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        """Generate historical weather features."""
        print(">> Generate historical weather features...")
        base_hwth = self.ds.base_hwth
        wstn_loc2county = self.ds.wstn_loc2county

        # Preprocess historical weather features
        stns_to_drop = ["57.624.2", "57.623.2"]
        base_hwth = (
            base_hwth.rename(COORD_COL2ABBR)
            .with_columns(*CAST_COORDS, pl.concat_str([pl.col("lat"), pl.col("lon")], separator="").alias("loc"))
            .filter(~pl.col("loc").is_in(stns_to_drop))
            .join(wstn_loc2county, on=LOC_COLS, how="left")
            .drop(["loc"])
        )

        # Specify base features
        cols_to_skip = LOC_COLS + ["datetime", "county"]
        hwth_feats = [c for c in base_hwth.columns if c not in cols_to_skip]

        # Generate historical weather features
        agg_stats = [
            *[pl.col(feat).mean().alias(f"{feat}_local_mean_hist") for feat in hwth_feats],
        ]
        hwth_stats_by_county = (
            base_hwth.filter(pl.col("county").is_not_null()).group_by(self.HWTH_LGP_KEYS).agg(agg_stats)
        )
        hwth_l_feats = [f"{feat}_local_mean_hist" for feat in hwth_feats]

        agg_stats = [
            *[pl.col(feat).mean().alias(f"{feat}_global_mean_hist") for feat in hwth_feats],
        ]
        hwth_stats = base_hwth.group_by(self.HWTH_GGP_KEYS).agg(agg_stats)
        hwth_g_feats = [f"{feat}_global_mean_hist" for feat in hwth_feats]

        l_feats_map = {feat: f"{feat}_7d" for feat in hwth_l_feats}
        g_feats_map = {feat: f"{feat}_7d" for feat in hwth_g_feats}
        hwth_stats_by_county = hwth_stats_by_county.with_columns(pl.col("datetime") + pl.duration(days=7))
        hwth_stats = hwth_stats.with_columns(pl.col("datetime") + pl.duration(days=7))
        df_feats = (
            df_feats.join(hwth_stats_by_county, on=self.HWTH_LGP_KEYS, how="left")
            .rename(l_feats_map)
            .join(hwth_stats, on=self.HWTH_GGP_KEYS, how="left")
            .rename(g_feats_map)
        )
        self._feats.extend(list(l_feats_map.values()))
        self._feats.extend(list(g_feats_map.values()))

        return df_feats

    def _gen_tgt_feats(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        """Generate revealed target features."""
        print(">> Generate target features...")
        if self.ds.mode == "online":
            base_tgt = self.ds.base_tgt
            base_tgt = base_tgt.with_columns(
                (pl.col("target") / pl.col("installed_capacity_lag2d")).alias("target_div_cap_lag2d")
            )
        else:
            df_feats = df_feats.with_columns(
                (pl.col("target") / pl.col("installed_capacity_lag2d")).alias("target_div_cap_lag2d")
            )
            tgt_cols = ["target", "target_div_cap_lag2d"]
            base_tgt = df_feats.select(TGT_PK_COLS + ["is_consumption", "datetime"] + tgt_cols)

        aux_cols = ["installed_capacity_lag2d"]
        base_tgt = base_tgt.drop(aux_cols)
        revealed_tgt_fe = _RevealedTgtFE(tgt_cols=tgt_cols, n_days_to_look=6, cross_pc=self.tgt_feats_xpc)
        df_tmp, tgt_feats = revealed_tgt_fe.run(base_tgt)

        df_feats = df_feats.join(df_tmp, on=self.REVEALED_TGT_JOIN_KEYS, how="left")
        self._feats.extend(tgt_feats)

        return df_feats

    def _to_pandas(self, df_feats: pl.DataFrame) -> pl.DataFrame:
        # Convert to pandas and specify cat feats...

        return df_feats


class _RevealedTgtFE(object):
    """Feature engineer considering lookback targets.

    Revealed targets include the raw and processed ones (e.g., rolling,
    divided by `installed_capacity`).

    Args:
        n_days_to_look: number of lookback days, starting at 2 days
        tgt_cols: target columns to lookback
        cross_pc: if True, revealed prod/cons is used as features for
            cons/prod
    """

    BASE_LOOKBACK_DAY: int = 2
    BASE_LOOKBACK_HOUR: int = 25
    DAY_ENC_DIM: int = 23  # Closet hour-gap revealed targets

    JOIN_KEYS = TGT_PK_COLS + ["is_consumption", "datetime"]

    def __init__(self, tgt_cols: List[str], n_days_to_look: int = 6, cross_pc: bool = True) -> None:
        self.tgt_cols = tgt_cols
        self.n_days_to_look = n_days_to_look
        self.cross_pc = cross_pc
        self.n_tgt_cols = len(tgt_cols)

        self._lookback_dcols = [
            f"{tgt_col}_lag{d+self.BASE_LOOKBACK_DAY}d" for d in range(n_days_to_look) for tgt_col in tgt_cols
        ]
        # self._lookback_hcols = []
        self._feats = self._lookback_dcols  # + self._lookback_hcols

    @property
    def feats(self) -> List[str]:
        return self._feats

    def run(self, df_y: pl.DataFrame) -> pl.LazyFrame:
        """Run feature engineering

        Args:
            df_y: DataFrame containing target information

        Returns:
            df_y_lookback: DataFrame containing revealed targets as
                features
        """
        df_y_raw = df_y.lazy()
        df_y_lookback = df_y.lazy()

        for d in range(self.n_days_to_look):
            feats_map = dict(zip(self.tgt_cols, self._lookback_dcols[d * self.n_tgt_cols : (d + 1) * self.n_tgt_cols]))

            if d == 0:
                # No need to shift base lookback (2 day here)
                df_y_lookback = df_y_lookback.rename(feats_map)
            else:
                df_y_shifted = df_y_raw.with_columns(
                    (pl.col("datetime") + pl.duration(days=d)).alias("datetime")
                ).rename(feats_map)
                df_y_lookback = df_y_lookback.join(df_y_shifted, on=self.JOIN_KEYS, how="left")
        # Add base lookback day as base common shift
        df_y_lookback = df_y_lookback.with_columns(pl.col("datetime") + pl.duration(days=self.BASE_LOOKBACK_DAY))

        if self.cross_pc:
            # Prod/Cons long to wide
            df_yp_lookback = df_y_lookback.filter(pl.col("is_consumption") == 0).drop("is_consumption")
            df_yc_lookback = df_y_lookback.filter(pl.col("is_consumption") == 1).drop("is_consumption")
            df_y_lookback = (
                df_yp_lookback.join(df_yc_lookback, on=TGT_PK_COLS + ["datetime"], suffix="_cons").rename(
                    {feat: f"{feat}_prod" for feat in self.feats}
                )
            ).collect()

            # ===
            # Tmp. workaround
            tgt_col_prefix = "|".join([f"{tgt_col}_lag" for tgt_col in self.tgt_cols])
            tgt_feats = [c for c in df_y_lookback.columns if re.search(f"{tgt_col_prefix}.*", c)]
            # ===

        return df_y_lookback, tgt_feats
