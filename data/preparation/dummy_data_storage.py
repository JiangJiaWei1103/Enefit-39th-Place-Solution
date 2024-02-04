"""
Dummy data storage, a local mirror of OnlineDataStorage during infer.

To keep data processing aligned, data storage follows the rules,
1. Always drop `data_block_id`
    * Only keep `data_block_id` for forecast weather for safe join
2. Don't use `prediction_unit_id`, which is already covered by the
    composite primary keys (`county`, `is_business`, `product_type`)

* [ ] Truncate every df, except for `base_tgt`
* [ ] Dynamically drop sample pointed by `dt_tail` (increase with test
    iter)

Author: JiaWei Jiang
"""
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import holidays
import numpy as np
import pandas as pd
import polars as pl

from metadata import CAST_COORDS, CAST_COUNTY, COORD_COL2ABBR, TGT_PK_COLS


class DataStorage(object):
    """Data storage.

    Args:
        mode: mode of data storage, specify "online" for inference

    Attributes:
        base_tgt: revealed targets
    """

    RAW_DATA_PATH = Path("./data/raw/")
    DT_TAIL: datetime = datetime(2022, 1, 1, 0)

    # Define columns
    CLIENT_COLS = ["product_type", "county", "eic_count", "installed_capacity", "is_business", "date"]
    FWTH_COLS = [
        "latitude",
        "longitude",
        "hours_ahead",
        "temperature",
        "dewpoint",
        "cloudcover_high",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_total",
        "10_metre_u_wind_component",
        "10_metre_v_wind_component",
        "forecast_datetime",
        "direct_solar_radiation",
        "surface_solar_radiation_downwards",
        "snowfall",
        "total_precipitation",
    ]
    HWTH_COLS = [
        "datetime",
        "temperature",
        "dewpoint",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloudcover_total",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",
        "direct_solar_radiation",
        "diffuse_radiation",
        "latitude",
        "longitude",
    ]
    ELEC_COLS = ["forecast_date", "euros_per_mwh"]
    GAS_COLS = ["forecast_date", "lowest_price_per_mwh", "highest_price_per_mwh"]
    TGT_COLS = [  # For revealed targets
        "county",
        "is_business",
        "product_type",
        "target",
        "is_consumption",
        "datetime",
    ]
    TEST_COLS = [  # For the current (predicting) test DataFrame
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
        "row_id",
    ]

    def __init__(
        self,
        mode: str = "offline",
        trunc_outdated: bool = False,
        reduce_mem: bool = False,
    ) -> None:
        self.mode = mode
        self.trunc_oudated = trunc_outdated
        self.reduce_mem = reduce_mem

        # Load data
        (
            self.base_client,
            self.base_fwth,
            self.base_hwth,
            self.base_elec,
            self.base_gas,
            self.base_tgt,
        ) = self._load_data()
        if mode == "online" and trunc_outdated:
            self._trunc_outdated()
        if reduce_mem:
            self._reduce_pl_mem()

        # Setup schema
        self.client_schema = self.base_client.schema
        self.fwth_schema = self.base_fwth.schema
        self.hwth_schema = self.base_hwth.schema
        self.elec_schema = self.base_elec.schema
        self.gas_schema = self.base_gas.schema
        if mode == "online":
            self.tgt_schema = self.base_tgt.schema
            self.test_schema = self.tgt_schema.copy()
            del self.test_schema["target"]
            self.test_schema["row_id"] = pl.Int64

        # Load static auxiliary data
        self.wstn_loc2county = (
            pl.read_csv("./data/processed/wth_station_latlon2county.csv")
            .drop("")
            .rename(COORD_COL2ABBR)
            .with_columns(CAST_COUNTY + CAST_COORDS)
        )
        # holidays...
        self.holidays = list(holidays.country_holidays("EE", years=range(2021, 2026)).keys())

    def _load_data(self) -> Tuple[pl.DataFrame, ...]:
        """Specify columns to load as constant???"""
        base_client = pl.read_csv(self.RAW_DATA_PATH / "client.csv", columns=self.CLIENT_COLS, try_parse_dates=True)
        base_fwth = pl.read_csv(
            self.RAW_DATA_PATH / "forecast_weather.csv", columns=self.FWTH_COLS, try_parse_dates=True
        )
        base_hwth = pl.read_csv(
            self.RAW_DATA_PATH / "historical_weather.csv", columns=self.HWTH_COLS, try_parse_dates=True
        )
        base_elec = pl.read_csv(
            self.RAW_DATA_PATH / "electricity_prices.csv", columns=self.ELEC_COLS, try_parse_dates=True
        )
        base_gas = pl.read_csv(self.RAW_DATA_PATH / "gas_prices.csv", columns=self.GAS_COLS, try_parse_dates=True)
        if self.mode == "online":
            base_tgt = pl.read_csv(self.RAW_DATA_PATH / "train.csv", columns=self.TGT_COLS, try_parse_dates=True)
        else:
            base_tgt = None

        return (base_client, base_fwth, base_hwth, base_elec, base_gas, base_tgt)

    def _trunc_outdated(self) -> None:
        """Truncate outdated data (i.e., those dummy for fe)."""
        self.base_tgt = self.base_tgt.filter(pl.col("datetime") >= self.DT_TAIL)

    def _reduce_pl_mem(self) -> None:
        """Reduce memory usage of polars DataFrame."""
        self.base_client = _reduce_memory_usage(self.base_client, data_name="client")
        self.base_fwth = _reduce_memory_usage(self.base_fwth, data_name="fwth")
        self.base_hwth = _reduce_memory_usage(self.base_hwth, data_name="hwth")
        self.base_elec = _reduce_memory_usage(self.base_elec, data_name="elec")
        self.base_gas = _reduce_memory_usage(self.base_gas, data_name="gas")
        if self.mode == "online":
            self.base_tgt = _reduce_memory_usage(self.base_tgt, data_name="target")

    def update(
        self,
        new_client: pd.DataFrame,
        new_fwth: pd.DataFrame,
        new_hwth: pd.DataFrame,
        new_elec: pd.DataFrame,
        new_gas: pd.DataFrame,
        new_tgt: pd.DataFrame,
    ) -> None:
        new_client = pl.from_pandas(new_client[self.CLIENT_COLS], schema_overrides=self.client_schema)
        new_fwth = pl.from_pandas(new_fwth[self.FWTH_COLS], schema_overrides=self.fwth_schema)
        new_hwth = pl.from_pandas(new_hwth[self.HWTH_COLS], schema_overrides=self.hwth_schema)
        new_elec = pl.from_pandas(new_elec[self.ELEC_COLS], schema_overrides=self.elec_schema)
        new_gas = pl.from_pandas(new_gas[self.GAS_COLS], schema_overrides=self.gas_schema)
        if self.mode == "online":
            new_tgt = pl.from_pandas(new_tgt[self.TGT_COLS], schema_overrides=self.tgt_schema)

        # Truncate outdated
        if self.mode == "online" and self.trunc_oudated:
            # self.dt_tail = self.dt_tail + timedelta(days=1)
            self._trunc_outdated()

        self.base_client = pl.concat([self.base_client, new_client]).unique(TGT_PK_COLS + ["date"])
        self.base_fwth = pl.concat([self.base_fwth, new_fwth]).unique(
            ["latitude", "longitude", "hours_ahead", "forecast_datetime"]
        )
        self.base_hwth = pl.concat([self.base_hwth, new_hwth]).unique(["datetime", "latitude", "longitude"])
        self.base_elec = pl.concat([self.base_elec, new_elec]).unique(["forecast_date"])
        self.base_gas = pl.concat([self.base_gas, new_gas]).unique(["forecast_date"])
        if self.mode == "online":
            self.base_tgt = pl.concat([self.base_tgt, new_tgt]).unique(TGT_PK_COLS + ["is_consumption", "datetime"])

    def preprocess_test(self, df_test: pd.DataFrame) -> pl.DataFrame:
        df_test = df_test.rename(columns={"prediction_datetime": "datetime"})
        df_test = pl.from_pandas(df_test[self.TEST_COLS], schema_overrides=self.test_schema)
        return df_test


def _reduce_memory_usage(df: pl.DataFrame, cols_to_skip: List[str] = [], data_name: str = "") -> pl.DataFrame:
    """Reduce memory usage by dtype casting.

    Args:
        df: raw DataFrame
        cols_to_skip: columns to skip dtype casting

    Returns:
        df: DataFrame with reduced memory footprint
    """
    start_mem = df.estimated_size("mb")
    print(f"Memory usage of {data_name} DataFrame is {start_mem:.2f} MB.")

    # pl.Uint8, pl.UInt16, pl.UInt32, pl.UInt64
    NUM_INT_TYPES = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
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
    print(f"Memory usage became: {end_mem:.2f} MB.")
    print(f"-> Total {(start_mem-end_mem) / start_mem * 100}% reduced.")

    return df
