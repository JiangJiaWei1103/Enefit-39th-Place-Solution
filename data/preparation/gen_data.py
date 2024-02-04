"""
Script for generating processed data for local CV.

* [ ] Add sanity check and test for generated features
    * A simple visualization of a sampled unit is naive
    * How to consider corner cases for generated features (e.g., nan)

Author: JiaWei Jiang
"""
from pathlib import Path

import polars as pl

from data.fe import FeatureEngineer

from .dummy_data_storage import DataStorage, _reduce_memory_usage

# Define data path
RAW_DATA_PATH = Path("./data/raw/")
PROC_DATA_PATH = Path("./data/processed/")

# Define const config
DS_MODE = "offline"
TRUNC_OUTDATED = False
REDUCE_MEM = True

# Define configuration for feature engineering
fe_cfg = {"fill_wth_null": False, "tgt_feats_xpc": True}


def main() -> None:
    off_ds = DataStorage(DS_MODE, TRUNC_OUTDATED, REDUCE_MEM)
    fe = FeatureEngineer(off_ds, **fe_cfg)

    # train DataFrame is the offline version of online test
    # Also, it plays the role of `base_train` for target feature engineering
    train_cols = off_ds.TEST_COLS + ["target"]
    train = pl.read_csv(RAW_DATA_PATH / "train.csv", columns=train_cols, try_parse_dates=True)
    train = _reduce_memory_usage(train)

    # Preprocess data and generate features
    df_tr = fe.gen_feats(train)
    df_tr.write_parquet(PROC_DATA_PATH / "base_feats.parquet")


if __name__ == "__main__":
    main()
