"""
Project metadata for global access.
Author: JiaWei
"""
import polars as pl

# == Data ==
UNIT_ID_COL = "prediction_unit_id"
TGT_COL = "target"
DBI = "data_block_id"
TGT_PK_COLS = ["county", "is_business", "product_type"]
REVEALED_TGT_COLS = [UNIT_ID_COL, "datetime", TGT_COL, "is_consumption"]
COORD_COL2ABBR = {"latitude": "lat", "longitude": "lon"}
LOC_COLS = ["lat", "lon"]
PRODUCT_TYPE2NAME = {0: "Combined", 1: "Fixed", 2: "General service", 3: "Spot"}

# == Join keys ==
CLI_JOIN_KEYS = ["county", "is_business", "product_type", "date"]
# FWTH_JOIN_KEYS = ["county", "datetime"]
# HWTH_JOIN_KEYS = ["county", "datetime"]
REVEALED_TGT_JOIN_KEYS = [UNIT_ID_COL, "datetime", "is_consumption"]
REVEALED_TGT_ROLLING_JOIN_KEYS = [UNIT_ID_COL, "datetime", "is_consumption"]

# == Groupby keys ==
FWTH_LGP_KEYS = ["county", "datetime", DBI]
FWTH_GGP_KEYS = ["datetime", DBI]

HWTH_LGP_KEYS = ["county", "datetime"]
HWTH_GGP_KEYS = ["datetime"]
REVEALED_TGT_ROLLING_GP_KEYS = [UNIT_ID_COL, "is_consumption"]

# == Polars expressions ==
# Casting
CAST_COUNTY = [pl.col("county").cast(pl.Int8)]
CAST_COORDS = [pl.col("lat").cast(pl.Float32), pl.col("lon").cast(pl.Float32)]

# == Temp ==
# Time zone has been fixed in the w data
FIX_FWTH_TIMEZONE = False
