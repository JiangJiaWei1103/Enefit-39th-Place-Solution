"""
Project metadata for global access.
Author: JiaWei
"""
# == Data ==
UNIT_ID_COL = "prediction_unit_id"
TGT_COL = "target"
DBI = "data_block_id"
TGT_PK_COLS = ["county", "is_business", "product_type"]
REVEALED_TGT_COLS = [UNIT_ID_COL, "datetime", TGT_COL, "is_consumption"]
COORD_COL2ABBR = {"latitude": "lat", "longitude": "lon"}
PRODUCT_TYPE2NAME = {0: "Combined", 1: "Fixed", 2: "General service", 3: "Spot"}

# == Join keys ==
CLI_JOIN_KEYS = ["county", "is_business", "product_type", DBI]
FWTH_JOIN_KEYS = ["county", DBI, "datetime"]  # datetime is the fixed version of forecast_datetime
HWTH_JOIN_KEYS = ["county", "datetime"]
REVEALED_TGT_JOIN_KEYS = [UNIT_ID_COL, "datetime", "is_consumption"]
REVEALED_TGT_ROLLING_JOIN_KEYS = [UNIT_ID_COL, "datetime", "is_consumption"]

# == Groupby keys ==
REVEALED_TGT_ROLLING_GP_KEYS = [UNIT_ID_COL, "is_consumption"]

# == Temp ==
# Time zone has been fixed in the raw data
FIX_FWTH_TIMEZONE = False
