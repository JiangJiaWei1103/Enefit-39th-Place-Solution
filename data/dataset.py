"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.
"""
import logging
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from metadata import TGT_PK_COLS, UNIT_ID_COL

fwth_local_mean_max = {
    "temperature_local_mean": 31.688623428344727,
    "dewpoint_local_mean": 21.145593643188477,
    "cloudcover_high_local_mean": 1.0000076293945312,
    "cloudcover_low_local_mean": 1.0000076293945312,
    "cloudcover_mid_local_mean": 1.0000076293945312,
    "cloudcover_total_local_mean": 1.0000076293945312,
    "10_metre_u_wind_component_local_mean": 15.26400375366211,
    "10_metre_v_wind_component_local_mean": 14.370864868164062,
    "direct_solar_radiation_local_mean": 917.0106201171875,
    "surface_solar_radiation_downwards_local_mean": 834.2266845703125,
    "snowfall_local_mean": 0.0032722156029194593,
    "total_precipitation_local_mean": 0.01346588134765625,
}
hwth_local_mean_max = {
    "temperature_local_mean_hist": 30.799999237060547,
    "dewpoint_local_mean_hist": 21.174999237060547,
    "rain_local_mean_hist": 5.650000095367432,
    "snowfall_local_mean_hist": 1.9600000381469727,
    "surface_pressure_local_mean_hist": 1049.0167236328125,
    "cloudcover_total_local_mean_hist": 100.0,
    "cloudcover_low_local_mean_hist": 100.0,
    "cloudcover_mid_local_mean_hist": 100.0,
    "cloudcover_high_local_mean_hist": 100.0,
    "windspeed_10m_local_mean_hist": 15.967171669006348,
    "winddirection_10m_local_mean_hist": 360.0,
    "shortwave_radiation_local_mean_hist": 826.2000122070312,
    "direct_solar_radiation_local_mean_hist": 710.5999755859375,
    "diffuse_radiation_local_mean_hist": 376.25,
}


class TSDataset(Dataset):
    """Time series Dataset.

    Args:
        data: processed data
        data_split: data split

    Attributes:
        _n_samples: number of samples
        _infer: if True, the dataset is constructed for inference
            *Note: Ground truth is not provided.
    """

    TID_FEATS: List[str] = ["quarter", "month", "day", "weekday", "hour", "dayofyear", "holiday_type"]
    TS_DF_IDX: List[str] = [UNIT_ID_COL] + TGT_PK_COLS + ["is_consumption"]
    NAN_TOKEN: float = -1.0
    TID_BASE: np.ndarray = np.array([4, 12, 31, 7, 23, 365, 3]).reshape((-1, 1))
    CLI_ATTR_BASE: np.ndarray = np.array([15, 1, 3])

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        data_split: str,
        **dataset_cfg: Any,
    ) -> None:
        self.data = data
        self.data_split = data_split
        self.dataset_cfg = dataset_cfg

        self.tgt_col = dataset_cfg["tgt_col"]

        self.ts_df = data.pivot(values=self.tgt_col, index=self.TS_DF_IDX, columns="datetime").reset_index()
        tid_df = data[["datetime"] + self.TID_FEATS].drop_duplicates().sort_values("datetime")
        self.tid_arr_base = tid_df[self.TID_FEATS].to_numpy().T
        cli_attr_tmp = self.ts_df[[UNIT_ID_COL] + TGT_PK_COLS].drop_duplicates().to_numpy()
        self.cli_attr_base = {
            cli_attr_tmp[i, 0]: cli_attr_tmp[i, 1:] / self.CLI_ATTR_BASE for i in range(len(cli_attr_tmp))
        }
        if "div_cap" in self.tgt_col:
            assert "installed_capacity" in data, "Please provide installed_capacity."
            cap_df = (
                data[[UNIT_ID_COL, "datetime", "installed_capacity"]]
                .drop_duplicates()
                .reset_index(drop=True)
                .pivot(values="installed_capacity", index=UNIT_ID_COL, columns="datetime")
                .sort_index()
            )
            self.cap = {uid: np.array(r) for uid, r in cap_df.iterrows()}
        else:
            self.cap = None

        # Forecast weather
        fwth_feats = [c for c in data.columns if c.endswith("_local_mean")]
        if len(fwth_feats) > 0:  # Tmp
            fwth_df = data[[UNIT_ID_COL, "datetime"] + fwth_feats].drop_duplicates().reset_index(drop=True)
            # ===
            # Max-norm
            # This is wrong for validation set ...
            # fwth_df[fwth_feats] = fwth_df[fwth_feats] / fwth_df[fwth_feats].max()
            for feat in fwth_feats:
                fwth_df[feat] = fwth_df[feat] / fwth_local_mean_max[feat]
            # ===
            fwth_df["fwth_feats"] = fwth_df[fwth_feats].values.tolist()
            fwth_df = fwth_df.pivot(values="fwth_feats", index=UNIT_ID_COL, columns="datetime")
            self.fwth_base = dict(zip(fwth_df.index, fwth_df.values))

            dummy = [np.NaN for _ in range(12)]
            for k, v in self.fwth_base.items():
                fix = []
                for vv in v:
                    if isinstance(vv, list):
                        fix.append(vv)
                    else:
                        fix.append(dummy)
                self.fwth_base[k] = np.array(fix)
                del fix
        else:
            self.fwth_base = None
        # Historical weather
        hwth_feats = [c for c in data.columns if c.endswith("_local_mean_hist")]
        if len(hwth_feats) > 0:  # Tmp
            hwth_df = data[[UNIT_ID_COL, "datetime"] + hwth_feats].drop_duplicates().reset_index(drop=True)
            # ===
            # Max-norm
            for feat in hwth_feats:
                hwth_df[feat] = hwth_df[feat] / hwth_local_mean_max[feat]
            # ===
            hwth_df["hwth_feats"] = hwth_df[hwth_feats].values.tolist()
            hwth_df = hwth_df.pivot(values="hwth_feats", index=UNIT_ID_COL, columns="datetime")
            self.hwth_base = dict(zip(hwth_df.index, hwth_df.values))

            dummy = [np.NaN for _ in range(14)]
            for k, v in self.hwth_base.items():
                fix = []
                for vv in v:
                    if isinstance(vv, list):
                        fix.append(vv)
                    else:
                        fix.append(dummy)
                self.hwth_base[k] = np.array(fix)
                del fix
        else:
            self.hwth_base = None
        # Price
        price_feats = [c for c in data.columns if "per_mwh" in c]
        if len(price_feats) > 0:
            price_df = data[[UNIT_ID_COL, "datetime"] + price_feats].drop_duplicates().reset_index(drop=True)
            # ===
            # Log-transform
            for feat in price_feats:
                price_df[feat] = np.log1p(price_df[feat])
            # ===
            price_df["price_feats"] = price_df[price_feats].values.tolist()
            price_df = price_df.pivot(values="price_feats", index=UNIT_ID_COL, columns="datetime")
            self.price_base = dict(zip(price_df.index, price_df.values))

            dummy = [np.NaN for _ in range(4)]
            for k, v in self.price_base.items():
                fix = []
                for vv in v:
                    if isinstance(vv, list):
                        fix.append(vv)
                    else:
                        fix.append(dummy)
                self.price_base[k] = np.array(fix)
                del fix
        else:
            self.price_base = None

        self.t_window = dataset_cfg["t_window"]
        self.gap = dataset_cfg["gap"]
        self.horizon = dataset_cfg["horizon"]
        self.offset = self.t_window + self.gap + self.horizon
        self.n_dates = self.tid_arr_base.shape[1]

        self.add_x_tids = dataset_cfg["add_x_tids"]
        self.drop_x_nan_pct = dataset_cfg["drop_x_nan_pct"]

        self.data_chunks = self._chunk_data()
        self._set_n_samples()
        self._infer = False

    def _set_n_samples(self) -> None:
        self._n_samples = len(self.data_chunks["X"])

    def _chunk_data(self) -> Dict[str, Any]:
        """Chunk the data."""
        data_chunks: Dict[str, List[np.ndarray]] = {
            "X": [],
            "X_tids": [],
            "hwth": [],
            "price": [],
            "y": [],
            "tids": [],
            "cli_attr": [],
            "fwth": [],
            "cap": [],
        }
        cnt, drop_cnt = 0, 0
        for uid, ts_df_uid in self.ts_df.groupby(UNIT_ID_COL):
            # Production first
            ts_df_uid = ts_df_uid.sort_values("is_consumption")

            # tgt
            # ts_arr_uid = np.log1p(ts_df_uid.to_numpy()[:, 5:])
            # tgt_div_cap
            ts_arr_uid = ts_df_uid.to_numpy()[:, 5:]
            #
            # ts_arr_uid[0] = (ts_arr_uid[0] - 0.062498) / 0.137278
            # ts_arr_uid[1] = (ts_arr_uid[1] - 0.302223) / 0.559241
            ts_arr_uid = np.log1p(ts_arr_uid)

            for dt_s in range(0, self.n_dates - self.offset + 1, 24):
                cnt += 1
                X_s, X_e = dt_s, dt_s + self.t_window
                y_s, y_e = dt_s + self.t_window + self.gap, dt_s + self.offset
                x = np.nan_to_num(ts_arr_uid[:, X_s:X_e], nan=self.NAN_TOKEN)
                y = ts_arr_uid[:, y_s:y_e]
                if self._drop_sample(x, y):
                    drop_cnt += 1
                    continue

                # Historical observed values
                data_chunks["X"].append(x)
                if self.add_x_tids:
                    data_chunks["X_tids"].append(self.tid_arr_base[:, X_s:X_e] / self.TID_BASE)
                if self.hwth_base is not None:
                    data_chunks["hwth"].append(np.nan_to_num(np.vstack(self.hwth_base[uid][X_s:X_e]).T, 0))
                if self.price_base is not None:
                    data_chunks["price"].append(
                        np.nan_to_num(np.vstack(self.price_base[uid][X_s + 24 : X_e + 24]).T, 0)
                    )

                data_chunks["y"].append(y)
                data_chunks["tids"].append(self.tid_arr_base[:, y_s:y_e] / self.TID_BASE)
                data_chunks["cli_attr"].append(self.cli_attr_base[uid])
                if self.fwth_base is not None:
                    data_chunks["fwth"].append(np.nan_to_num(np.vstack(self.fwth_base[uid][y_s:y_e]).T, 0))  # (C, T)

                # installed_capacity is at day-level (i.e., same val for the same day)
                if self.cap is not None:
                    data_chunks["cap"].append(self.cap[uid][y_s])
        logging.info(f"#Samples in {self.data_split} set | total {cnt}, drop {drop_cnt} => remain {cnt-drop_cnt}")

        return data_chunks

    def _drop_sample(self, x: np.ndarray, y: np.ndarray) -> bool:
        drop = False

        # Take prod to determine missing ratio (p/c miss at the same time)
        drop_x = (x[0, :] == self.NAN_TOKEN).sum() / self.t_window * 100 >= self.drop_x_nan_pct
        drop_y = np.isnan(y).all()
        if self.data_split == "train":
            drop = drop_x or drop_y
        else:
            # Consider severe missing lookback during evaluation
            drop = drop_y

        return drop

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # Construct data sample here...
        data_sample = {
            "x": torch.tensor(self.data_chunks["X"][idx], dtype=torch.float32),
            "tids": torch.tensor(self.data_chunks["tids"][idx], dtype=torch.float32),
            "cli_attr": torch.tensor(self.data_chunks["cli_attr"][idx], dtype=torch.float32),
        }
        if self.add_x_tids:
            data_sample["x_tids"] = torch.tensor(self.data_chunks["x_tids"][idx], dtype=torch.float32)
        if self.hwth_base is not None:
            data_sample["hwth"] = torch.tensor(self.data_chunks["hwth"][idx], dtype=torch.float32)
        if self.price_base is not None:
            data_sample["price"] = torch.tensor(self.data_chunks["price"][idx], dtype=torch.float32)
        if self.fwth_base is not None:
            data_sample["fwth"] = torch.tensor(self.data_chunks["fwth"][idx], dtype=torch.float32)

        if not self._infer:
            data_sample["y"] = torch.tensor(self.data_chunks["y"][idx], dtype=torch.float32).reshape(
                -1,
            )
        if self.cap is not None:
            data_sample["cap"] = torch.tensor(self.data_chunks["cap"][idx], dtype=torch.float32)

        return data_sample
