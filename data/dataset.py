"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.
"""
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from metadata import TGT_PK_COLS, UNIT_ID_COL


class TSDataset(Dataset):
    """Time series Dataset.

    Args:
        data: processed data
        split: data split

    Attributes:
        _n_samples: number of samples
        _infer: if True, the dataset is constructed for inference
            *Note: Ground truth is not provided.
    """

    TID_FEATS: List[str] = ["quarter", "month", "day", "weekday", "hour", "dayofyear", "holiday_type"]
    TS_DF_IDX: List[str] = [UNIT_ID_COL] + TGT_PK_COLS + ["is_consumption"]
    NAN_TOKEN: float = -1.0
    TID_BASE: np.ndarray = np.array([4, 12, 31, 7, 23, 365, 3]).reshape((-1, 1))

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        **dataset_cfg: Any,
    ) -> None:
        self.data = data
        self.dataset_cfg = dataset_cfg

        self.tgt_col = dataset_cfg["tgt_col"]

        self.ts_df = data.pivot(values=self.tgt_col, index=self.TS_DF_IDX, columns="datetime").reset_index()
        tid_df = data[["datetime"] + self.TID_FEATS].drop_duplicates().sort_values("datetime")
        self.tid_arr_base = tid_df[self.TID_FEATS].to_numpy().T
        cli_attr_tmp = self.ts_df[[UNIT_ID_COL] + TGT_PK_COLS].drop_duplicates().to_numpy()
        self.cli_attr_base = {cli_attr_tmp[i, 0]: cli_attr_tmp[i, 1:] for i in range(len(cli_attr_tmp))}
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

        self.t_window = dataset_cfg["t_window"]
        self.gap = dataset_cfg["gap"]
        self.horizon = dataset_cfg["horizon"]
        self.offset = self.t_window + self.gap + self.horizon
        self.n_dates = self.tid_arr_base.shape[1]

        self.data_chunks = self._chunk_data()
        self._set_n_samples()
        self._infer = False

    def _chunk_data(self) -> Dict[str, Any]:
        """Chunk the data."""
        data_chunks: Dict[str, List[np.ndarray]] = {"X": [], "y": [], "tids": [], "cli_attr": [], "cap": []}
        for uid, ts_df_uid in self.ts_df.groupby(UNIT_ID_COL):
            # Production first
            ts_df_uid = ts_df_uid.sort_values("is_consumption")

            # tgt
            # ts_arr_uid = np.log1p(ts_df_uid.to_numpy()[:, 5:])
            # tgt_div_cap
            ts_arr_uid = ts_df_uid.to_numpy()[:, 5:]
            # ts_arr_uid[0] = (ts_arr_uid[0] - 0.062498) / 0.137278
            # ts_arr_uid[1] = (ts_arr_uid[1] - 0.302223) / 0.559241
            ts_arr_uid = np.log1p(ts_arr_uid)

            for dt_s in range(0, self.n_dates - self.offset + 1, 24):
                X_s, X_e = dt_s, dt_s + self.t_window
                y_s, y_e = dt_s + self.t_window + self.gap, dt_s + self.offset

                data_chunks["X"].append(np.nan_to_num(ts_arr_uid[:, X_s:X_e], nan=self.NAN_TOKEN))
                data_chunks["y"].append(ts_arr_uid[:, y_s:y_e])
                data_chunks["tids"].append(self.tid_arr_base[:, y_s:y_e] / self.TID_BASE)
                data_chunks["cli_attr"].append(self.cli_attr_base[uid])

                # installed_capacity is at day-level (i.e., same val for the same day)
                if self.cap is not None:
                    data_chunks["cap"].append(self.cap[uid][y_s])

        return data_chunks

    def _set_n_samples(self) -> None:
        self._n_samples = len(self.data_chunks["X"])

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # Construct data sample here...
        data_sample = {
            "x": torch.tensor(self.data_chunks["X"][idx], dtype=torch.float32),
            "tids": torch.tensor(self.data_chunks["tids"][idx], dtype=torch.int32),
            "cli_attr": torch.tensor(self.data_chunks["cli_attr"][idx], dtype=torch.int32),
        }
        if self.cap is not None:
            data_sample["cap"] = torch.tensor(self.data_chunks["cap"][idx], dtype=torch.float32)
        if not self._infer:
            data_sample["y"] = torch.tensor(self.data_chunks["y"][idx], dtype=torch.float32).reshape(-1)

        return data_sample
