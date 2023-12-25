"""
Dataloader building logic.
Author: JiaWei Jiang

This file contains the basic logic of building dataloaders for training
and evaluation processes.
"""
from typing import Any, Union

import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from .dataset import TSDataset


def build_dataloader(
    data: Union[pd.DataFrame, np.ndarray], data_split: str, dataset_cfg: DictConfig, **dataloader_cfg: Any
) -> DataLoader:
    """Cretae and return dataloader.

    Args:
        data: data to be fed into torch Dataset
        data_split: data split
        dataset_cfg: hyperparameters of dataset
        dataloader_cfg: hyperparameters of dataloader

    Returns:
        dataloader: dataloader
    """
    collate = None
    shuffle = dataloader_cfg["shuffle"] if data_split == "train" else False
    dataloader = DataLoader(
        TSDataset(data, **dataset_cfg),
        batch_size=dataloader_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=dataloader_cfg["num_workers"],
        collate_fn=collate,
        pin_memory=dataloader_cfg["pin_memory"],
        drop_last=dataloader_cfg["drop_last"],
    )

    return dataloader
