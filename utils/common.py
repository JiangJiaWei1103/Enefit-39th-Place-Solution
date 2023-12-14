"""
Common utility functions used in training and evaluation processes.
Author: JiaWei Jiang
"""
import time
from collections import defaultdict
from decimal import Decimal
from typing import Dict, List

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module

import wandb


def count_params(model: Module) -> str:
    """Count number of parameters in model.

    Args:
        model: model instance

    Returns:
        n_params: number of parameters in model, represented in
            scientific notation
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = f"{Decimal(str(n_params)):.4E}"

    return n_params


def dictconfig2dict(cfg: DictConfig) -> Dict:
    """Convert OmegaConf config object to primitive container."""
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)

    return cfg_resolved


class Profiler(object):
    """Profiler for probing time cost of the designated process."""

    t_start: float

    def __init__(self) -> None:
        self.proc_type = ""
        self.t_elapsed: Dict[str, List[float]] = defaultdict(list)

    def start(self, proc_type: str = "train") -> None:
        self.proc_type = proc_type
        self.t_start = time.time()

    def stop(self, record: bool = True) -> None:
        if record:
            self.t_elapsed[self.proc_type].append(time.time() - self.t_start)

    def summarize(self, log_wnb: bool = True) -> None:
        print("\n=====Profile Summary=====")
        for proc_type, t_elapsed in self.t_elapsed.items():
            t_avg = np.mean(t_elapsed)
            t_std = np.std(t_elapsed)
            print(f"{proc_type.upper()} takes {t_avg:.2f} ± {t_std:.2f} (sec/epoch)")

            if log_wnb:
                wandb.log({f"{proc_type}_time": {"avg": t_avg, "t_std": t_std}})


class AvgMeter(object):
    """Meter computing and storing current and average values.

    Args:
        name: name of the value to track
    """

    _val: float
    _sum: float
    _cnt: int
    _avg: float

    def __init__(self, name: str) -> None:
        self.val_name = name

        self._reset()

    def update(self, val: float, n: int = 1) -> None:
        self._val = val
        self._sum += val * n
        self._cnt += n
        self._avg = self._sum / self._cnt

    @property
    def val_cur(self) -> float:
        """Return current value."""
        return self._val

    @property
    def avg_cur(self) -> float:
        """Return current average value."""
        return self._avg

    def _reset(self) -> None:
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._cnt = 0
