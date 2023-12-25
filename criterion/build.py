"""
Loss criterion building logic.
Author: JiaWei Jiang

This file contains the basic logic of building loss criterion for
training and evaluation processes.
"""
from typing import Any, Optional

import torch.nn as nn
from torch.nn.modules.loss import _Loss

from .custom import MaskedLoss


def build_criterion(**loss_fn_cfg: Any) -> Optional[_Loss]:
    """Build and return the loss criterion.

    In some scenarios, it's better to define loss criterion along
    with model architecture to enable access to intermediate
    representation (e.g., reconstruction loss). In this case, default
    building of loss criterion is disabled and nothing is returned.

    Args:
        loss_fn_cfg: hyperparameters for building loss function

    Returns:
        criterion: loss criterion
    """
    loss_fn = loss_fn_cfg["name"]

    criterion: _Loss
    if loss_fn == "l1":
        criterion = nn.L1Loss()
    elif loss_fn == "l2":
        criterion = nn.MSELoss()
    elif loss_fn == "ml1":
        criterion = MaskedLoss("l1")
    elif loss_fn == "ml1":
        criterion = MaskedLoss("l2")
    elif loss_fn == "mtl":
        print("Loss criterion default building is disabled...")
        criterion = None
    else:
        raise RuntimeError(f"Loss criterion {loss_fn} isn't registered...")

    return criterion
