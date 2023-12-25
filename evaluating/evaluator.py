"""
Evaluator definition.
Author: JiaWei Jiang

This file contains the definition of evaluator used during evaluation
process.

* [ ] Integrate `torchmetrics`?
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from criterion.custom import MaskedLoss


class Evaluator(object):
    """Custom evaluator.

    Args:
        metric_names: evaluation metrics
    """

    eval_metrics: Dict[str, Callable[..., float]] = {}
    EPS: float = 1e-6

    def __init__(self, metric_names: List[str]) -> None:
        self.metric_names = metric_names

        self._build()

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        scaler: Optional[object] = None,
    ) -> Dict[str, float]:
        """Run evaluation using pre-specified metrics.

        Args:
            y_true: ground truth
            y_pred: prediction
            scaler: scaling object

        Returns:
            eval_result: evaluation performance report
        """
        if scaler is not None:
            # Do inverse transformation to rescale y values
            y_pred, y_true = self._rescale_y(y_pred, y_true, scaler)

        eval_result = {}
        for metric_name, metric in self.eval_metrics.items():
            y_pred = y_pred.reshape(-1, 2, 24)
            y_true = y_true.reshape(-1, 2, 24)

            yp_pred, yp_true = y_pred[:, 0, :], y_true[:, 0, :]
            yc_pred, yc_true = y_pred[:, 1, :], y_true[:, 1, :]

            eval_result[f"prod_{metric_name}"] = metric(yp_pred, yp_true)
            eval_result[f"cons_{metric_name}"] = metric(yc_pred, yc_true)
            eval_result[metric_name] = metric(y_pred, y_true)

        return eval_result

    def _build(self) -> None:
        """Build evaluation metric instances."""
        for metric_name in self.metric_names:
            if metric_name == "mmae":
                self.eval_metrics[metric_name] = self._MMAE
            elif metric_name == "mrmse":
                self.eval_metrics[metric_name] = self._MRMSE
            elif metric_name == "rmse":
                self.eval_metrics[metric_name] = self._RMSE
            elif metric_name == "mae":
                self.eval_metrics[metric_name] = self._MAE
            elif metric_name == "rrse":
                self.eval_metrics[metric_name] = self._RRSE
            elif metric_name == "rae":
                self.eval_metrics[metric_name] = self._RAE
            elif metric_name == "corr":
                self.eval_metrics[metric_name] = self._CORR

    def _rescale_y(self, y_pred: Tensor, y_true: Tensor, scaler: Any) -> Tuple[Tensor, Tensor]:
        """Rescale y to the original scale.

        Args:
            y_pred: prediction
            y_true: ground truth
            scaler: scaling object

        Returns:
            y_pred: rescaled prediction
            y_true: rescaled ground truth
        """
        # Do inverse transform...

        return y_pred, y_true

    def _MMAE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Masked mean absolute error.

        Args:
            y_pred: prediction
            y_true: groudtruths

        Returns:
            mmae: masked mean absolute error
        """
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        mmae = MaskedLoss("l1")(y_pred, y_true).item()

        return mmae

    def _MRMSE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Masked root mean squared error.

        Args:
            y_pred: prediction
            y_true: groudtruths

        Returns:
            mrmse: masked root mean squared error
        """
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        mrmse = MaskedLoss("l2")(y_pred, y_true).item()

        return mrmse

    def _RMSE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Root mean squared error.

        Args:
            y_pred: prediction
            y_true: groudtruths

        Returns:
            rmse: root mean squared error
        """
        mse = nn.MSELoss()
        rmse = torch.sqrt(mse(y_pred, y_true)).item()

        return rmse

    def _MAE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Mean absolute error.

        Args:
            y_pred: prediction
            y_true: groudtruths

        Returns:
            mae: root mean squared error
        """
        mae = nn.L1Loss()(y_pred, y_true).item()

        return mae

    def _RRSE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Root relative squared error.

        Args:
            y_pred: prediction
            y_true: groudtruths

        Returns:
            rrse: root relative squared error
        """
        mse = nn.MSELoss()
        rrse = (torch.sqrt(mse(y_pred, y_true)) / torch.std(y_true)).item()

        return rrse

    def _RAE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Relative absolute error.

        Args:
            y_pred: prediction
            y_true: groudtruths

        Returns:
            rae: relative absolute error
        """
        gt_mean = torch.mean(y_true)

        sae = nn.L1Loss(reduction="sum")  # Sum absolute error
        rae = (sae(y_pred, y_true) / sae(gt_mean.expand(y_true.shape), y_true)).item()

        return rae

    def _CORR(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Empirical correlation coefficient.

        Args:
            y_pred: prediction
            y_true: groudtruths

        Returns:
            corr: empirical correlation coefficient
        """
        pred_mean = torch.mean(y_pred, dim=0)
        pred_std = torch.std(y_pred, dim=0)
        gt_mean = torch.mean(y_true, dim=0)
        gt_std = torch.std(y_true, dim=0)

        gt_idx_leg = gt_std != 0
        idx_leg = gt_idx_leg
        corr_per_ts = torch.mean(((y_pred - pred_mean) * (y_true - gt_mean)), dim=0) / (pred_std * gt_std)
        corr = torch.mean(corr_per_ts[idx_leg]).item()

        return corr
