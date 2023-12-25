"""
Early stopping tracker.
Author: JiaWei Jiang
"""
from typing import Optional


class EarlyStopping(object):
    """Monitor whether the specified metric improves or not.

    If metric doesn't improve for the `patience` epochs, then the
    training and evaluation processes will stop early.

    Args:
        patience: tolerance for number of epochs when model can't
            improve the specified score (e.g., loss, metric)
        mode: performance determination mode, the choices can be,
            {'min', 'max'}
        tr_loss_thres: stop training immediately once training loss
            reaches this threshold
    """

    _best_score: float
    _stop: bool
    _wait_count: int

    def __init__(
        self,
        patience: int = 10,
        mode: str = "min",
        tr_loss_thres: Optional[float] = None,
    ):
        self.patience = patience
        self.mode = mode
        self.tr_loss_thres = tr_loss_thres
        self._setup()

    def _setup(self) -> None:
        """Setup es tracker."""
        if self.mode == "min":
            self._best_score = 1e18
        elif self.mode == "max":
            self._best_score = -1 * 1e-18
        self._stop = False
        self._wait_count = 0

    def step(self, score: float) -> None:
        """Update states of es tracker.

        Args:
            score: specified score in the current epoch
        """
        if self.tr_loss_thres is not None:
            if score <= self.tr_loss_thres:
                self._stop = True
        else:
            score_adj = score if self.mode == "min" else -score
            if score_adj < self._best_score:
                self._best_score = score_adj
                self._wait_count = 0
            else:
                self._wait_count += 1

            if self._wait_count >= self.patience:
                self._stop = True

    def stop(self) -> bool:
        return self._stop
