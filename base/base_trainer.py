"""
Base class definition for all customized trainers.
Author: JiaWei Jiang

* [ ] Design better profiling workflow.
* [ ] Support ReduceLROnPlateau with the specified metric.
* [ ] Support early stopping with the specified metric.
* [ ] Optionally run final evaluation on training set.
* [ ] Return prediction from final evaluation or not.
* [ ] Design `eval()` for datasets with ground truth and rewrite `test`
    for pure inference (ref: lightning)?
"""
import json
from abc import abstractmethod
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import wandb
from evaluating.evaluator import Evaluator

# from utils.common import Profiler
from utils.early_stopping import EarlyStopping
from utils.model_checkpoint import ModelCheckpoint


class BaseTrainer:
    """Base class for all customized trainers.

    Args:
        logger: message logger
        trainer_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_skd: learning rate scheduler
        ckpt_path: path to save model checkpoints
        es: early stopping tracker
        evaluator: task-specific evaluator
        use_wandb: if True, training and evaluation processes are
            tracked with wandb
    """

    train_loader: DataLoader  # Tmp. workaround
    eval_loader: DataLoader  # Tmp. workaround
    # profiler: Profiler = Profiler()

    def __init__(
        self,
        logger: Logger,
        trainer_cfg: DictConfig,
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
        ckpt_path: Path,
        evaluator: Evaluator,
        use_wandb: bool,
    ):
        self.logger = logger
        self.trainer_cfg = trainer_cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_skd = lr_skd
        self.ckpt_path = ckpt_path
        self.evaluator = evaluator
        self.use_wandb = use_wandb

        self.device = trainer_cfg["device"]
        self.epochs = trainer_cfg["epochs"]
        self.one_batch_only = trainer_cfg["one_batch_only"]
        self.grad_accum_steps = trainer_cfg["grad_accum_steps"]
        self.step_per_batch = trainer_cfg["step_per_batch"]

        # Model checkpoint
        self.model_ckpt = ModelCheckpoint(ckpt_path, **trainer_cfg["model_ckpt"])

        # Early stopping
        if trainer_cfg["es"]["patience"] != 0:
            self.es = EarlyStopping(**trainer_cfg["es"])
        else:
            self.es = None

        self._iter = 0
        self._track_best_model = True  # (Deprecated)

    def train_eval(self, proc_id: int) -> None:
        """Run training and evaluation processes.

        Args:
            proc_id: identifier of the current process
        """
        self.logger.info("Start training and evaluation processes...")
        for epoch in range(self.epochs):
            self.epoch = epoch  # For interior use
            train_loss = self._train_epoch()
            val_loss, val_result, _ = self._eval_epoch()

            # Adjust learning rate
            if self.lr_skd is not None and not self.step_per_batch:
                if isinstance(self.lr_skd, lr_scheduler.ReduceLROnPlateau):
                    self.lr_skd.step(val_loss)
                else:
                    self.lr_skd.step()

            # Track and log process result (by epoch)
            self._log_proc(epoch, train_loss, val_loss, val_result)

            # Record the best checkpoint
            self.model_ckpt.step(
                epoch, self.model, val_loss, val_result, last_epoch=False if epoch != self.epochs - 1 else True
            )

            # Check early stopping is triggered or not
            if self.es is not None:
                self.es.step(val_loss)
                if self.es.stop:
                    self.logger.info(f"Early stopping is triggered at epoch {epoch}, training process is halted.")
                    break
        if self.use_wandb:
            wandb.log({"best_epoch": self.model_ckpt.best_epoch + 1})  # `epoch` starts from 0

        # Run final evaluation
        final_prf_report, y_preds = self._run_final_eval()
        self._log_best_prf(final_prf_report)

    def train_only(self) -> None:
        """Run training-only process w/o `_eval_epoch`.

        This is commonly used in full-train mode, in which all of the
        available training data is fed into the model.

        Returns:
            None
        """
        for epoch in range(self.epochs):
            self.epoch = epoch
            train_loss = self._train_epoch()

            # Adjust learning rate
            if self.lr_skd is not None and not self.step_per_batch:
                if isinstance(self.lr_skd, lr_scheduler.ReduceLROnPlateau):
                    self.lr_skd.step(train_loss)
                else:
                    self.lr_skd.step()

            # Track and log process result
            self._log_proc(epoch, train_loss)

            # Save the last ckpt
            if epoch == self.epochs - 1:
                self.model_ckpt.save_ckpt(self.model, "last")

    def test(self, proc_id: int, test_loader: DataLoader) -> Tensor:
        """Run evaluation process on unseen test data using the
        designated model checkpoint.

        Args:
            proc_id: identifier of the current process
            test_loader: test data loader

        Returns:
            y_pred: prediction on test set
        """
        self.eval_loader = test_loader
        _, eval_result, y_pred = self._eval_epoch(return_output=True)
        test_prf_report = {"test": eval_result}
        self._log_best_prf(test_prf_report)

        return y_pred

    @abstractmethod
    def _train_epoch(self) -> Union[float, Dict[str, float]]:
        """Run training process for one epoch.

        Returns:
            train_loss_avg: average training loss over batches
                *Note: If MTL is used, returned object will be dict
                    containing losses of sub-tasks and the total loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(self, return_output: bool = False) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Args:
            return_output: whether to return prediction

        Returns:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: prediction
        """
        raise NotImplementedError

    def _log_proc(
        self,
        epoch: int,
        train_loss: Union[float, Dict[str, float]],
        val_loss: Optional[float] = None,
        val_result: Optional[Dict[str, float]] = None,
        proc_id: Optional[str] = None,
    ) -> None:
        """Log message of training process.

        Args:
            epoch: current epoch number
            train_loss: training loss
            val_loss: validation loss
            val_result: evaluation performance report
            proc_id: identifier of the current process
        """
        proc_msg = [f"Epoch{epoch} [{epoch+1}/{self.epochs}]"]

        # Construct training loss message
        if isinstance(train_loss, float):
            proc_msg.append(f"Training loss {train_loss:.4f}")
        else:
            for loss_k, loss_v in train_loss.items():
                loss_name = loss_k.split("_")[0].capitalize()
                proc_msg.append(f"{loss_name} loss {round(loss_v, 4)}")

        # Construct eval prf message
        if val_loss is not None:
            proc_msg.append(f"Validation loss {val_loss:.4f}")
        if val_result is not None:
            for metric, score in val_result.items():
                proc_msg.append(f"{metric.upper()} {round(score, 4)}")

        proc_msg = " | ".join(proc_msg)
        self.logger.info(proc_msg)

        if self.use_wandb:
            # Process loss dict and log
            log_dict = train_loss if isinstance(train_loss, dict) else {"train_loss": train_loss}
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
            if val_result is not None:
                for metric, score in val_result.items():
                    log_dict[metric] = score

            if proc_id is not None:
                log_dict = {f"{k}_{proc_id}": v for k, v in log_dict.items()}

            wandb.log(log_dict)

    def _run_final_eval(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tensor]]:
        """Run final evaluation process with designated model checkpoint.

        Returns:
            final_prf_report: performance report of final evaluation
            y_preds: prediction on different datasets
        """
        # Load the best model checkpoint
        self.model = self.model_ckpt.load_best_ckpt(self.model, self.device)

        # Reconstruct dataloaders
        self._disable_shuffle()
        val_loader = self.eval_loader

        final_prf_report, y_preds = {}, {}
        for data_split, dataloader in {
            "train": self.train_loader,
            "val": val_loader,
        }.items():
            self.eval_loader = dataloader
            _, eval_result, y_pred = self._eval_epoch(return_output=True)
            final_prf_report[data_split] = eval_result
            y_preds[data_split] = y_pred

        return final_prf_report, y_preds

    def _disable_shuffle(self) -> None:
        """Disable shuffle in train dataloader for final evaluation."""
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,  # Reset shuffle to False
            num_workers=self.train_loader.num_workers,
            collate_fn=self.train_loader.collate_fn,
        )

    def _log_best_prf(self, prf_report: Dict[str, Any]) -> None:
        """Log performance evaluated with the best model checkpoint.

        Args:
            prf_report: performance report
        """
        self.logger.info(">>>>> Performance Report - Best Ckpt <<<<<")
        self.logger.info(json.dumps(prf_report, indent=4))

        if self.use_wandb:
            wandb.log(prf_report)
