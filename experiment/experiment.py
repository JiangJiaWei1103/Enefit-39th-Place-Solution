"""
Experiment tracker.
Author: JiaWei Jiang

This file contains the definition of experiment tracker for experiment
configuration, message logging, object dumping, etc.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from sklearn.base import BaseEstimator
from torch.nn import Module
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

import wandb
from utils.common import dictconfig2dict
from utils.logger import Logger


class Experiment(object):
    """Experiment tracker.

    Args:
        cfg: configuration driving the designated process
        log_file: file to log experiment process
        infer: if True, the experiment is for inference only

    Attributes:
        model_name: model name
    """

    exp_dump_path: Path
    ckpt_path: Path  # Path to save model checkpoints
    _cv_score: float = 0

    def __init__(
        self,
        cfg: DictConfig,
        log_file: str = "train_eval.log",
        infer: bool = False,
    ) -> None:
        # Setup experiment identifier
        if cfg.exp_id is None:
            cfg.exp_id = datetime.now().strftime("%m%d-%H_%M_%S")
        self.exp_id = cfg.exp_id

        self.cfg = cfg
        self.log_file = log_file
        self.infer = infer

        # Make buffer to dump output objects
        self.DUMP_PATH = Path(cfg["paths"]["DUMP_PATH"])
        self._mkbuf()

        # Configure the experiment
        if infer:
            self._evoke_cfg()
        else:
            self.data_cfg = cfg.data
            self.model_cfg = cfg.model
            self._set_model_name()
            if "trainer" in cfg:
                self.trainer_cfg = cfg.trainer

        # Setup experiment logger
        if cfg.use_wandb:
            assert cfg.project_name is not None, "Please specify project name of wandb."
            self.exp_supr = self.add_wnb_run(cfg=cfg, job_type="supervise", name="supr")
        self.logger = Logger(logging_file=self.exp_dump_path / log_file).get_logger()

    def _mkbuf(self) -> None:
        """Make local buffer to dump experiment output objects."""
        self.DUMP_PATH.mkdir(parents=True, exist_ok=True)
        self.exp_dump_path = self.DUMP_PATH / self.exp_id
        self.ckpt_path = self.exp_dump_path / "models"

        if self.infer:
            assert self.exp_dump_path.exists(), "There exists no output objects for your specified experiment."
        else:
            self.exp_dump_path.mkdir(parents=True, exist_ok=False)
            for sub_dir in ["config", "trafos", "models", "preds", "feats", "imps"]:
                sub_path = self.exp_dump_path / sub_dir
                sub_path.mkdir(parents=True, exist_ok=False)
            for pred_type in ["oof", "holdout", "final"]:
                sub_path = self.exp_dump_path / "preds" / pred_type
                sub_path.mkdir(parents=True, exist_ok=False)

    def _set_model_name(self) -> None:
        self._model_name = HydraConfig.get().runtime.choices.model

    def _evoke_cfg(self) -> None:
        """Retrieve configuration of the pre-dumped experiment."""
        pass

    def __enter__(self) -> Experiment:
        self._log_cfg_to_wnb()
        if self.cfg.use_wandb:
            self.exp_supr.finish()

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_inst: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._halt()

    @property
    def model_name(self) -> str:
        return self._model_name

    def log(self, msg: str) -> None:
        """Log the provided message."""
        self.logger.info(msg)

    def log_prfs(self, prfs: Dict[str, Any]) -> None:
        """Log dict-like performance."""
        self.log("-" * 50)
        self.log(">>>>> Performance Report <<<<<")
        self.log(json.dumps(prfs, indent=4))

    def dump_cfg(self, cfg: Union[DictConfig, Dict[str, Any]], file_name: str) -> None:
        """Dump configuration under corresponding path.

        Args:
            cfg: configuration
            file_name: config name with .yaml extension
        """
        file_name = file_name if file_name.endswith(".yaml") else f"{file_name}.yaml"
        dump_path = self.exp_dump_path / "config" / file_name
        if isinstance(cfg, Dict):
            cfg = OmegaConf.create(cfg)
        OmegaConf.save(cfg, dump_path)

    def dump_ndarr(self, arr: np.ndarray, file_name: str) -> None:
        """Dump np.ndarray to corresponding path.

        Args:
            arr: array to dump
            file_name: array name with .npy extension
        """
        dump_path = self.exp_dump_path / "preds" / file_name
        np.save(dump_path, arr)

    def dump_df(self, df: pd.DataFrame, file_name: str) -> None:
        """Dump DataFrame (e.g., feature imp) to corresponding path.

        Args:
            df: DataFrame to dump
            file_name: df name with .csv (by default) extension
        """
        if "." not in file_name:
            file_name = f"{file_name}.csv"
        dump_path = self.exp_dump_path / file_name

        if file_name.endswith(".csv"):
            df.to_csv(dump_path, index=False)
        elif file_name.endswith(".parquet"):
            df.to_parquet(dump_path, index=False)
        elif file_name.endswith(".pkl"):
            df.to_pickle(dump_path)

    def dump_model(self, model: Union[BaseEstimator, Module], file_name: str) -> None:
        """Dump the best model checkpoint to corresponding path.

        Args:
            model: well-trained estimator/model
            file_name: estimator/model name with .pkl/.pth extension
        """
        if isinstance(model, BaseEstimator):
            file_name = f"{file_name}.pkl"
        elif isinstance(model, Module):
            file_name = f"{file_name}.pth"
        dump_path = self.exp_dump_path / "models" / file_name

        if isinstance(model, BaseEstimator):
            with open(dump_path, "wb") as f:
                pickle.dump(model, f)
        elif isinstance(model, Module):
            torch.save(model.state_dict(), dump_path)

    def dump_trafo(self, trafo: Any, file_name: str) -> None:
        """Dump data transfomer (e.g., scaler) to corresponding path.

        Args:
            trafo: fitted data transformer
            file_name: transformer name with .pkl extension
        """
        file_name = file_name if file_name.endswith(".pkl") else f"{file_name}.pkl"
        dump_path = self.exp_dump_path / "trafos" / file_name
        with open(dump_path, "wb") as f:
            pickle.dump(trafo, f)

    def set_cv_score(self, cv_score: float) -> None:
        """Set final CV score for recording.

        Args:
            cv_score: final CV score
        """
        self._cv_score = cv_score

    def add_wnb_run(
        self,
        cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
        job_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Union[Run, RunDisabled, None]:
        """Initialize an wandb run for experiment tracking.

        Args:
            cfg: experiment config
                *Note: current random seed is recorded.
            job_type: type of run
            name: name of run

        Returns:
            run: wandb run to track the current experiment
        """
        if cfg is not None and isinstance(cfg, DictConfig):
            cfg = dictconfig2dict(cfg)
        run = wandb.init(project=self.cfg.project_name, config=cfg, group=self.exp_id, job_type=job_type, name=name)

        return run

    def _log_cfg_to_wnb(self) -> None:
        """Log experiment config to wandb."""
        self.log(f"===== Experiment {self.exp_id} =====")
        self.log(f"-> CFG: {self.cfg}\n")

    def _halt(self) -> None:
        if self.cfg.use_wandb:
            dump_entry = self.add_wnb_run(None, job_type="dumping")

            # Log final CV score if exists
            if self._cv_score is not None:
                dump_entry.log({"cv_score": self._cv_score})

            # Push artifacts to remote
            artif = wandb.Artifact(name=self.model_name.upper(), type="output")
            artif.add_dir(self.exp_dump_path)
            dump_entry.log_artifact(artif)
            dump_entry.finish()

        self.log(f"===== End of Experiment {self.exp_id} =====")
