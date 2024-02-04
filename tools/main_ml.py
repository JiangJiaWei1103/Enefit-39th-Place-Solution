"""
Main script for running local CV for ML models.
Author: JiaWei Jiang

* [ ] Use `instantiate` to build objects (e.g., cv, model).
"""
import gc
import pickle
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from sklearn.base import BaseEstimator

from cv.build import build_cv
from cv.validation import _downsample_quasi0_prod, cross_validate
from data.data_processor import DataProcessor
from experiment.experiment import Experiment
from modeling.build import build_ml_models
from utils.common import dictconfig2dict
from utils.traits import is_gbdt_instance  # type: ignore

warnings.simplefilter("ignore")


@hydra.main(config_path="../config", config_name="main_ml")
def main(cfg: DictConfig) -> None:
    """Run training and evaluation processes.

    Args:
        cfg: configuration driving training and evaluation processes
    """
    # Configure experiment
    experiment = Experiment(cfg)

    with experiment as exp:
        exp.dump_cfg(exp.cfg, "main")

        # Prepare data
        dp = DataProcessor(Path(exp.data_cfg["data_path"]), **exp.data_cfg["dp"])
        dp.run_before_splitting()
        data = dp.get_data_cv()
        # ===
        # X_cols = dp.feats + ["is_consumption", "seg"]
        X_cols = dp.feats + ["is_consumption"]
        # ===
        X, y = data[X_cols], data[dp.tgt_cols]
        exp.log(f"Data shape | X {X.shape}, y {y.shape}")
        exp.log(f"-> #Samples {len(X)}")
        exp.log(f"-> #Features {X.shape[1] - 1}")

        # Build cross-validator
        cv = build_cv(**{"scheme": "tscv", **exp.data_cfg["cv"]})
        if "groups" in exp.data_cfg["cv"]:
            groups = data[exp.data_cfg["cv"]["groups"]]
        else:
            groups = None

        # Build models
        tgt_types = exp.data_cfg["tgt_types"]
        n_models = exp.data_cfg["cv"]["n_folds"] if not exp.cfg["one_fold_only"] else 1
        model_params = exp.model_cfg["model_params"]
        fit_params = exp.model_cfg["fit_params"]
        models = {tgt_type: build_ml_models(exp.model_name, n_models, **model_params) for tgt_type in tgt_types}

        # Run cross-validation
        cv_result = {}
        for tgt_type in tgt_types:
            exp.log(f"Run cross-validation for {tgt_type} models...")
            cv_result[tgt_type] = cross_validate(
                exp=exp,
                X=X,
                y=y,
                models=models[tgt_type],
                cv=cv,
                fit_params=fit_params,
                groups=groups,
                tgt_type=tgt_type,
                downsamp_quasi0=exp.data_cfg["dp"]["downsamp_quasi0"],
                one_fold_only=exp.cfg["one_fold_only"],
                model_type=exp.cfg["model_type"],
            )

        # Dump CV output objects
        if exp.cfg["dump_fold_models"]:
            with open(exp.exp_dump_path / "models/models.pkl", "wb") as f:
                pickle.dump(models, f)
        oof_pred = np.zeros(cv_result[tgt_types[0]].oof_pred.shape)
        for tgt_type in tgt_types:
            oof_pred = oof_pred + cv_result[tgt_type].oof_pred
        exp.dump_ndarr(oof_pred, "oof")
        prfs = {tgt_type: cv_result[tgt_type].oof_prfs for tgt_type in tgt_types}
        exp.log_prfs(prfs)
        feat_imps = pd.concat(
            list(chain(*[cv_result[tgt_type].feat_imps for tgt_type in tgt_types])), ignore_index=True
        )
        exp.dump_df(feat_imps, file_name="feat_imps.parquet")

        # Run optional full-train
        if cfg["full_train"]:

            def _get_refit_iter(models: List[BaseEstimator]) -> int:
                best_iters = [model.best_iteration for model in models]
                best_iter = np.median(best_iters)
                # ===
                # Adjust w/ tr / val sizes
                # https://www.kaggle.com/code/iglovikov/xgb-1114/comments#141375
                best_iter = int(best_iter / 0.8)
                # ===
                return best_iter

            # ===
            # if roll:
            # Use the latest 1-year as training data
            # Submit this prod with cc_raw & cb_raw_base
            # Act as the base score to verify if retrain at start work
            # Submit another only retrain at start with exactly the same setting as base
            # hope the score is the same or close
            # Submit another monthly retraining only on cc and cb
            if exp.data_cfg["cv"]["max_train_size"] == 12:
                DT_TAIL = datetime(2022, 6, 1, 0)
                data_full = data[data["datetime"] >= DT_TAIL].reset_index(drop=True)
                X, y = data_full[dp.feats + ["is_consumption"]], data_full[dp.tgt_cols]
                exp.log(f"## Full-Training Months ## {data_full['datetime'].min()} ~ {data_full['datetime'].max()}")
            else:
                exp.log(f"## Full-Training Months ## {data['datetime'].min()} ~ {data['datetime'].max()}")
            # ===

            best_iter = {tgt_type: _get_refit_iter(models[tgt_type]) for tgt_type in tgt_types}
            del models
            gc.collect()
            models = defaultdict(list)
            for tgt_type in tgt_types:
                model_params_ft = model_params.copy()
                # model_params_ft["n_estimators"] = best_iter[tgt_type]
                model_params_ft["n_estimators"] = model_params["n_estimators"]
                for seed in range(1):
                    model_params_ft["seed"] = seed
                    models[tgt_type].append(build_ml_models(exp.model_name, 1, **model_params_ft)[0])

            for tgt_type in tgt_types:
                # exp.log(f"Run full-training for {tgt_type} models with best iter {best_iter[tgt_type]}...")
                exp.log(f"Run full-training for {tgt_type} models with tuned iter {model_params['n_estimators']}...")

                if tgt_type == "prod":
                    tgt_mask = X["is_consumption"] == 0
                elif tgt_type == "cons":
                    tgt_mask = X["is_consumption"] == 1
                elif tgt_type == "cons_c":
                    tgt_mask = (X["is_consumption"] == 1) & (X["is_business"] == 0)
                elif tgt_type == "cons_b":
                    # Business consumption
                    tgt_mask = (X["is_consumption"] == 1) & (X["is_business"] == 1)
                cols_to_drop = ["is_consumption"]
                if tgt_type in ["cons_c", "cons_b"]:
                    cols_to_drop.append("is_business")
                X_tr = X[tgt_mask].drop(cols_to_drop, axis=1)
                y_tr = y.iloc[:, 0][X_tr.index]
                # ===
                if tgt_type.startswith("prod") and exp.data_cfg["dp"]["downsamp_quasi0"]["ratio"] != 1:
                    exp.log(">> Downsample quasi-zero production values...")
                    X_tr, y_tr = _downsample_quasi0_prod(X_tr, y_tr, **exp.data_cfg["dp"]["downsamp_quasi0"])
                # ===

                for seed in range(1):
                    if cfg["use_wandb"]:
                        seed_run = exp.add_wnb_run(
                            job_type=f"ft_{tgt_type}",
                            name=f"seed{seed}",
                        )

                    exp.log(f"===== Seed{seed} =====")
                    if fit_params is None:
                        fit_params_fold = {}
                    else:
                        fit_params_fold = dictconfig2dict(fit_params).copy()
                    if is_gbdt_instance(models[tgt_type][seed], ["xgb", "lgbm", "cat"]):
                        fit_params_fold["eval_set"] = [(X_tr, y_tr)]
                        # Add categorical features...
                        # ...
                    models[tgt_type][seed].fit(X_tr, y_tr, **fit_params_fold)

                    if cfg["use_wandb"]:
                        seed_run.finish()

            # Dump models
            with open(exp.exp_dump_path / "models/models_ft.pkl", "wb") as f:
                pickle.dump(models, f)


if __name__ == "__main__":
    # Launch main function
    main()
