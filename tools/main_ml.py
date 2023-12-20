"""
Main script for running local CV for ML models.
Author: JiaWei Jiang

* [ ] Use `instantiate` to build objects (e.g., cv, model).
"""
import gc
import pickle
import warnings
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from sklearn.base import BaseEstimator

from cv.build import build_cv
from cv.validation import cross_validate
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
        X, y = data[dp.feats + ["is_consumption"]], data[dp.tgt_cols]
        exp.log(f"Data shape | X {X.shape}, y {y.shape}")
        exp.log(f"-> #Samples {len(X)}")
        exp.log(f"-> #Features {X.shape[1] - 1}")

        # Run cross-validation
        cv = build_cv(**{"scheme": "tscv", **exp.data_cfg["cv"]})
        if "groups" in exp.data_cfg["cv"]:
            groups = data[exp.data_cfg["cv"]["groups"]]
        else:
            groups = None

        # Build models
        n_models = exp.data_cfg["cv"]["n_folds"]
        model_params = exp.model_cfg["model_params"]
        fit_params = exp.model_cfg["fit_params"]
        models = {
            "prod": build_ml_models(exp.model_name, n_models, **model_params),
            "cons": build_ml_models(exp.model_name, n_models, **model_params),
        }

        # Run cross-validation
        cv_result = {}
        for tgt_type in ["prod", "cons"]:
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
            )

        # Dump CV output objects
        with open(exp.exp_dump_path / "models/models.pkl", "wb") as f:
            pickle.dump(models, f)
        oof_pred = cv_result["prod"].oof_pred + cv_result["cons"].oof_pred
        exp.dump_ndarr(oof_pred, "oof")
        prfs = {"prod": cv_result["prod"].oof_prfs, "cons": cv_result["cons"].oof_prfs}
        exp.log_prfs(prfs)
        feat_imps = pd.concat(cv_result["prod"].feat_imps + cv_result["cons"].feat_imps, ignore_index=True)
        exp.dump_df(feat_imps, file_name="feat_imps.parquet")

        # Run optional full-train
        if cfg["full_train"]:

            def _get_refit_iter(models: List[BaseEstimator]) -> int:
                best_iters = [model.best_iteration for model in models]
                best_iter = np.median(best_iters)
                # ===
                # Adjust w/ tr / val sizes
                best_iter = int(best_iter / 0.8)
                # ===
                return best_iter

            best_iter = {"prod": _get_refit_iter(models["prod"]), "cons": _get_refit_iter(models["cons"])}
            del models
            gc.collect()
            models = {"prod": [], "cons": []}
            for tgt_type in models.keys():
                model_params_ft = model_params.copy()
                model_params_ft["n_estimators"] = best_iter[tgt_type]
                for seed in range(3):
                    model_params_ft["seed"] = seed
                    models[tgt_type].append(build_ml_models(exp.model_name, 1, **model_params_ft)[0])

            for tgt_type in ["prod", "cons"]:
                exp.log(f"Run full-training for {tgt_type} models with best iter {best_iter[tgt_type]}...")

                if tgt_type == "prod":
                    tgt_mask = X["is_consumption"] == 0
                else:
                    tgt_mask = X["is_consumption"] == 1
                X_tr = X[tgt_mask].drop("is_consumption", axis=1)
                y_tr = y.iloc[:, 0][X_tr.index]

                for seed in range(3):
                    if cfg["use_wandb"]:
                        seed_run = exp.add_wnb_run(
                            # cfg={"model": {"model_params": model_params}},
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
