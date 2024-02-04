"""
Main script for running local CV for ML models.
Author: JiaWei Jiang

* [ ] Use `instantiate` to build objects (e.g., cv, model).
"""
import warnings
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import optuna
from omegaconf.dictconfig import DictConfig

from cv.build import build_cv
from cv.validation import cross_validate
from data.data_processor import DataProcessor
from experiment.experiment import Experiment
from modeling.build import build_ml_models

optuna.logging.enable_propagation()
warnings.simplefilter("ignore")


def _objective(  # type: ignore
    trial, exp, X, y, cv, fit_params, groups, tgt_type, downsamp_quasi0, model_type
) -> float:
    xgb_params = {
        # "n_estimators": trial.suggest_int("n_estimators", 1000, 3500),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        # "max_depth": trial.suggest_int("max_depth", 4, 18),
        "max_depth": trial.suggest_int("max_depth", 4, 15),
        "learning_rate": trial.suggest_uniform("learning_rate", 0.005, 0.1),
        # "subsample": trial.suggest_uniform("subsample", 0.5, 0.9),
        "subsample": trial.suggest_uniform("subsample", 0.5, 0.85),
        # "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 0.9),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 0.8),
        "min_child_weight": trial.suggest_int("min_child_weight", 32, 128),
        "reg_lambda": trial.suggest_uniform("reg_lambda", 1, 10),
        "reg_alpha": trial.suggest_uniform("reg_alpha", 0, 10),
        # Fixed
        # "early_stopping_rounds": 50,
        "eval_metric": "mae",
        "tree_method": "gpu_hist",
        "seed": 42,
    }
    models = build_ml_models("xgb", 1, **xgb_params)

    # Run CV
    cv_result = cross_validate(
        exp=exp,
        X=X,
        y=y,
        models=models,
        cv=cv,
        fit_params=fit_params,
        groups=groups,
        tgt_type=tgt_type,
        downsamp_quasi0=downsamp_quasi0,
        one_fold_only=False,
        tune=True,
    )
    # ===
    # Start from simple mean
    prf = np.mean(cv_result.oof_prfs)
    # Weighted-sum by abs value
    # `p_raw`: 65.6, 5.5, 19.6
    # `p_dcap`: 53.1, 5.5, 18.0
    # `cc_raw`: 58.9, 59.9, 52.1
    # `cc_dcap`: 20.7, 22.2, 12.3
    # `cb_raw`: 59.1, 60.4, 51.0
    # prfs = cv_result.oof_prfs
    # if model_type == "p_raw":
    #     prf = prfs[0] / 65.6 + prfs[1] / 5.5 + prfs[2] / 19.6
    # elif model_type == "p_dcap":
    #     prf = prfs[0] / 53.1 + prfs[1] / 5.5 + prfs[2] / 18
    # elif model_type == "cc_raw":
    #     prf = prfs[0] / 58.9 + prfs[1] / 59.9 + prfs[2] / 52.1
    # elif model_type == "cc_dcap":
    #     prf = prfs[0] / 20.7 + prfs[1] / 22.2 + prfs[2] / 12.3
    # elif model_type == "cb_raw":
    #     prf = prfs[0] / 59.1 + prfs[1] / 60.4 + prfs[2] / 51
    # ===

    return prf


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

        # Build cross-validator
        cv = build_cv(**{"scheme": "tscv", **exp.data_cfg["cv"]})
        if "groups" in exp.data_cfg["cv"]:
            groups = data[exp.data_cfg["cv"]["groups"]]
        else:
            groups = None

        tgt_type = exp.data_cfg["tgt_types"][0]
        fit_params = exp.model_cfg["fit_params"]

        # Run optuna study
        opt_fn = partial(
            _objective,
            exp=exp,
            X=X,
            y=y,
            cv=cv,
            fit_params=fit_params,
            groups=groups,
            tgt_type=tgt_type,
            downsamp_quasi0=exp.data_cfg["dp"]["downsamp_quasi0"],
            model_type=exp.data_cfg["model_type"],
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(opt_fn, n_trials=40)


if __name__ == "__main__":
    # Launch main function
    main()
