"""
Main script for running local CV for ML models.
Author: JiaWei Jiang

* [ ] Use `instantiate` to build objects (e.g., cv, model).
"""
import pickle
import warnings
from pathlib import Path

import hydra
import pandas as pd
from omegaconf.dictconfig import DictConfig

from cv.build import build_cv
from cv.validation import cross_validate
from data.data_processor import DataProcessor
from experiment.experiment import Experiment
from modeling.build import build_ml_models

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
                fit_params=exp.model_cfg["fit_params"],
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


if __name__ == "__main__":
    # Launch main function
    main()
