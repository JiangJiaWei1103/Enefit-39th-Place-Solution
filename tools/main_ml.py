"""
Main script for running local CV for ML models.
Author: JiaWei Jiang

* [ ] Use `instantiate` to build objects (e.g., cv, model).
"""
import pickle
import warnings
from pathlib import Path
from typing import List

import hydra
import pandas as pd
from omegaconf.dictconfig import DictConfig
from sklearn.metrics import mean_absolute_error as mae

from cv.build import build_cv
from cv.validation import cross_validate
from data.data_processor import DataProcessor
from experiment.experiment import Experiment
from metadata import TGT_COL
from modeling.build import build_ml_models

warnings.simplefilter("ignore")


def _load_feats(feats_path: Path) -> List[str]:
    feats = []
    with open(feats_path, "r") as f:
        for line in f.readlines():
            feats.append(line.strip())

    return feats


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
        if exp.data_cfg["proc_data_ver"] is not None:
            dp = None
            proc_data_path = Path(exp.cfg["paths"]["PROC_DATA_PATH"])
            # ===
            # 1. Data only one version, keeps growing
            # 2. Version control on feature set
            # 3. Target also needs specify
            data = pd.read_parquet(proc_data_path / "data.parquet")
            data = data[data[TGT_COL].notna()].reset_index(drop=True)
            feats = _load_feats(proc_data_path / "feats" / f"v{exp.data_cfg['proc_data_ver']}.txt")
            X, y = data[feats], data[TGT_COL]
            # ===
        else:
            dp = DataProcessor(**exp.data_cfg["dp"])
            dp.run_before_splitting()
            data = dp.get_data_cv()

        # Run cross-validation
        cv = build_cv(**{"scheme": "gpkf", **exp.data_cfg["cv"]})
        if "groups" in exp.data_cfg["cv"]:
            groups = data[exp.data_cfg["cv"]["groups"]]

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
        exp.set_cv_score(mae(data[TGT_COL], oof_pred))


if __name__ == "__main__":
    # Launch main function
    main()
