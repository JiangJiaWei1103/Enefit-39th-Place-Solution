"""
Cross validation core logic.
Author: JiaWei Jiang

This file contains the core logic of running cross validation.
"""
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple

import lightgbm
import numpy as np
import pandas as pd

# from category_encoders.utils import convert_input, convert_input_vector
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import BaseCrossValidator

import wandb
from experiment.experiment import Experiment
from utils.common import dictconfig2dict
from utils.traits import is_gbdt_instance  # type: ignore

CVResult = namedtuple("CVResult", ["oof_pred", "oof_prfs", "feat_imps"])


def cross_validate(
    exp: Experiment,
    X: pd.DataFrame,
    y: pd.Series,
    models: List[BaseEstimator],
    cv: BaseCrossValidator,
    fit_params: Dict[str, Any],
    eval_fn: Optional[Callable] = None,
    imp_type: str = "gain",
    stratified: Optional[str] = None,
    groups: Optional[pd.Series] = None,
    **kwargs: Any,
) -> CVResult:
    """Run cross validation and return evaluated performance and
    predicting results.

    Parameters:
        exp: experiment tracker
        X: feature matrix
        y: target
        models: list of model instances to train
        cv: cross validator
        fit_params: parameters passed to `fit()` of the estimator
        eval_fn: evaluation function used to derive performance score
        imp_type: how the feature importance is calculated
        stratified: column acting as stratified determinant, used to
            preserve the percentage of samples for each class
        group: group labels

    Return:
        cv_result: output of cross validatin process
    """

    def _predict(model: BaseEstimator, x: pd.DataFrame) -> np.ndarray:
        """Do inference with the well-trained estimator.

        Parameters:
            model: well-trained estimator used to do inference
            x: data to predict

        Return:
            y_pred: prediction
        """
        y_pred = model.predict(x)

        return y_pred

    # Split modeling matters
    tgt_type = kwargs["tgt_type"]

    # Define CV output objects
    tscv = True if cv.__class__.__name__ == "TSCV" else False
    if tscv:
        oof = np.zeros((len(X), cv.get_n_splits()))
    else:
        oof = np.zeros((len(X), 1))
    prfs: List[float] = []
    feat_imps: List[pd.DataFrame] = []

    if groups is not None:
        cv_iter = cv.split(X=X, groups=groups)
    else:
        cv_iter = cv.split(X=X)
    for fold, (tr_idx, val_idx) in enumerate(cv_iter):
        # Configure wandb run
        if exp.cfg.use_wandb:
            fold_run = exp.add_wnb_run(cfg=exp.cfg, job_type=f"train_eval_{tgt_type}", name=f"fold{fold}")

        exp.log(f"===== Fold{fold} =====")
        if tscv:
            exp.log(f"## Training Months ## {groups[tr_idx].min()} ~ {groups[tr_idx].max()}")
            exp.log(f"## Validation Months ## {groups[val_idx].min()} ~ {groups[val_idx].max()}")
        X_tr, X_val = X.iloc[tr_idx, :], X.iloc[val_idx, :]
        y_tr, y_val = y.iloc[tr_idx, 0], y.iloc[val_idx, 0]

        # Further split by target types
        X_tr, y_tr = _split_data_by_tgt_type(X_tr, y_tr, tgt_type)
        X_val, y_val = _split_data_by_tgt_type(X_val, y_val, tgt_type)

        if fit_params is None:
            fit_params_fold = {}
        else:
            fit_params_fold = dictconfig2dict(fit_params).copy()
        if is_gbdt_instance(models[fold], ["xgb", "lgbm", "cat"]):
            fit_params_fold["eval_set"] = [(X_tr, y_tr), (X_val, y_val)]
            if is_gbdt_instance(models[fold], "lgbm"):
                fit_params_fold["callbacks"] = [
                    lightgbm.early_stopping(50),
                    lightgbm.log_evaluation(200),
                ]
            # Add categorical features...
            # ...
        models[fold].fit(X_tr, y_tr, **fit_params_fold)

        # Evaluate on oof
        val_idx = X_val.index
        oof_pred = _predict(models[fold], X_val)
        # ===
        # oof_pred = oof_pred * y.iloc[val_idx, 1]  # Inverse transform prediction
        oof_pred = np.clip(oof_pred, 0, np.inf)  # Clip prediction
        # ===
        if tscv:
            oof[val_idx, fold] = oof_pred
        else:
            oof[val_idx] = oof_pred
        # prf = mae(y_val * y.iloc[val_idx, 1], oof_pred)
        prf = mae(y_val, oof_pred)
        prfs.append(prf)
        exp.log(f"-> MAE: {prf}")

        # Store feature importance
        if is_gbdt_instance(models[fold], ["xgb", "lgbm", "cat"]):
            feat_imp = _get_feat_imp(models[fold], list(X_tr.columns), imp_type)
            feat_imp["fold"] = fold
            feat_imp["is_cons"] = True if tgt_type == "cons" else False
            feat_imps.append(feat_imp)

        if exp.cfg.use_wandb:
            wandb.log({"val_mae": prf})
            fold_run.finish()

    cv_result = CVResult(oof, prfs, feat_imps)

    return cv_result


def _split_data_by_tgt_type(X: pd.DataFrame, y: pd.Series, tgt_type: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split datasetes by target type."""
    if tgt_type == "prod":
        tgt_mask = X["is_consumption"] == 0
    else:
        tgt_mask = X["is_consumption"] == 1

    X_ = X[tgt_mask].drop("is_consumption", axis=1)
    y_ = y[X_.index]
    assert y.notna().all()

    return X_, y_


def _get_feat_imp(model: BaseEstimator, feat_names: List[str], imp_type: str) -> pd.DataFrame:
    """Generate and return feature importance DataFrame.

    Parameters:
        model: well-trained estimator
        feat_names: list of feature names
        imp_type: how the feature importance is calculated

    Return:
        feat_imp: feature importance
    """
    feat_imp = pd.DataFrame(feat_names, columns=["feature"])

    if is_gbdt_instance(model, "lgbm"):
        feat_imp[f"importance_{imp_type}"] = model.booster_.feature_importance(imp_type)
    elif is_gbdt_instance(model, ("xgb", "cat")):
        feat_imp[f"importance_{imp_type}"] = model.feature_importances_

    return feat_imp
