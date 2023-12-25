"""
Evaluator building logic.
Author: JiaWei Jiang

This file contains the basic logic of building evaluator for evaluation
process.
"""
from typing import Any

from .evaluator import Evaluator


def build_evaluator(**evaluator_cfg: Any) -> Evaluator:
    """Build and return the evaluator.

    Args:
        evaluator_cfg: hyperparameters of the evaluator

    Returns:
        evaluator: evaluator
    """
    eval_metrics = evaluator_cfg["eval_metrics"]
    evaluator = Evaluator(eval_metrics)

    return evaluator
