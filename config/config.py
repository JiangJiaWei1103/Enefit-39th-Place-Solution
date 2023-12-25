"""
Experiment configuration logic.
Author: JiaWei Jiang

This file includes utility functions for setting up experiments.
"""
import os
import random
import string
from typing import List

import numpy as np
import torch


def gen_exp_id(model_name: str) -> str:
    """Generate unique experiment identifier.

    Args:
        model_name: name of model architecture

    Returns:
        exp_id: experiment identifier
    """
    chars = string.ascii_lowercase + string.digits
    exp_id = "".join(random.SystemRandom().choice(chars) for _ in range(8))
    exp_id = f"{model_name}-{exp_id}"

    return exp_id


def get_seeds(n_seeds: int = 3) -> List[int]:
    """Generate and return a list of random seeds

    Args:
        n_seeds: number of seeds

    Returns:
        seeds: a list of random seeds
    """
    seeds = [random.randint(1, 2**32 - 1) for _ in range(n_seeds)]

    return seeds


def seed_everything(seed: int) -> None:
    """Seed current experiment to guarantee reproducibility.

    Args:
        seed: manually specified seed number
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running with cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


# def setup_dp() -> Dict[str, Any]:
#     """Return hyperparameters controlling data processing.

#     Returns:
#         dp_cfg: hyperparameters of data processor
#     """
#     cfg_path = os.path.join(CONFIG_PATH, "dp.yaml")
#     with open(cfg_path, "r") as f:
#         dp_cfg = yaml.full_load(f)

#     return dp_cfg


# def setup_model(model_name: str) -> Dict[str, Any]:
#     """Return hyperparameters of the specified model.

#     Args:
#         model_name: name of model architecture

#     Returns:
#         model_cfg: hyperparameters of the specified model
#     """
#     cfg_path = os.path.join(CONFIG_PATH, f"model/{model_name}.yaml")
#     with open(cfg_path, "r") as f:
#         model_cfg = yaml.full_load(f)

#     return model_cfg


# def setup_proc(seed: Optional[int] = None) -> Dict[str, Any]:
#     """Return hyperparameters for training and evaluation processes,
#     and clear local output buffer to dump outputs.

#     Args:
#         seed: random seed for training and evaluation processes
#             *Note: For processes with any CV scheme, seed is fixed;
#                 hence, there's no need to specify.

#     Returns:
#         proc_cfg: hyperparameters for training and evaluation processes
#     """
#     # Load in config file for training and evaluation processes
#     cfg_path = os.path.join(CONFIG_PATH, "defaults.yaml")
#     with open(cfg_path, "r") as f:
#         proc_cfg = yaml.full_load(f)

#     # Seed experiment
#     if seed is not None:
#         # Dynamically adjust random seed (e.g., train on whole dataset
#         # with different random seeds for blending)
#         proc_cfg["seed"] = seed
#     seed_everything(proc_cfg["seed"])

#     return proc_cfg
