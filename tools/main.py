"""
Main script for training and evaluation processes for DL.
Author: JiaWei Jiang
"""
import gc
import math
import warnings
from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig

import wandb
from base.base_trainer import BaseTrainer
from config.config import get_seeds, seed_everything
from criterion.build import build_criterion
from cv.build import build_cv
from data.build import build_dataloader
from data.data_processor import DataProcessor
from evaluating.build import build_evaluator
from experiment.experiment import Experiment
from modeling.build import build_model
from solver.build import build_lr_scheduler, build_optimizer
from trainer.trainer import MainTrainer
from utils.common import count_params

warnings.simplefilter("ignore")


@hydra.main(config_path="../config", config_name="main_dl")
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
        # ===
        # Remember to process the scale of target
        # ===
        dp = DataProcessor(Path(exp.data_cfg["data_path"]), **exp.data_cfg["dp"])
        dp.run_before_splitting()
        data = dp.get_data_cv()

        # Build cross-validator
        cv = build_cv(**{"scheme": "tscv", **exp.data_cfg["cv"]})
        if "groups" in exp.data_cfg["cv"]:
            groups = data[exp.data_cfg["cv"]["groups"]]
        else:
            groups = None

        # Run cross-validation
        one_fold_only = exp.cfg["one_fold_only"]
        manual_seed = exp.cfg["seed"]
        for s_i, seed in enumerate(get_seeds(exp.cfg["n_seeds"])):
            seed = seed if manual_seed is None else manual_seed
            exp.log(f"\nSeed the experiment with {seed}...")
            seed_everything(seed)
            cfg_seed = exp.cfg.copy()
            cfg_seed["seed"] = seed

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X=data, groups=groups)):
                # Configure sub-entry for tracking current fold
                seed_name, fold_name = f"seed{s_i}", f"fold{fold}"
                proc_id = f"{seed_name}_{fold_name}"
                if exp.cfg["use_wandb"]:
                    tr_eval_run = exp.add_wnb_run(
                        cfg=cfg_seed,
                        job_type=fold_name if one_fold_only else seed_name,
                        name=seed_name if one_fold_only else fold_name,
                    )
                exp.log(f"== Train and Eval Process - Fold{fold} ==")

                # Build dataloaders
                data_tr, data_val = data.iloc[tr_idx].reset_index(drop=True), data.iloc[val_idx].reset_index(drop=True)
                # data_tr, data_val, scaler = dp.run_after_splitting(data_tr, data_val)
                train_loader = build_dataloader(
                    data_tr, "train", exp.data_cfg["dataset"], **exp.trainer_cfg["dataloader"]
                )
                val_loader = build_dataloader(
                    data_val, "val", exp.data_cfg["dataset"], **exp.trainer_cfg["dataloader"]
                )
                # y_val = val_loader.dataset.data_chunks["y"]
                # import pickle
                # with open("./y_val.pkl", "wb") as f:
                #     pickle.dump(y_val, f)
                # import sys; sys.exit(1)

                # Build model
                model = build_model(exp.model_name, **exp.model_cfg["model_params"])
                model.to(exp.trainer_cfg["device"])
                if exp.cfg["use_wandb"]:
                    wandb.log({"model": {"n_params": count_params(model)}})
                    wandb.watch(model, log="all", log_graph=True)

                # Build criterion
                loss_fn = build_criterion(**exp.trainer_cfg["loss_fn"])

                # Build solvers
                optimizer = build_optimizer(model, **exp.trainer_cfg["optimizer"])
                num_training_steps = (
                    math.ceil(
                        len(train_loader.dataset)
                        / exp.trainer_cfg["dataloader"]["batch_size"]
                        / exp.trainer_cfg["grad_accum_steps"]
                    )
                    * exp.trainer_cfg["epochs"]
                )
                lr_skd = build_lr_scheduler(optimizer, num_training_steps, **exp.trainer_cfg["lr_skd"])

                # Build evaluator
                evaluator = build_evaluator(**exp.trainer_cfg["evaluator"])

                # Build trainer
                trainer: BaseTrainer = None
                trainer = MainTrainer(
                    logger=exp.logger,
                    trainer_cfg=exp.trainer_cfg,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    lr_skd=lr_skd,
                    ckpt_path=exp.ckpt_path,
                    evaluator=evaluator,
                    scaler=None,
                    train_loader=train_loader,
                    eval_loader=val_loader,
                    use_wandb=exp.cfg["use_wandb"],
                )

                # Run main training and evaluation for one fold
                trainer.train_eval(fold)

                # Run evaluation on unseen test set
                if False:
                    data_test = dp.get_data_test()
                    test_loader = build_dataloader(
                        data_test, "test", exp.data_cfg["dataset"], **exp.trainer_cfg["dataloader"]
                    )
                    _ = trainer.test(fold, test_loader)

                # Dump output objects
                # if scaler is not None:
                #     exp.dump_trafo(scaler, f"scaler_{proc_id}")
                for model_path in exp.ckpt_path.glob("*.pth"):
                    if "seed" in str(model_path) or "fold" in str(model_path):
                        continue

                    # Rename model file
                    model_file_name_dst = f"{model_path.stem}_{proc_id}.pth"
                    model_path_dst = exp.ckpt_path / model_file_name_dst
                    model_path.rename(model_path_dst)

                # Free mem.
                del (data_tr, data_val, train_loader, val_loader, model, optimizer, lr_skd, evaluator, trainer)
                _ = gc.collect()

                if exp.cfg["use_wandb"]:
                    tr_eval_run.finish()
                if one_fold_only:
                    exp.log("Cross-validatoin stops at first fold!!!")
                    break

        if exp.cfg["full_train"]:
            exp.log("\nStart full-training process on the whole dataset...")
            for s_i, seed in enumerate(get_seeds(3)):
                exp.log(f"\nSeed the experiment with {seed}...")
                seed_everything(seed)
                cfg_seed = exp.cfg.copy()
                cfg_seed["seed"] = seed

                # Configure sub-entry for tracking current fold
                if exp.cfg["use_wandb"]:
                    ft_run = exp.add_wnb_run(cfg=cfg_seed, job_type="full_train", name=f"seed{s_i}")
                exp.log(f"\n== Full-Training Process - Seed{s_i} ==")

                # Build dataloaders
                train_loader = build_dataloader(
                    data, "train", exp.data_cfg["dataset"], **exp.trainer_cfg["dataloader"]
                )
                val_loader = None

                # Build model
                model = build_model(exp.model_name, **exp.model_cfg["model_params"])
                model.to(exp.trainer_cfg["device"])
                if exp.cfg["use_wandb"]:
                    wandb.log({"model": {"n_params": count_params(model)}})
                    wandb.watch(model, log="all", log_graph=True)

                # Build criterion
                loss_fn = build_criterion(**exp.trainer_cfg["loss_fn"])

                # Build solvers
                optimizer = build_optimizer(model, **exp.trainer_cfg["optimizer"])
                num_training_steps = (
                    math.ceil(
                        len(train_loader.dataset)
                        / exp.trainer_cfg["dataloader"]["batch_size"]
                        / exp.trainer_cfg["grad_accum_steps"]
                    )
                    * exp.trainer_cfg["epochs"]
                )
                lr_skd = build_lr_scheduler(optimizer, num_training_steps, **exp.trainer_cfg["lr_skd"])

                # Build evaluator
                evaluator = build_evaluator(**exp.trainer_cfg["evaluator"])

                # Build trainer
                trainer = MainTrainer(
                    logger=exp.logger,
                    trainer_cfg=exp.trainer_cfg,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    lr_skd=lr_skd,
                    ckpt_path=exp.ckpt_path,
                    evaluator=evaluator,
                    scaler=None,
                    train_loader=train_loader,
                    eval_loader=val_loader,
                    use_wandb=exp.cfg["use_wandb"],
                )

                # Run full-training process for one seed
                trainer.train_only()

                # Dump output objects
                for model_path in exp.ckpt_path.glob("*.pth"):
                    if "seed" in str(model_path) or "fold" in str(model_path):
                        continue

                    # Rename model file
                    model_file_name_dst = f"{model_path.stem}_ft_seed{s_i}.pth"
                    model_path_dst = exp.ckpt_path / model_file_name_dst
                    model_path.rename(model_path_dst)

                # Free mem.
                del (train_loader, val_loader, model, optimizer, lr_skd, evaluator, trainer)
                _ = gc.collect()

                if exp.cfg["use_wandb"]:
                    ft_run.finish()


if __name__ == "__main__":
    # Launch main function
    main()
