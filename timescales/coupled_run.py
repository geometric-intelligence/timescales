"""
Standalone training entry point for the Coupled RNN architecture.

Mirrors single_run.py but imports from coupled_rnn instead of
multitimescale_rnn.  No changes are made to any existing file.

Usage:
    python coupled_run.py --config flip_flop_coupled.yaml
"""

import argparse
import datetime
import os
import sys

import torch
import torch.nn as nn
import yaml
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from callbacks import LossLoggerCallback

from timescales.datamodules import (
    PathIntegrationDataModule,
    PathIntegration1DDataModule,
    HierarchicalCounterDataModule,
    FlipFlopDataModule,
    NullDataModule,
    TeacherStudentDataModule,
)
from timescales.rnns.coupled_rnn import CoupledRNN, CoupledRNNLightning

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))


# ============================================================================
# Datamodule factory (identical to single_run.py — duplicated to stay isolated)
# ============================================================================

def create_datamodule(config: dict):
    """Create datamodule based on task type in config."""
    task = config.get("task", "flip_flop")

    if task == "path_integration":
        datamodule = PathIntegrationDataModule(
            trajectory_type=config["trajectory_type"],
            velocity_representation=config["velocity_representation"],
            dt=config["dt"],
            num_time_steps=config["num_time_steps"],
            arena_size=config["arena_size"],
            num_place_cells=config["num_place_cells"],
            place_cell_rf=config["place_cell_rf"],
            DoG=config["DoG"],
            surround_scale=config["surround_scale"],
            place_cell_layout=config["place_cell_layout"],
            linear_speed_mean=config.get("linear_speed_mean"),
            linear_speed_std=config.get("linear_speed_std"),
            behavioral_timescale_mean=config.get("behavioral_timescale_mean"),
            behavioral_timescale_std=config.get("behavioral_timescale_std"),
            linear_speed_tau=config.get("linear_speed_tau", 1.0),
            angular_speed_mean=config.get("angular_speed_mean", 0.0),
            angular_speed_std=config.get("angular_speed_std", 1.0),
            angular_speed_tau=config.get("angular_speed_tau", 0.4),
            num_trajectories=config["num_trajectories"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            train_val_split=config["train_val_split"],
        )
        config["input_size"] = {"cartesian": 2, "polar": 2, "sincos_polar": 3}[
            config["velocity_representation"]
        ]
        config["output_size"] = config["num_place_cells"]

    elif task == "path_integration_1d":
        datamodule = PathIntegration1DDataModule(
            dt=config["dt"],
            num_time_steps=config["num_time_steps"],
            arena_size=config["arena_size"],
            num_place_cells=config["num_place_cells"],
            place_cell_rf=config["place_cell_rf"],
            DoG=config.get("DoG", False),
            surround_scale=config.get("surround_scale", 2.0),
            place_cell_layout=config.get("place_cell_layout", "uniform"),
            velocity_mean=config.get("velocity_mean", 0.0),
            velocity_std=config.get("velocity_std", 0.5),
            velocity_tau=config.get("velocity_tau", 1.0),
            num_trajectories=config["num_trajectories"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            train_val_split=config["train_val_split"],
        )
        config["input_size"] = 1
        config["output_size"] = config["num_place_cells"]

    elif task == "binary_counter":
        datamodule = HierarchicalCounterDataModule(
            n_levels=config["n_levels"],
            base_flip_prob=config["base_flip_prob"],
            noise_std=config.get("noise_std", 0.1),
            num_time_steps=config["num_time_steps"],
            num_trajectories=config["num_trajectories"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            train_val_split=config["train_val_split"],
            observe_all_levels=config.get("observe_all_levels", False),
            input_encoding=config.get("input_encoding", "noisy_binary"),
        )
        config["input_size"] = datamodule.input_size
        config["output_size"] = datamodule.output_size

    elif task == "flip_flop":
        datamodule = FlipFlopDataModule(
            n_bits=config["n_bits"],
            p_pulse=config["p_pulse"],
            pulse_amplitude=config.get("pulse_amplitude", 1.0),
            num_time_steps=config["num_time_steps"],
            num_val_trajectories=config.get("num_val_trajectories", 2000),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        config["input_size"] = datamodule.input_size
        config["output_size"] = datamodule.output_size

    elif task == "teacher_student":
        datamodule = TeacherStudentDataModule(
            teacher_hidden_size=config["teacher_hidden_size"],
            teacher_recurrent_gain=config["teacher_recurrent_gain"],
            teacher_timescale=config["teacher_timescale"],
            teacher_activation=config.get("teacher_activation", "Tanh"),
            teacher_wrec_init=config.get("teacher_wrec_init", "normal_scaled"),
            teacher_seed=config.get("teacher_seed", 42),
            input_dim=config.get("input_dim", 2),
            output_dim=config.get("output_dim", 2),
            num_time_steps=config["num_time_steps"],
            dt=config["dt"],
            num_sequences=config.get("num_sequences", 500),
            train_val_split=config.get("train_val_split", 0.8),
            savgol_window=config.get("savgol_window", 5),
            savgol_polyorder=config.get("savgol_polyorder", 2),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        config["input_size"] = datamodule.input_size
        config["output_size"] = datamodule.output_size

    elif task == "null":
        datamodule = NullDataModule(
            input_size=config.get("input_size", 1),
            num_time_steps=config["num_time_steps"],
            num_trajectories=config["num_trajectories"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            train_val_split=config["train_val_split"],
        )
        config["input_size"] = datamodule.input_size
        config["output_size"] = datamodule.output_size

    else:
        raise ValueError(f"Unknown task: {task}")

    return datamodule


# ============================================================================
# Model factory
# ============================================================================

def create_coupled_rnn_model(config: dict):
    """Create CoupledRNN model and its Lightning wrapper."""
    model = CoupledRNN(
        input_size=config["input_size"],
        r_hidden_size=config["r_hidden_size"],
        s_hidden_size=config["s_hidden_size"],
        output_size=config["output_size"],
        dt=config["dt"],
        tau_r=config["tau_r"],
        tau_s=config["tau_s"],
        activation=getattr(nn, config.get("activation", "Tanh")),
        zero_diag_wrec=config.get("zero_diag_wrec", True),
        recurrent_gain=config.get("recurrent_gain", 1.0),
        noise_std=config.get("noise_std", 0.0),
        wrec_init=config.get("wrec_init", "orthogonal"),
        w_s_scale=config.get("w_s_scale", 1.0),
    )

    if "max_steps" in config:
        lr_step_size = config.get("lr_step_size", 1000)
        lr_interval = "step"
    else:
        lr_step_size = config["step_size"]
        lr_interval = "epoch"

    lightning_module = CoupledRNNLightning(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        step_size=lr_step_size,
        gamma=config["gamma"],
        task=config.get("task", "flip_flop"),
        lr_interval=lr_interval,
    )

    return model, lightning_module


# ============================================================================
# Training loop
# ============================================================================

def single_seed(config: dict) -> dict:
    """
    Train a CoupledRNN for a single seed.  Can be called from the CLI
    entry point below or from the sweep runner.
    """
    seed_everything(config["seed"], workers=True)
    seed = config["seed"]
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[coupled_run] Starting training — seed={seed}, run_id={run_id}")

    # Directory structure
    if "sweep_dir" in config and "experiment_name" in config:
        run_dir = os.path.join(
            config["sweep_dir"], config["experiment_name"], f"seed_{seed}"
        )
        wandb_name = (
            f"{config['project_name']}_{config['experiment_name']}_seed{seed}_{run_id}"
        )
        wandb_group = os.path.basename(config["sweep_dir"])
        wandb_job_type = config["experiment_name"]
    else:
        run_dir = os.path.join(log_dir, "single_runs", f"coupled_{run_id}")
        wandb_name = f"{config['project_name']}_coupled_{run_id}"
        wandb_group = None
        wandb_job_type = "single_run"

    wandb_tags = ["coupled"]
    if config.get("activation"):
        wandb_tags.append(config["activation"])
    if "experiment_name" in config:
        wandb_tags.append(config["experiment_name"])

    checkpoints_dir = os.path.join(run_dir, "checkpoints")

    wandb_logger = WandbLogger(
        project=config["project_name"],
        name=wandb_name,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=wandb_tags,
        dir=log_dir,
        save_dir=log_dir,
        config=config,
    )

    # Data
    task = config.get("task", "flip_flop")
    datamodule = create_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Model
    model, lightning_module = create_coupled_rnn_model(config)
    print(f"CoupledRNN: r_hidden={model.r_hidden_size}, s_hidden={model.s_hidden_size}")

    @rank_zero_only
    def create_directories():
        os.makedirs(checkpoints_dir, exist_ok=True)

    create_directories()

    @rank_zero_only
    def save_untrained_model():
        untrained_ckpt_path = os.path.join(checkpoints_dir, "untrained.ckpt")
        checkpoint = {
            "state_dict": lightning_module.state_dict(),
            "lr_schedulers": [],
            "epoch": 0,
            "global_step": 0,
            "hyper_parameters": dict(config),
        }
        torch.save(checkpoint, untrained_ckpt_path)
        print(f"Untrained model saved to: {untrained_ckpt_path}")

    save_untrained_model()

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="best-model-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    loss_logger = LossLoggerCallback(save_dir=run_dir)
    callbacks = [checkpoint_callback, loss_logger]

    checkpoint_every_n = config.get("save_checkpoint_every_n_epochs", None)
    if checkpoint_every_n is not None and checkpoint_every_n > 0:
        periodic_checkpoint = ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="checkpoint-epoch={epoch:03d}",
            every_n_epochs=checkpoint_every_n,
            save_top_k=-1,
        )
        callbacks.append(periodic_checkpoint)

    # Device / strategy
    devices = config.get("devices", "auto")
    accelerator = config.get("accelerator", "auto")
    if devices == "auto" and "device" in config:
        device_str = config["device"]
        if device_str.startswith("cuda:"):
            gpu_ids = device_str.replace("cuda:", "").split(",")
            devices = [int(g.strip()) for g in gpu_ids]
            accelerator = "gpu"
    strategy = config.get("strategy", "auto")
    if isinstance(devices, list) and len(devices) == 1:
        strategy = "auto"
    elif strategy == "auto" and task != "path_integration":
        strategy = "ddp_find_unused_parameters_true"

    # Trainer
    if "max_steps" in config:
        val_every = config.get("val_every_n_steps", 50)
        max_steps = config["max_steps"]
        max_epochs = (max_steps + val_every - 1) // val_every
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=max_epochs,
            limit_train_batches=val_every,
            default_root_dir=log_dir,
            callbacks=callbacks,
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
        )
    else:
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=config["max_epochs"],
            default_root_dir=log_dir,
            callbacks=callbacks,
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
        )

    skip_training = config.get("skip_training", False)
    final_val_loss = None

    if skip_training:
        print("skip_training=True — saving random (untrained) network only.")
    else:
        print("Training…")
        trainer.fit(lightning_module, train_loader, val_loader)
        print("Training complete!")
        if hasattr(lightning_module, "trainer") and lightning_module.trainer.callback_metrics:
            final_val_loss = lightning_module.trainer.callback_metrics.get("val_loss")
            if final_val_loss is not None:
                final_val_loss = float(final_val_loss)

    @rank_zero_only
    def save_additional_artifacts():
        model_path = os.path.join(run_dir, f"final_model_seed{seed}.pth")
        torch.save(lightning_module.model.state_dict(), model_path)
        config_path = os.path.join(run_dir, f"config_seed{seed}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        print(f"Artifacts saved to: {run_dir}")

    save_additional_artifacts()
    wandb.finish()

    return {"final_val_loss": final_val_loss}


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Coupled RNN (single run)")
    parser.add_argument(
        "--config", type=str, default="flip_flop_coupled.yaml",
        help="Config file name under base_configs/",
    )
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "base_configs", args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    single_seed(config)
