"""
Unified training entry point for all RNN architectures.

Dispatches to the correct model factory based on model_type in config:
  - "rnn"     : standard RNN with per-unit time constants
  - "coupled" : two coupled populations (nonlinear r + linear s)
  - "schur"   : Schur-decomposed recurrent weights

Usage:
    python train.py --config configs/rnn/flip_flop.yaml
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

from callbacks import (
    LossLoggerCallback,
    PositionDecodingCallback,
    TrajectoryVisualizationCallback,
    GradientStatisticsCallback,
)
from timescales.analysis.measurements import PositionDecodingMeasurement
from timescales.datamodules import (
    PathIntegrationDataModule,
    PathIntegration1DDataModule,
    HierarchicalCounterDataModule,
    FlipFlopDataModule,
    NullDataModule,
    TeacherStudentDataModule,
)

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))


# ============================================================================
# Datamodule factory (shared across all architectures)
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
# Model factories
# ============================================================================

def _create_rnn_model(config: dict):
    from timescales.rnns.rnn import RNN, RNNLightning

    model = RNN(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        dt=config["dt"],
        time_constants_config=config.get("time_constants_config"),
        activation=getattr(nn, config["activation"]),
        learn_time_constants=config["learn_time_constants"],
        init_time_constant=config.get("init_time_constant"),
        shared_time_constant=config["shared_time_constant"],
        normalize_hidden=config["normalize_hidden"],
        zero_diag_wrec=config["zero_diag_wrec"],
        recurrent_gain=config["recurrent_gain"],
        noise_std=config["noise_std"],
        wrec_init=config["wrec_init"],
        alpha_parameterization=config["alpha_parameterization"],
        stability_param=config.get("stability_param", 2.0),
        dynamics_type=config["dynamics_type"],
    )

    if "max_steps" in config:
        lr_step_size = config.get("lr_step_size", 1000)
        lr_interval = "step"
    else:
        lr_step_size = config["step_size"]
        lr_interval = "epoch"

    lightning_module = RNNLightning(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        step_size=lr_step_size,
        gamma=config["gamma"],
        task=config.get("task", "path_integration"),
        precondition_gradients=config.get("precondition_gradients", False),
        eps_alpha=config.get("eps_alpha", 1e-2),
        lr_interval=lr_interval,
    )
    return model, lightning_module


def _create_coupled_model(config: dict):
    from timescales.rnns.coupled_rnn import CoupledRNN, CoupledRNNLightning

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
        w_s_gain=config.get("w_s_gain", 1.0),
        trainable_w_s=config.get("trainable_w_s", False),
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


def _create_schur_model(config: dict):
    from timescales.rnns.schur_rnn import SchurRNN, SchurRNNLightning

    model = SchurRNN(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        dt=config["dt"],
        tau=config["tau"],
        recurrent_gain=config.get("recurrent_gain", 1.0),
        noise_std=config.get("noise_std", 0.0),
        wrec_init=config.get("wrec_init", "normal_scaled"),
        train_t=config.get("train_t", True),
        train_q=config.get("train_q", False),
        q_parameterization=config.get("q_parameterization", "cayley"),
    )

    if "max_steps" in config:
        lr_step_size = config.get("lr_step_size", 1000)
        lr_interval = "step"
    else:
        lr_step_size = config["step_size"]
        lr_interval = "epoch"

    lightning_module = SchurRNNLightning(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        step_size=lr_step_size,
        gamma=config["gamma"],
        task=config.get("task", "flip_flop"),
        lr_interval=lr_interval,
    )
    return model, lightning_module


MODEL_FACTORIES = {
    "rnn": _create_rnn_model,
    "multitimescale": _create_rnn_model,  # backward compat
    "coupled": _create_coupled_model,
    "schur": _create_schur_model,
}


# ============================================================================
# Callback assembly
# ============================================================================

def _build_callbacks(config: dict, run_dir: str, datamodule=None):
    """Build the list of Lightning callbacks based on config."""
    task = config.get("task", "flip_flop")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="best-model-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    loss_logger = LossLoggerCallback(save_dir=run_dir)
    callbacks = [checkpoint_callback, loss_logger]

    if config.get("track_gradients", False):
        callbacks.append(GradientStatisticsCallback(
            save_dir=run_dir,
            log_every_n_steps=config.get("grad_log_every_n_steps", 100),
            track_per_weight_matrix=config.get("grad_track_per_weight_matrix", True),
        ))

    if task == "path_integration" and datamodule is not None:
        callbacks.append(PositionDecodingCallback(
            measurement=PositionDecodingMeasurement(),
            datamodule=datamodule,
            log_every_n_epochs=config["log_every_n_epochs"],
            save_dir=run_dir,
        ))
        callbacks.append(TrajectoryVisualizationCallback(
            place_cell_centers=datamodule.place_cell_centers,
            arena_size=config["arena_size"],
            log_every_n_epochs=config["viz_log_every_n_epochs"],
            num_trajectories_to_plot=3,
        ))

    checkpoint_every_n = config.get("save_checkpoint_every_n_epochs", None)
    if checkpoint_every_n is not None and checkpoint_every_n > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(run_dir, "checkpoints"),
            filename="checkpoint-epoch={epoch:03d}",
            every_n_epochs=checkpoint_every_n,
            save_top_k=-1,
        ))

    return callbacks


# ============================================================================
# Training loop (shared across all architectures)
# ============================================================================

def single_seed(config: dict) -> dict:
    """Train a single seed. Used by both single runs and sweeps."""
    seed_everything(config["seed"], workers=True)

    model_type = config["model_type"]
    seed = config["seed"]
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[train] model_type={model_type}, seed={seed}, run_id={run_id}")

    if model_type not in MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Supported: {list(MODEL_FACTORIES.keys())}"
        )

    # Directory layout
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
        run_dir = os.path.join(log_dir, "single_runs", f"{model_type}_{run_id}")
        wandb_name = f"{config['project_name']}_{model_type}_{run_id}"
        wandb_group = None
        wandb_job_type = "single_run"

    wandb_tags = [model_type]
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
    factory = MODEL_FACTORIES[model_type]
    model, lightning_module = factory(config)

    if config.get("freeze_wrec", False) and hasattr(model, "rnn_step"):
        wrec = getattr(model.rnn_step, "W_rec", None)
        if wrec is not None:
            for p in wrec.parameters():
                p.requires_grad_(False)
            print("  freeze_wrec=True — W_rec frozen (only W_in, W_out train)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model.__class__.__name__}, trainable params: {n_params}")

    @rank_zero_only
    def create_directories():
        os.makedirs(checkpoints_dir, exist_ok=True)
    create_directories()

    @rank_zero_only
    def save_untrained_model():
        torch.save({
            "state_dict": lightning_module.state_dict(),
            "lr_schedulers": [],
            "epoch": 0,
            "global_step": 0,
            "hyper_parameters": dict(config),
        }, os.path.join(checkpoints_dir, "untrained.ckpt"))
    save_untrained_model()

    # Callbacks
    callbacks = _build_callbacks(config, run_dir, datamodule=datamodule)

    # Trainer
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
        print("skip_training=True — saving untrained network only.")
    else:
        print("Training…")
        trainer.fit(lightning_module, train_loader, val_loader)
        print("Training complete!")

        if hasattr(lightning_module, "trainer") and lightning_module.trainer.callback_metrics:
            fvl = lightning_module.trainer.callback_metrics.get("val_loss")
            if fvl is not None:
                final_val_loss = float(fvl)

    @rank_zero_only
    def save_artifacts():
        torch.save(
            lightning_module.model.state_dict(),
            os.path.join(run_dir, f"final_model_seed{seed}.pth"),
        )
        with open(os.path.join(run_dir, f"config_seed{seed}.yaml"), "w") as f:
            yaml.dump(config, f)
        if task == "path_integration":
            torch.save(
                datamodule.place_cell_centers,
                os.path.join(run_dir, f"place_cell_centers_seed{seed}.pt"),
            )
        print(f"  Artifacts saved: {run_dir}")

    save_artifacts()
    wandb.finish()
    return {"final_val_loss": final_val_loss}


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RNN (any architecture)")
    parser.add_argument(
        "--config", type=str, default="configs/rnn/flip_flop.yaml",
        help="Path to config file (relative to this script's directory)",
    )
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    single_seed(config)
