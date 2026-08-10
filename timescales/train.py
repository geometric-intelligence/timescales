"""
Unified training entry point for the RNN model.

The RNN has per-unit time constants; model_type in config selects the factory
("rnn"), leaving room to register additional architectures later.

Usage:
    python train.py --config configs/rnn/flip_flop.yaml
"""

import argparse
import datetime
import json
import os

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
    GradientStatisticsCallback,
    TauTrajectoryCallback,
    SpectralSnapshotCallback,
    SpectralTrajectoryCallback,
    RecurrentRestructuringCallback,
)
from timescales.datamodules import (
    FlipFlopDataModule,
    SignedFlipFlopDataModule,
    NullDataModule,
    SineWaveDataModule,
)
from timescales import run_ids
from timescales import convergence as convergence_metrics
from timescales import provenance
from timescales.presets import resolve_presets

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))


def _capture_rng_state() -> dict:
    """Snapshot torch + numpy + cuda RNG states for reproducibility."""
    import numpy as _np
    state = {
        "torch": torch.get_rng_state(),
        "numpy": _np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


@rank_zero_only
def _dump_rng_state(run_dir: str, tag: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"rng_state_{tag}.pt")
    torch.save(_capture_rng_state(), path)
    print(f"RNG state ({tag}) saved to: {path}")


@rank_zero_only
def _write_completion_marker(run_dir: str, result: dict) -> None:
    """Write the completion marker that makes reruns skippable/resumable."""
    os.makedirs(run_dir, exist_ok=True)
    with open(run_ids.marker_path(run_dir), "w") as f:
        yaml.dump(result, f, default_flow_style=False)


# ============================================================================
# Datamodule factory (shared across all architectures)
# ============================================================================

def create_datamodule(config: dict):
    """Create datamodule based on task type in config."""
    task = config.get("task", "flip_flop")

    if task == "flip_flop":
        datamodule = FlipFlopDataModule(
            n_bits=config["n_bits"],
            p_pulse=config["p_pulse"],
            pulse_amplitude=config.get("pulse_amplitude", 1.0),
            force_initial_pulse=config.get("force_initial_pulse", False),
            num_time_steps=config["num_time_steps"],
            num_val_trajectories=config.get("num_val_trajectories", 2000),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        config["input_size"] = datamodule.input_size
        config["output_size"] = datamodule.output_size

    elif task == "signed_flip_flop":
        datamodule = SignedFlipFlopDataModule(
            n_bits=config["n_bits"],
            p_pulse=config["p_pulse"],
            pulse_amplitude=config.get("pulse_amplitude", 1.0),
            num_time_steps=config["num_time_steps"],
            num_val_trajectories=config.get("num_val_trajectories", 100),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        config["input_size"] = datamodule.input_size
        config["output_size"] = datamodule.output_size

    elif task == "sine_wave":
        datamodule = SineWaveDataModule(
            n_pairs=config.get("n_pairs", 1),
            periods=config.get("periods", config.get("period", 10.0)),
            dt=config["dt"],
            num_time_steps=config["num_time_steps"],
            init_hidden_value=config.get("init_hidden_value", 1.0),
            random_phase=config.get("random_phase", False),
            num_val_trajectories=config.get("num_val_trajectories", 200),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        config["input_size"] = datamodule.input_size
        config["output_size"] = datamodule.output_size
        config["init_hidden_value"] = datamodule.init_hidden_value

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
        init_time_constants_config=config.get("init_time_constants_config"),
        shared_time_constant=config["shared_time_constant"],
        normalize_hidden=config["normalize_hidden"],
        zero_diag_wrec=config["zero_diag_wrec"],
        recurrent_gain=config["recurrent_gain"],
        noise_std=config["noise_std"],
        wrec_init=config["wrec_init"],
        wrec_init_config=config.get("wrec_init_config"),
        wrec_init_scale=config.get("wrec_init_scale", 1.0),
        recurrent_parameterization=config.get(
            "recurrent_parameterization", "standard"
        ),
        output_coupling_gamma=config.get("output_coupling_gamma"),
        use_biases=config.get("use_biases", True),
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
        gamma=config.get("gamma", 1.0),
        task=config.get("task", "flip_flop"),
        precondition_gradients=config.get("precondition_gradients", False),
        eps_alpha=config.get("eps_alpha", 1e-2),
        lr_interval=lr_interval,
        init_hidden_value=config.get("init_hidden_value"),
        signed_output_threshold=config.get("signed_output_threshold", 0.33),
        optimizer_name=config.get("optimizer_name", "adam"),
        use_lr_scheduler=config.get("use_lr_scheduler", True),
        lr_scheduler_gamma=config.get("lr_scheduler_gamma"),
        sgld_beta=config.get("sgld_beta", 2000.0),
        sgld_add_noise=config.get("sgld_add_noise", True),
        sgld_parameter_prior=config.get("sgld_parameter_prior", False),
        clark_loss_scaling=config.get("clark_loss_scaling", True),
    )
    return model, lightning_module


MODEL_FACTORIES = {
    "rnn": _create_rnn_model,
    "multitimescale": _create_rnn_model,  # backward compat
}


# ============================================================================
# Callback assembly
# ============================================================================

def _build_callbacks(config: dict, run_dir: str, datamodule=None):
    """Build the list of Lightning callbacks based on config."""
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

    if config.get("track_tau_trajectory", True):
        callbacks.append(TauTrajectoryCallback(save_dir=run_dir))

    if config.get("track_spectral_snapshots", True) and config.get("model_type") == "rnn":
        callbacks.append(SpectralSnapshotCallback(
            save_dir=run_dir,
            eps_real_axis=config.get("eps_real_axis", 0.1),
        ))

    if config.get("track_spectral_trajectory", False) and config.get("model_type") == "rnn":
        callbacks.append(SpectralTrajectoryCallback(
            save_dir=run_dir,
            top_k=config.get("spectral_trajectory_top_k", 2),
            log_every_n_validation_epochs=config.get(
                "spectral_trajectory_log_every_n_validation_epochs", 1
            ),
            include_wrec=config.get("spectral_trajectory_include_wrec", True),
        ))

    if config.get("track_gradients", False):
        callbacks.append(GradientStatisticsCallback(
            save_dir=run_dir,
            log_every_n_steps=config.get("grad_log_every_n_steps", 100),
            track_per_weight_matrix=config.get("grad_track_per_weight_matrix", True),
        ))

    if config.get("track_recurrent_restructuring", False):
        callbacks.append(RecurrentRestructuringCallback(save_dir=run_dir))

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

    # Deterministic identity: same (config, seed) -> same run_id and output dir,
    # so reruns are idempotent and sweeps resumable. started_at is kept only for
    # human-readable logging/wandb, not for the directory layout.
    # NOTE: fingerprint the *as-authored* config BEFORE preset resolution, so
    # sweep-level resume (which fingerprints the authored config) agrees.
    fingerprint = run_ids.config_fingerprint(config)
    run_id = run_ids.run_id_for(config, seed)
    started_at = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Expand high-level presets (trainable / tau_init) into low-level keys; the
    # resolved config is what gets saved alongside the run.
    resolve_presets(config)

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

    # Idempotent rerun: if this exact run already finished, skip it (unless forced).
    if run_ids.is_complete(run_dir, fingerprint) and not config.get("force_rerun", False):
        print(f"  [skip] run already complete: {run_dir}")
        marker = run_ids.load_marker(run_dir) or {}
        return {
            "final_val_loss": marker.get("final_val_loss"),
            "run_id": run_id,
            "fingerprint": fingerprint,
            "seed": seed,
            "skipped": True,
        }

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
            print("  freeze_wrec=True — W_rec frozen")

    if config.get("freeze_win", False) and hasattr(model, "rnn_step"):
        win = getattr(model.rnn_step, "W_in", None)
        if win is not None:
            for p in win.parameters():
                p.requires_grad_(False)
            print("  freeze_win=True — W_in frozen")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model.__class__.__name__}, trainable params: {n_params}")

    @rank_zero_only
    def create_directories():
        os.makedirs(checkpoints_dir, exist_ok=True)
    create_directories()

    _dump_rng_state(run_dir, tag="init")

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
    elif strategy == "auto":
        strategy = "ddp_find_unused_parameters_true"

    if "max_steps" in config:
        val_every = config.get("val_every_n_steps", 50)
        max_steps = config["max_steps"]
        max_epochs = (max_steps + val_every - 1) // val_every
        trainer_kwargs = dict(
            logger=wandb_logger,
            max_epochs=max_epochs,
            limit_train_batches=val_every,
            default_root_dir=log_dir,
            callbacks=callbacks,
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
        )
        gradient_clip_val = config.get("gradient_clip_val")
        if gradient_clip_val is not None:
            trainer_kwargs["gradient_clip_val"] = gradient_clip_val
        trainer = Trainer(**trainer_kwargs)
    else:
        trainer_kwargs = dict(
            logger=wandb_logger,
            max_epochs=config["max_epochs"],
            default_root_dir=log_dir,
            callbacks=callbacks,
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
        )
        gradient_clip_val = config.get("gradient_clip_val")
        if gradient_clip_val is not None:
            trainer_kwargs["gradient_clip_val"] = gradient_clip_val
        trainer = Trainer(**trainer_kwargs)

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
        print(f"  Artifacts saved: {run_dir}")

    save_artifacts()
    _dump_rng_state(run_dir, tag="final")
    wandb.finish()

    # Steps-to-convergence from the full validation curve LossLoggerCallback wrote.
    # Every curve-based candidate is computed and stored, so the headline metric
    # can be chosen at aggregation time without rerunning (config.convergence_metric
    # selects the headline; None defers the choice).
    convergence = None
    curve_path = os.path.join(run_dir, "training_losses.json")
    if not skip_training and os.path.exists(curve_path):
        with open(curve_path) as f:
            curve = json.load(f)
        convergence = convergence_metrics.compute_convergence(curve, config)

    result = {
        "final_val_loss": final_val_loss,
        "run_id": run_id,
        "fingerprint": fingerprint,
        "seed": seed,
        "started_at": started_at,
        "completed_at": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "skipped": False,
        "convergence": convergence,
        "steps_to_convergence": (convergence or {}).get("steps_to_convergence"),
        "provenance": provenance.collect_provenance(),
    }
    _write_completion_marker(run_dir, result)
    return result


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
