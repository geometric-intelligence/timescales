import torch.nn as nn
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import yaml
import torch
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from callbacks import (
    LossLoggerCallback,
    PositionDecodingCallback,
    TrajectoryVisualizationCallback,
    TimescaleVisualizationCallback,
    GradientStatisticsCallback,
)

from timescales.analysis.measurements import PositionDecodingMeasurement

from timescales.datamodules import PathIntegrationDataModule, HierarchicalCounterDataModule

from timescales.rnns.rnn import RNN, RNNLightning


def create_datamodule(config: dict):
    """Create datamodule based on task type in config."""
    task = config.get("task", "path_integration")  # Default to path integration
    
    if task == "path_integration":
        datamodule = PathIntegrationDataModule(
            # Trajectory configuration 
            trajectory_type=config["trajectory_type"],
            velocity_representation=config["velocity_representation"],
            dt=config["dt"],
            num_time_steps=config["num_time_steps"],
            arena_size=config["arena_size"],
            
            # Place cell parameters
            num_place_cells=config["num_place_cells"],
            place_cell_rf=config["place_cell_rf"],
            DoG=config["DoG"],
            surround_scale=config["surround_scale"],
            place_cell_layout=config["place_cell_layout"],
            
            # Speed/behavioral timescale parameters
            linear_speed_mean=config.get("linear_speed_mean"),
            linear_speed_std=config.get("linear_speed_std"),
            behavioral_timescale_mean=config.get("behavioral_timescale_mean"),
            behavioral_timescale_std=config.get("behavioral_timescale_std"),
            
            # OU dynamics
            linear_speed_tau=config.get("linear_speed_tau", 1.0),
            angular_speed_mean=config.get("angular_speed_mean", 0.0),
            angular_speed_std=config.get("angular_speed_std", 1.0),
            angular_speed_tau=config.get("angular_speed_tau", 0.4),
            
            # DataLoader parameters
            num_trajectories=config["num_trajectories"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            train_val_split=config["train_val_split"],
        )
        # For path integration, output_size is num_place_cells
        config["input_size"] = {"cartesian": 2, "polar": 2, "sincos_polar": 3}[
            config["velocity_representation"]
        ]
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
        # Set input/output sizes from datamodule
        config["input_size"] = datamodule.input_size
        config["output_size"] = datamodule.output_size
        
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return datamodule
from timescales.rnns.multitimescale_rnn import MultiTimescaleRNN, MultiTimescaleRNNLightning

import datetime

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
print("Log directory:", log_dir)


def create_vanilla_rnn_model(
    config: dict,
):
    """Create vanilla PathIntRNN model and lightning module."""
    model = RNN(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        alpha=config["alpha"],
        activation=getattr(nn, config["activation"]),
    )

    lightning_module = RNNLightning(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        step_size=config["step_size"],
        gamma=config["gamma"],
        task=config.get("task", "path_integration"),
    )

    return model, lightning_module


def create_multitimescale_rnn_model(
    config: dict,
):
    """Create MultiTimescaleRNN model and lightning module."""
    # Check if timescales should be learned
    learn_timescales = config["learn_timescales"]
    
    model = MultiTimescaleRNN(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        dt=config["dt"],
        timescales_config=config.get("timescales_config"),  # Can be None if learning
        activation=getattr(nn, config["activation"]),
        learn_timescales=learn_timescales,
        init_timescale=config.get("init_timescale"),  # Uniform init if provided
        normalize_hidden=config.get("normalize_hidden", False),
        zero_diag_wrec=config.get("zero_diag_wrec", True),
    )

    lightning_module = MultiTimescaleRNNLightning(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        step_size=config["step_size"],
        gamma=config["gamma"],
        task=config.get("task", "path_integration"),
    )

    return model, lightning_module


def single_seed(config: dict) -> dict:
    """
    Main training function for a single seed.
    Used by both single runs and parameter sweeps.

    Returns:
        dict: Training results including final validation loss
    """
    # Set global seed
    seed_everything(config["seed"], workers=True)
    print(f"Global seed set to: {config['seed']}")

    model_type = config["model_type"]
    seed = config["seed"]
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Starting training run: {run_id}")
    print(f"Model type: {model_type}, Seed: {seed}")

    # Determine save directory structure
    if "sweep_dir" in config and "experiment_name" in config:
        # Parameter sweep mode: save in sweep_dir/experiment_name/seed_{seed}/
        run_dir = os.path.join(
            config["sweep_dir"], config["experiment_name"], f"seed_{seed}"
        )
        wandb_name = (
            f"{config['project_name']}_{config['experiment_name']}_seed{seed}_{run_id}"
        )
        wandb_group = os.path.basename(config["sweep_dir"])  # Group by sweep name
    else:
        # Single run mode: save in log_dir/single_runs/{model_type}_{run_id}/
        run_dir = os.path.join(log_dir, "single_runs", f"{model_type}_{run_id}")
        wandb_name = f"{config['project_name']}_{model_type}_{run_id}"
        wandb_group = None

    # Create checkpoints subdirectory
    checkpoints_dir = os.path.join(run_dir, "checkpoints")

    wandb_logger = WandbLogger(
        project=config["project_name"],
        name=wandb_name,
        group=wandb_group,
        dir=log_dir,
        save_dir=log_dir,
        config=config,
    )
    print("Wandb initialized. Find logs at: ", log_dir)
    print(f"Wandb run name: {wandb_name}")

    # Create datamodule based on task type
    task = config.get("task", "path_integration")
    datamodule = create_datamodule(config)

    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print("Data prepared")

    # Create model based on type
    if model_type == "vanilla":
        model, lightning_module = create_vanilla_rnn_model(config)
        print("Vanilla PathIntRNN initialized")
    elif model_type == "multitimescale":
        model, lightning_module = create_multitimescale_rnn_model(config)
        print("MultiTimescaleRNN initialized")
        timescale_stats = model.get_timescale_stats()
        print(f"Timescale statistics: {timescale_stats}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"{model_type.capitalize()} Lightning module initialized")

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

    # Best model checkpoint (monitors validation loss)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="best-model-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    loss_logger = LossLoggerCallback(save_dir=run_dir)

    # Gradient statistics callback (task-agnostic)
    gradient_stats_callback = GradientStatisticsCallback(
        save_dir=run_dir,
        log_every_n_steps=config.get("grad_log_every_n_steps", 100),
        track_per_weight_matrix=config.get("grad_track_per_weight_matrix", True),
    )

    # Build callbacks list
    callbacks = [
        checkpoint_callback,
        loss_logger,
        gradient_stats_callback,
    ]
    
    # Task-specific callbacks
    if task == "path_integration":
        position_decoding_callback = PositionDecodingCallback(
            measurement=PositionDecodingMeasurement(config["decode_k"]),
            datamodule=datamodule,
            log_every_n_epochs=config["log_every_n_epochs"],
            save_dir=run_dir,
        )

        trajectory_viz_callback = TrajectoryVisualizationCallback(
            place_cell_centers=datamodule.place_cell_centers,
            arena_size=config["arena_size"],
            decode_k=config["decode_k"],
            log_every_n_epochs=config["viz_log_every_n_epochs"],
            num_trajectories_to_plot=3,
        )
        callbacks.extend([position_decoding_callback, trajectory_viz_callback])
    
    elif task == "binary_counter":
        # TODO: Add binary counter specific callbacks (e.g., per-level accuracy)
        pass

    # Periodic checkpoint (every N epochs)
    checkpoint_every_n = config.get("save_checkpoint_every_n_epochs", None)
    if checkpoint_every_n is not None and checkpoint_every_n > 0:
        periodic_checkpoint = ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="checkpoint-epoch={epoch:03d}",
            every_n_epochs=checkpoint_every_n,
            save_top_k=-1,  # Save all periodic checkpoints
        )
        callbacks.append(periodic_checkpoint)
        print(f"Periodic checkpoints enabled: saving every {checkpoint_every_n} epochs")

    if model_type == "multitimescale":
        timescale_viz_callback = TimescaleVisualizationCallback(
            save_dir=run_dir,
            log_every_n_epochs=config.get("viz_log_every_n_epochs", 10),
        )
        callbacks.append(timescale_viz_callback)

    # Use devices and accelerator directly from config if specified
    devices = config.get("devices", "auto")
    accelerator = config.get("accelerator", "auto")

    # Fallback to old device parsing if not specified
    if devices == "auto" and "device" in config:
        device_str = config["device"]
        if device_str.startswith("cuda:"):
            gpu_ids = device_str.replace("cuda:", "").split(",")
            devices = [int(gpu_id.strip()) for gpu_id in gpu_ids]
            accelerator = "gpu"

    # Use DDP with find_unused_parameters for tasks that don't use all model params
    # (e.g., binary_counter doesn't use W_h_init when init_context=None)
    strategy = config.get("strategy", "auto")
    if strategy == "auto" and task != "path_integration":
        strategy = "ddp_find_unused_parameters_true"
    
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config["max_epochs"],
        default_root_dir=log_dir,
        callbacks=callbacks,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
    )

    print("Trainer initialized")
    print("Training...")

    #TODO: Add optimizer options (Adam, SGD, etc.; currently only Adam is supported)
    trainer.fit(lightning_module, train_loader, val_loader)

    print("Training complete!")

    # Get final validation loss
    final_val_loss = None
    if (
        hasattr(lightning_module, "trainer")
        and lightning_module.trainer.callback_metrics
    ):
        final_val_loss = lightning_module.trainer.callback_metrics.get("val_loss", None)
        if final_val_loss is not None:
            final_val_loss = float(final_val_loss)

    # Save artifacts
    @rank_zero_only
    def save_additional_artifacts():
        model_path = os.path.join(run_dir, f"final_model_seed{seed}.pth")
        torch.save(lightning_module.model.state_dict(), model_path)

        config_path = os.path.join(run_dir, f"config_seed{seed}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Save task-specific artifacts
        if task == "path_integration":
            place_cells_path = os.path.join(run_dir, f"place_cell_centers_seed{seed}.pt")
            torch.save(datamodule.place_cell_centers, place_cells_path)

        print(f"All artifacts saved to: {run_dir}")

    save_additional_artifacts()

    return {"final_val_loss": final_val_loss}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single RNN training")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "base_configs", args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    single_seed(config)
