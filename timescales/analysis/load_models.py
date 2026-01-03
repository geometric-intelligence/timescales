import os
import glob
import yaml
import torch
import torch.nn as nn
import json
from collections import defaultdict
from timescales.rnns.multitimescale_rnn import MultiTimescaleRNN, MultiTimescaleRNNLightning


def load_experiment_sweep(
    sweep_dir: str,
    device: str,
    use_lightning_checkpoint: bool = True,
    checkpoint_type: str = "best",
) -> dict:
    """
    Load all models from an experiment sweep.

    Args:
        sweep_dir: Path to the sweep directory
        device: Device to load models on
        use_lightning_checkpoint: If True, load Lightning checkpoints; if False, load final PyTorch models
        checkpoint_type: "best" or "last" (only used if use_lightning_checkpoint=True)

    Returns:
        dict: Nested dictionary with structure [experiment_name][seed] = (model, config)
    """

    # Read sweep metadata
    metadata_path = os.path.join(sweep_dir, "sweep_metadata.yaml")
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    # Read sweep summary (optional for incomplete sweeps)
    summary_path = os.path.join(sweep_dir, "sweep_summary.yaml")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = yaml.safe_load(f)
    else:
        print("⚠ sweep_summary.yaml not found (sweep may be incomplete)")

    models = defaultdict(dict)
    failed_loads = []
    skipped_incomplete = []

    print(
        f"Loading {metadata['n_experiments']} experiments with {metadata['n_seeds']} seeds each..."
    )
    print(f"Total models to load: {metadata['total_runs']}")
    print(
        f"Using {'Lightning checkpoints' if use_lightning_checkpoint else 'PyTorch final models'}"
    )
    print()

    for experiment_name in metadata["experiments"]:
        print(f"Loading experiment: {experiment_name}")
        experiment_dir = os.path.join(sweep_dir, experiment_name)
        
        # Skip if experiment directory doesn't exist yet
        if not os.path.exists(experiment_dir):
            print(f"  ⚠ Experiment directory not found, skipping")
            skipped_incomplete.append((experiment_name, "all", "Experiment directory not found"))
            continue

        # Read experiment summary (optional for incomplete experiments)
        exp_summary_path = os.path.join(experiment_dir, "experiment_summary.yaml")
        exp_summary = None
        if os.path.exists(exp_summary_path):
            with open(exp_summary_path, "r") as f:
                exp_summary = yaml.safe_load(f)

        for seed in range(metadata["n_seeds"]):
            seed_dir = os.path.join(experiment_dir, f"seed_{seed}")

            try:
                # Check if seed directory exists
                if not os.path.exists(seed_dir):
                    skipped_incomplete.append((experiment_name, seed, "Seed directory not found"))
                    continue
                
                # Check if critical files exist before attempting to load
                config_path = os.path.join(seed_dir, f"config_seed{seed}.yaml")
                if not os.path.exists(config_path):
                    skipped_incomplete.append((experiment_name, seed, "Config file not found"))
                    continue
                
                # Check if model/checkpoint exists
                if use_lightning_checkpoint:
                    checkpoint_dir = os.path.join(seed_dir, "checkpoints")
                    if not os.path.exists(checkpoint_dir):
                        skipped_incomplete.append((experiment_name, seed, "Checkpoints directory not found"))
                        continue
                    
                    if checkpoint_type == "best":
                        checkpoint_files = glob.glob(
                            os.path.join(checkpoint_dir, "best-model-*.ckpt")
                        )
                    else:  # last
                        checkpoint_files = glob.glob(
                            os.path.join(checkpoint_dir, "last.ckpt")
                        )
                    
                    if not checkpoint_files:
                        skipped_incomplete.append((experiment_name, seed, f"No {checkpoint_type} checkpoint found"))
                        continue
                else:
                    model_path = os.path.join(seed_dir, f"final_model_seed{seed}.pth")
                    if not os.path.exists(model_path):
                        skipped_incomplete.append((experiment_name, seed, "Final model not found"))
                        continue

                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                # Load training losses
                losses_path = os.path.join(seed_dir, "training_losses.json")
                training_data = None
                if os.path.exists(losses_path):
                    with open(losses_path, "r") as f:
                        training_data = json.load(f)
                else:
                    print(
                        f"  ⚠ No training_losses.json found for {experiment_name}/seed_{seed}"
                    )

                # Load decoding errors
                decoding_errors_path = os.path.join(
                    seed_dir, "position_decoding_errors.json"
                )
                decoding_errors = None
                if os.path.exists(decoding_errors_path):
                    with open(decoding_errors_path, "r") as f:
                        decoding_errors = json.load(f)
                else:
                    print(
                        f"  ⚠ No position_decoding_errors.json found for {experiment_name}/seed_{seed}"
                    )

                # Load place_cell_centers
                place_cells_path = os.path.join(
                    seed_dir, f"place_cell_centers_seed{seed}.pt"
                )
                place_cell_centers = None
                if os.path.exists(place_cells_path):
                    place_cell_centers = torch.load(
                        place_cells_path, map_location=device
                    )
                else:
                    print(
                        f"  ⚠ No place_cell_centers found for {experiment_name}/seed_{seed}"
                    )

                if use_lightning_checkpoint:
                    checkpoint_path = checkpoint_files[0]

                    # Create base model
                    model = MultiTimescaleRNN(
                        input_size=config["input_size"],
                        hidden_size=config["hidden_size"],
                        output_size=config["num_place_cells"],
                        dt=config["dt"],
                        timescales_config=config["timescales_config"],
                        activation=getattr(nn, config["activation"]),
                    )

                    # Load Lightning model
                    model_lightning = MultiTimescaleRNNLightning.load_from_checkpoint(
                        checkpoint_path,
                        model=model,
                        learning_rate=config["learning_rate"],
                        weight_decay=config["weight_decay"],
                        step_size=config["step_size"],
                        gamma=config["gamma"],
                    )

                    model = model_lightning.model

                else:
                    # Load PyTorch final model
                    # Create model
                    model = MultiTimescaleRNN(
                        input_size=config["input_size"],
                        hidden_size=config["hidden_size"],
                        output_size=config["num_place_cells"],
                        dt=config["dt"],
                        timescales_config=config["timescales_config"],
                        activation=getattr(nn, config["activation"]),
                    )

                    # Load state dict
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)

                # Move to device and set to eval mode
                model.to(device)
                model.eval()

                # Prepare training data dictionary
                training_losses_dict = {}
                if training_data:
                    # Extract and align the data as done in plot_loss_curves
                    epochs = training_data["epochs"][
                        :-1
                    ]  # Remove last epoch if incomplete
                    train_losses = training_data["train_losses_epoch"]
                    val_losses = training_data["val_losses_epoch"][
                        :-1
                    ]  # Align with epochs

                    training_losses_dict = {
                        "epochs": epochs,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "raw_data": training_data,  # Keep full data if needed
                    }

                # Get final val loss from experiment summary if available
                final_val_loss = None
                if exp_summary and "run_details" in exp_summary:
                    final_val_loss = next(
                        (run["final_val_loss"]
                        for run in exp_summary["run_details"]
                        if run["seed"] == seed),
                        None
                    )

                # Store model and config
                models[experiment_name][seed] = {
                    "model": model,
                    "config": config,
                    "final_val_loss": final_val_loss,
                    "training_losses": training_losses_dict,
                    "position_decoding_errors": decoding_errors,
                    "place_cell_centers": place_cell_centers,
                }

                print(f"  ✓ Loaded {experiment_name}/seed_{seed}")

            except Exception as e:
                failed_loads.append((experiment_name, seed, str(e)))
                print(f"  ✗ Failed to load {experiment_name}/seed_{seed}: {e}")

    print()
    print(
        f"Successfully loaded: {sum(len(seeds) for seeds in models.values())}/{metadata['total_runs']} models"
    )

    if skipped_incomplete:
        print(f"Skipped incomplete: {len(skipped_incomplete)}")
        for exp, seed, reason in skipped_incomplete:
            print(f"  - {exp}/seed_{seed}: {reason}")

    if failed_loads:
        print(f"Failed loads: {len(failed_loads)}")
        for exp, seed, error in failed_loads:
            print(f"  - {exp}/seed_{seed}: {error}")

    return dict(models), metadata, summary
