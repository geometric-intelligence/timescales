#!/usr/bin/env python
"""
Linear Path Integration Theory - Training Script

Validates theoretical predictions about linear RNNs for path integration:
1. N=1 with self-connections (alpha ≈ 1): Should work
2. N=1 without self-connections: Should fail for any alpha
3. N=2 without self-connections: Should work via off-diagonal coupling

Usage:
    python train.py                    # Run all conditions
    python train.py --condition n1     # Run only N=1 experiments  
    python train.py --condition n2     # Run only N=2 experiments
    python train.py --epochs 200       # Override max epochs
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lightning import Trainer, seed_everything

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from timescales.rnns.multitimescale_rnn import MultiTimescaleRNN, MultiTimescaleRNNLightning
from timescales.datamodules import PathIntegration1DDataModule

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent / "results"

TASK_CONFIG = {
    "dt": 0.02,  # 20ms timestep
    "num_time_steps": 100,
    "arena_size": 2.0,  # 2m arena
    "num_place_cells": 32,
    "place_cell_rf": 0.15,
    "DoG": True,
    "surround_scale": 2.0,
    "place_cell_layout": "uniform",
    "velocity_mean": 0.0,
    "velocity_std": 0.5,
    "velocity_tau": 1.0,
    "num_trajectories": 5000,
    "batch_size": 64,
    "num_workers": 4,
    "train_val_split": 0.8,
}

TRAIN_CONFIG = {
    "max_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
}

# Alpha values to sweep for "without self-connections" experiments
ALPHA_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


# ============================================================================
# Helper Functions
# ============================================================================

def alpha_to_timescale(alpha: float, dt: float) -> float:
    """Convert alpha to timescale: alpha = 1 - exp(-dt/tau) => tau = -dt/ln(1-alpha)"""
    if alpha >= 1.0:
        return 1e-6  # Very small timescale for alpha ~ 1
    return -dt / np.log(1 - alpha)


def create_model(hidden_size: int, zero_diag_wrec: bool, alpha: float) -> MultiTimescaleRNN:
    """Create a linear MultiTimescaleRNN model."""
    dt = TASK_CONFIG["dt"]
    tau = alpha_to_timescale(alpha, dt)
    
    return MultiTimescaleRNN(
        input_size=1,
        hidden_size=hidden_size,
        output_size=TASK_CONFIG["num_place_cells"],
        dt=dt,
        timescales_config={"type": "discrete", "values": [tau]},
        activation=nn.Identity,
        learn_timescales=False,
        normalize_hidden=False,
        zero_diag_wrec=zero_diag_wrec,
    )


def create_lightning_module(model: MultiTimescaleRNN) -> MultiTimescaleRNNLightning:
    """Wrap model in Lightning module."""
    return MultiTimescaleRNNLightning(
        model=model,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        step_size=50,
        gamma=0.5,
        task="path_integration_1d",
    )


def train_model(
    model: MultiTimescaleRNN,
    datamodule: PathIntegration1DDataModule,
    max_epochs: int,
    seed: int = 42,
) -> dict:
    """Train a model and return results."""
    seed_everything(seed, workers=True)
    
    lightning_module = create_lightning_module(model)
    
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    
    trainer.fit(lightning_module, datamodule)
    
    final_val_loss = trainer.callback_metrics.get("val_loss", None)
    if final_val_loss is not None:
        final_val_loss = float(final_val_loss)
    
    return {
        "final_val_loss": final_val_loss,
        "model": model,
    }


def extract_model_params(model: MultiTimescaleRNN) -> dict:
    """Extract learned parameters from model."""
    W_rec = model.rnn_step.W_rec.weight.detach().cpu().numpy()
    W_in = model.rnn_step.W_in.weight.detach().cpu().numpy()
    alpha = model.rnn_step.current_alphas.detach().cpu().numpy()
    
    # Compute eigenvalues for N >= 2
    if W_rec.shape[0] >= 2:
        eigenvalues = np.linalg.eigvals(W_rec)
        eig_info = {
            "eigenvalues_real": [float(e.real) for e in eigenvalues],
            "eigenvalues_imag": [float(e.imag) for e in eigenvalues],
            "max_abs_eigenvalue": float(np.max(np.abs(eigenvalues))),
        }
    else:
        eig_info = {}
    
    return {
        "W_rec": W_rec.tolist(),
        "W_in": W_in.tolist(),
        "alpha": alpha.tolist(),
        **eig_info,
    }


def save_result(
    save_path: Path,
    model: MultiTimescaleRNN,
    config: dict,
    final_val_loss: float,
):
    """Save model checkpoint and metadata."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), save_path.with_suffix(".pt"))
    
    # Save metadata as JSON
    metadata = {
        "config": config,
        "final_val_loss": final_val_loss,
        "learned_params": extract_model_params(model),
    }
    with open(save_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved to {save_path}")


# ============================================================================
# Training Functions
# ============================================================================

def train_n1_with_selfconn(datamodule: PathIntegration1DDataModule, max_epochs: int):
    """Train N=1 with self-connections (alpha ≈ 1). Should work."""
    print("\n" + "=" * 60)
    print("Section A: N=1 with self-connections (should work)")
    print("=" * 60)
    
    alpha = 0.99
    config = {"hidden_size": 1, "zero_diag_wrec": False, "alpha": alpha}
    
    print(f"\nTraining: hidden_size=1, self-connections=True, alpha={alpha}")
    
    model = create_model(hidden_size=1, zero_diag_wrec=False, alpha=alpha)
    result = train_model(model, datamodule, max_epochs)
    
    print(f"Final validation loss: {result['final_val_loss']:.4f}")
    
    # Extract and print learned parameters
    params = extract_model_params(model)
    print(f"Learned W_rec = {params['W_rec'][0][0]:.4f} (theory: 1.0)")
    print(f"Learned W_in  = {params['W_in'][0][0]:.4f} (theory: dt={TASK_CONFIG['dt']})")
    
    save_result(
        RESULTS_DIR / "n1_with_selfconn",
        model,
        config,
        result["final_val_loss"],
    )
    
    return {"alpha": alpha, "final_val_loss": result["final_val_loss"]}


def train_n1_without_selfconn(datamodule: PathIntegration1DDataModule, max_epochs: int):
    """Train N=1 without self-connections. Should fail for any alpha."""
    print("\n" + "=" * 60)
    print("Section B: N=1 without self-connections (should fail)")
    print("=" * 60)
    
    results = {}
    
    for alpha in ALPHA_SWEEP:
        config = {"hidden_size": 1, "zero_diag_wrec": True, "alpha": alpha}
        
        print(f"\nTraining: hidden_size=1, self-connections=False, alpha={alpha}")
        
        model = create_model(hidden_size=1, zero_diag_wrec=True, alpha=alpha)
        result = train_model(model, datamodule, max_epochs)
        
        print(f"Final validation loss: {result['final_val_loss']:.4f}")
        
        save_result(
            RESULTS_DIR / "n1_without_selfconn" / f"alpha_{alpha}",
            model,
            config,
            result["final_val_loss"],
        )
        
        results[alpha] = result["final_val_loss"]
    
    print("\nSummary - N=1 without self-connections:")
    for alpha, loss in results.items():
        print(f"  α={alpha:.2f}: val_loss={loss:.4f}")
    
    return results


def train_n2_without_selfconn(datamodule: PathIntegration1DDataModule, max_epochs: int):
    """Train N=2 without self-connections. Should work via off-diagonal coupling."""
    print("\n" + "=" * 60)
    print("Section C: N=2 without self-connections (should work)")
    print("=" * 60)
    
    alphas_to_test = [0.5, 0.9, 0.99]
    results = {}
    
    for alpha in alphas_to_test:
        config = {"hidden_size": 2, "zero_diag_wrec": True, "alpha": alpha}
        
        print(f"\nTraining: hidden_size=2, self-connections=False, alpha={alpha}")
        
        model = create_model(hidden_size=2, zero_diag_wrec=True, alpha=alpha)
        result = train_model(model, datamodule, max_epochs)
        
        print(f"Final validation loss: {result['final_val_loss']:.4f}")
        
        # Print eigenvalue info
        params = extract_model_params(model)
        eigs = [complex(r, i) for r, i in zip(params["eigenvalues_real"], params["eigenvalues_imag"])]
        print(f"W_rec eigenvalues: {eigs}")
        print(f"Max |eigenvalue|: {params['max_abs_eigenvalue']:.4f}")
        
        save_result(
            RESULTS_DIR / "n2_without_selfconn" / f"alpha_{alpha}",
            model,
            config,
            result["final_val_loss"],
        )
        
        results[alpha] = {
            "final_val_loss": result["final_val_loss"],
            "max_abs_eigenvalue": params["max_abs_eigenvalue"],
        }
    
    print("\nSummary - N=2 without self-connections:")
    for alpha, info in results.items():
        print(f"  α={alpha:.2f}: val_loss={info['final_val_loss']:.4f}, max|λ|={info['max_abs_eigenvalue']:.4f}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train linear path integration models")
    parser.add_argument(
        "--condition",
        choices=["n1", "n2", "all"],
        default="all",
        help="Which condition to train (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAIN_CONFIG["max_epochs"],
        help=f"Max training epochs (default: {TRAIN_CONFIG['max_epochs']})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Linear Path Integration Theory - Training")
    print("=" * 60)
    print(f"Condition: {args.condition}")
    print(f"Max epochs: {args.epochs}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create data module (shared across all experiments)
    print("\nCreating data module...")
    seed_everything(args.seed)
    datamodule = PathIntegration1DDataModule(**TASK_CONFIG)
    datamodule.prepare_data()
    datamodule.setup()
    print(f"Training samples: {len(datamodule.train_dataset)}")
    print(f"Validation samples: {len(datamodule.val_dataset)}")
    
    # Store all results for summary
    all_results = {}
    
    # Run requested conditions
    if args.condition in ("n1", "all"):
        all_results["n1_with_selfconn"] = train_n1_with_selfconn(datamodule, args.epochs)
        all_results["n1_without_selfconn"] = train_n1_without_selfconn(datamodule, args.epochs)
    
    if args.condition in ("n2", "all"):
        all_results["n2_without_selfconn"] = train_n2_without_selfconn(datamodule, args.epochs)
    
    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()