#!/usr/bin/env python3
"""
Parameter sweep experiment runner.

Takes an experiment configuration file and runs all parameter combinations
with multiple seeds for uncertainty quantification.
"""

import os
import yaml
import argparse
import datetime
import copy
import itertools
from typing import List, Dict, Any, Tuple
from single_run import single_seed
import numpy as np
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def deep_merge_dict(base: Dict, override: Dict) -> Dict:
    """Deep merge override dictionary into base dictionary."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_sweep_config(sweep_file: str) -> Dict:
    """Load sweep configuration with base config and experiments."""
    if not os.path.exists(sweep_file):
        raise FileNotFoundError(f"Sweep file not found: {sweep_file}")

    with open(sweep_file, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Load base configuration
    base_config_path = sweep_config["base_config"]
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    sweep_config["_base_config"] = base_config
    return sweep_config


def generate_grid_experiments(sweep_config: Dict) -> List[Tuple[str, Dict]]:
    """Generate experiment configurations from grid specification.
    
    Expects sweep_config to have a 'grid' key with parameter specifications:
    
    grid:
      linear_speed_mean: [0.05, 0.1, 0.15, ...]
      timescale_values: [0.949, 0.448, 0.280, ...]  # For discrete timescales
    
    Or for complex nested parameters:
    
    grid:
      linear_speed_mean: [0.05, 0.1, 0.15, ...]
      timescales_config__values: [[0.949], [0.448], [0.280], ...]  # Use __ for nested keys
    
    Optional fixed_overrides: Applied to all experiments before grid params
    Optional naming: Custom naming format specification
    """
    base_config = sweep_config["_base_config"]
    grid_spec = sweep_config["grid"]
    fixed_overrides = sweep_config.get("fixed_overrides", {})
    naming_config = sweep_config.get("naming", {})
    
    # Apply fixed overrides to base config first
    base_with_fixed = deep_merge_dict(base_config, fixed_overrides)
    
    # Extract parameter names and values
    param_names = list(grid_spec.keys())
    param_values = [grid_spec[name] for name in param_names]
    
    # Generate all combinations (Cartesian product)
    combinations = list(itertools.product(*param_values))
    
    print(f"Generating grid sweep: {len(combinations)} experiments")
    print(f"Grid dimensions: {' × '.join([f'{len(v)}' for v in param_values])} = {len(combinations)}")
    if fixed_overrides:
        print(f"Fixed overrides applied: {list(fixed_overrides.keys())}")
    
    experiment_configs = []
    for combo in combinations:
        # Create overrides dict for this combination
        overrides = {}
        name_parts = {}  # Use dict to track specific naming components
        
        for param_name, value in zip(param_names, combo):
            # Handle nested keys (e.g., "timescales_config__std")
            if "__" in param_name:
                keys = param_name.split("__")
                current = overrides
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
                
                # Special handling for timescales_config__values to compute alpha
                if param_name == "timescales_config__values" and isinstance(value, list) and len(value) == 1:
                    # Compute alpha from timescale: alpha = 1 - exp(-dt / timescale)
                    dt = base_config.get("dt", 0.1)  # Default dt = 0.1s
                    timescale = value[0]
                    alpha = 1 - np.exp(-dt / timescale) if timescale > 0.001 else 1.0
                    name_parts["alpha"] = f"{alpha:.1f}"
                else:
                    # Store the nested param for naming
                    # Use full param name for naming clarity
                    if isinstance(value, float):
                        name_parts[param_name] = f"{value:.3g}"
                    else:
                        name_parts[param_name] = str(value)
            else:
                overrides[param_name] = value
                
                # Format value for name based on parameter type
                if param_name == "linear_speed_mean":
                    name_parts["speed"] = f"{value:.2f}"
                elif isinstance(value, float):
                    name_parts[param_name] = f"{value:.3g}"
                else:
                    name_parts[param_name] = str(value)
        
        # Generate experiment name
        if "format" in naming_config:
            # Use custom format string
            # Replace {param_name} with values
            exp_name = naming_config["format"]
            for param_name, value in zip(param_names, combo):
                if isinstance(value, float):
                    formatted_value = f"{value:.3g}"
                else:
                    formatted_value = str(value)
                exp_name = exp_name.replace("{" + param_name + "}", formatted_value)
        elif "speed" in name_parts and "alpha" in name_parts:
            # Legacy: speed_X_alpha_Y format if both exist
            exp_name = f"speed_{name_parts['speed']}_alpha_{name_parts['alpha']}"
        else:
            # Fallback to generic naming
            exp_name = "_".join(f"{k}_{v}" for k, v in name_parts.items())
        
        # Merge: base_with_fixed + grid overrides
        merged_config = deep_merge_dict(base_with_fixed, overrides)
        experiment_configs.append((exp_name, merged_config))
    
    return experiment_configs


def generate_experiment_configs(sweep_config: Dict) -> List[Tuple[str, Dict]]:
    """Generate all individual experiment configurations from sweep.
    
    Supports two modes:
    1. 'experiments' list: Manually specified experiments
    2. 'grid' dict: Automatic grid generation from parameter lists
    """
    base_config = sweep_config["_base_config"]
    
    # Check if grid mode
    if "grid" in sweep_config:
        return generate_grid_experiments(sweep_config)
    
    # Original experiments list mode
    experiments = sweep_config["experiments"]
    experiment_configs = []
    for exp in experiments:
        exp_name = exp["name"]
        overrides = exp.get("overrides", {})

        # Merge base config with overrides
        merged_config = deep_merge_dict(base_config, overrides)
        experiment_configs.append((exp_name, merged_config))

    return experiment_configs


@rank_zero_only
def save_sweep_metadata(
    sweep_dir: str, sweep_config: Dict, experiment_configs: List[Tuple[str, Dict]]
) -> None:
    """Save sweep metadata and configurations."""

    # Save sweep metadata
    sweep_metadata = {
        "sweep_name": os.path.basename(sweep_dir),
        "base_config_file": sweep_config["base_config"],
        "n_seeds": sweep_config["n_seeds"],
        "n_experiments": len(experiment_configs),
        "total_runs": len(experiment_configs) * sweep_config["n_seeds"],
        "created_at": datetime.datetime.now().isoformat(),
        "experiments": [name for name, _ in experiment_configs],
    }

    metadata_path = os.path.join(sweep_dir, "sweep_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(sweep_metadata, f, default_flow_style=False, indent=2)

    # Save individual experiment configs
    configs_dir = os.path.join(sweep_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    for exp_name, config in experiment_configs:
        config_path = os.path.join(configs_dir, f"{exp_name}_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Sweep metadata saved to: {metadata_path}")


def run_experiment(
    exp_name: str, config: Dict, seeds: List[int], sweep_dir: str
) -> List[Dict[str, Any]]:
    """Run a single experiment configuration with multiple seeds."""

    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}")

    # Create experiment directory
    exp_dir = os.path.join(sweep_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Run each seed
    run_results = []
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'-'*60}")
        print(f"EXPERIMENT {exp_name} - SEED {seed_idx + 1}/{len(seeds)}: seed={seed}")
        print(f"{'-'*60}")

        # Create seed config for single_seed
        seed_config = config.copy()
        seed_config["seed"] = seed
        seed_config["sweep_dir"] = sweep_dir
        seed_config["experiment_name"] = exp_name

        # The seed directory will be: sweep_dir/exp_name/seed_{seed}/
        seed_dir = os.path.join(exp_dir, f"seed_{seed}")

        try:
            result = single_seed(seed_config)

            # Save run summary
            run_summary = {
                "experiment_name": exp_name,
                "seed": seed,
                "status": "completed",
                "seed_dir": seed_dir,
                "final_val_loss": result.get("final_val_loss", None),
                "completed_at": datetime.datetime.now().isoformat(),
            }

            summary_path = os.path.join(seed_dir, "run_summary.yaml")
            with open(summary_path, "w") as f:
                yaml.dump(run_summary, f, default_flow_style=False, indent=2)

            print(f"✓ {exp_name} seed {seed} completed successfully")
            run_results.append(run_summary)

        except Exception as e:
            print(f"✗ {exp_name} seed {seed} failed with error: {str(e)}")

            # Save error summary
            error_summary = {
                "experiment_name": exp_name,
                "seed": seed,
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.datetime.now().isoformat(),
            }

            os.makedirs(seed_dir, exist_ok=True)
            error_path = os.path.join(seed_dir, "error_summary.yaml")
            with open(error_path, "w") as f:
                yaml.dump(error_summary, f, default_flow_style=False, indent=2)

            run_results.append(error_summary)

    # Generate experiment summary
    successful_runs = [r for r in run_results if r["status"] == "completed"]
    val_losses = [
        r["final_val_loss"]
        for r in successful_runs
        if r.get("final_val_loss") is not None
    ]

    exp_summary = {
        "experiment_name": exp_name,
        "experiment_completed_at": datetime.datetime.now().isoformat(),
        "total_seeds": len(run_results),
        "successful_runs": len(successful_runs),
        "failed_runs": len(run_results) - len(successful_runs),
        "success_rate": len(successful_runs) / len(run_results) if run_results else 0,
    }

    if val_losses:
        exp_summary["validation_loss_stats"] = {
            "mean": float(np.mean(val_losses)),
            "std": float(np.std(val_losses)),
            "min": float(np.min(val_losses)),
            "max": float(np.max(val_losses)),
            "median": float(np.median(val_losses)),
        }

    exp_summary["run_details"] = run_results

    # Save experiment summary
    summary_path = os.path.join(exp_dir, "experiment_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(exp_summary, f, default_flow_style=False, indent=2)

    print(f"\nExperiment {exp_name} complete!")
    print(f"Successful runs: {len(successful_runs)}/{len(run_results)}")
    if val_losses:
        print(
            f"Validation loss: {exp_summary['validation_loss_stats']['mean']:.4f} ± {exp_summary['validation_loss_stats']['std']:.4f}"
        )

    return run_results


@rank_zero_only
def generate_sweep_summary(
    sweep_dir: str, all_results: Dict[str, List[Dict[str, Any]]]
) -> None:
    """Generate overall sweep summary."""

    # Aggregate statistics across all experiments
    total_runs = sum(len(results) for results in all_results.values())
    total_successful = sum(
        len([r for r in results if r["status"] == "completed"])
        for results in all_results.values()
    )

    # Per-experiment statistics
    experiment_stats = {}
    for exp_name, results in all_results.items():
        successful = [r for r in results if r["status"] == "completed"]
        val_losses = [
            r["final_val_loss"]
            for r in successful
            if r.get("final_val_loss") is not None
        ]

        stats = {
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(results) - len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
        }

        if val_losses:
            stats["validation_loss_stats"] = {
                "mean": float(np.mean(val_losses)),
                "std": float(np.std(val_losses)),
                "min": float(np.min(val_losses)),
                "max": float(np.max(val_losses)),
                "median": float(np.median(val_losses)),
            }

        experiment_stats[exp_name] = stats

    # Overall sweep summary
    sweep_summary = {
        "sweep_completed_at": datetime.datetime.now().isoformat(),
        "total_experiments": len(all_results),
        "total_runs": total_runs,
        "total_successful_runs": total_successful,
        "total_failed_runs": total_runs - total_successful,
        "overall_success_rate": total_successful / total_runs if total_runs else 0,
        "experiment_statistics": experiment_stats,
    }

    # Save sweep summary
    summary_path = os.path.join(sweep_dir, "sweep_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(sweep_summary, f, default_flow_style=False, indent=2)

    print(f"\n{'='*80}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {total_successful}/{total_runs}")
    print(f"Overall success rate: {sweep_summary['overall_success_rate']:.2%}")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary: {summary_path}")

    # Print per-experiment summary
    print("\nPer-experiment results:")
    for exp_name, stats in experiment_stats.items():
        print(
            f"  {exp_name}: {stats['successful_runs']}/{stats['total_runs']} successful",
            end="",
        )
        if "validation_loss_stats" in stats:
            print(
                f" (val_loss: {stats['validation_loss_stats']['mean']:.4f} ± {stats['validation_loss_stats']['std']:.4f})"
            )
        else:
            print()


@rank_zero_only
def create_sweep_directory_only(sweep_dir: str) -> None:
    """Create sweep directory structure (rank 0 only)."""
    os.makedirs(sweep_dir, exist_ok=True)


def run_parameter_sweep(sweep_file: str):
    """Run full parameter sweep experiment."""
    print(f"Loading parameter sweep configuration: {sweep_file}")

    # Load sweep configuration
    sweep_config = load_sweep_config(sweep_file)
    n_seeds = sweep_config["n_seeds"]
    experiment_configs = generate_experiment_configs(sweep_config)

    print("Parameter sweep configuration:")
    print(f"  Base config: {sweep_config['base_config']}")
    print(f"  Number of experiments: {len(experiment_configs)}")
    print(f"  Seeds per experiment: {n_seeds}")
    print(f"  Total runs: {len(experiment_configs) * n_seeds}")
    print(f"  Experiments: {[name for name, _ in experiment_configs]}")

    # Create sweep directory
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    sweep_name = os.path.splitext(os.path.basename(sweep_file))[0]

    sweep_file_mtime = os.path.getmtime(sweep_file)
    timestamp = datetime.datetime.fromtimestamp(sweep_file_mtime).strftime(
        "%Y%m%d_%H%M%S"
    )
    sweep_dir = os.path.join(log_dir, "experiments", f"{sweep_name}_{timestamp}")

    # Rest stays the same...
    create_sweep_directory_only(sweep_dir)  # Already has @rank_zero_only
    save_sweep_metadata(
        sweep_dir, sweep_config, experiment_configs
    )  # Already has @rank_zero_only

    seeds = list(range(n_seeds))

    all_results = {}
    for exp_name, config in experiment_configs:
        results = run_experiment(exp_name, config, seeds, sweep_dir)
        all_results[exp_name] = results

    generate_sweep_summary(sweep_dir, all_results)  # Already has @rank_zero_only


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep experiment")
    parser.add_argument(
        "--sweep", type=str, required=True, help="Path to sweep configuration file"
    )
    args = parser.parse_args()

    run_parameter_sweep(args.sweep)


if __name__ == "__main__":
    main()
