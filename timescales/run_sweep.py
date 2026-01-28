#!/usr/bin/env python3
"""
Parameter sweep experiment runner with multi-GPU scheduling.

Takes an experiment configuration file and runs all parameter combinations
with multiple seeds, distributed across available GPUs.

Usage:
    # Run on all 4 GPUs in parallel
    python run_sweep.py --sweep sweep_configs/my_sweep.yaml --gpus 0,1,2,3
    
    # Run on single GPU (sequential)
    python run_sweep.py --sweep sweep_configs/my_sweep.yaml --gpus 0
    
    # Run sequentially on default GPU
    python run_sweep.py --sweep sweep_configs/my_sweep.yaml
"""

import os
import sys
import yaml
import argparse
import datetime
import copy
import itertools
import subprocess
import tempfile
from typing import List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue
from threading import Lock
import numpy as np


# ============================================================================
# Configuration Loading and Generation
# ============================================================================

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
    """Generate experiment configurations from grid specification."""
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
        name_parts = {}
        
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
                
                if isinstance(value, float):
                    name_parts[param_name] = f"{value:.3g}"
                else:
                    name_parts[param_name] = str(value)
            else:
                overrides[param_name] = value
                if isinstance(value, float):
                    name_parts[param_name] = f"{value:.3g}"
                else:
                    name_parts[param_name] = str(value)
        
        # Generate experiment name
        if "format" in naming_config:
            exp_name = naming_config["format"]
            for param_name, value in zip(param_names, combo):
                if isinstance(value, float):
                    formatted_value = f"{value:.3g}"
                else:
                    formatted_value = str(value)
                exp_name = exp_name.replace("{" + param_name + "}", formatted_value)
        else:
            exp_name = "_".join(f"{k}_{v}" for k, v in name_parts.items())
        
        # Merge: base_with_fixed + grid overrides
        merged_config = deep_merge_dict(base_with_fixed, overrides)
        experiment_configs.append((exp_name, merged_config))
    
    return experiment_configs


def generate_experiment_configs(sweep_config: Dict) -> List[Tuple[str, Dict]]:
    """Generate all individual experiment configurations from sweep."""
    base_config = sweep_config["_base_config"]
    
    if "grid" in sweep_config:
        return generate_grid_experiments(sweep_config)
    
    # Original experiments list mode
    experiments = sweep_config["experiments"]
    fixed_overrides = sweep_config.get("fixed_overrides", {})
    base_with_fixed = deep_merge_dict(base_config, fixed_overrides)
    
    experiment_configs = []
    for exp in experiments:
        exp_name = exp["name"]
        overrides = exp.get("overrides", {})
        merged_config = deep_merge_dict(base_with_fixed, overrides)
        experiment_configs.append((exp_name, merged_config))

    return experiment_configs


# ============================================================================
# Job Definition
# ============================================================================

class Job:
    """A single training job (one experiment + one seed)."""
    def __init__(self, exp_name: str, config: Dict, seed: int, sweep_dir: str):
        self.exp_name = exp_name
        self.config = config
        self.seed = seed
        self.sweep_dir = sweep_dir
        self.gpu_id = None  # Assigned when scheduled
    
    @property
    def job_id(self) -> str:
        return f"{self.exp_name}_seed{self.seed}"
    
    @property
    def seed_dir(self) -> str:
        return os.path.join(self.sweep_dir, self.exp_name, f"seed_{self.seed}")


def create_jobs(
    experiment_configs: List[Tuple[str, Dict]], 
    seeds: List[int], 
    sweep_dir: str
) -> List[Job]:
    """Create all jobs for the sweep."""
    jobs = []
    for exp_name, config in experiment_configs:
        for seed in seeds:
            jobs.append(Job(exp_name, config, seed, sweep_dir))
    return jobs


# ============================================================================
# Job Execution (Subprocess-based for isolation)
# ============================================================================

def run_job_subprocess(job: Job, gpu_id: int) -> Dict[str, Any]:
    """
    Run a single job in a subprocess for complete isolation.
    
    This avoids DDP issues and ensures clean WandB run separation.
    """
    # Create seed directory
    os.makedirs(job.seed_dir, exist_ok=True)
    
    # Prepare config with GPU and sweep info
    run_config = copy.deepcopy(job.config)
    run_config["seed"] = job.seed
    run_config["sweep_dir"] = job.sweep_dir
    run_config["experiment_name"] = job.exp_name
    # Force single GPU (no DDP)
    # Use device [0] because CUDA_VISIBLE_DEVICES makes the target GPU appear as device 0
    run_config["devices"] = [0]
    run_config["accelerator"] = "gpu"
    
    # Write config to temp file
    config_file = os.path.join(job.seed_dir, "run_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)
    
    # Run training in subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(script_dir, "run_single_job.py"),
        "--config", config_file,
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] Starting {job.job_id}")
    
    import time
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=script_dir,
        )
        
        elapsed_time = time.time() - start_time
        elapsed_str = _format_duration(elapsed_time)
        
        if result.returncode == 0:
            # Try to read the result file
            result_file = os.path.join(job.seed_dir, "job_result.yaml")
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    job_result = yaml.safe_load(f)
            else:
                job_result = {"final_val_loss": None}
            
            print(f"[GPU {gpu_id}] ✓ {job.job_id} completed in {elapsed_str}")
            return {
                "experiment_name": job.exp_name,
                "seed": job.seed,
                "status": "completed",
                "seed_dir": job.seed_dir,
                "final_val_loss": job_result.get("final_val_loss"),
                "runtime_seconds": elapsed_time,
                "runtime_str": elapsed_str,
                "completed_at": datetime.datetime.now().isoformat(),
            }
        else:
            print(f"[GPU {gpu_id}] ✗ {job.job_id} failed after {elapsed_str}")
            print(f"  stderr: {result.stderr[:500] if result.stderr else 'None'}")
            return {
                "experiment_name": job.exp_name,
                "seed": job.seed,
                "status": "failed",
                "error": result.stderr[:1000] if result.stderr else "Unknown error",
                "runtime_seconds": elapsed_time,
                "failed_at": datetime.datetime.now().isoformat(),
            }
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[GPU {gpu_id}] ✗ {job.job_id} exception: {e}")
        return {
            "experiment_name": job.exp_name,
            "seed": job.seed,
            "status": "failed",
            "error": str(e),
            "runtime_seconds": elapsed_time,
            "failed_at": datetime.datetime.now().isoformat(),
        }


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# ============================================================================
# GPU Pool Scheduler
# ============================================================================

class GPUScheduler:
    """Manages GPU allocation for parallel job execution."""
    
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.n_gpus = len(gpu_ids)
        
    def run_jobs_parallel(self, jobs: List[Job]) -> List[Dict[str, Any]]:
        """Run jobs in parallel across available GPUs."""
        results = []
        
        if self.n_gpus == 0:
            print("No GPUs specified, running sequentially on CPU")
            for job in jobs:
                result = run_job_subprocess(job, gpu_id=0)
                results.append(result)
            return results
        
        print(f"\nScheduling {len(jobs)} jobs across {self.n_gpus} GPUs: {self.gpu_ids}")
        print("=" * 60)
        
        # Use ProcessPoolExecutor with max_workers = n_gpus
        with ProcessPoolExecutor(max_workers=self.n_gpus) as executor:
            # Submit jobs round-robin across GPUs
            future_to_job = {}
            for i, job in enumerate(jobs):
                gpu_id = self.gpu_ids[i % self.n_gpus]
                future = executor.submit(run_job_subprocess, job, gpu_id)
                future_to_job[future] = job
            
            # Collect results as they complete
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Job {job.job_id} raised exception: {e}")
                    results.append({
                        "experiment_name": job.exp_name,
                        "seed": job.seed,
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.datetime.now().isoformat(),
                    })
        
        return results


# ============================================================================
# Sweep Management
# ============================================================================

def save_sweep_metadata(
    sweep_dir: str, sweep_config: Dict, experiment_configs: List[Tuple[str, Dict]]
) -> None:
    """Save sweep metadata and configurations."""
    os.makedirs(sweep_dir, exist_ok=True)
    
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


def generate_sweep_summary(
    sweep_dir: str, all_results: List[Dict[str, Any]]
) -> None:
    """Generate overall sweep summary."""
    
    # Group results by experiment
    exp_results = {}
    for result in all_results:
        exp_name = result["experiment_name"]
        if exp_name not in exp_results:
            exp_results[exp_name] = []
        exp_results[exp_name].append(result)
    
    # Aggregate statistics
    total_runs = len(all_results)
    total_successful = len([r for r in all_results if r["status"] == "completed"])
    
    # Calculate total runtime
    all_runtimes = [r.get("runtime_seconds", 0) for r in all_results]
    total_runtime = sum(all_runtimes)

    # Per-experiment statistics
    experiment_stats = {}
    for exp_name, results in exp_results.items():
        successful = [r for r in results if r["status"] == "completed"]
        val_losses = [
            r["final_val_loss"]
            for r in successful
            if r.get("final_val_loss") is not None
        ]
        runtimes = [r.get("runtime_seconds", 0) for r in results]

        stats = {
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(results) - len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_runtime_seconds": sum(runtimes),
            "total_runtime_str": _format_duration(sum(runtimes)),
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
        "total_experiments": len(exp_results),
        "total_runs": total_runs,
        "total_successful_runs": total_successful,
        "total_failed_runs": total_runs - total_successful,
        "overall_success_rate": total_successful / total_runs if total_runs else 0,
        "total_runtime_seconds": total_runtime,
        "total_runtime_str": _format_duration(total_runtime),
        "experiment_statistics": experiment_stats,
    }

    # Save sweep summary
    summary_path = os.path.join(sweep_dir, "sweep_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(sweep_summary, f, default_flow_style=False, indent=2)

    print(f"\n{'='*80}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(exp_results)}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {total_successful}/{total_runs}")
    print(f"Overall success rate: {sweep_summary['overall_success_rate']:.2%}")
    print(f"Total runtime: {_format_duration(total_runtime)}")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary: {summary_path}")

    # Print per-experiment summary
    print("\nPer-experiment results:")
    for exp_name, stats in experiment_stats.items():
        runtime_str = stats.get("total_runtime_str", "?")
        status_str = f"{stats['successful_runs']}/{stats['total_runs']} successful"
        
        if "validation_loss_stats" in stats:
            loss_str = f"val_loss: {stats['validation_loss_stats']['mean']:.4f} ± {stats['validation_loss_stats']['std']:.4f}"
            print(f"  {exp_name}: {status_str}, {loss_str}, runtime: {runtime_str}")
        else:
            print(f"  {exp_name}: {status_str}, runtime: {runtime_str}")


def run_parameter_sweep(sweep_file: str, gpu_ids: List[int] = None):
    """Run full parameter sweep experiment with multi-GPU scheduling."""
    print(f"Loading parameter sweep configuration: {sweep_file}")

    # Load sweep configuration
    sweep_config = load_sweep_config(sweep_file)
    n_seeds = sweep_config["n_seeds"]
    experiment_configs = generate_experiment_configs(sweep_config)
    
    # Get GPUs from config if not specified on command line
    if gpu_ids is None:
        gpu_ids = sweep_config.get("gpus", [0])
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]

    print("\nParameter sweep configuration:")
    print(f"  Base config: {sweep_config['base_config']}")
    print(f"  Number of experiments: {len(experiment_configs)}")
    print(f"  Seeds per experiment: {n_seeds}")
    print(f"  Total runs: {len(experiment_configs) * n_seeds}")
    print(f"  Experiments: {[name for name, _ in experiment_configs]}")
    print(f"  GPUs: {gpu_ids}")

    # Create sweep directory
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    sweep_name = os.path.splitext(os.path.basename(sweep_file))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(log_dir, "experiments", f"{sweep_name}_{timestamp}")

    os.makedirs(sweep_dir, exist_ok=True)
    print(f"\nSweep directory: {sweep_dir}")

    # Save metadata
    save_sweep_metadata(sweep_dir, sweep_config, experiment_configs)

    # Create all jobs
    seeds = list(range(n_seeds))
    jobs = create_jobs(experiment_configs, seeds, sweep_dir)
    
    # Run jobs with GPU scheduler
    if gpu_ids is None:
        gpu_ids = [0]  # Default to GPU 0
    
    scheduler = GPUScheduler(gpu_ids)
    all_results = scheduler.run_jobs_parallel(jobs)

    # Generate summary
    generate_sweep_summary(sweep_dir, all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep with multi-GPU scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on GPUs 0,1,2,3 in parallel
  python run_sweep.py --sweep sweep_configs/my_sweep.yaml --gpus 0,1,2,3
  
  # Run on single GPU
  python run_sweep.py --sweep sweep_configs/my_sweep.yaml --gpus 2
  
  # Run sequentially on default GPU
  python run_sweep.py --sweep sweep_configs/my_sweep.yaml
        """
    )
    parser.add_argument(
        "--sweep", type=str, required=True, 
        help="Path to sweep configuration file"
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). Overrides config file."
    )
    args = parser.parse_args()

    # Parse GPU IDs (None if not specified, let config file decide)
    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    
    run_parameter_sweep(args.sweep, gpu_ids=gpu_ids)


if __name__ == "__main__":
    main()
