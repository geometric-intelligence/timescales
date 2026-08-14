#!/usr/bin/env python3
"""
Unified parameter sweep runner for all sequence-model architectures.

model_type in the config YAML determines which architecture is trained.

Usage:
    python sweep.py --sweep sweep_configs/rnn/flip_flop_gain_sweep.yaml --gpus 0,1,2,3
"""

import os
import sys
import copy
import yaml
import argparse
import datetime
import itertools
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any
import numpy as np

from timescales import run_ids


# ============================================================================
# Configuration loading and generation
# ============================================================================

def _deep_merge_dict(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_sweep_config(sweep_file: str) -> dict:
    if not os.path.exists(sweep_file):
        raise FileNotFoundError(f"Sweep file not found: {sweep_file}")

    with open(sweep_file, "r") as f:
        sweep_config = yaml.safe_load(f)

    base_config_path = sweep_config["base_config"]
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    sweep_config["_base_config"] = base_config
    return sweep_config


def _generate_grid_experiments(sweep_config: dict) -> list[tuple[str, dict]]:
    base_config = sweep_config["_base_config"]
    grid_spec = sweep_config["grid"]
    fixed_overrides = sweep_config.get("fixed_overrides", {})
    naming_config = sweep_config.get("naming", {})

    base_with_fixed = _deep_merge_dict(base_config, fixed_overrides)

    param_names = list(grid_spec.keys())
    param_values = [grid_spec[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    print(f"Generating grid sweep: {len(combinations)} experiments")
    print(f"Grid dimensions: {' x '.join([str(len(v)) for v in param_values])} = {len(combinations)}")
    if fixed_overrides:
        print(f"Fixed overrides applied: {list(fixed_overrides.keys())}")

    experiment_configs = []
    for combo in combinations:
        overrides = {}
        name_parts = {}

        for param_name, value in zip(param_names, combo):
            if "__" in param_name:
                keys = param_name.split("__")
                current = overrides
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
                name_parts[param_name] = f"{value:.3g}" if isinstance(value, float) else str(value)
            else:
                overrides[param_name] = value
                name_parts[param_name] = f"{value:.3g}" if isinstance(value, float) else str(value)

        if "format" in naming_config:
            exp_name = naming_config["format"]
            for param_name, value in zip(param_names, combo):
                formatted_value = f"{value:.3g}" if isinstance(value, float) else str(value)
                exp_name = exp_name.replace("{" + param_name + "}", formatted_value)
        else:
            exp_name = "_".join(f"{k}_{v}" for k, v in name_parts.items())

        merged_config = _deep_merge_dict(base_with_fixed, overrides)
        experiment_configs.append((exp_name, merged_config))

    return experiment_configs


def _format_value(value) -> str:
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def _generate_variant_experiments(sweep_config: dict) -> list[tuple[str, dict]]:
    """Cartesian product over named variant axes, optionally crossed with `grid`.

    `variants` maps axis -> {variant_name: overrides_dict}. Each combination
    deep-merges the base config with one overrides dict per axis (in axis order),
    then applies scalar `grid` overrides on top. Naming uses `naming.format` with
    {axis_name} / {grid_param} placeholders, else joins variant names and grid
    values with underscores.
    """
    base_config = sweep_config["_base_config"]
    variants_spec = sweep_config["variants"]
    fixed_overrides = sweep_config.get("fixed_overrides", {})
    naming_config = sweep_config.get("naming", {})
    base_with_fixed = _deep_merge_dict(base_config, fixed_overrides)

    axis_names = list(variants_spec.keys())
    axis_options = [list(variants_spec[a].items()) for a in axis_names]

    grid_spec = sweep_config.get("grid", {})
    grid_names = list(grid_spec.keys())
    grid_values = [grid_spec[n] for n in grid_names]

    n_total = 1
    for opts in axis_options:
        n_total *= len(opts)
    for vals in grid_values:
        n_total *= len(vals)
    print(f"Generating variant sweep: {n_total} experiments "
          f"({' x '.join(f'{a}:{len(o)}' for a, o in zip(axis_names, axis_options, strict=True))}"
          f"{' x ' if grid_names else ''}"
          f"{' x '.join(f'{n}:{len(v)}' for n, v in zip(grid_names, grid_values, strict=True))})")

    experiment_configs = []
    for variant_combo in itertools.product(*axis_options):
        for grid_combo in itertools.product(*grid_values):
            merged = base_with_fixed
            substitutions: dict[str, str] = {}
            for axis, (vname, overrides) in zip(axis_names, variant_combo, strict=True):
                merged = _deep_merge_dict(merged, overrides or {})
                substitutions[axis] = str(vname)

            grid_overrides: dict = {}
            for gname, gval in zip(grid_names, grid_combo, strict=True):
                if "__" in gname:
                    keys = gname.split("__")
                    current = grid_overrides
                    for key in keys[:-1]:
                        current = current.setdefault(key, {})
                    current[keys[-1]] = gval
                else:
                    grid_overrides[gname] = gval
                substitutions[gname] = _format_value(gval)
            merged = _deep_merge_dict(merged, grid_overrides)

            if "format" in naming_config:
                exp_name = naming_config["format"]
                for key, val in substitutions.items():
                    exp_name = exp_name.replace("{" + key + "}", val)
            else:
                exp_name = "_".join(substitutions.values())

            experiment_configs.append((exp_name, merged))

    return experiment_configs


def generate_experiment_configs(sweep_config: dict) -> list[tuple[str, dict]]:
    if "variants" in sweep_config:
        return _generate_variant_experiments(sweep_config)
    if "grid" in sweep_config:
        return _generate_grid_experiments(sweep_config)

    base_config = sweep_config["_base_config"]
    experiments = sweep_config["experiments"]
    fixed_overrides = sweep_config.get("fixed_overrides", {})
    base_with_fixed = _deep_merge_dict(base_config, fixed_overrides)

    experiment_configs = []
    for exp in experiments:
        exp_name = exp["name"]
        overrides = exp.get("overrides", {})
        merged_config = _deep_merge_dict(base_with_fixed, overrides)
        experiment_configs.append((exp_name, merged_config))

    return experiment_configs


# ============================================================================
# Job definition
# ============================================================================

class Job:
    """A single training job (one experiment + one seed)."""
    def __init__(self, exp_name: str, config: dict, seed: int, sweep_dir: str):
        self.exp_name = exp_name
        self.config = config
        self.seed = seed
        self.sweep_dir = sweep_dir
        self.gpu_id = None

    @property
    def job_id(self) -> str:
        return f"{self.exp_name}_seed{self.seed}"

    @property
    def seed_dir(self) -> str:
        return os.path.join(self.sweep_dir, self.exp_name, f"seed_{self.seed}")


def create_jobs(
    experiment_configs: list[tuple[str, dict]],
    seeds: list[int],
    sweep_dir: str,
    seeds_outermost: bool = True,
) -> list[Job]:
    """Build the list of Job objects.

    With seeds_outermost=True (default), the iteration runs all experiments
    at seed=s before moving to seed=s+1. Combined with the GPU pool's
    submission order, this means the first complete pass over the grid
    (one seed per condition) finishes before any second-seed work starts —
    useful for "fast first pass, accumulate later" schedules.
    """
    jobs = []
    if seeds_outermost:
        for seed in seeds:
            for exp_name, config in experiment_configs:
                jobs.append(Job(exp_name, config, seed, sweep_dir))
    else:
        for exp_name, config in experiment_configs:
            for seed in seeds:
                jobs.append(Job(exp_name, config, seed, sweep_dir))
    return jobs


# ============================================================================
# Helpers
# ============================================================================

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def save_sweep_metadata(
    sweep_dir: str, sweep_config: dict, experiment_configs: list[tuple[str, dict]]
) -> None:
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

    configs_dir = os.path.join(sweep_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    for exp_name, config in experiment_configs:
        config_path = os.path.join(configs_dir, f"{exp_name}_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Sweep metadata saved to: {metadata_path}")


def generate_sweep_summary(
    sweep_dir: str, all_results: list[dict[str, Any]]
) -> None:
    exp_results: dict[str, list] = {}
    for result in all_results:
        exp_name = result["experiment_name"]
        if exp_name not in exp_results:
            exp_results[exp_name] = []
        exp_results[exp_name].append(result)

    total_runs = len(all_results)
    total_successful = len([r for r in all_results if r["status"] == "completed"])
    total_runtime = sum(r.get("runtime_seconds", 0) for r in all_results)

    experiment_stats = {}
    for exp_name, results in exp_results.items():
        successful = [r for r in results if r["status"] == "completed"]
        val_losses = [
            r["final_val_loss"]
            for r in successful
            if r.get("final_val_loss") is not None
        ]
        runtimes = [r.get("runtime_seconds", 0) for r in results]

        stats: dict[str, Any] = {
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

    summary_path = os.path.join(sweep_dir, "sweep_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(sweep_summary, f, default_flow_style=False, indent=2)

    print(f"\n{'=' * 80}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total experiments: {len(exp_results)}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {total_successful}/{total_runs}")
    print(f"Overall success rate: {sweep_summary['overall_success_rate']:.2%}")
    print(f"Total runtime: {_format_duration(total_runtime)}")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary: {summary_path}")

    print("\nPer-experiment results:")
    for exp_name, stats in experiment_stats.items():
        runtime_str = stats.get("total_runtime_str", "?")
        status_str = f"{stats['successful_runs']}/{stats['total_runs']} successful"
        if "validation_loss_stats" in stats:
            loss_str = f"val_loss: {stats['validation_loss_stats']['mean']:.4f} +/- {stats['validation_loss_stats']['std']:.4f}"
            print(f"  {exp_name}: {status_str}, {loss_str}, runtime: {runtime_str}")
        else:
            print(f"  {exp_name}: {status_str}, runtime: {runtime_str}")


# ============================================================================
# Subprocess execution (always spawns run_job.py)
# ============================================================================

def run_job_subprocess(job: Job, gpu_id: int | None) -> dict[str, Any]:
    """Run a single job in a subprocess for complete isolation.

    ``gpu_id=None`` runs on CPU (used for local smoke sweeps and CI); otherwise the
    job is pinned to the given GPU via CUDA_VISIBLE_DEVICES.
    """
    os.makedirs(job.seed_dir, exist_ok=True)
    use_cpu = gpu_id is None

    run_config = copy.deepcopy(job.config)
    run_config["seed"] = job.seed
    run_config["sweep_dir"] = job.sweep_dir
    run_config["experiment_name"] = job.exp_name
    run_config["devices"] = 1 if use_cpu else [0]
    run_config["accelerator"] = "cpu" if use_cpu else "gpu"

    config_file = os.path.join(job.seed_dir, "run_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(script_dir, "run_job.py"),
        "--config", config_file,
    ]

    env = os.environ.copy()
    if not use_cpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tag = f"[{'CPU' if use_cpu else f'GPU {gpu_id}'}] {job.job_id}"
    print(f"{tag} — starting")

    start_time = time.time()
    PROGRESS_KEYWORDS = {"val_accuracy"}

    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd=script_dir,
        )

        output_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if any(kw in line for kw in PROGRESS_KEYWORDS):
                print(f"{tag} | {line}", flush=True)
            output_lines.append(line)

        proc.wait()
        elapsed = time.time() - start_time
        elapsed_str = _format_duration(elapsed)

        if proc.returncode == 0:
            result_file = os.path.join(job.seed_dir, "job_result.yaml")
            if os.path.exists(result_file):
                with open(result_file) as f:
                    job_result = yaml.safe_load(f)
            else:
                job_result = {"final_val_loss": None}

            print(f"{tag} — completed in {elapsed_str}")
            return {
                "experiment_name": job.exp_name,
                "seed": job.seed,
                "status": "completed",
                "seed_dir": job.seed_dir,
                "final_val_loss": job_result.get("final_val_loss"),
                "runtime_seconds": elapsed,
                "runtime_str": elapsed_str,
                "completed_at": datetime.datetime.now().isoformat(),
            }
        else:
            combined = "\n".join(output_lines[-20:])
            print(f"{tag} — failed after {elapsed_str}")
            print(f"  last output: {combined[:500]}")
            return {
                "experiment_name": job.exp_name,
                "seed": job.seed,
                "status": "failed",
                "error": combined[:1000],
                "runtime_seconds": elapsed,
                "failed_at": datetime.datetime.now().isoformat(),
            }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"{tag} — exception: {e}")
        return {
            "experiment_name": job.exp_name,
            "seed": job.seed,
            "status": "failed",
            "error": str(e),
            "runtime_seconds": elapsed,
            "failed_at": datetime.datetime.now().isoformat(),
        }


# ============================================================================
# GPU scheduler
# ============================================================================

class GPUScheduler:
    def __init__(self, gpu_ids: list[int]):
        self.gpu_ids = gpu_ids
        self.n_gpus = len(gpu_ids)

    def run_jobs_parallel(self, jobs: list[Job]) -> list[dict[str, Any]]:
        results = []
        if self.n_gpus == 0:
            print("No GPUs specified, running sequentially on CPU")
            for job in jobs:
                results.append(run_job_subprocess(job, gpu_id=None))
            return results

        print(f"\nScheduling {len(jobs)} jobs across {self.n_gpus} GPUs: {self.gpu_ids}")
        with ProcessPoolExecutor(max_workers=self.n_gpus) as executor:
            future_to_job = {}
            for i, job in enumerate(jobs):
                gpu_id = self.gpu_ids[i % self.n_gpus]
                future = executor.submit(run_job_subprocess, job, gpu_id)
                future_to_job[future] = job
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    results.append(future.result())
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
# Sweep entry point
# ============================================================================

def _skipped_result(job: Job, marker: dict) -> dict[str, Any]:
    """Build a summary-shaped result for a run skipped via its completion marker."""
    return {
        "experiment_name": job.exp_name,
        "seed": job.seed,
        "status": "completed",
        "seed_dir": job.seed_dir,
        "final_val_loss": marker.get("final_val_loss"),
        "runtime_seconds": 0,
        "runtime_str": "resumed",
        "completed_at": marker.get("completed_at"),
    }


def run_parameter_sweep(
    sweep_file: str, gpu_ids: list[int] | None = None, resume: bool = True
):
    print(f"Loading sweep: {sweep_file}")

    sweep_config = load_sweep_config(sweep_file)
    n_seeds = sweep_config["n_seeds"]
    experiment_configs = generate_experiment_configs(sweep_config)

    if gpu_ids is None:
        gpu_ids = sweep_config.get("gpus", [0])
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]

    print(f"  Experiments: {len(experiment_configs)}")
    print(f"  Seeds: {n_seeds}  |  Total runs: {len(experiment_configs) * n_seeds}")
    print(f"  GPUs: {gpu_ids}")

    # Deterministic sweep dir (no timestamp) so re-invoking the same sweep reuses
    # the same tree and can skip runs that already finished.
    log_dir_local = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    sweep_name = os.path.splitext(os.path.basename(sweep_file))[0]
    sweep_dir = os.path.join(log_dir_local, "experiments", sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"  Sweep dir: {sweep_dir}  (resume={resume})")

    save_sweep_metadata(sweep_dir, sweep_config, experiment_configs)

    seeds = list(range(n_seeds))
    seeds_outermost = sweep_config.get("seeds_outermost", True)
    jobs = create_jobs(experiment_configs, seeds, sweep_dir,
                       seeds_outermost=seeds_outermost)

    # Resume: partition into already-complete (skip, but keep for the summary) and
    # pending. A run counts as complete only if its marker's fingerprint matches
    # the job about to run, so a changed config forces a fresh run.
    skipped_results: list[dict[str, Any]] = []
    if resume:
        pending = []
        for job in jobs:
            fp = run_ids.config_fingerprint(job.config)
            marker = run_ids.load_marker(job.seed_dir) if run_ids.is_complete(
                job.seed_dir, fp
            ) else None
            if marker is not None:
                skipped_results.append(_skipped_result(job, marker))
            else:
                pending.append(job)
        jobs = pending
        if skipped_results:
            print(f"  Resume: {len(skipped_results)} already complete, "
                  f"{len(jobs)} to run.")
    else:
        for job in jobs:
            job.config["force_rerun"] = True  # override single_seed's own skip guard

    if not jobs:
        print("  Nothing to run — all runs already complete.")
        generate_sweep_summary(sweep_dir, skipped_results)
        return

    scheduler = GPUScheduler(gpu_ids)  # empty list => sequential CPU
    all_results = scheduler.run_jobs_parallel(jobs)

    generate_sweep_summary(sweep_dir, skipped_results + all_results)


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep (any architecture)")
    parser.add_argument("--sweep", type=str, required=True)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (no GPUs).")
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Re-run every job even if a completed marker already exists.",
    )
    args = parser.parse_args()

    gpu_ids = None
    if args.cpu:
        gpu_ids = []
    elif args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

    run_parameter_sweep(args.sweep, gpu_ids=gpu_ids, resume=args.resume)


if __name__ == "__main__":
    main()
