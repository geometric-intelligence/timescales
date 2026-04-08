#!/usr/bin/env python3
"""
Parameter sweep runner for the Coupled RNN architecture.

Reuses the sweep infrastructure from run_sweep.py (config loading,
grid generation, metadata, summary) but launches coupled_single_job.py
in each subprocess instead of run_single_job.py.

Usage:
    python coupled_sweep.py --sweep sweep_configs/my_coupled_sweep.yaml --gpus 0,1,2,3
"""

import os
import sys
import copy
import yaml
import argparse
import datetime
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_sweep import (
    load_sweep_config,
    generate_experiment_configs,
    create_jobs,
    Job,
    save_sweep_metadata,
    generate_sweep_summary,
    _format_duration,
)


# ============================================================================
# Subprocess execution (points to coupled_single_job.py)
# ============================================================================

def run_job_subprocess(job: Job, gpu_id: int) -> dict[str, Any]:
    """Run a single coupled-RNN job in an isolated subprocess."""
    os.makedirs(job.seed_dir, exist_ok=True)

    run_config = copy.deepcopy(job.config)
    run_config["seed"] = job.seed
    run_config["sweep_dir"] = job.sweep_dir
    run_config["experiment_name"] = job.exp_name
    run_config["devices"] = [0]
    run_config["accelerator"] = "gpu"

    config_file = os.path.join(job.seed_dir, "run_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(script_dir, "coupled_single_job.py"),
        "--config", config_file,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tag = f"[GPU {gpu_id}] {job.job_id}"
    print(f"{tag} — starting")

    start_time = time.time()
    PROGRESS_KEYWORDS = {"val_accuracy"}

    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd=script_dir,
        )

        stderr_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if any(kw in line for kw in PROGRESS_KEYWORDS):
                print(f"{tag} | {line}", flush=True)
            stderr_lines.append(line)

        proc.wait()
        elapsed = time.time() - start_time
        elapsed_str = _format_duration(elapsed)

        if proc.returncode == 0:
            result_file = os.path.join(job.seed_dir, "job_result.yaml")
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
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
            combined = "\n".join(stderr_lines[-20:])
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
# GPU scheduler (mirrors run_sweep.GPUScheduler)
# ============================================================================

class GPUScheduler:
    def __init__(self, gpu_ids: list[int]):
        self.gpu_ids = gpu_ids
        self.n_gpus = len(gpu_ids)

    def run_jobs_parallel(self, jobs: list[Job]) -> list[dict[str, Any]]:
        results = []
        if self.n_gpus == 0:
            for job in jobs:
                results.append(run_job_subprocess(job, gpu_id=0))
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
# Main
# ============================================================================

def run_parameter_sweep(sweep_file: str, gpu_ids: list[int] | None = None):
    print(f"Loading coupled-RNN sweep: {sweep_file}")

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

    log_dir_local = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    sweep_name = os.path.splitext(os.path.basename(sweep_file))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(log_dir_local, "experiments", f"{sweep_name}_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"  Sweep dir: {sweep_dir}")

    save_sweep_metadata(sweep_dir, sweep_config, experiment_configs)

    seeds = list(range(n_seeds))
    jobs = create_jobs(experiment_configs, seeds, sweep_dir)

    scheduler = GPUScheduler(gpu_ids or [0])
    all_results = scheduler.run_jobs_parallel(jobs)

    generate_sweep_summary(sweep_dir, all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Run coupled-RNN parameter sweep with multi-GPU scheduling",
    )
    parser.add_argument("--sweep", type=str, required=True, help="Sweep config YAML")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

    run_parameter_sweep(args.sweep, gpu_ids=gpu_ids)


if __name__ == "__main__":
    main()
