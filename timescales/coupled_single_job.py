#!/usr/bin/env python3
"""
Single job runner for coupled-RNN sweep execution.

Called by coupled_sweep.py in a subprocess — mirrors run_single_job.py
but imports from coupled_run instead of single_run.

Usage (called by coupled_sweep.py, not directly):
    python coupled_single_job.py --config /path/to/run_config.yaml
"""

import os
import sys
import yaml
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coupled_run import single_seed


def main():
    parser = argparse.ArgumentParser(description="Run a single coupled-RNN training job")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to job configuration YAML file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    try:
        result = single_seed(config)

        seed_dir = os.path.join(
            config["sweep_dir"],
            config["experiment_name"],
            f"seed_{config['seed']}",
        )
        result_file = os.path.join(seed_dir, "job_result.yaml")
        with open(result_file, "w") as f:
            yaml.dump(result, f, default_flow_style=False)

        print(f"Job completed. Result saved to {result_file}")
        sys.exit(0)

    except Exception as e:
        print(f"Job failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
