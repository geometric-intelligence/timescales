#!/usr/bin/env python3
"""
Single job runner for sweep execution.
Called by sweep.py in a subprocess.

Usage (called by sweep.py, not directly):
    python run_job.py --config /path/to/run_config.yaml
"""

import os
import sys
import yaml
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
repository_root = os.path.dirname(script_dir)
# Prefer the checkout containing this launcher over an editable installation
# that may point at a different worktree.  Keep script_dir available for the
# historical top-level ``from train import ...`` import below.
sys.path.insert(0, repository_root)
if script_dir not in sys.path:
    sys.path.insert(1, script_dir)

from train import single_seed


def main():
    parser = argparse.ArgumentParser(description="Run a single training job")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    try:
        # single_seed() writes the completion marker (job_result.yaml) itself,
        # so it is the single source of truth for resume — no duplicate write here.
        single_seed(config)
        print("Job completed.")
        sys.exit(0)

    except Exception as e:
        print(f"Job failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
