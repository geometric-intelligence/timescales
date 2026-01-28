#!/usr/bin/env python3
"""
Single job runner for subprocess-based sweep execution.

This script is called by run_sweep.py to run a single training job
in an isolated subprocess. This ensures clean GPU allocation and 
proper WandB run separation.

Usage (called by run_sweep.py, not directly):
    python run_single_job.py --config /path/to/run_config.yaml
"""

import os
import sys
import yaml
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from single_run import single_seed


def main():
    parser = argparse.ArgumentParser(description="Run a single training job")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to job configuration YAML file"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Run training
    try:
        result = single_seed(config)
        
        # Save result for the parent process to read
        seed_dir = os.path.join(
            config["sweep_dir"], 
            config["experiment_name"], 
            f"seed_{config['seed']}"
        )
        result_file = os.path.join(seed_dir, "job_result.yaml")
        with open(result_file, "w") as f:
            yaml.dump(result, f, default_flow_style=False)
        
        print(f"Job completed successfully. Result saved to {result_file}")
        sys.exit(0)
        
    except Exception as e:
        print(f"Job failed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

