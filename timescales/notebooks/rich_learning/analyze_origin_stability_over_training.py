#!/usr/bin/env python3
"""Track origin stability across every saved recurrent-network checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from analyze_forced_pulse_checkpoints import _load_model, _model_linearization
from analyze_linearity_over_training import _checkpoint_steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, default=None)
    # Full eigendecompositions of these 512 x 512 Jacobians are substantially
    # faster on the CPU in the environments where this analysis is run.
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    out_file = (
        args.out_file.resolve()
        if args.out_file is not None
        else run_dir / "analysis" / "origin_stability_over_training.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with (run_dir / "run_config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    if config.get("task") != "flip_flop" or config.get("activation") != "Tanh":
        raise ValueError("This analysis expects the tanh flip-flop run")
    config.setdefault("input_size", int(config["n_bits"]))
    config.setdefault("output_size", int(config["n_bits"]))
    if config.get("dynamics_type") != "voltage":
        raise ValueError("This analysis assumes voltage dynamics")
    if config.get("use_biases", True):
        raise ValueError("The origin is not a fixed point when biases are enabled")

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    checkpoints = _checkpoint_steps(run_dir, config)
    print(f"device={device} checkpoints={len(checkpoints)}", flush=True)

    rows = []
    for index, (step, path) in enumerate(checkpoints, start=1):
        print(f"[{index:02d}/{len(checkpoints):02d}] step={step} {path.name}", flush=True)
        model = _load_model(config, path, device)
        linearization = _model_linearization(model)
        jacobian = torch.as_tensor(
            linearization["discrete_jacobian"], dtype=torch.float64, device=device
        )
        with torch.inference_mode():
            eigvals = torch.linalg.eigvals(jacobian).cpu().numpy()
        magnitudes = np.abs(eigvals)
        order = np.argsort(magnitudes)[::-1]
        dominant = eigvals[order[0]]
        row = {
            "step": step,
            "checkpoint": path.name,
            "spectral_radius": float(magnitudes[order[0]]),
            "stability_margin": float(1.0 - magnitudes[order[0]]),
            "stable_origin": bool(np.all(magnitudes < 1.0)),
            "n_eigenvalues_at_or_outside_unit_circle": int(
                np.sum(magnitudes >= 1.0)
            ),
            "n_eigenvalues_within_0.01_of_unit_circle": int(
                np.sum(np.abs(magnitudes - 1.0) <= 0.01)
            ),
            "dominant_eigenvalue_real": float(dominant.real),
            "dominant_eigenvalue_imag": float(dominant.imag),
            "largest_eigenvalues": [
                {
                    "real": float(eigvals[mode].real),
                    "imag": float(eigvals[mode].imag),
                    "magnitude": float(magnitudes[mode]),
                }
                for mode in order[:5]
            ],
        }
        rows.append(row)
        print(
            f"  rho={row['spectral_radius']:.8f} "
            f"unstable={row['n_eigenvalues_at_or_outside_unit_circle']}",
            flush=True,
        )
        del model, jacobian
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "run_dir": str(run_dir),
        "activation": config["activation"],
        "criterion": "strictly stable iff all eigenvalues of DF(0) have magnitude < 1",
        "rows": rows,
    }
    with out_file.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved {out_file}", flush=True)


if __name__ == "__main__":
    main()
