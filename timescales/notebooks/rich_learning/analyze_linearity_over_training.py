#!/usr/bin/env python3
"""Measure nonlinear-field and Jacobian residuals across saved checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from analyze_forced_pulse_checkpoints import (
    _load_model,
    _model_linearization,
    _trajectory_inputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--n-trajectories", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--analysis-seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca-geometry", type=Path, default=None)
    parser.add_argument("--pca-components", type=int, default=3)
    return parser.parse_args()


def _checkpoint_priority(path: Path) -> int:
    if path.suffix == ".pth":
        return 4
    if path.name.startswith("checkpoint-"):
        return 3
    if path.name.startswith("best-model-"):
        return 2
    if path.name == "last.ckpt":
        return 1
    return 0


def _checkpoint_steps(run_dir: Path, config: dict) -> list[tuple[int, Path]]:
    by_step: dict[int, Path] = {}
    for path in sorted((run_dir / "checkpoints").glob("*.ckpt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        step = int(payload.get("global_step", 0))
        current = by_step.get(step)
        if current is None or _checkpoint_priority(path) > _checkpoint_priority(current):
            by_step[step] = path
        del payload

    final_path = run_dir / f"final_model_seed{int(config.get('seed', 0))}.pth"
    if final_path.exists():
        by_step[int(config["max_steps"])] = final_path
    return sorted(by_step.items())


def _linearity_metrics(
    model,
    inputs: np.ndarray,
    linearization: dict,
    batch_size: int,
    device: torch.device,
    pca_basis: np.ndarray | None = None,
) -> dict[str, float | int]:
    recurrent = torch.as_tensor(
        linearization["recurrent"], dtype=torch.float32, device=device
    )
    tau_inv = torch.as_tensor(
        1.0 / linearization["tau"], dtype=torch.float32, device=device
    )
    field_norm = float(np.linalg.norm(linearization["field_jacobian"], ord="fro"))
    weighted_recurrent = (1.0 / linearization["tau"])[:, None] * linearization[
        "recurrent"
    ]
    column_norm_sq = torch.as_tensor(
        np.sum(weighted_recurrent**2, axis=0),
        dtype=torch.float32,
        device=device,
    )

    basis = None
    reduced_jacobian_norm = None
    reduced_weighted_recurrent = None
    if pca_basis is not None:
        basis = torch.as_tensor(pca_basis, dtype=torch.float32, device=device)
        reduced_jacobian = pca_basis.T @ linearization["field_jacobian"] @ pca_basis
        reduced_jacobian_norm = float(np.linalg.norm(reduced_jacobian, ord="fro"))
        reduced_weighted_recurrent = basis.T @ torch.as_tensor(
            weighted_recurrent, dtype=torch.float32, device=device
        )

    residual_sq = 0.0
    field_sq = 0.0
    jacobian_difference_sq = 0.0
    projected_residual_sq = 0.0
    projected_field_sq = 0.0
    projected_jacobian_difference_sq = 0.0
    n_states = 0
    hidden_size = model.hidden_size
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(inputs.shape[0], start + batch_size)
        input_batch = torch.from_numpy(inputs[start:stop]).to(device)
        with torch.inference_mode():
            hidden, _ = model(inputs=input_batch)
        flat = hidden.reshape(-1, hidden_size)
        for state_start in range(0, flat.shape[0], 8192):
            states = flat[state_start : state_start + 8192]
            with torch.inference_mode():
                phi = torch.tanh(states)
                field = (
                    -states + torch.nn.functional.linear(phi, recurrent)
                ) * tau_inv
                residual = (
                    torch.nn.functional.linear(phi - states, recurrent) * tau_inv
                )
                delta_derivative = -(phi**2)
                jacobian_fro_sq = (
                    delta_derivative.square() * column_norm_sq
                ).sum(dim=1)
                if basis is not None:
                    projected_field = field @ basis
                    projected_residual = residual @ basis
                    derivative_times_basis = delta_derivative.unsqueeze(-1) * basis
                    projected_jacobian_difference = torch.matmul(
                        reduced_weighted_recurrent.unsqueeze(0),
                        derivative_times_basis,
                    )
            residual_sq += float(residual.square().sum(dtype=torch.float64).cpu())
            field_sq += float(field.square().sum(dtype=torch.float64).cpu())
            jacobian_difference_sq += float(
                jacobian_fro_sq.sum(dtype=torch.float64).cpu()
            )
            if basis is not None:
                projected_residual_sq += float(
                    projected_residual.square().sum(dtype=torch.float64).cpu()
                )
                projected_field_sq += float(
                    projected_field.square().sum(dtype=torch.float64).cpu()
                )
                projected_jacobian_difference_sq += float(
                    projected_jacobian_difference.square()
                    .sum(dtype=torch.float64)
                    .cpu()
                )
            n_states += states.shape[0]
        del hidden, flat, input_batch

    metrics = {
        "eta_field": math.sqrt(residual_sq / max(field_sq, np.finfo(float).tiny)),
        "eta_jacobian": (
            math.sqrt(jacobian_difference_sq / n_states)
            / max(field_norm, np.finfo(float).tiny)
        ),
        "n_states": n_states,
    }
    if basis is not None:
        metrics.update(
            {
                "eta_field_pca": math.sqrt(
                    projected_residual_sq
                    / max(projected_field_sq, np.finfo(float).tiny)
                ),
                "eta_jacobian_pca": (
                    math.sqrt(projected_jacobian_difference_sq / n_states)
                    / max(reduced_jacobian_norm, np.finfo(float).tiny)
                ),
            }
        )
    return metrics


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    out_file = (
        args.out_file.resolve()
        if args.out_file is not None
        else run_dir / "analysis" / "linearity_over_training.json"
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
        raise ValueError("This analysis assumes an origin fixed point without biases")

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    inputs, _ = _trajectory_inputs(config, args.n_trajectories, args.analysis_seed)
    pca_geometry = (
        args.pca_geometry.resolve()
        if args.pca_geometry is not None
        else run_dir / "analysis" / "checkpoint_500_vs_final" / "geometry.npz"
    )
    with np.load(pca_geometry) as geometry:
        pca_components = geometry["pca_components"].astype(np.float64)
        explained_variance = geometry["pca_explained_variance_ratio"].astype(
            np.float64
        )
    if not 1 <= args.pca_components <= pca_components.shape[0]:
        raise ValueError(
            f"pca-components must be between 1 and {pca_components.shape[0]}"
        )
    pca_basis = pca_components[: args.pca_components].T
    checkpoints = _checkpoint_steps(run_dir, config)
    print(
        f"device={device} checkpoints={len(checkpoints)} "
        f"states/checkpoint={inputs.shape[0] * inputs.shape[1]}",
        flush=True,
    )

    rows = []
    for index, (step, path) in enumerate(checkpoints, start=1):
        print(f"[{index:02d}/{len(checkpoints):02d}] step={step} {path.name}", flush=True)
        model = _load_model(config, path, device)
        linearization = _model_linearization(model)
        metrics = _linearity_metrics(
            model=model,
            inputs=inputs,
            linearization=linearization,
            batch_size=args.batch_size,
            device=device,
            pca_basis=pca_basis,
        )
        row = {"step": step, "checkpoint": path.name, **metrics}
        rows.append(row)
        print(
            f"  eta_field={metrics['eta_field']:.6f} "
            f"eta_J={metrics['eta_jacobian']:.6f} "
            f"eta_field_pca={metrics['eta_field_pca']:.6f} "
            f"eta_J_pca={metrics['eta_jacobian_pca']:.6f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "run_dir": str(run_dir),
        "n_trajectories": args.n_trajectories,
        "num_time_steps": int(inputs.shape[1]),
        "n_states_per_checkpoint": int(inputs.shape[0] * inputs.shape[1]),
        "analysis_seed": args.analysis_seed,
        "pca_geometry": str(pca_geometry),
        "pca_components": args.pca_components,
        "pca_explained_variance": float(
            explained_variance[: args.pca_components].sum()
        ),
        "rows": rows,
    }
    with out_file.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved {out_file}", flush=True)


if __name__ == "__main__":
    main()
