#!/usr/bin/env python3
"""Compare an intermediate and final tanh flip-flop checkpoint.

The analysis is tailored to the voltage RNN used by the rich-learning runs.
It computes:

1. The spectrum of the zero-input discrete map Jacobian at the origin.
2. A pooled PCA basis for hidden activity from both checkpoints.
3. Output-matched Jacobian eigenmodes using ``|W_out @ V|``.
4. Full-state nonlinear-field and Jacobian-residual metrics.

Trajectory statistics use many independently generated forced-pulse trials.
Compact arrays for interactive visualization are saved separately from the
JSON summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA


def _configure_import_path() -> None:
    here = Path.cwd().resolve()
    script = Path(__file__).resolve()
    candidates = [here, here.parent, script.parents[2], script.parents[3]]
    for candidate in candidates:
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


_configure_import_path()

from train import _create_rnn_model  # noqa: E402
from timescales.datamodules.flip_flop import (  # noqa: E402
    simulate_flip_flop_trajectories,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-trajectories", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pca-points", type=int, default=30000)
    parser.add_argument("--cloud-points", type=int, default=8000)
    parser.add_argument("--plot-trajectories", type=int, default=8)
    parser.add_argument("--plot-stride", type=int, default=5)
    parser.add_argument("--analysis-seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _checkpoint_path(run_dir: Path, step: int, val_every: int) -> Path:
    if step <= 0 or step % val_every:
        raise ValueError(
            f"checkpoint step must be a positive multiple of {val_every}, got {step}"
        )
    epoch = step // val_every - 1
    matches = sorted((run_dir / "checkpoints").glob(f"checkpoint-*{epoch:03d}.ckpt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one epoch-{epoch:03d} checkpoint, found {matches}"
        )
    checkpoint = torch.load(matches[0], map_location="cpu", weights_only=False)
    if int(checkpoint.get("global_step", -1)) != step:
        raise ValueError(
            f"{matches[0]} contains global_step={checkpoint.get('global_step')}, "
            f"expected {step}"
        )
    return matches[0]


def _load_model(config: dict, path: Path, device: torch.device):
    torch.manual_seed(int(config.get("seed", 0)))
    model, _ = _create_rnn_model(dict(config))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in payload:
        state = {
            key[len("model.") :]: value
            for key, value in payload["state_dict"].items()
            if key.startswith("model.")
        }
    else:
        state = payload
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _model_linearization(model) -> dict[str, np.ndarray | float]:
    step = model.rnn_step
    recurrent = (
        step.recurrent_weight_scale
        * step.W_rec.weight.detach().cpu().double().numpy()
    )
    alpha = step.current_alphas.detach().cpu().double().numpy().reshape(-1)
    tau = step.current_time_constants.detach().cpu().double().numpy().reshape(-1)
    if tau.size == 1:
        tau = np.repeat(tau, model.hidden_size)
    if alpha.size == 1:
        alpha = np.repeat(alpha, model.hidden_size)

    identity = np.eye(model.hidden_size, dtype=np.float64)
    discrete_jacobian = np.diag(1.0 - alpha) + alpha[:, None] * recurrent
    field_jacobian = (1.0 / tau)[:, None] * (-identity + recurrent)
    readout = model.effective_readout_weight.detach().cpu().double().numpy()
    return {
        "recurrent": recurrent,
        "alpha": alpha,
        "tau": tau,
        "discrete_jacobian": discrete_jacobian,
        "field_jacobian": field_jacobian,
        "readout": readout,
    }


def _spectral_analysis(linearization: dict, output_timescales: list[float]) -> dict:
    jacobian = linearization["discrete_jacobian"]
    eigvals, eigvecs = np.linalg.eig(jacobian)
    eigvecs_inv = np.linalg.inv(eigvecs)
    complex_coupling = linearization["readout"] @ eigvecs
    coupling = np.abs(complex_coupling)
    selected = np.argmax(coupling, axis=1)
    magnitudes = np.abs(eigvals)

    selected_rows = []
    for output_index, mode_index in enumerate(selected.tolist()):
        eigenvalue = eigvals[mode_index]
        selected_rows.append(
            {
                "output": output_index,
                "task_timescale_steps": float(output_timescales[output_index]),
                "mode_index": int(mode_index),
                "eigenvalue_real": float(eigenvalue.real),
                "eigenvalue_imag": float(eigenvalue.imag),
                "eigenvalue_magnitude": float(abs(eigenvalue)),
                "coupling": float(coupling[output_index, mode_index]),
                "coupling_phase": float(
                    np.angle(complex_coupling[output_index, mode_index])
                ),
            }
        )

    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "eigvecs_inv": eigvecs_inv,
        "complex_coupling": complex_coupling,
        "selected": selected,
        "summary": {
            "spectral_radius": float(magnitudes.max()),
            "stable_origin": bool(np.all(magnitudes < 1.0)),
            "n_eigenvalues_at_or_outside_unit_circle": int(np.sum(magnitudes >= 1.0)),
            "n_eigenvalues_within_0.01_of_unit_circle": int(
                np.sum(np.abs(magnitudes - 1.0) <= 0.01)
            ),
            "eigenvector_condition_number": float(np.linalg.cond(eigvecs)),
            "n_unique_output_matched_modes": int(np.unique(selected).size),
            "selected_modes": selected_rows,
        },
    }


def _trajectory_inputs(config: dict, n_trajectories: int, seed: int):
    np.random.seed(seed)
    inputs, targets, _ = simulate_flip_flop_trajectories(
        num_trajectories=n_trajectories,
        num_time_steps=int(config["num_time_steps"]),
        n_bits=int(config["n_bits"]),
        p_pulse=config["p_pulse"],
        pulse_amplitude=float(config.get("pulse_amplitude", 1.0)),
        force_initial_pulse=bool(config.get("force_initial_pulse", False)),
    )
    return inputs, targets


def _state_codes(targets: np.ndarray) -> np.ndarray:
    powers = (1 << np.arange(targets.shape[-1], dtype=np.int64)).reshape(1, 1, -1)
    return np.sum(targets.astype(np.int64) * powers, axis=-1)


def _collect_activity(
    model,
    inputs: np.ndarray,
    targets: np.ndarray,
    linearization: dict,
    sample_indices: np.ndarray,
    batch_size: int,
    plot_trajectories: int,
    plot_stride: int,
    device: torch.device,
) -> dict:
    n_trajectories, n_steps, hidden_size = (
        inputs.shape[0],
        inputs.shape[1],
        model.hidden_size,
    )
    recurrent = torch.as_tensor(
        linearization["recurrent"], dtype=torch.float32, device=device
    )
    tau_inv = torch.as_tensor(
        1.0 / linearization["tau"], dtype=torch.float32, device=device
    )
    field_jacobian = linearization["field_jacobian"]
    field_norm = float(np.linalg.norm(field_jacobian, ord="fro"))
    weighted_recurrent = (1.0 / linearization["tau"])[:, None] * linearization[
        "recurrent"
    ]
    column_norm_sq = torch.as_tensor(
        np.sum(weighted_recurrent**2, axis=0),
        dtype=torch.float32,
        device=device,
    )

    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    sampled_states = []
    sampled_codes = []
    plot_states = []
    plot_codes = []
    residual_sq = 0.0
    field_sq = 0.0
    jacobian_difference_sq = 0.0
    n_states = 0

    all_codes = _state_codes(targets)
    for trajectory_start in range(0, n_trajectories, batch_size):
        trajectory_stop = min(n_trajectories, trajectory_start + batch_size)
        input_batch = torch.from_numpy(inputs[trajectory_start:trajectory_stop]).to(device)
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
                residual = torch.nn.functional.linear(phi - states, recurrent) * tau_inv
                delta_derivative = -(phi**2)
                jacobian_fro_sq = (delta_derivative.square() * column_norm_sq).sum(dim=1)
            residual_sq += float(residual.square().sum(dtype=torch.float64).cpu())
            field_sq += float(field.square().sum(dtype=torch.float64).cpu())
            jacobian_difference_sq += float(
                jacobian_fro_sq.sum(dtype=torch.float64).cpu()
            )
            n_states += states.shape[0]

        global_start = trajectory_start * n_steps
        global_stop = trajectory_stop * n_steps
        lo = int(np.searchsorted(sample_indices, global_start, side="left"))
        hi = int(np.searchsorted(sample_indices, global_stop, side="left"))
        if hi > lo:
            local_indices = sample_indices[lo:hi] - global_start
            sampled_states.append(
                flat[torch.as_tensor(local_indices, device=device)].cpu().numpy()
            )
            flat_codes = all_codes[trajectory_start:trajectory_stop].reshape(-1)
            sampled_codes.append(flat_codes[local_indices])

        if trajectory_start < plot_trajectories:
            take = min(trajectory_stop, plot_trajectories) - trajectory_start
            plot_states.append(
                hidden[:take, ::plot_stride].detach().cpu().numpy()
            )
            plot_codes.append(
                all_codes[trajectory_start : trajectory_start + take, ::plot_stride]
            )
        del hidden, flat, input_batch

    return {
        "sampled_states": np.concatenate(sampled_states, axis=0).astype(np.float32),
        "sampled_codes": np.concatenate(sampled_codes, axis=0).astype(np.int8),
        "plot_states": np.concatenate(plot_states, axis=0).astype(np.float32),
        "plot_codes": np.concatenate(plot_codes, axis=0).astype(np.int8),
        "eta_field": math.sqrt(residual_sq / max(field_sq, np.finfo(float).tiny)),
        "eta_jacobian": (
            math.sqrt(jacobian_difference_sq / n_states)
            / max(field_norm, np.finfo(float).tiny)
        ),
        "n_states": n_states,
    }


def _phase_aligned_selected_coordinates(
    states: np.ndarray,
    spectral: dict,
) -> np.ndarray:
    flat = states.reshape(-1, states.shape[-1]).astype(np.float64)
    selected = spectral["selected"]
    inverse_rows = spectral["eigvecs_inv"][selected, :]
    modal = flat @ inverse_rows.T
    phases = np.asarray(
        [
            spectral["complex_coupling"][output, mode]
            / max(abs(spectral["complex_coupling"][output, mode]), np.finfo(float).tiny)
            for output, mode in enumerate(selected)
        ]
    )
    coordinates = np.real(modal * phases[None, :])
    return coordinates.reshape(*states.shape[:-1], len(selected)).astype(np.float32)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else run_dir / "analysis" / f"checkpoint_{args.checkpoint_step}_vs_final"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "run_config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    if config.get("task") != "flip_flop" or config.get("activation") != "Tanh":
        raise ValueError("This analysis expects the tanh flip-flop rich-learning run")
    if config.get("dynamics_type") != "voltage":
        raise ValueError("This analysis currently assumes voltage dynamics")
    if config.get("use_biases", True):
        raise ValueError("The origin is not necessarily a fixed point when biases are enabled")

    val_every = int(config.get("val_every_n_steps", 10))
    intermediate_path = _checkpoint_path(run_dir, args.checkpoint_step, val_every)
    final_path = run_dir / f"final_model_seed{int(config.get('seed', 0))}.pth"
    if not final_path.exists():
        raise FileNotFoundError(final_path)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    print(f"device={device} | output={out_dir}", flush=True)

    inputs, targets = _trajectory_inputs(
        config, args.n_trajectories, args.analysis_seed
    )
    total_states = inputs.shape[0] * inputs.shape[1]
    rng = np.random.default_rng(args.analysis_seed)
    n_sample = min(args.pca_points, total_states)
    sample_indices = np.sort(rng.choice(total_states, size=n_sample, replace=False))
    output_timescales = [1.0 / float(p) for p in config["p_pulse"]]

    checkpoint_specs = [
        (f"step_{args.checkpoint_step}", intermediate_path, args.checkpoint_step),
        ("final", final_path, int(config["max_steps"])),
    ]
    analyses = {}
    for tag, path, step in checkpoint_specs:
        print(f"{tag}: loading {path.name}", flush=True)
        model = _load_model(config, path, device)
        linearization = _model_linearization(model)
        spectral = _spectral_analysis(linearization, output_timescales)
        activity = _collect_activity(
            model=model,
            inputs=inputs,
            targets=targets,
            linearization=linearization,
            sample_indices=sample_indices,
            batch_size=args.batch_size,
            plot_trajectories=args.plot_trajectories,
            plot_stride=args.plot_stride,
            device=device,
        )
        analyses[tag] = {
            "step": step,
            "path": str(path),
            "linearization": linearization,
            "spectral": spectral,
            "activity": activity,
        }
        summary = spectral["summary"]
        print(
            f"  rho={summary['spectral_radius']:.6f} "
            f"stable={summary['stable_origin']} "
            f"eta_field={activity['eta_field']:.6f} "
            f"eta_J={activity['eta_jacobian']:.6f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pooled = np.concatenate(
        [analyses[tag]["activity"]["sampled_states"] for tag, _, _ in checkpoint_specs],
        axis=0,
    )
    pca = PCA(n_components=10, svd_solver="randomized", random_state=args.analysis_seed)
    pca.fit(pooled)

    output_arrays = {
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "sample_state_codes": analyses[checkpoint_specs[0][0]]["activity"][
            "sampled_codes"
        ][: args.cloud_points],
        "trajectory_state_codes": analyses[checkpoint_specs[0][0]]["activity"][
            "plot_codes"
        ],
    }
    summary = {
        "run_dir": str(run_dir),
        "config": {
            "hidden_size": int(config["hidden_size"]),
            "dt": float(config["dt"]),
            "p_pulse": [float(value) for value in config["p_pulse"]],
            "task_timescales_steps": output_timescales,
            "num_time_steps": int(config["num_time_steps"]),
            "force_initial_pulse": bool(config.get("force_initial_pulse", False)),
        },
        "sampling": {
            "n_trajectories": args.n_trajectories,
            "n_states_per_checkpoint": total_states,
            "pca_points_per_checkpoint": n_sample,
            "cloud_points": min(args.cloud_points, n_sample),
            "plot_trajectories": args.plot_trajectories,
            "plot_stride": args.plot_stride,
            "analysis_seed": args.analysis_seed,
        },
        "pca": {
            "basis": "pooled across intermediate and final sampled activity",
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_top3": float(
                pca.explained_variance_ratio_[:3].sum()
            ),
        },
        "checkpoints": {},
    }

    for tag, _, _ in checkpoint_specs:
        analysis = analyses[tag]
        activity = analysis["activity"]
        spectral = analysis["spectral"]
        sample = activity["sampled_states"]
        plot_states = activity["plot_states"]
        output_arrays[f"eigvals_{tag}"] = spectral["eigvals"].astype(np.complex64)
        output_arrays[f"selected_modes_{tag}"] = spectral["selected"].astype(np.int32)
        output_arrays[f"coupling_{tag}"] = np.abs(
            spectral["complex_coupling"]
        ).astype(np.float32)
        output_arrays[f"cloud_pca_{tag}"] = pca.transform(
            sample[: args.cloud_points]
        )[:, :3].astype(np.float32)
        output_arrays[f"cloud_eigen_{tag}"] = _phase_aligned_selected_coordinates(
            sample[: args.cloud_points], spectral
        )
        output_arrays[f"trajectory_pca_{tag}"] = pca.transform(
            plot_states.reshape(-1, plot_states.shape[-1])
        )[:, :3].reshape(*plot_states.shape[:-1], 3).astype(np.float32)
        output_arrays[f"trajectory_eigen_{tag}"] = _phase_aligned_selected_coordinates(
            plot_states, spectral
        )
        summary["checkpoints"][tag] = {
            "step": analysis["step"],
            **spectral["summary"],
            "eta_field": float(activity["eta_field"]),
            "eta_jacobian": float(activity["eta_jacobian"]),
        }

    with (out_dir / "summary.json").open("w") as handle:
        json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)
    np.savez_compressed(out_dir / "geometry.npz", **output_arrays)
    print(f"saved {out_dir / 'summary.json'}", flush=True)
    print(f"saved {out_dir / 'geometry.npz'}", flush=True)


if __name__ == "__main__":
    main()
