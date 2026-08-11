#!/usr/bin/env python3
"""Track sine-output-coupled oscillatory Jacobian modes across checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from timescales.mode_analysis import pair_indices

from analyze_forced_pulse_checkpoints import _load_model, _model_linearization
from analyze_linearity_over_training import _checkpoint_steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-step", type=int, default=400)
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _orthonormal_basis(eigvecs: np.ndarray, indices: list[int]) -> np.ndarray:
    basis, _ = np.linalg.qr(eigvecs[:, indices])
    return basis


def _oscillation_period(eigvals: np.ndarray, indices: list[int]) -> float | None:
    theta = float(np.mean(np.abs(np.angle(eigvals[indices]))))
    if theta <= 1e-10:
        return None
    return float(2.0 * np.pi / theta)


def _frequency_coupling(
    readout: np.ndarray,
    eigvecs: np.ndarray,
    groups: list[list[int]],
    n_pairs: int,
) -> np.ndarray:
    """Frobenius coupling between each sin/cos output pair and eigenmode pair."""
    raw = np.abs(readout @ eigvecs)
    per_output_mode = np.stack(
        [np.sqrt(np.sum(raw[:, group] ** 2, axis=1)) for group in groups], axis=1
    )
    return np.stack(
        [
            np.sqrt(
                per_output_mode[2 * frequency] ** 2
                + per_output_mode[2 * frequency + 1] ** 2
            )
            for frequency in range(n_pairs)
        ],
        axis=0,
    )


def _mode_row(
    *,
    frequency: int,
    group_index: int,
    spectrum: dict,
    task_period: float,
    overlap: float | None = None,
) -> dict:
    indices = spectrum["groups"][group_index]
    eigenvalues = spectrum["eigvals"][indices]
    magnitudes = np.abs(eigenvalues)
    period = _oscillation_period(spectrum["eigvals"], indices)
    row = {
        "frequency": frequency,
        "output_indices": [2 * frequency, 2 * frequency + 1],
        "task_timescale_steps": task_period,
        "mode_group_index": int(group_index),
        "mode_indices": [int(value) for value in indices],
        "eigenvalue_magnitude": float(np.mean(magnitudes)),
        "max_eigenvalue_magnitude": float(np.max(magnitudes)),
        "characteristic_timescale_steps": period,
        "regime": "decay" if np.all(magnitudes < 1.0) else "growth",
        "coupling": float(spectrum["coupling"][frequency, group_index]),
    }
    if overlap is not None:
        row["adjacent_eigensubspace_overlap"] = overlap
    return row


def _match_adjacent_groups(
    previous_spectrum: dict,
    previous_groups: np.ndarray,
    current_spectrum: dict,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_groups = current_spectrum["oscillatory_groups"]
    overlaps = np.empty((previous_groups.size, candidate_groups.size), dtype=float)
    for row, group_index in enumerate(previous_groups):
        previous_basis = previous_spectrum["bases"][int(group_index)]
        for column, candidate in enumerate(candidate_groups):
            current_basis = current_spectrum["bases"][int(candidate)]
            dimension = min(previous_basis.shape[1], current_basis.shape[1])
            overlaps[row, column] = float(
                np.linalg.norm(previous_basis.conj().T @ current_basis, ord="fro")
                / np.sqrt(dimension)
            )
    # Track each output's reference mode independently. Multiple output pairs
    # may legitimately select the same recurrent mode, so a one-to-one
    # assignment across outputs would spuriously split those tracks.
    columns = np.argmax(overlaps, axis=1)
    matched = candidate_groups[columns]
    quality = overlaps[np.arange(previous_groups.size), columns]
    return matched, quality


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    out_file = (
        args.out_file.resolve()
        if args.out_file is not None
        else run_dir / "analysis" / "coupled_modes_over_training.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with (run_dir / "run_config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    if config.get("task") != "sine_wave" or config.get("activation") != "Tanh":
        raise ValueError("This analysis expects the tanh sine-wave run")
    config.setdefault("input_size", 1)
    config.setdefault("output_size", 2 * int(config["n_pairs"]))
    if config.get("dynamics_type") != "voltage":
        raise ValueError("This analysis assumes voltage dynamics")
    if config.get("use_biases", True):
        raise ValueError("The origin is not a fixed point when biases are enabled")

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    checkpoint_specs = _checkpoint_steps(run_dir, config)
    steps = [step for step, _ in checkpoint_specs]
    if args.reference_step not in steps:
        raise ValueError(
            f"Reference step {args.reference_step} is not a saved checkpoint; "
            f"available steps: {steps}"
        )

    n_pairs = int(config["n_pairs"])
    task_periods = [float(period) / float(config["dt"]) for period in config["periods"]]
    if len(task_periods) != n_pairs:
        raise ValueError("n_pairs must match the number of configured periods")

    spectra = []
    print(f"device={device} checkpoints={len(checkpoint_specs)}", flush=True)
    for index, (step, path) in enumerate(checkpoint_specs, start=1):
        print(f"[{index:02d}/{len(checkpoint_specs):02d}] step={step} {path.name}", flush=True)
        model = _load_model(config, path, device)
        linearization = _model_linearization(model)
        eigvals, eigvecs = np.linalg.eig(linearization["discrete_jacobian"])
        eigvecs /= np.maximum(
            np.linalg.norm(eigvecs, axis=0, keepdims=True), np.finfo(float).tiny
        )
        groups = pair_indices(eigvals)
        periods = np.asarray(
            [
                np.nan if (period := _oscillation_period(eigvals, group)) is None else period
                for group in groups
            ]
        )
        oscillatory_groups = np.flatnonzero(np.isfinite(periods)).astype(np.int64)
        if oscillatory_groups.size < n_pairs:
            raise ValueError("Too few oscillatory eigenmode pairs for sine outputs")
        coupling = _frequency_coupling(
            linearization["readout"], eigvecs, groups, n_pairs
        )
        instantaneous = oscillatory_groups[
            np.argmax(coupling[:, oscillatory_groups], axis=1)
        ]
        bases = [_orthonormal_basis(eigvecs, group) for group in groups]
        spectra.append(
            {
                "step": step,
                "checkpoint": path.name,
                "eigvals": eigvals,
                "groups": groups,
                "oscillatory_groups": oscillatory_groups,
                "bases": bases,
                "coupling": coupling,
                "instantaneous": instantaneous,
            }
        )
        print(
            "  instantaneous_groups="
            + ",".join(str(value) for value in instantaneous),
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reference_index = steps.index(args.reference_step)
    tracked_groups: list[np.ndarray | None] = [None] * len(spectra)
    tracked_overlaps: list[np.ndarray | None] = [None] * len(spectra)
    tracked_groups[reference_index] = spectra[reference_index]["instantaneous"].copy()
    tracked_overlaps[reference_index] = np.ones(n_pairs, dtype=np.float64)

    for index in range(reference_index + 1, len(spectra)):
        matched, quality = _match_adjacent_groups(
            spectra[index - 1], tracked_groups[index - 1], spectra[index]
        )
        tracked_groups[index] = matched
        tracked_overlaps[index] = quality
    for index in range(reference_index - 1, -1, -1):
        matched, quality = _match_adjacent_groups(
            spectra[index + 1], tracked_groups[index + 1], spectra[index]
        )
        tracked_groups[index] = matched
        tracked_overlaps[index] = quality

    rows = []
    for index, spectrum in enumerate(spectra):
        rows.append(
            {
                "step": spectrum["step"],
                "checkpoint": spectrum["checkpoint"],
                "instantaneous": [
                    _mode_row(
                        frequency=frequency,
                        group_index=int(spectrum["instantaneous"][frequency]),
                        spectrum=spectrum,
                        task_period=task_periods[frequency],
                    )
                    for frequency in range(n_pairs)
                ],
                "tracked_from_reference": [
                    _mode_row(
                        frequency=frequency,
                        group_index=int(tracked_groups[index][frequency]),
                        spectrum=spectrum,
                        task_period=task_periods[frequency],
                        overlap=float(tracked_overlaps[index][frequency]),
                    )
                    for frequency in range(n_pairs)
                ],
            }
        )

    result = {
        "run_dir": str(run_dir),
        "task": config["task"],
        "activation": config["activation"],
        "reference_step": args.reference_step,
        "task_timescales_steps": task_periods,
        "timescale_definition": "2*pi/abs(arg(lambda)) discrete steps",
        "regime_definition": "decay for every eigenvalue magnitude < 1; growth otherwise",
        "instantaneous_definition": "argmax pair-level Frobenius coupling between each sin/cos output pair and each conjugate Jacobian eigenmode pair",
        "tracked_definition": "select by coupling at the reference checkpoint, then track each output independently between adjacent checkpoints by maximum eigensubspace overlap",
        "rows": rows,
    }
    with out_file.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved {out_file}", flush=True)


if __name__ == "__main__":
    main()
