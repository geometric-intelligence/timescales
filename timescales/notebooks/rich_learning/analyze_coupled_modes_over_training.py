#!/usr/bin/env python3
"""Track output-coupled Jacobian eigenmodes across training checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from analyze_forced_pulse_checkpoints import _load_model, _model_linearization
from analyze_linearity_over_training import _checkpoint_steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-step", type=int, default=400)
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _characteristic_timescale(eigenvalue: complex) -> tuple[float | None, str]:
    """Return the positive e-folding time 1 / |log |lambda|| in steps.

    The regime records whether this is a decay time (|lambda| < 1) or a growth
    time (|lambda| > 1).  This lets the same positive log-scale plot remain
    readable when a mode crosses the unit circle.
    """
    magnitude = abs(eigenvalue)
    if magnitude <= 0.0:
        return None, "zero"
    log_magnitude = float(np.log(magnitude))
    if abs(log_magnitude) < 1e-12:
        return None, "neutral"
    regime = "decay" if log_magnitude < 0.0 else "growth"
    return float(1.0 / abs(log_magnitude)), regime


def _mode_row(
    *,
    output: int,
    mode_index: int,
    eigvals: np.ndarray,
    coupling: np.ndarray,
    task_timescale: float,
    overlap: float | None = None,
) -> dict:
    eigenvalue = eigvals[mode_index]
    timescale, regime = _characteristic_timescale(eigenvalue)
    row = {
        "output": output,
        "task_timescale_steps": task_timescale,
        "mode_index": int(mode_index),
        "eigenvalue_real": float(eigenvalue.real),
        "eigenvalue_imag": float(eigenvalue.imag),
        "eigenvalue_magnitude": float(abs(eigenvalue)),
        "characteristic_timescale_steps": timescale,
        "regime": regime,
        "coupling": float(coupling[output, mode_index]),
    }
    if overlap is not None:
        row["adjacent_eigenvector_overlap"] = overlap
    return row


def _match_adjacent_modes(
    previous_eigvecs: np.ndarray,
    previous_indices: np.ndarray,
    current_eigvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Continue selected modes by maximum adjacent-checkpoint vector overlap."""
    selected_previous = previous_eigvecs[:, previous_indices]
    overlaps = np.abs(selected_previous.conj().T @ current_eigvecs)
    rows, columns = linear_sum_assignment(-overlaps)
    matched = np.empty(previous_indices.size, dtype=np.int64)
    quality = np.empty(previous_indices.size, dtype=np.float64)
    matched[rows] = columns
    quality[rows] = overlaps[rows, columns]
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
    checkpoint_specs = _checkpoint_steps(run_dir, config)
    steps = [step for step, _ in checkpoint_specs]
    if args.reference_step not in steps:
        raise ValueError(
            f"Reference step {args.reference_step} is not a saved checkpoint; "
            f"available steps: {steps}"
        )
    task_timescales = [1.0 / float(p) for p in config["p_pulse"]]

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
        coupling = np.abs(linearization["readout"] @ eigvecs)
        instantaneous = np.argmax(coupling, axis=1).astype(np.int64)
        spectra.append(
            {
                "step": step,
                "checkpoint": path.name,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
                "coupling": coupling,
                "instantaneous": instantaneous,
            }
        )
        print(
            "  instantaneous=" + ",".join(str(value) for value in instantaneous),
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reference_index = steps.index(args.reference_step)
    tracked_indices: list[np.ndarray | None] = [None] * len(spectra)
    tracked_overlaps: list[np.ndarray | None] = [None] * len(spectra)
    tracked_indices[reference_index] = spectra[reference_index]["instantaneous"].copy()
    tracked_overlaps[reference_index] = np.ones(len(task_timescales), dtype=np.float64)

    for index in range(reference_index + 1, len(spectra)):
        matched, quality = _match_adjacent_modes(
            spectra[index - 1]["eigvecs"],
            tracked_indices[index - 1],
            spectra[index]["eigvecs"],
        )
        tracked_indices[index] = matched
        tracked_overlaps[index] = quality
    for index in range(reference_index - 1, -1, -1):
        matched, quality = _match_adjacent_modes(
            spectra[index + 1]["eigvecs"],
            tracked_indices[index + 1],
            spectra[index]["eigvecs"],
        )
        tracked_indices[index] = matched
        tracked_overlaps[index] = quality

    rows = []
    for index, spectrum in enumerate(spectra):
        instantaneous_rows = [
            _mode_row(
                output=output,
                mode_index=int(spectrum["instantaneous"][output]),
                eigvals=spectrum["eigvals"],
                coupling=spectrum["coupling"],
                task_timescale=task_timescales[output],
            )
            for output in range(len(task_timescales))
        ]
        tracked_rows = [
            _mode_row(
                output=output,
                mode_index=int(tracked_indices[index][output]),
                eigvals=spectrum["eigvals"],
                coupling=spectrum["coupling"],
                task_timescale=task_timescales[output],
                overlap=float(tracked_overlaps[index][output]),
            )
            for output in range(len(task_timescales))
        ]
        rows.append(
            {
                "step": spectrum["step"],
                "checkpoint": spectrum["checkpoint"],
                "instantaneous": instantaneous_rows,
                "tracked_from_reference": tracked_rows,
            }
        )

    result = {
        "run_dir": str(run_dir),
        "activation": config["activation"],
        "reference_step": args.reference_step,
        "task_timescales_steps": task_timescales,
        "timescale_definition": "1 / abs(log(abs(lambda))) discrete steps",
        "regime_definition": "decay for abs(lambda)<1; growth for abs(lambda)>1",
        "instantaneous_definition": "argmax_j abs((W_out V)_ij) independently at each checkpoint",
        "tracked_definition": "select by coupling at the reference checkpoint, then continue between adjacent checkpoints by maximum normalized right-eigenvector overlap with one-to-one assignment",
        "rows": rows,
    }
    with out_file.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved {out_file}", flush=True)


if __name__ == "__main__":
    main()
