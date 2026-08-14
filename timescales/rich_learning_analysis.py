"""Lightweight descriptors for rich-learning loss curves.

This module intentionally does not collapse "staircaseness" into one score.  A
curve can look step-like because it has one abrupt transition, because task
components are acquired at different times, or simply because it has not
finished training.  The descriptors below keep those possibilities separate:

* latency: steps at 10%, 50%, and 90% of the observed log-loss improvement;
* abruptness: how much of the downward motion occurs in the largest 10% of
  validation-to-validation drops;
* censoring: the fraction of net improvement made in the last 20% of training
  and the terminal log-loss slope;
* component timing: the same latency calculation for individual flip-flop bits
  or adjacent sine/cosine output pairs.

All shape calculations use a five-validation running median in log10(loss).
Metrics that describe the *shape* of improvement are omitted unless the loss
falls by at least 0.1 decade, preventing nearly flat noisy curves from receiving
an apparently strong staircase score.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_PERSISTENCE = 3
DEFAULT_MEANINGFUL_IMPROVEMENT_DECADES = 0.1
DEFAULT_DROP_FRACTION = 0.1
DEFAULT_TAIL_FRACTION = 0.2

CONFIG_FIELDS = (
    "experiment_name",
    "seed",
    "task",
    "activation",
    "optimizer_name",
    "wrec_init_scale",
    "output_coupling_gamma",
    "recurrent_gain",
    "hidden_size",
    "max_steps",
    "architecture",
    "residual_init_scale",
    "residual_gain",
    "attention_logit_scale",
    "context_length",
)

RESTRUCTURING_FIELDS = (
    "wrec_relative_delta_fro",
    "wrec_cosine_from_init",
    "wrec_delta_orthogonal_fro",
    "effective_wrec_delta_fro",
    "effective_wrec_delta_rms",
)


def _median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    radius = window // 2
    return np.asarray(
        [
            np.median(values[max(0, i - radius) : min(len(values), i + radius + 1)])
            for i in range(len(values))
        ],
        dtype=float,
    )


def _persistent_crossing(
    steps: np.ndarray,
    log_losses: np.ndarray,
    fraction: float,
    persistence: int,
) -> int | None:
    net_improvement = log_losses[0] - log_losses[-1]
    if net_improvement <= 0:
        return None

    target = log_losses[0] - fraction * net_improvement
    width = max(1, min(persistence, len(log_losses)))
    for i in range(len(log_losses) - width + 1):
        if np.all(log_losses[i : i + width] <= target):
            return int(steps[i])
    return None


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.all(x == x[0]):
        return None
    centered_x = x - np.mean(x)
    return float(np.dot(centered_x, y - np.mean(y)) / np.dot(centered_x, centered_x))


def compute_curve_shape(
    steps: Iterable[int],
    losses: Iterable[float],
    *,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    persistence: int = DEFAULT_PERSISTENCE,
    meaningful_improvement_decades: float = DEFAULT_MEANINGFUL_IMPROVEMENT_DECADES,
    drop_fraction: float = DEFAULT_DROP_FRACTION,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> dict[str, float | int | bool | None]:
    """Compute descriptive shape metrics for one validation-loss curve."""
    steps_array = np.asarray(list(steps), dtype=int)
    losses_array = np.asarray(list(losses), dtype=float)
    n_points = min(len(steps_array), len(losses_array))
    steps_array = steps_array[:n_points]
    losses_array = losses_array[:n_points]

    if n_points == 0:
        return {"n_points": 0}
    if np.any(~np.isfinite(losses_array)) or np.any(losses_array <= 0):
        raise ValueError("losses must be finite and strictly positive")
    if np.any(np.diff(steps_array) <= 0):
        raise ValueError("steps must be strictly increasing")
    if not 0 < drop_fraction <= 1:
        raise ValueError("drop_fraction must lie in (0, 1]")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must lie in (0, 1]")

    log_losses = _median_smooth(np.log10(losses_array), smoothing_window)
    log10_improvement = float(log_losses[0] - log_losses[-1])
    meaningful = log10_improvement >= meaningful_improvement_decades

    t10 = _persistent_crossing(steps_array, log_losses, 0.1, persistence)
    t50 = _persistent_crossing(steps_array, log_losses, 0.5, persistence)
    t90 = _persistent_crossing(steps_array, log_losses, 0.9, persistence)

    interval_drops = np.maximum(log_losses[:-1] - log_losses[1:], 0.0)
    total_downward_motion = float(np.sum(interval_drops))
    drop_concentration = None
    monotonicity = None
    if meaningful and len(interval_drops) and total_downward_motion > 0:
        n_largest = max(1, math.ceil(drop_fraction * len(interval_drops)))
        drop_concentration = float(
            np.sum(np.sort(interval_drops)[-n_largest:]) / total_downward_motion
        )
        monotonicity = float(max(log10_improvement, 0.0) / total_downward_motion)

    first_step = int(steps_array[0])
    last_step = int(steps_array[-1])
    horizon = last_step - first_step
    tail_start_step = last_step - tail_fraction * horizon
    tail_start_index = int(np.searchsorted(steps_array, tail_start_step))
    tail_gain_fraction = None
    if meaningful and log10_improvement > 0:
        tail_gain_fraction = float(
            (log_losses[tail_start_index] - log_losses[-1]) / log10_improvement
        )
    tail_slope = _linear_slope(
        steps_array[tail_start_index:].astype(float), log_losses[tail_start_index:]
    )

    transition_width_steps = None
    transition_width_fraction = None
    if meaningful and t10 is not None and t90 is not None:
        transition_width_steps = t90 - t10
        if horizon > 0:
            transition_width_fraction = transition_width_steps / horizon

    return {
        "n_points": n_points,
        "first_step": first_step,
        "last_step": last_step,
        "start_loss": float(10 ** log_losses[0]),
        "final_loss": float(10 ** log_losses[-1]),
        "best_raw_loss": float(np.min(losses_array)),
        "log10_improvement": log10_improvement,
        "meaningful_improvement": meaningful,
        "t10_steps": t10 if meaningful else None,
        "t50_steps": t50 if meaningful else None,
        "t90_steps": t90 if meaningful else None,
        "transition_width_steps": transition_width_steps,
        "transition_width_fraction": transition_width_fraction,
        "drop_concentration_top_10pct": drop_concentration,
        "monotonicity": monotonicity,
        "tail_gain_fraction": tail_gain_fraction,
        "tail_log10_slope_per_100_steps": (
            100.0 * tail_slope if tail_slope is not None else None
        ),
    }


def _component_curves(
    curve: dict[str, Any],
    task: str,
    component_groups: list[list[int]] | None = None,
) -> list[np.ndarray]:
    channels = curve.get("val_losses_per_bit") or {}
    ordered = []
    channel_index = 0
    while f"channel_{channel_index}" in channels:
        ordered.append(np.asarray(channels[f"channel_{channel_index}"], dtype=float))
        channel_index += 1

    if component_groups is not None:
        grouped = []
        for group in component_groups:
            valid = [ordered[index] for index in group if index < len(ordered)]
            if valid:
                grouped.append(np.mean(np.stack(valid, axis=0), axis=0))
        return grouped
    if task == "sine_wave":
        return [
            (ordered[i] + ordered[i + 1]) / 2.0
            for i in range(0, len(ordered) - 1, 2)
        ]
    return ordered


def compute_component_timing(
    curve: dict[str, Any],
    task: str,
    component_groups: list[list[int]] | None = None,
) -> dict[str, float | int | None]:
    """Describe acquisition timing for bits or sine/cosine frequency pairs."""
    steps = curve.get("steps") or []
    component_shapes = [
        compute_curve_shape(steps, losses)
        for losses in _component_curves(curve, task, component_groups)
    ]

    result: dict[str, float | int | None] = {
        "n_components": len(component_shapes),
        "n_components_meaningfully_improved": sum(
            bool(shape.get("meaningful_improvement")) for shape in component_shapes
        ),
    }
    meaningful_t50 = []
    for index, shape in enumerate(component_shapes):
        t50 = shape.get("t50_steps")
        result[f"component_{index}_t50_steps"] = t50
        result[f"component_{index}_log10_improvement"] = shape.get(
            "log10_improvement"
        )
        result[f"component_{index}_final_loss"] = shape.get("final_loss")
        if t50 is not None:
            meaningful_t50.append(int(t50))

    result["component_t50_spread_steps"] = (
        max(meaningful_t50) - min(meaningful_t50)
        if len(meaningful_t50) >= 2
        else None
    )
    return result


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"null", "None", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip('"\'')


def _load_selected_config(path: Path) -> dict[str, Any]:
    """Read the flat scalar fields needed here without requiring a YAML import."""
    wanted = set(CONFIG_FIELDS)
    config: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        if key in wanted:
            config[key] = _parse_scalar(raw_value)
    return config


def _load_final_restructuring(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    records = json.loads(path.read_text()).get("records") or []
    if not records:
        return {}
    final = records[-1]
    return {f"final_{field}": final.get(field) for field in RESTRUCTURING_FIELDS}


def analyze_run(run_dir: Path) -> dict[str, Any]:
    """Analyze one ``<experiment>/seed_<n>`` directory."""
    curve = json.loads((run_dir / "training_losses.json").read_text())
    config = _load_selected_config(run_dir / "run_config.yaml")
    metadata_path = run_dir / "task_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    component_groups = metadata.get("component_groups")
    component_timing = compute_component_timing(
        curve,
        str(config.get("task", "")),
        component_groups,
    )
    for index, reference in enumerate(metadata.get("linear_reference_losses", [])):
        component_timing[f"component_{index}_linear_reference_loss"] = reference
        final_loss = component_timing.get(f"component_{index}_final_loss")
        if final_loss is not None and reference > 0:
            component_timing[f"component_{index}_final_to_linear_ratio"] = (
                final_loss / reference
            )
    for index, reference in enumerate(metadata.get("bayes_reference_losses", [])):
        component_timing[f"component_{index}_bayes_reference_loss"] = reference
        final_loss = component_timing.get(f"component_{index}_final_loss")
        if final_loss is not None and reference > 0:
            component_timing[f"component_{index}_final_to_bayes_ratio"] = (
                final_loss / reference
            )
    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        **config,
        **compute_curve_shape(curve.get("steps") or [], curve.get("val_losses") or []),
        **component_timing,
        **_load_final_restructuring(run_dir / "recurrent_restructuring.json"),
    }
    return row


def collect_sweep_metrics(sweep_dir: Path) -> list[dict[str, Any]]:
    """Analyze every completed seed directory in a sweep."""
    loss_paths = sorted(sweep_dir.glob("*/seed_*/training_losses.json"))
    return [analyze_run(path.parent) for path in loss_paths]


def write_metrics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with (output_dir / "curve_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "definitions": {
            "smoothing": "5-validation running median of log10(loss)",
            "meaningful_improvement": "at least 0.1 decade net loss reduction",
            "t10_t50_t90": "first persistent crossing of 10/50/90% of observed log-loss improvement",
            "drop_concentration_top_10pct": "fraction of downward motion in the largest 10% of validation-interval drops",
            "tail": "last 20% of the observed training horizon",
            "component": "task_metadata component_groups when available; otherwise one bit for flip-flop or adjacent sine/cosine pair",
        },
        "rows": rows,
    }
    (output_dir / "curve_metrics.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = collect_sweep_metrics(args.sweep_dir)
    if not rows:
        raise SystemExit(f"No training curves found under {args.sweep_dir}")
    write_metrics(rows, args.output_dir)
    print(f"Wrote metrics for {len(rows)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
