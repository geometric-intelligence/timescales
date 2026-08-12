#!/usr/bin/env python3
"""Estimate task-target autocorrelations and poles without using an RNN.

The estimator is deliberately data-driven:

1. Generate independent target trajectories.
2. Estimate centered, normalized autocorrelations by pooled FFT products.
3. Recover a finite-dimensional continuation with a matrix-pencil estimator.
4. Map discrete poles ``z`` to the continuous Laplace plane with
   ``s = log(z) / dt``.

Analytic poles are computed only after fitting, as a validation reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", type=Path, default=Path("target_poles.json"))
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-samples", type=int, default=48)
    return parser.parse_args()


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _raw_fft_statistics(
    values: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return summed lag products and value sums for a trajectory batch."""
    n_time = values.shape[1]
    n_fft = _next_power_of_two(2 * n_time)
    spectrum = np.fft.rfft(values, n=n_fft, axis=1)
    products = np.fft.irfft(spectrum * spectrum.conj(), n=n_fft, axis=1)
    lag_products = products[:, : max_lag + 1, :].sum(axis=0).T
    value_sums = values.sum(axis=(0, 1), dtype=np.float64)
    return lag_products, value_sums


def _statistics_to_acf(
    lag_products: np.ndarray,
    value_sums: np.ndarray,
    n_trajectories: int,
    n_time: int,
) -> np.ndarray:
    lags = np.arange(lag_products.shape[-1])
    counts = n_trajectories * (n_time - lags)
    mean = value_sums / (n_trajectories * n_time)
    covariance = lag_products / counts[None, :] - mean[:, None] ** 2
    return covariance / covariance[:, :1]


def _flip_flop_targets(
    rng: np.random.Generator,
    n_trajectories: int,
    n_time: int,
    pulse_probabilities: np.ndarray,
) -> np.ndarray:
    """Generate centered forced-pulse flip-flop targets in {-1, +1}."""
    n_channels = pulse_probabilities.size
    values = np.empty((n_trajectories, n_time, n_channels), dtype=np.float32)
    state = rng.choice(np.array([-1.0, 1.0], dtype=np.float32),
                       size=(n_trajectories, n_channels))
    values[:, 0, :] = state
    for time_index in range(1, n_time):
        pulse = rng.random((n_trajectories, n_channels)) < pulse_probabilities
        new_sign = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=(n_trajectories, n_channels),
        )
        state = np.where(pulse, new_sign, state)
        values[:, time_index, :] = state
    return values


def _sine_targets(
    rng: np.random.Generator,
    n_trajectories: int,
    n_time: int,
    periods: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Generate independent random-phase sine/cosine target pairs."""
    time = np.arange(n_time, dtype=np.float64) * dt
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_trajectories, periods.size))
    values = np.empty(
        (n_trajectories, n_time, 2 * periods.size), dtype=np.float32
    )
    for index, period in enumerate(periods):
        angle = phase[:, index, None] + 2.0 * np.pi * time[None, :] / period
        values[:, :, 2 * index] = np.cos(angle)
        values[:, :, 2 * index + 1] = np.sin(angle)
    return values


def _estimate_group_acfs(
    *,
    generator,
    rng: np.random.Generator,
    n_groups: int,
    trajectories_per_group: int,
    n_time: int,
    max_lag: int,
    pair_channels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    raw_groups = []
    sum_groups = []
    for group_index in range(n_groups):
        values = generator(rng, trajectories_per_group, n_time)
        lag_products, value_sums = _raw_fft_statistics(values, max_lag)
        raw_groups.append(lag_products)
        sum_groups.append(value_sums)
        print(f"  target group {group_index + 1:02d}/{n_groups}", flush=True)

    raw_groups = np.stack(raw_groups)
    sum_groups = np.stack(sum_groups)
    group_acfs = np.stack(
        [
            _statistics_to_acf(
                raw_groups[index],
                sum_groups[index],
                trajectories_per_group,
                n_time,
            )
            for index in range(n_groups)
        ]
    )
    full_acf = _statistics_to_acf(
        raw_groups.sum(axis=0),
        sum_groups.sum(axis=0),
        n_groups * trajectories_per_group,
        n_time,
    )

    if pair_channels:
        group_acfs = 0.5 * (group_acfs[:, 0::2] + group_acfs[:, 1::2])
        full_acf = 0.5 * (full_acf[0::2] + full_acf[1::2])
    return full_acf, group_acfs


def _matrix_pencil(
    sequence: np.ndarray,
    *,
    rank: int | None = None,
    max_rank: int = 8,
) -> dict:
    """Fit a sum of discrete exponentials to a real scalar sequence."""
    values = np.asarray(sequence, dtype=np.float64)
    n_values = values.size
    n_rows = max(8, n_values // 3)
    n_columns = n_values - n_rows
    windows = np.lib.stride_tricks.sliding_window_view(values, n_columns + 1)
    h0 = windows[:n_rows, :-1]
    h1 = windows[:n_rows, 1:]
    u, singular_values, vh = np.linalg.svd(h0, full_matrices=False)

    if rank is None:
        n_candidates = min(max_rank, singular_values.size - 1)
        gaps = singular_values[:n_candidates] / np.maximum(
            singular_values[1 : n_candidates + 1], np.finfo(float).tiny
        )
        rank = int(np.argmax(gaps) + 1)
    else:
        gaps = singular_values[:max_rank] / np.maximum(
            singular_values[1 : max_rank + 1], np.finfo(float).tiny
        )

    ur = u[:, :rank]
    vr = vh[:rank].conj().T
    reduced = (
        np.diag(1.0 / singular_values[:rank])
        @ ur.conj().T
        @ h1
        @ vr
    )
    poles = np.linalg.eigvals(reduced)
    vandermonde = poles[None, :] ** np.arange(n_values)[:, None]
    residues = np.linalg.lstsq(vandermonde, values, rcond=None)[0]
    fitted = np.real(vandermonde @ residues)
    return {
        "rank": rank,
        "poles": poles,
        "residues": residues,
        "fitted": fitted,
        "singular_values": singular_values,
        "gaps": gaps,
    }


def _complex_rows(values: np.ndarray) -> list[dict[str, float]]:
    return [
        {"real": float(value.real), "imag": float(value.imag)} for value in values
    ]


def _fit_components(
    *,
    full_acf: np.ndarray,
    group_acfs: np.ndarray,
    dt: float,
    labels: list[str],
    analytic_s_poles: list[np.ndarray],
    bootstrap_samples: int,
    rng: np.random.Generator,
    max_plot_lag: int,
) -> list[dict]:
    fit_points = min(512, full_acf.shape[-1])
    bootstrap_fit_points = min(256, full_acf.shape[-1])
    plot_stride = max(1, max_plot_lag // 400)
    plot_indices = np.arange(0, max_plot_lag + 1, plot_stride)
    components = []

    for channel, label in enumerate(labels):
        fit = _matrix_pencil(full_acf[channel, :fit_points])
        z_poles = fit["poles"]
        s_poles = np.log(z_poles.astype(np.complex128)) / dt

        bootstrap_poles = []
        for _ in range(bootstrap_samples):
            sample = rng.integers(0, group_acfs.shape[0], group_acfs.shape[0])
            bootstrap_acf = group_acfs[sample, channel].mean(axis=0)
            bootstrap_fit = _matrix_pencil(
                bootstrap_acf[:bootstrap_fit_points], rank=fit["rank"]
            )
            bootstrap_s = np.log(
                bootstrap_fit["poles"].astype(np.complex128)
            ) / dt
            bootstrap_poles.extend(_complex_rows(bootstrap_s))

        n_plot = max_plot_lag + 1
        plot_fit = _matrix_pencil(full_acf[channel, :fit_points], rank=fit["rank"])
        fitted_long = np.real(
            plot_fit["poles"][None, :] ** np.arange(n_plot)[:, None]
            @ plot_fit["residues"]
        )
        analytic_acf = np.real(
            np.sum(
                np.exp(
                    analytic_s_poles[channel][None, :]
                    * (np.arange(n_plot) * dt)[:, None]
                ),
                axis=1,
            )
            / analytic_s_poles[channel].size
        )

        components.append(
            {
                "label": label,
                "selected_rank": int(fit["rank"]),
                "singular_values_relative": [
                    float(value / fit["singular_values"][0])
                    for value in fit["singular_values"][:8]
                ],
                "z_poles": _complex_rows(z_poles),
                "s_poles": _complex_rows(s_poles),
                "bootstrap_s_poles": bootstrap_poles,
                "analytic_s_poles": _complex_rows(analytic_s_poles[channel]),
                "acf_lag": [float(index * dt) for index in plot_indices],
                "acf_empirical": [
                    float(value) for value in full_acf[channel, plot_indices]
                ],
                "acf_fitted": [float(value) for value in fitted_long[plot_indices]],
                "acf_analytic": [float(value) for value in analytic_acf[plot_indices]],
            }
        )
    return components


def _add_laplace_slices(task: dict, full_acf: np.ndarray) -> None:
    dt = float(task["dt"])
    lag = np.arange(full_acf.shape[-1]) * dt
    for channel, component in enumerate(task["components"]):
        fitted_poles = np.asarray(
            [complex(row["real"], row["imag"]) for row in component["s_poles"]]
        )
        fit = _matrix_pencil(
            full_acf[channel, : min(512, full_acf.shape[-1])],
            rank=component["selected_rank"],
        )
        residues = fit["residues"]

        if task["task"] == "flip_flop":
            coordinate = np.linspace(-0.13, 0.08, 320)
            sample_s = coordinate.astype(np.complex128)
            axis = "real"
        else:
            coordinate = np.linspace(0.0, 0.4, 320)
            sample_s = 0.005 + 1j * coordinate
            axis = "imaginary_at_real_0.005"

        empirical = dt * np.sum(
            full_acf[channel, :, None]
            * np.exp(-lag[:, None] * sample_s[None, :]),
            axis=0,
        )
        rational = np.sum(
            residues[:, None] / (sample_s[None, :] - fitted_poles[:, None]),
            axis=0,
        )
        component["laplace_slice"] = {
            "axis": axis,
            "coordinate": [float(value) for value in coordinate],
            "empirical_log10_magnitude": [
                float(np.log10(max(abs(value), 1e-12))) for value in empirical
            ],
            "rational_log10_magnitude": [
                float(np.log10(max(abs(value), 1e-12))) for value in rational
            ],
        }


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    pulse_probabilities = np.asarray([0.01, 0.005, 0.0025])
    flip_dt = 0.1
    print("Estimating flip-flop target autocorrelations", flush=True)
    flip_acf, flip_groups = _estimate_group_acfs(
        generator=lambda group_rng, n_trajectories, n_time: _flip_flop_targets(
            group_rng,
            n_trajectories,
            n_time,
            pulse_probabilities,
        ),
        rng=rng,
        n_groups=16,
        trajectories_per_group=128,
        n_time=4096,
        max_lag=1600,
        pair_channels=False,
    )
    flip_analytic = [
        np.asarray([np.log(1.0 - probability) / flip_dt], dtype=np.complex128)
        for probability in pulse_probabilities
    ]
    flip_task = {
        "task": "flip_flop",
        "dt": flip_dt,
        "trajectory_steps": 4096,
        "n_trajectories": 2048,
        "parameters": {"p_pulse": pulse_probabilities.tolist()},
        "components": _fit_components(
            full_acf=flip_acf,
            group_acfs=flip_groups,
            dt=flip_dt,
            labels=["mean hold 100 steps", "mean hold 200 steps", "mean hold 400 steps"],
            analytic_s_poles=flip_analytic,
            bootstrap_samples=args.bootstrap_samples,
            rng=rng,
            max_plot_lag=1600,
        ),
    }
    _add_laplace_slices(flip_task, flip_acf)

    periods = np.asarray([20.0, 50.0, 100.0])
    sine_dt = 1.0
    print("Estimating random-phase sine target autocorrelations", flush=True)
    sine_acf, sine_groups = _estimate_group_acfs(
        generator=lambda group_rng, n_trajectories, n_time: _sine_targets(
            group_rng, n_trajectories, n_time, periods, sine_dt
        ),
        rng=rng,
        n_groups=16,
        trajectories_per_group=64,
        n_time=2000,
        max_lag=500,
        pair_channels=True,
    )
    sine_analytic = [
        np.asarray(
            [-1j * 2.0 * np.pi / period, 1j * 2.0 * np.pi / period],
            dtype=np.complex128,
        )
        for period in periods
    ]
    sine_task = {
        "task": "sine_wave",
        "dt": sine_dt,
        "trajectory_steps": 2000,
        "n_trajectories": 1024,
        "parameters": {"periods": periods.tolist(), "random_phase": True},
        "components": _fit_components(
            full_acf=sine_acf,
            group_acfs=sine_groups,
            dt=sine_dt,
            labels=["period 20", "period 50", "period 100"],
            analytic_s_poles=sine_analytic,
            bootstrap_samples=args.bootstrap_samples,
            rng=rng,
            max_plot_lag=500,
        ),
    }
    _add_laplace_slices(sine_task, sine_acf)

    result = {
        "method": {
            "autocorrelation": "pooled unbiased lag products, centered by the empirical ensemble mean and normalized at lag zero",
            "pole_estimator": "matrix pencil with rank selected by the largest leading Hankel singular-value gap",
            "continuous_mapping": "s = log(z) / dt using the principal branch",
            "uncertainty": "trajectory-group bootstrap",
            "analytic_values_used_for_fitting": False,
        },
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "tasks": [flip_task, sine_task],
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved {args.out_file}", flush=True)


if __name__ == "__main__":
    main()
