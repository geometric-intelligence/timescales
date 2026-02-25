#!/usr/bin/env python3
"""
Linear Network Timescales Experiment

Replicates theoretical results on power-law timescale distributions
in linear recurrent networks.

Experiment A: Verifies eigenvalue → timescale mapping produces expected power-law slopes.
Experiment B: Validates simulated autocorrelation timescales match eigenvalue predictions.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import curve_fit


# =============================================================================
# Experiment A: Eigenvalue Distribution Functions
# =============================================================================


def generate_eigenvalues_uniform(
    n: int, low: float, high: float, rng: np.random.Generator
) -> NDArray:
    """Generate uniformly distributed eigenvalues."""
    return rng.uniform(low, high, n)


def generate_eigenvalues_gamma(
    n: int, shape: float, scale: float, rng: np.random.Generator
) -> NDArray:
    """Generate gamma-distributed eigenvalues."""
    return rng.gamma(shape, scale, n)


def generate_eigenvalues_semicircle(
    n: int, g: float, rng: np.random.Generator
) -> NDArray:
    """
    Generate eigenvalues from the semicircle distribution.

    Creates an N×N Gaussian random matrix W, computes its eigenvalues,
    and returns effective eigenvalues: λ_eff = 1 - g·Re(λ_W)
    """
    W = rng.standard_normal((n, n)) / np.sqrt(n)
    eigenvalues_W = np.linalg.eigvals(W)
    lambda_eff = 1 - g * np.real(eigenvalues_W)
    return lambda_eff


def eigenvalues_to_timescales(eigenvalues: NDArray, tau_syn: float) -> NDArray:
    """
    Convert eigenvalues to timescales.

    τ = τ_syn / Re(λ)

    Only returns positive timescales (λ > 0).
    """
    real_eigenvalues = np.real(eigenvalues)
    positive_mask = real_eigenvalues > 0
    timescales = tau_syn / real_eigenvalues[positive_mask]
    return timescales


def fit_powerlaw_slope(
    timescales: NDArray, n_bins: int
) -> Tuple[float, float, NDArray, NDArray]:
    """
    Fit power-law slope to timescale distribution using log-log histogram.

    Returns:
        slope: Power-law exponent γ
        intercept: Log-space intercept
        bin_centers: Centers of histogram bins (log-spaced)
        hist_density: Histogram density values
    """
    log_tau = np.log10(timescales)
    bins = np.logspace(log_tau.min(), log_tau.max(), n_bins + 1)
    hist, bin_edges = np.histogram(timescales, bins=bins, density=True)

    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    valid_mask = hist > 0
    log_centers = np.log10(bin_centers[valid_mask])
    log_hist = np.log10(hist[valid_mask])

    slope, intercept, _, _, _ = stats.linregress(log_centers, log_hist)

    return slope, intercept, bin_centers, hist


def run_experiment_a(config: dict, output_dir: Path, rng: np.random.Generator) -> None:
    """
    Run Experiment A: Theoretical Scaling.

    Verifies eigenvalue → timescale mapping produces expected power-law slopes.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Theoretical Scaling")
    print("=" * 60)

    n = config["n_eigenvalues"]
    tau_syn_values = config["tau_syn_values"]
    n_bins = config["n_bins"]

    distributions = {
        "Uniform": {
            "generate": lambda: generate_eigenvalues_uniform(n, 0.01, 1.0, rng),
            "expected_gamma": -2.0,
        },
        "Gamma(0.5)": {
            "generate": lambda: generate_eigenvalues_gamma(n, 0.5, 0.5, rng),
            "expected_gamma": -1.5,
        },
        "Gamma(2.0)": {
            "generate": lambda: generate_eigenvalues_gamma(n, 2.0, 0.25, rng),
            "expected_gamma": -3.0,
        },
        "Semi-circle": {
            "generate": lambda: generate_eigenvalues_semicircle(n, 0.9, rng),
            "expected_gamma": -2.5,
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    results = {}

    for idx, (name, dist_config) in enumerate(distributions.items()):
        ax = axes[idx]
        eigenvalues = dist_config["generate"]()
        expected_gamma = dist_config["expected_gamma"]

        print(f"\n{name} Distribution:")
        print(f"  Expected γ: {expected_gamma}")

        for tau_syn in tau_syn_values:
            timescales = eigenvalues_to_timescales(eigenvalues, tau_syn)

            if len(timescales) < 10:
                print(f"  τ_syn={tau_syn}: Too few valid timescales ({len(timescales)})")
                continue

            slope, intercept, bin_centers, hist = fit_powerlaw_slope(
                timescales, n_bins
            )

            valid_mask = hist > 0
            ax.loglog(
                bin_centers[valid_mask],
                hist[valid_mask],
                "o-",
                label=f"τ_syn={tau_syn}, γ={slope:.2f}",
                alpha=0.7,
            )

            print(f"  τ_syn={tau_syn}: Fitted γ = {slope:.2f}")

        ax.set_xlabel("Timescale τ")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"{name} (expected γ ≈ {expected_gamma})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        results[name] = {
            "expected_gamma": expected_gamma,
            "eigenvalues": eigenvalues,
        }

    plt.tight_layout()
    output_path = output_dir / "eigenvalue_distributions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved: {output_path}")

    print("\n--- τ_syn Scaling Verification ---")
    print("Horizontal shift without slope change should be observed.")
    print("Check that slopes are consistent across τ_syn values in each panel.")


# =============================================================================
# Experiment B: Physics Validation Functions
# =============================================================================


def generate_weight_matrix(n: int, rng: np.random.Generator) -> NDArray:
    """Generate random weight matrix W ~ N(0, 1/N)."""
    return rng.standard_normal((n, n)) / np.sqrt(n)


def compute_eigendecomposition(W: NDArray) -> Tuple[NDArray, NDArray]:
    """Compute eigenvalues and eigenvectors of weight matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(W)
    return eigenvalues, eigenvectors


def compute_theoretical_timescales(
    eigenvalues: NDArray, g: float, tau_syn: float
) -> NDArray:
    """
    Compute theoretical timescales from weight matrix eigenvalues.

    τ_theory = τ_syn / (1 - g·Re(λ_W))

    Only returns positive timescales where (1 - g·Re(λ_W)) > 0.
    """
    lambda_eff = 1 - g * np.real(eigenvalues)
    positive_mask = lambda_eff > 0
    timescales = tau_syn / lambda_eff[positive_mask]
    return timescales, positive_mask


def simulate_linear_ode(
    W: NDArray,
    g: float,
    tau_syn: float,
    dt: float,
    duration: float,
    noise_std: float,
    rng: np.random.Generator,
) -> NDArray:
    """
    Simulate linear ODE using Euler-Maruyama method.

    τ_syn · dx/dt = -x + g·W·x + η

    Discretized: x(t+dt) = x(t) + (dt/τ_syn) * (-x(t) + g·W·x(t)) + sqrt(dt/τ_syn)·noise_std·η

    Returns:
        activity: Array of shape (n_steps, n_neurons)
    """
    n = W.shape[0]
    n_steps = int(duration / dt)

    activity = np.zeros((n_steps, n))
    x = rng.standard_normal(n) * 0.1

    dt_tau = dt / tau_syn
    sqrt_dt_tau = np.sqrt(dt_tau)
    gW = g * W

    print(f"  Simulating {n_steps} steps...")

    for t in range(n_steps):
        activity[t] = x

        noise = rng.standard_normal(n) * noise_std * sqrt_dt_tau
        dx = dt_tau * (-x + gW @ x) + noise
        x = x + dx

        if t % 10000 == 0 and t > 0:
            print(f"    Step {t}/{n_steps}")

    return activity


def compute_autocorrelation_fft(signal: NDArray, max_lag: int) -> NDArray:
    """
    Compute autocorrelation using FFT for efficiency.

    Args:
        signal: 1D signal array
        max_lag: Maximum lag to compute

    Returns:
        autocorr: Normalized autocorrelation from lag 0 to max_lag
    """
    n = len(signal)
    signal_centered = signal - np.mean(signal)

    fft_len = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    fft_signal = np.fft.fft(signal_centered, fft_len)
    power_spectrum = fft_signal * np.conj(fft_signal)
    autocorr_full = np.real(np.fft.ifft(power_spectrum))

    autocorr = autocorr_full[:max_lag + 1]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]

    return autocorr


def fit_exponential_timescale(
    autocorr: NDArray, dt: float, fit_range: Tuple[int, int] = (1, None)
) -> float:
    """
    Fit exponential decay to autocorrelation to extract timescale.

    Fits in log-space: log(AC) = -t/τ + const

    Args:
        autocorr: Normalized autocorrelation
        dt: Time step
        fit_range: (start_lag, end_lag) for fitting

    Returns:
        timescale: Fitted timescale τ
    """
    start_lag = fit_range[0]
    end_lag = fit_range[1] if fit_range[1] is not None else len(autocorr)

    lags = np.arange(start_lag, end_lag)
    ac_segment = autocorr[start_lag:end_lag]

    positive_mask = ac_segment > 0.01
    if np.sum(positive_mask) < 5:
        return np.nan

    lags_fit = lags[positive_mask]
    ac_fit = ac_segment[positive_mask]

    log_ac = np.log(ac_fit)
    times = lags_fit * dt

    slope, intercept, _, _, _ = stats.linregress(times, log_ac)

    if slope >= 0:
        return np.nan

    timescale = -1.0 / slope
    return timescale


def project_onto_eigenvectors(
    activity: NDArray, eigenvectors: NDArray
) -> NDArray:
    """
    Project neural activity onto eigenvectors.

    Args:
        activity: Shape (n_steps, n_neurons)
        eigenvectors: Shape (n_neurons, n_neurons), columns are eigenvectors

    Returns:
        projections: Shape (n_steps, n_modes) - complex-valued
    """
    eigenvectors_inv = np.linalg.inv(eigenvectors)
    projections = activity @ eigenvectors_inv.T
    return projections


def run_experiment_b(config: dict, output_dir: Path, rng: np.random.Generator) -> None:
    """
    Run Experiment B: Physics Validation.

    Verifies simulated autocorrelation timescales match eigenvalue predictions.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Physics Validation")
    print("=" * 60)

    n = config["n_neurons"]
    g = config["g_coupling"]
    tau_syn = config["tau_syn"]
    dt = config["dt"]
    duration = config["duration"]
    noise_std = config["noise_std"]
    max_lag = config["max_lag"]

    print(f"\nParameters:")
    print(f"  N = {n} neurons")
    print(f"  g = {g}")
    print(f"  τ_syn = {tau_syn} ms")
    print(f"  dt = {dt} ms")
    print(f"  duration = {duration} ms ({duration/1000:.1f} s)")
    print(f"  noise_std = {noise_std}")

    print("\n1. Generating weight matrix...")
    W = generate_weight_matrix(n, rng)

    print("2. Computing eigendecomposition...")
    eigenvalues, eigenvectors = compute_eigendecomposition(W)

    print("3. Computing theoretical timescales...")
    tau_theory, valid_mask = compute_theoretical_timescales(eigenvalues, g, tau_syn)
    print(f"   {len(tau_theory)} valid timescales (positive effective eigenvalues)")
    print(f"   τ range: [{tau_theory.min():.2f}, {tau_theory.max():.2f}] ms")

    print("\n4. Simulating linear ODE...")
    activity = simulate_linear_ode(W, g, tau_syn, dt, duration, noise_std, rng)

    print("\n5. Computing autocorrelations and fitting timescales...")
    tau_fitted = np.zeros(n)

    discard_steps = int(1000 / dt)
    activity_steady = activity[discard_steps:]

    for i in range(n):
        autocorr = compute_autocorrelation_fft(activity_steady[:, i], max_lag)
        tau_fitted[i] = fit_exponential_timescale(autocorr, dt)

        if i % 100 == 0:
            print(f"   Processed {i}/{n} neurons")

    valid_fitted = ~np.isnan(tau_fitted)
    print(f"   {np.sum(valid_fitted)} neurons with valid fitted timescales")

    print("\n6. Creating comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    n_bins = 50

    tau_theory_finite = tau_theory[np.isfinite(tau_theory) & (tau_theory > 0)]
    tau_fitted_valid = tau_fitted[valid_fitted & (tau_fitted > 0) & np.isfinite(tau_fitted)]

    if len(tau_theory_finite) > 0 and len(tau_fitted_valid) > 0:
        all_tau = np.concatenate([tau_theory_finite, tau_fitted_valid])
        bins = np.logspace(
            np.log10(np.percentile(all_tau, 1)),
            np.log10(np.percentile(all_tau, 99)),
            n_bins,
        )

        ax1.hist(
            tau_theory_finite,
            bins=bins,
            alpha=0.5,
            label=f"Theory (n={len(tau_theory_finite)})",
            density=True,
        )
        ax1.hist(
            tau_fitted_valid,
            bins=bins,
            alpha=0.5,
            label=f"Simulation (n={len(tau_fitted_valid)})",
            density=True,
        )

        ax1.set_xscale("log")
        ax1.set_xlabel("Timescale τ (ms)")
        ax1.set_ylabel("Probability Density")
        ax1.set_title("Theory vs Simulation Timescale Distributions")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    ax2 = axes[1]

    n_modes_to_plot = 5
    print("\n7. Sanity check: Eigenvector projections...")

    real_eigenvalue_indices = np.where(
        (np.abs(np.imag(eigenvalues)) < 1e-10) & valid_mask
    )[0]

    if len(real_eigenvalue_indices) >= n_modes_to_plot:
        tau_real = tau_syn / (1 - g * np.real(eigenvalues[real_eigenvalue_indices]))
        sorted_indices = real_eigenvalue_indices[np.argsort(tau_real)[::-1]]
        selected_indices = sorted_indices[:n_modes_to_plot]
    else:
        selected_indices = np.where(valid_mask)[0][:n_modes_to_plot]

    projections = project_onto_eigenvectors(activity_steady, eigenvectors)

    colors = plt.cm.viridis(np.linspace(0, 1, n_modes_to_plot))

    for i, mode_idx in enumerate(selected_indices):
        proj = np.real(projections[:, mode_idx])
        autocorr = compute_autocorrelation_fft(proj, max_lag)

        tau_mode_theory = tau_syn / (1 - g * np.real(eigenvalues[mode_idx]))
        tau_mode_fitted = fit_exponential_timescale(autocorr, dt)

        lags = np.arange(len(autocorr)) * dt
        ax2.plot(
            lags,
            autocorr,
            color=colors[i],
            label=f"Mode {mode_idx}: τ_theory={tau_mode_theory:.1f}, τ_fit={tau_mode_fitted:.1f}",
            alpha=0.8,
        )

        if tau_mode_theory > 0 and np.isfinite(tau_mode_theory):
            ax2.plot(
                lags,
                np.exp(-lags / tau_mode_theory),
                "--",
                color=colors[i],
                alpha=0.5,
            )

    ax2.set_xlabel("Lag (ms)")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title("Eigenvector Projection Autocorrelations")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max_lag * dt])

    plt.tight_layout()
    output_path = output_dir / "simulation_vs_theory.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved: {output_path}")

    print("\n8. Creating detailed eigenvector projection plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    n_detailed = min(6, len(selected_indices) + 1)

    for i in range(n_detailed - 1):
        if i >= len(selected_indices):
            break

        mode_idx = selected_indices[i]
        ax = axes[i]

        proj = np.real(projections[:, mode_idx])
        autocorr = compute_autocorrelation_fft(proj, max_lag)

        tau_mode_theory = tau_syn / (1 - g * np.real(eigenvalues[mode_idx]))
        tau_mode_fitted = fit_exponential_timescale(autocorr, dt)

        lags = np.arange(len(autocorr)) * dt

        ax.semilogy(lags, np.maximum(autocorr, 1e-10), "b-", label="Simulated", linewidth=2)

        if tau_mode_theory > 0 and np.isfinite(tau_mode_theory):
            ax.semilogy(
                lags,
                np.exp(-lags / tau_mode_theory),
                "r--",
                label=f"Theory (τ={tau_mode_theory:.1f})",
                linewidth=2,
            )

        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Mode {mode_idx}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([1e-2, 1.5])

    ax_summary = axes[-1]
    if len(tau_theory_finite) > 0 and len(tau_fitted_valid) > 0:
        ax_summary.scatter(
            tau_theory_finite[:len(tau_fitted_valid)],
            tau_fitted_valid[:len(tau_theory_finite)],
            alpha=0.3,
            s=10,
        )

        max_val = max(tau_theory_finite.max(), tau_fitted_valid.max())
        min_val = min(tau_theory_finite.min(), tau_fitted_valid.min())
        ax_summary.plot([min_val, max_val], [min_val, max_val], "r--", label="Identity")

        ax_summary.set_xscale("log")
        ax_summary.set_yscale("log")
        ax_summary.set_xlabel("Theoretical τ (ms)")
        ax_summary.set_ylabel("Fitted τ (ms)")
        ax_summary.set_title("Theory vs Fitted Timescales")
        ax_summary.legend()
        ax_summary.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "eigenvector_projections.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    print("\n--- Summary Statistics ---")
    if len(tau_fitted_valid) > 0:
        print(f"Theoretical τ: median={np.median(tau_theory):.2f}, "
              f"mean={np.mean(tau_theory):.2f}")
        print(f"Fitted τ: median={np.median(tau_fitted_valid):.2f}, "
              f"mean={np.mean(tau_fitted_valid):.2f}")


# =============================================================================
# Main Script
# =============================================================================


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Linear Network Timescales Experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/<timestamp>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = script_dir / args.config

    if not config_path.exists():
        config_path = Path(args.config)

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    rng = np.random.default_rng(seed)
    print(f"Random seed: {seed}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / "results" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)

    if config.get("run_experiment_a", True):
        run_experiment_a(config["experiment_a"], output_dir, rng)

    if config.get("run_experiment_b", True):
        run_experiment_b(config["experiment_b"], output_dir, rng)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
