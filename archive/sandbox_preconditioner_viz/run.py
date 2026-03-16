#!/usr/bin/env python
"""
Toy 2D RNN example demonstrating gradient preconditioning for multi-timescale networks.

This script visualizes how heterogeneous timescales create ill-conditioned loss landscapes
and how preconditioning (scaling gradients by 1/α) improves optimization.

The example uses a teacher-student setup where a 2-neuron RNN tries to match
the dynamics of a teacher network with known weights.

Usage:
    python run.py --config configs/default.yaml
    python run.py --config configs/default.yaml --n_steps 10000
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

try:
    import yaml
except ImportError:
    yaml = None


# =============================================================================
# Configuration
# =============================================================================

RESULTS_BASE = Path(__file__).parent / "results"


@dataclass
class Config:
    """Configuration for the preconditioning visualization experiment."""

    # Timescales
    alpha_fast: float = 0.9
    alpha_slow: float = 0.1

    # Teacher weights (target)
    w12_teacher: float = 1.0
    w21_teacher: float = 0.5

    # Simulation parameters
    rollout_length: int = 20
    initial_state: tuple[float, float] = (0.5, -0.3)

    # Optimization parameters
    n_steps: int = 5000
    learning_rate: float = 0.05

    # Multiple trajectory settings
    n_trajectories: int = 8
    init_range: tuple[float, float] = (-0.5, 1.5)

    # Visualization grid
    grid_resolution: int = 15
    w12_range: tuple[float, float] = (-0.5, 2.0)
    w21_range: tuple[float, float] = (-0.5, 1.5)

    # Output
    seed: int = 42

    @property
    def alphas(self) -> NDArray[np.floating]:
        """Return timescales as numpy array."""
        return np.array([self.alpha_fast, self.alpha_slow])

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        if yaml is None:
            raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle list -> tuple conversions
        if "initial_state" in data and isinstance(data["initial_state"], list):
            data["initial_state"] = tuple(data["initial_state"])
        if "init_range" in data and isinstance(data["init_range"], list):
            data["init_range"] = tuple(data["init_range"])
        if "w12_range" in data and isinstance(data["w12_range"], list):
            data["w12_range"] = tuple(data["w12_range"])
        if "w21_range" in data and isinstance(data["w21_range"], list):
            data["w21_range"] = tuple(data["w21_range"])

        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        if yaml is None:
            raise ImportError("PyYAML is required to save config files.")

        data = {
            "alpha_fast": self.alpha_fast,
            "alpha_slow": self.alpha_slow,
            "w12_teacher": self.w12_teacher,
            "w21_teacher": self.w21_teacher,
            "rollout_length": self.rollout_length,
            "initial_state": list(self.initial_state),
            "n_steps": self.n_steps,
            "learning_rate": self.learning_rate,
            "n_trajectories": self.n_trajectories,
            "init_range": list(self.init_range),
            "grid_resolution": self.grid_resolution,
            "w12_range": list(self.w12_range),
            "w21_range": list(self.w21_range),
            "seed": self.seed,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_timestamped_results_dir() -> Path:
    """Create and return a timestamped results directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_BASE / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# =============================================================================
# Core Functions
# =============================================================================


def tanh_activation(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Tanh activation function."""
    return np.tanh(x)


def run_rnn(
    W: NDArray[np.floating],
    alphas: NDArray[np.floating],
    r0: NDArray[np.floating],
    T: int,
) -> NDArray[np.floating]:
    """
    Run RNN dynamics for T steps.

    Implements leaky integration: r(t+1) = (1-α)*r(t) + α*tanh(W*r(t))

    :param W: Weight matrix of shape (2, 2).
    :param alphas: Per-neuron timescales of shape (2,).
    :param r0: Initial state of shape (2,).
    :param T: Number of timesteps to simulate.
    :return: Trajectory of shape (T+1, 2).
    """
    trajectory = [r0.copy()]
    r = r0.copy()

    for _ in range(T):
        r_new = (1 - alphas) * r + alphas * tanh_activation(W @ r)
        trajectory.append(r_new.copy())
        r = r_new

    return np.array(trajectory)


def compute_loss(
    w12: float,
    w21: float,
    config: Config,
) -> float:
    """
    Compute loss by rolling out dynamics and comparing trajectories.

    :param w12: Student weight into fast neuron.
    :param w21: Student weight into slow neuron.
    :param config: Experiment configuration.
    :return: MSE loss between student and teacher trajectories.
    """
    W_student = np.array([[0.0, w12], [w21, 0.0]])
    W_teacher = np.array([[0.0, config.w12_teacher], [config.w21_teacher, 0.0]])
    r0 = np.array(config.initial_state)

    traj_student = run_rnn(W_student, config.alphas, r0, config.rollout_length)
    traj_teacher = run_rnn(W_teacher, config.alphas, r0, config.rollout_length)

    return float(np.mean((traj_student - traj_teacher) ** 2))


def compute_gradients_finite_diff(
    w12: float,
    w21: float,
    config: Config,
    eps: float = 1e-5,
) -> tuple[float, float, float]:
    """
    Compute gradients using finite differences.

    :param w12: Current w12 value.
    :param w21: Current w21 value.
    :param config: Experiment configuration.
    :param eps: Finite difference step size.
    :return: Tuple of (loss, grad_w12, grad_w21).
    """
    loss_center = compute_loss(w12, w21, config)
    loss_plus_w12 = compute_loss(w12 + eps, w21, config)
    loss_plus_w21 = compute_loss(w12, w21 + eps, config)

    grad_w12 = (loss_plus_w12 - loss_center) / eps
    grad_w21 = (loss_plus_w21 - loss_center) / eps

    return loss_center, grad_w12, grad_w21


# =============================================================================
# Optimization
# =============================================================================


@dataclass
class OptimizationResult:
    """Results from running optimization."""

    w12_trajectory: list[float]
    w21_trajectory: list[float]
    loss_trajectory: list[float]
    final_loss: float
    initial_loss: float

    @property
    def loss_reduction(self) -> float:
        """Compute the loss reduction factor."""
        if self.final_loss > 0:
            return self.initial_loss / self.final_loss
        return float("inf")


def run_optimization(
    w12_init: float,
    w21_init: float,
    config: Config,
    precondition: bool = False,
) -> OptimizationResult:
    """
    Run gradient descent optimization.

    :param w12_init: Initial w12 value.
    :param w21_init: Initial w21 value.
    :param config: Experiment configuration.
    :param precondition: If True, scale gradients by 1/α (preconditioning).
    :return: OptimizationResult with trajectories and losses.
    """
    w12_traj = [w12_init]
    w21_traj = [w21_init]
    loss_traj = []

    w12, w21 = w12_init, w21_init

    for _ in range(config.n_steps):
        loss_val, grad12, grad21 = compute_gradients_finite_diff(w12, w21, config)

        if precondition:
            w12 -= config.learning_rate * grad12 / config.alpha_fast
            w21 -= config.learning_rate * grad21 / config.alpha_slow
        else:
            w12 -= config.learning_rate * grad12
            w21 -= config.learning_rate * grad21

        w12_traj.append(w12)
        w21_traj.append(w21)
        loss_traj.append(loss_val)

    return OptimizationResult(
        w12_trajectory=w12_traj,
        w21_trajectory=w21_traj,
        loss_trajectory=loss_traj,
        final_loss=loss_traj[-1],
        initial_loss=loss_traj[0],
    )


def generate_initial_points(
    config: Config,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Generate random initial points for optimization trajectories."""
    low, high = config.init_range
    return [(rng.uniform(low, high), rng.uniform(low, high)) for _ in range(config.n_trajectories)]


# =============================================================================
# Visualization
# =============================================================================


def compute_gradient_field(
    config: Config,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Compute gradient field on a grid."""
    w12_vals = np.linspace(*config.w12_range, config.grid_resolution)
    w21_vals = np.linspace(*config.w21_range, config.grid_resolution)
    W12, W21 = np.meshgrid(w12_vals, w21_vals)

    dL_dw12 = np.zeros_like(W12)
    dL_dw21 = np.zeros_like(W21)
    loss_grid = np.zeros_like(W12)

    for i in range(W12.shape[0]):
        for j in range(W12.shape[1]):
            loss_val, grad12, grad21 = compute_gradients_finite_diff(
                W12[i, j], W21[i, j], config
            )
            dL_dw12[i, j] = grad12
            dL_dw21[i, j] = grad21
            loss_grid[i, j] = loss_val

    return W12, W21, dL_dw12, dL_dw21, loss_grid


def plot_gradient_fields(
    config: Config,
    W12: NDArray,
    W21: NDArray,
    dL_dw12: NDArray,
    dL_dw21: NDArray,
    loss_grid: NDArray,
    output_path: str,
) -> None:
    """Plot raw vs preconditioned gradient fields."""
    precond_dL_dw12 = dL_dw12 / config.alpha_fast
    precond_dL_dw21 = dL_dw21 / config.alpha_slow

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw gradients
    ax1 = axes[0]
    magnitude_raw = np.sqrt(dL_dw12**2 + dL_dw21**2)
    scale_raw = np.max(magnitude_raw)
    ax1.quiver(W12, W21, -dL_dw12 / scale_raw, -dL_dw21 / scale_raw,
               alpha=0.7, width=0.005, color="blue", scale=8)
    ax1.plot(config.w12_teacher, config.w21_teacher, "r*", markersize=20,
             label="Teacher weights", zorder=5)
    ax1.contour(W12, W21, loss_grid, levels=10, alpha=0.4, colors="gray", linewidths=1)
    ax1.set_xlabel(rf"$w_{{12}}$ (into fast neuron, $\alpha_1={config.alpha_fast}$)", fontsize=12)
    ax1.set_ylabel(rf"$w_{{21}}$ (into slow neuron, $\alpha_2={config.alpha_slow}$)", fontsize=12)
    ax1.set_title("Raw Gradient Field", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect("equal")

    # Preconditioned gradients
    ax2 = axes[1]
    magnitude_precond = np.sqrt(precond_dL_dw12**2 + precond_dL_dw21**2)
    scale_precond = np.max(magnitude_precond)
    ax2.quiver(W12, W21, -precond_dL_dw12 / scale_precond, -precond_dL_dw21 / scale_precond,
               alpha=0.7, width=0.005, color="green", scale=8)
    ax2.plot(config.w12_teacher, config.w21_teacher, "r*", markersize=20,
             label="Teacher weights", zorder=5)
    ax2.contour(W12, W21, loss_grid, levels=10, alpha=0.4, colors="gray", linewidths=1)
    ax2.set_xlabel(rf"$w_{{12}}$ (into fast neuron, $\alpha_1={config.alpha_fast}$)", fontsize=12)
    ax2.set_ylabel(rf"$w_{{21}}$ (into slow neuron, $\alpha_2={config.alpha_slow}$)", fontsize=12)
    ax2.set_title(r"Preconditioned Gradient Field ($A^{-1}\nabla L$)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_optimization_trajectories(
    config: Config,
    W12: NDArray,
    W21: NDArray,
    loss_grid: NDArray,
    standard_results: list[OptimizationResult],
    precond_results: list[OptimizationResult],
    output_path: str,
) -> None:
    """Plot optimization trajectories and loss curves for multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: trajectories in weight space
    ax_traj = axes[0]
    contourf = ax_traj.contourf(W12, W21, loss_grid, levels=50, cmap="viridis", alpha=0.6)
    plt.colorbar(contourf, ax=ax_traj, label="Loss")

    for i, (std_res, pre_res) in enumerate(zip(standard_results, precond_results)):
        alpha_line = 0.7 if i == 0 else 0.4
        lw = 2.0 if i == 0 else 1.2

        ax_traj.plot(std_res.w12_trajectory, std_res.w21_trajectory, "o-",
                     color="cyan", alpha=alpha_line, linewidth=lw, markersize=1.5,
                     markevery=max(1, config.n_steps // 20),
                     label="Standard GD" if i == 0 else None)

        ax_traj.plot(pre_res.w12_trajectory, pre_res.w21_trajectory, "s-",
                     color="red", alpha=alpha_line, linewidth=lw, markersize=1.5,
                     markevery=max(1, config.n_steps // 20),
                     label="Preconditioned GD" if i == 0 else None)

        ax_traj.plot(std_res.w12_trajectory[0], std_res.w21_trajectory[0], "ko",
                     markersize=8, zorder=5, markeredgewidth=1.5, markeredgecolor="white")

    ax_traj.plot(config.w12_teacher, config.w21_teacher, "w*", markersize=25,
                 label="Teacher", zorder=5, markeredgewidth=1.5, markeredgecolor="black")
    ax_traj.contour(W12, W21, loss_grid, levels=30, alpha=0.3, colors="white", linewidths=0.8)
    ax_traj.set_xlabel(rf"$w_{{12}}$ (into fast neuron, $\alpha_1={config.alpha_fast}$)", fontsize=12)
    ax_traj.set_ylabel(rf"$w_{{21}}$ (into slow neuron, $\alpha_2={config.alpha_slow}$)", fontsize=12)
    ax_traj.set_title(f"Optimization Trajectories ({config.n_trajectories} runs)",
                      fontsize=14, fontweight="bold")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(fontsize=10, loc="upper right")
    ax_traj.set_aspect("equal")

    # Right: loss curves
    ax_loss = axes[1]
    for i, (std_res, pre_res) in enumerate(zip(standard_results, precond_results)):
        alpha_line = 1.0 if i == 0 else 0.3
        lw = 2.5 if i == 0 else 1.0

        ax_loss.semilogy(std_res.loss_trajectory, color="blue", linewidth=lw,
                         alpha=alpha_line, label="Standard GD" if i == 0 else None)
        ax_loss.semilogy(pre_res.loss_trajectory, color="green", linewidth=lw,
                         alpha=alpha_line, label="Preconditioned GD" if i == 0 else None)

    ax_loss.set_xlabel("Optimization Step", fontsize=12)
    ax_loss.set_ylabel("Loss (log scale)", fontsize=12)
    ax_loss.set_title(f"Loss vs. Iteration ({config.n_steps} steps)", fontsize=14, fontweight="bold")
    ax_loss.grid(True, alpha=0.3, which="both")
    ax_loss.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_summary(
    config: Config,
    standard_results: list[OptimizationResult],
    precond_results: list[OptimizationResult],
) -> str:
    """Generate summary statistics string."""
    std_final = [r.final_loss for r in standard_results]
    pre_final = [r.final_loss for r in precond_results]

    lines = [
        "",
        "=" * 70,
        "RESULTS SUMMARY",
        "=" * 70,
        "Configuration:",
        f"  Timescales: alpha_fast={config.alpha_fast}, alpha_slow={config.alpha_slow}",
        f"  Rollout length: T={config.rollout_length}",
        f"  Optimization steps: {config.n_steps}",
        f"  Number of trajectories: {config.n_trajectories}",
        "",
        f"Standard GD (n={len(standard_results)}):",
        f"  Final loss: {np.mean(std_final):.8f} +/- {np.std(std_final):.8f}",
        f"  Best: {np.min(std_final):.8f}, Worst: {np.max(std_final):.8f}",
        "",
        f"Preconditioned GD (n={len(precond_results)}):",
        f"  Final loss: {np.mean(pre_final):.8f} +/- {np.std(pre_final):.8f}",
        f"  Best: {np.min(pre_final):.8f}, Worst: {np.max(pre_final):.8f}",
        "",
        f"Average improvement: {np.mean(std_final) / np.mean(pre_final):.1f}x lower final loss with preconditioning",
        "=" * 70,
    ]
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main(config: Config, results_dir: Path) -> None:
    """Run the full preconditioning visualization experiment."""
    rng = np.random.default_rng(config.seed)

    print("RNN Preconditioner Visualization")
    print("=" * 50)
    print(f"Timescales: alpha_fast={config.alpha_fast}, alpha_slow={config.alpha_slow}")
    print(f"Teacher weights: w12={config.w12_teacher}, w21={config.w21_teacher}")
    print(f"Optimization: {config.n_steps} steps, lr={config.learning_rate}")
    print(f"Trajectories: {config.n_trajectories}")
    print(f"Results dir: {results_dir}")
    print()

    # Compute gradient field
    print("Computing gradient field...")
    W12, W21, dL_dw12, dL_dw21, loss_grid = compute_gradient_field(config)

    # Plot gradient fields
    gradient_path = results_dir / "gradient_fields.png"
    plot_gradient_fields(config, W12, W21, dL_dw12, dL_dw21, loss_grid, str(gradient_path))
    print(f"  Saved: {gradient_path}")

    # Generate initial points
    initial_points = generate_initial_points(config, rng)

    # Run optimizations
    print(f"\nRunning {config.n_trajectories} optimization trajectories...")
    standard_results = []
    precond_results = []

    for i, (w12_init, w21_init) in enumerate(initial_points):
        print(f"  Trajectory {i + 1}/{config.n_trajectories}: init=({w12_init:.2f}, {w21_init:.2f})")

        std_res = run_optimization(w12_init, w21_init, config, precondition=False)
        standard_results.append(std_res)

        pre_res = run_optimization(w12_init, w21_init, config, precondition=True)
        precond_results.append(pre_res)

        print(f"    Standard: {std_res.final_loss:.6f}, Precond: {pre_res.final_loss:.6f}")

    # Plot trajectories
    traj_path = results_dir / "optimization_trajectories.png"
    plot_optimization_trajectories(
        config, W12, W21, loss_grid, standard_results, precond_results, str(traj_path)
    )
    print(f"\nSaved: {traj_path}")

    # Summary
    summary = print_summary(config, standard_results, precond_results)
    print(summary)

    # Save summary to file
    summary_path = results_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    # Save config
    config.to_yaml(str(results_dir / "config.yaml"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RNN Preconditioner Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py --config configs/default.yaml
    python run.py --config configs/default.yaml --n_steps 10000
    python run.py --config configs/default.yaml --alpha_fast 0.95 --alpha_slow 0.05
        """
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Allow overriding any config parameter
    parser.add_argument("--alpha_fast", type=float, help="Fast neuron timescale")
    parser.add_argument("--alpha_slow", type=float, help="Slow neuron timescale")
    parser.add_argument("--n_steps", type=int, help="Number of optimization steps")
    parser.add_argument("--n_trajectories", type=int, help="Number of trajectories")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--rollout_length", type=int, help="RNN rollout length")
    parser.add_argument("--grid_resolution", type=int, help="Grid resolution for visualization")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Apply command-line overrides
    if args.alpha_fast is not None:
        config.alpha_fast = args.alpha_fast
    if args.alpha_slow is not None:
        config.alpha_slow = args.alpha_slow
    if args.n_steps is not None:
        config.n_steps = args.n_steps
    if args.n_trajectories is not None:
        config.n_trajectories = args.n_trajectories
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.rollout_length is not None:
        config.rollout_length = args.rollout_length
    if args.grid_resolution is not None:
        config.grid_resolution = args.grid_resolution
    if args.seed is not None:
        config.seed = args.seed

    # Create results directory
    results_dir = get_timestamped_results_dir()

    main(config, results_dir)
