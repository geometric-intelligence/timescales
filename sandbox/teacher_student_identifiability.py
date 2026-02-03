#!/usr/bin/env python
"""
Teacher-Student Timescale Identifiability Experiment

Goal: Test whether a student RNN can recover teacher dynamics by changing W_rec
when the student has different timescales (alpha_i).

Interpretation: If student FAILS to match teacher -> alphas are identifiable.
                If student SUCCEEDS -> alphas are NOT identifiable (absorbed into W_rec).

Experimental sweep:
- Activation: ReLU vs Tanh
- Diagonal of W_rec: zero vs non-zero (self-connections)
- Optimizer preconditioning by alpha^-1 vs none
- Network size: N=2, N=4, N=8

Date: 2026-02-03
"""

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Literal
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Results directory (will be set with timestamp in main)
RESULTS_BASE = Path(__file__).parent / "results" / "teacher_student_identifiability"
RESULTS_DIR = RESULTS_BASE  # Will be updated with timestamp


def get_timestamped_results_dir() -> Path:
    """Create and return a timestamped results directory."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_BASE / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# BabyRNN Model (from notebook 14)
# =============================================================================

class BabyRNN(nn.Module):
    """
    Minimal N-neuron multi-timescale RNN (no external input).

    Update rule: r_{t+1} = (1 - alpha) * r_t + alpha * phi(W_rec @ r_t + b)
    """

    def __init__(
        self,
        num_neurons: int,
        alphas: torch.Tensor,
        activation: type[nn.Module] = nn.Tanh,
        zero_diag: bool = False,
    ) -> None:
        super().__init__()

        self.register_buffer("alphas", alphas.clone())
        self.num_neurons = num_neurons
        self.W_rec = nn.Linear(num_neurons, num_neurons, bias=True)
        self.activation = activation()

        self.zero_diag = zero_diag
        if zero_diag:
            self.W_rec.weight.data.fill_diagonal_(0)
            self.W_rec.weight.register_hook(lambda g: g.clone().fill_diagonal_(0))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """One-step update."""
        pre_activation = self.W_rec(r)
        activated = self.activation(pre_activation)
        return (1 - self.alphas) * r + self.alphas * activated

    def delta(self, r: torch.Tensor) -> torch.Tensor:
        """Compute delta_r = r_{t+1} - r_t."""
        pre_activation = self.W_rec(r)
        activated = self.activation(pre_activation)
        return self.alphas * (-r + activated)

    def unroll(self, r0: torch.Tensor, T: int) -> torch.Tensor:
        """Unroll RNN for T steps. Returns shape (batch, T+1, N)."""
        batch_size = r0.shape[0]
        trajectory = torch.zeros(batch_size, T + 1, self.num_neurons, device=r0.device)
        trajectory[:, 0] = r0

        r = r0
        for t in range(T):
            r = self.forward(r)
            trajectory[:, t + 1] = r

        return trajectory

    def copy_weights_from(self, other: "BabyRNN") -> None:
        """Copy W_rec and bias from another BabyRNN."""
        self.W_rec.weight.data.copy_(other.W_rec.weight.data)
        self.W_rec.bias.data.copy_(other.W_rec.bias.data)


# =============================================================================
# Training Functions
# =============================================================================

def apply_alpha_preconditioning(student: BabyRNN, eps: float = 1e-8) -> None:
    """Scale gradients by 1/alpha to compensate for timescale differences."""
    alphas = student.alphas.detach()
    preconditioner = 1.0 / (alphas + eps)

    if student.W_rec.weight.grad is not None:
        student.W_rec.weight.grad *= preconditioner.unsqueeze(1)
    if student.W_rec.bias.grad is not None:
        student.W_rec.bias.grad *= preconditioner


def train_student(
    teacher: BabyRNN,
    student: BabyRNN,
    n_epochs: int = 1000,
    batch_size: int = 64,
    T: int = 50,
    lr: float = 1e-2,
    init_scale: float = 2.0,
    lag: int = 10,
    precondition: bool = False,
    loss_type: Literal["trajectory", "vector_field"] = "trajectory",
    verbose: bool = False,
) -> dict[str, list[float]]:
    """
    Train student to match teacher.

    Args:
        teacher: Frozen teacher network.
        student: Student network with different alphas.
        loss_type: "trajectory" (MSE on unrolled states) or "vector_field" (MSE on delta).
        precondition: Scale gradients by 1/alpha.

    Returns:
        Dictionary with loss history.
    """
    num_neurons = teacher.num_neurons
    optimizer = torch.optim.Adam(student.W_rec.parameters(), lr=lr)

    losses = {"loss": [], "traj_loss": [], "vf_loss": []}

    for epoch in range(n_epochs):
        # Random state samples (on same device as model)
        r = torch.randn(batch_size, num_neurons, device=device) * init_scale

        if loss_type == "trajectory":
            # Trajectory loss
            with torch.no_grad():
                teacher_traj = teacher.unroll(r, T)
            student_traj = student.unroll(r, T)
            loss = F.mse_loss(student_traj[:, lag:, :], teacher_traj[:, lag:, :])
        else:
            # Vector field loss
            with torch.no_grad():
                teacher_delta = teacher.delta(r)
            student_delta = student.delta(r)
            loss = F.mse_loss(student_delta, teacher_delta)

        optimizer.zero_grad()
        loss.backward()

        if precondition:
            apply_alpha_preconditioning(student)

        optimizer.step()

        losses["loss"].append(loss.item())

        # Track both losses for monitoring
        with torch.no_grad():
            r_eval = torch.randn(batch_size, num_neurons, device=device) * init_scale
            # Trajectory
            teacher_traj = teacher.unroll(r_eval, T)
            student_traj = student.unroll(r_eval, T)
            traj_loss = F.mse_loss(student_traj[:, lag:, :], teacher_traj[:, lag:, :])
            # VF
            teacher_delta = teacher.delta(r_eval)
            student_delta = student.delta(r_eval)
            vf_loss = F.mse_loss(student_delta, teacher_delta)

            losses["traj_loss"].append(traj_loss.item())
            losses["vf_loss"].append(vf_loss.item())

        # Progress output every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1:4d}/{n_epochs} | Loss: {loss.item():.6f}", flush=True)

    return losses


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    num_neurons: int
    activation: str  # "Tanh" or "ReLU"
    zero_diag: bool
    precondition: bool
    loss_type: str  # "trajectory" or "vector_field"

    # Training params
    n_epochs: int = 2000
    batch_size: int = 128
    T: int = 60
    lr: float = 1e-2
    lag: int = 10
    init_scale: float = 2.0

    # Random seeds
    teacher_seed: int = 42
    student_alpha_seed: int = 123

    def name(self) -> str:
        diag_str = "nodiag" if self.zero_diag else "diag"
        precond_str = "precond" if self.precondition else "noprecond"
        return f"N{self.num_neurons}_{self.activation}_{diag_str}_{precond_str}_{self.loss_type}"


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: ExperimentConfig
    teacher_alphas: list[float]
    student_alphas: list[float]
    final_loss: float
    final_traj_loss: float
    final_vf_loss: float
    loss_history: dict[str, list[float]]
    teacher_W: list[list[float]]
    student_W_init: list[list[float]]
    student_W_final: list[list[float]]


# =============================================================================
# Run Single Experiment
# =============================================================================

def run_experiment(config: ExperimentConfig, verbose: bool = False) -> ExperimentResult:
    """Run a single teacher-student experiment."""

    N = config.num_neurons
    activation_cls = nn.Tanh if config.activation == "Tanh" else nn.ReLU

    # Generate teacher alphas (diverse timescales)
    torch.manual_seed(config.teacher_seed)
    teacher_alphas = torch.rand(N) * 0.8 + 0.1  # (0.1, 0.9)

    # Generate student alphas (different from teacher)
    torch.manual_seed(config.student_alpha_seed)
    student_alphas = torch.rand(N) * 0.8 + 0.1

    # Create teacher and move to device
    torch.manual_seed(config.teacher_seed + 1000)  # Different seed for weights
    teacher = BabyRNN(N, teacher_alphas, activation=activation_cls, zero_diag=config.zero_diag)
    teacher = teacher.to(device)

    # Create student (copy teacher's weights initially) and move to device
    student = BabyRNN(N, student_alphas, activation=activation_cls, zero_diag=config.zero_diag)
    student = student.to(device)
    student.copy_weights_from(teacher)

    # Store initial student weights
    student_W_init = student.W_rec.weight.data.cpu().clone()

    # Train
    losses = train_student(
        teacher, student,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        T=config.T,
        lr=config.lr,
        init_scale=config.init_scale,
        lag=config.lag,
        precondition=config.precondition,
        loss_type=config.loss_type,
        verbose=verbose,
    )

    return ExperimentResult(
        config=config,
        teacher_alphas=teacher_alphas.tolist(),
        student_alphas=student_alphas.tolist(),
        final_loss=losses["loss"][-1],
        final_traj_loss=losses["traj_loss"][-1],
        final_vf_loss=losses["vf_loss"][-1],
        loss_history=losses,
        teacher_W=teacher.W_rec.weight.data.cpu().tolist(),
        student_W_init=student_W_init.tolist(),
        student_W_final=student.W_rec.weight.data.cpu().tolist(),
    )


# =============================================================================
# Multi-GPU Parallel Execution
# =============================================================================

def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def run_sweep_parallel(
    configs: list[ExperimentConfig],
    gpu_ids: list[int],
    results_dir: Path,
) -> list[ExperimentResult]:
    """
    Run experiments in parallel across multiple GPUs using direct subprocess spawning.

    Spawns up to len(gpu_ids) processes at once, each with its own GPU.
    """
    import time

    n_gpus = len(gpu_ids)
    n_experiments = len(configs)
    script_path = Path(__file__).resolve()

    print(f"\nScheduling {n_experiments} experiments across {n_gpus} GPUs: {gpu_ids}")
    print("=" * 60, flush=True)

    # Prepare jobs: (config, gpu_id, config_file, result_file)
    jobs = []
    for i, config in enumerate(configs):
        config_dict = asdict(config)
        config_dict["name"] = config.name()
        config_dict["results_dir"] = str(results_dir)
        config_file = results_dir / f"_config_{i}.json"
        result_file = results_dir / f"_result_{i}.json"
        log_file = results_dir / f"_log_{i}.txt"
        gpu_id = gpu_ids[i % n_gpus]

        # Write config
        with open(config_file, "w") as f:
            json.dump(config_dict, f)

        jobs.append({
            "config": config,
            "config_dict": config_dict,
            "gpu_id": gpu_id,
            "config_file": config_file,
            "result_file": result_file,
            "log_file": log_file,
            "process": None,
            "start_time": None,
        })

    # Track running and completed jobs
    running = []
    pending = list(range(len(jobs)))
    completed_results = []

    def start_job(job_idx: int):
        job = jobs[job_idx]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])

        cmd = [
            sys.executable, str(script_path),
            "--single", str(job["config_file"]),
            "--output", str(job["result_file"]),
        ]

        log_fh = open(job["log_file"], "w")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(script_path.parent),
        )
        job["process"] = proc
        job["log_fh"] = log_fh
        job["start_time"] = time.time()

        exp_name = job["config_dict"]["name"]
        print(f"[GPU {job['gpu_id']}] Starting {exp_name} (PID {proc.pid})", flush=True)

    def check_completed():
        still_running = []
        for job_idx in running:
            job = jobs[job_idx]
            proc = job["process"]
            retcode = proc.poll()

            if retcode is not None:
                # Process finished
                elapsed = time.time() - job["start_time"]
                job["log_fh"].close()
                exp_name = job["config_dict"]["name"]

                if retcode == 0 and job["result_file"].exists():
                    with open(job["result_file"], "r") as f:
                        result_data = json.load(f)
                    result = ExperimentResult(
                        config=ExperimentConfig(**result_data["config"]),
                        teacher_alphas=result_data["teacher_alphas"],
                        student_alphas=result_data["student_alphas"],
                        final_loss=result_data["final_loss"],
                        final_traj_loss=result_data["final_traj_loss"],
                        final_vf_loss=result_data["final_vf_loss"],
                        loss_history=result_data["loss_history"],
                        teacher_W=result_data["teacher_W"],
                        student_W_init=result_data["student_W_init"],
                        student_W_final=result_data["student_W_final"],
                    )
                    completed_results.append(result)
                    print(f"[GPU {job['gpu_id']}] ✓ {exp_name} completed in {elapsed:.1f}s "
                          f"(loss: {result.final_traj_loss:.2e})", flush=True)
                else:
                    print(f"[GPU {job['gpu_id']}] ✗ {exp_name} failed after {elapsed:.1f}s", flush=True)
                    # Read log for error
                    if job["log_file"].exists():
                        with open(job["log_file"], "r") as f:
                            log_content = f.read()
                        if log_content:
                            print(f"    Log: {log_content[-300:]}", flush=True)
            else:
                still_running.append(job_idx)
        return still_running

    # Main loop: keep n_gpus jobs running at a time
    while pending or running:
        # Start new jobs if we have capacity
        while pending and len(running) < n_gpus:
            job_idx = pending.pop(0)
            start_job(job_idx)
            running.append(job_idx)

        # Wait a bit then check for completed jobs
        time.sleep(0.5)
        running = check_completed()

    # Clean up temp files
    for job in jobs:
        for f in [job["config_file"], job["result_file"], job["log_file"]]:
            if f.exists():
                f.unlink()

    return completed_results


def run_single_from_config(config_file: str, output_file: str) -> None:
    """
    Run a single experiment from a config file (called by subprocess).
    """
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    # Remove extra keys not part of ExperimentConfig
    config_dict.pop("name", None)
    config_dict.pop("results_dir", None)

    config = ExperimentConfig(**config_dict)
    result = run_experiment(config, verbose=False)

    # Serialize result to JSON
    result_data = {
        "config": asdict(result.config),
        "teacher_alphas": result.teacher_alphas,
        "student_alphas": result.student_alphas,
        "final_loss": result.final_loss,
        "final_traj_loss": result.final_traj_loss,
        "final_vf_loss": result.final_vf_loss,
        "loss_history": result.loss_history,
        "teacher_W": result.teacher_W,
        "student_W_init": result.student_W_init,
        "student_W_final": result.student_W_final,
    }

    with open(output_file, "w") as f:
        json.dump(result_data, f)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_phase_portrait_2d(
    rnn: BabyRNN,
    ax: plt.Axes,
    xlim: tuple[float, float] = (-2, 2),
    ylim: tuple[float, float] = (-2, 2),
    n_grid: int = 20,
    trajectories: list[torch.Tensor] | None = None,
    title: str = "",
) -> None:
    """Plot 2D vector field."""
    assert rnn.num_neurons == 2

    r1 = np.linspace(xlim[0], xlim[1], n_grid)
    r2 = np.linspace(ylim[0], ylim[1], n_grid)
    R1, R2 = np.meshgrid(r1, r2)

    r_grid = torch.tensor(
        np.stack([R1.flatten(), R2.flatten()], axis=1),
        dtype=torch.float32
    )

    with torch.no_grad():
        delta = rnn.delta(r_grid).numpy()

    U = delta[:, 0].reshape(R1.shape)
    V = delta[:, 1].reshape(R1.shape)

    # Normalize arrows
    speed = np.sqrt(U**2 + V**2)
    speed[speed == 0] = 1.0
    U_norm = U / speed
    V_norm = V / speed

    ax.quiver(R1, R2, U_norm, V_norm, angles="xy", scale_units="xy", scale=3.5,
              width=0.004, alpha=0.6)

    if trajectories is not None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(trajectories)))
        for traj, color in zip(trajectories, colors):
            traj_np = traj.numpy()
            ax.plot(traj_np[:, 0], traj_np[:, 1], "-", color=color, linewidth=1.5, alpha=0.8)
            ax.scatter(traj_np[0, 0], traj_np[0, 1], s=40, color=color, marker="o", zorder=5)
            ax.scatter(traj_np[-1, 0], traj_np[-1, 1], s=40, color=color, marker="x", zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$r_1$")
    ax.set_ylabel(r"$r_2$")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def create_comparison_figure(
    teacher: BabyRNN,
    student_init: BabyRNN,
    student_trained: BabyRNN,
    losses: dict,
    config: ExperimentConfig,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Create a figure with:
    - Row 1: Flow maps (teacher, untrained student, trained student)
    - Row 2: Sample trajectories comparison, Loss curve
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1])

    # Generate trajectories from same initial conditions
    torch.manual_seed(999)
    n_traj = 5
    T_viz = 80
    r0 = torch.randn(n_traj, 2) * 1.5

    with torch.no_grad():
        trajs_teacher = [teacher.unroll(r0[i:i+1], T_viz).squeeze(0) for i in range(n_traj)]
        trajs_init = [student_init.unroll(r0[i:i+1], T_viz).squeeze(0) for i in range(n_traj)]
        trajs_trained = [student_trained.unroll(r0[i:i+1], T_viz).squeeze(0) for i in range(n_traj)]

    lim = 2.5

    # Row 1: Phase portraits
    ax1 = fig.add_subplot(gs[0, 0])
    plot_phase_portrait_2d(teacher, ax1, (-lim, lim), (-lim, lim),
                           trajectories=trajs_teacher, title="Teacher")

    ax2 = fig.add_subplot(gs[0, 1])
    plot_phase_portrait_2d(student_init, ax2, (-lim, lim), (-lim, lim),
                           trajectories=trajs_init, title="Student (untrained)")

    ax3 = fig.add_subplot(gs[0, 2])
    plot_phase_portrait_2d(student_trained, ax3, (-lim, lim), (-lim, lim),
                           trajectories=trajs_trained, title="Student (trained)")

    # Row 1, col 4: Alpha comparison
    ax4 = fig.add_subplot(gs[0, 3])
    x = np.arange(2)
    width = 0.35
    ax4.bar(x - width/2, teacher.alphas.numpy(), width, label="Teacher", color="tab:blue")
    ax4.bar(x + width/2, student_trained.alphas.numpy(), width, label="Student", color="tab:orange")
    ax4.set_xticks(x)
    ax4.set_xticklabels([r"$\alpha_1$", r"$\alpha_2$"])
    ax4.set_ylabel(r"$\alpha$")
    ax4.set_title("Timescales")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Row 2: Trajectory comparison for one IC
    ax5 = fig.add_subplot(gs[1, 0:2])
    t_steps = np.arange(T_viz + 1)
    traj_t = trajs_teacher[0].numpy()
    traj_s = trajs_trained[0].numpy()
    ax5.plot(t_steps, traj_t[:, 0], "b-", label=r"Teacher $r_1$", linewidth=2)
    ax5.plot(t_steps, traj_t[:, 1], "b--", label=r"Teacher $r_2$", linewidth=2)
    ax5.plot(t_steps, traj_s[:, 0], "r-", label=r"Student $r_1$", linewidth=2, alpha=0.7)
    ax5.plot(t_steps, traj_s[:, 1], "r--", label=r"Student $r_2$", linewidth=2, alpha=0.7)
    ax5.set_xlabel("Time step")
    ax5.set_ylabel("Activation")
    ax5.set_title("Trajectory Comparison (one IC)")
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)

    # Row 2: Loss curve
    ax6 = fig.add_subplot(gs[1, 2:4])
    epochs = np.arange(1, len(losses["loss"]) + 1)
    ax6.semilogy(epochs, losses["traj_loss"], label="Trajectory Loss", linewidth=2)
    ax6.semilogy(epochs, losses["vf_loss"], label="VF Loss", linewidth=2, alpha=0.7)
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Loss (log)")
    ax6.set_title(f"Training Loss (optimizing {config.loss_type})")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Title
    diag_str = "no self-conn" if config.zero_diag else "with self-conn"
    precond_str = "preconditioned" if config.precondition else "no precond"
    fig.suptitle(
        f"{config.activation} | {diag_str} | {precond_str} | Final traj loss: {losses['traj_loss'][-1]:.2e}",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Main Sweep
# =============================================================================

def run_sweep(
    gpu_ids: list[int] | None = None,
    parallel: bool = True,
    results_dir: Path | None = None,
):
    """Run the full experimental sweep.

    Args:
        gpu_ids: List of GPU IDs to use. If None, uses all available GPUs.
        parallel: If True and multiple GPUs available, run in parallel.
        results_dir: Directory to save results. If None, creates timestamped dir.

    Returns:
        Tuple of (results list, results directory path)
    """
    global RESULTS_DIR

    # Create timestamped results directory
    if results_dir is None:
        results_dir = get_timestamped_results_dir()
    RESULTS_DIR = results_dir

    print(f"Results will be saved to: {results_dir}")

    # Sweep parameters
    activations = ["Tanh", "ReLU"]
    zero_diags = [False, True]  # with/without self-connections
    preconditions = [False, True]
    loss_types = ["trajectory"]  # Focus on trajectory loss for now
    num_neurons_list = [2]  # Start with 2 for visualization

    # Generate all combinations
    combinations = list(itertools.product(
        num_neurons_list, activations, zero_diags, preconditions, loss_types
    ))

    configs = [
        ExperimentConfig(
            num_neurons=N,
            activation=act,
            zero_diag=zd,
            precondition=pc,
            loss_type=lt,
        )
        for N, act, zd, pc, lt in combinations
    ]

    print(f"Running {len(configs)} experiments...")
    print("=" * 60)

    # Determine GPU setup
    if gpu_ids is None:
        gpu_ids = get_available_gpus()

    # Use parallel execution if multiple GPUs and parallel=True
    if parallel and len(gpu_ids) > 1:
        print(f"Using parallel execution on GPUs: {gpu_ids}")
        all_results = run_sweep_parallel(configs, gpu_ids, results_dir)
    else:
        # Sequential execution
        gpu_id = gpu_ids[0] if gpu_ids else 0
        print(f"Using sequential execution on GPU: {gpu_id}")

        all_results = []
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] {config.name()}")
            result = run_experiment(config, verbose=False)
            all_results.append(result)

            print(f"  Final traj loss: {result.final_traj_loss:.4e}")
            print(f"  Final VF loss:   {result.final_vf_loss:.4e}")
            print(f"  Teacher alphas:  {np.array(result.teacher_alphas).round(3)}")
            print(f"  Student alphas:  {np.array(result.student_alphas).round(3)}")

    return all_results, results_dir


def generate_visualizations(results: list[ExperimentResult]):
    """Generate comparison figures for N=2 experiments."""

    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    for result in results:
        if result.config.num_neurons != 2:
            continue

        config = result.config
        N = config.num_neurons
        activation_cls = nn.Tanh if config.activation == "Tanh" else nn.ReLU

        # Recreate networks
        teacher_alphas = torch.tensor(result.teacher_alphas)
        student_alphas = torch.tensor(result.student_alphas)

        teacher = BabyRNN(N, teacher_alphas, activation=activation_cls, zero_diag=config.zero_diag)
        teacher.W_rec.weight.data = torch.tensor(result.teacher_W)

        student_init = BabyRNN(N, student_alphas, activation=activation_cls, zero_diag=config.zero_diag)
        student_init.W_rec.weight.data = torch.tensor(result.student_W_init)
        student_init.W_rec.bias.data = teacher.W_rec.bias.data.clone()  # Same bias initially

        student_trained = BabyRNN(N, student_alphas, activation=activation_cls, zero_diag=config.zero_diag)
        student_trained.W_rec.weight.data = torch.tensor(result.student_W_final)
        # Note: bias is also learned but we didn't save it separately

        # Generate figure
        save_path = RESULTS_DIR / f"{config.name()}.png"
        create_comparison_figure(
            teacher, student_init, student_trained,
            result.loss_history, config,
            save_path=save_path
        )
        plt.close()


def create_summary_figure(results: list[ExperimentResult]):
    """Create a summary bar chart of final losses across conditions."""

    # Filter to N=2 trajectory experiments
    results_2d = [r for r in results if r.config.num_neurons == 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, activation in enumerate(["Tanh", "ReLU"]):
        ax = axes[ax_idx]

        # Get results for this activation
        act_results = [r for r in results_2d if r.config.activation == activation]

        # Group by (zero_diag, precondition)
        conditions = []
        losses = []
        colors = []

        for r in act_results:
            diag_str = "No self-conn" if r.config.zero_diag else "Self-conn"
            precond_str = "Precond" if r.config.precondition else "No precond"
            conditions.append(f"{diag_str}\n{precond_str}")
            losses.append(r.final_traj_loss)

            # Color by success (low loss = green, high = red)
            colors.append("tab:green" if r.final_traj_loss < 0.01 else "tab:red")

        x = np.arange(len(conditions))
        bars = ax.bar(x, losses, color=colors, alpha=0.7, edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel("Final Trajectory Loss")
        ax.set_title(f"{activation} Activation")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, loss in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f"{loss:.2e}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Teacher-Student Identifiability: Can student match teacher with different alphas?",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    save_path = RESULTS_DIR / "summary.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved summary: {save_path}")

    return fig


def save_results_json(results: list[ExperimentResult]):
    """Save results to JSON."""

    data = []
    for r in results:
        data.append({
            "name": r.config.name(),
            "config": asdict(r.config),
            "teacher_alphas": r.teacher_alphas,
            "student_alphas": r.student_alphas,
            "final_loss": r.final_loss,
            "final_traj_loss": r.final_traj_loss,
            "final_vf_loss": r.final_vf_loss,
        })

    save_path = RESULTS_DIR / "results.json"
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results: {save_path}")


# =============================================================================
# Main
# =============================================================================

class TeeOutput:
    """Write to both stdout and a log file."""
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    parser = argparse.ArgumentParser(
        description="Teacher-Student Timescale Identifiability Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on all available GPUs in parallel
    python teacher_student_identifiability.py

    # Run on specific GPUs
    python teacher_student_identifiability.py --gpus 0,1,2,3

    # Run sequentially on single GPU
    python teacher_student_identifiability.py --gpus 0 --sequential

    # Run sequentially (no parallelism)
    python teacher_student_identifiability.py --sequential
        """
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). Default: all available."
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run experiments sequentially (no parallelism)"
    )
    # Hidden arguments for subprocess execution
    parser.add_argument("--single", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Handle subprocess mode (--single)
    if args.single:
        run_single_from_config(args.single, args.output)
        return

    # Create timestamped results directory first (for logging)
    results_dir = get_timestamped_results_dir()

    # Set up logging to file
    log_file = results_dir / "experiment.log"
    tee = TeeOutput(log_file)
    sys.stdout = tee
    sys.stderr = tee

    print(f"Log file: {log_file}")
    print(f"Monitor with: tail -f {log_file}")
    print()

    try:
        # Parse GPU IDs
        gpu_ids = None
        if args.gpus is not None:
            gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

        # Run sweep
        parallel = not args.sequential
        results, results_dir = run_sweep(
            gpu_ids=gpu_ids, parallel=parallel, results_dir=results_dir
        )

        # Save results
        save_results_json(results)

        # Generate visualizations
        generate_visualizations(results)

        # Create summary figure
        create_summary_figure(results)

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print(f"Results saved to: {results_dir}")
        print(f"Log file: {log_file}")
        print("=" * 60)

    finally:
        # Restore stdout/stderr
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
