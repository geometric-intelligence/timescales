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
# Effective Timescales Analysis
# =============================================================================

def get_activation_derivative(rnn: BabyRNN):
    """
    Return the derivative function for the RNN's activation.

    For tanh: phi'(z) = 1 - tanh(z)^2
    For ReLU: phi'(z) = (z > 0).float()
    For Identity: phi'(z) = 1
    """
    if isinstance(rnn.activation, nn.Tanh):
        def phi_prime(z):
            return 1 - torch.tanh(z) ** 2
        return phi_prime
    elif isinstance(rnn.activation, nn.ReLU):
        def phi_prime(z):
            return (z > 0).float()
        return phi_prime
    elif isinstance(rnn.activation, nn.Identity):
        def phi_prime(z):
            return torch.ones_like(z)
        return phi_prime
    else:
        raise ValueError(f"Unknown activation: {type(rnn.activation)}")


def compute_jacobian(rnn: BabyRNN, r: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian J = (I - A) + A @ D @ W_rec at state r.

    Where:
        - A = diag(alpha) is the timescale matrix
        - D = diag(phi'(z)) is the gain matrix at z = W_rec @ r + b
        - W_rec is the recurrent weight matrix

    Args:
        rnn: The BabyRNN model.
        r: State vector, shape (num_neurons,) or (batch, num_neurons).

    Returns:
        Jacobian matrix, shape (num_neurons, num_neurons) or (batch, num_neurons, num_neurons).
    """
    # Handle batch dimension
    if r.dim() == 1:
        r = r.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    N = rnn.num_neurons

    # Get weight matrix and bias
    W = rnn.W_rec.weight.data  # (N, N)
    b = rnn.W_rec.bias.data    # (N,)
    alpha = rnn.alphas         # (N,)

    # Compute pre-activation: z = W @ r + b
    z = F.linear(r, W, b)  # (batch, N)

    # Compute gain: d = phi'(z)
    phi_prime = get_activation_derivative(rnn)
    d = phi_prime(z)  # (batch, N)

    # Build Jacobian: J = (I - diag(alpha)) + diag(alpha * d) @ W
    I_minus_A = torch.diag(1 - alpha)  # (N, N)

    # g = alpha * d, shape (batch, N)
    g = alpha.unsqueeze(0) * d  # (batch, N)

    # diag(g) @ W = row-scale W by g
    J = I_minus_A.unsqueeze(0) + g.unsqueeze(2) * W.unsqueeze(0)  # (batch, N, N)

    if squeeze_output:
        J = J.squeeze(0)

    return J


def compute_effective_timescales(
    rnn: BabyRNN,
    T: int = 500,
    burn_in: int = 100,
    stride: int = 5,
    r0: torch.Tensor | None = None,
    init_scale: float = 1.0,
) -> dict:
    """
    Compute effective timescales along a rollout.

    Args:
        rnn: The BabyRNN model.
        T: Total rollout steps.
        burn_in: Steps to skip at the beginning.
        stride: Compute every `stride` steps after burn-in.
        r0: Initial state (optional, random if None).
        init_scale: Scale for random initial state.

    Returns:
        Dictionary with:
            - eigvals: (n_samples, N) complex eigenvalues
            - rho: (n_samples,) spectral radius (max |eigenvalue|)
            - sigma: (n_samples,) largest singular value
            - tau_dom: (n_samples,) dominant stable timescale
            - tau_all: (n_samples, N) all timescales (inf for unstable)
            - times: (n_samples,) time indices
            - trajectory: (T+1, N) full trajectory
    """
    N = rnn.num_neurons

    # Initialize
    if r0 is None:
        r0 = torch.randn(1, N, device=next(rnn.parameters()).device) * init_scale
    elif r0.dim() == 1:
        r0 = r0.unsqueeze(0)

    # Rollout
    with torch.no_grad():
        trajectory = rnn.unroll(r0, T).squeeze(0).cpu()  # (T+1, N)

    # Sample times
    sample_times = list(range(burn_in, T, stride))
    n_samples = len(sample_times)

    # Storage
    eigvals_all = np.zeros((n_samples, N), dtype=np.complex128)
    rho_all = np.zeros(n_samples)
    sigma_all = np.zeros(n_samples)
    tau_dom_all = np.zeros(n_samples)
    tau_all = np.zeros((n_samples, N))

    eps = 1e-8

    for i, t in enumerate(sample_times):
        r_t = trajectory[t]  # (N,)

        # Compute Jacobian (move to CPU for numpy operations)
        J = compute_jacobian(rnn, r_t.to(next(rnn.parameters()).device)).cpu().numpy()

        # Eigenvalues
        eigvals = np.linalg.eigvals(J)
        eigvals_all[i] = eigvals

        # Spectral radius
        mags = np.abs(eigvals)
        rho = np.max(mags)
        rho_all[i] = rho

        # Largest singular value
        sigma = np.linalg.svd(J, compute_uv=False)[0]
        sigma_all[i] = sigma

        # Convert to timescales
        taus = np.zeros(N)
        for j, mag in enumerate(mags):
            if mag < 1 - eps:
                taus[j] = -1.0 / np.log(mag + eps)  # positive, finite
            else:
                taus[j] = np.inf  # marginal or unstable
        tau_all[i] = taus

        # Dominant stable timescale (largest finite)
        finite_taus = taus[np.isfinite(taus)]
        tau_dom_all[i] = np.max(finite_taus) if len(finite_taus) > 0 else np.inf

    return {
        "eigvals": eigvals_all,
        "rho": rho_all,
        "sigma": sigma_all,
        "tau_dom": tau_dom_all,
        "tau_all": tau_all,
        "times": np.array(sample_times),
        "trajectory": trajectory.numpy(),
    }


def compute_timescale_summary(results: dict) -> dict:
    """
    Compute summary statistics from effective timescale results.

    Args:
        results: Output from compute_effective_timescales().

    Returns:
        Dictionary with summary statistics.
    """
    rho = results["rho"]
    sigma = results["sigma"]
    tau_dom = results["tau_dom"]
    eigvals = results["eigvals"]

    # Filter finite tau_dom values
    finite_tau = tau_dom[np.isfinite(tau_dom)]

    # Non-normality ratio
    kappa = sigma / (rho + 1e-8)

    return {
        "rho_median": float(np.median(rho)),
        "rho_q25": float(np.percentile(rho, 25)),
        "rho_q75": float(np.percentile(rho, 75)),
        "sigma_median": float(np.median(sigma)),
        "tau_dom_median": float(np.median(finite_tau)) if len(finite_tau) > 0 else float("inf"),
        "tau_dom_q25": float(np.percentile(finite_tau, 25)) if len(finite_tau) > 0 else float("inf"),
        "tau_dom_q75": float(np.percentile(finite_tau, 75)) if len(finite_tau) > 0 else float("inf"),
        "frac_unstable": float(np.mean(rho >= 1.0)),
        "kappa_median": float(np.median(kappa)),
        "kappa_max": float(np.max(kappa)),
        "eigval_mean_real": float(np.mean(eigvals.real)),
        "eigval_mean_imag": float(np.mean(np.abs(eigvals.imag))),
    }


# =============================================================================
# Training Functions
# =============================================================================

def compute_vf_loss(
    teacher_delta: torch.Tensor,
    student_delta: torch.Tensor,
    normalize_mode: str = "none",
) -> torch.Tensor:
    """
    Compute vector field loss between teacher and student deltas.

    Args:
        teacher_delta: Teacher's delta (r_next - r), shape (batch, num_neurons).
        student_delta: Student's delta, shape (batch, num_neurons).
        normalize_mode: "none" (MSE), "cosine" (1 - cos similarity), or "angular_mse" (MSE on unit vectors).

    Returns:
        Scalar loss tensor.
    """
    eps = 1e-8
    if normalize_mode == "none":
        return F.mse_loss(student_delta, teacher_delta)
    elif normalize_mode == "cosine":
        cos_sim = F.cosine_similarity(student_delta, teacher_delta, dim=-1, eps=eps)
        return (1 - cos_sim).mean()
    elif normalize_mode == "angular_mse":
        teacher_norm = teacher_delta / (teacher_delta.norm(dim=-1, keepdim=True) + eps)
        student_norm = student_delta / (student_delta.norm(dim=-1, keepdim=True) + eps)
        return F.mse_loss(student_norm, teacher_norm)
    else:
        raise ValueError(f"Unknown normalize_mode: {normalize_mode}")


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
    vf_normalize_mode: str = "none",
    verbose: bool = False,
) -> dict[str, list[float]]:
    """
    Train student to match teacher.

    Args:
        teacher: Frozen teacher network.
        student: Student network with different alphas.
        loss_type: "trajectory" (MSE on unrolled states) or "vector_field" (VF loss).
        vf_normalize_mode: Normalization for VF loss ("none", "cosine", "angular_mse").
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
            # Vector field loss with normalization mode
            with torch.no_grad():
                teacher_delta = teacher.delta(r)
            student_delta = student.delta(r)
            loss = compute_vf_loss(teacher_delta, student_delta, vf_normalize_mode)

        optimizer.zero_grad()
        loss.backward()

        if precondition:
            apply_alpha_preconditioning(student)

        optimizer.step()

        losses["loss"].append(loss.item())

        # Track both losses for monitoring (always use "none" mode for consistent comparison)
        with torch.no_grad():
            r_eval = torch.randn(batch_size, num_neurons, device=device) * init_scale
            # Trajectory
            teacher_traj = teacher.unroll(r_eval, T)
            student_traj = student.unroll(r_eval, T)
            traj_loss = F.mse_loss(student_traj[:, lag:, :], teacher_traj[:, lag:, :])
            # VF (always MSE for monitoring)
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

    # Vector field loss normalization mode: "none", "cosine", or "angular_mse"
    vf_normalize_mode: str = "none"

    # Random seeds (used when custom params are not provided)
    teacher_seed: int = 42
    student_alpha_seed: int = 123

    # Custom teacher/student parameters (None = random initialization)
    teacher_W: list[list[float]] | None = None
    teacher_b: list[float] | None = None
    teacher_alphas: list[float] | None = None
    student_alphas: list[float] | None = None

    def name(self) -> str:
        diag_str = "nodiag" if self.zero_diag else "diag"
        precond_str = "precond" if self.precondition else "noprecond"
        vf_mode_str = f"_{self.vf_normalize_mode}" if self.loss_type == "vector_field" else ""
        custom_str = "_custom" if self.teacher_W is not None else ""
        return f"N{self.num_neurons}_{self.activation}_{diag_str}_{precond_str}_{self.loss_type}{vf_mode_str}{custom_str}"


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
    teacher_b: list[float]
    student_W_init: list[list[float]]
    student_W_final: list[list[float]]
    student_b_final: list[float]
    # Effective timescale summaries
    teacher_timescale_summary: dict | None = None
    student_timescale_summary: dict | None = None


# =============================================================================
# Run Single Experiment
# =============================================================================

def get_activation_class(name: str) -> type[nn.Module]:
    """Get activation class from string name."""
    if name == "Tanh":
        return nn.Tanh
    elif name == "ReLU":
        return nn.ReLU
    elif name == "Identity":
        return nn.Identity
    else:
        raise ValueError(f"Unknown activation: {name}. Supported: Tanh, ReLU, Identity")


def run_experiment(config: ExperimentConfig, verbose: bool = False) -> ExperimentResult:
    """Run a single teacher-student experiment."""

    N = config.num_neurons
    activation_cls = get_activation_class(config.activation)

    # Use custom teacher alphas or generate random ones
    if config.teacher_alphas is not None:
        teacher_alphas = torch.tensor(config.teacher_alphas, dtype=torch.float32)
    else:
        torch.manual_seed(config.teacher_seed)
        teacher_alphas = torch.rand(N) * 0.8 + 0.1  # (0.1, 0.9)

    # Use custom student alphas or generate random ones
    if config.student_alphas is not None:
        student_alphas = torch.tensor(config.student_alphas, dtype=torch.float32)
    else:
        torch.manual_seed(config.student_alpha_seed)
        student_alphas = torch.rand(N) * 0.8 + 0.1

    # Create teacher and move to device
    torch.manual_seed(config.teacher_seed + 1000)  # Different seed for weights
    teacher = BabyRNN(N, teacher_alphas, activation=activation_cls, zero_diag=config.zero_diag)
    teacher = teacher.to(device)

    # Apply custom teacher weights if provided
    if config.teacher_W is not None:
        teacher.W_rec.weight.data = torch.tensor(config.teacher_W, dtype=torch.float32, device=device)
    if config.teacher_b is not None:
        teacher.W_rec.bias.data = torch.tensor(config.teacher_b, dtype=torch.float32, device=device)

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
        vf_normalize_mode=config.vf_normalize_mode,
        verbose=verbose,
    )

    # Compute effective timescale summaries
    teacher_ts_summary = None
    student_ts_summary = None
    try:
        teacher_ts = compute_effective_timescales(teacher, T=200, burn_in=30, stride=5)
        student_ts = compute_effective_timescales(student, T=200, burn_in=30, stride=5)
        teacher_ts_summary = compute_timescale_summary(teacher_ts)
        student_ts_summary = compute_timescale_summary(student_ts)
    except Exception as e:
        print(f"Warning: Could not compute timescale summaries: {e}")

    return ExperimentResult(
        config=config,
        teacher_alphas=teacher_alphas.tolist(),
        student_alphas=student_alphas.tolist(),
        final_loss=losses["loss"][-1],
        final_traj_loss=losses["traj_loss"][-1],
        final_vf_loss=losses["vf_loss"][-1],
        loss_history=losses,
        teacher_W=teacher.W_rec.weight.data.cpu().tolist(),
        teacher_b=teacher.W_rec.bias.data.cpu().tolist(),
        student_W_init=student_W_init.tolist(),
        student_W_final=student.W_rec.weight.data.cpu().tolist(),
        student_b_final=student.W_rec.bias.data.cpu().tolist(),
        teacher_timescale_summary=teacher_ts_summary,
        student_timescale_summary=student_ts_summary,
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
                        teacher_b=result_data["teacher_b"],
                        student_W_init=result_data["student_W_init"],
                        student_W_final=result_data["student_W_final"],
                        student_b_final=result_data["student_b_final"],
                        teacher_timescale_summary=result_data.get("teacher_timescale_summary"),
                        student_timescale_summary=result_data.get("student_timescale_summary"),
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
        "teacher_b": result.teacher_b,
        "student_W_init": result.student_W_init,
        "student_W_final": result.student_W_final,
        "student_b_final": result.student_b_final,
        "teacher_timescale_summary": result.teacher_timescale_summary,
        "student_timescale_summary": result.student_timescale_summary,
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


def plot_effective_timescales(
    teacher: BabyRNN,
    student_init: BabyRNN,
    student_trained: BabyRNN,
    config: ExperimentConfig,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot effective timescales comparison:
    - Left: Jacobian eigenvalues (teacher vs untrained student)
    - Right: Jacobian eigenvalues (teacher vs trained student)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Compute effective timescales for all three
    teacher_ts = compute_effective_timescales(teacher, T=300, burn_in=50, stride=5)
    student_init_ts = compute_effective_timescales(student_init, T=300, burn_in=50, stride=5)
    student_trained_ts = compute_effective_timescales(student_trained, T=300, burn_in=50, stride=5)

    teacher_eigs = teacher_ts["eigvals"].flatten()
    student_init_eigs = student_init_ts["eigvals"].flatten()
    student_trained_eigs = student_trained_ts["eigvals"].flatten()

    # Unit circle coordinates
    theta = np.linspace(0, 2 * np.pi, 100)

    # Left: Teacher vs Untrained Student
    ax = axes[0]
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit circle")
    ax.scatter(teacher_eigs.real, teacher_eigs.imag, alpha=0.6, s=25, label="Teacher", c="tab:blue")
    ax.scatter(student_init_eigs.real, student_init_eigs.imag, alpha=0.6, s=25, label="Student (init)", c="tab:orange")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Teacher vs Untrained Student")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # Right: Teacher vs Trained Student
    ax = axes[1]
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit circle")
    ax.scatter(teacher_eigs.real, teacher_eigs.imag, alpha=0.6, s=25, label="Teacher", c="tab:blue")
    ax.scatter(student_trained_eigs.real, student_trained_eigs.imag, alpha=0.6, s=25, label="Student (trained)", c="tab:green")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Teacher vs Trained Student")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # Title
    diag_str = "no self-conn" if config.zero_diag else "with self-conn"
    precond_str = "precond" if config.precondition else "no precond"
    fig.suptitle(
        f"Jacobian Eigenvalues: {config.activation} | {diag_str} | {precond_str}",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def create_loss_curves_figure(
    results: list[ExperimentResult],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Create a figure showing VF loss curves for all conditions on one plot.
    Only includes vector_field training results.
    Uses shared y-axis for adequate scale comparison.
    """
    # Filter to only vector_field training
    vf_results = [r for r in results if r.config.loss_type == "vector_field"]

    if not vf_results:
        print("No VF training results available for loss curves figure")
        return None

    # Determine global y-limits
    all_losses = []
    for r in vf_results:
        all_losses.extend(r.loss_history.get("loss", []))

    if not all_losses:
        print("No loss data available for loss curves figure")
        return None

    y_min = min(all_losses) * 0.5
    y_max = max(all_losses) * 2.0

    # Create figure with subplots per activation
    activations = sorted(set(r.config.activation for r in vf_results))
    n_activations = len(activations)

    fig, axes = plt.subplots(1, n_activations, figsize=(6 * n_activations, 5), sharey=True)
    if n_activations == 1:
        axes = [axes]

    # Color map for different conditions
    # Group by (zero_diag, precondition, vf_normalize_mode)
    condition_colors = {
        ("diag", "noprecond", "none"): "tab:blue",
        ("diag", "noprecond", "cosine"): "tab:cyan",
        ("diag", "noprecond", "angular_mse"): "deepskyblue",
        ("diag", "precond", "none"): "tab:orange",
        ("diag", "precond", "cosine"): "gold",
        ("diag", "precond", "angular_mse"): "coral",
        ("nodiag", "noprecond", "none"): "tab:green",
        ("nodiag", "noprecond", "cosine"): "limegreen",
        ("nodiag", "noprecond", "angular_mse"): "mediumseagreen",
        ("nodiag", "precond", "none"): "tab:red",
        ("nodiag", "precond", "cosine"): "lightcoral",
        ("nodiag", "precond", "angular_mse"): "indianred",
    }

    linestyles = {
        "none": "-",
        "cosine": "--",
        "angular_mse": ":",
    }

    for ax_idx, activation in enumerate(activations):
        ax = axes[ax_idx]

        # Get results for this activation
        act_results = [r for r in vf_results if r.config.activation == activation]

        for r in act_results:
            n_epochs = len(r.loss_history["loss"])
            epochs = np.arange(1, n_epochs + 1)

            # Build condition key
            diag_str = "nodiag" if r.config.zero_diag else "diag"
            precond_str = "precond" if r.config.precondition else "noprecond"
            vf_mode = r.config.vf_normalize_mode

            condition_key = (diag_str, precond_str, vf_mode)
            color = condition_colors.get(condition_key, "gray")
            linestyle = linestyles.get(vf_mode, "-")

            # Build label
            label = f"{diag_str}, {precond_str}, {vf_mode}"

            ax.semilogy(epochs, r.loss_history["loss"],
                       label=label, color=color, linewidth=1.5,
                       linestyle=linestyle, alpha=0.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("VF Loss (log scale)")
        ax.set_title(f"{activation}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)

    fig.suptitle("Vector Field Training: Loss Curves by Condition", fontsize=14, fontweight="bold")
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
    custom_teacher_params: dict | None = None,
    sweep_params: dict | None = None,
):
    """Run the full experimental sweep.

    Args:
        gpu_ids: List of GPU IDs to use. If None, uses all available GPUs.
        parallel: If True and multiple GPUs available, run in parallel.
        results_dir: Directory to save results. If None, creates timestamped dir.
        custom_teacher_params: Optional dict with teacher_W, teacher_b, teacher_alphas, student_alphas.
        sweep_params: Optional dict with sweep configuration (activations, loss_types, n_epochs, etc.).

    Returns:
        Tuple of (results list, results directory path)
    """
    global RESULTS_DIR

    # Create timestamped results directory
    if results_dir is None:
        results_dir = get_timestamped_results_dir()
    RESULTS_DIR = results_dir

    print(f"Results will be saved to: {results_dir}")

    # Extract custom teacher params if provided
    teacher_W = custom_teacher_params.get("teacher_W") if custom_teacher_params else None
    teacher_b = custom_teacher_params.get("teacher_b") if custom_teacher_params else None
    teacher_alphas = custom_teacher_params.get("teacher_alphas") if custom_teacher_params else None
    student_alphas = custom_teacher_params.get("student_alphas") if custom_teacher_params else None

    if custom_teacher_params:
        print("Using custom teacher parameters:")
        print(f"  teacher_W: {teacher_W}")
        print(f"  teacher_b: {teacher_b}")
        print(f"  teacher_alphas: {teacher_alphas}")
        print(f"  student_alphas: {student_alphas}")

    # Sweep parameters - use sweep_params if provided, otherwise defaults
    if sweep_params:
        activations = sweep_params.get("activations", ["Tanh", "ReLU", "Identity"])
        zero_diags = sweep_params.get("zero_diags", [False, True])
        preconditions = sweep_params.get("preconditions", [False, True])
        loss_types = sweep_params.get("loss_types", ["trajectory", "vector_field"])
        vf_normalize_modes = sweep_params.get("vf_normalize_modes", ["none", "cosine", "angular_mse"])
        num_neurons_list = sweep_params.get("num_neurons_list", [2])
        # Training hyperparameters
        n_epochs = sweep_params.get("n_epochs", 2000)
        batch_size = sweep_params.get("batch_size", 128)
        T = sweep_params.get("T", 60)
        lr = sweep_params.get("lr", 0.01)
        lag = sweep_params.get("lag", 10)
        init_scale = sweep_params.get("init_scale", 2.0)
        print("Using sweep config:")
        print(f"  activations: {activations}")
        print(f"  zero_diags: {zero_diags}")
        print(f"  preconditions: {preconditions}")
        print(f"  loss_types: {loss_types}")
        print(f"  vf_normalize_modes: {vf_normalize_modes}")
        print(f"  n_epochs: {n_epochs}, batch_size: {batch_size}, T: {T}, lr: {lr}")
    else:
        # Default sweep parameters
        activations = ["Tanh", "ReLU", "Identity"]
        zero_diags = [False, True]
        preconditions = [False, True]
        loss_types = ["trajectory", "vector_field"]
        vf_normalize_modes = ["none", "cosine", "angular_mse"]
        num_neurons_list = [2]
        n_epochs = 2000
        batch_size = 128
        T = 60
        lr = 0.01
        lag = 10
        init_scale = 2.0

    # Generate all combinations
    # For trajectory loss, vf_normalize_mode doesn't matter, so we only use "none"
    # For vector_field loss, we sweep over vf_normalize_modes
    configs = []
    for N in num_neurons_list:
        for act in activations:
            for zd in zero_diags:
                for pc in preconditions:
                    for lt in loss_types:
                        if lt == "trajectory":
                            # Only one config for trajectory loss
                            configs.append(ExperimentConfig(
                                num_neurons=N,
                                activation=act,
                                zero_diag=zd,
                                precondition=pc,
                                loss_type=lt,
                                vf_normalize_mode="none",
                                n_epochs=n_epochs,
                                batch_size=batch_size,
                                T=T,
                                lr=lr,
                                lag=lag,
                                init_scale=init_scale,
                                teacher_W=teacher_W,
                                teacher_b=teacher_b,
                                teacher_alphas=teacher_alphas,
                                student_alphas=student_alphas,
                            ))
                        else:
                            # Sweep over vf_normalize_modes for vector_field loss
                            for vf_mode in vf_normalize_modes:
                                configs.append(ExperimentConfig(
                                    num_neurons=N,
                                    activation=act,
                                    zero_diag=zd,
                                    precondition=pc,
                                    loss_type=lt,
                                    vf_normalize_mode=vf_mode,
                                    n_epochs=n_epochs,
                                    batch_size=batch_size,
                                    T=T,
                                    lr=lr,
                                    lag=lag,
                                    init_scale=init_scale,
                                    teacher_W=teacher_W,
                                    teacher_b=teacher_b,
                                    teacher_alphas=teacher_alphas,
                                    student_alphas=student_alphas,
                                ))

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
        activation_cls = get_activation_class(config.activation)

        # Recreate networks
        teacher_alphas = torch.tensor(result.teacher_alphas)
        student_alphas = torch.tensor(result.student_alphas)

        teacher = BabyRNN(N, teacher_alphas, activation=activation_cls, zero_diag=config.zero_diag)
        teacher.W_rec.weight.data = torch.tensor(result.teacher_W)
        teacher.W_rec.bias.data = torch.tensor(result.teacher_b)

        student_init = BabyRNN(N, student_alphas, activation=activation_cls, zero_diag=config.zero_diag)
        student_init.W_rec.weight.data = torch.tensor(result.student_W_init)
        student_init.W_rec.bias.data = teacher.W_rec.bias.data.clone()  # Same bias initially

        student_trained = BabyRNN(N, student_alphas, activation=activation_cls, zero_diag=config.zero_diag)
        student_trained.W_rec.weight.data = torch.tensor(result.student_W_final)
        student_trained.W_rec.bias.data = torch.tensor(result.student_b_final)

        # Generate main comparison figure
        save_path = RESULTS_DIR / f"{config.name()}.png"
        create_comparison_figure(
            teacher, student_init, student_trained,
            result.loss_history, config,
            save_path=save_path
        )
        plt.close()

        # Generate effective timescales figure (Jacobian eigenvalues)
        ts_save_path = RESULTS_DIR / f"{config.name()}_timescales.png"
        try:
            plot_effective_timescales(teacher, student_init, student_trained, config, save_path=ts_save_path)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate timescales plot for {config.name()}: {e}")

    # Generate loss curves comparison figure
    if results:
        loss_curves_path = RESULTS_DIR / "loss_curves_comparison.png"
        try:
            create_loss_curves_figure(results, save_path=loss_curves_path)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate loss curves figure: {e}")


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

    # Run with custom teacher dynamics
    python teacher_student_identifiability.py --teacher-config configs/example_teacher.yaml

    # Run with custom sweep config
    python teacher_student_identifiability.py --sweep-config configs/sweep_config.yaml

    # Combine both
    python teacher_student_identifiability.py \\
        --sweep-config configs/sweep_config.yaml \\
        --teacher-config configs/example_teacher.yaml

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
    parser.add_argument(
        "--teacher-config", type=str, default=None,
        help="Path to YAML file with custom teacher parameters (teacher_W, teacher_b, teacher_alphas, student_alphas)"
    )
    parser.add_argument(
        "--sweep-config", type=str, default=None,
        help="Path to YAML file with sweep parameters (activations, loss_types, n_epochs, etc.)"
    )
    # Hidden arguments for subprocess execution
    parser.add_argument("--single", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Handle subprocess mode (--single)
    if args.single:
        run_single_from_config(args.single, args.output)
        return

    # Load sweep config if provided
    sweep_params = None
    if args.sweep_config:
        import yaml as yaml_loader
        with open(args.sweep_config, "r") as f:
            sweep_params = yaml_loader.safe_load(f)
        print(f"Loaded sweep config from: {args.sweep_config}")

    # Load custom teacher config if provided
    custom_teacher_params = None
    if args.teacher_config:
        import yaml as yaml_loader
        with open(args.teacher_config, "r") as f:
            custom_teacher_params = yaml_loader.safe_load(f)
        print(f"Loaded custom teacher config from: {args.teacher_config}")

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
            gpu_ids=gpu_ids,
            parallel=parallel,
            results_dir=results_dir,
            custom_teacher_params=custom_teacher_params,
            sweep_params=sweep_params,
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
