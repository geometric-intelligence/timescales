# %% [markdown]
# # Signed 2-Bit Asymmetric Flip-Flop Comparison
#
# Compare linear and nonlinear signed flip-flop runs.
#
# The compact figure focuses on:
# 1. Validation loss and validation accuracy over training.
# 2. Task timescales and tracked network effective timescales, when
#    `spectral_trajectory.pt` is available for a run.

# %%
import json
import itertools
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.optimize import minimize as sp_minimize
from sklearn.decomposition import PCA

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()
os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())


# %% Configuration
RUNS = {
    "Linear rate": "/home/facosta/timescales/timescales/logs/single_runs/rnn_20260516_012357",
    "Tanh rate": "/home/facosta/timescales/timescales/logs/single_runs/rnn_20260516_011340",
}

SAVE_FIGS = False
EIGEN_TRACK_ANCHOR = "final"  # "final" tracks learned modes backward; "initial" tracks initial modes forward
FIGS_DIR = Path("notebooks/flip_flop/figs/asym_2bit_demo")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

RUN_FIXED_POINT_ANALYSIS = True
FIXED_POINT_LABEL = "Tanh rate"
FIXED_POINT_N_TRAJ_INITS = 800
FIXED_POINT_N_RANDOM_INITS = 128
FIXED_POINT_MAX_ITERS = 5000
FIXED_POINT_FTOL = 1e-20
FIXED_POINT_GTOL = 1e-12
FIXED_POINT_CLUSTER_TOL = 1.0
FIXED_POINT_Q_THRESH = 1e-10
PLOT_BASIN_SHADING = True
BASIN_GRID_N = 120
BASIN_ROLLOUT_STEPS = 250

RUN_EDMD_ANALYSIS = True
EDMD_LABEL = "Tanh rate"
EDMD_N_TRAJ = 200
EDMD_NUM_TIME_STEPS = 500
EDMD_MAX_PAIRS = 50_000
EDMD_RANDOM_FEATURES = 150
EDMD_RANDOM_FEATURE_SCALE = 1.0
EDMD_RIDGE = 1e-5
EDMD_NOISE_STD = 0.01
EDMD_NOISE_SEED = 123
EDMD_DICTIONARIES = ["random_tanh", "carleman"]
CARLEMAN_MAX_ORDER = 2
CARLEMAN_INCLUDE_PAIRWISE_ORDER2 = True
CARLEMAN_MAX_PAIRWISE_TERMS = 300


def _torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def effective_timescale(eigvals: np.ndarray, dt: float) -> np.ndarray:
    """Convert discrete-time eigenvalues to effective timescales."""
    eig_abs = np.abs(eigvals)
    eig_abs_stable = np.where(eig_abs < 1.0, np.clip(eig_abs, 1e-12, None), np.nan)
    return -dt / np.log(eig_abs_stable)


def smooth_tau_branches_accel_dp(tau_net: np.ndarray) -> np.ndarray:
    """Choose tau-branch colors by minimizing global acceleration in log(tau).

    This intentionally tracks smooth branches in timescale space rather than
    eigenmode identity. It is appropriate for display when eigenvalue identities
    become ambiguous near crossings.
    """
    if tau_net.size == 0 or tau_net.shape[1] <= 1:
        return tau_net

    n_steps, n_tracks = tau_net.shape
    permutations = list(itertools.permutations(range(n_tracks)))
    n_perm = len(permutations)
    permuted = np.stack([tau_net[:, p] for p in permutations], axis=1)
    log_permuted = np.log(
        np.where((permuted > 0) & np.isfinite(permuted), permuted, np.nan)
    )

    def _vel_cost(curr, prev):
        diff = curr - prev
        if np.all(~np.isfinite(diff)):
            return np.inf
        return np.nansum(diff**2)

    def _accel_cost(curr, prev, prevprev):
        accel = curr - 2 * prev + prevprev
        if np.all(~np.isfinite(accel)):
            return np.inf
        return np.nansum(accel**2)

    dp = np.full((n_steps, n_perm, n_perm), np.inf)
    backptr = np.full((n_steps, n_perm, n_perm), -1, dtype=int)

    for p0 in range(n_perm):
        for p1 in range(n_perm):
            dp[1, p0, p1] = _vel_cost(log_permuted[1, p1], log_permuted[0, p0])

    for t in range(2, n_steps):
        for p_prev in range(n_perm):
            for p_curr in range(n_perm):
                costs = np.array(
                    [
                        dp[t - 1, p_prevprev, p_prev]
                        + _accel_cost(
                            log_permuted[t, p_curr],
                            log_permuted[t - 1, p_prev],
                            log_permuted[t - 2, p_prevprev],
                        )
                        for p_prevprev in range(n_perm)
                    ]
                )
                best_prevprev = int(np.nanargmin(costs))
                dp[t, p_prev, p_curr] = costs[best_prevprev]
                backptr[t, p_prev, p_curr] = best_prevprev

    end_prev, end_curr = np.unravel_index(np.nanargmin(dp[-1]), dp[-1].shape)
    chosen = np.empty(n_steps, dtype=int)
    chosen[-2] = end_prev
    chosen[-1] = end_curr

    for t in range(n_steps - 1, 1, -1):
        chosen[t - 2] = backptr[t, chosen[t - 1], chosen[t]]

    out = np.empty_like(tau_net)
    for t in range(n_steps):
        out[t] = permuted[t, chosen[t]]
    return out


def smooth_tau_branches_from_records(
    records: list[dict],
    dt: float,
    top_k: int,
) -> np.ndarray:
    """Build smooth top-k timescale branches from saved spectral records."""
    rank_eigs = np.stack([r["top_jacobian"] for r in records])[:, :top_k]
    rank_tau = effective_timescale(rank_eigs, dt)
    return smooth_tau_branches_accel_dp(rank_tau)


def load_run(label: str, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    config_path = sorted(run_dir.glob("config_seed*.yaml"))[0]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loss_path = run_dir / "training_losses.json"
    with open(loss_path) as f:
        losses = json.load(f)

    steps = np.asarray(losses.get("steps", []), dtype=float)
    if len(steps) == 0:
        steps = np.arange(1, len(losses.get("val_losses", [])) + 1, dtype=float)

    run = {
        "label": label,
        "run_dir": run_dir,
        "config": config,
        "steps": steps,
        "steps_for_log": np.maximum(steps, 1.0),
        "val_losses": np.asarray(losses.get("val_losses", []), dtype=float),
        "val_accs": np.asarray(losses.get("val_accuracies", []), dtype=float),
        "spectral_missing": False,
    }

    spectral_path = run_dir / "spectral_trajectory.pt"
    if spectral_path.exists():
        spectral = _torch_load(spectral_path, map_location="cpu")
        records = spectral["records"]
        spec_steps = np.asarray([r["step"] for r in records], dtype=float)
        dt = float(config["dt"])
        p_pulse = np.asarray(config["p_pulse"], dtype=float)
        task_tau = -dt / np.log1p(-p_pulse)
        top_k = int(spectral.get("top_k", len(task_tau)))
        eigvals_jacobian = [np.asarray(r["eigvals_jacobian"]) for r in records]
        run.update(
            {
                "spec_steps_for_log": np.maximum(spec_steps, 1.0),
                "tau_net": smooth_tau_branches_from_records(records, dt, top_k),
                "spectral_radius": np.asarray(
                    [np.max(np.abs(eigs)) for eigs in eigvals_jacobian],
                    dtype=float,
                ),
            }
        )
    else:
        run["spectral_missing"] = True

    p_pulse = np.asarray(config["p_pulse"], dtype=float)
    dt = float(config["dt"])
    run["task_tau"] = -dt / np.log1p(-p_pulse)
    return run


runs = [load_run(label, path) for label, path in RUNS.items()]
for run in runs:
    cfg = run["config"]
    print(
        f"{run['label']}: activation={cfg['activation']}, "
        f"dynamics={cfg['dynamics_type']}, "
        f"last_acc={run['val_accs'][-1] if len(run['val_accs']) else np.nan:.3f}, "
        f"spectral_missing={run['spectral_missing']}"
    )


# %% Comparison figure
fig, axes = plt.subplots(
    len(runs),
    3,
    figsize=(16, 3.8 * len(runs)),
    squeeze=False,
    sharex="col",
)

for row, run in enumerate(runs):
    row_color = f"C{row}"
    ax_loss = axes[row, 0]
    ax_acc = ax_loss.twinx()
    ax_tau = axes[row, 1]
    ax_stab = axes[row, 2]

    label = run["label"]
    steps = run["steps_for_log"]
    val_losses = run["val_losses"]
    val_accs = run["val_accs"]

    if len(val_losses):
        ax_loss.semilogx(
            steps[: len(val_losses)],
            val_losses,
            color=row_color,
            lw=1.8,
            label="val loss",
        )
    if len(val_accs):
        ax_acc.semilogx(
            steps[: len(val_accs)],
            val_accs,
            color="0.25",
            lw=1.4,
            ls="--",
            label="val accuracy",
        )

    ax_loss.set_title(label, loc="left", fontweight="bold")
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("validation loss")
    ax_loss.set_yscale("log")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_ylim(0, 1.02)

    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
    ax_loss.legend(
        handles_loss + handles_acc,
        labels_loss + labels_acc,
        loc="upper left",
        bbox_to_anchor=(1.12, 1.0),
        borderaxespad=0,
        fontsize=8,
    )

    for k, tau in enumerate(run["task_tau"]):
        ax_tau.axhline(
            tau,
            color=f"C{k}",
            ls="--",
            lw=1.4,
            alpha=0.75,
            label=rf"$\tau_{{task,{k}}}$",
        )

    if run["spectral_missing"]:
        ax_tau.text(
            0.03,
            0.03,
            "Missing spectral_trajectory.pt",
            transform=ax_tau.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
        )
    else:
        tau_net = run["tau_net"]
        for k in range(tau_net.shape[1]):
            ax_tau.semilogx(
                run["spec_steps_for_log"],
                tau_net[:, k],
                color=f"C{k}",
                lw=1.8,
                alpha=0.95,
                ls="-",
                label=rf"$\tau_{{net,{k}}}$",
            )

    ax_tau.set_title(f"{label}: effective timescales", loc="left", fontweight="bold")
    ax_tau.set_xlabel("training step")
    ax_tau.set_ylabel("effective timescale")
    ax_tau.set_yscale("log")
    ax_tau.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)

    if run["spectral_missing"]:
        ax_stab.text(
            0.03,
            0.03,
            "Missing spectral_trajectory.pt",
            transform=ax_stab.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
        )
    else:
        ax_stab.semilogx(
            run["spec_steps_for_log"],
            run["spectral_radius"],
            color=row_color,
            lw=1.8,
            label=r"$\rho(J_0)$",
        )
        unstable = run["spectral_radius"] > 1.0
        if unstable.any():
            ax_stab.fill_between(
                run["spec_steps_for_log"],
                1.0,
                run["spectral_radius"],
                where=unstable,
                color=row_color,
                alpha=0.15,
                interpolate=True,
            )
    ax_stab.set_yscale("log")
    ax_stab.axhline(1.0, color="k", ls="--", lw=1.0, label="stability boundary")
    ax_stab.set_title(f"{label}: origin stability", loc="left", fontweight="bold")
    ax_stab.set_xlabel("training step")
    ax_stab.set_ylabel(r"spectral radius $\rho(J_0)$")
    ax_stab.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)

fig.suptitle(
    "Signed 2-Bit Asymmetric Flip-Flop: Linear vs Nonlinear RNN",
    y=1.02,
)
plt.tight_layout()

if SAVE_FIGS:
    out_path = FIGS_DIR / "linear_vs_nonlinear_summary.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print("Saved:", out_path)

plt.show()


# %% Final-step fixed-point analysis
def final_model_path(run: dict) -> Path | None:
    paths = sorted(run["run_dir"].glob(f"final_model_seed{run['config']['seed']}.pth"))
    if paths:
        return paths[0]
    paths = sorted(run["run_dir"].glob("final_model_seed*.pth"))
    return paths[0] if paths else None


def load_final_model(run: dict):
    path = final_model_path(run)
    if path is None:
        return None, None

    from train import _create_rnn_model

    config = dict(run["config"])
    model, _ = _create_rnn_model(config)
    state = _torch_load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, path


def load_untrained_model(run: dict):
    from lightning.pytorch import seed_everything
    from train import _create_rnn_model

    config = dict(run["config"])
    untrained_ckpt = run["run_dir"] / "checkpoints" / "untrained.ckpt"
    model, lightning_module = _create_rnn_model(config)
    if untrained_ckpt.exists():
        ckpt = _torch_load(untrained_ckpt, map_location="cpu")
        lightning_module.load_state_dict(ckpt["state_dict"])
        source = untrained_ckpt
    else:
        seed_everything(config["seed"], workers=True)
        model, _ = _create_rnn_model(config)
        source = "reinitialized from config seed"
    model.eval()
    return model, source


def extract_rate_tanh_fixed_point_params(model) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return gW, bias, W_out for the zero-input rate-Tanh fixed-point equation."""
    if model.rnn_step.dynamics_type != "rate":
        raise ValueError("This fixed-point helper currently supports dynamics_type='rate' only.")
    if model.rnn_step.activation.__class__.__name__ != "Tanh":
        raise ValueError("This fixed-point helper currently supports activation='Tanh' only.")

    rnn_step = model.rnn_step
    g = float(rnn_step.recurrent_gain)
    W_rec = rnn_step.W_rec.weight.detach().cpu().numpy().astype(np.float64)
    b_rec = (
        rnn_step.W_rec.bias.detach().cpu().numpy().astype(np.float64)
        if rnn_step.W_rec.bias is not None
        else np.zeros(model.hidden_size, dtype=np.float64)
    )
    b_in = (
        rnn_step.W_in.bias.detach().cpu().numpy().astype(np.float64)
        if rnn_step.W_in.bias is not None
        else np.zeros(model.hidden_size, dtype=np.float64)
    )
    W_out = model.W_out.weight.detach().cpu().numpy().astype(np.float64)
    return g * W_rec, g * b_rec + b_in, W_out


def fixed_point_initial_conditions(model, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use hidden states from task trajectories plus random inits, as in notebook 16."""
    from datamodules.signed_flip_flop import simulate_signed_flip_flop_trajectories

    inputs_np, targets_np, _ = simulate_signed_flip_flop_trajectories(
        num_trajectories=100,
        num_time_steps=config["num_time_steps"],
        n_bits=config["n_bits"],
        p_pulse=config["p_pulse"],
        pulse_amplitude=config.get("pulse_amplitude", 1.0),
    )
    inputs = torch.from_numpy(inputs_np).float()

    with torch.no_grad():
        hidden_seq, _ = model(inputs)
    hidden_all = hidden_seq.detach().cpu().numpy().reshape(-1, model.hidden_size)
    targets_all = targets_np.reshape(-1, config["n_bits"])

    rng = np.random.RandomState(42)
    n_traj = min(FIXED_POINT_N_TRAJ_INITS, len(hidden_all))
    traj_inits = hidden_all[rng.choice(len(hidden_all), size=n_traj, replace=False)]
    h_std = max(float(hidden_all.std()), 1e-3)
    random_inits = rng.randn(FIXED_POINT_N_RANDOM_INITS, model.hidden_size) * h_std
    init_points = np.vstack([np.zeros((1, model.hidden_size)), traj_inits, random_inits])
    return init_points, hidden_all, targets_all


def find_fixed_point_candidates_lbfgs(
    model,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Find zero-input fixed points with analytic-gradient L-BFGS-B."""
    gW, bias, _ = extract_rate_tanh_fixed_point_params(model)
    gW_T = gW.T.copy()
    init_points, hidden_all, targets_all = fixed_point_initial_conditions(model, config)

    def _fp_speed(r):
        residual = r - np.tanh(gW @ r + bias)
        return 0.5 * np.dot(residual, residual)

    def _fp_grad(r):
        act = np.tanh(gW @ r + bias)
        residual = r - act
        sech2 = 1.0 - act**2
        return residual - gW_T @ (sech2 * residual)

    converged = []
    q_vals = []
    report_every = max(1, len(init_points) // 10)
    print(
        f"Running {len(init_points)} L-BFGS fixed-point inits "
        f"({FIXED_POINT_N_TRAJ_INITS} trajectory, "
        f"{FIXED_POINT_N_RANDOM_INITS} random, plus origin)..."
    )
    for idx, r0 in enumerate(init_points):
        res = sp_minimize(
            _fp_speed,
            r0.astype(np.float64),
            jac=_fp_grad,
            method="L-BFGS-B",
            options={
                "maxiter": FIXED_POINT_MAX_ITERS,
                "ftol": FIXED_POINT_FTOL,
                "gtol": FIXED_POINT_GTOL,
            },
        )
        if res.fun < FIXED_POINT_Q_THRESH:
            converged.append(res.x)
            q_vals.append(res.fun)
        if (idx + 1) % report_every == 0:
            print(f"  [{idx + 1}/{len(init_points)}] {len(converged)} converged")

    if not converged:
        return np.empty((0, model.hidden_size)), np.empty((0,)), hidden_all, targets_all

    return np.asarray(converged), np.sqrt(2 * np.asarray(q_vals)), hidden_all, targets_all


def cluster_fixed_points(
    candidates: np.ndarray,
    residual_norm: np.ndarray,
    residual_tol: float = np.sqrt(2 * FIXED_POINT_Q_THRESH),
    cluster_tol: float = FIXED_POINT_CLUSTER_TOL,
) -> list[dict]:
    good_idx = np.where(residual_norm < residual_tol)[0]
    order = good_idx[np.argsort(residual_norm[good_idx])]
    clusters: list[dict] = []

    for idx in order:
        point = candidates[idx]
        assigned = False
        for cluster in clusters:
            if np.linalg.norm(point - cluster["center"]) < cluster_tol:
                cluster["members"].append(idx)
                member_points = candidates[cluster["members"]]
                cluster["center"] = member_points.mean(axis=0)
                cluster["best_residual"] = min(
                    cluster["best_residual"],
                    float(residual_norm[idx]),
                )
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "center": point.copy(),
                    "members": [idx],
                    "best_residual": float(residual_norm[idx]),
                }
            )

    return clusters


def fixed_point_jacobian_eigendecomp(
    model,
    fixed_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gW, bias, _ = extract_rate_tanh_fixed_point_params(model)
    alpha = model.rnn_step.current_alphas.detach().cpu().numpy().astype(np.float64)
    pre = gW @ fixed_point + bias
    sech2 = 1.0 - np.tanh(pre) ** 2
    jac = np.diag(1.0 - alpha) + alpha[:, None] * (sech2[:, None] * gW)
    return np.linalg.eig(jac)


def output_mode_coupling(model, eigvecs: np.ndarray) -> np.ndarray:
    """Connectivity coupling from local eigenmodes to output channels."""
    W_out = model.W_out.weight.detach().cpu().numpy().astype(np.float64)
    return np.abs(W_out @ eigvecs)


def summarize_fixed_points(
    model,
    clusters: list[dict],
) -> list[dict]:
    rows = []
    with torch.no_grad():
        for i, cluster in enumerate(clusters):
            center = cluster["center"]
            h = torch.tensor(center, dtype=torch.float32).unsqueeze(0)
            output = model.W_out(h).squeeze(0).cpu().numpy()
            eigvals, eigvecs = fixed_point_jacobian_eigendecomp(model, center)
            coupling = output_mode_coupling(model, eigvecs)
            spectral_radius = float(np.max(np.abs(eigvals)))
            rows.append(
                {
                    "idx": i,
                    "n_members": len(cluster["members"]),
                    "hidden_norm": float(np.linalg.norm(center)),
                    "output": output,
                    "eigvals": eigvals,
                    "eigvecs": eigvecs,
                    "output_coupling": coupling,
                    "residual": cluster["best_residual"],
                    "spectral_radius": spectral_radius,
                    "stable": spectral_radius < 1.0,
                }
            )
    return rows


def basin_partition_output_plane(
    model,
    stable_centers: np.ndarray,
    xlim: tuple[float, float] = (-1.2, 1.2),
    ylim: tuple[float, float] = (-1.2, 1.2),
    grid_n: int = BASIN_GRID_N,
    rollout_steps: int = BASIN_ROLLOUT_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Approximate basins on the 2D readout plane by zero-input rollouts.

    The grid is embedded into hidden state by the minimum-norm pseudoinverse of
    W_out, then assigned to the nearest stable fixed point after rollout.
    """
    if stable_centers.size == 0:
        return None

    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    xx, yy = np.meshgrid(xs, ys)
    y_grid = np.column_stack([xx.ravel(), yy.ravel()])

    W_out = model.W_out.weight.detach().cpu().numpy().astype(np.float64)
    decoder = W_out.T @ np.linalg.pinv(W_out @ W_out.T)
    hidden0 = y_grid @ decoder.T

    hidden = torch.tensor(hidden0, dtype=torch.float32)
    zero_input = torch.zeros(hidden.shape[0], model.input_size, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(rollout_steps):
            hidden = model.rnn_step(zero_input, hidden)

    hidden_final = hidden.cpu().numpy()
    dists = np.linalg.norm(
        hidden_final[:, None, :] - stable_centers[None, :, :],
        axis=2,
    )
    basin_labels = np.argmin(dists, axis=1).reshape(grid_n, grid_n)
    return xx, yy, basin_labels


if RUN_FIXED_POINT_ANALYSIS:
    fp_run = next((run for run in runs if run["label"] == FIXED_POINT_LABEL), None)
    if fp_run is None:
        print(f"Fixed-point analysis skipped: no run labeled {FIXED_POINT_LABEL!r}.")
    else:
        fp_model, fp_model_path = load_final_model(fp_run)
        if fp_model is None:
            print(
                "Fixed-point analysis skipped: no final model found for "
                f"{FIXED_POINT_LABEL} at {fp_run['run_dir']}."
            )
        else:
            print(f"Fixed-point analysis for {FIXED_POINT_LABEL}: {fp_model_path}")
            candidates, residual_norm, hidden_all, targets_all = find_fixed_point_candidates_lbfgs(
                fp_model,
                fp_run["config"],
            )
            clusters = cluster_fixed_points(candidates, residual_norm)
            fp_rows = summarize_fixed_points(fp_model, clusters)

            print(f"Converged candidates: {len(candidates)}")
            print(f"Unique fixed points: {len(fp_rows)}")
            for row in fp_rows:
                output = np.array2string(row["output"], precision=3, suppress_small=True)
                print(
                    f"  #{row['idx']:02d} members={row['n_members']:3d} "
                    f"||h||={row['hidden_norm']:.3f} output={output} "
                    f"resid={row['residual']:.2e} "
                    f"rho={row['spectral_radius']:.3f} stable={row['stable']}"
                )

            if fp_rows:
                outputs = np.stack([row["output"] for row in fp_rows])
                stable = np.asarray([row["stable"] for row in fp_rows])
                n_members = np.asarray([row["n_members"] for row in fp_rows])
                sizes = 70 + 170 * np.sqrt(n_members / n_members.max())
                centers = np.stack([cluster["center"] for cluster in clusters])
                stable_centers = centers[stable]

                fig_fp, axes_fp = plt.subplots(1, 2, figsize=(11, 4.5))
                ax_out, ax_pca = axes_fp

                if PLOT_BASIN_SHADING:
                    basin = basin_partition_output_plane(fp_model, stable_centers)
                    if basin is not None:
                        _, _, basin_labels = basin
                        basin_colors = plt.cm.Set3(
                            np.linspace(0, 1, max(1, stable_centers.shape[0]))
                        )
                        basin_img = basin_colors[basin_labels]
                        basin_img[..., -1] = 0.28
                        ax_out.imshow(
                            basin_img,
                            origin="lower",
                            extent=(-1.2, 1.2, -1.2, 1.2),
                            interpolation="nearest",
                            aspect="auto",
                        )

                ax_out.scatter(
                    outputs[stable, 0],
                    outputs[stable, 1],
                    s=sizes[stable],
                    c="#1f77b4",
                    edgecolors="white",
                    linewidths=1.0,
                    marker="o",
                    label="stable",
                    alpha=0.95,
                )
                ax_out.scatter(
                    outputs[~stable, 0],
                    outputs[~stable, 1],
                    s=sizes[~stable],
                    c="#d62728",
                    edgecolors="white",
                    linewidths=1.0,
                    marker="o",
                    label="unstable",
                    alpha=0.95,
                )
                for idx, out in enumerate(outputs):
                    ax_out.text(out[0] + 0.035, out[1] + 0.035, str(idx), fontsize=8)
                ax_out.axhline(0, color="0.82", lw=0.8)
                ax_out.axvline(0, color="0.82", lw=0.8)
                ax_out.set_xlim(-1.2, 1.2)
                ax_out.set_ylim(-1.2, 1.2)
                ax_out.set_xlabel("output channel 0")
                ax_out.set_ylabel("output channel 1")
                ax_out.set_title("Fixed points in output space", fontweight="bold")
                ax_out.axis("equal")
                ax_out.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0,
                    fontsize=8,
                )

                pca = PCA(n_components=2)
                pca.fit(hidden_all)
                hidden_pca = pca.transform(hidden_all)
                fp_pca = pca.transform(centers)
                ax_pca.scatter(
                    hidden_pca[::5, 0],
                    hidden_pca[::5, 1],
                    s=1.0,
                    c="0.75",
                    alpha=0.08,
                    rasterized=True,
                    label="trajectory states",
                )
                ax_pca.scatter(
                    fp_pca[stable, 0],
                    fp_pca[stable, 1],
                    s=sizes[stable],
                    c="#1f77b4",
                    edgecolors="white",
                    linewidths=1.0,
                    marker="o",
                    label="stable",
                    alpha=0.95,
                )
                ax_pca.scatter(
                    fp_pca[~stable, 0],
                    fp_pca[~stable, 1],
                    s=sizes[~stable],
                    c="#d62728",
                    edgecolors="white",
                    linewidths=1.0,
                    marker="o",
                    label="unstable",
                    alpha=0.95,
                )
                for idx, xy in enumerate(fp_pca):
                    ax_pca.text(xy[0], xy[1], f" {idx}", fontsize=8)
                ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                ax_pca.set_title("Fixed points in hidden-state PCA", fontweight="bold")
                ax_pca.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0,
                    fontsize=8,
                )

                fig_fp.suptitle(
                    f"{FIXED_POINT_LABEL}: final-step fixed points "
                    f"({stable.sum()} stable, {(~stable).sum()} unstable)",
                    y=1.02,
                )
                plt.tight_layout()

                if SAVE_FIGS:
                    out_path = FIGS_DIR / "tanh_rate_final_fixed_points.pdf"
                    fig_fp.savefig(out_path, bbox_inches="tight", dpi=150)
                    print("Saved:", out_path)

                plt.show()

                stable_rows = [row for row in fp_rows if row["stable"]]
                if stable_rows:
                    dt = float(fp_run["config"]["dt"])
                    print("\nStable fixed-point local timescales:")
                    for row in stable_rows:
                        tau_eff = effective_timescale(row["eigvals"], dt)
                        tau_eff = tau_eff[np.isfinite(tau_eff)]
                        tau_eff = np.sort(tau_eff)[::-1]
                        top_tau = tau_eff[:8]
                        top_tau_str = ", ".join(f"{v:.3g}" for v in top_tau)
                        print(
                            f"  FP #{row['idx']:02d}: rho={row['spectral_radius']:.3f}, "
                            f"top tau_eff=[{top_tau_str}]"
                        )
                    print("\nStrongest output-coupled local modes:")
                    for row in stable_rows:
                        tau_eff_all = effective_timescale(row["eigvals"], dt)
                        valid = np.isfinite(tau_eff_all)
                        order = np.argsort(tau_eff_all[valid])[::-1]
                        valid_idx = np.where(valid)[0][order]
                        tau_sorted = tau_eff_all[valid_idx]
                        coupling_sorted = row["output_coupling"][:, valid_idx]
                        parts = []
                        for ch in range(coupling_sorted.shape[0]):
                            mode_rank = int(np.argmax(coupling_sorted[ch]))
                            parts.append(
                                f"ch{ch}: rank={mode_rank + 1}, "
                                f"tau={tau_sorted[mode_rank]:.3g}, "
                                f"coupling={coupling_sorted[ch, mode_rank]:.3g}"
                            )
                        print(f"  FP #{row['idx']:02d}: " + " | ".join(parts))

                    task_tau = -dt / np.log1p(
                        -np.asarray(fp_run["config"]["p_pulse"], dtype=float)
                    )

                    n_stable = len(stable_rows)
                    n_cols = min(2, n_stable)
                    n_rows = int(np.ceil(n_stable / n_cols))
                    fig_tau, axes_tau = plt.subplots(
                        n_rows,
                        n_cols,
                        figsize=(5.2 * n_cols, 3.4 * n_rows),
                        squeeze=False,
                        sharex=True,
                        sharey=True,
                    )
                    flat_axes_tau = axes_tau.ravel()
                    output_channel_handles = [
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="none",
                            markerfacecolor=f"C{k}",
                            markeredgecolor="white",
                            markersize=7,
                            label=f"max coupling to output ch{k}",
                        )
                        for k in range(fp_model.output_size)
                    ]

                    for ax, row in zip(flat_axes_tau, stable_rows):
                        for k, tau in enumerate(task_tau):
                            ax.axhline(
                                tau,
                                color=f"C{k}",
                                ls="--",
                                lw=1.2,
                                alpha=0.75,
                                label=rf"$\tau_{{task,{k}}}$" if row is stable_rows[0] else None,
                            )
                        tau_eff = effective_timescale(row["eigvals"], dt)
                        valid = np.isfinite(tau_eff)
                        order = np.argsort(tau_eff[valid])[::-1]
                        valid_idx = np.where(valid)[0][order]
                        tau_eff = tau_eff[valid_idx]
                        ax.plot(
                            np.arange(1, len(tau_eff) + 1),
                            tau_eff,
                            color="0.75",
                            lw=0.9,
                            zorder=1,
                        )
                        ax.scatter(
                            np.arange(1, len(tau_eff) + 1),
                            tau_eff,
                            s=18,
                            c="0.55",
                            edgecolors="none",
                            alpha=0.45,
                            zorder=2,
                        )
                        coupling = row["output_coupling"][:, valid_idx]
                        for ch in range(coupling.shape[0]):
                            mode_rank = int(np.argmax(coupling[ch]))
                            ax.scatter(
                                mode_rank + 1,
                                tau_eff[mode_rank],
                                s=95,
                                c=f"C{ch}",
                                edgecolors="white",
                                linewidths=0.9,
                                zorder=3,
                            )
                        output = np.array2string(row["output"], precision=2, suppress_small=True)
                        ax.set_title(
                            f"FP #{row['idx']}  output={output}\n"
                            f"rho={row['spectral_radius']:.3f}, members={row['n_members']}",
                            fontsize=10,
                        )
                        ax.set_yscale("log")
                        ax.set_xlabel("local mode rank")
                        ax.set_ylabel(r"local effective timescale")

                    for ax in flat_axes_tau[n_stable:]:
                        ax.axis("off")

                    handles, labels = flat_axes_tau[0].get_legend_handles_labels()
                    handles = handles + output_channel_handles
                    labels = labels + [h.get_label() for h in output_channel_handles]
                    if handles:
                        fig_tau.legend(
                            handles,
                            labels,
                            loc="upper left",
                            bbox_to_anchor=(1.01, 0.98),
                            borderaxespad=0,
                            fontsize=8,
                        )

                    fig_tau.suptitle(
                        f"{FIXED_POINT_LABEL}: local timescales by stable fixed point\n"
                        f"ch0 p={fp_run['config']['p_pulse'][0]} (slow), "
                        f"ch1 p={fp_run['config']['p_pulse'][1]} (fast)",
                        y=1.02,
                    )
                    plt.tight_layout()

                    if SAVE_FIGS:
                        out_path = FIGS_DIR / "tanh_rate_stable_fp_timescales_by_fp.pdf"
                        fig_tau.savefig(out_path, bbox_inches="tight", dpi=150)
                        print("Saved:", out_path)

                    plt.show()
else:
    print("Fixed-point analysis disabled. Set RUN_FIXED_POINT_ANALYSIS = True to run it.")


# %% EDMD / Koopman estimate for final Tanh model
def edmd_transition_pairs(model, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Sample hidden states from task trajectories and advance them at zero input."""
    from datamodules.signed_flip_flop import simulate_signed_flip_flop_trajectories

    inputs_np, _, _ = simulate_signed_flip_flop_trajectories(
        num_trajectories=EDMD_N_TRAJ,
        num_time_steps=EDMD_NUM_TIME_STEPS,
        n_bits=config["n_bits"],
        p_pulse=config["p_pulse"],
        pulse_amplitude=config.get("pulse_amplitude", 1.0),
    )
    inputs = torch.from_numpy(inputs_np).float()

    with torch.no_grad():
        hidden_seq, _ = model(inputs)
    X = hidden_seq.detach().cpu().numpy().reshape(-1, model.hidden_size)

    if len(X) > EDMD_MAX_PAIRS:
        rng = np.random.RandomState(123)
        keep = rng.choice(len(X), size=EDMD_MAX_PAIRS, replace=False)
        X = X[keep]

    with torch.no_grad():
        X_t = torch.from_numpy(X).float()
        zero_input = torch.zeros(X_t.shape[0], model.input_size)
        Y_t = model.rnn_step(zero_input, X_t)
    Y = Y_t.detach().cpu().numpy()
    if EDMD_NOISE_STD > 0:
        rng = np.random.RandomState(EDMD_NOISE_SEED)
        Y = Y + rng.randn(*Y.shape) * EDMD_NOISE_STD
    return X, Y


def make_random_tanh_features(X_ref: np.ndarray):
    """Dictionary: constant + state + random tanh features."""
    rng = np.random.RandomState(7)
    hidden_size = X_ref.shape[1]
    W = (
        rng.randn(hidden_size, EDMD_RANDOM_FEATURES)
        * EDMD_RANDOM_FEATURE_SCALE
        / np.sqrt(hidden_size)
    )
    b = rng.uniform(-1.0, 1.0, size=EDMD_RANDOM_FEATURES)

    def features(X: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.ones((X.shape[0], 1)),
                X,
                np.tanh(X @ W + b),
            ],
            axis=1,
        )

    return features


def make_carleman_features(X_ref: np.ndarray):
    """Truncated monomial/Carleman-style dictionary.

    Default is constant + linear + diagonal powers, with a capped set of
    pairwise quadratic terms when CARLEMAN_MAX_ORDER >= 2.
    """
    hidden_size = X_ref.shape[1]
    rng = np.random.RandomState(11)
    if CARLEMAN_INCLUDE_PAIRWISE_ORDER2 and CARLEMAN_MAX_ORDER >= 2:
        all_pairs = [(i, j) for i in range(hidden_size) for j in range(i + 1, hidden_size)]
        n_pairs = min(CARLEMAN_MAX_PAIRWISE_TERMS, len(all_pairs))
        pair_idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs = [all_pairs[i] for i in pair_idx]
    else:
        pairs = []

    def features(X: np.ndarray) -> np.ndarray:
        feats = [np.ones((X.shape[0], 1)), X]
        for order in range(2, CARLEMAN_MAX_ORDER + 1):
            feats.append(X**order)
        if pairs:
            feats.append(np.column_stack([X[:, i] * X[:, j] for i, j in pairs]))
        return np.concatenate(feats, axis=1)

    return features


def make_edmd_features(X_ref: np.ndarray, dictionary: str):
    if dictionary == "random_tanh":
        return make_random_tanh_features(X_ref)
    if dictionary == "carleman":
        return make_carleman_features(X_ref)
    raise ValueError(f"Unknown EDMD dictionary: {dictionary!r}")


def fit_edmd_koopman(
    X: np.ndarray,
    Y: np.ndarray,
    dictionary: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit Phi(Y) ≈ Phi(X) K with ridge regularization."""
    feature_fn = make_edmd_features(X, dictionary)
    Phi_X = feature_fn(X)
    Phi_Y = feature_fn(Y)

    G = Phi_X.T @ Phi_X / Phi_X.shape[0]
    A = Phi_X.T @ Phi_Y / Phi_X.shape[0]
    K = np.linalg.solve(G + EDMD_RIDGE * np.eye(G.shape[0]), A)
    eigvals = np.linalg.eigvals(K)
    Phi_Y_pred = Phi_X @ K
    hidden_size = X.shape[1]
    Y_pred = Phi_Y_pred[:, 1 : 1 + hidden_size]
    feature_norm = max(np.linalg.norm(Phi_Y), np.finfo(float).eps)
    state_norm = max(np.sqrt(np.mean(Y**2)), np.finfo(float).eps)
    state_rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
    metrics = {
        "feature_rel_fro": np.linalg.norm(Phi_Y - Phi_Y_pred) / feature_norm,
        "state_rmse": state_rmse,
        "state_rel_rmse": state_rmse / state_norm,
    }
    return K, eigvals, metrics


def compute_edmd_summary(model, config: dict, dictionary: str) -> dict:
    X_edmd, Y_edmd = edmd_transition_pairs(model, config)
    K_edmd, eigs, metrics = fit_edmd_koopman(X_edmd, Y_edmd, dictionary)
    dt = float(config["dt"])
    tau = effective_timescale(eigs, dt)
    tau_finite = np.sort(tau[np.isfinite(tau)])[::-1]
    return {
        "X": X_edmd,
        "K": K_edmd,
        "eigs": eigs,
        "tau": tau,
        "tau_finite": tau_finite,
        "metrics": metrics,
    }


if RUN_EDMD_ANALYSIS:
    edmd_run = next((run for run in runs if run["label"] == EDMD_LABEL), None)
    if edmd_run is None:
        print(f"EDMD skipped: no run labeled {EDMD_LABEL!r}.")
    else:
        edmd_model, edmd_model_path = load_final_model(edmd_run)
        if edmd_model is None:
            print(
                "EDMD skipped: no final model found for "
                f"{EDMD_LABEL} at {edmd_run['run_dir']}."
            )
        else:
            untrained_model, untrained_source = load_untrained_model(edmd_run)
            edmd_cases = {
                "untrained": (untrained_model, untrained_source),
                "trained": (edmd_model, edmd_model_path),
            }
            dt_edmd = float(edmd_run["config"]["dt"])
            edmd_results = {}
            for dictionary in EDMD_DICTIONARIES:
                edmd_results[dictionary] = {}
                for case_name, (case_model, source) in edmd_cases.items():
                    print(
                        f"EDMD / Koopman estimate for {EDMD_LABEL} "
                        f"({case_name}, {dictionary}): {source}"
                    )
                    summary = compute_edmd_summary(
                        case_model, edmd_run["config"], dictionary
                    )
                    edmd_results[dictionary][case_name] = summary
                    print(
                        f"  EDMD pairs={len(summary['X'])}, "
                        f"feature_dim={summary['K'].shape[0]}, "
                        f"ridge={EDMD_RIDGE:g}, noise_std={EDMD_NOISE_STD:g}"
                    )
                    metrics = summary["metrics"]
                    print(
                        "  Fit residuals: "
                        f"feature_rel_fro={metrics['feature_rel_fro']:.3g}, "
                        f"state_rmse={metrics['state_rmse']:.3g}, "
                        f"state_rel_rmse={metrics['state_rel_rmse']:.3g}"
                    )
                    print(
                        f"  Top {case_name} {dictionary} Koopman tau_eff: "
                        + ", ".join(f"{v:.3g}" for v in summary["tau_finite"][:12])
                    )

            n_rows = len(EDMD_DICTIONARIES) * len(edmd_cases)
            fig_edmd, axes_edmd = plt.subplots(
                n_rows,
                3,
                figsize=(14.0, 3.8 * n_rows),
                squeeze=False,
            )

            task_tau = -dt_edmd / np.log1p(
                -np.asarray(edmd_run["config"]["p_pulse"], dtype=float)
            )
            theta = np.linspace(0, 2 * np.pi, 256)
            all_positive_tau = [
                tau
                for tau in task_tau
                if np.isfinite(tau) and tau > 0
            ]
            for dictionary_results in edmd_results.values():
                for summary in dictionary_results.values():
                    all_positive_tau.extend(
                        tau
                        for tau in summary["tau_finite"]
                        if np.isfinite(tau) and tau > 0
                    )
            if all_positive_tau:
                tau_min = min(all_positive_tau)
                tau_max = max(all_positive_tau)
                shared_tau_ylim = (tau_min / 1.4, tau_max * 1.4)
            else:
                shared_tau_ylim = None

            row_idx = 0
            for dictionary in EDMD_DICTIONARIES:
                for case_name, summary in edmd_results[dictionary].items():
                    row_label = f"{dictionary}, {case_name}"
                    ax_complex, ax_tau, ax_resid = axes_edmd[row_idx]
                    eigs = summary["eigs"]
                    tau_finite = summary["tau_finite"]
                    metrics = summary["metrics"]

                    ax_complex.plot(
                        np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.5
                    )
                    ax_complex.scatter(
                        eigs.real,
                        eigs.imag,
                        s=14,
                        c=np.abs(eigs),
                        cmap="viridis",
                        alpha=0.75,
                        edgecolors="none",
                    )
                    ax_complex.axhline(0, color="0.85", lw=0.8)
                    ax_complex.axvline(0, color="0.85", lw=0.8)
                    ax_complex.set_xlabel("Re")
                    ax_complex.set_ylabel("Im")
                    ax_complex.set_title(
                        f"{row_label}: EDMD Koopman spectrum", fontweight="bold"
                    )
                    ax_complex.axis("equal")

                    for k, tau in enumerate(task_tau):
                        ax_tau.axhline(
                            tau,
                            color=f"C{k}",
                            ls="--",
                            lw=1.2,
                            alpha=0.75,
                            label=rf"$\tau_{{task,{k}}}$" if row_idx == 0 else None,
                        )

                    ax_tau.plot(
                        np.arange(1, len(tau_finite) + 1),
                        tau_finite,
                        marker="o",
                        ms=3.0,
                        lw=1.0,
                        color="0.25",
                        alpha=0.85,
                    )
                    ax_tau.set_yscale("log")
                    ax_tau.set_xlabel("Koopman mode rank")
                    ax_tau.set_ylabel(r"effective timescale")
                    ax_tau.set_title(
                        f"{row_label}: EDMD Koopman timescales\n"
                        f"rel. feature residual={metrics['feature_rel_fro']:.2g}, "
                        f"rel. state RMSE={metrics['state_rel_rmse']:.2g}",
                        fontweight="bold",
                    )
                    if shared_tau_ylim is not None:
                        ax_tau.set_ylim(*shared_tau_ylim)
                    ax_tau.legend(
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0,
                        fontsize=8,
                    )

                    residual_labels = ["feature\nrel.", "state\nrel. RMSE"]
                    residual_values = [
                        metrics["feature_rel_fro"],
                        metrics["state_rel_rmse"],
                    ]
                    bars = ax_resid.bar(
                        residual_labels,
                        residual_values,
                        color=["0.45", "C3"],
                        alpha=0.85,
                    )
                    ax_resid.set_ylim(0, max(1.0, 1.15 * max(residual_values)))
                    ax_resid.set_ylabel("relative residual")
                    ax_resid.set_title(
                        f"{row_label}: EDMD fit error\n"
                        f"state RMSE={metrics['state_rmse']:.3g}",
                        fontweight="bold",
                    )
                    ax_resid.axhline(1.0, color="0.75", ls="--", lw=0.8)
                    ax_resid.bar_label(bars, fmt="%.2g", padding=3, fontsize=8)
                    row_idx += 1

            fig_edmd.suptitle(
                f"{EDMD_LABEL}: before/after zero-input noisy EDMD Koopman estimates "
                f"(noise std={EDMD_NOISE_STD:g})",
                y=1.02,
            )
            plt.tight_layout()

            if SAVE_FIGS:
                out_path = FIGS_DIR / "tanh_rate_edmd_koopman.pdf"
                fig_edmd.savefig(out_path, bbox_inches="tight", dpi=150)
                print("Saved:", out_path)

            plt.show()
else:
    print("EDMD analysis disabled. Set RUN_EDMD_ANALYSIS = True to run it.")

# %%
