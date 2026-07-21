# %% Signed 2-bit asymmetric flip-flop dynamics grid summary
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA


# %%
SWEEP_DIR = Path(
    "/home/facosta/timescales/timescales/logs/experiments/"
    "flip_flop_asym_2bit_dynamics_grid_20260601_214403"
)
REPO_ROOT = Path("/home/facosta/timescales")
PACKAGE_DIR = REPO_ROOT / "timescales"
FIGS_DIR = Path("/home/facosta/timescales/timescales/notebooks/figs/flip_flop")
SAVE_FIGS = True
SAVE_ANIMATIONS = True
ANIMATION_DIR = FIGS_DIR / "animations"
ANIMATION_FPS = 20
ANIMATION_DURATION_SECONDS = 10
ANIMATION_DPI = 100
ANIMATION_BITRATE = 1800

# Set any of these to a subset for smaller diagnostic figures, e.g.
# PLOT_VARIANTS = ["tanh_rate", "tanh_voltage"]
PLOT_VARIANTS: list[str] | None = None
PLOT_TAUS: list[float] | None = None
PLOT_GAINS: list[float] | None = None

FOCUS_VARIANT = "tanh_rate"
FOCUS_TAU = 0.02
FOCUS_GAIN = 0.1
ADDITIONAL_FOCUS_RUNS = [
    {"variant": "tanh_voltage", "tau": 0.01, "gain": 0.2},
]
PCA_FOCUS_VARIANT = "tanh_voltage"
PCA_FOCUS_TAU = 0.01
PCA_FOCUS_GAIN = 0.2
PCA_N_TRAJ = 80
PCA_NUM_TIME_STEPS = 500
PCA_LINE_STRIDE = 2
PCA_SCATTER_STRIDE = 10

for path in (REPO_ROOT, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

VARIANT_ORDER = ["linear", "tanh_rate", "tanh_voltage", "relu_rate", "relu_voltage"]
TAU_ORDER = [0.005, 0.01, 0.02]
GAIN_ORDER = [0.05, 0.1, 0.2]
TOP_K_FALLBACK = 2


# %%
def _torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def effective_timescale(eigvals: np.ndarray, dt: float) -> np.ndarray:
    """Convert discrete-time eigenvalues to stable effective timescales."""
    eig_abs = np.abs(eigvals)
    eig_abs_stable = np.where(eig_abs < 1.0, np.clip(eig_abs, 1e-12, None), np.nan)
    return -dt / np.log(eig_abs_stable)


def smooth_tau_branches_accel_dp(tau_net: np.ndarray) -> np.ndarray:
    """Choose smooth display branches in log-timescale space."""
    if tau_net.size == 0 or tau_net.shape[0] <= 1 or tau_net.shape[1] <= 1:
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


def parse_experiment_name(name: str) -> dict:
    match = re.match(r"^(?P<variant>.+)_tau(?P<tau>[\d.]+)_g(?P<gain>[\d.]+)$", name)
    if match is None:
        return {
            "variant": name,
            "tau": np.nan,
            "gain": np.nan,
        }
    return {
        "variant": match.group("variant"),
        "tau": float(match.group("tau")),
        "gain": float(match.group("gain")),
    }


def variant_label(variant: str) -> str:
    return {
        "linear": "Linear",
        "tanh_rate": "Tanh rate",
        "tanh_voltage": "Tanh voltage",
        "relu_rate": "ReLU rate",
        "relu_voltage": "ReLU voltage",
    }.get(variant, variant)


def run_sort_key(run: dict) -> tuple:
    variant = run["variant"]
    tau = run["tau"]
    gain = run["gain"]
    variant_idx = VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else 999
    tau_idx = TAU_ORDER.index(tau) if tau in TAU_ORDER else 999
    gain_idx = GAIN_ORDER.index(gain) if gain in GAIN_ORDER else 999
    return variant_idx, tau_idx, gain_idx, run["label"]


def discover_run_dirs(sweep_dir: Path) -> list[Path]:
    return sorted(path.parent for path in sweep_dir.glob("*/seed_*/training_losses.json"))


def load_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    experiment_name = run_dir.parent.name
    parsed = parse_experiment_name(experiment_name)

    config_paths = sorted(run_dir.glob("config_seed*.yaml"))
    if config_paths:
        config_path = config_paths[0]
    else:
        config_path = run_dir / "run_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loss_path = run_dir / "training_losses.json"
    with open(loss_path) as f:
        losses = json.load(f)

    steps = np.asarray(losses.get("steps", []), dtype=float)
    if len(steps) == 0:
        steps = np.arange(1, len(losses.get("val_losses", [])) + 1, dtype=float)

    label = (
        f"{variant_label(parsed['variant'])} "
        rf"$\tau$={parsed['tau']:g}, g={parsed['gain']:g}"
    )
    run = {
        "label": label,
        "experiment_name": experiment_name,
        "run_dir": run_dir,
        "config": config,
        "variant": parsed["variant"],
        "tau": parsed["tau"],
        "gain": parsed["gain"],
        "steps": steps,
        "steps_for_log": np.maximum(steps, 1.0),
        "val_losses": np.asarray(losses.get("val_losses", []), dtype=float),
        "val_accs": np.asarray(losses.get("val_accuracies", []), dtype=float),
        "spectral_missing": False,
    }

    p_pulse = np.asarray(config["p_pulse"], dtype=float)
    dt = float(config["dt"])
    run["task_tau"] = -dt / np.log1p(-p_pulse)

    spectral_path = run_dir / "spectral_trajectory.pt"
    if spectral_path.exists():
        spectral = _torch_load(spectral_path, map_location="cpu")
        records = spectral.get("records", [])
        if records:
            spec_steps = np.asarray([r["step"] for r in records], dtype=float)
            top_k = int(spectral.get("top_k", len(run["task_tau"]) or TOP_K_FALLBACK))
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
    else:
        run["spectral_missing"] = True

    return run


def filter_runs(runs: list[dict]) -> list[dict]:
    out = runs
    if PLOT_VARIANTS is not None:
        out = [run for run in out if run["variant"] in PLOT_VARIANTS]
    if PLOT_TAUS is not None:
        out = [run for run in out if run["tau"] in PLOT_TAUS]
    if PLOT_GAINS is not None:
        out = [run for run in out if run["gain"] in PLOT_GAINS]
    return out


# %%
all_run_dirs = discover_run_dirs(SWEEP_DIR)
runs = sorted((load_run(path) for path in all_run_dirs), key=run_sort_key)
runs = filter_runs(runs)

print(f"Loaded {len(runs)} runs from {SWEEP_DIR}")
for run in runs:
    cfg = run["config"]
    last_loss = run["val_losses"][-1] if len(run["val_losses"]) else np.nan
    last_acc = run["val_accs"][-1] if len(run["val_accs"]) else np.nan
    print(
        f"{run['experiment_name']}: activation={cfg['activation']}, "
        f"dynamics={cfg['dynamics_type']}, "
        f"tau={run['tau']:g}, g={run['gain']:g}, "
        f"last_loss={last_loss:.4g}, last_acc={last_acc:.3f}, "
        f"spectral_missing={run['spectral_missing']}"
    )


# # %% Main summary figure
# fig, axes = plt.subplots(
#     len(runs),
#     3,
#     figsize=(16, 3.8 * len(runs)),
#     squeeze=False,
#     sharex="col",
# )

# for row, run in enumerate(runs):
#     row_color = f"C{row % 10}"
#     ax_loss = axes[row, 0]
#     ax_acc = ax_loss.twinx()
#     ax_tau = axes[row, 1]
#     ax_stab = axes[row, 2]

#     label = run["label"]
#     steps = run["steps_for_log"]
#     val_losses = run["val_losses"]
#     val_accs = run["val_accs"]

#     if len(val_losses):
#         ax_loss.semilogx(
#             steps[: len(val_losses)],
#             val_losses,
#             color=row_color,
#             lw=1.8,
#             label="val loss",
#         )
#     if len(val_accs):
#         ax_acc.semilogx(
#             steps[: len(val_accs)],
#             val_accs,
#             color="0.25",
#             lw=1.4,
#             ls="--",
#             label="val accuracy",
#         )

#     ax_loss.set_title(label, loc="left", fontweight="bold")
#     ax_loss.set_xlabel("training step")
#     ax_loss.set_ylabel("validation loss")
#     ax_loss.set_yscale("log")
#     ax_acc.set_ylabel("validation accuracy")
#     ax_acc.set_ylim(0, 1.02)

#     handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
#     handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
#     ax_loss.legend(
#         handles_loss + handles_acc,
#         labels_loss + labels_acc,
#         loc="upper left",
#         bbox_to_anchor=(1.12, 1.0),
#         borderaxespad=0,
#         fontsize=8,
#     )

#     for k, tau in enumerate(run["task_tau"]):
#         ax_tau.axhline(
#             tau,
#             color=f"C{k}",
#             ls="--",
#             lw=1.4,
#             alpha=0.75,
#             label=rf"$\tau_{{task,{k}}}$",
#         )

#     if run["spectral_missing"]:
#         ax_tau.text(
#             0.03,
#             0.03,
#             "Missing spectral_trajectory.pt",
#             transform=ax_tau.transAxes,
#             fontsize=8,
#             va="bottom",
#             ha="left",
#         )
#     else:
#         tau_net = run["tau_net"]
#         for k in range(tau_net.shape[1]):
#             ax_tau.semilogx(
#                 run["spec_steps_for_log"],
#                 tau_net[:, k],
#                 color=f"C{k}",
#                 lw=1.8,
#                 alpha=0.95,
#                 ls="-",
#                 label=rf"$\tau_{{net,{k}}}$",
#             )

#     ax_tau.set_title(f"{label}: effective timescales", loc="left", fontweight="bold")
#     ax_tau.set_xlabel("training step")
#     ax_tau.set_ylabel("effective timescale")
#     ax_tau.set_yscale("log")
#     ax_tau.legend(
#         loc="upper left",
#         bbox_to_anchor=(1.02, 1.0),
#         borderaxespad=0,
#         fontsize=8,
#     )

#     if run["spectral_missing"]:
#         ax_stab.text(
#             0.03,
#             0.03,
#             "Missing spectral_trajectory.pt",
#             transform=ax_stab.transAxes,
#             fontsize=8,
#             va="bottom",
#             ha="left",
#         )
#     else:
#         ax_stab.semilogx(
#             run["spec_steps_for_log"],
#             run["spectral_radius"],
#             color=row_color,
#             lw=1.8,
#             label=r"$\rho(J_0)$",
#         )
#         unstable = run["spectral_radius"] > 1.0
#         if unstable.any():
#             ax_stab.fill_between(
#                 run["spec_steps_for_log"],
#                 1.0,
#                 run["spectral_radius"],
#                 where=unstable,
#                 color=row_color,
#                 alpha=0.15,
#                 interpolate=True,
#             )
#     ax_stab.set_yscale("log")
#     ax_stab.axhline(1.0, color="k", ls="--", lw=1.0, label="stability boundary")
#     ax_stab.set_title(f"{label}: origin stability", loc="left", fontweight="bold")
#     ax_stab.set_xlabel("training step")
#     ax_stab.set_ylabel(r"spectral radius $\rho(J_0)$")
#     ax_stab.legend(
#         loc="upper left",
#         bbox_to_anchor=(1.02, 1.0),
#         borderaxespad=0,
#         fontsize=8,
#     )

# fig.suptitle(
#     "Signed 2-Bit Asymmetric Flip-Flop: Linear vs Nonlinear RNN Grid",
#     y=1.002,
# )
# plt.tight_layout()

# if SAVE_FIGS:
#     FIGS_DIR.mkdir(parents=True, exist_ok=True)
#     out_path = FIGS_DIR / "linear_vs_nonlinear_dynamics_grid_summary.pdf"
#     fig.savefig(out_path, bbox_inches="tight", dpi=150)
#     print("Saved:", out_path)

# plt.show()


# %% Focused loss/timescale views
def find_focus_run(variant: str, tau: float, gain: float) -> dict | None:
    return next(
        (
            run
            for run in runs
            if run["variant"] == variant
            and np.isclose(run["tau"], tau)
            and np.isclose(run["gain"], gain)
        ),
        None,
    )


def plot_loss_tau_focus(run: dict, *, save_fig: bool = SAVE_FIGS):
    fig_focus, axes_focus = plt.subplots(
        2,
        1,
        figsize=(9.5, 6.8),
        sharex=True,
        height_ratios=[1.0, 1.25],
    )
    ax_loss = axes_focus[0]
    ax_acc = ax_loss.twinx()
    ax_tau = axes_focus[1]

    steps = run["steps_for_log"]
    val_losses = run["val_losses"]
    val_accs = run["val_accs"]

    if len(val_losses):
        ax_loss.semilogx(
            steps[: len(val_losses)],
            val_losses,
            color="C0",
            lw=2.0,
            label="val loss",
        )
    if len(val_accs):
        ax_acc.semilogx(
            steps[: len(val_accs)],
            val_accs,
            color="0.25",
            lw=1.6,
            ls="--",
            label="val accuracy",
        )

    ax_loss.set_ylabel("validation loss")
    ax_loss.set_yscale("log")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_ylim(0, 1.02)
    ax_loss.grid(True, which="both", axis="x", alpha=0.18)
    ax_loss.grid(True, which="major", axis="y", alpha=0.18)

    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
    ax_loss.legend(
        handles_loss + handles_acc,
        labels_loss + labels_acc,
        loc="best",
        fontsize=9,
    )

    for k, tau in enumerate(run["task_tau"]):
        ax_tau.axhline(
            tau,
            color=f"C{k}",
            ls="--",
            lw=1.5,
            alpha=0.8,
            label=rf"$\tau_{{task,{k}}}$",
        )

    if run["spectral_missing"]:
        ax_tau.text(
            0.5,
            0.5,
            "Missing spectral_trajectory.pt\n"
            "Cannot plot network effective timescales over training.",
            transform=ax_tau.transAxes,
            fontsize=10,
            va="center",
            ha="center",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75"},
        )
    else:
        tau_net = run["tau_net"]
        for k in range(tau_net.shape[1]):
            ax_tau.semilogx(
                run["spec_steps_for_log"],
                tau_net[:, k],
                color=f"C{k}",
                lw=2.0,
                alpha=0.95,
                ls="-",
                label=rf"$\tau_{{net,{k}}}$",
            )

    ax_tau.set_xlabel("training step")
    ax_tau.set_ylabel("effective timescale")
    ax_tau.set_yscale("log")
    ax_tau.grid(True, which="both", axis="x", alpha=0.18)
    ax_tau.grid(True, which="major", axis="y", alpha=0.18)
    ax_tau.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=9,
    )

    title = (
        f"{variant_label(run['variant'])} RNN Focus: "
        rf"$\tau$={run['tau']:g}, g={run['gain']:g}"
    )
    fig_focus.suptitle(title, y=1.01, fontweight="bold")
    plt.tight_layout()

    if save_fig:
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = (
            FIGS_DIR
            / f"{run['variant']}_tau{run['tau']:g}_g{run['gain']:g}_loss_tau_focus.pdf"
        )
        fig_focus.savefig(out_path, bbox_inches="tight", dpi=150)
        print("Saved:", out_path)

    plt.show()
    return fig_focus


focus_specs = [
    {"variant": FOCUS_VARIANT, "tau": FOCUS_TAU, "gain": FOCUS_GAIN},
    *ADDITIONAL_FOCUS_RUNS,
]
focus_run = None
focus_runs = []
for focus_spec in focus_specs:
    run = find_focus_run(
        focus_spec["variant"],
        focus_spec["tau"],
        focus_spec["gain"],
    )
    if run is None:
        print(
            "Focused run not found: "
            f"variant={focus_spec['variant']}, "
            f"tau={focus_spec['tau']:g}, "
            f"g={focus_spec['gain']:g}"
        )
        continue
    if (
        focus_spec["variant"] == FOCUS_VARIANT
        and np.isclose(focus_spec["tau"], FOCUS_TAU)
        and np.isclose(focus_spec["gain"], FOCUS_GAIN)
    ):
        focus_run = run
    focus_runs.append(run)
    plot_loss_tau_focus(run)


# %% Hidden-state PCA trajectories for a focused final model
def final_model_path(run: dict) -> Path | None:
    paths = sorted(run["run_dir"].glob(f"final_model_seed{run['config']['seed']}.pth"))
    if paths:
        return paths[0]
    paths = sorted(run["run_dir"].glob("final_model_seed*.pth"))
    if paths:
        return paths[0]
    ckpt_paths = sorted((run["run_dir"] / "checkpoints").glob("best-model-*.ckpt"))
    return ckpt_paths[0] if ckpt_paths else None


def load_final_model(run: dict):
    path = final_model_path(run)
    if path is None:
        return None, None

    from train import _create_rnn_model

    config = dict(run["config"])
    model, lightning_module = _create_rnn_model(config)
    state = _torch_load(path, map_location="cpu")
    if path.suffix == ".ckpt":
        lightning_module.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, path


def signed_state_label(state: np.ndarray) -> str:
    return f"({int(state[0]):+d}, {int(state[1]):+d})"


def generate_hidden_trajectories(model, config: dict):
    from datamodules.signed_flip_flop import simulate_signed_flip_flop_trajectories

    inputs_np, targets_np, _ = simulate_signed_flip_flop_trajectories(
        num_trajectories=PCA_N_TRAJ,
        num_time_steps=PCA_NUM_TIME_STEPS,
        n_bits=config["n_bits"],
        p_pulse=config["p_pulse"],
        pulse_amplitude=config.get("pulse_amplitude", 1.0),
    )
    inputs = torch.from_numpy(inputs_np).float()
    with torch.no_grad():
        hidden_seq, outputs = model(inputs)
    return (
        hidden_seq.detach().cpu().numpy(),
        outputs.detach().cpu().numpy(),
        targets_np,
    )


def plot_hidden_pca_trajectories(run: dict) -> None:
    model, model_path = load_final_model(run)
    if model is None:
        print(
            "Hidden PCA skipped: no final model found for "
            f"{run['experiment_name']} at {run['run_dir']}."
        )
        print("Expected final_model_seed*.pth or checkpoints/best-model-*.ckpt.")
        return

    hidden_seq, outputs, targets = generate_hidden_trajectories(model, run["config"])
    n_traj, n_steps, hidden_size = hidden_seq.shape
    hidden_flat = hidden_seq.reshape(n_traj * n_steps, hidden_size)

    pca = PCA(n_components=2)
    hidden_pca = pca.fit_transform(hidden_flat).reshape(n_traj, n_steps, 2)
    final_states = targets[:, -1, :]

    state_values = np.array(
        [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1],
        ],
        dtype=np.float32,
    )
    state_colors = {
        tuple(state): f"C{i}" for i, state in enumerate(state_values.astype(int))
    }

    fig_pca, ax = plt.subplots(figsize=(7.0, 6.2))
    for traj_idx in range(n_traj):
        state = tuple(final_states[traj_idx].astype(int))
        color = state_colors.get(state, "0.4")
        xy = hidden_pca[traj_idx]
        ax.plot(
            xy[::PCA_LINE_STRIDE, 0],
            xy[::PCA_LINE_STRIDE, 1],
            color=color,
            lw=0.8,
            alpha=0.22,
        )
        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            s=8,
            color=color,
            alpha=0.35,
            marker="o",
            edgecolors="none",
        )
        ax.scatter(
            xy[-1, 0],
            xy[-1, 1],
            s=18,
            color=color,
            alpha=0.85,
            marker="x",
        )

    hidden_sample = hidden_pca.reshape(-1, 2)[::PCA_SCATTER_STRIDE]
    ax.scatter(
        hidden_sample[:, 0],
        hidden_sample[:, 1],
        s=1.0,
        c="0.2",
        alpha=0.025,
        rasterized=True,
        label="hidden states",
    )

    for state in state_values.astype(int):
        ax.plot(
            [],
            [],
            color=state_colors[tuple(state)],
            lw=2.0,
            label=f"final target {signed_state_label(state)}",
        )
    ax.scatter([], [], s=12, color="k", marker="o", label="trajectory start")
    ax.scatter([], [], s=24, color="k", marker="x", label="trajectory end")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(
        f"{variant_label(run['variant'])} hidden-state PCA trajectories\n"
        rf"$\tau$={run['tau']:g}, g={run['gain']:g}; model={model_path.name}",
        fontweight="bold",
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=8,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.18)
    plt.tight_layout()

    if SAVE_FIGS:
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = (
            FIGS_DIR
            / f"{run['variant']}_tau{run['tau']:g}_g{run['gain']:g}_hidden_pca.pdf"
        )
        fig_pca.savefig(out_path, bbox_inches="tight", dpi=150)
        print("Saved:", out_path)

    print(
        f"PCA trajectories: hidden_shape={hidden_seq.shape}, "
        f"output_shape={outputs.shape}, "
        f"explained_var={pca.explained_variance_ratio_[0]:.2%}, "
        f"{pca.explained_variance_ratio_[1]:.2%}"
    )
    plt.show()


pca_run = find_focus_run(PCA_FOCUS_VARIANT, PCA_FOCUS_TAU, PCA_FOCUS_GAIN)
if pca_run is None:
    print(
        "Hidden PCA run not found: "
        f"variant={PCA_FOCUS_VARIANT}, tau={PCA_FOCUS_TAU:g}, g={PCA_FOCUS_GAIN:g}"
    )
else:
    plot_hidden_pca_trajectories(pca_run)


# %% Focused animations
def positive_log_ylim(values: list[np.ndarray], pad_decades: float = 0.08) -> tuple[float, float]:
    finite_values = np.concatenate(
        [
            np.asarray(value, dtype=float).ravel()
            for value in values
            if len(np.asarray(value).ravel())
        ]
    )
    finite_values = finite_values[np.isfinite(finite_values) & (finite_values > 0)]
    if len(finite_values) == 0:
        return 1e-3, 1.0
    lo = np.nanmin(finite_values)
    hi = np.nanmax(finite_values)
    log_lo = np.log10(lo)
    log_hi = np.log10(hi)
    span = max(log_hi - log_lo, 0.5)
    return 10 ** (log_lo - pad_decades * span), 10 ** (log_hi + pad_decades * span)


def save_focus_animation(anim, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gif":
        writer = animation.PillowWriter(fps=ANIMATION_FPS)
    else:
        writer = animation.FFMpegWriter(fps=ANIMATION_FPS, bitrate=ANIMATION_BITRATE)
    anim.save(path, writer=writer, dpi=ANIMATION_DPI)
    print("Saved:", path)


def log_spaced_animation_steps(step_arrays: list[np.ndarray]) -> np.ndarray:
    steps = np.concatenate(
        [
            np.asarray(step_array, dtype=float).ravel()
            for step_array in step_arrays
            if len(np.asarray(step_array).ravel())
        ]
    )
    steps = steps[np.isfinite(steps) & (steps >= 1.0)]
    if len(steps) == 0:
        return np.array([1.0])

    start = float(np.nanmin(steps))
    stop = float(np.nanmax(steps))
    n_frames = max(2, int(round(ANIMATION_DURATION_SECONDS * ANIMATION_FPS)))
    if np.isclose(start, stop):
        return np.full(n_frames, start)
    return np.geomspace(start, stop, n_frames)


def focus_animation_data(run: dict) -> dict:
    loss_steps = run["steps_for_log"][: len(run["val_losses"])]
    acc_steps = run["steps_for_log"][: len(run["val_accs"])]
    data = {
        "loss_steps": loss_steps,
        "loss": run["val_losses"],
        "acc_steps": acc_steps,
        "acc": run["val_accs"],
        "tau_steps": np.array([], dtype=float),
        "tau_net": None,
    }
    if not run["spectral_missing"]:
        data["tau_steps"] = run["spec_steps_for_log"]
        data["tau_net"] = run["tau_net"]
    data["frame_steps"] = log_spaced_animation_steps(
        [loss_steps, acc_steps, data["tau_steps"]]
    )
    return data


def setup_focus_animation_axes(run: dict, title: str):
    data = focus_animation_data(run)
    fig_anim, axes_anim = plt.subplots(
        2,
        1,
        figsize=(9.5, 6.8),
        sharex=True,
        height_ratios=[1.0, 1.25],
    )
    ax_loss = axes_anim[0]
    ax_acc = ax_loss.twinx()
    ax_tau = axes_anim[1]

    x_values = [data["loss_steps"], data["acc_steps"]]
    if len(data["tau_steps"]):
        x_values.append(data["tau_steps"])
    x_all = np.concatenate([x for x in x_values if len(x)])
    x_min = max(float(np.nanmin(x_all)), 1.0)
    x_max = float(np.nanmax(x_all))

    ax_loss.set_xscale("log")
    ax_tau.set_xscale("log")
    ax_loss.set_xlim(x_min, x_max)
    ax_loss.set_yscale("log")
    ax_loss.set_ylim(*positive_log_ylim([data["loss"]]))
    ax_loss.set_ylabel("validation loss")
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_ylabel("validation accuracy")

    tau_ylim_values = [run["task_tau"]]
    if data["tau_net"] is not None:
        tau_ylim_values.append(data["tau_net"])
    ax_tau.set_yscale("log")
    ax_tau.set_ylim(*positive_log_ylim(tau_ylim_values))
    ax_tau.set_xlabel("training step")
    ax_tau.set_ylabel("effective timescale")

    for ax in (ax_loss, ax_tau):
        ax.grid(True, which="both", axis="x", alpha=0.18)
        ax.grid(True, which="major", axis="y", alpha=0.18)

    for k, tau in enumerate(run["task_tau"]):
        ax_tau.axhline(
            tau,
            color=f"C{k}",
            ls="--",
            lw=1.5,
            alpha=0.8,
            label=rf"$\tau_{{task,{k}}}$",
        )

    if run["spectral_missing"]:
        ax_tau.text(
            0.5,
            0.5,
            "Missing spectral_trajectory.pt\n"
            "Cannot animate network effective timescales.",
            transform=ax_tau.transAxes,
            fontsize=10,
            va="center",
            ha="center",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75"},
        )

    fig_anim.suptitle(title, y=1.01, fontweight="bold")
    return fig_anim, ax_loss, ax_acc, ax_tau, data


def make_focus_animations(run: dict) -> None:
    run_stem = f"{run['variant']}_tau{run['tau']:g}_g{run['gain']:g}"
    title_prefix = (
        f"{variant_label(run['variant'])} RNN Focus: "
        rf"$\tau$={run['tau']:g}, g={run['gain']:g}"
    )

    # Version 1: progressively reveal the curves over log-spaced training steps.
    fig_reveal, ax_loss_reveal, ax_acc_reveal, ax_tau_reveal, anim_data = (
        setup_focus_animation_axes(
            run,
            f"{title_prefix}: progressive training curves",
        )
    )
    (loss_line,) = ax_loss_reveal.plot([], [], color="C0", lw=2.0, label="val loss")
    (acc_line,) = ax_acc_reveal.plot(
        [], [], color="0.25", lw=1.6, ls="--", label="val accuracy"
    )
    tau_lines = []
    if anim_data["tau_net"] is not None:
        for k in range(anim_data["tau_net"].shape[1]):
            (tau_line,) = ax_tau_reveal.plot(
                [], [], color=f"C{k}", lw=2.0, alpha=0.95, label=rf"$\tau_{{net,{k}}}$"
            )
            tau_lines.append(tau_line)

    handles_loss, labels_loss = ax_loss_reveal.get_legend_handles_labels()
    handles_acc, labels_acc = ax_acc_reveal.get_legend_handles_labels()
    ax_loss_reveal.legend(
        handles_loss + handles_acc,
        labels_loss + labels_acc,
        loc="best",
        fontsize=9,
    )
    ax_tau_reveal.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=9,
    )
    time_text = ax_loss_reveal.text(
        0.02,
        0.92,
        "",
        transform=ax_loss_reveal.transAxes,
        fontsize=10,
        fontweight="bold",
    )

    def update_reveal(frame_idx: int):
        step = anim_data["frame_steps"][frame_idx]
        loss_mask = anim_data["loss_steps"] <= step
        acc_mask = anim_data["acc_steps"] <= step
        loss_line.set_data(anim_data["loss_steps"][loss_mask], anim_data["loss"][loss_mask])
        acc_line.set_data(anim_data["acc_steps"][acc_mask], anim_data["acc"][acc_mask])
        if anim_data["tau_net"] is not None:
            tau_mask = anim_data["tau_steps"] <= step
            for k, tau_line in enumerate(tau_lines):
                tau_line.set_data(
                    anim_data["tau_steps"][tau_mask],
                    anim_data["tau_net"][tau_mask, k],
                )
        time_text.set_text(f"step = {int(step)}")
        return [loss_line, acc_line, *tau_lines, time_text]

    reveal_anim = animation.FuncAnimation(
        fig_reveal,
        update_reveal,
        frames=len(anim_data["frame_steps"]),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )
    plt.tight_layout()
    if SAVE_ANIMATIONS:
        save_focus_animation(
            reveal_anim,
            ANIMATION_DIR / f"{run_stem}_reveal.mp4",
        )
    plt.show()

    # Version 2: show all curves, then sweep a vertical time marker left to right.
    fig_marker, ax_loss_marker, ax_acc_marker, ax_tau_marker, anim_data = (
        setup_focus_animation_axes(
            run,
            f"{title_prefix}: moving training-time marker",
        )
    )
    ax_loss_marker.plot(
        anim_data["loss_steps"], anim_data["loss"], color="C0", lw=2.0, label="val loss"
    )
    ax_acc_marker.plot(
        anim_data["acc_steps"],
        anim_data["acc"],
        color="0.25",
        lw=1.6,
        ls="--",
        label="val accuracy",
    )
    if anim_data["tau_net"] is not None:
        for k in range(anim_data["tau_net"].shape[1]):
            ax_tau_marker.plot(
                anim_data["tau_steps"],
                anim_data["tau_net"][:, k],
                color=f"C{k}",
                lw=2.0,
                alpha=0.95,
                label=rf"$\tau_{{net,{k}}}$",
            )

    loss_marker = ax_loss_marker.axvline(1.0, color="k", lw=1.2, alpha=0.8)
    tau_marker = ax_tau_marker.axvline(1.0, color="k", lw=1.2, alpha=0.8)
    marker_text = ax_loss_marker.text(
        0.02,
        0.92,
        "",
        transform=ax_loss_marker.transAxes,
        fontsize=10,
        fontweight="bold",
    )

    handles_loss, labels_loss = ax_loss_marker.get_legend_handles_labels()
    handles_acc, labels_acc = ax_acc_marker.get_legend_handles_labels()
    ax_loss_marker.legend(
        handles_loss + handles_acc,
        labels_loss + labels_acc,
        loc="best",
        fontsize=9,
    )
    ax_tau_marker.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=9,
    )

    def update_marker(frame_idx: int):
        step = anim_data["frame_steps"][frame_idx]
        loss_marker.set_xdata([step, step])
        tau_marker.set_xdata([step, step])
        marker_text.set_text(f"step = {int(step)}")
        return [loss_marker, tau_marker, marker_text]

    marker_anim = animation.FuncAnimation(
        fig_marker,
        update_marker,
        frames=len(anim_data["frame_steps"]),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )
    plt.tight_layout()
    if SAVE_ANIMATIONS:
        save_focus_animation(
            marker_anim,
            ANIMATION_DIR / f"{run_stem}_marker.mp4",
        )
    plt.show()


for run in focus_runs:
    make_focus_animations(run)

