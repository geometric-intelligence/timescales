# %% [markdown]
# # Reservoir Computing Phase Transition — Frozen $W_\text{rec}$, Gain Sweep
#
# $W_\text{rec}$ is **frozen** at its random initialisation.  Only $W_\text{in}$
# and $W_\text{out}$ are trained.  We sweep the recurrent gain $g$ so that
# the Jacobian $J = (1-\alpha)I + \alpha g W_\text{rec}$ goes from heavily
# damped ($g \ll 1$, no long timescales) to edge-of-chaos ($g \approx 1$).
#
# **Key questions:**
# 1. Is there a critical $g$ below which the task cannot be solved?
# 2. Do networks at low $g$ use a qualitatively different mechanism
#    (e.g. direct pathway, transient amplification, complex-pair rotation)
#    compared to networks near $g = 1$ (slow eigenvalue modes)?

# %%
import os
import sys
import subprocess
import json
import glob

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()

os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())

from rnns.rnn import RNN, RNNLightning
from datamodules.flip_flop import FlipFlopDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_FIGS = True
FIGS_DIR = os.path.join("notebooks", "figs", "reservoir")
os.makedirs(FIGS_DIR, exist_ok=True)

NUM_MODES = 30   # number of top modes (by |λ|) to show in coupling heatmaps

# %% Discover latest sweep
LOGS_DIR = "logs/experiments"
sweep_dirs = sorted(
    [d for d in os.listdir(LOGS_DIR)
     if d.startswith("flip_flop_reservoir_g_sweep")],
    reverse=True,
)
if not sweep_dirs:
    raise RuntimeError(
        f"No flip_flop_reservoir_g_sweep directories in {LOGS_DIR}.\n"
        "Run:  python sweep.py --sweep sweep_configs/rnn/flip_flop_reservoir_g_sweep.yaml --gpus 0"
    )
sweep_dir = os.path.join(LOGS_DIR, sweep_dirs[0])
print(f"Using sweep: {sweep_dir}")

# %% Load data and training curves
records = []

for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
        continue
    if not exp_name.startswith("g_"):
        continue

    parts = exp_name.split("_")
    try:
        gain = float(parts[1])
    except (IndexError, ValueError):
        continue

    for sdn in sorted(os.listdir(exp_dir)):
        if not sdn.startswith("seed_"):
            continue
        seed = int(sdn.split("_")[1])
        seed_path = os.path.join(exp_dir, sdn)

        fvl = None
        rf = os.path.join(seed_path, "job_result.yaml")
        if os.path.exists(rf):
            with open(rf) as f:
                fvl = yaml.safe_load(f).get("final_val_loss")

        val_losses, val_accs, steps = [], [], []
        val_losses_per_bit, val_accs_per_bit = {}, {}
        lf = os.path.join(seed_path, "training_losses.json")
        if os.path.exists(lf):
            with open(lf) as f:
                ld = json.load(f)
            val_losses = ld.get("val_losses", [])
            val_accs = ld.get("val_accuracies", [])
            steps = ld.get("steps", [])
            val_losses_per_bit = ld.get("val_losses_per_bit", {})
            val_accs_per_bit = ld.get("val_accuracies_per_bit", {})

        if fvl is None and val_losses:
            fvl = val_losses[-1]
        if fvl is None:
            continue

        run_config_path = os.path.join(seed_path, "run_config.yaml")
        run_config = {}
        if os.path.exists(run_config_path):
            with open(run_config_path) as f:
                run_config = yaml.safe_load(f)

        records.append(dict(
            exp_name=exp_name, gain=gain, seed=seed,
            seed_path=seed_path, final_val_loss=fvl,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
            val_losses_per_bit=val_losses_per_bit,
            val_accs_per_bit=val_accs_per_bit,
            run_config=run_config,
        ))

df = pd.DataFrame(records)
gains = sorted(df["gain"].unique())
print(f"Loaded {len(df)} runs, gains: {gains}")

# %% Color palette
_palette = plt.cm.viridis(np.linspace(0.15, 0.95, len(gains)))
COLORS = {g: _palette[i] for i, g in enumerate(gains)}

# %% Plot 1: Validation loss vs training step — all gains
fig, ax = plt.subplots(figsize=(10, 5))
plotted = set()
for _, row in df.iterrows():
    g = row["gain"]
    vl = row["val_losses"]
    st = row["steps"][:len(vl)] if row["steps"] else list(range(1, len(vl) + 1))
    label = f"g = {g}" if g not in plotted else None
    ax.plot(st, vl, linewidth=1.4, color=COLORS[g], alpha=0.7, label=label)
    plotted.add(g)

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation loss", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2)
fig.suptitle("Reservoir ($W_{\\mathrm{rec}}$ frozen) — Validation Loss",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "reservoir_val_loss.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 2: Validation accuracy vs training step
fig, ax = plt.subplots(figsize=(10, 5))
plotted = set()
for _, row in df.iterrows():
    g = row["gain"]
    va = row["val_accs"]
    st = row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1))
    if not va:
        continue
    label = f"g = {g}" if g not in plotted else None
    ax.plot(st, va, linewidth=1.4, color=COLORS[g], alpha=0.7, label=label)
    plotted.add(g)

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation accuracy", fontsize=12)
ax.set_ylim(0.4, 1.02)
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2, loc="lower right")
fig.suptitle("Reservoir ($W_{\\mathrm{rec}}$ frozen) — Validation Accuracy",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "reservoir_val_accuracy.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 3: Final val loss vs gain (phase transition curve)
summary = (
    df.groupby("gain")["final_val_loss"]
    .agg(["mean", "std", "count"])
    .sort_index()
)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(summary.index, summary["mean"], yerr=summary["std"],
            fmt="o-", capsize=4, linewidth=1.8, markersize=7,
            color="#264653", ecolor="#adb5bd")
ax.set_xlabel("Recurrent gain $g$", fontsize=12)
ax.set_ylabel("Final validation loss", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
           label="$g = 1$ (edge of chaos)")
ax.legend(fontsize=10)
fig.suptitle("Phase Transition? — Final Loss vs Gain\n"
             "$W_{\\mathrm{rec}}$ frozen, only $W_{\\mathrm{in}}$/$W_{\\mathrm{out}}$ trained",
             fontsize=13, fontweight="bold", y=1.04)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "reservoir_phase_transition.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 4: Per-bit final accuracy vs gain
first_config = df.iloc[0]["run_config"] if len(df) > 0 else {}
n_bits = first_config.get("n_bits", 6)
p_pulse = first_config.get("p_pulse", 0.05)
pp_list = p_pulse if isinstance(p_pulse, list) else [p_pulse] * n_bits

bit_acc_records = []
for _, row in df.iterrows():
    apb = row["val_accs_per_bit"]
    if not apb:
        continue
    for ch_key, vals in apb.items():
        if vals:
            bit_idx = int(ch_key.replace("channel_", ""))
            bit_acc_records.append(dict(
                gain=row["gain"], seed=row["seed"],
                bit=bit_idx, final_acc=vals[-1],
            ))

if bit_acc_records:
    df_bit = pd.DataFrame(bit_acc_records)
    fig, ax = plt.subplots(figsize=(9, 5))
    bit_colors = [f"C{i}" for i in range(n_bits)]

    for bit_i in range(n_bits):
        sub = df_bit[df_bit["bit"] == bit_i]
        means = sub.groupby("gain")["final_acc"].mean()
        stds = sub.groupby("gain")["final_acc"].std().fillna(0)
        hold = 1.0 / pp_list[bit_i]
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    fmt="o-", capsize=3, linewidth=1.4, markersize=5,
                    color=bit_colors[bit_i],
                    label=f"Bit {bit_i} (p={pp_list[bit_i]}, hold≈{hold:.0f})")

    ax.set_xlabel("Recurrent gain $g$", fontsize=12)
    ax.set_ylabel("Final validation accuracy", fontsize=12)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="chance (50%)")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.suptitle("Per-Bit Accuracy vs Gain — Which bits need long timescales?",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, "reservoir_per_bit_accuracy.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()
else:
    print("No per-bit accuracy data found (re-run sweep with updated callback)")

# %% [markdown]
# ## Eigenspectrum & Mechanism Analysis
#
# For each gain value, load the trained (frozen) Jacobian, extract
# $W_\text{in}$, $W_\text{out}$, and compute timescales, transient
# amplification, and coupling structure.

# %% Load models and compute eigenspectra
eig_data = {}

for g in gains:
    row_data = df[df["gain"] == g].iloc[0]
    seed_path = row_data["seed_path"]
    config_file = os.path.join(seed_path, "run_config.yaml")
    with open(config_file) as f:
        run_config = yaml.safe_load(f)

    n_bits_g = run_config["n_bits"]
    p_pulse_cfg = run_config["p_pulse"]
    dt = run_config["dt"]
    tau = run_config["time_constants_config"]["values"][0]
    alpha = 1.0 - np.exp(-dt / tau)

    best_ckpts = glob.glob(
        os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        print(f"g={g}: no checkpoint found, skipping")
        continue

    ckpt = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)

    W_rec = W_out = W_in = None
    for key, val in ckpt["state_dict"].items():
        if "W_rec.weight" in key:
            W_rec = val.numpy()
        elif "W_out.weight" in key:
            W_out = val.numpy()
        elif "W_in.weight" in key:
            W_in = val.numpy()

    if W_rec is None:
        continue

    N = W_rec.shape[0]
    J = (1.0 - alpha) * np.eye(N) + alpha * g * W_rec
    eigenvalues, eigvecs = np.linalg.eig(J)

    eig_data[g] = dict(
        eigs=eigenvalues, eigvecs=eigvecs, J=J,
        W_out=W_out, W_in=W_in,
        alpha=alpha, n_bits=n_bits_g, p_pulse=p_pulse_cfg,
    )

print(f"Loaded eigenspectra for {len(eig_data)} gain values")

# %% Plot 5: Eigenvalue spectra (complex plane) — all gains
theta = np.linspace(0, 2 * np.pi, 200)

n_g = len(eig_data)
ncols = min(4, n_g)
nrows = (n_g + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                         squeeze=False)

for idx, g in enumerate(sorted(eig_data.keys())):
    ax = axes[idx // ncols, idx % ncols]
    eigs = eig_data[g]["eigs"]
    rho = np.max(np.abs(eigs))

    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.5, alpha=0.4)
    ax.scatter(eigs.real, eigs.imag, s=10, alpha=0.6,
               color=COLORS[g], edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(f"$g = {g}$\n$\\rho = {rho:.3f}$", fontsize=11)
    ax.set_xlabel("Re", fontsize=9)
    ax.set_ylabel("Im", fontsize=9)
    ax.grid(True, alpha=0.15)

for idx in range(n_g, nrows * ncols):
    axes[idx // ncols, idx % ncols].set_visible(False)

fig.suptitle("Jacobian Eigenspectra — Frozen $W_{\\mathrm{rec}}$, varying $g$",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "reservoir_eigenspectra.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 5b: Zoomed eigenvalue spectra (near unit circle)
ZOOM_XLIM = (0.85, 1.05)
ZOOM_YLIM = (-0.15, 0.15)

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                         squeeze=False)

for idx, g in enumerate(sorted(eig_data.keys())):
    ax = axes[idx // ncols, idx % ncols]
    eigs = eig_data[g]["eigs"]
    abs_eigs = np.abs(eigs)
    n_bits_g = eig_data[g]["n_bits"]
    rho = np.max(abs_eigs)

    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.5, alpha=0.4)
    ax.axhline(0, color="#bbb", linewidth=0.3, zorder=0)
    ax.axvline(0, color="#bbb", linewidth=0.3, zorder=0)

    top_idx = np.argsort(abs_eigs)[-n_bits_g:]
    rest_idx = np.argsort(abs_eigs)[:-n_bits_g]

    ax.scatter(eigs.real[rest_idx], eigs.imag[rest_idx], s=15, alpha=0.6,
               color=COLORS[g], edgecolors="none", zorder=3)
    ax.scatter(eigs.real[top_idx], eigs.imag[top_idx], s=25, alpha=0.95,
               color=COLORS[g], edgecolors="black", linewidths=0.5,
               zorder=4, label=f"Top {n_bits_g}")

    ax.set_xlim(*ZOOM_XLIM)
    ax.set_ylim(*ZOOM_YLIM)
    ax.set_title(f"$g = {g}$, $\\rho = {rho:.4f}$", fontsize=10)
    ax.set_xlabel("Re", fontsize=9)
    ax.set_ylabel("Im", fontsize=9)
    ax.grid(True, alpha=0.15)

for idx in range(n_g, nrows * ncols):
    axes[idx // ncols, idx % ncols].set_visible(False)

if n_g > 0:
    axes[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.7)

fig.suptitle("Eigenspectra (zoom) — Frozen $W_{\\mathrm{rec}}$, varying $g$",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "reservoir_eigenspectra_zoom.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 6: Effective timescale scree plot per gain
_sample = next(iter(eig_data.values()))
_pp = _sample["p_pulse"]
_pp_list = _pp if isinstance(_pp, list) else [_pp]
_holds = [1.0 / p for p in _pp_list]
_n_bits = _sample["n_bits"]
TAU_VIS_MAX = max(_holds) * 20   # cap display at 20× longest hold

fig, axes = plt.subplots(1, n_g, figsize=(4.5 * n_g, 4.5),
                         squeeze=False, sharey=True)

for col, g in enumerate(sorted(eig_data.keys())):
    ax = axes[0, col]
    data = eig_data[g]
    eigs = data["eigs"]
    abs_sorted = np.sort(np.abs(eigs))[::-1]
    N = len(abs_sorted)
    ranks = np.arange(1, N + 1)

    log_abs = np.log(np.clip(abs_sorted, 1e-12, None))
    tau_eff = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)
    tau_eff = np.clip(tau_eff, None, TAU_VIS_MAX * 5)

    n_clipped = np.sum(tau_eff >= TAU_VIS_MAX * 5)

    ax.scatter(ranks[_n_bits:], tau_eff[_n_bits:], s=12, color="#adb5bd",
               edgecolors="none", alpha=0.5, zorder=3)
    for ti in range(min(_n_bits, len(tau_eff))):
        ax.scatter(ranks[ti], tau_eff[ti], s=50, color=f"C{ti}",
                   edgecolors="white", linewidths=0.6, zorder=4)

    for hi, hold_val in enumerate(_holds):
        ax.axhline(hold_val, color=f"C{hi}", linewidth=0.7, linestyle=":",
                   alpha=0.4, zorder=1)

    ax.set_xlabel("Eigenvalue rank", fontsize=10)
    if col == 0:
        ax.set_ylabel(r"$\tau_{\mathrm{eff}} = -1/\ln|\lambda|$", fontsize=10)
    rho = np.max(np.abs(eigs))
    subtitle = f"$g = {g}$, $\\rho = {rho:.3f}$"
    if n_clipped > 0:
        subtitle += f"\n({n_clipped} modes clipped)"
    ax.set_title(subtitle, fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(0.5, TAU_VIS_MAX)
    ax.grid(True, alpha=0.1, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Effective Timescales — Frozen $W_{\\mathrm{rec}}$",
             fontsize=14, fontweight="bold")
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "reservoir_tau_eff_scree.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 6b: PCA dimensionality of latent dynamics
PCA_GAIN = 0.8#gains[-1]   # which network to analyze
PCA_N_TRAJ = 100

_pca_row = df[df["gain"] == PCA_GAIN].iloc[0]
_pca_seed_path = _pca_row["seed_path"]
_pca_cfg_file = os.path.join(_pca_seed_path, "run_config.yaml")
with open(_pca_cfg_file) as f:
    _pca_cfg = yaml.safe_load(f)

_pca_ckpts = glob.glob(os.path.join(_pca_seed_path, "checkpoints", "best-model-*.ckpt"))
assert _pca_ckpts, f"No checkpoint found for g={PCA_GAIN}"

_pca_model = RNN(
    input_size=_pca_cfg["n_bits"],
    hidden_size=_pca_cfg["hidden_size"],
    output_size=_pca_cfg["n_bits"],
    dt=_pca_cfg["dt"],
    time_constants_config=_pca_cfg.get("time_constants_config"),
    activation=getattr(nn, _pca_cfg["activation"]),
    learn_time_constants=_pca_cfg["learn_time_constants"],
    init_time_constant=_pca_cfg.get("init_time_constant"),
    shared_time_constant=_pca_cfg["shared_time_constant"],
    normalize_hidden=_pca_cfg["normalize_hidden"],
    zero_diag_wrec=_pca_cfg["zero_diag_wrec"],
    recurrent_gain=_pca_cfg["recurrent_gain"],
    noise_std=0.0,
    wrec_init=_pca_cfg["wrec_init"],
    alpha_parameterization=_pca_cfg["alpha_parameterization"],
    dynamics_type=_pca_cfg["dynamics_type"],
)
_pca_lit = RNNLightning(
    model=_pca_model,
    learning_rate=_pca_cfg["learning_rate"],
    weight_decay=_pca_cfg["weight_decay"],
    step_size=_pca_cfg.get("lr_step_size", _pca_cfg.get("step_size", 1000)),
    gamma=_pca_cfg["gamma"],
    task="flip_flop",
)
_pca_ckpt = torch.load(_pca_ckpts[0], map_location=device, weights_only=False)
_pca_lit.load_state_dict(_pca_ckpt["state_dict"])
_pca_lit.eval().to(device)

_pca_dm = FlipFlopDataModule(
    n_bits=_pca_cfg["n_bits"],
    p_pulse=_pca_cfg["p_pulse"],
    pulse_amplitude=_pca_cfg["pulse_amplitude"],
    num_time_steps=_pca_cfg["num_time_steps"],
    num_val_trajectories=PCA_N_TRAJ,
    batch_size=PCA_N_TRAJ,
)
_pca_dm.setup()
_pca_inp, _, _ = _pca_dm.val_dataset.tensors

with torch.no_grad():
    _pca_h_seq, _ = _pca_lit.model(_pca_inp.to(device), init_context=None)
    _pca_h = _pca_h_seq.cpu().numpy()   # (n_traj, T, hidden_size)

_pca_n_traj, _pca_T, _pca_H = _pca_h.shape
_pca_data = _pca_h.reshape(-1, _pca_H)  # (n_traj * T, hidden_size)
_pca_data -= _pca_data.mean(axis=0, keepdims=True)

from numpy.linalg import svd as _svd
_, _pca_s, _ = _svd(_pca_data, full_matrices=False)
_pca_var = _pca_s ** 2
_pca_var_frac = _pca_var / _pca_var.sum()
_pca_cum = np.cumsum(_pca_var_frac)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

MAX_COMP_SHOW = min(60, len(_pca_cum))
ax1.plot(np.arange(1, MAX_COMP_SHOW + 1), _pca_cum[:MAX_COMP_SHOW],
         "o-", markersize=4, linewidth=1.5, color="#264653")
for thresh in [0.90, 0.95, 0.99]:
    n_needed = int(np.searchsorted(_pca_cum, thresh)) + 1
    ax1.axhline(thresh, color="#adb5bd", linewidth=0.8, linestyle=":")
    ax1.annotate(f"{thresh*100:.0f}% \u2192 {n_needed} PCs",
                 xy=(n_needed, thresh),
                 textcoords="offset points", xytext=(8, -10),
                 fontsize=9, color="#e76f51", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#e76f51", lw=0.8))
ax1.set_xlabel("Number of principal components", fontsize=12)
ax1.set_ylabel("Cumulative variance explained", fontsize=12)
ax1.set_ylim(0, 1.02)
ax1.set_xlim(0, MAX_COMP_SHOW + 1)
ax1.grid(True, alpha=0.2)
ax1.set_title("Cumulative variance explained", fontsize=12)

ax2.bar(np.arange(1, MAX_COMP_SHOW + 1), _pca_var_frac[:MAX_COMP_SHOW],
        color="#2a9d8f", edgecolor="none", alpha=0.8)
ax2.set_xlabel("Principal component", fontsize=12)
ax2.set_ylabel("Fraction of variance", fontsize=12)
ax2.set_xlim(0, MAX_COMP_SHOW + 1)
ax2.grid(True, alpha=0.2, axis="y")
ax2.set_title("Variance per component", fontsize=12)

fig.suptitle(f"PCA Dimensionality — Reservoir g={PCA_GAIN}, "
             f"{PCA_N_TRAJ} traj, N={_pca_H}, n_bits={_pca_cfg['n_bits']}",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"reservoir_pca_dimensionality_g{PCA_GAIN}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

print(f"\nPCA summary for g={PCA_GAIN}:")
print(f"  Data matrix: {_pca_data.shape[0]} points x {_pca_data.shape[1]} dims")
for thresh in [0.80, 0.90, 0.95, 0.99]:
    n_needed = int(np.searchsorted(_pca_cum, thresh)) + 1
    print(f"  {thresh*100:.0f}% variance -> {n_needed} PCs")






# # %% [markdown]
# # ## Mechanism Diagnostics
# #
# # For each gain value we compute:
# # - **Transient amplification** $\|J^k\|_2$ and input→output gain
# #   $|W_\text{out} J^k W_\text{in}|$
# # - **Henrici non-normality**
# # - **Direct pathway** $W_\text{out} W_\text{in}$
# # - **Output coupling heatmap** $|W_\text{out} V|$

# # # %% Plot 7: Transient amplification — ||J^k||_2 overlay
# # K_MAX = 500

# # fig, ax = plt.subplots(figsize=(10, 5))
# # for g in sorted(eig_data.keys()):
# #     J = eig_data[g]["J"]
# #     Jk = np.eye(J.shape[0])
# #     norms = np.empty(K_MAX + 1)
# #     norms[0] = 1.0
# #     for k in range(1, K_MAX + 1):
# #         Jk = Jk @ J
# #         norms[k] = np.linalg.norm(Jk, ord=2)
# #     ax.semilogy(np.arange(K_MAX + 1), norms, linewidth=1.4,
# #                 color=COLORS[g], label=f"g = {g}")

# # ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
# # ax.set_xlabel("$k$ (time steps)", fontsize=12)
# # ax.set_ylabel("$\\|J^k\\|_2$", fontsize=12)
# # ax.legend(fontsize=9, ncol=2)
# # ax.grid(True, alpha=0.2)
# # ax.set_xlim(0, K_MAX)
# # fig.suptitle("Transient Amplification — Frozen $W_{\\mathrm{rec}}$",
# #              fontsize=14, fontweight="bold", y=1.02)
# # plt.tight_layout()
# # if SAVE_FIGS:
# #     fig.savefig(os.path.join(FIGS_DIR, "reservoir_transient_amp.pdf"),
# #                 bbox_inches="tight", dpi=150)
# # plt.show()

# # # %% Plot 8: Input→output transient gain per bit
# # fig, axes = plt.subplots(2, (n_g + 1) // 2,
# #                          figsize=(5 * ((n_g + 1) // 2), 9),
# #                          squeeze=False, sharey=True)

# # for idx, g in enumerate(sorted(eig_data.keys())):
# #     ax = axes[idx // ((n_g + 1) // 2), idx % ((n_g + 1) // 2)]
# #     data = eig_data[g]
# #     J = data["J"]
# #     W_out = data["W_out"]
# #     W_in = data["W_in"]
# #     n_bits_g = data["n_bits"]
# #     pp = data["p_pulse"]
# #     pp_list_g = pp if isinstance(pp, list) else [pp] * n_bits_g

# #     if W_out is None or W_in is None:
# #         ax.text(0.5, 0.5, "missing weights", transform=ax.transAxes, ha="center")
# #         continue

# #     Jk = np.eye(J.shape[0])
# #     io_gain = np.empty((K_MAX + 1, n_bits_g))
# #     for k in range(K_MAX + 1):
# #         G = W_out @ Jk @ W_in
# #         for b in range(n_bits_g):
# #             io_gain[k, b] = abs(G[b, b])
# #         Jk = Jk @ J

# #     ks = np.arange(K_MAX + 1)
# #     for b in range(n_bits_g):
# #         ax.semilogy(ks, io_gain[:, b], linewidth=1.0,
# #                     label=f"Bit {b} (p={pp_list_g[b]})")

# #     ax.set_xlabel("$k$", fontsize=10)
# #     ax.set_title(f"$g = {g}$", fontsize=11)
# #     ax.legend(fontsize=6, loc="upper right", ncol=2)
# #     ax.grid(True, alpha=0.15)
# #     ax.set_xlim(0, K_MAX)
# #     ax.spines["top"].set_visible(False)
# #     ax.spines["right"].set_visible(False)

# # axes[0, 0].set_ylabel("$|W_{\\mathrm{out}} J^k W_{\\mathrm{in}}|_{ii}$",
# #                        fontsize=11)
# # for idx in range(n_g, axes.shape[0] * axes.shape[1]):
# #     axes[idx // ((n_g + 1) // 2), idx % ((n_g + 1) // 2)].set_visible(False)

# # fig.suptitle("Input→Output Gain per Bit — How long does each bit persist?",
# #              fontsize=13, fontweight="bold", y=1.02)
# # plt.tight_layout()
# # if SAVE_FIGS:
# #     fig.savefig(os.path.join(FIGS_DIR, "reservoir_io_gain.pdf"),
# #                 bbox_inches="tight", dpi=150)
# # plt.show()

# # # %% Non-normality & direct pathway summary
# # print("\n" + "=" * 75)
# # print("Non-normality & Direct Pathway Summary")
# # print("=" * 75)

# # summary_rows = []
# # for g in sorted(eig_data.keys()):
# #     data = eig_data[g]
# #     J = data["J"]
# #     eigs = data["eigs"]
# #     W_out = data["W_out"]
# #     W_in = data["W_in"]
# #     N = J.shape[0]

# #     rho = np.max(np.abs(eigs))
# #     frob_sq = np.linalg.norm(J, "fro") ** 2
# #     eig_sq = np.sum(np.abs(eigs) ** 2)
# #     henrici = np.sqrt(max(frob_sq - eig_sq, 0.0))
# #     henrici_norm = henrici / np.linalg.norm(J, "fro")

# #     Jk = np.eye(N)
# #     peak = 1.0
# #     for k in range(1, K_MAX + 1):
# #         Jk = Jk @ J
# #         peak = max(peak, np.linalg.norm(Jk, ord=2))

# #     direct_diag = np.diag(W_out @ W_in) if (W_out is not None and W_in is not None) else np.zeros(1)
# #     direct_norm = np.linalg.norm(W_out @ W_in) if (W_out is not None and W_in is not None) else 0.0

# #     summary_rows.append(dict(
# #         g=g, rho=rho, henrici=henrici, henrici_norm=henrici_norm,
# #         peak_Jk=peak, amp_ratio=peak / max(rho, 1e-12),
# #         direct_norm=direct_norm,
# #         direct_diag_mean=np.mean(np.abs(direct_diag)),
# #     ))

# #     print(f"\n  g = {g}:")
# #     print(f"    ρ(J) = {rho:.4f}")
# #     print(f"    Henrici = {henrici:.3f}  (norm'd = {henrici_norm:.3f})")
# #     print(f"    peak ||J^k||₂ = {peak:.3f}  ({peak/max(rho,1e-12):.1f}× spectral)")
# #     print(f"    ||W_out W_in||_F = {direct_norm:.4f}")
# #     diag_str = "  ".join(f"{d:+.3f}" for d in direct_diag[:6])
# #     print(f"    diag(W_out W_in) = [{diag_str}]")

# # df_summary = pd.DataFrame(summary_rows)

# # # %% Plot 9: Summary — ρ, Henrici, peak amplification vs g
# # fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# # axes[0].plot(df_summary["g"], df_summary["rho"], "o-", color="#264653",
# #              linewidth=1.5, markersize=6)
# # axes[0].axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
# # axes[0].set_xlabel("$g$", fontsize=12)
# # axes[0].set_ylabel("$\\rho(J)$", fontsize=12)
# # axes[0].set_title("Spectral Radius", fontsize=12)
# # axes[0].grid(True, alpha=0.3)

# # axes[1].plot(df_summary["g"], df_summary["henrici_norm"], "o-",
# #              color="#e76f51", linewidth=1.5, markersize=6)
# # axes[1].set_xlabel("$g$", fontsize=12)
# # axes[1].set_ylabel("Henrici / $\\|J\\|_F$", fontsize=12)
# # axes[1].set_title("Normalised Non-normality", fontsize=12)
# # axes[1].grid(True, alpha=0.3)

# # axes[2].plot(df_summary["g"], df_summary["amp_ratio"], "o-",
# #              color="#2a9d8f", linewidth=1.5, markersize=6)
# # axes[2].axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
# # axes[2].set_xlabel("$g$", fontsize=12)
# # axes[2].set_ylabel("peak $\\|J^k\\| / \\rho^k$", fontsize=12)
# # axes[2].set_title("Transient Amplification Ratio", fontsize=12)
# # axes[2].grid(True, alpha=0.3)

# # fig.suptitle("Reservoir Diagnostics vs Gain",
# #              fontsize=14, fontweight="bold", y=1.04)
# # plt.tight_layout()
# # if SAVE_FIGS:
# #     fig.savefig(os.path.join(FIGS_DIR, "reservoir_diagnostics_vs_g.pdf"),
# #                 bbox_inches="tight", dpi=150)
# # plt.show()

# %% Plot 10: Output & Input coupling heatmaps — top NUM_MODES modes per gain
for g in sorted(eig_data.keys()):
    data = eig_data[g]
    eigs = data["eigs"]
    V = data["eigvecs"]
    W_out = data["W_out"]
    W_in = data["W_in"]
    n_bits_g = data["n_bits"]
    pp = data["p_pulse"]
    pp_list_g = pp if isinstance(pp, list) else [pp] * n_bits_g
    N = len(eigs)

    if W_out is None or W_in is None:
        continue

    n_show = min(NUM_MODES, N)
    abs_rank_order = np.argsort(np.abs(eigs))[::-1]
    top_idx = abs_rank_order[:n_show]

    abs_top = np.abs(eigs[top_idx])
    log_abs = np.log(np.clip(abs_top, 1e-12, None))
    tau_top = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)
    tau_top = np.clip(tau_top, 0, 1e6)

    rho = np.max(np.abs(eigs))

    is_complex = np.abs(eigs[top_idx].imag) > 1e-8

    xtick_labels = []
    xtick_colors = []
    for m in range(n_show):
        tau_str = f"{tau_top[m]:.0f}" if tau_top[m] < 1e5 else "∞"
        marker = "*" if is_complex[m] else ""
        xtick_labels.append(f"{m+1}{marker}\n(τ={tau_str})")
        xtick_colors.append("#2563eb" if is_complex[m] else "black")

    def _apply_pair_markers(ax, n_show, is_complex, n_bits_g):
        """Add bracket annotations connecting adjacent conjugate pairs."""
        ax.set_xticks(range(n_show))
        ax.set_xticklabels(xtick_labels, fontsize=6)
        for tick_label, color in zip(ax.get_xticklabels(), xtick_colors):
            tick_label.set_color(color)
            if color != "black":
                tick_label.set_fontweight("bold")

        m = 0
        while m < n_show - 1:
            if is_complex[m] and is_complex[m + 1]:
                lam_a = eigs[top_idx[m]]
                lam_b = eigs[top_idx[m + 1]]
                if abs(lam_a - lam_b.conjugate()) < 1e-6:
                    y_bot = n_bits_g - 0.5
                    ax.annotate("", xy=(m, y_bot + 0.15), xytext=(m + 1, y_bot + 0.15),
                                arrowprops=dict(arrowstyle="-", color="#2563eb",
                                                lw=1.8), annotation_clip=False)
                    ax.text((m + m + 1) / 2, y_bot + 0.35, "cc",
                            ha="center", va="bottom", fontsize=5,
                            color="#2563eb", clip_on=False,
                            transform=ax.transData)
                    m += 2
                    continue
            m += 1

    # --- Output coupling ---
    coupling_out = np.abs(W_out @ V)[:, top_idx]    # (n_bits, n_show)

    fig, ax = plt.subplots(figsize=(max(n_show * 0.45, 8), max(n_bits_g * 0.7, 3)))
    im = ax.imshow(coupling_out, cmap="YlOrRd", aspect="auto",
                   interpolation="nearest")

    ax.set_yticks(range(n_bits_g))
    ax.set_yticklabels([f"Bit {i} (p={pp_list_g[i]})" for i in range(n_bits_g)],
                       fontsize=9)
    _apply_pair_markers(ax, n_show, is_complex, n_bits_g)
    ax.set_xlabel("Eigenmode rank (by $|\\lambda|$) — * = complex, cc = conjugate pair",
                  fontsize=10)
    ax.set_ylabel("Output bit", fontsize=11)
    ax.axvline(n_bits_g - 0.5, color="white", linewidth=1.5, linestyle="--",
               alpha=0.8)

    for i in range(n_bits_g):
        for j in range(n_show):
            val = coupling_out[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if val > 0.5 * coupling_out.max() else "black")

    plt.colorbar(im, ax=ax, label="$|W_{\\mathrm{out}} V|$", shrink=0.8)
    fig.suptitle(f"Output Coupling — $g = {g}$, $\\rho = {rho:.3f}$\n"
                 f"$|W_{{\\mathrm{{out}}}} V|$, top {n_show} of {N} modes",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"reservoir_output_coupling_g{g}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

    # --- Input coupling ---
    V_inv = np.linalg.pinv(V)
    coupling_in = np.abs(V_inv @ W_in)[top_idx]     # (n_show, n_bits)

    fig, ax = plt.subplots(figsize=(max(n_show * 0.45, 8), max(n_bits_g * 0.7, 3)))
    im = ax.imshow(coupling_in.T, cmap="YlGnBu", aspect="auto",
                   interpolation="nearest")

    ax.set_yticks(range(n_bits_g))
    ax.set_yticklabels([f"Input {i} (p={pp_list_g[i]})" for i in range(n_bits_g)],
                       fontsize=9)
    _apply_pair_markers(ax, n_show, is_complex, n_bits_g)
    ax.set_xlabel("Eigenmode rank (by $|\\lambda|$) — * = complex, cc = conjugate pair",
                  fontsize=10)
    ax.set_ylabel("Input bit", fontsize=11)
    ax.axvline(n_bits_g - 0.5, color="white", linewidth=1.5, linestyle="--",
               alpha=0.8)

    for i in range(n_bits_g):
        for j in range(n_show):
            val = coupling_in[j, i]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if val > 0.5 * coupling_in.max() else "black")

    plt.colorbar(im, ax=ax, label="$|V^{-1} W_{\\mathrm{in}}|$", shrink=0.8)
    fig.suptitle(f"Input Coupling — $g = {g}$, $\\rho = {rho:.3f}$\n"
                 f"$|V^{{-1}} W_{{\\mathrm{{in}}}}|$, top {n_show} of {N} modes",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"reservoir_input_coupling_g{g}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

    # Dominant mode per bit
    dominant = np.argmax(coupling_out, axis=1)
    print(f"\ng = {g} (ρ = {rho:.3f}): Dominant mode per output bit:")
    for bit_i in range(n_bits_g):
        mode_j = dominant[bit_i]
        tau_str = f"{tau_top[mode_j]:.1f}" if tau_top[mode_j] < 1e5 else "∞"
        print(f"  Bit {bit_i} (p={pp_list_g[bit_i]}, "
              f"hold~{1.0/pp_list_g[bit_i]:.0f})  <--  "
              f"Mode {mode_j+1} (τ_eff={tau_str})")

# %% Plot 11: Mode coupling analysis — configurable scatter plots
# Options:
INCLUDE_REAL = False      # include real eigenvalues (θ=0 or π)
USE_THETA_TAU = False     # x-axis: True = θ·τ, False = |θ|
USE_INV_THETA = False     # x-axis: True = 1/|θ| (rotational timescale), overrides USE_THETA_TAU
PER_BIT = False           # True = one subplot per bit, False = max across bits
COUPLING_VS_TAU = True    # also produce coupling vs τ_eff scatter

from scipy.stats import spearmanr as _spearmanr

INV_THETA_CAP = 200       # cap 1/|θ| for visualization (modes with θ→0)

_mode_records = []

for g in sorted(eig_data.keys()):
    data = eig_data[g]
    eigs = data["eigs"]
    V = data["eigvecs"]
    W_out = data["W_out"]
    n_bits_g = data["n_bits"]
    N = len(eigs)
    if W_out is None:
        continue

    abs_rank_order = np.argsort(np.abs(eigs))[::-1]
    coupling_raw = np.abs(W_out @ V)
    row_sums = coupling_raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
    coupling_norm = coupling_raw / row_sums

    visited = set()
    for rank, idx in enumerate(abs_rank_order):
        if idx in visited:
            continue
        lam = eigs[idx]
        r_mag = abs(lam)
        log_r = np.log(max(r_mag, 1e-12))
        tau = -1.0 / min(log_r, -1e-10)
        theta = abs(np.angle(lam))

        if abs(lam.imag) < 1e-8:
            if not INCLUDE_REAL:
                visited.add(idx)
                continue
            visited.add(idx)
            mode_coupling_norm = coupling_norm[:, idx]
            is_pair = False
        else:
            partner = None
            for rank2, idx2 in enumerate(abs_rank_order):
                if idx2 != idx and idx2 not in visited:
                    if abs(eigs[idx2] - lam.conjugate()) < 1e-6:
                        partner = idx2
                        break
            if partner is None:
                visited.add(idx)
                continue
            visited.update([idx, partner])
            mode_coupling_norm = coupling_norm[:, idx] + coupling_norm[:, partner]
            is_pair = True

        theta_tau = theta * tau
        inv_theta = min(1.0 / max(theta, 1e-6), INV_THETA_CAP)
        base = dict(g=g, theta=theta, r_mag=r_mag, tau=tau,
                    theta_tau=theta_tau, inv_theta=inv_theta,
                    rank=rank + 1, is_pair=is_pair)

        if PER_BIT:
            for bit_i in range(n_bits_g):
                _mode_records.append({**base, "coupling": float(mode_coupling_norm[bit_i]),
                                      "bit": bit_i})
        else:
            _mode_records.append({**base, "coupling": float(mode_coupling_norm.max()),
                                  "bit": int(np.argmax(mode_coupling_norm))})

if _mode_records:
    _mdf = pd.DataFrame(_mode_records)
    _all_gains = sorted(_mdf["g"].unique())
    _n_g = len(_all_gains)
    _tau_thresh = min(_holds)

    if USE_INV_THETA:
        _x_col, _x_lab, _x_short = "inv_theta", "$1/|\\theta|$ (rot. timescale)", "1/θ"
    elif USE_THETA_TAU:
        _x_col, _x_lab, _x_short = "theta_tau", "$\\theta \\cdot \\tau_{\\mathrm{eff}}$", "θτ"
    else:
        _x_col, _x_lab, _x_short = "theta", "$|\\theta|$", "θ"

    def _sig_stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    def _plot_scatter(df, gains, x_col, x_label, y_col, y_label,
                      title, fname, include_real):
        """Generic per-gain scatter with background/eligible split."""
        bits = sorted(df["bit"].unique()) if PER_BIT else [None]
        ncols = min(4, len(gains))
        nrows = (len(gains) + ncols - 1) // ncols

        for bit_i in bits:
            if bit_i is not None:
                df_b = df[df["bit"] == bit_i]
                pp_list_g = eig_data[gains[0]]["p_pulse"]
                if not isinstance(pp_list_g, list):
                    pp_list_g = [pp_list_g] * (bit_i + 1)
                hold_i = 1.0 / max(pp_list_g[bit_i], 1e-8)
                bit_suffix = f" — Bit {bit_i} (hold ≈ {hold_i:.0f})"
                fname_suffix = f"_bit{bit_i}"
            else:
                df_b = df
                bit_suffix = ""
                fname_suffix = ""

            fig, axes = plt.subplots(nrows, ncols,
                                      figsize=(4.5 * ncols, 4 * nrows),
                                      squeeze=False)
            for i, g in enumerate(gains):
                ax = axes[i // ncols, i % ncols]
                sub = df_b[df_b["g"] == g]
                bg = sub[sub["tau"] < _tau_thresh]
                elig = sub[sub["tau"] >= _tau_thresh]

                if include_real:
                    bg_pair = bg[bg["is_pair"]]
                    bg_real = bg[~bg["is_pair"]]
                    elig_pair = elig[elig["is_pair"]]
                    elig_real = elig[~elig["is_pair"]]

                    ax.scatter(bg_pair[x_col], bg_pair[y_col],
                               s=14, color="#d3d3d3", edgecolors="#aaa",
                               linewidths=0.2, alpha=0.45, zorder=2, marker="o",
                               label="bg pair" if i == 0 else None)
                    ax.scatter(bg_real[x_col], bg_real[y_col],
                               s=20, color="#d3d3d3", edgecolors="#aaa",
                               linewidths=0.2, alpha=0.45, zorder=2, marker="D",
                               label="bg real" if i == 0 else None)
                    if len(elig_pair) > 0:
                        ax.scatter(elig_pair[x_col], elig_pair[y_col],
                                   s=50, color="#e63946", edgecolors="black",
                                   linewidths=0.4, alpha=0.8, zorder=4, marker="o",
                                   label="elig pair" if i == 0 else None)
                    if len(elig_real) > 0:
                        ax.scatter(elig_real[x_col], elig_real[y_col],
                                   s=65, color="#2563eb", edgecolors="black",
                                   linewidths=0.4, alpha=0.8, zorder=5, marker="D",
                                   label="elig real" if i == 0 else None)
                else:
                    ax.scatter(bg[x_col], bg[y_col],
                               s=14, color="#d3d3d3", edgecolors="#aaa",
                               linewidths=0.2, alpha=0.45, zorder=2,
                               label=f"τ < {_tau_thresh:.0f}" if i == 0 else None)
                    if len(elig) > 0:
                        ax.scatter(elig[x_col], elig[y_col],
                                   s=50, color="#e63946", edgecolors="black",
                                   linewidths=0.4, alpha=0.8, zorder=4,
                                   label=f"τ ≥ {_tau_thresh:.0f}" if i == 0 else None)

                if len(elig) > 3:
                    rho_e, pval_e = _spearmanr(elig[x_col], elig[y_col])
                    z = np.polyfit(elig[x_col], elig[y_col], 1)
                    x_fit = np.linspace(elig[x_col].min(), elig[x_col].max(), 50)
                    ax.plot(x_fit, np.polyval(z, x_fit), "k--", linewidth=1.0,
                            alpha=0.5, zorder=3)
                    ax.text(0.97, 0.97,
                            f"n = {len(elig)}\n"
                            f"ρ = {rho_e:.2f} {_sig_stars(pval_e)}\n"
                            f"p = {pval_e:.2g}",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=7.5,
                            bbox=dict(facecolor="white", alpha=0.85,
                                      edgecolor="#333", boxstyle="round,pad=0.3"))
                elif len(elig) == 0:
                    ax.text(0.5, 0.5, "no eligible\nmodes",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=9, color="#999")

                ax.set_title(f"$g = {g}$", fontsize=11, fontweight="bold")
                ax.set_xlabel(x_label, fontsize=10)
                if i % ncols == 0:
                    ax.set_ylabel(y_label, fontsize=10)
                ax.grid(True, alpha=0.1)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            for i in range(len(gains), nrows * ncols):
                axes[i // ncols, i % ncols].set_visible(False)

            axes[0, 0].legend(fontsize=6.5, loc="center right", framealpha=0.8)
            fig.suptitle(f"{title}{bit_suffix}", fontsize=12,
                         fontweight="bold", y=1.04)
            plt.tight_layout()
            if SAVE_FIGS:
                fig.savefig(os.path.join(FIGS_DIR, f"{fname}{fname_suffix}.pdf"),
                            bbox_inches="tight", dpi=150)
            plt.show()

    # --- Scatter 1: x vs coupling ---
    _real_tag = " (incl. real)" if INCLUDE_REAL else ""
    _plot_scatter(_mdf, _all_gains, _x_col, _x_lab, "coupling",
                  "Norm. output coupling",
                  f"{_x_lab} vs Coupling{_real_tag}",
                  "reservoir_rotation_vs_coupling", INCLUDE_REAL)

    # --- Scatter 2: τ_eff vs coupling ---
    if COUPLING_VS_TAU:
        _plot_scatter(_mdf, _all_gains, "tau", "$\\tau_{\\mathrm{eff}}$",
                      "coupling", "Norm. output coupling",
                      f"$\\tau_{{\\mathrm{{eff}}}}$ vs Coupling{_real_tag}",
                      "reservoir_tau_vs_coupling", INCLUDE_REAL)

    # ================================================================
    # Print summary
    # ================================================================
    print("\n=== Mode Coupling Summary ===")
    print(f"  INCLUDE_REAL={INCLUDE_REAL}, USE_THETA_TAU={USE_THETA_TAU}, "
          f"USE_INV_THETA={USE_INV_THETA}, PER_BIT={PER_BIT}")
    print(f"  x-axis: {_x_short}  |  Eligibility: τ_eff ≥ {_tau_thresh:.0f}\n")

    if not PER_BIT:
        for g in _all_gains:
            sub = _mdf[_mdf["g"] == g]
            elig = sub[sub["tau"] >= _tau_thresh]
            n_real_e = (~elig["is_pair"]).sum() if len(elig) > 0 else 0
            top = sub.sort_values("coupling", ascending=False).head(5)

            rho_a, pval_a = (_spearmanr(sub[_x_col], sub["coupling"])
                             if len(sub) > 3 else (0.0, 1.0))
            if len(elig) > 3:
                rho_e, pval_e = _spearmanr(elig[_x_col], elig["coupling"])
            else:
                rho_e, pval_e = 0.0, 1.0

            print(f"g = {g}: {len(elig)}/{len(sub)} eligible "
                  f"({n_real_e}r + {len(elig)-n_real_e}cc)  |  "
                  f"ρ_all({_x_short}) = {rho_a:.3f} (p={pval_a:.2g})  |  "
                  f"ρ_elig({_x_short}) = {rho_e:.3f} (p={pval_e:.2g}) "
                  f"{_sig_stars(pval_e)}")
            for _, r in top.iterrows():
                kind = "real" if not r["is_pair"] else "pair"
                print(f"  Rank {int(r['rank'])} [{kind}]: "
                      f"θ={r['theta']:.4f} ({np.degrees(r['theta']):.1f}°), "
                      f"|λ|={r['r_mag']:.4f}, τ={r['tau']:.0f}, "
                      f"coupling={r['coupling']:.4f} → bit {int(r['bit'])}")

    # ================================================================
    # Multiple regression: coupling ~ β₁·|θ| + β₂·τ_eff  (standardised)
    # Also reports model with 1/|θ| for comparison.
    # ================================================================
    from numpy.linalg import lstsq as _lstsq
    from scipy.stats import t as _tdist

    def _ols_summary(y, X_raw, names):
        """OLS on z-scored predictors → standardised β, t, p, R²."""
        n, k = X_raw.shape
        mu = X_raw.mean(axis=0); sd = X_raw.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        X_z = (X_raw - mu) / sd
        X = np.column_stack([X_z, np.ones(n)])
        beta, res, _, _ = _lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        dof = max(n - k - 1, 1)
        mse = ss_res / dof
        try:
            cov_b = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov_b)[:k])
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)
        t_vals = beta[:k] / np.where(se > 1e-12, se, np.inf)
        p_vals = 2 * _tdist.sf(np.abs(t_vals), dof)
        rows = []
        for i, nm in enumerate(names):
            rows.append(dict(name=nm, beta=beta[i], se=se[i],
                             t=t_vals[i], p=p_vals[i]))
        return dict(r2=r2, n=n, rows=rows)

    print("\n=== Multiple Regression (eligible modes, per gain) ===")
    print("  coupling ~ β₁·|θ| + β₂·τ  (z-scored predictors → "
          "standardised β, comparable magnitudes)")
    print("  Also: coupling ~ β₁·(1/|θ|) + β₂·τ\n")

    if not PER_BIT:
        hdr = (f"{'g':>5s} {'n':>3s} │ "
               f"{'R²':>5s}  {'β_θ':>6s} {'p':>7s}  {'β_τ':>6s} {'p':>7s} │ "
               f"{'R²':>5s}  {'β_1/θ':>6s} {'p':>7s}  {'β_τ':>6s} {'p':>7s}")
        print(f"{'':>10s}  {'model: |θ| + τ':^33s} │ {'model: 1/|θ| + τ':^33s}")
        print(hdr)
        print("─" * len(hdr))
        for g in _all_gains:
            elig = _mdf[(_mdf["g"] == g) & (_mdf["tau"] >= _tau_thresh)]
            if len(elig) < 5:
                print(f"{g:5.2f} {len(elig):3d} │ (too few eligible modes)")
                continue
            c = elig["coupling"].values
            X_theta = np.column_stack([elig["theta"].values,
                                       elig["tau"].values])
            X_inv   = np.column_stack([elig["inv_theta"].values,
                                       elig["tau"].values])
            s1 = _ols_summary(c, X_theta, ["|θ|", "τ"])
            s2 = _ols_summary(c, X_inv,   ["1/|θ|", "τ"])
            r1, r2_ = s1["rows"], s2["rows"]
            print(f"{g:5.2f} {s1['n']:3d} │ "
                  f"{s1['r2']:5.3f}  {r1[0]['beta']:+6.3f} "
                  f"{r1[0]['p']:7.1e}  {r1[1]['beta']:+6.3f} "
                  f"{r1[1]['p']:7.1e} │ "
                  f"{s2['r2']:5.3f}  {r2_[0]['beta']:+6.3f} "
                  f"{r2_[0]['p']:7.1e}  {r2_[1]['beta']:+6.3f} "
                  f"{r2_[1]['p']:7.1e}")

# %% Plot 12: Direct pathway heatmap — W_out @ W_in for each gain
n_g_direct = len([g for g in eig_data if eig_data[g]["W_out"] is not None
                  and eig_data[g]["W_in"] is not None])
if n_g_direct > 0:
    ncols_d = min(4, n_g_direct)
    nrows_d = (n_g_direct + ncols_d - 1) // ncols_d
    fig, axes = plt.subplots(nrows_d, ncols_d,
                             figsize=(3.5 * ncols_d, 3 * nrows_d),
                             squeeze=False)

    vmax_all = max(
        np.abs(eig_data[g]["W_out"] @ eig_data[g]["W_in"]).max()
        for g in eig_data
        if eig_data[g]["W_out"] is not None and eig_data[g]["W_in"] is not None
    )

    for idx, g in enumerate(sorted(eig_data.keys())):
        data = eig_data[g]
        if data["W_out"] is None or data["W_in"] is None:
            continue
        ax = axes[idx // ncols_d, idx % ncols_d]
        d = data["W_out"] @ data["W_in"]
        n_b = d.shape[0]
        im = ax.imshow(d, cmap="RdBu_r", aspect="equal",
                       vmin=-vmax_all, vmax=vmax_all)
        ax.set_xticks(range(n_b))
        ax.set_xticklabels([f"{i}" for i in range(n_b)], fontsize=7)
        ax.set_yticks(range(n_b))
        ax.set_yticklabels([f"{i}" for i in range(n_b)], fontsize=7)
        for i in range(n_b):
            for j in range(n_b):
                ax.text(j, i, f"{d[i,j]:.2f}", ha="center", va="center",
                        fontsize=5,
                        color="white" if abs(d[i,j]) > 0.6*vmax_all else "black")
        ax.set_title(f"$g = {g}$", fontsize=10)

    for idx in range(n_g_direct, nrows_d * ncols_d):
        axes[idx // ncols_d, idx % ncols_d].set_visible(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                 label="$W_{\\mathrm{out}} W_{\\mathrm{in}}$")
    fig.suptitle("Direct Pathway $W_{\\mathrm{out}} W_{\\mathrm{in}}$ — Frozen $W_{\\mathrm{rec}}$",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, "reservoir_direct_pathway.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% Example trajectories — best seed per gain
SEQ_IDX = 0

for g in sorted(eig_data.keys()):
    row_data = df[df["gain"] == g].sort_values("final_val_loss").iloc[0]
    seed_path = row_data["seed_path"]
    config_file = os.path.join(seed_path, "run_config.yaml")
    with open(config_file) as f:
        run_config = yaml.safe_load(f)

    n_bits_g = run_config["n_bits"]
    p_pulse_cfg = run_config["p_pulse"]
    pp_list_g = p_pulse_cfg if isinstance(p_pulse_cfg, list) else [p_pulse_cfg] * n_bits_g
    num_ts = run_config["num_time_steps"]

    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        continue

    model = RNN(
        input_size=n_bits_g,
        hidden_size=run_config["hidden_size"],
        output_size=n_bits_g,
        dt=run_config["dt"],
        time_constants_config=run_config.get("time_constants_config"),
        activation=getattr(nn, run_config["activation"]),
        learn_time_constants=run_config["learn_time_constants"],
        init_time_constant=run_config.get("init_time_constant"),
        shared_time_constant=run_config["shared_time_constant"],
        normalize_hidden=run_config["normalize_hidden"],
        zero_diag_wrec=run_config["zero_diag_wrec"],
        recurrent_gain=run_config["recurrent_gain"],
        noise_std=0.0,
        wrec_init=run_config["wrec_init"],
        alpha_parameterization=run_config["alpha_parameterization"],
        dynamics_type=run_config["dynamics_type"],
    )
    lit = RNNLightning(
        model=model,
        learning_rate=run_config["learning_rate"],
        weight_decay=run_config["weight_decay"],
        step_size=run_config.get("lr_step_size", run_config.get("step_size", 1000)),
        gamma=run_config["gamma"],
        task="flip_flop",
    )
    ckpt_data = torch.load(best_ckpts[0], map_location=device, weights_only=False)
    lit.load_state_dict(ckpt_data["state_dict"])
    lit.eval().to(device)

    dm = FlipFlopDataModule(
        n_bits=n_bits_g,
        p_pulse=p_pulse_cfg,
        pulse_amplitude=run_config.get("pulse_amplitude", 1.0),
        num_time_steps=num_ts,
        num_val_trajectories=10,
        batch_size=10,
    )
    dm.setup()
    inp, _, tgt = dm.val_dataset.tensors

    with torch.no_grad():
        _, out = lit.model(inp.to(device), init_context=None)
        out_prob = torch.sigmoid(out).cpu()

    t_arr = np.arange(num_ts)
    fig, axes_t = plt.subplots(n_bits_g, 1, figsize=(14, 1.8 * n_bits_g), sharex=True)
    if n_bits_g == 1:
        axes_t = [axes_t]

    for bit in range(n_bits_g):
        ax = axes_t[bit]
        pulse = inp[SEQ_IDX, :, bit].numpy()
        set_mask = pulse > 0.5
        reset_mask = pulse < -0.5
        if set_mask.any():
            ax.scatter(t_arr[set_mask], np.full(set_mask.sum(), 0.65),
                       marker=6, s=18, color="C2", zorder=3,
                       label="set" if bit == 0 else None)
        if reset_mask.any():
            ax.scatter(t_arr[reset_mask], np.full(reset_mask.sum(), 0.35),
                       marker=7, s=18, color="C3", zorder=3,
                       label="reset" if bit == 0 else None)
        ax.step(t_arr, tgt[SEQ_IDX, :, bit].numpy(), where="post",
                color="black", linewidth=1.2, label="target" if bit == 0 else None)
        ax.plot(t_arr, out_prob[SEQ_IDX, :, bit].numpy(),
                color=COLORS[g], linewidth=1.0, alpha=0.9,
                label="output" if bit == 0 else None)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1])
        hold = 1.0 / max(pp_list_g[bit], 1e-8)
        ax.set_ylabel(f"Bit {bit}\n(~{hold:.0f})", fontsize=9)

    axes_t[-1].set_xlabel("Timestep", fontsize=11)
    axes_t[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Reservoir Trajectories — $g = {g}$ (seed {row_data['seed']})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"reservoir_traj_g{g}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% [markdown]
# ## 2D Eigenmode Projections
#
# For a complex conjugate pair $\lambda, \bar\lambda$, the 2D invariant
# subspace is spanned by $\mathrm{Re}(v)$ and $\mathrm{Im}(v)$.
# We project the hidden-state trajectory onto this plane and color
# by the target state of the most-coupled output bit.

# %% Plot 13: Eigenmode projection — hidden activity in mode subspaces
# Each entry: (mode_rank,) for 1D, or (r1, r2) for 2D conjugate pair.
# Append a bit index to override auto-detection: (r1, r2, bit) or (r, None, bit).
PROJECTION_G = 0.6
PROJECTION_MODES = [
    (12, 13, 1),        # pair, auto-detect best-coupled bit
]
NUM_VAL_TRAJS = 1

from matplotlib.colors import Normalize as _MplNormalize
from scipy.stats import pearsonr as _pearsonr


def _time_since_pulse(pulse_signal):
    """Compute per-timestep distance to most recent pulse."""
    T = len(pulse_signal)
    out = np.full(T, T, dtype=float)
    last = -1
    for t in range(T):
        if abs(pulse_signal[t]) > 0.5:
            last = t
        if last >= 0:
            out[t] = t - last
    return out


def _scatter_with_alpha(ax, x, y, color_vals, alpha_arr, cmap="coolwarm",
                        vmin=0, vmax=1, s=8):
    """Scatter with per-point alpha via RGBA array."""
    cm = plt.get_cmap(cmap)
    norm = _MplNormalize(vmin=vmin, vmax=vmax)
    rgba = cm(norm(color_vals))
    rgba[:, 3] = alpha_arr
    sc = ax.scatter(x, y, c=rgba, s=s, zorder=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def _lda_accuracy_and_boundary(X, y):
    """Fisher LDA on 2D points. Returns accuracy, normal vector w, threshold."""
    mask0 = y < 0.5
    mask1 = y > 0.5
    if mask0.sum() == 0 or mask1.sum() == 0:
        return 0.5, np.array([1.0, 0.0]), 0.0
    mu0 = X[mask0].mean(axis=0)
    mu1 = X[mask1].mean(axis=0)
    S0 = np.cov(X[mask0].T) if mask0.sum() > 2 else np.eye(2) * 1e-6
    S1 = np.cov(X[mask1].T) if mask1.sum() > 2 else np.eye(2) * 1e-6
    Sw = S0 + S1
    Sw += np.eye(2) * 1e-8
    w = np.linalg.solve(Sw, mu1 - mu0)
    w /= np.linalg.norm(w) + 1e-15
    proj = X @ w
    thresh = 0.5 * (mu0 @ w + mu1 @ w)
    preds = (proj > thresh).astype(float)
    acc = (preds == y).mean()
    if acc < 0.5:
        w = -w
        thresh = -thresh
        acc = 1.0 - acc
    return acc, w, thresh


if PROJECTION_G in eig_data:
    _proj_data = eig_data[PROJECTION_G]
    _proj_eigs = _proj_data["eigs"]
    _proj_V = _proj_data["eigvecs"]
    _proj_W_out = _proj_data["W_out"]
    _proj_n_bits = _proj_data["n_bits"]
    _proj_pp = _proj_data["p_pulse"]
    _proj_pp_list = _proj_pp if isinstance(_proj_pp, list) else [_proj_pp] * _proj_n_bits
    _proj_N = len(_proj_eigs)

    _proj_abs_rank = np.argsort(np.abs(_proj_eigs))[::-1]

    _proj_row = df[df["gain"] == PROJECTION_G].sort_values("final_val_loss").iloc[0]
    _proj_seed_path = _proj_row["seed_path"]
    _proj_cfg_file = os.path.join(_proj_seed_path, "run_config.yaml")
    with open(_proj_cfg_file) as f:
        _proj_cfg = yaml.safe_load(f)

    _proj_best = glob.glob(
        os.path.join(_proj_seed_path, "checkpoints", "best-model-*.ckpt"))
    if _proj_best:
        _proj_model = RNN(
            input_size=_proj_n_bits,
            hidden_size=_proj_cfg["hidden_size"],
            output_size=_proj_n_bits,
            dt=_proj_cfg["dt"],
            time_constants_config=_proj_cfg.get("time_constants_config"),
            activation=getattr(nn, _proj_cfg["activation"]),
            learn_time_constants=_proj_cfg["learn_time_constants"],
            init_time_constant=_proj_cfg.get("init_time_constant"),
            shared_time_constant=_proj_cfg["shared_time_constant"],
            normalize_hidden=_proj_cfg["normalize_hidden"],
            zero_diag_wrec=_proj_cfg["zero_diag_wrec"],
            recurrent_gain=_proj_cfg["recurrent_gain"],
            noise_std=0.0,
            wrec_init=_proj_cfg["wrec_init"],
            alpha_parameterization=_proj_cfg["alpha_parameterization"],
            dynamics_type=_proj_cfg["dynamics_type"],
        )
        _proj_lit = RNNLightning(
            model=_proj_model,
            learning_rate=_proj_cfg["learning_rate"],
            weight_decay=_proj_cfg["weight_decay"],
            step_size=_proj_cfg.get("lr_step_size", _proj_cfg.get("step_size", 1000)),
            gamma=_proj_cfg["gamma"],
            task="flip_flop",
        )
        _proj_ckpt = torch.load(_proj_best[0], map_location=device, weights_only=False)
        _proj_lit.load_state_dict(_proj_ckpt["state_dict"])
        _proj_lit.eval().to(device)

        _proj_dm = FlipFlopDataModule(
            n_bits=_proj_n_bits,
            p_pulse=_proj_pp,
            pulse_amplitude=_proj_cfg.get("pulse_amplitude", 1.0),
            num_time_steps=_proj_cfg["num_time_steps"],
            num_val_trajectories=NUM_VAL_TRAJS,
            batch_size=NUM_VAL_TRAJS,
        )
        _proj_dm.setup()
        _proj_inp, _, _proj_tgt = _proj_dm.val_dataset.tensors

        with torch.no_grad():
            _proj_hidden, _proj_out = _proj_lit.model(
                _proj_inp.to(device), init_context=None)
            _proj_hidden = _proj_hidden.cpu().numpy()

        _proj_seq = 0
        _proj_h = _proj_hidden[_proj_seq]        # (T, hidden_size)
        _proj_t_arr = np.arange(_proj_h.shape[0])
        T_len = _proj_h.shape[0]

        coupling_full = np.abs(_proj_W_out @ _proj_V)

        for entry in PROJECTION_MODES:
            # Parse entry: (r1,), (r1, r2), (r1, bit), (r1, r2, bit)
            if len(entry) == 1:
                r1 = entry[0]
                r2 = None
                force_bit = None
            elif len(entry) == 2:
                r1, r2 = entry
                force_bit = None
            elif len(entry) == 3:
                r1, r2, force_bit = entry
            else:
                print(f"Skipping invalid entry: {entry}")
                continue

            is_pair = r2 is not None
            idx1 = _proj_abs_rank[r1 - 1]
            lam = _proj_eigs[idx1]
            v = _proj_V[:, idx1]
            r_mag = np.abs(lam)
            log_abs_lam = np.log(np.clip(r_mag, 1e-12, None))
            tau_mode = -1.0 / min(log_abs_lam, -1e-10)

            if is_pair:
                idx2 = _proj_abs_rank[r2 - 1]
                mode_coupling = coupling_full[:, idx1] + coupling_full[:, idx2]
            else:
                idx2 = None
                mode_coupling = coupling_full[:, idx1]

            if force_bit is not None:
                best_bit = force_bit
            else:
                best_bit = int(np.argmax(mode_coupling))

            tgt_bit = _proj_tgt[_proj_seq, :, best_bit].numpy()
            pulse_bit = _proj_inp[_proj_seq, :, best_bit].numpy()
            hold = 1.0 / max(_proj_pp_list[best_bit], 1e-8)

            tsp = _time_since_pulse(pulse_bit)
            alpha_arr = 1.0 - np.clip(tsp / hold, 0, 1) * 0.8
            alpha_arr = np.clip(alpha_arr, 0.1, 1.0)

            set_t = np.where(pulse_bit > 0.5)[0]
            reset_t = np.where(pulse_bit < -0.5)[0]

            # ============================================================
            # 2D conjugate-pair projection
            # ============================================================
            if is_pair:
                theta = np.angle(lam)

                u1 = v.real.copy()
                u2 = v.imag.copy()
                u1 /= np.linalg.norm(u1) + 1e-15
                u2 -= u1 * np.dot(u1, u2)
                u2 /= np.linalg.norm(u2) + 1e-15

                proj1 = _proj_h @ u1
                proj2 = _proj_h @ u2

                # --- LDA separability ---
                X_2d = np.column_stack([proj1, proj2])
                lda_acc, lda_w, lda_thresh = _lda_accuracy_and_boundary(X_2d, tgt_bit)

                # --- Readout boundary ---
                w_out_bit = _proj_W_out[best_bit]
                w_proj = np.array([w_out_bit @ u1, w_out_bit @ u2])
                w_proj_norm = np.linalg.norm(w_proj)

                fig, axes_p = plt.subplots(1, 2, figsize=(14, 5.5))

                # Left: 2D phase portrait with alpha
                ax = axes_p[0]
                sm = _scatter_with_alpha(ax, proj1, proj2, tgt_bit, alpha_arr,
                                        cmap="coolwarm", s=8)

                # Time-flow arrows
                arrow_step = max(1, T_len // 20)
                for t in range(0, T_len - 1, arrow_step):
                    ax.annotate("", xy=(proj1[t+1], proj2[t+1]),
                                xytext=(proj1[t], proj2[t]),
                                arrowprops=dict(arrowstyle="->", color="gray",
                                                lw=0.5, alpha=0.3))

                if len(set_t) > 0:
                    ax.scatter(proj1[set_t], proj2[set_t], marker="^", s=40,
                               color="C2", edgecolors="black", linewidths=0.4,
                               zorder=5, label="set pulse")
                if len(reset_t) > 0:
                    ax.scatter(proj1[reset_t], proj2[reset_t], marker="v", s=40,
                               color="C3", edgecolors="black", linewidths=0.4,
                               zorder=5, label="reset pulse")

                # LDA decision boundary
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
                perp = np.array([-lda_w[1], lda_w[0]])
                mid = lda_w * lda_thresh
                p0 = mid - perp * span
                p1 = mid + perp * span
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", linewidth=1.2,
                        alpha=0.6, label=f"LDA ({lda_acc:.1%})")
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                # Readout boundary
                if w_proj_norm > 1e-8:
                    perp_r = np.array([-w_proj[1], w_proj[0]]) / w_proj_norm
                    p0r = -perp_r * span
                    p1r = perp_r * span
                    ax.plot([p0r[0], p1r[0]], [p0r[1], p1r[1]], "--",
                            color="#9333ea", linewidth=1.2, alpha=0.7,
                            label="readout boundary")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

                plt.colorbar(sm, ax=ax, label=f"Bit {best_bit} target")
                ax.set_xlabel("$\\mathrm{Re}(v)$ projection", fontsize=11)
                ax.set_ylabel("$\\mathrm{Im}(v)$ projection", fontsize=11)
                ax.legend(fontsize=7, loc="upper left")
                ax.set_title(f"Modes {r1},{r2}: $|\\lambda|={r_mag:.4f}$, "
                             f"$\\theta={theta:.3f}$, $\\tau={tau_mode:.0f}$\n"
                             f"LDA acc = {lda_acc:.1%}  |  "
                             f"opacity = recency of pulse",
                             fontsize=9)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.15)

                # Right: projections over time
                ax2 = axes_p[1]
                ax2.plot(_proj_t_arr, proj1, linewidth=0.8, alpha=0.8,
                         label="Re(v) proj", color="#2563eb")
                ax2.plot(_proj_t_arr, proj2, linewidth=0.8, alpha=0.8,
                         label="Im(v) proj", color="#e76f51")

                ax2_twin = ax2.twinx()
                ax2_twin.step(_proj_t_arr, tgt_bit, where="post",
                              color="black", linewidth=1.0, alpha=0.4,
                              label=f"Bit {best_bit} target")
                ax2_twin.set_ylabel(f"Bit {best_bit} target", fontsize=10)
                ax2_twin.set_ylim(-0.2, 1.3)

                ax2.set_xlabel("Timestep", fontsize=11)
                ax2.set_ylabel("Projection amplitude", fontsize=11)
                ax2.legend(fontsize=8, loc="upper left")
                ax2.grid(True, alpha=0.15)
                ax2.set_title("Projections over time", fontsize=10)

                bit_label = f"bit {best_bit}" + ("" if force_bit is None else " (forced)")
                fig.suptitle(
                    f"2D Eigenmode Projection — $g = {PROJECTION_G}$, "
                    f"modes {r1}&{r2}, {bit_label}\n"
                    f"$\\lambda = {lam.real:.4f} \\pm {abs(lam.imag):.4f}i$",
                    fontsize=12, fontweight="bold", y=1.04)
                plt.tight_layout()
                if SAVE_FIGS:
                    fig.savefig(os.path.join(FIGS_DIR,
                                f"reservoir_2d_proj_g{PROJECTION_G}_m{r1}_{r2}_b{best_bit}.pdf"),
                                bbox_inches="tight", dpi=150)
                plt.show()

                print(f"  Modes {r1},{r2} -> bit {best_bit}: "
                      f"LDA accuracy = {lda_acc:.3f}")

            # ============================================================
            # 1D single-mode projection
            # ============================================================
            else:
                u1 = v.real.copy()
                u1 /= np.linalg.norm(u1) + 1e-15
                proj = _proj_h @ u1

                corr, pval = _pearsonr(proj, tgt_bit)
                if corr < 0:
                    proj = -proj
                    u1 = -u1
                    corr = -corr

                # Readout projection onto this direction
                w_out_bit = _proj_W_out[best_bit]
                w_proj_1d = w_out_bit @ u1

                # Class means and threshold
                mask0 = tgt_bit < 0.5
                mask1 = tgt_bit > 0.5
                mu0 = proj[mask0].mean() if mask0.any() else 0.0
                mu1 = proj[mask1].mean() if mask1.any() else 0.0
                thresh_1d = 0.5 * (mu0 + mu1)
                preds_1d = (proj > thresh_1d).astype(float)
                acc_1d = (preds_1d == tgt_bit).mean()

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

                # Left: projection over time
                ax1.plot(_proj_t_arr, proj, linewidth=0.8, color="#2563eb",
                         alpha=0.8, label=f"Mode {r1} proj")
                ax1.axhline(thresh_1d, color="gray", linestyle="--",
                            linewidth=1, alpha=0.5, label=f"threshold = {thresh_1d:.3f}")

                if len(set_t) > 0:
                    ax1.scatter(set_t, proj[set_t], marker="^", s=30,
                                color="C2", edgecolors="black", linewidths=0.3,
                                zorder=5, label="set")
                if len(reset_t) > 0:
                    ax1.scatter(reset_t, proj[reset_t], marker="v", s=30,
                                color="C3", edgecolors="black", linewidths=0.3,
                                zorder=5, label="reset")

                ax1_twin = ax1.twinx()
                ax1_twin.step(_proj_t_arr, tgt_bit, where="post",
                              color="black", linewidth=1.0, alpha=0.3)
                ax1_twin.set_ylabel(f"Bit {best_bit} target", fontsize=10)
                ax1_twin.set_ylim(-0.2, 1.3)

                ax1.set_xlabel("Timestep", fontsize=11)
                ax1.set_ylabel("Projection onto Re(v)", fontsize=11)
                ax1.legend(fontsize=7, loc="upper left")
                ax1.grid(True, alpha=0.15)
                ax1.set_title(f"Projection over time  |  r = {corr:.3f}, "
                              f"p = {pval:.2e}", fontsize=10)

                # Right: histogram split by target state
                bins = np.linspace(proj.min(), proj.max(), 50)
                if mask0.any():
                    ax2.hist(proj[mask0], bins=bins, alpha=0.6, color="#4361ee",
                             label=f"target = 0 (n={mask0.sum()})", density=True)
                if mask1.any():
                    ax2.hist(proj[mask1], bins=bins, alpha=0.6, color="#e63946",
                             label=f"target = 1 (n={mask1.sum()})", density=True)
                ax2.axvline(thresh_1d, color="gray", linestyle="--",
                            linewidth=1.2, label=f"thresh ({acc_1d:.1%} acc)")
                ax2.set_xlabel("Projection value", fontsize=11)
                ax2.set_ylabel("Density", fontsize=11)
                ax2.legend(fontsize=8)
                ax2.set_title("Distribution by target state", fontsize=10)
                ax2.grid(True, alpha=0.15)

                bit_label = f"bit {best_bit}" + ("" if force_bit is None else " (forced)")
                fig.suptitle(
                    f"1D Mode Projection — $g = {PROJECTION_G}$, "
                    f"mode {r1}, {bit_label}\n"
                    f"$\\lambda = {lam.real:.4f}"
                    f"{'+' if lam.imag >= 0 else ''}{lam.imag:.4f}i$"
                    f", $\\tau = {tau_mode:.0f}$"
                    f"  |  Pearson $r = {corr:.3f}$, threshold acc = {acc_1d:.1%}",
                    fontsize=11, fontweight="bold", y=1.04)
                plt.tight_layout()
                if SAVE_FIGS:
                    fig.savefig(os.path.join(FIGS_DIR,
                                f"reservoir_1d_proj_g{PROJECTION_G}_m{r1}_b{best_bit}.pdf"),
                                bbox_inches="tight", dpi=150)
                plt.show()

                print(f"  Mode {r1} -> bit {best_bit}: "
                      f"Pearson r = {corr:.3f} (p={pval:.2e}), "
                      f"threshold acc = {acc_1d:.3f}")

    else:
        print(f"No checkpoint found for g={PROJECTION_G}")
else:
    print(f"PROJECTION_G={PROJECTION_G} not in eig_data; "
          f"available: {sorted(eig_data.keys())}")

# %%
