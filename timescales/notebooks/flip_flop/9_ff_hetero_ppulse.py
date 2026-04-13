# %% [markdown]
# # Flip-Flop with Heterogeneous Pulse Rates
#
# Each bit receives pulses at a different rate (p_pulse per bit),
# creating hold intervals of different lengths. The key question: does the
# trained network develop eigenvalues with **different** effective timescales
# matching the different hold intervals?

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

# %% Specify sweep directory
sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/flip_flop_hetero_ppulse_20260412_081857"

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
        lf = os.path.join(seed_path, "training_losses.json")
        if os.path.exists(lf):
            with open(lf) as f:
                ld = json.load(f)
            val_losses = ld.get("val_losses", ld.get("val_losses_epoch", []))
            val_accs = ld.get("val_accuracies", [])
            steps = ld.get("steps", [])

        if fvl is None and val_losses:
            fvl = val_losses[-1]
        if fvl is None:
            continue

        records.append(dict(
            exp_name=exp_name, gain=gain, seed=seed,
            seed_path=seed_path, final_val_loss=fvl,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
        ))

df = pd.DataFrame(records)
gains = sorted(df["gain"].unique())
print(f"Loaded {len(df)} runs, gains: {gains}")

# %% Plot 1: Loss vs training step
_palette = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#f4a261", "#606c38", "#457b9d"]
COLORS = {g: _palette[i % len(_palette)] for i, g in enumerate(gains)}

fig, ax = plt.subplots(figsize=(8, 4))
for _, row in df.iterrows():
    g = row["gain"]
    vl = row["val_losses"]
    st = row["steps"][:len(vl)] if row["steps"] else list(range(1, len(vl) + 1))
    ax.plot(st, vl, linewidth=1.8, color=COLORS[g], label=f"g = {g}")

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation loss", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
fig.suptitle("Hetero p_pulse Flip-Flop: Training Curves (Identity)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% Plot 1b: Final val loss vs gain
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
fig.suptitle("Hetero p_pulse — Final Val Loss vs Gain (Identity)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% Plot 1c: Accuracy vs training step
fig, ax = plt.subplots(figsize=(8, 4))
for _, row in df.iterrows():
    g = row["gain"]
    va = row["val_accs"]
    st = row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1))
    if not va:
        continue
    ax.plot(st, va, linewidth=1.8, color=COLORS[g], label=f"g = {g}")

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation accuracy", fontsize=12)
ax.set_ylim(0.4, 1.02)
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="lower right")
fig.suptitle("Hetero p_pulse Flip-Flop: Accuracy (Identity)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% Plot 1d: Steps to X% accuracy vs gain
ACC_THRESHOLD = 0.9  # configurable

steps_to_thresh = []
for _, row in df.iterrows():
    va = row["val_accs"]
    st = row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1))
    if not va:
        continue
    reached = None
    for s_i, a_i in zip(st, va):
        if a_i >= ACC_THRESHOLD:
            reached = s_i
            break
    steps_to_thresh.append(dict(gain=row["gain"], steps_to=reached))

if steps_to_thresh:
    df_stt = pd.DataFrame(steps_to_thresh)
    df_stt_valid = df_stt.dropna(subset=["steps_to"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not df_stt_valid.empty:
        means = df_stt_valid.groupby("gain")["steps_to"].mean()
        stds = df_stt_valid.groupby("gain")["steps_to"].std().fillna(0)
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    fmt="o-", capsize=4, linewidth=1.8, markersize=7,
                    color="#2a9d8f", ecolor="#adb5bd")

    never_reached = df_stt[df_stt["steps_to"].isna()].groupby("gain").size()
    for g_nr, count in never_reached.items():
        ax.scatter(g_nr, ax.get_ylim()[1] * 0.95, marker="x", s=80,
                   color="red", zorder=5)
        ax.annotate(f"did not reach", (g_nr, ax.get_ylim()[1] * 0.95),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=8, color="red", va="center")

    ax.set_xlabel("Recurrent gain $g$", fontsize=12)
    ax.set_ylabel(f"Steps to {ACC_THRESHOLD*100:.0f}% accuracy", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Steps to {ACC_THRESHOLD*100:.0f}% Accuracy vs Gain",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

# %% Plot 2: Example trajectories (input pulses, ground truth, network output)
SEQ_IDX = 0

for _, row in df.iterrows():
    g = row["gain"]
    seed_path = row["seed_path"]

    config_file = os.path.join(seed_path, "run_config.yaml")
    with open(config_file) as f:
        run_config = yaml.safe_load(f)

    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        continue

    n_bits = run_config["n_bits"]
    p_pulse = run_config["p_pulse"]

    model = RNN(
        input_size=n_bits,
        hidden_size=run_config["hidden_size"],
        output_size=n_bits,
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
    ckpt = torch.load(best_ckpts[0], map_location=device, weights_only=False)
    lit.load_state_dict(ckpt["state_dict"])
    lit.eval().to(device)

    dm = FlipFlopDataModule(
        n_bits=n_bits,
        p_pulse=p_pulse,
        pulse_amplitude=run_config["pulse_amplitude"],
        num_time_steps=run_config["num_time_steps"],
        num_val_trajectories=10,
        batch_size=10,
    )
    dm.setup()
    inp, _, tgt = dm.val_dataset.tensors

    with torch.no_grad():
        _, out = lit.model(inp.to(device), init_context=None)
        out_prob = torch.sigmoid(out).cpu()

    p_list = p_pulse if isinstance(p_pulse, list) else [p_pulse] * n_bits
    t_arr = np.arange(run_config["num_time_steps"])

    fig, axes = plt.subplots(n_bits, 1, figsize=(12, 2 * n_bits), sharex=True)
    if n_bits == 1:
        axes = [axes]

    for bit in range(n_bits):
        ax = axes[bit]
        pulse = inp[SEQ_IDX, :, bit].numpy()
        set_mask = pulse > 0.5
        reset_mask = pulse < -0.5
        if set_mask.any():
            ax.scatter(t_arr[set_mask], np.full(set_mask.sum(), 0.65),
                       marker=6, s=25, color="C2", zorder=3,
                       label="set (+1)" if bit == 0 else None)
        if reset_mask.any():
            ax.scatter(t_arr[reset_mask], np.full(reset_mask.sum(), 0.35),
                       marker=7, s=25, color="C3", zorder=3,
                       label="reset (-1)" if bit == 0 else None)
        ax.step(t_arr, tgt[SEQ_IDX, :, bit].numpy(), where="post",
                color="black", linewidth=1.5, label="target" if bit == 0 else None)
        ax.plot(t_arr, out_prob[SEQ_IDX, :, bit].numpy(),
                color="steelblue", linewidth=1.2, alpha=0.9,
                label="output" if bit == 0 else None)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1])
        avg_hold = 1.0 / max(p_list[bit], 1e-8)
        ax.set_ylabel(f"Bit {bit}\n(p={p_list[bit]}, ~{avg_hold:.0f}steps)", fontsize=10)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Hetero p_pulse Trajectories — g = {g}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


# %% Plot 3: Jacobian eigenvalue spectrum — untrained vs trained
theta = np.linspace(0, 2 * np.pi, 200)
UNTRAINED_COLOR = "#8da0b5"
TRAINED_COLOR = "#e76f51"

n_gains = len(gains)
fig, axes = plt.subplots(n_gains, 2, figsize=(7.5, 3.2 * n_gains), squeeze=False)

eig_data = {}

for row_idx, g in enumerate(gains):
    row_data = df[df["gain"] == g].iloc[0]
    seed_path = row_data["seed_path"]
    config_file = os.path.join(seed_path, "run_config.yaml")
    with open(config_file) as f:
        run_config = yaml.safe_load(f)

    n_bits = run_config["n_bits"]
    p_pulse_cfg = run_config["p_pulse"]
    dt = run_config["dt"]
    tau = run_config["time_constants_config"]["values"][0]
    alpha = 1.0 - np.exp(-dt / tau)

    ckpt_map = [
        ("Untrained", os.path.join(seed_path, "checkpoints", "untrained.ckpt")),
        ("Trained",   glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))),
    ]
    ckpt_map[1] = ("Trained", ckpt_map[1][1][0] if ckpt_map[1][1] else None)

    for col_idx, (label, ckpt_path) in enumerate(ckpt_map):
        ax = axes[row_idx, col_idx]
        if ckpt_path is None or not os.path.exists(ckpt_path):
            ax.text(0.5, 0.5, "no ckpt", transform=ax.transAxes, ha="center")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        W_rec = None
        for key, val in ckpt["state_dict"].items():
            if "W_rec.weight" in key:
                W_rec = val.numpy()
                break
        if W_rec is None:
            continue

        W_out = None
        for key, val in ckpt["state_dict"].items():
            if "W_out.weight" in key:
                W_out = val.numpy()
                break

        W_in = None
        for key, val in ckpt["state_dict"].items():
            if "W_in.weight" in key:
                W_in = val.numpy()
                break

        N = W_rec.shape[0]
        J = (1.0 - alpha) * np.eye(N) + alpha * g * W_rec
        eigenvalues, eigvecs = np.linalg.eig(J)
        eigs = eigenvalues
        abs_eigs = np.abs(eigs)
        max_abs = np.max(abs_eigs)

        if col_idx == 1:
            eig_data[g] = dict(
                eigs=eigs, eigvecs=eigvecs, W_out=W_out, W_in=W_in,
                alpha=alpha, n_bits=n_bits, p_pulse=p_pulse_cfg,
            )

        ax.plot(np.cos(theta), np.sin(theta), color="#c0392b",
                linewidth=1.2, alpha=0.5, linestyle="--", zorder=1,
                label="$|\\lambda|=1$")
        ax.axhline(0, color="#bbb", linewidth=0.3, zorder=0)
        ax.axvline(0, color="#bbb", linewidth=0.3, zorder=0)

        top_idx = np.argsort(abs_eigs)[-n_bits:]
        rest_idx = np.argsort(abs_eigs)[:-n_bits]
        pt_color = UNTRAINED_COLOR if col_idx == 0 else TRAINED_COLOR

        ax.scatter(eigs.real[rest_idx], eigs.imag[rest_idx], s=18, alpha=0.7,
                   c=pt_color, edgecolors="none", zorder=3)
        if col_idx == 1:
            ax.scatter(eigs.real[top_idx], eigs.imag[top_idx], s=18, alpha=0.95,
                       c=TRAINED_COLOR, edgecolors="black", linewidths=0.4,
                       zorder=4, label=f"Top {n_bits}")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)

        if row_idx == 0:
            ax.set_title(label, fontsize=13, fontweight="bold")
        ax.annotate(f"|$\\lambda$|$_{{max}}$={max_abs:.3f}",
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        if col_idx == 0:
            ax.set_ylabel(f"g = {g}\nIm($\\lambda$)", fontsize=11)
        if row_idx == n_gains - 1:
            ax.set_xlabel("Re($\\lambda$)", fontsize=11)

axes[0, 0].legend(fontsize=8, loc="lower left", framealpha=0.7)
axes[0, 1].legend(fontsize=8, loc="lower left", framealpha=0.7)

fig.suptitle("Jacobian Eigenvalue Spectrum — Hetero p_pulse (Identity)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# %% Plot 3b: Zoomed-in eigenvalue spectrum (near unit circle)
ZOOM_XLIM = (0.9, 1.05)
ZOOM_YLIM = (-0.1, 0.1)

fig, axes = plt.subplots(n_gains, 2, figsize=(7.5, 3.2 * n_gains), squeeze=False)

for row_idx, g in enumerate(gains):
    row_data = df[df["gain"] == g].iloc[0]
    seed_path = row_data["seed_path"]
    config_file = os.path.join(seed_path, "run_config.yaml")
    with open(config_file) as f:
        run_config = yaml.safe_load(f)

    n_bits_g = run_config["n_bits"]
    dt = run_config["dt"]
    tau_val = run_config["time_constants_config"]["values"][0]
    alpha = 1.0 - np.exp(-dt / tau_val)

    ckpt_map = [
        ("Untrained", os.path.join(seed_path, "checkpoints", "untrained.ckpt")),
        ("Trained",   glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))),
    ]
    ckpt_map[1] = ("Trained", ckpt_map[1][1][0] if ckpt_map[1][1] else None)

    for col_idx, (label, ckpt_path) in enumerate(ckpt_map):
        ax = axes[row_idx, col_idx]
        if ckpt_path is None or not os.path.exists(ckpt_path):
            ax.text(0.5, 0.5, "no ckpt", transform=ax.transAxes, ha="center")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        W_rec = None
        for key, val in ckpt["state_dict"].items():
            if "W_rec.weight" in key:
                W_rec = val.numpy()
                break
        if W_rec is None:
            continue

        N = W_rec.shape[0]
        J = (1.0 - alpha) * np.eye(N) + alpha * g * W_rec
        eigs = np.linalg.eigvals(J)
        abs_eigs = np.abs(eigs)

        ax.plot(np.cos(theta), np.sin(theta), color="#c0392b",
                linewidth=1.2, alpha=0.5, linestyle="--", zorder=1)
        ax.axhline(0, color="#bbb", linewidth=0.3, zorder=0)
        ax.axvline(0, color="#bbb", linewidth=0.3, zorder=0)

        top_idx = np.argsort(abs_eigs)[-n_bits_g:]
        rest_idx = np.argsort(abs_eigs)[:-n_bits_g]
        pt_color = UNTRAINED_COLOR if col_idx == 0 else TRAINED_COLOR

        ax.scatter(eigs.real[rest_idx], eigs.imag[rest_idx], s=25, alpha=0.7,
                   c=pt_color, edgecolors="none", zorder=3)
        if col_idx == 1:
            ax.scatter(eigs.real[top_idx], eigs.imag[top_idx], s=30, alpha=0.95,
                       c=TRAINED_COLOR, edgecolors="black", linewidths=0.5,
                       zorder=4, label=f"Top {n_bits_g}")

        ax.set_xlim(*ZOOM_XLIM)
        ax.set_ylim(*ZOOM_YLIM)
        ax.grid(True, alpha=0.15)

        if row_idx == 0:
            ax.set_title(label, fontsize=13, fontweight="bold")
        ax.annotate(f"|$\\lambda$|$_{{max}}$={np.max(abs_eigs):.4f}",
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        if col_idx == 0:
            ax.set_ylabel(f"g = {g}\nIm($\\lambda$)", fontsize=11)
        if row_idx == n_gains - 1:
            ax.set_xlabel("Re($\\lambda$)", fontsize=11)

if n_gains > 0:
    axes[0, 1].legend(fontsize=8, loc="upper left", framealpha=0.7)

fig.suptitle("Jacobian Eigenvalue Spectrum (zoom) — Hetero p_pulse (Identity)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# %% Plot 4: Effective timescale scree plot — tau_eff = -1/ln|lambda|

FIGS_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

_sample = next(iter(eig_data.values()))
_pp = _sample["p_pulse"]
_pp_list = _pp if isinstance(_pp, list) else [_pp]
_holds = [1.0 / p for p in _pp_list]

bit_colors = [f"C{i}" for i in range(len(_holds))]
from matplotlib.lines import Line2D as _Line2D

fig, axes = plt.subplots(1, n_gains, figsize=(5 * n_gains, 4.5),
                         squeeze=False, sharey=True)

for col, g in enumerate(gains):
    ax = axes[0, col]
    if g not in eig_data:
        continue

    data = eig_data[g]
    eigs = data["eigs"]
    V = data["eigvecs"]
    W_out = data["W_out"]
    n_bits_g = data["n_bits"]
    N = len(eigs)

    abs_rank = np.argsort(np.abs(eigs))[::-1]
    abs_sorted = np.abs(eigs[abs_rank])
    log_abs = np.log(np.clip(abs_sorted, 1e-12, None))
    tau_eff = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)
    ranks = np.arange(1, N + 1)

    coupling = np.abs(W_out @ V) if W_out is not None else None
    coupling_ranked = coupling[:, abs_rank] if coupling is not None else None

    # For each bit, find the rank of the mode with strongest coupling
    best_rank_for_bit = {}
    if coupling_ranked is not None:
        for bi in range(n_bits_g):
            best_rank_for_bit[bi] = int(np.argmax(coupling_ranked[bi]))

    highlighted_ranks = set(best_rank_for_bit.values())

    # Plot all modes in gray first
    non_highlighted = [r for r in range(N) if r not in highlighted_ranks]
    ax.scatter(ranks[non_highlighted], tau_eff[non_highlighted],
               s=12, color=UNTRAINED_COLOR, edgecolors="none",
               alpha=0.5, zorder=3)

    # Highlight best-coupling mode for each bit
    for bi, ri in best_rank_for_bit.items():
        c = bit_colors[bi]
        ax.scatter(ranks[ri], tau_eff[ri], s=80, color=c,
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(f"bit {bi}\n$\\tau$={tau_eff[ri]:.0f}",
                    (ranks[ri], tau_eff[ri]),
                    textcoords="offset points", xytext=(12, 0),
                    fontsize=7.5, color=c, fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5, alpha=0.4))

    for bi in range(len(_holds)):
        c = bit_colors[bi]
        ax.axhline(_holds[bi], color=c, linewidth=0.9, linestyle=":",
                   alpha=0.5, zorder=1)
        ax.text(N * 0.92, _holds[bi] * 1.15,
                f"hold≈{_holds[bi]:.0f}", fontsize=6.5,
                color=c, ha="right", alpha=0.7)

    ax.set_xlabel("Eigenvalue rank", fontsize=11)
    if col == 0:
        ax.set_ylabel(r"$\tau_{\mathrm{eff}} = -1\,/\,\ln|\lambda|$  (steps)",
                      fontsize=11)
    ax.set_title(f"$g = {g}$", fontsize=12)
    ax.grid(True, alpha=0.12, which="both")
    ax.set_yscale("log")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

_leg = [_Line2D([0], [0], marker="o", color="w", markerfacecolor=bit_colors[i],
                markeredgecolor="black", markeredgewidth=0.8,
                markersize=8, label=f"Bit {i} (hold≈{_holds[i]:.0f})")
        for i in range(len(_holds))]
_leg.append(_Line2D([0], [0], marker="o", color="w", markerfacecolor=UNTRAINED_COLOR,
                     markersize=5, label="Other modes"))
axes[0, -1].legend(handles=_leg, fontsize=7, loc="center right", framealpha=0.85)

fig.suptitle("Effective Timescales — best-coupling mode per bit highlighted",
             fontsize=14, fontweight="bold")
plt.tight_layout()

if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "tau_eff_scree.pdf"),
                bbox_inches="tight", dpi=150)
    print(f"Saved to {FIGS_DIR}/tau_eff_scree.pdf")
plt.show()

# %% Plot 5: Mode-to-output coupling — |W_out @ V| for top modes
# (W_out @ V)[i, j] tells us how much eigenvector mode j contributes to output bit i.
# We expect each bit to be dominated by a single slow mode whose effective timescale
# matches that bit's hold interval.

for g in gains:
    if g not in eig_data:
        continue
    data = eig_data[g]
    eigs = data["eigs"]
    V = data["eigvecs"]
    W_out = data["W_out"]
    n_bits = data["n_bits"]
    pp = data["p_pulse"]
    pp_list = pp if isinstance(pp, list) else [pp] * n_bits

    if W_out is None:
        print(f"g={g}: no W_out found, skipping")
        continue

    abs_sorted_idx = np.argsort(np.abs(eigs))[::-1]
    top_mode_idx = abs_sorted_idx[:n_bits]

    abs_eigs_sorted = np.abs(eigs[top_mode_idx])
    log_abs = np.log(np.clip(abs_eigs_sorted, 1e-12, None))
    tau_modes = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)

    top_eigs = eigs[top_mode_idx]
    IMAG_THRESH = 1e-8
    is_complex = np.abs(top_eigs.imag) > IMAG_THRESH
    theta_modes = np.abs(np.angle(top_eigs))

    coupling = W_out @ V
    coupling_top = np.abs(coupling[:, top_mode_idx])

    fig, ax = plt.subplots(figsize=(max(n_bits * 1.1, 5), max(n_bits * 0.65, 3)))
    im = ax.imshow(coupling_top, cmap="YlOrRd", aspect="auto")

    ax.set_yticks(range(n_bits))
    ax.set_yticklabels([f"Bit {i} (p={pp_list[i]})" for i in range(n_bits)], fontsize=9)
    ax.set_xticks(range(n_bits))
    mode_labels = []
    for mi in range(n_bits):
        lbl = f"Mode {mi+1}\n$\\tau$={tau_modes[mi]:.0f}"
        if is_complex[mi]:
            lbl += f"\n$\\theta$={theta_modes[mi]:.2f} (C)"
        else:
            sign = "+" if top_eigs[mi].real >= 0 else "−"
            lbl += f"\n({sign}real)"
        mode_labels.append(lbl)
    ax.set_xticklabels(mode_labels, fontsize=7)
    ax.set_xlabel("Eigenmode (ranked by $|\\lambda|$) — C = complex pair", fontsize=10)
    ax.set_ylabel("Output bit", fontsize=11)

    for i in range(n_bits):
        for j in range(n_bits):
            val = coupling_top[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val > 0.5 * coupling_top.max() else "black")

    plt.colorbar(im, ax=ax, label="$|W_{\\mathrm{out}} V|$", shrink=0.8)
    fig.suptitle(f"Mode-to-Output Coupling — g = {g}\n"
                 f"Which slow mode drives which output bit?",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"mode_output_coupling_g{g}.pdf"),
                    bbox_inches="tight", dpi=150)
        print(f"Saved to {FIGS_DIR}/mode_output_coupling_g{g}.pdf")
    plt.show()

    # Identify dominant mode per bit
    dominant = np.argmax(coupling_top, axis=1)
    print(f"\ng = {g}: Dominant mode per output bit:")
    for bit_i in range(n_bits):
        mode_j = dominant[bit_i]
        print(f"  Bit {bit_i} (p_pulse={pp_list[bit_i]}, hold~{1.0/pp_list[bit_i]:.0f} steps)"
              f"  <--  Mode {mode_j+1} (tau_eff={tau_modes[mode_j]:.1f} steps)")

    # --- Extended: coupling profile across ALL modes ---
    coupling_all = np.abs(W_out @ V)   # (n_bits, N)
    abs_all = np.abs(eigs)
    abs_rank_order = np.argsort(abs_all)[::-1]
    coupling_ranked = coupling_all[:, abs_rank_order]   # re-order columns by |λ|

    fig, axes_ext = plt.subplots(n_bits, 1, figsize=(12, 2.2 * n_bits), sharex=True)
    if n_bits == 1:
        axes_ext = [axes_ext]
    for bit_i, ax_e in enumerate(axes_ext):
        ax_e.plot(np.arange(1, N + 1), coupling_ranked[bit_i], linewidth=0.8,
                  color=TRAINED_COLOR, alpha=0.7)
        ax_e.axvspan(1, n_bits + 0.5, alpha=0.08, color="C0", label="slow modes")
        ax_e.set_ylabel(f"Bit {bit_i}\n(p={pp_list[bit_i]})", fontsize=9)
        #ax_e.set_yscale("log")
        ax_e.grid(True, alpha=0.1)
        ax_e.spines["top"].set_visible(False)
        ax_e.spines["right"].set_visible(False)
        if bit_i == 0:
            ax_e.legend(fontsize=8, loc="upper right")
    axes_ext[-1].set_xlabel("Eigenmode rank (by $|\\lambda|$)", fontsize=11)
    fig.suptitle(r"$|W_{\mathrm{out}}\, V|$ across all modes" + f" — g = {g}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"output_coupling_all_modes_g{g}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% Plot 5b: W_in projection onto all eigenmodes (input-to-mode coupling)
# V^{-1} W_in tells us which modes are driven by the input.
# We use left eigenvectors (rows of V^{-1} = pinv(V)) for non-symmetric J.

for g in gains:
    if g not in eig_data:
        continue
    data = eig_data[g]
    eigs = data["eigs"]
    V = data["eigvecs"]
    W_in = data["W_in"]
    n_bits = data["n_bits"]
    pp = data["p_pulse"]
    pp_list = pp if isinstance(pp, list) else [pp] * n_bits

    if W_in is None:
        print(f"g={g}: no W_in found, skipping")
        continue

    N = V.shape[0]
    abs_rank_order = np.argsort(np.abs(eigs))[::-1]
    tau_all = -1.0 / np.where(
        np.log(np.clip(np.abs(eigs), 1e-12, None)) < -1e-10,
        np.log(np.clip(np.abs(eigs), 1e-12, None)),
        -1e-10,
    )
    tau_ranked = tau_all[abs_rank_order]

    # Left eigenvectors: rows of V^{-1}
    V_inv = np.linalg.pinv(V)                    # (N, N)
    input_coupling = np.abs(V_inv @ W_in)        # (N, n_bits) — mode × input_bit
    input_coupling_ranked = input_coupling[abs_rank_order]

    fig, axes_in = plt.subplots(n_bits, 1, figsize=(12, 2.2 * n_bits), sharex=True)
    if n_bits == 1:
        axes_in = [axes_in]
    for bit_i, ax_i in enumerate(axes_in):
        ax_i.plot(np.arange(1, N + 1), input_coupling_ranked[:, bit_i],
                  linewidth=0.8, color="#2a9d8f", alpha=0.8)
        ax_i.axvspan(1, n_bits + 0.5, alpha=0.08, color="C0", label="slow modes")
        ax_i.set_ylabel(f"Bit {bit_i}\n(p={pp_list[bit_i]})", fontsize=9)
        ax_i.set_yscale("log")
        ax_i.grid(True, alpha=0.1)
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        if bit_i == 0:
            ax_i.legend(fontsize=8, loc="upper right")
    axes_in[-1].set_xlabel("Eigenmode rank (by $|\\lambda|$)", fontsize=11)
    fig.suptitle(f"$|V^{{-1}} W_{{\\mathrm{{in}}}}|$ — Input-to-mode coupling, g = {g}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"input_coupling_all_modes_g{g}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

    # Heatmap: top n_bits slow modes vs all input bits
    coupling_top_in = input_coupling_ranked[:n_bits, :]   # (n_bits, n_bits)
    fig2, ax2 = plt.subplots(figsize=(max(n_bits * 0.9, 4), max(n_bits * 0.6, 3)))
    im2 = ax2.imshow(coupling_top_in.T, cmap="Blues", aspect="auto")
    ax2.set_yticks(range(n_bits))
    ax2.set_yticklabels([f"Input bit {i} (p={pp_list[i]})" for i in range(n_bits)],
                        fontsize=9)
    ax2.set_xticks(range(n_bits))
    ax2.set_xticklabels([f"Mode {i+1}" for i in range(n_bits)], fontsize=9)
    ax2.set_xlabel("Slow eigenmode (rank)", fontsize=11)
    ax2.set_ylabel("Input bit", fontsize=11)
    for i in range(n_bits):
        for j in range(n_bits):
            val = coupling_top_in[j, i]
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if val > 0.5 * coupling_top_in.max() else "black")
    plt.colorbar(im2, ax=ax2, label="$|V^{-1} W_{\\mathrm{in}}|$", shrink=0.8)
    fig2.suptitle(f"Input-to-slow-mode coupling — g = {g}",
                  fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig2.savefig(os.path.join(FIGS_DIR, f"input_coupling_heatmap_g{g}.pdf"),
                     bbox_inches="tight", dpi=150)
    plt.show()


# %% Plot 5c: PCA dimensionality of latent dynamics
# Collect hidden states from many trajectories, run PCA, and plot
# cumulative variance explained vs number of components.

PCA_GAIN = gains[0]   # which network to analyze (pick one gain)
PCA_N_TRAJ = 100      # number of trajectories to collect

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
    ax1.annotate(f"{thresh*100:.0f}% → {n_needed} PCs",
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

_pp_pca = _pca_cfg["p_pulse"]
_pp_pca = _pp_pca if isinstance(_pp_pca, list) else [_pp_pca]
fig.suptitle(f"PCA Dimensionality of Latent Dynamics — g={PCA_GAIN}, "
             f"{PCA_N_TRAJ} trajectories, "
             f"N={_pca_H}, n_bits={_pca_cfg['n_bits']}",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"pca_dimensionality_g{PCA_GAIN}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

print(f"\nPCA summary for g={PCA_GAIN}:")
print(f"  Data matrix: {_pca_data.shape[0]} points × {_pca_data.shape[1]} dims")
for thresh in [0.80, 0.90, 0.95, 0.99]:
    n_needed = int(np.searchsorted(_pca_cum, thresh)) + 1
    print(f"  {thresh*100:.0f}% variance → {n_needed} PCs")


# %% Plot 6: Eigenvector orthogonality — slow (top |λ|) vs bulk modes
# J is non-symmetric, so right eigenvectors need not be orthogonal.
# We show |V^H V| for the top-n_bits modes, the next n_bits modes (bulk), and cross overlap.


def _unit_norm_columns(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return M / norms


def _offdiag_abs(G: np.ndarray) -> np.ndarray:
    n = G.shape[0]
    m = np.abs(G)
    return m[~np.eye(n, dtype=bool)]


for g in gains:
    if g not in eig_data:
        continue
    data = eig_data[g]
    eigs = data["eigs"]
    V_full = data["eigvecs"]
    n_bits = data["n_bits"]
    N = V_full.shape[0]

    abs_sorted_idx = np.argsort(np.abs(eigs))[::-1]
    top_idx = abs_sorted_idx[:n_bits]
    all_bulk_idx = abs_sorted_idx[n_bits:]

    # How many bulk modes to show in the heatmap (keep manageable)
    N_BULK_HEATMAP = min(20, len(all_bulk_idx))
    bulk_heatmap_idx = all_bulk_idx[:N_BULK_HEATMAP]

    if len(all_bulk_idx) == 0:
        print(f"g={g}: not enough modes for bulk comparison, skipping orthogonality plot")
        continue

    Vs = _unit_norm_columns(V_full[:, top_idx])
    Vb_all = _unit_norm_columns(V_full[:, all_bulk_idx])
    Vb_hm = _unit_norm_columns(V_full[:, bulk_heatmap_idx])

    G_slow = Vs.conj().T @ Vs
    G_bulk_hm = Vb_hm.conj().T @ Vb_hm
    G_cross_hm = Vs.conj().T @ Vb_hm

    abs_slow = np.abs(G_slow)
    abs_bulk = np.abs(G_bulk_hm)
    abs_cross = np.abs(G_cross_hm)

    od_slow = _offdiag_abs(G_slow)
    od_bulk = _offdiag_abs(G_bulk_hm)
    od_cross_all = np.abs(Vs.conj().T @ Vb_all).ravel()

    print(f"\ng = {g} — eigenvector overlap (|inner product|)")
    print(f"  slow–slow off-diag:      mean={od_slow.mean():.4f}, max={od_slow.max():.4f}  (n_modes={n_bits})")
    print(f"  bulk–bulk off-diag:      mean={od_bulk.mean():.4f}, max={od_bulk.max():.4f}  (heatmap, n_modes={N_BULK_HEATMAP})")
    print(f"  slow–bulk (all {len(all_bulk_idx)} bulk): mean={od_cross_all.mean():.4f}, max={od_cross_all.max():.4f}")

    fig = plt.figure(figsize=(13, 5.2), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.35, wspace=0.35)

    vmax = max(abs_slow.max(), abs_bulk.max(), abs_cross.max(), 1e-8)
    n_bulk_hm = N_BULK_HEATMAP

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(abs_slow, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
    ax0.set_xticks(range(n_bits))
    ax0.set_yticks(range(n_bits))
    ax0.set_xticklabels([f"{i+1}" for i in range(n_bits)], fontsize=9)
    ax0.set_yticklabels([f"{i+1}" for i in range(n_bits)], fontsize=9)
    ax0.set_xlabel("slow mode $j$", fontsize=10)
    ax0.set_ylabel("slow mode $i$", fontsize=10)
    ax0.set_title("$|v_i^H v_j|$ — top $|\\lambda|$ modes", fontsize=11)

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(abs_bulk, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax1.set_xticks(range(n_bulk_hm))
    ax1.set_yticks(range(n_bulk_hm))
    ax1.set_xticklabels([f"{n_bits+i+1}" for i in range(n_bulk_hm)], fontsize=7,
                        rotation=90)
    ax1.set_yticklabels([f"{n_bits+i+1}" for i in range(n_bulk_hm)], fontsize=7)
    ax1.set_xlabel("bulk mode rank $j$", fontsize=10)
    ax1.set_ylabel("bulk mode rank $i$", fontsize=10)
    ax1.set_title(f"$|v_i^H v_j|$ — top {n_bulk_hm} bulk modes\n"
                  f"(ranks {n_bits+1}…{n_bits+n_bulk_hm})", fontsize=10)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(abs_cross, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
    ax2.set_xticks(range(n_bulk_hm))
    ax2.set_yticks(range(n_bits))
    ax2.set_xticklabels([f"{n_bits+j+1}" for j in range(n_bulk_hm)], fontsize=7,
                        rotation=90)
    ax2.set_yticklabels([f"s{i+1}" for i in range(n_bits)], fontsize=8)
    ax2.set_xlabel("bulk mode rank", fontsize=10)
    ax2.set_ylabel("slow mode", fontsize=10)
    ax2.set_title(f"slow $\\times$ bulk: $|v_s^H v_b|$\n"
                  f"(top {n_bulk_hm} bulk modes)", fontsize=10)

    cbar = fig.colorbar(im2, ax=[ax0, ax1, ax2], shrink=0.85, aspect=28, pad=0.02)
    cbar.set_label(r"$|\langle v_i, v_j \rangle|$", fontsize=10)

    od_bulk_all = np.abs((Vb_all.conj().T @ Vb_all)[~np.eye(len(all_bulk_idx), dtype=bool)])

    axh = fig.add_subplot(gs[1, :])
    bins = np.linspace(0, min(1.0, max(od_slow.max(), od_bulk_all.max(), 0.05) * 1.1), 50)
    axh.hist(od_slow, bins=bins, alpha=0.75, label=f"slow–slow off-diag ({n_bits} modes)",
             color=TRAINED_COLOR, density=True)
    axh.hist(od_bulk_all, bins=bins, alpha=0.45,
             label=f"bulk–bulk off-diag (all {len(all_bulk_idx)} bulk modes)",
             color=UNTRAINED_COLOR, density=True)
    axh.axvline(od_cross_all.mean(), color="#6a4c93", linestyle="--", linewidth=1.2,
                label=f"slow–bulk mean = {od_cross_all.mean():.3f} ({len(all_bulk_idx)} bulk modes)")
    axh.set_xlabel(r"$|\langle v_i, v_j \rangle|$ (off-diagonal or cross pairs)", fontsize=11)
    axh.set_ylabel("density", fontsize=11)
    axh.legend(fontsize=9, loc="upper right")
    axh.set_title("Pairwise overlap magnitudes", fontsize=11)
    axh.spines["top"].set_visible(False)
    axh.spines["right"].set_visible(False)

    fig.suptitle(f"Eigenmode orthogonality — trained Jacobian, $g = {g}$",
                 fontsize=13, fontweight="bold")
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"eigenmode_orthogonality_g{g}.pdf"),
                    bbox_inches="tight", dpi=150)
        print(f"Saved to {FIGS_DIR}/eigenmode_orthogonality_g{g}.pdf")
    plt.show()


# %% Summary: print top-N effective timescales vs expected hold intervals
_sample = next(iter(eig_data.values()))
_n_bits = _sample["n_bits"]
_pp = _sample["p_pulse"]
_pp_list = _pp if isinstance(_pp, list) else [_pp] * _n_bits
_holds = [f"{1.0/p:.0f}" for p in _pp_list]

print("\n" + "=" * 60)
print(f"Top-{_n_bits} effective timescales vs expected hold intervals")
print(f"p_pulse = {_pp_list} -> expected holds: {_holds} steps")
print("=" * 60)

for g in gains:
    if g not in eig_data:
        continue
    data = eig_data[g]
    n_bits = data["n_bits"]
    eigs = data["eigs"]
    abs_sorted = np.sort(np.abs(eigs))[::-1]
    log_abs = np.log(np.clip(abs_sorted[:n_bits], 1e-12, None))
    tau_top = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)
    print(f"\n  g = {g}:")
    for i, tau in enumerate(tau_top):
        print(f"    mode {i+1}: tau_eff = {tau:.1f} steps")


# %% Plot 7: Condition number of eigenvector matrix — how (non-)diagonalisable is J?
# cond(V) = sigma_max(V) / sigma_min(V).  Large => ill-conditioned change of basis,
# meaning the "mode" picture is fragile (eigenvectors nearly linearly dependent).
# We compare across all gains and against an identity-basis reference.

print("\n" + "=" * 55)
print("Condition number of eigenvector matrix V")
print("=" * 55)

cond_results = {}
for g in gains:
    if g not in eig_data:
        continue
    V = eig_data[g]["eigvecs"]
    svd_vals = np.linalg.svd(V, compute_uv=False)
    cond = svd_vals[0] / max(svd_vals[-1], 1e-12)
    cond_results[g] = dict(cond=cond, svd=svd_vals)
    print(f"  g = {g}:  cond(V) = {cond:.2f}  "
          f"(σ_max={svd_vals[0]:.4f}, σ_min={svd_vals[-1]:.4e})")

fig, axes_c = plt.subplots(1, len(cond_results), figsize=(5 * len(cond_results), 3.8),
                            squeeze=False)

for col, (g, res) in enumerate(cond_results.items()):
    ax_c = axes_c[0, col]
    svd = res["svd"]
    ranks_sv = np.arange(1, len(svd) + 1)
    ax_c.semilogy(ranks_sv, svd, linewidth=1.2, color=TRAINED_COLOR)
    ax_c.axhline(svd[0] / np.sqrt(len(svd)), color="#aaa", linestyle=":",
                 linewidth=1, label="$\\sigma_{\\max}/\\sqrt{N}$ (Haar ref.)")
    ax_c.set_xlabel("Singular value rank", fontsize=11)
    if col == 0:
        ax_c.set_ylabel("Singular value of $V$", fontsize=11)
    ax_c.set_title(f"$g = {g}$\ncond($V$) = {res['cond']:.1f}", fontsize=11)
    ax_c.grid(True, alpha=0.12)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.legend(fontsize=8)

fig.suptitle("Singular value spectrum of eigenvector matrix $V$",
             fontsize=13, fontweight="bold")
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "eigvec_condition_number.pdf"),
                bbox_inches="tight", dpi=150)
    print(f"Saved to {FIGS_DIR}/eigvec_condition_number.pdf")
plt.show()
