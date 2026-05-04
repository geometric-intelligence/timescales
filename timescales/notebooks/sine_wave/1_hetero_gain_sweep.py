# %% [markdown]
# # Sine-Wave Generation — Heterogeneous Periods, Gain Sweep
#
# Multi-frequency autonomous oscillation with **linear** (Identity) activation.
# The network generates $N$ (cos, sin) pairs at distinct periods
# $T_1, T_2, \ldots, T_N$ from a uniform initial state $r_0 = 1$,
# with no external input.
#
# We sweep recurrent gain $g$ from sub-critical to super-critical.
#
# **Key analyses:**
# 1. Training curves (MSE loss / $R^2$) and per-pair breakdown
# 2. Sample output waveforms vs targets
# 3. Global Jacobian eigendecomposition (linear system — state-independent)

# %%
import os
import sys
import subprocess
import json
import glob

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
from datamodules.sine_wave import SineWaveDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_FIGS = True
FIGS_DIR = os.path.join("notebooks", "figs", "sine_wave_hetero")
os.makedirs(FIGS_DIR, exist_ok=True)

# %% Specify sweep directory
sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/sine_wave_hetero_gain_sweep_20260414_014800"

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
        val_losses_per_ch, val_accs_per_ch = {}, {}
        lf = os.path.join(seed_path, "training_losses.json")
        if os.path.exists(lf):
            with open(lf) as f:
                ld = json.load(f)
            val_losses = ld.get("val_losses", [])
            val_accs = ld.get("val_accuracies", [])
            steps = ld.get("steps", [])
            val_losses_per_ch = ld.get("val_losses_per_bit", {})
            val_accs_per_ch = ld.get("val_accuracies_per_bit", {})

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
            val_losses_per_ch=val_losses_per_ch,
            val_accs_per_ch=val_accs_per_ch,
            run_config=run_config,
        ))

df = pd.DataFrame(records)
gains = sorted(df["gain"].unique())
print(f"Loaded {len(df)} runs, gains: {gains}")

# %% Extract task parameters from first config
first_config = df.iloc[0]["run_config"] if len(df) > 0 else {}
n_pairs = first_config.get("n_pairs", 1)
periods_cfg = first_config.get("periods", first_config.get("period", 10.0))
if isinstance(periods_cfg, (int, float)):
    periods_list = [float(periods_cfg)] * n_pairs
else:
    periods_list = [float(p) for p in periods_cfg]
n_channels = 2 * n_pairs
init_hidden_value = first_config.get("init_hidden_value", 1.0)
dt = first_config.get("dt", 1.0)

channel_labels = []
for k in range(n_pairs):
    channel_labels.append(f"cos (T={periods_list[k]:.0f})")
    channel_labels.append(f"sin (T={periods_list[k]:.0f})")

print(f"n_pairs={n_pairs}, periods={periods_list}, dt={dt}, "
      f"init_hidden={init_hidden_value}")

# %% Color palette
palette = plt.cm.viridis(np.linspace(0.15, 0.95, len(gains)))
COLORS = {g: palette[i] for i, g in enumerate(gains)}

# %% Plot 1: Validation loss vs training step
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
ax.set_ylabel("Validation loss (MSE)", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2)
fig.suptitle("Sine-Wave RNN — Validation Loss by Gain",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "sw_val_loss.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 2: Validation R² vs training step
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
ax.set_ylabel("Validation $R^2$", fontsize=12)
ax.set_ylim(-0.1, 1.05)
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2, loc="lower right")
fig.suptitle("Sine-Wave RNN — Validation $R^2$ by Gain",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "sw_val_r2.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 3: Final val loss vs gain
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
ax.set_ylabel("Final validation loss (MSE)", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
           label="$g = 1$")
ax.legend(fontsize=10)
fig.suptitle("Sine-Wave RNN — Final Loss vs Gain",
             fontsize=13, fontweight="bold", y=1.04)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "sw_final_loss_vs_gain.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 4: Per-channel R² curves (one subplot per gain)
ch_colors = [f"C{i}" for i in range(n_channels)]

ncols_pb = min(4, len(gains))
nrows_pb = (len(gains) + ncols_pb - 1) // ncols_pb
fig, axes = plt.subplots(nrows_pb, ncols_pb,
                         figsize=(5 * ncols_pb, 3.5 * nrows_pb),
                         squeeze=False, sharey=True)

for gi, g in enumerate(gains):
    ax = axes[gi // ncols_pb, gi % ncols_pb]
    sub = df[df["gain"] == g]
    for _, row in sub.iterrows():
        apb = row["val_accs_per_ch"]
        if not apb:
            continue
        st = row["steps"]
        for ch_key, vals in apb.items():
            if not vals:
                continue
            ch_idx = int(ch_key.replace("channel_", ""))
            x = st[:len(vals)] if st else list(range(1, len(vals) + 1))
            ax.plot(x, vals, linewidth=1.2, color=ch_colors[ch_idx],
                    alpha=0.6)
    ax.set_title(f"g = {g}", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.5, 1.05)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.grid(True, alpha=0.2)
    if gi // ncols_pb == nrows_pb - 1:
        ax.set_xlabel("Step", fontsize=10)
    if gi % ncols_pb == 0:
        ax.set_ylabel("Per-channel $R^2$", fontsize=10)

for i in range(len(gains), nrows_pb * ncols_pb):
    axes[i // ncols_pb, i % ncols_pb].set_visible(False)

legend_handles = [Line2D([0], [0], color=ch_colors[i], linewidth=2,
                         label=channel_labels[i])
                  for i in range(n_channels)]
axes[0, -1].legend(handles=legend_handles, fontsize=7, loc="lower right")
fig.suptitle("Sine-Wave RNN — Per-Channel $R^2$ During Training",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "sw_per_channel_r2_curves.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 5: Sample waveforms — model output vs target for best seed per gain
fig, axes = plt.subplots(len(gains), 1,
                         figsize=(12, 3.0 * len(gains)),
                         squeeze=False, sharex=True)

for gi, g in enumerate(gains):
    ax = axes[gi, 0]
    sub = df[df["gain"] == g]
    row = sub.loc[sub["final_val_loss"].idxmin()]
    seed_path = row["seed_path"]
    rc = row["run_config"]

    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        ax.text(0.5, 0.5, "no checkpoint", ha="center", va="center",
                transform=ax.transAxes)
        continue

    ckpt = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    n_p = rc.get("n_pairs", 1)
    model = RNN(
        input_size=rc.get("input_size", 1),
        hidden_size=rc["hidden_size"],
        output_size=2 * n_p,
        dt=rc["dt"],
        time_constants_config=rc.get("time_constants_config"),
        activation=getattr(nn, rc["activation"]),
        learn_time_constants=rc["learn_time_constants"],
        init_time_constant=rc.get("init_time_constant"),
        shared_time_constant=rc["shared_time_constant"],
        normalize_hidden=rc["normalize_hidden"],
        zero_diag_wrec=rc["zero_diag_wrec"],
        recurrent_gain=rc["recurrent_gain"],
        noise_std=0.0,
        wrec_init=rc["wrec_init"],
        alpha_parameterization=rc["alpha_parameterization"],
        dynamics_type=rc["dynamics_type"],
    )
    lit = RNNLightning(
        model=model,
        learning_rate=rc["learning_rate"],
        weight_decay=rc["weight_decay"],
        step_size=rc.get("lr_step_size", rc.get("step_size", 1000)),
        gamma=rc["gamma"],
        task="sine_wave",
        init_hidden_value=rc.get("init_hidden_value", 1.0),
    )
    lit.load_state_dict(sd)
    lit.eval()

    T_steps = rc["num_time_steps"]
    dummy_inp = torch.zeros(1, T_steps, 1)
    h_init = torch.full((1, rc["hidden_size"]),
                        rc.get("init_hidden_value", 1.0))
    with torch.no_grad():
        _, out = model(dummy_inp, init_hidden=h_init)
    out_np = out[0].numpy()

    t_ax = np.arange(T_steps) * rc["dt"]
    for k in range(n_p):
        ci = 2 * k
        period_k = periods_list[k]
        phase_k = 2.0 * np.pi * t_ax / period_k
        ax.plot(t_ax, np.cos(phase_k), "--", color=ch_colors[ci],
                linewidth=0.8, alpha=0.5)
        ax.plot(t_ax, out_np[:, ci], color=ch_colors[ci], linewidth=1.2,
                label=f"cos T={period_k:.0f}" if gi == 0 else None)
        ax.plot(t_ax, np.sin(phase_k), "--", color=ch_colors[ci + 1],
                linewidth=0.8, alpha=0.5)
        ax.plot(t_ax, out_np[:, ci + 1], color=ch_colors[ci + 1], linewidth=1.2,
                label=f"sin T={period_k:.0f}" if gi == 0 else None)

    ax.set_ylabel(f"g={g}", fontsize=11, fontweight="bold")
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.2)
    if gi == 0:
        ax.legend(fontsize=7, ncol=n_p, loc="upper right")

axes[-1, 0].set_xlabel("Time", fontsize=12)
fig.suptitle("Sine-Wave RNN — Output vs Target (dashed) by Gain",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "sw_waveforms.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## Generalization Test — Extended Rollout
#
# The training horizon is $T = 500$ steps.  If the network learned genuine
# oscillatory modes (eigenvalues at the correct frequencies on the unit
# circle), the output should remain accurate indefinitely.  If it merely
# memorised the finite training sequence via interference of many modes,
# the output should degrade past $t = 500$.

# %% Plot 6: Extended rollout — output vs target well past training horizon
ROLLOUT_STEPS = 2500  # 5× training length
R2_WINDOW = 100       # window size for windowed R²

fig, axes = plt.subplots(len(gains), 1,
                         figsize=(14, 3.0 * len(gains)),
                         squeeze=False, sharex=True)

r2_by_gain = {}

for gi, g in enumerate(gains):
    ax = axes[gi, 0]
    sub = df[df["gain"] == g]
    row = sub.loc[sub["final_val_loss"].idxmin()]
    seed_path = row["seed_path"]
    rc = row["run_config"]
    train_T = rc["num_time_steps"]

    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        ax.text(0.5, 0.5, "no checkpoint", ha="center", va="center",
                transform=ax.transAxes)
        continue

    ckpt = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    n_p = rc.get("n_pairs", 1)
    model = RNN(
        input_size=rc.get("input_size", 1),
        hidden_size=rc["hidden_size"],
        output_size=2 * n_p,
        dt=rc["dt"],
        time_constants_config=rc.get("time_constants_config"),
        activation=getattr(nn, rc["activation"]),
        learn_time_constants=rc["learn_time_constants"],
        init_time_constant=rc.get("init_time_constant"),
        shared_time_constant=rc["shared_time_constant"],
        normalize_hidden=rc["normalize_hidden"],
        zero_diag_wrec=rc["zero_diag_wrec"],
        recurrent_gain=rc["recurrent_gain"],
        noise_std=0.0,
        wrec_init=rc["wrec_init"],
        alpha_parameterization=rc["alpha_parameterization"],
        dynamics_type=rc["dynamics_type"],
    )
    lit = RNNLightning(
        model=model,
        learning_rate=rc["learning_rate"],
        weight_decay=rc["weight_decay"],
        step_size=rc.get("lr_step_size", rc.get("step_size", 1000)),
        gamma=rc["gamma"],
        task="sine_wave",
        init_hidden_value=rc.get("init_hidden_value", 1.0),
    )
    lit.load_state_dict(sd)
    lit.eval()

    dummy_inp = torch.zeros(1, ROLLOUT_STEPS, 1)
    h_init = torch.full((1, rc["hidden_size"]),
                        rc.get("init_hidden_value", 1.0))
    with torch.no_grad():
        _, out = model(dummy_inp, init_hidden=h_init)
    out_np = out[0].numpy()

    t_ax = np.arange(ROLLOUT_STEPS) * rc["dt"]
    for k in range(n_p):
        ci = 2 * k
        period_k = periods_list[k]
        phase_k = 2.0 * np.pi * t_ax / period_k
        ax.plot(t_ax, np.cos(phase_k), "--", color=ch_colors[ci],
                linewidth=0.6, alpha=0.4)
        ax.plot(t_ax, out_np[:, ci], color=ch_colors[ci], linewidth=1.0,
                label=f"cos T={period_k:.0f}" if gi == 0 else None)
        ax.plot(t_ax, np.sin(phase_k), "--", color=ch_colors[ci + 1],
                linewidth=0.6, alpha=0.4)
        ax.plot(t_ax, out_np[:, ci + 1], color=ch_colors[ci + 1], linewidth=1.0,
                label=f"sin T={period_k:.0f}" if gi == 0 else None)

    ax.axvline(train_T * rc["dt"], color="black", linestyle=":",
               linewidth=1.5, alpha=0.7, label="training horizon" if gi == 0 else None)
    ax.set_ylabel(f"g={g}", fontsize=11, fontweight="bold")
    ax.set_ylim(-2.0, 2.0)
    ax.grid(True, alpha=0.2)
    if gi == 0:
        ax.legend(fontsize=7, ncol=n_p + 1, loc="upper right")

    # Windowed R² per channel
    targets_ext = np.zeros((ROLLOUT_STEPS, 2 * n_p))
    for k in range(n_p):
        ci = 2 * k
        period_k = periods_list[k]
        phase_k = 2.0 * np.pi * t_ax / period_k
        targets_ext[:, ci] = np.cos(phase_k)
        targets_ext[:, ci + 1] = np.sin(phase_k)

    n_windows = ROLLOUT_STEPS // R2_WINDOW
    win_centers = []
    win_r2_total = []
    win_r2_per_ch = [[] for _ in range(2 * n_p)]
    for wi in range(n_windows):
        s = wi * R2_WINDOW
        e = s + R2_WINDOW
        tgt_w = targets_ext[s:e]
        out_w = out_np[s:e]
        ss_res = np.sum((tgt_w - out_w) ** 2)
        ss_tot = np.sum((tgt_w - tgt_w.mean(axis=0)) ** 2)
        r2_total = 1.0 - ss_res / max(ss_tot, 1e-12)
        win_centers.append((s + e) / 2 * rc["dt"])
        win_r2_total.append(r2_total)
        for ch in range(2 * n_p):
            ss_r = np.sum((tgt_w[:, ch] - out_w[:, ch]) ** 2)
            ss_t = np.sum((tgt_w[:, ch] - tgt_w[:, ch].mean()) ** 2)
            win_r2_per_ch[ch].append(1.0 - ss_r / max(ss_t, 1e-12))

    r2_by_gain[g] = dict(centers=win_centers, total=win_r2_total,
                         per_ch=win_r2_per_ch, train_T=train_T * rc["dt"])

axes[-1, 0].set_xlabel("Time", fontsize=12)
fig.suptitle("Extended Rollout — Output vs Target (dashed)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "sw_extended_rollout.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 7: Windowed R² across the rollout
ncols_r2 = min(4, len(gains))
nrows_r2 = (len(gains) + ncols_r2 - 1) // ncols_r2
fig, axes = plt.subplots(nrows_r2, ncols_r2,
                         figsize=(5 * ncols_r2, 3.5 * nrows_r2),
                         squeeze=False, sharey=True)

for gi, g in enumerate(gains):
    ax = axes[gi // ncols_r2, gi % ncols_r2]
    if g not in r2_by_gain:
        ax.set_visible(False)
        continue
    rd = r2_by_gain[g]
    ax.plot(rd["centers"], rd["total"], "k-", linewidth=2, label="overall $R^2$")
    for ch in range(n_channels):
        ax.plot(rd["centers"], rd["per_ch"][ch], linewidth=1.0,
                color=ch_colors[ch], alpha=0.7, label=channel_labels[ch] if gi == 0 else None)
    ax.axvline(rd["train_T"], color="black", linestyle=":",
               linewidth=1.5, alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title(f"g = {g}", fontsize=11, fontweight="bold")
    ax.set_ylim(-1.5, 1.1)
    ax.grid(True, alpha=0.2)
    if gi // ncols_r2 == nrows_r2 - 1:
        ax.set_xlabel("Time", fontsize=10)
    if gi % ncols_r2 == 0:
        ax.set_ylabel("Windowed $R^2$", fontsize=10)

for i in range(len(gains), nrows_r2 * ncols_r2):
    axes[i // ncols_r2, i % ncols_r2].set_visible(False)

if len(gains) > 0 and 0 in r2_by_gain:
    legend_handles = [Line2D([0], [0], color="k", linewidth=2, label="overall")]
    for ch in range(n_channels):
        legend_handles.append(Line2D([0], [0], color=ch_colors[ch], linewidth=1.5,
                                     label=channel_labels[ch]))
    axes[0, -1].legend(handles=legend_handles, fontsize=6, loc="lower left")

fig.suptitle(f"Windowed $R^2$ (window={R2_WINDOW} steps) — "
             f"dotted line = training horizon",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "sw_windowed_r2.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## Global Jacobian Analysis
#
# With **Identity** (linear) activation, the discrete-time dynamics are:
# $$r_{t+1} = (1-\alpha)\, r_t + \alpha\,(g\, W_\text{rec}\, r_t + b)$$
#
# The Jacobian is **state-independent** (no nonlinearity):
# $$J = (1-\alpha)\, I + \alpha\, g\, W_\text{rec}$$
#
# so there is no need for fixed-point finding — the spectrum is a global
# property of the trained weight matrix.  Eigenvalues with $|\lambda| < 1$
# correspond to decaying modes; complex-conjugate pairs on or near the unit
# circle produce sustained oscillations at angular frequency
# $\omega = \arg(\lambda)$, i.e.\ period $T = 2\pi / \omega$.

# %% Compute global Jacobian for each gain
TAU_VIS_MAX = None
NUM_MODES = 30
ZOOM_XLIM = (0.7, 1.05)  # e.g. (0.7, 1.05) to zoom; None = full view
ZOOM_YLIM = (-0.5, 0.5)  # e.g. (-0.5, 0.5) to zoom; None = full view
pair_colors = ["#2a9d8f", "#e76f51", "#264653"][:n_pairs]

jac_data = {}

for _, row in df.drop_duplicates(subset="gain").iterrows():
    g = row["gain"]
    seed_path = row["seed_path"]
    rc = row["run_config"]

    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        print(f"g={g}: no checkpoint, skipping")
        continue

    ckpt = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    W_rec = W_out = b_out = None
    for key, val in sd.items():
        if "W_rec.weight" in key:
            W_rec = val.numpy()
        elif "W_out.weight" in key or "readout.weight" in key:
            W_out = val.numpy()
        elif "W_out.bias" in key or "readout.bias" in key:
            b_out = val.numpy()

    if W_rec is None:
        continue

    N = W_rec.shape[0]
    dt_rc = rc["dt"]
    tau_val = rc["time_constants_config"]["values"][0]
    alpha = 1.0 - np.exp(-dt_rc / tau_val)

    J = np.diag(np.full(N, 1 - alpha)) + alpha * g * W_rec
    eigs, V = np.linalg.eig(J)

    abs_eigs = np.abs(eigs)
    log_abs = np.log(np.clip(abs_eigs, 1e-12, None))
    tau_eff = -1.0 / np.clip(log_abs, None, -1e-10)

    osc_periods = np.full_like(abs_eigs, np.nan, dtype=float)
    imag_mask = np.abs(eigs.imag) > 1e-10
    osc_periods[imag_mask] = 2.0 * np.pi / np.abs(np.angle(eigs[imag_mask]))

    jac_data[g] = dict(
        eigs=eigs, V=V, abs_eigs=abs_eigs, tau_eff=tau_eff,
        osc_periods=osc_periods, alpha=alpha, N=N, J=J,
        W_out=W_out, b_out=b_out,
    )

    print(f"g={g}: α={alpha:.4f}, N={N}")
    print(f"  Top-5 |λ|: {np.sort(abs_eigs)[-5:][::-1]}")
    print(f"  # eigs with |λ| > 0.99: {np.sum(abs_eigs > 0.99)}")
    print(f"  spectral radius: {abs_eigs.max():.6f}")
    top_osc = osc_periods[imag_mask]
    if len(top_osc) > 0:
        top_osc_sorted = np.sort(top_osc[np.isfinite(top_osc)])
        print(f"  oscillatory modes: {np.sum(imag_mask)} "
              f"(periods from {top_osc_sorted[0]:.1f} to {top_osc_sorted[-1]:.1f})")

# %% Plot: Output-to-mode coupling heatmap — top NUM_MODES per gain
for g in sorted(jac_data):
    d = jac_data[g]
    eigs = d["eigs"]
    V = d["V"]
    W_out = d["W_out"]
    N = d["N"]

    if W_out is None:
        continue

    n_show = min(NUM_MODES, N)
    abs_rank_order = np.argsort(np.abs(eigs))[::-1]
    top_idx = abs_rank_order[:n_show]

    abs_top = np.abs(eigs[top_idx])
    log_abs_top = np.log(np.clip(abs_top, 1e-12, None))
    tau_top = -1.0 / np.where(log_abs_top < -1e-10, log_abs_top, -1e-10)
    tau_top = np.clip(tau_top, 0, 1e6)

    is_complex = np.abs(eigs[top_idx].imag) > 1e-8

    xtick_labels = []
    xtick_colors = []
    for m in range(n_show):
        tau_str = f"{tau_top[m]:.0f}" if tau_top[m] < 1e5 else "∞"
        marker = "*" if is_complex[m] else ""
        xtick_labels.append(f"{m+1}{marker}\n(τ={tau_str})")
        xtick_colors.append("#2563eb" if is_complex[m] else "black")

    coupling_out = np.abs(W_out @ V)[:, top_idx]  # (n_channels, n_show)

    fig, ax = plt.subplots(figsize=(max(n_show * 0.45, 8),
                                    max(n_channels * 0.6, 3)))
    im = ax.imshow(coupling_out, cmap="YlOrRd", aspect="auto",
                   interpolation="nearest")

    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_labels, fontsize=9)
    ax.set_xticks(range(n_show))
    ax.set_xticklabels(xtick_labels, fontsize=6)
    for tick_label, color in zip(ax.get_xticklabels(), xtick_colors):
        tick_label.set_color(color)
        if color != "black":
            tick_label.set_fontweight("bold")

    # Mark conjugate pairs
    m = 0
    while m < n_show - 1:
        if is_complex[m] and is_complex[m + 1]:
            lam_a = eigs[top_idx[m]]
            lam_b = eigs[top_idx[m + 1]]
            if abs(lam_a - lam_b.conjugate()) < 1e-6:
                y_bot = n_channels - 0.5
                ax.annotate("", xy=(m, y_bot + 0.15),
                            xytext=(m + 1, y_bot + 0.15),
                            arrowprops=dict(arrowstyle="-", color="#2563eb",
                                            lw=1.8),
                            annotation_clip=False)
                ax.text((m + m + 1) / 2, y_bot + 0.35, "cc",
                        ha="center", va="bottom", fontsize=5,
                        color="#2563eb", clip_on=False,
                        transform=ax.transData)
                m += 2
                continue
        m += 1

    for i in range(n_channels):
        for j in range(n_show):
            val = coupling_out[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if val > 0.5 * coupling_out.max() else "black")

    ax.set_xlabel("Eigenmode rank (by $|\\lambda|$) — * = complex, cc = conjugate pair",
                  fontsize=10)
    ax.set_ylabel("Output channel", fontsize=11)
    plt.colorbar(im, ax=ax, label="$|W_{\\mathrm{out}} V|$", shrink=0.8)
    rho = np.max(np.abs(eigs))
    fig.suptitle(f"Output-to-Mode Coupling — g={g}, ρ={rho:.3f}\n"
                 f"$|W_{{\\mathrm{{out}}}} V|$, top {n_show} of {N} modes",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"sw_output_coupling_g{g}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% Plot: Eigenspectrum — one per gain
# HIGHLIGHT_MODE: "coupling" = top-coupled modes per output pair (colored by pair)
#                 "top_n"    = top N modes by |λ| (red)
HIGHLIGHT_MODE = "top_n"
COUPLING_TOP_N = 2 * n_pairs  # per-pair count for "coupling" mode
TOP_N = 20                    # total count for "top_n" mode
theta_circle = np.linspace(0, 2 * np.pi, 200)

for g in sorted(jac_data):
    d = jac_data[g]
    eigs = d["eigs"]
    abs_eigs = d["abs_eigs"]
    V = d["V"]
    W_out = d["W_out"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(np.cos(theta_circle), np.sin(theta_circle),
            "k-", linewidth=0.6, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)

    if HIGHLIGHT_MODE == "coupling" and W_out is not None:
        coupling_full = np.abs(W_out @ V)  # (n_channels, N)
        pair_coupling = np.zeros((n_pairs, len(eigs)))
        for k in range(n_pairs):
            pair_coupling[k] = coupling_full[2*k] + coupling_full[2*k + 1]

        highlighted = set()
        eig_pair_assignment = {}
        for k in range(n_pairs):
            top_for_pair = np.argsort(pair_coupling[k])[-COUPLING_TOP_N:]
            for idx in top_for_pair:
                if idx not in highlighted:
                    highlighted.add(idx)
                    eig_pair_assignment[idx] = k

        bulk_mask = np.array([i not in highlighted for i in range(len(eigs))])
        ax.scatter(eigs[bulk_mask].real, eigs[bulk_mask].imag,
                   s=6, color="#ccc", alpha=0.5, edgecolors="none",
                   rasterized=True, zorder=2, label="bulk")

        for k in range(n_pairs):
            idxs = np.array([i for i, pk in eig_pair_assignment.items()
                             if pk == k])
            if len(idxs) == 0:
                continue
            c_strength = pair_coupling[k, idxs]
            sizes = 30 + 120 * (c_strength / max(pair_coupling.max(), 1e-12))
            ax.scatter(eigs[idxs].real, eigs[idxs].imag,
                       s=sizes, color=pair_colors[k], alpha=0.85,
                       edgecolors="black", linewidths=0.4, zorder=5,
                       label=f"coupled to T={periods_list[k]:.0f}")

    else:
        top_idx = np.argsort(abs_eigs)[-TOP_N:]
        rest_mask = np.ones(len(eigs), dtype=bool)
        rest_mask[top_idx] = False

        ax.scatter(eigs[rest_mask].real, eigs[rest_mask].imag,
                   s=6, color="#ccc", alpha=0.5, edgecolors="none",
                   rasterized=True, zorder=2, label="bulk")
        ax.scatter(eigs[top_idx].real, eigs[top_idx].imag,
                   s=35, color="#c1121f", edgecolors="black",
                   linewidths=0.4, alpha=0.85, zorder=5,
                   label=f"top {TOP_N} by |λ|")

    # Target frequencies as thin crosshairs
    for k in range(n_pairs):
        omega_k = 2.0 * np.pi / periods_list[k]
        lam_re = np.cos(omega_k)
        lam_im = np.sin(omega_k)
        cross_len = 0.04
        for sign in [1, -1]:
            li = sign * lam_im
            ax.plot([lam_re - cross_len, lam_re + cross_len], [li, li],
                    "-", color=pair_colors[k], linewidth=1.8, alpha=0.9,
                    zorder=9)
            ax.plot([lam_re, lam_re], [li - cross_len, li + cross_len],
                    "-", color=pair_colors[k], linewidth=1.8, alpha=0.9,
                    zorder=9)
        ax.plot([], [], "+", color=pair_colors[k], markersize=8,
                markeredgewidth=2, label=f"T={periods_list[k]:.0f} target")

    ax.set_title(f"Global Jacobian Eigenspectrum — g={g}\n"
                 f"(spectral radius = {abs_eigs.max():.4f})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Re(λ)", fontsize=11)
    ax.set_ylabel("Im(λ)", fontsize=11)
    ax.set_aspect("equal")
    if ZOOM_XLIM is not None:
        ax.set_xlim(*ZOOM_XLIM)
    else:
        ax.set_xlim(-1.15, 1.15)
    if ZOOM_YLIM is not None:
        ax.set_ylim(*ZOOM_YLIM)
    else:
        ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    if SAVE_FIGS:
        suffix = "_zoom" if ZOOM_XLIM is not None or ZOOM_YLIM is not None else ""
        fig.savefig(os.path.join(FIGS_DIR, f"sw_eigenspectrum_g{g}{suffix}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% Plot: τ_eff scree — one per gain
for g in sorted(jac_data):
    d = jac_data[g]
    tau_sorted = np.sort(np.clip(d["tau_eff"], 0, TAU_VIS_MAX))[::-1]
    ranks = np.arange(1, len(tau_sorted) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ranks, tau_sorted, marker="o", color="#264653")

    for k in range(n_pairs):
        ax.axhline(periods_list[k], color=pair_colors[k], linestyle="--",
                   linewidth=0.8, alpha=0.6,
                   label=f"pair {k} period = {periods_list[k]:.0f}")

    ax.set_xlabel("Eigenmode rank", fontsize=12)
    ax.set_ylabel("$\\tau_{\\mathrm{eff}}$", fontsize=13)
    ax.set_yscale("log")
    cap_str = f" (capped at {TAU_VIS_MAX})" if TAU_VIS_MAX is not None else ""
    ax.set_title(f"τ_eff Scree — g={g}{cap_str}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"sw_tau_scree_g{g}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% Plot: |λ| rank — one per gain
for g in sorted(jac_data):
    d = jac_data[g]
    abs_sorted = np.sort(d["abs_eigs"])[::-1]
    ranks = np.arange(1, len(abs_sorted) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ranks, abs_sorted, linewidth=1.5, color="#264653")
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1, alpha=0.5,
               label="|λ| = 1")

    ax.set_xlabel("Eigenmode rank", fontsize=12)
    ax.set_ylabel("$|\\lambda|$", fontsize=13)
    ax.set_title(f"|λ| Rank — g={g}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"sw_eigval_mag_g{g}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% Plot: Oscillatory period spectrum — nearest eigenvalue to each target
fig, ax = plt.subplots(figsize=(9, 5))

for k in range(n_pairs):
    target_omega = 2.0 * np.pi / periods_list[k]
    target_lam = np.exp(1j * target_omega)

    matched_abs = []
    for g in sorted(jac_data):
        eigs = jac_data[g]["eigs"]
        dists = np.abs(eigs - target_lam)
        best_idx = np.argmin(dists)
        matched_abs.append(np.abs(eigs[best_idx]))

    ax.plot(sorted(jac_data), matched_abs, "o-", color=pair_colors[k],
            linewidth=1.8, markersize=7,
            label=f"T={periods_list[k]:.0f}")

ax.axhline(1.0, color="red", linestyle=":", linewidth=1, alpha=0.5,
           label="|λ| = 1 (sustained)")
ax.set_xlabel("Recurrent gain $g$", fontsize=12)
ax.set_ylabel("|λ| of nearest eigenvalue to target frequency", fontsize=11)
ax.set_title("Does the network learn eigenvalues at the target frequencies?",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.legend(fontsize=9)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "sw_target_freq_match.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()
