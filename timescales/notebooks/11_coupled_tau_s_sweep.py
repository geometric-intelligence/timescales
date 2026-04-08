# %% [markdown]
# # Coupled RNN — $\tau_s$ Sweep on 3-bit Flip-Flop
#
# Sweep over the timescale of the linear s-network ($\tau_s$) while
# keeping $\tau_r$ fixed.  Questions:
# - How does the memory timescale affect learning speed and final performance?
# - Does the trained r-network adapt its eigenspectrum depending on $\tau_s$?
# - What do the output trajectories look like for different $\tau_s$?

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

from rnns.coupled_rnn import CoupledRNN, CoupledRNNLightning
from datamodules.flip_flop import FlipFlopDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Sweep directory
sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/flip_flop_coupled_tau_s_20260406_185430"

# %% Load data and training curves
records = []

for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
        continue
    if not exp_name.startswith("tau_s_"):
        continue

    tau_s_str = exp_name[len("tau_s_"):]
    try:
        tau_s = float(tau_s_str)
    except ValueError:
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
            val_losses = ld.get("val_losses", [])
            val_accs = ld.get("val_accuracies", [])
            steps = ld.get("steps", [])

        if fvl is None and val_losses:
            fvl = val_losses[-1]
        if fvl is None:
            continue

        records.append(dict(
            exp_name=exp_name, tau_s=tau_s, seed=seed,
            seed_path=seed_path, final_val_loss=fvl,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
        ))

df = pd.DataFrame(records)
tau_s_values = sorted(df["tau_s"].unique())
print(f"Loaded {len(df)} runs, tau_s values: {tau_s_values}")

# %% Color palette (monotonic — cool blues for small tau_s, warm reds for large)
_cmap = plt.cm.coolwarm
_norm = plt.Normalize(vmin=np.log10(min(tau_s_values)), vmax=np.log10(max(tau_s_values)))
COLORS = {ts: _cmap(_norm(np.log10(ts))) for ts in tau_s_values}

# %% Plot 1: Validation loss vs training step
fig, ax = plt.subplots(figsize=(9, 4.5))
plotted_labels = set()
for _, row in df.iterrows():
    ts = row["tau_s"]
    vl = row["val_losses"]
    st = row["steps"][:len(vl)] if row["steps"] else list(range(1, len(vl) + 1))
    label = f"$\\tau_s$ = {ts}" if ts not in plotted_labels else None
    ax.plot(st, vl, linewidth=1.6, color=COLORS[ts], alpha=0.8, label=label)
    plotted_labels.add(ts)

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation loss", fontsize=12)
ax.set_yscale("log")
# ax.set_xscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2)
fig.suptitle("Coupled RNN — Flip-Flop: Validation Loss vs Training Step",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% Plot 2: Validation accuracy vs training step
has_acc = df["val_accs"].apply(len).sum() > 0
if has_acc:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted_labels = set()
    for _, row in df.iterrows():
        ts = row["tau_s"]
        va = row["val_accs"]
        if not va:
            continue
        st = row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1))
        label = f"$\\tau_s$ = {ts}" if ts not in plotted_labels else None
        ax.plot(st, va, linewidth=1.6, color=COLORS[ts], alpha=0.8, label=label)
        plotted_labels.add(ts)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Validation accuracy", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.suptitle("Coupled RNN — Flip-Flop: Validation Accuracy vs Training Step",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

# %% Plot 3: Final validation loss vs tau_s (aggregated over seeds)
agg = df.groupby("tau_s")["final_val_loss"].agg(["mean", "std", "count"]).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(agg["tau_s"], agg["mean"], yerr=agg["std"], fmt="o-",
            color="#264653", capsize=4, linewidth=2, markersize=7)

for _, row in df.iterrows():
    ax.scatter(row["tau_s"], row["final_val_loss"], s=20,
               color=COLORS[row["tau_s"]], alpha=0.5, zorder=2)

ax.set_xlabel("$\\tau_s$ (s-network timescale)", fontsize=12)
ax.set_ylabel("Final validation loss", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
fig.suptitle("Final Loss vs $\\tau_s$", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# %% Plot 3b: Steps to convergence (accuracy-based) vs tau_s
ACC_THRESHOLD = 0.95

convergence_records = []
for _, row in df.iterrows():
    va = np.array(row["val_accs"])
    if len(va) == 0:
        convergence_records.append(dict(tau_s=row["tau_s"], seed=row["seed"],
                                        steps_to_convergence=np.nan))
        continue
    st = np.array(row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1)))
    above = np.where(va >= ACC_THRESHOLD)[0]
    steps_to_conv = int(st[above[0]]) if len(above) > 0 else np.nan
    convergence_records.append(dict(tau_s=row["tau_s"], seed=row["seed"],
                                    steps_to_convergence=steps_to_conv))

conv_df = pd.DataFrame(convergence_records)
conv_agg = conv_df.groupby("tau_s")["steps_to_convergence"].agg(["mean", "std", "count"]).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(conv_agg["tau_s"], conv_agg["mean"], yerr=conv_agg["std"], fmt="o-",
            color="#264653", capsize=4, linewidth=2, markersize=7)

for _, row in conv_df.iterrows():
    if not np.isnan(row["steps_to_convergence"]):
        ax.scatter(row["tau_s"], row["steps_to_convergence"], s=20,
                   color=COLORS[row["tau_s"]], alpha=0.5, zorder=2)

n_failed = conv_df["steps_to_convergence"].isna().sum()
if n_failed > 0:
    failed = conv_df[conv_df["steps_to_convergence"].isna()]
    failed_taus = sorted(failed["tau_s"].unique())
    ax.set_title(f"({n_failed} run(s) never reached threshold: "
                 f"$\\tau_s$ ∈ {{{', '.join(str(t) for t in failed_taus)}}})",
                 fontsize=9, style="italic")

ax.set_xlabel("$\\tau_s$ (s-network timescale)", fontsize=12)
ax.set_ylabel("Steps to convergence", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
fig.suptitle(f"Steps to Convergence (val acc ≥ {ACC_THRESHOLD}) vs $\\tau_s$",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# %% Helper: reconstruct CoupledRNN + Lightning from a run config
def load_trained_model(seed_path, device="cpu"):
    """Load run_config + best checkpoint, return (lit, run_config)."""
    config_file = os.path.join(seed_path, "run_config.yaml")
    with open(config_file) as f:
        run_config = yaml.safe_load(f)

    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        return None, run_config

    n_bits = run_config["n_bits"]
    model = CoupledRNN(
        input_size=n_bits,
        r_hidden_size=run_config["r_hidden_size"],
        s_hidden_size=run_config["s_hidden_size"],
        output_size=n_bits,
        dt=run_config["dt"],
        tau_r=run_config["tau_r"],
        tau_s=run_config["tau_s"],
        activation=getattr(nn, run_config.get("activation", "Tanh")),
        zero_diag_wrec=run_config.get("zero_diag_wrec", True),
        recurrent_gain=run_config.get("recurrent_gain", 1.0),
        noise_std=0.0,
        wrec_init=run_config.get("wrec_init", "orthogonal"),
        w_s_gain=run_config.get("w_s_gain", 1.0),
    )
    lit = CoupledRNNLightning(
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
    return lit, run_config


# %% Plot 4: Example trajectories for each tau_s (first seed only)
SEQ_IDX = 0

first_seeds = df.sort_values("seed").groupby("tau_s").first().reset_index()

for _, row in first_seeds.iterrows():
    ts = row["tau_s"]
    lit, run_config = load_trained_model(row["seed_path"], device=device)
    if lit is None:
        continue

    n_bits = run_config["n_bits"]
    p_pulse = run_config["p_pulse"]

    dm = FlipFlopDataModule(
        n_bits=n_bits,
        p_pulse=p_pulse,
        pulse_amplitude=run_config.get("pulse_amplitude", 1.0),
        num_time_steps=run_config["num_time_steps"],
        num_val_trajectories=10,
        batch_size=10,
    )
    dm.setup()
    inp, _, tgt = dm.val_dataset.tensors

    with torch.no_grad():
        _, _, out = lit.model(inp.to(device), init_context=None)
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
        ax.set_ylabel(f"Bit {bit}", fontsize=10)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Coupled RNN Trajectories — $\\tau_s$ = {ts}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


# %% Plot 5: r-network Jacobian eigenspectrum — untrained vs trained (one per tau_s)
theta = np.linspace(0, 2 * np.pi, 200)
UNTRAINED_COLOR = "#8da0b5"
TRAINED_COLOR = "#e76f51"

n_tau = len(tau_s_values)
fig, axes = plt.subplots(n_tau, 2, figsize=(7.5, 3.2 * n_tau), squeeze=False)

for row_idx, ts in enumerate(tau_s_values):
    row_data = df[df["tau_s"] == ts].iloc[0]
    seed_path = row_data["seed_path"]
    config_file = os.path.join(seed_path, "run_config.yaml")
    with open(config_file) as f:
        run_config = yaml.safe_load(f)

    dt = run_config["dt"]
    tau_r = run_config["tau_r"]
    gain = run_config.get("recurrent_gain", 1.0)
    alpha_r = 1.0 - np.exp(-dt / tau_r)

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
        J = (1.0 - alpha_r) * np.eye(N) + alpha_r * gain * W_rec
        eigs = np.linalg.eigvals(J)
        max_abs = np.max(np.abs(eigs))

        ax.plot(np.cos(theta), np.sin(theta), color="#c0392b",
                linewidth=1.2, alpha=0.5, linestyle="--", zorder=1,
                label="$|\\lambda|=1$")
        ax.axhline(0, color="#bbb", linewidth=0.3, zorder=0)
        ax.axvline(0, color="#bbb", linewidth=0.3, zorder=0)

        pt_color = UNTRAINED_COLOR if col_idx == 0 else TRAINED_COLOR
        ax.scatter(eigs.real, eigs.imag, s=18, alpha=0.7,
                   c=pt_color, edgecolors="none", zorder=3)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)

        if row_idx == 0:
            ax.set_title(label, fontsize=13, fontweight="bold")
        ax.annotate(f"$|\\lambda|_{{max}}$={max_abs:.3f}",
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        if col_idx == 0:
            ax.set_ylabel(f"$\\tau_s$={ts}\nIm($\\lambda$)", fontsize=11)
        if row_idx == n_tau - 1:
            ax.set_xlabel("Re($\\lambda$)", fontsize=11)

axes[0, 0].legend(fontsize=8, loc="lower left", framealpha=0.7)
fig.suptitle("r-network Jacobian Spectrum — Coupled RNN, varying $\\tau_s$",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()


# %% Plot 6: s-network discrete-time Jacobian spectrum (fixed W_s, varies only via alpha_s)
fig, axes = plt.subplots(1, n_tau, figsize=(3.5 * n_tau, 3.5), squeeze=False)

for col, ts in enumerate(tau_s_values):
    ax = axes[0, col]
    row_data = df[df["tau_s"] == ts].iloc[0]
    seed_path = row_data["seed_path"]

    ckpt_path = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not ckpt_path:
        continue
    ckpt = torch.load(ckpt_path[0], map_location="cpu", weights_only=False)

    W_s = None
    for key, val in ckpt["state_dict"].items():
        if "W_s" in key:
            W_s = val.numpy()
            break
    if W_s is None:
        continue

    config_file = os.path.join(seed_path, "run_config.yaml")
    with open(config_file) as f:
        run_config = yaml.safe_load(f)
    dt = run_config["dt"]
    alpha_s = 1.0 - np.exp(-dt / ts)

    N_s = W_s.shape[0]
    J_s = (1.0 - alpha_s) * np.eye(N_s) + alpha_s * W_s
    eigs_s = np.linalg.eigvals(J_s)
    max_abs_s = np.max(np.abs(eigs_s))

    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="$|\\lambda|=1$")
    ax.scatter(eigs_s.real, eigs_s.imag, s=18, alpha=0.7, c="#2a9d8f", edgecolors="none")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_title(f"$\\tau_s$ = {ts}", fontsize=11)
    ax.annotate(f"$|\\lambda|_{{max}}$={max_abs_s:.3f}",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.annotate(f"$\\alpha_s$={alpha_s:.4f}",
                xy=(0.97, 0.82), xycoords="axes fraction",
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    if col == 0:
        ax.set_ylabel("Im($\\lambda$)", fontsize=11)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.7)
    ax.set_xlabel("Re($\\lambda$)", fontsize=11)

fig.suptitle("s-network discrete Jacobian: $(1-\\alpha_s)I + \\alpha_s W^s$",
             fontsize=13, fontweight="bold", y=1.04)
plt.tight_layout()
plt.show()


# %% Summary table
print("\n" + "=" * 65)
print(f"{'tau_s':>8}  {'mean_loss':>10}  {'std_loss':>10}  {'n_seeds':>7}")
print("-" * 65)
for _, r in agg.iterrows():
    print(f"{r['tau_s']:8.1f}  {r['mean']:10.4f}  {r['std']:10.4f}  {int(r['count']):7d}")
print("=" * 65)
