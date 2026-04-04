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

from rnns.multitimescale_rnn import MultiTimescaleRNN, MultiTimescaleRNNLightning
from datamodules.flip_flop import FlipFlopDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Discover sweep
LOGS_DIR = "logs/experiments"

sweep_dirs = sorted(
    [d for d in os.listdir(LOGS_DIR) if d.startswith("flip_flop_hetero_ppulse")],
    reverse=True,
)
if not sweep_dirs:
    raise RuntimeError("No flip_flop_hetero_ppulse directories found")

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

        val_losses = []
        steps = []
        lf = os.path.join(seed_path, "training_losses.json")
        if os.path.exists(lf):
            with open(lf) as f:
                ld = json.load(f)
            val_losses = ld.get("val_losses", ld.get("val_losses_epoch", []))
            steps = ld.get("steps", [])

        if fvl is None and val_losses:
            fvl = val_losses[-1]
        if fvl is None:
            continue

        records.append(dict(
            exp_name=exp_name, gain=gain, seed=seed,
            seed_path=seed_path, final_val_loss=fvl,
            val_losses=val_losses, steps=steps,
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

    model = MultiTimescaleRNN(
        input_size=n_bits,
        hidden_size=run_config["hidden_size"],
        output_size=n_bits,
        dt=run_config["dt"],
        timescales_config=run_config.get("timescales_config"),
        activation=getattr(nn, run_config["activation"]),
        learn_timescales=run_config["learn_timescales"],
        init_timescale=run_config.get("init_timescale"),
        shared_timescale=run_config["shared_timescale"],
        normalize_hidden=run_config["normalize_hidden"],
        zero_diag_wrec=run_config["zero_diag_wrec"],
        recurrent_gain=run_config["recurrent_gain"],
        noise_std=0.0,
        wrec_init=run_config["wrec_init"],
        alpha_parameterization=run_config["alpha_parameterization"],
        dynamics_type=run_config["dynamics_type"],
    )
    lit = MultiTimescaleRNNLightning(
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
    tau = run_config["timescales_config"]["values"][0]
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

        N = W_rec.shape[0]
        J = (1.0 - alpha) * np.eye(N) + alpha * g * W_rec
        eigenvalues, eigvecs = np.linalg.eig(J)
        eigs = eigenvalues
        abs_eigs = np.abs(eigs)
        max_abs = np.max(abs_eigs)

        if col_idx == 1:
            eig_data[g] = dict(
                eigs=eigs, eigvecs=eigvecs, W_out=W_out,
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


# %% Plot 4: Effective timescale scree plot — tau_eff = -1/ln|lambda|
fig, axes = plt.subplots(1, n_gains, figsize=(3.5 * n_gains, 3.5), squeeze=False, sharey=True)

for col, g in enumerate(gains):
    ax = axes[0, col]
    if g not in eig_data:
        continue

    data = eig_data[g]
    eigs = data["eigs"]
    n_bits = data["n_bits"]
    abs_sorted = np.sort(np.abs(eigs))[::-1]
    N = len(abs_sorted)
    ranks = np.arange(1, N + 1)

    log_abs = np.log(np.clip(abs_sorted, 1e-12, None))
    tau_eff = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)

    ax.scatter(ranks[:n_bits], tau_eff[:n_bits], s=50, color=TRAINED_COLOR,
               edgecolors="none", zorder=3, label=f"Top {n_bits} (slow modes)")
    ax.scatter(ranks[n_bits:], tau_eff[n_bits:], s=30, color=UNTRAINED_COLOR,
               edgecolors="none", zorder=3, label="Remaining")

    for i in range(min(n_bits, len(tau_eff))):
        ax.annotate(f"{tau_eff[i]:.1f}", (ranks[i], tau_eff[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8,
                    color=TRAINED_COLOR)

    ax.set_xlabel("Rank", fontsize=11)
    if col == 0:
        ax.set_ylabel("$\\tau_{\\mathrm{eff}} = -1\\,/\\,\\ln|\\lambda|$\n(steps)",
                      fontsize=10)
    ax.set_title(f"g = {g}", fontsize=11)
    ax.grid(True, alpha=0.15, which="both")
    if col == 0:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.set_yscale("log")

# Build subtitle from actual config
_sample = next(iter(eig_data.values()))
_pp = _sample["p_pulse"]
_pp_str = _pp if isinstance(_pp, list) else [_pp]
_holds = [f"{1.0/p:.0f}" for p in _pp_str]
fig.suptitle(f"Effective Timescales (ranked) — Hetero p_pulse (Identity)\n"
             f"p_pulse = {_pp_str}  (hold intervals ~{_holds} steps)",
             fontsize=12, fontweight="bold", y=1.06)
plt.tight_layout()
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

    coupling = W_out @ V
    coupling_top = np.abs(coupling[:, top_mode_idx])

    fig, ax = plt.subplots(figsize=(max(n_bits * 0.9, 4), max(n_bits * 0.6, 3)))
    im = ax.imshow(coupling_top, cmap="YlOrRd", aspect="auto")

    ax.set_yticks(range(n_bits))
    ax.set_yticklabels([f"Bit {i} (p={pp_list[i]})" for i in range(n_bits)], fontsize=9)
    ax.set_xticks(range(n_bits))
    ax.set_xticklabels([f"Mode {i+1}\n($\\tau$={tau_modes[i]:.0f})" for i in range(n_bits)],
                       fontsize=8)
    ax.set_xlabel("Eigenmode (ranked by $|\\lambda|$)", fontsize=11)
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
    plt.show()

    # Identify dominant mode per bit
    dominant = np.argmax(coupling_top, axis=1)
    print(f"\ng = {g}: Dominant mode per output bit:")
    for bit_i in range(n_bits):
        mode_j = dominant[bit_i]
        print(f"  Bit {bit_i} (p_pulse={pp_list[bit_i]}, hold~{1.0/pp_list[bit_i]:.0f} steps)"
              f"  <--  Mode {mode_j+1} (tau_eff={tau_modes[mode_j]:.1f} steps)")


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
