# %% [markdown]
# # Coupled RNN — Trainable vs Fixed $W^s$ on 3-bit Flip-Flop
#
# Both runs use heterogeneous per-bit pulse rates (`p_pulse = [0.01, 0.05, 0.1]`)
# and a single $\tau_s = 1.0$.  The sweep axis is whether $W^s$ is learned
# (trainable) or kept at its random initialisation (fixed).
#
# Questions:
# - Does learning $W^s$ help the network solve the heterogeneous-timescale task?
# - How does the $W^s$ eigenspectrum change when it is trained?
# - Are per-bit accuracy differences affected by trainability of $W^s$?

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
sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/flip_flop_coupled_trainable_ws_20260407_010711"

# %% Load data and training curves
records = []

for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
        continue
    if not exp_name.startswith("trainable_ws_"):
        continue

    trainable_str = exp_name[len("trainable_ws_"):]
    trainable = trainable_str.lower() == "true"

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
            exp_name=exp_name, trainable_w_s=trainable, seed=seed,
            seed_path=seed_path, final_val_loss=fvl,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
        ))

df = pd.DataFrame(records)
print(f"Loaded {len(df)} runs")
print(df[["exp_name", "trainable_w_s", "seed", "final_val_loss"]])

# %% Colors / labels
COLORS = {False: "#264653", True: "#e76f51"}
LABELS = {False: "$W^s$ fixed", True: "$W^s$ trainable"}

# %% Plot 1: Validation loss vs training step
fig, ax = plt.subplots(figsize=(9, 4.5))
for _, row in df.iterrows():
    tw = row["trainable_w_s"]
    vl = row["val_losses"]
    st = row["steps"][:len(vl)] if row["steps"] else list(range(1, len(vl) + 1))
    ax.plot(st, vl, linewidth=1.8, color=COLORS[tw], alpha=0.85, label=LABELS[tw])

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation loss", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
fig.suptitle("Validation Loss — Trainable vs Fixed $W^s$",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% Plot 2: Validation accuracy vs training step
fig, ax = plt.subplots(figsize=(9, 4.5))
for _, row in df.iterrows():
    tw = row["trainable_w_s"]
    va = row["val_accs"]
    if not va:
        continue
    st = row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1))
    ax.plot(st, va, linewidth=1.8, color=COLORS[tw], alpha=0.85, label=LABELS[tw])

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation accuracy", fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
fig.suptitle("Validation Accuracy — Trainable vs Fixed $W^s$",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% Plot 3: Per-channel accuracy vs training step
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

for ax, (tw, grp) in zip(axes, df.groupby("trainable_w_s")):
    row = grp.iloc[0]
    seed_path = row["seed_path"]
    lf = os.path.join(seed_path, "training_losses.json")
    if not os.path.exists(lf):
        continue
    with open(lf) as f:
        ld = json.load(f)

    steps = ld.get("steps", [])
    ch_colors = ["#2a9d8f", "#e9c46a", "#e76f51"]

    config_file = os.path.join(seed_path, "run_config.yaml")
    with open(config_file) as f:
        run_config = yaml.safe_load(f)
    p_pulse = run_config["p_pulse"]
    p_list = p_pulse if isinstance(p_pulse, list) else [p_pulse] * run_config["n_bits"]

    for ch_idx in range(run_config["n_bits"]):
        key = f"val_accuracies_channel_{ch_idx}"
        ch_acc = ld.get(key, [])
        if not ch_acc:
            continue
        st = steps[:len(ch_acc)]
        ax.plot(st, ch_acc, linewidth=1.5, color=ch_colors[ch_idx % len(ch_colors)],
                alpha=0.85, label=f"bit {ch_idx} (p={p_list[ch_idx]})")

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(LABELS[tw], fontsize=13, fontweight="bold")

axes[0].set_ylabel("Validation accuracy", fontsize=12)
fig.suptitle("Per-Bit Accuracy — Trainable vs Fixed $W^s$",
             fontsize=14, fontweight="bold", y=1.04)
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
        trainable_w_s=run_config.get("trainable_w_s", False),
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


# %% Plot 4: Example trajectories — side by side
SEQ_IDX = 0

fig_rows = df.iloc[0]["val_accs"]  # just to check n_bits from config
row0 = df.iloc[0]
config_file = os.path.join(row0["seed_path"], "run_config.yaml")
with open(config_file) as f:
    rc0 = yaml.safe_load(f)
n_bits = rc0["n_bits"]
p_pulse = rc0["p_pulse"]
p_list = p_pulse if isinstance(p_pulse, list) else [p_pulse] * n_bits

fig, axes = plt.subplots(n_bits, 2, figsize=(16, 2.2 * n_bits), sharex=True, sharey=True)
if n_bits == 1:
    axes = axes.reshape(1, 2)

for col_idx, (_, row) in enumerate(df.iterrows()):
    tw = row["trainable_w_s"]
    lit, run_config = load_trained_model(row["seed_path"], device=device)
    if lit is None:
        continue

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

    t_arr = np.arange(run_config["num_time_steps"])

    for bit in range(n_bits):
        ax = axes[bit, col_idx]
        pulse = inp[SEQ_IDX, :, bit].numpy()
        set_mask = pulse > 0.5
        reset_mask = pulse < -0.5
        if set_mask.any():
            ax.scatter(t_arr[set_mask], np.full(set_mask.sum(), 0.65),
                       marker=6, s=25, color="C2", zorder=3,
                       label="set" if bit == 0 and col_idx == 0 else None)
        if reset_mask.any():
            ax.scatter(t_arr[reset_mask], np.full(reset_mask.sum(), 0.35),
                       marker=7, s=25, color="C3", zorder=3,
                       label="reset" if bit == 0 and col_idx == 0 else None)
        ax.step(t_arr, tgt[SEQ_IDX, :, bit].numpy(), where="post",
                color="black", linewidth=1.5,
                label="target" if bit == 0 and col_idx == 0 else None)
        ax.plot(t_arr, out_prob[SEQ_IDX, :, bit].numpy(),
                color=COLORS[tw], linewidth=1.2, alpha=0.9,
                label="output" if bit == 0 and col_idx == 0 else None)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1])
        if col_idx == 0:
            ax.set_ylabel(f"Bit {bit}\n(p={p_list[bit]})", fontsize=10)
        if bit == 0:
            ax.set_title(LABELS[tw], fontsize=12, fontweight="bold")

axes[-1, 0].set_xlabel("Timestep", fontsize=11)
axes[-1, 1].set_xlabel("Timestep", fontsize=11)
axes[0, 0].legend(fontsize=8, loc="upper right")
fig.suptitle("Output Trajectories — Trainable vs Fixed $W^s$",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# %% Plot 5: W_s eigenspectrum — untrained vs trained, side by side
theta = np.linspace(0, 2 * np.pi, 200)
UNTRAINED_COLOR = "#8da0b5"
TRAINED_COLOR = "#e76f51"

fig, axes = plt.subplots(2, 2, figsize=(9, 9), squeeze=False)

for row_idx, (_, row) in enumerate(df.iterrows()):
    tw = row["trainable_w_s"]
    seed_path = row["seed_path"]
    config_file = os.path.join(seed_path, "run_config.yaml")
    with open(config_file) as f:
        run_config = yaml.safe_load(f)

    dt = run_config["dt"]
    tau_s = run_config["tau_s"]
    w_s_gain = run_config.get("w_s_gain", 1.0)
    alpha_s = 1.0 - np.exp(-dt / tau_s)

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
        W_s = None
        for key, val in ckpt["state_dict"].items():
            if "W_s" in key:
                W_s = val.numpy()
                break
        if W_s is None:
            continue

        N_s = W_s.shape[0]
        J_s = (1.0 - alpha_s) * np.eye(N_s) + alpha_s * w_s_gain * W_s
        eigs_s = np.linalg.eigvals(J_s)
        max_abs_s = np.max(np.abs(eigs_s))

        ax.plot(np.cos(theta), np.sin(theta), color="#c0392b",
                linewidth=1.2, alpha=0.5, linestyle="--", zorder=1,
                label="$|\\lambda|=1$" if row_idx == 0 and col_idx == 0 else None)
        ax.axhline(0, color="#bbb", linewidth=0.3, zorder=0)
        ax.axvline(0, color="#bbb", linewidth=0.3, zorder=0)

        pt_color = UNTRAINED_COLOR if col_idx == 0 else TRAINED_COLOR
        ax.scatter(eigs_s.real, eigs_s.imag, s=18, alpha=0.7,
                   c=pt_color, edgecolors="none", zorder=3)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)

        if row_idx == 0:
            ax.set_title(label, fontsize=13, fontweight="bold")
        ax.annotate(f"$|\\lambda|_{{max}}$={max_abs_s:.3f}",
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        if col_idx == 0:
            ax.set_ylabel(f"{LABELS[tw]}\nIm($\\lambda$)", fontsize=11)
        if row_idx == 1:
            ax.set_xlabel("Re($\\lambda$)", fontsize=11)

axes[0, 0].legend(fontsize=8, loc="lower left", framealpha=0.7)
fig.suptitle("$W^s$ Discrete Jacobian Spectrum: $(1-\\alpha_s)I + \\alpha_s g_s W^s$",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# %% Plot 6: r-network Jacobian eigenspectrum — untrained vs trained
fig, axes = plt.subplots(2, 2, figsize=(9, 9), squeeze=False)

for row_idx, (_, row) in enumerate(df.iterrows()):
    tw = row["trainable_w_s"]
    seed_path = row["seed_path"]
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
                label="$|\\lambda|=1$" if row_idx == 0 and col_idx == 0 else None)
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
            ax.set_ylabel(f"{LABELS[tw]}\nIm($\\lambda$)", fontsize=11)
        if row_idx == 1:
            ax.set_xlabel("Re($\\lambda$)", fontsize=11)

axes[0, 0].legend(fontsize=8, loc="lower left", framealpha=0.7)
fig.suptitle("r-network Jacobian Spectrum — Trainable vs Fixed $W^s$",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# %% Summary table
print("\n" + "=" * 55)
print(f"{'trainable_w_s':>15}  {'final_val_loss':>15}  {'seed':>6}")
print("-" * 55)
for _, r in df.iterrows():
    print(f"{str(r['trainable_w_s']):>15}  {r['final_val_loss']:15.6f}  {r['seed']:6d}")
print("=" * 55)
