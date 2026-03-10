# %% [markdown]
# # Flip-Flop Gain Sweep
# Final validation loss vs recurrent gain `g`, for Identity vs Tanh activations.

# %%
import os
import sys
import subprocess
import json

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()

os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())

# %% Discover sweep
LOGS_DIR = "logs/experiments"

sweep_dirs = sorted(
    [d for d in os.listdir(LOGS_DIR) if d.startswith("flip_flop_gain_sweep")],
    reverse=True,
)
if not sweep_dirs:
    raise RuntimeError(f"No flip_flop_gain_sweep directories found in {LOGS_DIR}")

sweep_dir = os.path.join(LOGS_DIR, sweep_dirs[0])
print(f"Using sweep: {sweep_dir}")

#%%
print(sweep_dir)

# %% Load data
records = []

for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
        continue

    parts = exp_name.split("_")
    if len(parts) < 3 or parts[0] != "g":
        continue
    gain = float(parts[1])
    activation = parts[2]

    for seed_dir_name in sorted(os.listdir(exp_dir)):
        if not seed_dir_name.startswith("seed_"):
            continue
        seed = int(seed_dir_name.split("_")[1])
        seed_path = os.path.join(exp_dir, seed_dir_name)

        final_val_loss = None

        result_file = os.path.join(seed_path, "job_result.yaml")
        if os.path.exists(result_file):
            with open(result_file) as f:
                result = yaml.safe_load(f)
            final_val_loss = result.get("final_val_loss")

        if final_val_loss is None:
            losses_file = os.path.join(seed_path, "training_losses.json")
            if os.path.exists(losses_file):
                with open(losses_file) as f:
                    losses = json.load(f)
                val_losses = losses.get("val_losses_epoch", [])
                if val_losses:
                    final_val_loss = val_losses[-1]

        if final_val_loss is None:
            print(f"  No results found for {exp_name}/{seed_dir_name}")
            continue

        records.append({
            "exp_name": exp_name,
            "gain": gain,
            "activation": activation,
            "seed": seed,
            "final_val_loss": final_val_loss,
        })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} runs")
print(df)

# %% Plot: final loss vs g
if df.empty:
    raise RuntimeError("No completed runs found yet.")

activations = sorted(df["activation"].unique())
gains = sorted(df["gain"].unique())

fig, axes = plt.subplots(1, len(activations), figsize=(5 * len(activations), 4), sharey=True)
if len(activations) == 1:
    axes = [axes]

for ax, activation in zip(axes, activations):
    sub = df[df["activation"] == activation]
    grouped = sub.groupby("gain")["final_val_loss"]
    means = grouped.mean()
    sems = grouped.sem()

    ax.plot(means.index, means.values, marker="o", linewidth=2, color="steelblue")
    ax.fill_between(
        means.index,
        means.values - sems.values,
        means.values + sems.values,
        alpha=0.25,
        color="steelblue",
    )
    ax.set_xlabel("Recurrent gain $g$", fontsize=13)
    ax.set_ylabel("Final validation loss", fontsize=13)
    ax.set_title(activation, fontsize=14)
    ax.set_xticks(gains)
    ax.grid(True, alpha=0.3)

fig.suptitle("Flip-Flop: Final Loss vs Recurrent Gain", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

# %% Load training curves
curves = []

for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
        continue
    parts = exp_name.split("_")
    if len(parts) < 3 or parts[0] != "g":
        continue
    gain = float(parts[1])
    activation = parts[2]

    for seed_dir_name in sorted(os.listdir(exp_dir)):
        if not seed_dir_name.startswith("seed_"):
            continue
        losses_file = os.path.join(exp_dir, seed_dir_name, "training_losses.json")
        if not os.path.exists(losses_file):
            continue
        with open(losses_file) as f:
            losses = json.load(f)
        curves.append({
            "gain": gain,
            "activation": activation,
            "seed": int(seed_dir_name.split("_")[1]),
            "epochs": losses.get("epochs", []),
            "val_losses": losses.get("val_losses_epoch", []),
        })

# %% Plot: training curves
import matplotlib.cm as cm
import matplotlib.colors as mcolors

all_gains = sorted(set(c["gain"] for c in curves))
norm = mcolors.Normalize(vmin=min(all_gains), vmax=max(all_gains))
cmap = cm.plasma

fig, axes = plt.subplots(1, len(activations), figsize=(7 * len(activations), 4), sharey=True)
if len(activations) == 1:
    axes = [axes]

for ax, activation in zip(axes, activations):
    sub = [c for c in curves if c["activation"] == activation]
    for c in sub:
        color = cmap(norm(c["gain"]))
        epochs = c["epochs"] if c["epochs"] else range(len(c["val_losses"]))
        ax.plot(epochs, c["val_losses"], color=color, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Validation loss", fontsize=13)
    ax.set_title(activation, fontsize=14)
    ax.grid(True, alpha=0.3)

legend_handles = [
    Line2D([0], [0], color=cmap(norm(g)), linewidth=2, label=f"g = {g}")
    for g in all_gains
]
axes[-1].legend(handles=legend_handles, title="Recurrent gain $g$",
                fontsize=10, title_fontsize=11, loc="upper right")

fig.suptitle("Flip-Flop: Validation Loss Curves by Gain", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()