# %% [markdown]
# # Teacher-Student Gain Sweep
# Final validation MSE and R² vs student recurrent gain `g`, for Identity vs Tanh activations.

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
    [d for d in os.listdir(LOGS_DIR) if d.startswith("teacher_student_gain_sweep")],
    reverse=True,
)
if not sweep_dirs:
    raise RuntimeError(f"No teacher_student_gain_sweep directories found in {LOGS_DIR}")

sweep_dir = os.path.join(LOGS_DIR, sweep_dirs[0])
print(f"Using sweep: {sweep_dir}")

# %%
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
                val_losses = losses.get("val_losses", losses.get("val_losses_epoch", []))
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

# %% Plot: final MSE loss vs g
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
    ax.set_xlabel("Student recurrent gain $g$", fontsize=13)
    ax.set_ylabel("Final validation MSE", fontsize=13)
    ax.set_title(activation, fontsize=14)
    ax.set_xticks(gains)
    ax.grid(True, alpha=0.3)

fig.suptitle("Teacher-Student: Final MSE vs Student Gain", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

# %% Compute R² from best checkpoints
import torch
import torch.nn as nn
import glob
from rnns.rnn import RNN, RNNLightning
from datamodules.teacher_student import TeacherStudentDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r2_records = []

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
        seed_path = os.path.join(exp_dir, seed_dir_name)

        config_file = os.path.join(seed_path, "run_config.yaml")
        if not os.path.exists(config_file):
            continue
        with open(config_file) as f:
            run_config = yaml.safe_load(f)

        best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
        if not best_ckpts:
            print(f"  No best checkpoint for {exp_name}/{seed_dir_name}")
            continue
        ckpt_path = best_ckpts[0]

        model = RNN(
            input_size=run_config.get("input_dim", 2),
            hidden_size=run_config["hidden_size"],
            output_size=run_config.get("output_dim", 2),
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
            task="teacher_student",
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        lit.load_state_dict(ckpt["state_dict"])
        lit.eval().to(device)

        dm = TeacherStudentDataModule(
            teacher_hidden_size=run_config["teacher_hidden_size"],
            teacher_recurrent_gain=run_config["teacher_recurrent_gain"],
            teacher_timescale=run_config["teacher_timescale"],
            teacher_activation=run_config.get("teacher_activation", "Tanh"),
            teacher_wrec_init=run_config.get("teacher_wrec_init", "normal_scaled"),
            teacher_seed=run_config.get("teacher_seed", 42),
            input_dim=run_config.get("input_dim", 2),
            output_dim=run_config.get("output_dim", 2),
            num_time_steps=run_config["num_time_steps"],
            dt=run_config["dt"],
            num_sequences=run_config.get("num_sequences", 500),
            train_val_split=run_config.get("train_val_split", 0.8),
            savgol_window=run_config.get("savgol_window", 5),
            savgol_polyorder=run_config.get("savgol_polyorder", 2),
            batch_size=run_config["batch_size"],
        )
        dm.setup()
        inputs_val, _, targets_val = dm.val_dataset.tensors
        inputs_val = inputs_val.to(device)
        targets_val = targets_val.to(device)

        with torch.no_grad():
            _, outputs = lit.model(inputs_val, init_context=None)
            mse = ((outputs - targets_val) ** 2).mean().item()
            var = targets_val.var().item()
            r2 = 1.0 - mse / (var + 1e-8)

        r2_records.append({
            "exp_name": exp_name,
            "gain": gain,
            "activation": activation,
            "seed": int(seed_dir_name.split("_")[1]),
            "mse": mse,
            "r_squared": r2,
        })
        print(f"  {exp_name}/{seed_dir_name}: MSE = {mse:.6f}, R² = {r2:.4f}")

df_r2 = pd.DataFrame(r2_records)
print(f"\nLoaded R² for {len(df_r2)} runs")
print(df_r2)

# %% Plot: R² vs g
if not df_r2.empty:
    fig, axes = plt.subplots(1, len(activations), figsize=(5 * len(activations), 4), sharey=True)
    if len(activations) == 1:
        axes = [axes]

    for ax, activation in zip(axes, activations):
        sub = df_r2[df_r2["activation"] == activation]
        grouped = sub.groupby("gain")["r_squared"]
        means = grouped.mean()
        sems = grouped.sem()

        ax.plot(means.index, means.values, marker="o", linewidth=2, color="steelblue")
        if sems.notna().any():
            ax.fill_between(
                means.index,
                means.values - sems.fillna(0).values,
                means.values + sems.fillna(0).values,
                alpha=0.25,
                color="steelblue",
            )
        ax.set_xlabel("Student recurrent gain $g$", fontsize=13)
        ax.set_ylabel("$R^2$", fontsize=13)
        ax.set_title(activation, fontsize=14)
        ax.set_xticks(gains)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5, label="chance")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle("Teacher-Student: $R^2$ vs Student Gain", fontsize=15, y=1.02)
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
            "steps": losses.get("steps", losses.get("epochs", [])),
            "val_losses": losses.get("val_losses", losses.get("val_losses_epoch", [])),
            "val_accuracies": losses.get("val_accuracies", losses.get("val_accuracies_epoch", [])),
        })

# %% Plot: training loss curves
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
        x = np.array(c["steps"] if c["steps"] else list(range(1, len(c["val_losses"]) + 1)))
        x = np.maximum(x, 1)  # avoid log(0)
        ax.plot(x, c["val_losses"], color=color, linewidth=1.5, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=13)
    ax.set_ylabel("Validation MSE", fontsize=13)
    ax.set_title(activation, fontsize=14)
    ax.grid(True, alpha=0.3, which="both")

legend_handles = [
    Line2D([0], [0], color=cmap(norm(g)), linewidth=2, label=f"g = {g}")
    for g in all_gains
]
axes[-1].legend(handles=legend_handles, title="Student gain $g$",
                fontsize=10, title_fontsize=11, loc="upper right")

fig.suptitle("Teacher-Student: Validation MSE Curves by Student Gain (log-log)", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

# %% Plot: R² training curves
has_accuracy = any(len(c.get("val_accuracies", [])) > 0 for c in curves)

if has_accuracy:
    fig, axes = plt.subplots(1, len(activations), figsize=(7 * len(activations), 4), sharey=True)
    if len(activations) == 1:
        axes = [axes]

    for ax, activation in zip(axes, activations):
        sub = [c for c in curves if c["activation"] == activation]
        for c in sub:
            val_acc = c.get("val_accuracies", [])
            if not val_acc:
                continue
            color = cmap(norm(c["gain"]))
            x = np.array(c["steps"][:len(val_acc)] if c["steps"] else list(range(1, len(val_acc) + 1)))
            x = np.maximum(x, 1)
            ax.plot(x, val_acc, color=color, linewidth=1.5, alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel("Training step", fontsize=13)
        ax.set_ylabel("$R^2$", fontsize=13)
        ax.set_title(activation, fontsize=14)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, which="both")

    legend_handles = [
        Line2D([0], [0], color=cmap(norm(g)), linewidth=2, label=f"g = {g}")
        for g in all_gains
    ]
    axes[-1].legend(handles=legend_handles, title="Student gain $g$",
                    fontsize=10, title_fontsize=11, loc="lower right")

    fig.suptitle("Teacher-Student: $R^2$ Curves by Student Gain", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("No R² data found in training_losses.json — re-run sweep to generate.")

# %% Steps to convergence vs student gain
R2_THRESHOLD = 0.99  # R² threshold defining "converged"

convergence_records = []
for c in curves:
    val_acc = c.get("val_accuracies", [])
    if not val_acc:
        continue
    steps = np.array(c["steps"][:len(val_acc)]) if c["steps"] else np.arange(1, len(val_acc) + 1)
    val_acc = np.array(val_acc)
    hit = np.where(val_acc >= R2_THRESHOLD)[0]
    steps_to_converge = int(steps[hit[0]]) if len(hit) > 0 else np.nan
    convergence_records.append({
        "gain": c["gain"],
        "activation": c["activation"],
        "seed": c["seed"],
        "steps_to_convergence": steps_to_converge,
    })

df_conv = pd.DataFrame(convergence_records)
print(f"Convergence threshold: R² ≥ {R2_THRESHOLD}")
print(df_conv)

if not df_conv.empty:
    fig, axes = plt.subplots(1, len(activations), figsize=(5 * len(activations), 4), sharey=True)
    if len(activations) == 1:
        axes = [axes]

    for ax, activation in zip(axes, activations):
        sub = df_conv[df_conv["activation"] == activation].dropna(subset=["steps_to_convergence"])
        if sub.empty:
            ax.text(0.5, 0.5, "No runs converged", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)
        else:
            grouped = sub.groupby("gain")["steps_to_convergence"]
            means = grouped.mean()
            sems = grouped.sem()

            ax.plot(means.index, means.values, marker="o", linewidth=2, color="steelblue")
            if sems.notna().any():
                ax.fill_between(
                    means.index,
                    means.values - sems.fillna(0).values,
                    means.values + sems.fillna(0).values,
                    alpha=0.25,
                    color="steelblue",
                )

            # Mark gains that never converged (if any seeds didn't)
            never = df_conv[(df_conv["activation"] == activation)
                            & df_conv["steps_to_convergence"].isna()]
            if not never.empty:
                never_gains = never["gain"].unique()
                for g in never_gains:
                    ax.axvline(g, color="red", linestyle=":", alpha=0.5)
                ax.plot([], [], color="red", linestyle=":", alpha=0.5, label="did not converge")
                ax.legend(fontsize=10)

        ax.set_xlabel("Student recurrent gain $g$", fontsize=13)
        ax.set_ylabel(f"Steps to $R^2 \\geq {R2_THRESHOLD}$", fontsize=13)
        ax.set_title(activation, fontsize=14)
        ax.set_xticks(gains)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Teacher-Student: Convergence Speed ($R^2 \\geq {R2_THRESHOLD}$)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("No R² curves available — cannot compute convergence.")

# %% Per-timestep MSE across the sequence (log-log)
# Shows how prediction error evolves within a sequence for each gain/activation.
if not df_r2.empty:
    fig, axes = plt.subplots(1, len(activations), figsize=(7 * len(activations), 4), sharey=True)
    if len(activations) == 1:
        axes = [axes]

    for ax, activation in zip(axes, activations):
        sub = df_r2[df_r2["activation"] == activation]
        for _, row in sub.iterrows():
            seed_path = os.path.join(sweep_dir, row["exp_name"], f"seed_{int(row['seed'])}")
            config_file = os.path.join(seed_path, "run_config.yaml")
            with open(config_file) as f:
                run_config = yaml.safe_load(f)

            best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
            if not best_ckpts:
                continue
            ckpt_path = best_ckpts[0]

            model = RNN(
                input_size=run_config.get("input_dim", 2),
                hidden_size=run_config["hidden_size"],
                output_size=run_config.get("output_dim", 2),
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
                task="teacher_student",
            )
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            lit.load_state_dict(ckpt["state_dict"])
            lit.eval().to(device)

            dm = TeacherStudentDataModule(
                teacher_hidden_size=run_config["teacher_hidden_size"],
                teacher_recurrent_gain=run_config["teacher_recurrent_gain"],
                teacher_timescale=run_config["teacher_timescale"],
                teacher_activation=run_config.get("teacher_activation", "Tanh"),
                teacher_wrec_init=run_config.get("teacher_wrec_init", "normal_scaled"),
                teacher_seed=run_config.get("teacher_seed", 42),
                input_dim=run_config.get("input_dim", 2),
                output_dim=run_config.get("output_dim", 2),
                num_time_steps=run_config["num_time_steps"],
                dt=run_config["dt"],
                num_sequences=run_config.get("num_sequences", 500),
                train_val_split=run_config.get("train_val_split", 0.8),
                savgol_window=run_config.get("savgol_window", 5),
                savgol_polyorder=run_config.get("savgol_polyorder", 2),
                batch_size=run_config["batch_size"],
            )
            dm.setup()
            inputs_val, _, targets_val = dm.val_dataset.tensors
            inputs_val = inputs_val.to(device)
            targets_val = targets_val.to(device)

            with torch.no_grad():
                _, outputs = lit.model(inputs_val, init_context=None)
                # MSE per timestep, averaged over batch and output dims
                per_t_mse = ((outputs - targets_val) ** 2).mean(dim=(0, 2)).cpu().numpy()

            color = cmap(norm(row["gain"]))
            timesteps = np.arange(1, len(per_t_mse) + 1)
            ax.plot(timesteps, per_t_mse, color=color, linewidth=1.5, alpha=0.85)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Timestep $t$", fontsize=13)
        ax.set_ylabel("MSE at timestep $t$", fontsize=13)
        ax.set_title(activation, fontsize=14)
        ax.grid(True, alpha=0.3, which="both")

    legend_handles = [
        Line2D([0], [0], color=cmap(norm(g)), linewidth=2, label=f"g = {g}")
        for g in all_gains
    ]
    axes[-1].legend(handles=legend_handles, title="Student gain $g$",
                    fontsize=10, title_fontsize=11, loc="upper left")

    fig.suptitle("Teacher-Student: Per-Timestep MSE (log-log)", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()

# %% Output time series comparison (best run per activation)
if not df_r2.empty:
    fig, axes = plt.subplots(len(activations), 2, figsize=(14, 4 * len(activations)),
                              sharex=True)
    if len(activations) == 1:
        axes = axes.reshape(1, -1)

    for row, activation in enumerate(activations):
        sub = df_r2[df_r2["activation"] == activation]
        best_row = sub.loc[sub["r_squared"].idxmax()]
        best_exp = best_row["exp_name"]
        best_seed = int(best_row["seed"])

        seed_path = os.path.join(sweep_dir, best_exp, f"seed_{best_seed}")
        config_file = os.path.join(seed_path, "run_config.yaml")
        with open(config_file) as f:
            run_config = yaml.safe_load(f)

        best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
        ckpt_path = best_ckpts[0]

        model = RNN(
            input_size=run_config.get("input_dim", 2),
            hidden_size=run_config["hidden_size"],
            output_size=run_config.get("output_dim", 2),
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
            task="teacher_student",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        lit.load_state_dict(ckpt["state_dict"])
        lit.eval()

        dm = TeacherStudentDataModule(
            teacher_hidden_size=run_config["teacher_hidden_size"],
            teacher_recurrent_gain=run_config["teacher_recurrent_gain"],
            teacher_timescale=run_config["teacher_timescale"],
            teacher_activation=run_config.get("teacher_activation", "Tanh"),
            teacher_wrec_init=run_config.get("teacher_wrec_init", "normal_scaled"),
            teacher_seed=run_config.get("teacher_seed", 42),
            input_dim=run_config.get("input_dim", 2),
            output_dim=run_config.get("output_dim", 2),
            num_time_steps=run_config["num_time_steps"],
            dt=run_config["dt"],
            num_sequences=run_config.get("num_sequences", 500),
            train_val_split=run_config.get("train_val_split", 0.8),
            savgol_window=run_config.get("savgol_window", 5),
            savgol_polyorder=run_config.get("savgol_polyorder", 2),
            batch_size=run_config["batch_size"],
        )
        dm.setup()
        inputs_val, _, targets_val = dm.val_dataset.tensors

        with torch.no_grad():
            _, outputs = lit.model(inputs_val, init_context=None)

        # Plot first validation sequence, both output channels
        seq_idx = 0
        t = np.arange(run_config["num_time_steps"])
        for ch in range(2):
            ax = axes[row, ch]
            ax.plot(t, targets_val[seq_idx, :, ch].numpy(), "k-", linewidth=2, label="Teacher")
            ax.plot(t, outputs[seq_idx, :, ch].numpy(), "--", linewidth=2, color="steelblue",
                    label="Student")
            ax.set_ylabel(f"Output ch {ch}", fontsize=12)
            ax.set_title(f"{activation} (g={best_row['gain']}, R²={best_row['r_squared']:.3f})",
                         fontsize=12)
            ax.grid(True, alpha=0.3)
            if row == 0 and ch == 0:
                ax.legend(fontsize=10)

        axes[row, 0].set_xlabel("Time step", fontsize=12)
        axes[row, 1].set_xlabel("Time step", fontsize=12)

    fig.suptitle("Teacher-Student: Output Comparison (Best Run per Activation)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()
