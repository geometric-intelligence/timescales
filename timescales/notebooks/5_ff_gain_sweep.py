# %% [markdown]
# # Flip-Flop Gain Sweep
# Final validation loss vs recurrent gain `g`, for Identity vs Tanh activations.
# With low pulse frequency (p_pulse=0.01, avg hold ~100 steps) to stress-test
# linear vs nonlinear memory maintenance.

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
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

fig.suptitle("Flip-Flop: Final Loss vs Recurrent Gain", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

# %% Compute bit accuracy from best checkpoints
import torch
import torch.nn as nn
import glob
from rnns.multitimescale_rnn import MultiTimescaleRNN, MultiTimescaleRNNLightning
from datamodules.flip_flop import FlipFlopDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accuracy_records = []

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

        n_bits = run_config["n_bits"]
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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        lit.load_state_dict(ckpt["state_dict"])
        lit.eval().to(device)

        dm = FlipFlopDataModule(
            n_bits=n_bits,
            p_pulse=run_config["p_pulse"],
            pulse_amplitude=run_config["pulse_amplitude"],
            num_time_steps=run_config["num_time_steps"],
            num_trajectories=500,
            batch_size=500,
        )
        dm.setup()
        inputs_val, _, targets_val = dm.val_dataset.tensors
        inputs_val = inputs_val.to(device)
        targets_val = targets_val.to(device)

        with torch.no_grad():
            _, outputs = lit.model(inputs_val, init_context=None)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            bit_acc = (preds == targets_val).float().mean().item()

        accuracy_records.append({
            "exp_name": exp_name,
            "gain": gain,
            "activation": activation,
            "seed": int(seed_dir_name.split("_")[1]),
            "bit_accuracy": bit_acc,
        })
        print(f"  {exp_name}/{seed_dir_name}: bit accuracy = {bit_acc:.4f}")

df_acc = pd.DataFrame(accuracy_records)
print(f"\nLoaded accuracy for {len(df_acc)} runs")
print(df_acc)

# %% Plot: bit accuracy vs g
if not df_acc.empty:
    fig, axes = plt.subplots(1, len(activations), figsize=(5 * len(activations), 4), sharey=True)
    if len(activations) == 1:
        axes = [axes]

    for ax, activation in zip(axes, activations):
        sub = df_acc[df_acc["activation"] == activation]
        grouped = sub.groupby("gain")["bit_accuracy"]
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
        ax.set_xlabel("Recurrent gain $g$", fontsize=13)
        ax.set_ylabel("Bit accuracy", fontsize=13)
        ax.set_title(activation, fontsize=14)
        ax.set_xticks(gains)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle("Flip-Flop: Bit Accuracy vs Recurrent Gain", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()

# %% Example trajectories: input pulses, ground truth, network output
# Pick a few representative gains per activation to visually inspect behavior.

EXAMPLE_GAINS = [min(gains), gains[len(gains)//2], 0.8, 0.9, 0.95, max(gains)]
SEQ_IDX = 0  # which validation trajectory to plot

for activation in activations:
    sub = df_acc[df_acc["activation"] == activation]
    show_gains = [g for g in EXAMPLE_GAINS if g in sub["gain"].values]
    if not show_gains:
        continue

    n_bits_for_plot = None
    fig = None

    for col_idx, g in enumerate(show_gains):
        row = sub[sub["gain"] == g].iloc[0]
        seed_path = os.path.join(sweep_dir, row["exp_name"], f"seed_{int(row['seed'])}")
        config_file = os.path.join(seed_path, "run_config.yaml")
        with open(config_file) as f:
            run_config = yaml.safe_load(f)

        best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
        if not best_ckpts:
            continue
        ckpt_path = best_ckpts[0]

        n_bits = run_config["n_bits"]
        if n_bits_for_plot is None:
            n_bits_for_plot = n_bits
            fig, axes = plt.subplots(
                n_bits, len(show_gains),
                figsize=(5 * len(show_gains), 2 * n_bits),
                sharex=True, sharey=True,
            )
            if n_bits == 1:
                axes = axes.reshape(1, -1)
            if len(show_gains) == 1:
                axes = axes.reshape(-1, 1)

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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        lit.load_state_dict(ckpt["state_dict"])
        lit.eval().to(device)

        dm = FlipFlopDataModule(
            n_bits=n_bits,
            p_pulse=run_config["p_pulse"],
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

        t = np.arange(run_config["num_time_steps"])
        for bit in range(n_bits):
            ax = axes[bit, col_idx]
            pulse = inp[SEQ_IDX, :, bit].numpy()
            # Show input as small ticks centered at 0.5: +1→▲ at 0.65, -1→▼ at 0.35
            set_mask = pulse > 0.5
            reset_mask = pulse < -0.5
            if set_mask.any():
                ax.scatter(t[set_mask], np.full(set_mask.sum(), 0.65),
                           marker=6, s=25, color="C2", zorder=3,
                           label="set (+1)" if bit == 0 else None)
            if reset_mask.any():
                ax.scatter(t[reset_mask], np.full(reset_mask.sum(), 0.35),
                           marker=7, s=25, color="C3", zorder=3,
                           label="reset (-1)" if bit == 0 else None)
            ax.step(t, tgt[SEQ_IDX, :, bit].numpy(), where="post",
                    color="black", linewidth=1.5, label="target" if bit == 0 else None)
            ax.plot(t, out_prob[SEQ_IDX, :, bit].numpy(),
                    color="steelblue", linewidth=1.2, alpha=0.9,
                    label="output" if bit == 0 else None)
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 0.5, 1])
            if bit == 0:
                ax.set_title(f"g = {g} (acc={row['bit_accuracy']:.2f})", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"Bit {bit}", fontsize=11)
            if bit == n_bits - 1:
                ax.set_xlabel("Timestep", fontsize=11)

    if fig is not None:
        axes[0, 0].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Flip-Flop Trajectories — {activation}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

# %% Jacobian eigenvalue spectrum for Identity networks (untrained vs trained)
# For Identity activation, J = (1-α)I + α·g·W_rec is constant (linear system).
# Layout: rows = gain values, columns = [untrained, trained].
# Color: untrained in muted gray-blue, trained colored by |λ| to highlight slow modes.

identity_runs = df_acc[df_acc["activation"] == "Identity"].copy()

if not identity_runs.empty:
    id_gains = sorted(identity_runs["gain"].unique())
    n_gains = len(id_gains)
    theta = np.linspace(0, 2 * np.pi, 200)

    UNTRAINED_COLOR = "#8da0b5"
    TRAINED_COLOR = "#e76f51"

    # --- Complex plane: rows=gains, cols=[untrained, trained] ---
    fig, axes = plt.subplots(n_gains, 2, figsize=(7.5, 3.2 * n_gains),
                              squeeze=False)

    for row_idx, g in enumerate(id_gains):
        row_data = identity_runs[identity_runs["gain"] == g].iloc[0]
        seed_path = os.path.join(sweep_dir, row_data["exp_name"], f"seed_{int(row_data['seed'])}")
        config_file = os.path.join(seed_path, "run_config.yaml")
        with open(config_file) as f:
            run_config = yaml.safe_load(f)

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

            N = W_rec.shape[0]
            J = (1.0 - alpha) * np.eye(N) + alpha * g * W_rec
            eigs = np.linalg.eigvals(J)
            abs_eigs = np.abs(eigs)
            max_abs = np.max(abs_eigs)

            ax.plot(np.cos(theta), np.sin(theta), color="#c0392b",
                    linewidth=1.2, alpha=0.5, linestyle="--", zorder=1,
                    label="$|\\lambda|=1$ (slow)")
            ax.axhline(0, color="#bbb", linewidth=0.3, zorder=0)
            ax.axvline(0, color="#bbb", linewidth=0.3, zorder=0)

            top3_idx = np.argsort(abs_eigs)[-3:]
            rest_idx = np.argsort(abs_eigs)[:-3]
            pt_color = UNTRAINED_COLOR if col_idx == 0 else TRAINED_COLOR

            ax.scatter(eigs.real[rest_idx], eigs.imag[rest_idx], s=18, alpha=0.7,
                       c=pt_color, edgecolors="none", zorder=3)
            if col_idx == 1:
                ax.scatter(eigs.real[top3_idx], eigs.imag[top3_idx], s=18, alpha=0.95,
                           c=TRAINED_COLOR, edgecolors="black", linewidths=0.4, zorder=4, label="Top 3")

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

    fig.suptitle("Jacobian Eigenvalue Spectrum — Identity Networks",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

    # --- Scree plot: ordered |λ| showing the 3-vs-rest gap ---
    fig, axes = plt.subplots(1, len(id_gains), figsize=(3.2 * len(id_gains), 3),
                              squeeze=False, sharey=True)

    for col, g in enumerate(id_gains):
        ax = axes[0, col]
        row_data = identity_runs[identity_runs["gain"] == g].iloc[0]
        seed_path = os.path.join(sweep_dir, row_data["exp_name"], f"seed_{int(row_data['seed'])}")
        config_file = os.path.join(seed_path, "run_config.yaml")
        with open(config_file) as f:
            run_config = yaml.safe_load(f)
        dt = run_config["dt"]
        tau = run_config["timescales_config"]["values"][0]
        alpha = 1.0 - np.exp(-dt / tau)

        best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
        if not best_ckpts:
            continue
        ckpt = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)
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
        abs_sorted = np.sort(np.abs(eigs))[::-1]
        ranks = np.arange(1, N + 1)

        ax.scatter(ranks[:3], abs_sorted[:3], s=50, color=TRAINED_COLOR,
                   edgecolors="none", zorder=3, label="Top 3 (slow modes)")
        ax.scatter(ranks[3:], abs_sorted[3:], s=30, color=UNTRAINED_COLOR,
                   edgecolors="none", zorder=3, label="Remaining")
        ax.plot(ranks, abs_sorted, color="#999", linewidth=0.8, zorder=2)

        ax.set_xlabel("Rank", fontsize=11)
        if col == 0:
            ax.set_ylabel("|$\\lambda$|", fontsize=11)
        ax.set_title(f"g = {g}", fontsize=11)
        ax.grid(True, alpha=0.15)
        if col == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    fig.suptitle("Eigenvalue Magnitudes (ranked) — 3 slow modes per 3 flip-flop bits",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()

    # --- Print top-3 eigenvalues and eigenvectors ---
    for g in id_gains:
        row_data = identity_runs[identity_runs["gain"] == g].iloc[0]
        seed_path = os.path.join(sweep_dir, row_data["exp_name"], f"seed_{int(row_data['seed'])}")
        config_file = os.path.join(seed_path, "run_config.yaml")
        with open(config_file) as f:
            run_config = yaml.safe_load(f)
        dt = run_config["dt"]
        tau = run_config["timescales_config"]["values"][0]
        alpha = 1.0 - np.exp(-dt / tau)

        best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
        if not best_ckpts:
            continue
        ckpt = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)
        W_rec = None
        for key, val in ckpt["state_dict"].items():
            if "W_rec.weight" in key:
                W_rec = val.numpy()
                break
        if W_rec is None:
            continue

        N = W_rec.shape[0]
        J = (1.0 - alpha) * np.eye(N) + alpha * g * W_rec
        eigenvalues, eigvecs = np.linalg.eig(J)
        top3 = np.argsort(np.abs(eigenvalues))[-3:][::-1]

        print(f"\n{'='*60}")
        print(f"g = {g}  |  Top 3 eigenvalues of trained Identity network")
        print(f"{'='*60}")
        for rank, idx in enumerate(top3):
            ev = eigenvalues[idx]
            vec = eigvecs[:, idx]
            is_real = np.max(np.abs(vec.imag)) < 1e-10
            print(f"\n  λ_{rank+1}: {ev:.6f}  (|λ| = {np.abs(ev):.6f})")
            print(f"    purely real: {is_real}")
            if not is_real:
                print(f"    Re(λ) = {ev.real:.6f}, Im(λ) = {ev.imag:.6f}")
                print(f"    max |Im(v)| = {np.max(np.abs(vec.imag)):.6f}")
            print(f"    eigvec norm = {np.linalg.norm(vec):.4f}")
            print(f"    top-5 |components|: {np.sort(np.abs(vec))[-5:][::-1].real}")

    # --- Effective timescale scree plot: τ_eff = -1/log|λ| ---
    fig, axes = plt.subplots(1, len(id_gains), figsize=(3.2 * len(id_gains), 3),
                              squeeze=False, sharey=True)

    for col, g in enumerate(id_gains):
        ax = axes[0, col]
        row_data = identity_runs[identity_runs["gain"] == g].iloc[0]
        seed_path = os.path.join(sweep_dir, row_data["exp_name"], f"seed_{int(row_data['seed'])}")
        config_file = os.path.join(seed_path, "run_config.yaml")
        with open(config_file) as f:
            run_config = yaml.safe_load(f)
        dt = run_config["dt"]
        tau = run_config["timescales_config"]["values"][0]
        alpha = 1.0 - np.exp(-dt / tau)

        best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
        if not best_ckpts:
            continue
        ckpt = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)
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
        abs_sorted = np.sort(np.abs(eigs))[::-1]
        ranks = np.arange(1, N + 1)

        log_abs = np.log(np.clip(abs_sorted, 1e-12, None))
        tau_eff = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)

        ax.scatter(ranks[:3], tau_eff[:3], s=50, color=TRAINED_COLOR,
                   edgecolors="none", zorder=3, label="Top 3 (slow modes)")
        ax.scatter(ranks[3:], tau_eff[3:], s=30, color=UNTRAINED_COLOR,
                   edgecolors="none", zorder=3, label="Remaining")

        #ax.set_yscale("log")
        ax.set_xlabel("Rank", fontsize=11)
        if col == 0:
            ax.set_ylabel("$\\tau_{\\mathrm{eff}} = -1\\,/\\,\\ln|\\lambda|$\n(steps)",
                          fontsize=10)
        ax.set_title(f"g = {g}", fontsize=11)
        ax.grid(True, alpha=0.15, which="both")
        if col == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    fig.suptitle("Effective Timescales (ranked) — 3 slow modes per 3 flip-flop bits",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()

    # --- |λ| histograms: rows=gains, cols=[untrained, trained] ---
    fig, axes = plt.subplots(n_gains, 2, figsize=(7.5, 2.8 * n_gains),
                              squeeze=False, sharey="row")

    for row_idx, g in enumerate(id_gains):
        row_data = identity_runs[identity_runs["gain"] == g].iloc[0]
        seed_path = os.path.join(sweep_dir, row_data["exp_name"], f"seed_{int(row_data['seed'])}")
        config_file = os.path.join(seed_path, "run_config.yaml")
        with open(config_file) as f:
            run_config = yaml.safe_load(f)

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

            hist_color = UNTRAINED_COLOR if col_idx == 0 else "#e07b39"
            ax.hist(abs_eigs, bins=30, color=hist_color, alpha=0.75,
                    edgecolor="white", linewidth=0.5)
            ax.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.2,
                       alpha=0.8, label="|$\\lambda$|=1")
            ax.grid(True, alpha=0.15)

            if row_idx == 0:
                ax.set_title(label, fontsize=13, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"g = {g}\nCount", fontsize=11)
            if row_idx == n_gains - 1:
                ax.set_xlabel("|$\\lambda$|", fontsize=11)
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Eigenvalue Magnitudes — Identity Networks",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

# %% Hidden-state trajectories projected onto the 3 slow eigenvectors
# Rows = bits (colored by target value: 0 or 1). Columns = gain values.

if not identity_runs.empty:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    id_gains = sorted(identity_runs["gain"].unique())
    show_gains_proj = [g for g in id_gains if g <= 0.99]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_PROJ_TRAJS = 50
    SHARE_SCALE = True
    BIT_COLORS = {0: "#3b82f6", 1: "#ef4444"}

    proj_data = {}

    for g in show_gains_proj:
        row_data = identity_runs[identity_runs["gain"] == g].iloc[0]
        seed_path = os.path.join(sweep_dir, row_data["exp_name"], f"seed_{int(row_data['seed'])}")
        config_file = os.path.join(seed_path, "run_config.yaml")
        with open(config_file) as f:
            run_config = yaml.safe_load(f)

        best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
        if not best_ckpts:
            continue

        n_bits = run_config["n_bits"]
        dt = run_config["dt"]
        tau = run_config["timescales_config"]["values"][0]
        alpha = 1.0 - np.exp(-dt / tau)

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
            p_pulse=run_config["p_pulse"],
            pulse_amplitude=run_config["pulse_amplitude"],
            num_time_steps=run_config["num_time_steps"],
            num_val_trajectories=N_PROJ_TRAJS,
            batch_size=N_PROJ_TRAJS,
        )
        dm.setup()
        inp, _, tgt = dm.val_dataset.tensors

        with torch.no_grad():
            hidden_states, _ = lit.model(inp.to(device), init_context=None)
            hidden_np = hidden_states.cpu().numpy()

        W_rec = None
        for key, val in ckpt["state_dict"].items():
            if "W_rec.weight" in key:
                W_rec = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
                break
        N = W_rec.shape[0]
        J = (1.0 - alpha) * np.eye(N) + alpha * g * W_rec

        eigenvalues, V_right = np.linalg.eig(J)
        V_left = np.linalg.inv(V_right)
        top3_idx = np.argsort(np.abs(eigenvalues))[-3:][::-1]
        W_proj = V_left[top3_idx, :].real

        all_projected = np.stack([hidden_np[i] @ W_proj.T for i in range(N_PROJ_TRAJS)])
        tgt_np = tgt.numpy()
        eig_labels = [f"|λ|={np.abs(eigenvalues[i]):.3f}" for i in top3_idx]
        proj_data[g] = dict(projected=all_projected, targets=tgt_np,
                            eig_labels=eig_labels, n_bits=n_bits)

    if proj_data:
        n_bits_proj = list(proj_data.values())[0]["n_bits"]
        n_cols = len(proj_data)

        global_lim = None
        if SHARE_SCALE:
            all_vals = np.concatenate([d["projected"].ravel() for d in proj_data.values()])
            global_lim = 10#np.max(np.abs(all_vals)) * 1.05

        fig = plt.figure(figsize=(4.5 * n_cols, 4.5 * n_bits_proj))

        for col_idx, (g, data) in enumerate(sorted(proj_data.items())):
            all_proj = data["projected"]
            tgt_np = data["targets"]

            for bit in range(n_bits_proj):
                ax_idx = bit * n_cols + col_idx + 1
                ax = fig.add_subplot(n_bits_proj, n_cols, ax_idx, projection="3d")

                pts = all_proj.reshape(-1, 3)
                bit_vals = tgt_np[:, :, bit].ravel()
                mask0 = bit_vals < 0.5
                mask1 = ~mask0
                if mask0.any():
                    ax.scatter(pts[mask0, 0], pts[mask0, 1], pts[mask0, 2],
                               s=1, alpha=0.15, c=BIT_COLORS[0], depthshade=False)
                if mask1.any():
                    ax.scatter(pts[mask1, 0], pts[mask1, 1], pts[mask1, 2],
                               s=1, alpha=0.15, c=BIT_COLORS[1], depthshade=False)

                if global_lim is not None:
                    ax.set_xlim(-global_lim, global_lim)
                    ax.set_ylim(-global_lim, global_lim)
                    ax.set_zlim(-global_lim, global_lim)

                if bit == 0:
                    ax.set_title(f"g = {g}", fontsize=12)
                if col_idx == 0:
                    ax.set_ylabel(f"Bit {bit}\nMode 2", fontsize=9)
                else:
                    ax.set_ylabel("Mode 2", fontsize=8)
                ax.set_xlabel("Mode 1", fontsize=8)
                ax.set_zlabel("Mode 3", fontsize=8)
                ax.tick_params(labelsize=6)

        from matplotlib.lines import Line2D
        legend_els = [Line2D([0], [0], color=BIT_COLORS[0], lw=2, label="target = 0"),
                      Line2D([0], [0], color=BIT_COLORS[1], lw=2, label="target = 1")]
        fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=10,
                   framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle("Slow-Mode Projections Colored by Target Bit — Identity Networks",
                     fontsize=14, fontweight="bold", y=1.0)
        plt.tight_layout()
        plt.show()



### ----------------Tanh networks------------------- ###



# %% Jacobian analysis for Tanh networks (linearized at empirical operating point)
# J(r*) = (1-α)I + α · diag(sech²(g·W_rec·r*)) · g·W_rec
# We evaluate at the mean hidden state during "hold" periods (no recent pulse).

tanh_runs = df_acc[df_acc["activation"] == "Tanh"].copy()

if not tanh_runs.empty:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    tanh_gains = sorted(tanh_runs["gain"].unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_PROJ_TRAJS_TANH = 50
    theta = np.linspace(0, 2 * np.pi, 200)
    UNTRAINED_COLOR = "#8da0b5"
    TRAINED_COLOR = "#e76f51"
    BIT_COLORS = {0: "#3b82f6", 1: "#ef4444"}

    tanh_eig_data = {}

    for g in tanh_gains:
        row_data = tanh_runs[tanh_runs["gain"] == g].iloc[0]
        seed_path = os.path.join(sweep_dir, row_data["exp_name"], f"seed_{int(row_data['seed'])}")
        config_file = os.path.join(seed_path, "run_config.yaml")
        with open(config_file) as f:
            run_config = yaml.safe_load(f)

        best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
        untrained_ckpt = os.path.join(seed_path, "checkpoints", "untrained.ckpt")
        if not best_ckpts:
            continue

        n_bits = run_config["n_bits"]
        dt = run_config["dt"]
        tau = run_config["timescales_config"]["values"][0]
        alpha = 1.0 - np.exp(-dt / tau)

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

        # --- Trained: run trajectories to find operating point ---
        ckpt = torch.load(best_ckpts[0], map_location=device, weights_only=False)
        lit.load_state_dict(ckpt["state_dict"])
        lit.eval().to(device)

        dm = FlipFlopDataModule(
            n_bits=n_bits,
            p_pulse=run_config["p_pulse"],
            pulse_amplitude=run_config["pulse_amplitude"],
            num_time_steps=run_config["num_time_steps"],
            num_val_trajectories=N_PROJ_TRAJS_TANH,
            batch_size=N_PROJ_TRAJS_TANH,
        )
        dm.setup()
        inp, _, tgt = dm.val_dataset.tensors
        inp_dev = inp.to(device)

        with torch.no_grad():
            hidden_states, _ = lit.model(inp_dev, init_context=None)
            hidden_np = hidden_states.cpu().numpy()

        # Find "hold" timesteps: no pulse at current time
        pulse_mask = np.abs(inp.numpy()) > 0.5
        any_pulse = pulse_mask.any(axis=-1)
        hold_mask = ~any_pulse
        hold_hidden = hidden_np[hold_mask]
        r_star = hold_hidden.mean(axis=0)

        # Extract W_rec
        W_rec_trained = None
        for key, val in ckpt["state_dict"].items():
            if "W_rec.weight" in key:
                W_rec_trained = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
                break

        N = W_rec_trained.shape[0]
        pre_act = g * W_rec_trained @ r_star
        sech2 = 1.0 - np.tanh(pre_act) ** 2
        J_trained = (1.0 - alpha) * np.eye(N) + alpha * np.diag(sech2) @ (g * W_rec_trained)
        eigs_trained, V_right_trained = np.linalg.eig(J_trained)
        V_left_trained = np.linalg.inv(V_right_trained)
        top3_trained = np.argsort(np.abs(eigs_trained))[-3:][::-1]

        # --- Untrained: Jacobian at r*=0 (sech²=1, same as linear) ---
        eigs_untrained = None
        if os.path.exists(untrained_ckpt):
            ckpt_u = torch.load(untrained_ckpt, map_location="cpu", weights_only=False)
            W_rec_untrained = None
            for key, val in ckpt_u["state_dict"].items():
                if "W_rec.weight" in key:
                    W_rec_untrained = val.numpy()
                    break
            if W_rec_untrained is not None:
                J_untrained = (1.0 - alpha) * np.eye(N) + alpha * g * W_rec_untrained
                eigs_untrained = np.linalg.eigvals(J_untrained)

        # Project hidden states onto trained top-3 slow modes
        W_proj = V_left_trained[top3_trained, :].real
        all_projected = np.stack([hidden_np[i] @ W_proj.T for i in range(N_PROJ_TRAJS_TANH)])

        tanh_eig_data[g] = dict(
            eigs_trained=eigs_trained, eigs_untrained=eigs_untrained,
            top3_idx=top3_trained, projected=all_projected,
            targets=tgt.numpy(), inputs=inp.numpy(), n_bits=n_bits,
            eig_labels=[f"|λ|={np.abs(eigs_trained[i]):.3f}" for i in top3_trained],
        )

    # --- Complex plane: untrained (left) vs trained at operating point (right) ---
    if tanh_eig_data:
        n_tanh = len(tanh_eig_data)
        fig, axes = plt.subplots(n_tanh, 2, figsize=(7.5, 3.2 * n_tanh), squeeze=False)

        for row_idx, (g, data) in enumerate(sorted(tanh_eig_data.items())):
            for col_idx, (label, eigs) in enumerate([
                ("Untrained (J at r=0)", data["eigs_untrained"]),
                ("Trained (J at mean hold state)", data["eigs_trained"]),
            ]):
                ax = axes[row_idx, col_idx]
                if eigs is None:
                    ax.text(0.5, 0.5, "no ckpt", transform=ax.transAxes, ha="center")
                    continue

                abs_eigs = np.abs(eigs)
                max_abs = np.max(abs_eigs)

                ax.plot(np.cos(theta), np.sin(theta), color="#c0392b",
                        linewidth=1.2, alpha=0.5, linestyle="--", zorder=1,
                        label="$|\\lambda|=1$ (slow)")
                ax.axhline(0, color="#bbb", linewidth=0.3, zorder=0)
                ax.axvline(0, color="#bbb", linewidth=0.3, zorder=0)

                pt_color = UNTRAINED_COLOR if col_idx == 0 else TRAINED_COLOR
                ax.scatter(eigs.real, eigs.imag, s=18, alpha=0.8,
                           c=pt_color, edgecolors="none", zorder=3)

                ax.set_aspect("equal")
                ax.grid(True, alpha=0.15)

                if row_idx == 0:
                    ax.set_title(label, fontsize=11, fontweight="bold")
                ax.annotate(f"|$\\lambda$|$_{{max}}$={max_abs:.3f}",
                            xy=(0.97, 0.95), xycoords="axes fraction",
                            ha="right", va="top", fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
                if col_idx == 0:
                    ax.set_ylabel(f"g = {g}\nIm($\\lambda$)", fontsize=11)
                if row_idx == n_tanh - 1:
                    ax.set_xlabel("Re($\\lambda$)", fontsize=11)

        axes[0, 0].legend(fontsize=8, loc="lower left", framealpha=0.7)
        fig.suptitle("Jacobian Eigenvalue Spectrum — Tanh Networks",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.show()

        # --- Effective timescale scree plot for Tanh ---
        fig, axes = plt.subplots(1, n_tanh, figsize=(3.2 * n_tanh, 3),
                                  squeeze=False, sharey=True)
        for col, (g, data) in enumerate(sorted(tanh_eig_data.items())):
            ax = axes[0, col]
            abs_sorted = np.sort(np.abs(data["eigs_trained"]))[::-1]
            ranks = np.arange(1, len(abs_sorted) + 1)

            log_abs = np.log(np.clip(abs_sorted, 1e-12, None))
            tau_eff = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)

            ax.scatter(ranks[:3], tau_eff[:3], s=50, color=TRAINED_COLOR,
                       edgecolors="none", zorder=3, label="Top 3 (slow modes)")
            ax.scatter(ranks[3:], tau_eff[3:], s=30, color=UNTRAINED_COLOR,
                       edgecolors="none", zorder=3, label="Remaining")

            ax.set_yscale("log")
            ax.set_xlabel("Rank", fontsize=11)
            if col == 0:
                ax.set_ylabel("$\\tau_{\\mathrm{eff}} = -1\\,/\\,\\ln|\\lambda|$\n(steps)",
                              fontsize=10)
            ax.set_title(f"g = {g}", fontsize=11)
            ax.grid(True, alpha=0.15, which="both")
            if col == 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

        fig.suptitle("Effective Timescales (ranked) — Tanh Networks (at operating point)",
                     fontsize=13, fontweight="bold", y=1.03)
        plt.tight_layout()
        plt.show()

        # --- Slow-mode projections colored by target bit ---
        n_bits_proj = list(tanh_eig_data.values())[0]["n_bits"]
        n_cols = len(tanh_eig_data)
        SHARE_SCALE_TANH = True
        HOLD_ONLY = True
        PULSE_MARGIN = 10  # exclude this many timesteps after each pulse

        global_lim = None
        if SHARE_SCALE_TANH:
            all_vals = np.concatenate([d["projected"].ravel() for d in tanh_eig_data.values()])
            global_lim = np.max(np.abs(all_vals)) * 1.05

        fig = plt.figure(figsize=(4.5 * n_cols, 4.5 * n_bits_proj))

        for col_idx, (g, data) in enumerate(sorted(tanh_eig_data.items())):
            all_proj = data["projected"]
            tgt_np = data["targets"]
            inp_np = data["inputs"]

            # Build hold mask: True at timesteps far from any pulse
            if HOLD_ONLY:
                any_pulse = (np.abs(inp_np) > 0.5).any(axis=-1)
                near_pulse = np.zeros_like(any_pulse)
                n_seq, n_t = any_pulse.shape
                for dt_offset in range(PULSE_MARGIN + 1):
                    shifted = np.zeros_like(any_pulse)
                    if dt_offset <= n_t - 1:
                        shifted[:, dt_offset:] = any_pulse[:, :n_t - dt_offset]
                    near_pulse |= shifted
                hold_mask_flat = (~near_pulse).ravel()
            else:
                hold_mask_flat = np.ones(all_proj.shape[0] * all_proj.shape[1], dtype=bool)

            for bit in range(n_bits_proj):
                ax_idx = bit * n_cols + col_idx + 1
                ax = fig.add_subplot(n_bits_proj, n_cols, ax_idx, projection="3d")

                pts = all_proj.reshape(-1, 3)
                bit_vals = tgt_np[:, :, bit].ravel()

                pts_f = pts[hold_mask_flat]
                bits_f = bit_vals[hold_mask_flat]
                mask0 = bits_f < 0.5
                mask1 = ~mask0
                if mask0.any():
                    ax.scatter(pts_f[mask0, 0], pts_f[mask0, 1], pts_f[mask0, 2],
                               s=1, alpha=0.15, c=BIT_COLORS[0], depthshade=False)
                if mask1.any():
                    ax.scatter(pts_f[mask1, 0], pts_f[mask1, 1], pts_f[mask1, 2],
                               s=1, alpha=0.15, c=BIT_COLORS[1], depthshade=False)

                if global_lim is not None:
                    ax.set_xlim(-global_lim, global_lim)
                    ax.set_ylim(-global_lim, global_lim)
                    ax.set_zlim(-global_lim, global_lim)

                if bit == 0:
                    ax.set_title(f"g = {g}", fontsize=12)
                if col_idx == 0:
                    ax.set_ylabel(f"Bit {bit}\nMode 2", fontsize=9)
                else:
                    ax.set_ylabel("Mode 2", fontsize=8)
                ax.set_xlabel("Mode 1", fontsize=8)
                ax.set_zlabel("Mode 3", fontsize=8)
                ax.tick_params(labelsize=6)

        from matplotlib.lines import Line2D
        legend_els = [Line2D([0], [0], color=BIT_COLORS[0], lw=2, label="target = 0"),
                      Line2D([0], [0], color=BIT_COLORS[1], lw=2, label="target = 1")]
        fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=10,
                   framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

        hold_label = f" (hold periods only, ±{PULSE_MARGIN} steps excluded)" if HOLD_ONLY else ""
        fig.suptitle(f"Slow-Mode Projections Colored by Target Bit — Tanh Networks{hold_label}",
                     fontsize=13, fontweight="bold", y=1.0)
        plt.tight_layout()
        plt.show()

# %% Fixed-point finder for one Tanh network
# Minimize q(r) = ||r - tanh(g·W·r)||² from many initial conditions.

if tanh_eig_data:
    FP_GAIN = max(g for g in tanh_eig_data if tanh_eig_data[g]["eigs_trained"] is not None)
    print(f"Finding fixed points for Tanh network with g = {FP_GAIN}")

    fp_row = tanh_runs[tanh_runs["gain"] == FP_GAIN].iloc[0]
    fp_seed_path = os.path.join(sweep_dir, fp_row["exp_name"], f"seed_{int(fp_row['seed'])}")
    fp_config_file = os.path.join(fp_seed_path, "run_config.yaml")
    with open(fp_config_file) as f:
        fp_config = yaml.safe_load(f)

    fp_best = glob.glob(os.path.join(fp_seed_path, "checkpoints", "best-model-*.ckpt"))
    fp_ckpt = torch.load(fp_best[0], map_location="cpu", weights_only=False)
    fp_W_rec = None
    for key, val in fp_ckpt["state_dict"].items():
        if "W_rec.weight" in key:
            fp_W_rec = val.numpy()
            break

    fp_g = FP_GAIN
    fp_N = fp_W_rec.shape[0]
    fp_dt = fp_config["dt"]
    fp_tau = fp_config["timescales_config"]["values"][0]
    fp_alpha = 1.0 - np.exp(-fp_dt / fp_tau)
    gW = fp_g * fp_W_rec

    def fp_residual(r):
        return r - np.tanh(gW @ r)

    def fp_speed(r):
        res = fp_residual(r)
        return 0.5 * np.sum(res ** 2)

    def fp_speed_grad(r):
        res = fp_residual(r)
        sech2 = 1.0 - np.tanh(gW @ r) ** 2
        dres_dr = np.eye(fp_N) - np.diag(sech2) @ gW
        return dres_dr.T @ res

    from scipy.optimize import minimize as sp_minimize

    # Use hidden states from trajectories as initial conditions
    fp_proj = tanh_eig_data[FP_GAIN]
    fp_hidden_all = []
    fp_row_data = tanh_runs[tanh_runs["gain"] == FP_GAIN].iloc[0]
    fp_sp = os.path.join(sweep_dir, fp_row_data["exp_name"], f"seed_{int(fp_row_data['seed'])}")

    fp_model = MultiTimescaleRNN(
        input_size=fp_config["n_bits"],
        hidden_size=fp_config["hidden_size"],
        output_size=fp_config["n_bits"],
        dt=fp_config["dt"],
        timescales_config=fp_config.get("timescales_config"),
        activation=getattr(nn, fp_config["activation"]),
        learn_timescales=fp_config["learn_timescales"],
        init_timescale=fp_config.get("init_timescale"),
        shared_timescale=fp_config["shared_timescale"],
        normalize_hidden=fp_config["normalize_hidden"],
        zero_diag_wrec=fp_config["zero_diag_wrec"],
        recurrent_gain=fp_config["recurrent_gain"],
        noise_std=0.0,
        wrec_init=fp_config["wrec_init"],
        alpha_parameterization=fp_config["alpha_parameterization"],
        dynamics_type=fp_config["dynamics_type"],
    )
    fp_lit = MultiTimescaleRNNLightning(
        model=fp_model,
        learning_rate=fp_config["learning_rate"],
        weight_decay=fp_config["weight_decay"],
        step_size=fp_config.get("lr_step_size", fp_config.get("step_size", 1000)),
        gamma=fp_config["gamma"],
        task="flip_flop",
    )
    fp_lit.load_state_dict(fp_ckpt["state_dict"])
    fp_lit.eval()

    fp_dm = FlipFlopDataModule(
        n_bits=fp_config["n_bits"],
        p_pulse=fp_config["p_pulse"],
        pulse_amplitude=fp_config["pulse_amplitude"],
        num_time_steps=fp_config["num_time_steps"],
        num_val_trajectories=100,
        batch_size=100,
    )
    fp_dm.setup()
    fp_inp, _, fp_tgt = fp_dm.val_dataset.tensors

    with torch.no_grad():
        fp_h, _ = fp_lit.model(fp_inp, init_context=None)
        fp_hidden_all = fp_h.numpy().reshape(-1, fp_N)

    # Subsample initial conditions
    rng = np.random.RandomState(0)
    n_inits = 200
    init_idx = rng.choice(len(fp_hidden_all), size=n_inits, replace=False)
    init_points = fp_hidden_all[init_idx]

    # Also add the origin
    init_points = np.vstack([np.zeros((1, fp_N)), init_points])

    converged = []
    for r0 in init_points:
        res = sp_minimize(fp_speed, r0, jac=fp_speed_grad, method="L-BFGS-B",
                          options={"maxiter": 5000, "ftol": 1e-20, "gtol": 1e-12})
        if res.fun < 1e-10:
            converged.append(res.x)

    converged = np.array(converged)
    print(f"Converged from {len(converged)}/{len(init_points)} initial conditions")

    # Cluster by distance
    unique_fps = [converged[0]]
    for r in converged[1:]:
        dists = [np.linalg.norm(r - u) for u in unique_fps]
        if min(dists) > 0.1:
            unique_fps.append(r)
    unique_fps = np.array(unique_fps)
    print(f"Found {len(unique_fps)} unique fixed points")

    # Print fixed points: speed, norm, and output (via readout weights)
    fp_W_out = None
    fp_b_out = None
    for key, val in fp_ckpt["state_dict"].items():
        if "readout.weight" in key:
            fp_W_out = val.numpy()
        if "readout.bias" in key:
            fp_b_out = val.numpy()

    print(f"\n{'FP':>3} {'||r*||':>8} {'q(r*)':>10} {'output (sigmoid)':>30}")
    print("-" * 60)
    for i, r in enumerate(unique_fps):
        q = fp_speed(r)
        norm = np.linalg.norm(r)
        if fp_W_out is not None:
            logit = fp_W_out @ r + (fp_b_out if fp_b_out is not None else 0)
            out = 1.0 / (1.0 + np.exp(-logit))
            out_str = ", ".join(f"{v:.2f}" for v in out)
        else:
            out_str = "N/A"
        print(f"{i:3d} {norm:8.4f} {q:10.2e}    [{out_str}]")

    # Visualize fixed points in slow-mode subspace
    fp_W_proj = tanh_eig_data[FP_GAIN]["top3_idx"]
    eigs_t = tanh_eig_data[FP_GAIN]["eigs_trained"]
    V_r = np.linalg.eig(
        (1.0 - fp_alpha) * np.eye(fp_N)
        + fp_alpha * np.diag(1.0 - np.tanh(gW @ unique_fps.mean(axis=0))**2) @ gW
    )[1]
    V_l = np.linalg.inv(V_r)
    top3 = tanh_eig_data[FP_GAIN]["top3_idx"]
    W_proj_fp = V_l[top3, :].real

    fp_projected = unique_fps @ W_proj_fp.T

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectory cloud faintly
    traj_proj = tanh_eig_data[FP_GAIN]["projected"]
    pts = traj_proj.reshape(-1, 3)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               s=0.5, alpha=0.05, c="#999", depthshade=False)

    # Plot fixed points
    if fp_W_out is not None:
        fp_outputs = 1.0 / (1.0 + np.exp(-(fp_W_out @ unique_fps.T + fp_b_out[:, None]).T))
        fp_labels = ["".join(str(int(v > 0.5)) for v in out) for out in fp_outputs]
    else:
        fp_labels = [str(i) for i in range(len(unique_fps))]

    ax.scatter(fp_projected[:, 0], fp_projected[:, 1], fp_projected[:, 2],
               s=100, c=TRAINED_COLOR, edgecolors="black", linewidths=1, zorder=5)
    for i, label in enumerate(fp_labels):
        ax.text(fp_projected[i, 0], fp_projected[i, 1], fp_projected[i, 2],
                f"  {label}", fontsize=8, zorder=6)

    ax.set_xlabel("Mode 1", fontsize=10)
    ax.set_ylabel("Mode 2", fontsize=10)
    ax.set_zlabel("Mode 3", fontsize=10)
    ax.set_title(f"Fixed Points in Slow-Mode Space — Tanh g={FP_GAIN}\n"
                 f"({len(unique_fps)} fixed points found)", fontsize=12)
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
        x = c["steps"] if c["steps"] else range(len(c["val_losses"]))
        ax.plot(x, c["val_losses"], color=color, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Training step", fontsize=13)
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

# %% Plot: accuracy training curves
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
            x = c["steps"][:len(val_acc)] if c["steps"] else range(len(val_acc))
            ax.plot(x, val_acc, color=color, linewidth=1.5, alpha=0.85)

        ax.set_xlabel("Training step", fontsize=13)
        ax.set_ylabel("Validation accuracy", fontsize=13)
        ax.set_title(activation, fontsize=14)
        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color=cmap(norm(g)), linewidth=2, label=f"g = {g}")
        for g in all_gains
    ]
    axes[-1].legend(handles=legend_handles, title="Recurrent gain $g$",
                    fontsize=10, title_fontsize=11, loc="lower right")

    fig.suptitle("Flip-Flop: Validation Accuracy Curves by Gain", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("No accuracy data found in training_losses.json — re-run sweep to generate.")

# %% Steps to convergence vs recurrent gain
ACCURACY_THRESHOLD = 0.92  # bit accuracy threshold defining "converged"

convergence_records = []
for c in curves:
    val_acc = c.get("val_accuracies", [])
    if not val_acc:
        continue
    steps = np.array(c["steps"][:len(val_acc)]) if c["steps"] else np.arange(1, len(val_acc) + 1)
    val_acc = np.array(val_acc)
    hit = np.where(val_acc >= ACCURACY_THRESHOLD)[0]
    steps_to_converge = int(steps[hit[0]]) if len(hit) > 0 else np.nan
    convergence_records.append({
        "gain": c["gain"],
        "activation": c["activation"],
        "seed": c["seed"],
        "steps_to_convergence": steps_to_converge,
    })

df_conv = pd.DataFrame(convergence_records)
print(f"Convergence threshold: accuracy ≥ {ACCURACY_THRESHOLD}")
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

        ax.set_xlabel("Recurrent gain $g$", fontsize=13)
        ax.set_ylabel(f"Steps to accuracy ≥ {ACCURACY_THRESHOLD}", fontsize=13)
        ax.set_title(activation, fontsize=14)
        ax.set_xticks(gains)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Flip-Flop: Convergence Speed (accuracy ≥ {ACCURACY_THRESHOLD})",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("No accuracy curves available — cannot compute convergence.")
# %%
