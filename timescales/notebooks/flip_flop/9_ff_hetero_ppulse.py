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
sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/flip_flop_hetero_ppulse_20260409_021117"

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


# %% Plot 4: Effective timescale scree plot — tau_eff = -1/ln|lambda|

FIGS_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

_sample = next(iter(eig_data.values()))
_pp = _sample["p_pulse"]
_pp_list = _pp if isinstance(_pp, list) else [_pp]
_holds = [1.0 / p for p in _pp_list]

fig, axes = plt.subplots(1, n_gains, figsize=(5 * n_gains, 4.5),
                         squeeze=False, sharey=True)

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

    ax.scatter(ranks[n_bits:], tau_eff[n_bits:], s=20, color=UNTRAINED_COLOR,
               edgecolors="none", alpha=0.6, zorder=3, label="Other modes")

    # Auto-match each top-|λ| mode to its nearest hold in log-space.
    # Uses a greedy assignment: for each hold (sorted by size), find the
    # closest unmatched tau_eff; skip both if log-ratio > MATCH_TOLERANCE.
    MATCH_TOLERANCE = 0.5  # ~1.6x in linear space

    holds_arr = np.array(_holds)
    tau_top = tau_eff[:n_bits]

    hold_order = np.argsort(holds_arr)[::-1]   # largest hold first
    tau_order = np.argsort(tau_top)[::-1]       # largest tau_eff first

    matched_tau = {}   # tau_rank -> hold_idx
    used_holds = set()
    used_taus = set()

    for hi in hold_order:
        hold_val = holds_arr[hi]
        best_dist, best_ti = np.inf, None
        for ti in tau_order:
            if ti in used_taus:
                continue
            d = abs(np.log(tau_top[ti] + 1e-12) - np.log(hold_val + 1e-12))
            if d < best_dist:
                best_dist, best_ti = d, ti
        if best_ti is not None and best_dist <= MATCH_TOLERANCE:
            matched_tau[best_ti] = hi
            used_holds.add(hi)
            used_taus.add(best_ti)

    # One color per hold index so colors are stable across gains
    hold_colors = {hi: f"C{k}" for k, hi in enumerate(hold_order)}

    for ti in range(min(n_bits, len(tau_eff))):
        if ti in matched_tau:
            hi = matched_tau[ti]
            c = hold_colors[hi]
            ax.scatter(ranks[ti], tau_top[ti], s=70, color=c,
                       edgecolors="white", linewidths=0.8, zorder=4)
            ax.annotate(f"$\\tau$ = {tau_top[ti]:.0f}",
                        (ranks[ti], tau_top[ti]),
                        textcoords="offset points", xytext=(10, 0),
                        fontsize=9, color=c, fontweight="bold",
                        va="center",
                        arrowprops=dict(arrowstyle="-", color=c,
                                        lw=0.6, alpha=0.4))
        else:
            ax.scatter(ranks[ti], tau_top[ti], s=40, color=UNTRAINED_COLOR,
                       edgecolors="white", linewidths=0.6, zorder=4, alpha=0.7)

    for hi in range(len(_holds)):
        hold_val = holds_arr[hi]
        if hi not in used_holds:
            continue
        c = hold_colors[hi]
        ax.axhline(hold_val, color=c, linewidth=0.9, linestyle=":",
                   alpha=0.5, zorder=1)
        ax.text(N * 0.92, hold_val * 1.15,
                f"hold ≈ {hold_val:.0f}", fontsize=7.5,
                color=c, ha="right", alpha=0.7)

    ax.set_xlabel("Eigenvalue rank", fontsize=11)
    #ax.set_xscale("log")
    if col == 0:
        ax.set_ylabel(r"$\tau_{\mathrm{eff}} = -1\,/\,\ln|\lambda|$  (steps)",
                      fontsize=11)
    ax.set_title(f"$g = {g}$", fontsize=12)
    ax.grid(True, alpha=0.12, which="both")
    ax.legend(fontsize=8, loc="center right", framealpha=0.85,
              edgecolor="none")
    ax.set_yscale("log")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Effective Timescales (ranked eigenvalues)",
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
