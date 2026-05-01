# %% [markdown]
# # Schur-Parameterized RNN — Sweep Analysis
#
# Compares training conditions from the Schur decomposition sweep:
#
# | Condition | train_t | train_q | What learns |
# |-----------|---------|---------|-------------|
# | T only    | True    | False   | Eigenvalue structure (fixed basis) |
# | Both      | True    | True    | Full Schur decomposition |
# | Q only    | False   | True    | Basis rotation (fixed eigenvalues) |
# | Frozen    | False   | False   | Nothing recurrent (readout only) |
#
# $W_\text{rec}$ is reconstructed as $Q T Q^\top$ on every forward pass,
# where $T$ is quasi-upper-triangular and $Q$ is orthogonal (Cayley-parameterized).

# %%
import os
import sys
import subprocess
import json

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()

os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())

from rnns.schur_rnn import SchurRNN, SchurRNNLightning
from datamodules.flip_flop import FlipFlopDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_FIGS = True
FIG_DIR = os.path.join("notebooks", "figs", "schur")
os.makedirs(FIG_DIR, exist_ok=True)

# %% Specify sweep directory
sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/flip_flop_schur_sweep_20260410_020558"

#sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/flip_flop_schur_sweep_20260410_050424"

# %% Parse condition labels from experiment names
CONDITION_LABELS = {
    (True, True): "Both T & Q",
    (True, False): "T only (eigenvalues)",
    (False, True): "Q only (basis)",
    (False, False): "Frozen $W_\\mathrm{rec}$",
}


def _parse_condition(exp_name: str) -> tuple[bool, bool]:
    """Extract (train_t, train_q) from naming format 'trainT_{}_trainQ_{}'."""
    parts = exp_name.lower()
    train_t = "traint_true" in parts
    train_q = "trainq_true" in parts
    return train_t, train_q


# %% Load data and training curves
records = []

for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
        continue
    if not exp_name.startswith("trainT_"):
        continue

    train_t, train_q = _parse_condition(exp_name)
    label = CONDITION_LABELS[(train_t, train_q)]

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

        run_config_path = os.path.join(seed_path, "run_config.yaml")
        run_config = {}
        if os.path.exists(run_config_path):
            with open(run_config_path) as f:
                run_config = yaml.safe_load(f)

        records.append(dict(
            exp_name=exp_name,
            train_t=train_t,
            train_q=train_q,
            label=label,
            seed=seed,
            seed_path=seed_path,
            final_val_loss=fvl,
            val_losses=val_losses,
            val_accs=val_accs,
            steps=steps,
            run_config=run_config,
        ))

df = pd.DataFrame(records)
conditions = sorted(df["label"].unique())
print(f"Loaded {len(df)} runs across {len(conditions)} conditions:")
for c in conditions:
    n = len(df[df["label"] == c])
    print(f"  {c}: {n} seed(s)")

# %% Color palette for conditions
COLORS = {
    "T only (eigenvalues)": "#2a9d8f",
    "Q only (basis)":       "#e76f51",
    "Both T & Q":           "#264653",
    "Frozen $W_\\mathrm{rec}$": "#adb5bd",
}

# %% Plot 1: Validation loss vs training step
fig, ax = plt.subplots(figsize=(9, 4.5))
plotted_labels = set()

for _, row in df.iterrows():
    lab = row["label"]
    vl = row["val_losses"]
    st = row["steps"][:len(vl)] if row["steps"] else list(range(1, len(vl) + 1))
    show_label = lab if lab not in plotted_labels else None
    ax.plot(st, vl, linewidth=1.8, color=COLORS.get(lab, "gray"),
            alpha=0.7, label=show_label)
    plotted_labels.add(lab)

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation loss", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
fig.suptitle("Schur RNN — Flip-Flop: Validation Loss",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIG_DIR, "schur_val_loss.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 2: Validation accuracy vs training step
fig, ax = plt.subplots(figsize=(9, 4.5))
plotted_labels = set()

for _, row in df.iterrows():
    lab = row["label"]
    va = row["val_accs"]
    st = row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1))
    if not va:
        continue
    show_label = lab if lab not in plotted_labels else None
    ax.plot(st, va, linewidth=1.8, color=COLORS.get(lab, "gray"),
            alpha=0.7, label=show_label)
    plotted_labels.add(lab)

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation accuracy", fontsize=12)
ax.set_ylim(0.4, 1.02)
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="lower right")
fig.suptitle("Schur RNN — Flip-Flop: Validation Accuracy",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIG_DIR, "schur_val_accuracy.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 3: Final validation loss bar chart (mean +/- std across seeds)
summary = (
    df.groupby("label")["final_val_loss"]
    .agg(["mean", "std", "count"])
    .reindex(["T only (eigenvalues)", "Q only (basis)",
              "Both T & Q", "Frozen $W_\\mathrm{rec}$"])
    .dropna(subset=["mean"])
)

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(summary))
bars = ax.bar(x, summary["mean"], yerr=summary["std"],
              capsize=5, width=0.6, edgecolor="black", linewidth=0.8,
              color=[COLORS.get(lab, "gray") for lab in summary.index])
ax.set_xticks(x)
ax.set_xticklabels(summary.index, fontsize=10)
ax.set_ylabel("Final validation loss", fontsize=12)
ax.set_yscale("log")
ax.grid(True, axis="y", alpha=0.3)
fig.suptitle("Schur RNN — Final Validation Loss by Condition",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIG_DIR, "schur_final_loss_bar.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## Eigenspectrum Analysis
#
# Load trained checkpoints and inspect the eigenvalues of the
# effective Jacobian $J = Q T Q^\top$ for each condition.

# %%
def load_schur_model(seed_path: str) -> SchurRNN:
    """Load a trained SchurRNN from checkpoint."""
    ckpt_dir = os.path.join(seed_path, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        return None

    best = [c for c in ckpts if "best" in c.lower()]
    ckpt_file = best[0] if best else sorted(ckpts)[-1]
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)

    config_path = os.path.join(seed_path, "run_config.yaml")
    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_bits = cfg.get("n_bits", 3)
    input_size = n_bits
    output_size = n_bits

    model = SchurRNN(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        output_size=output_size,
        dt=cfg["dt"],
        tau=cfg["tau"],
        recurrent_gain=cfg.get("recurrent_gain", 1.0),
        noise_std=cfg.get("noise_std", 0.0),
        wrec_init=cfg.get("wrec_init", "normal_scaled"),
        train_t=cfg.get("train_t", True),
        train_q=cfg.get("train_q", False),
        q_parameterization=cfg.get("q_parameterization", "cayley"),
    )

    lit = SchurRNNLightning.load_from_checkpoint(
        ckpt_path, model=model,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        step_size=cfg.get("lr_step_size", cfg.get("step_size", 1000)),
        gamma=cfg["gamma"],
        task=cfg.get("task", "flip_flop"),
    )
    lit.eval()
    return lit.model


def get_jacobian_eigen(model: SchurRNN):
    """Compute J = Q T Q^T and its eigendecomposition."""
    with torch.no_grad():
        Q = model.rnn_step._effective_Q()
        T = model.rnn_step.T
        if model.rnn_step.train_q and model.rnn_step.q_parameterization == "free":
            I = torch.eye(Q.shape[0], device=Q.device)
            J = Q @ T @ torch.linalg.solve(Q, I)
        else:
            J = Q @ T @ Q.T
    J_np = J.cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(J_np)
    return eigenvalues, eigenvectors, J_np


# %% Compute eigenspectra
eigen_records = []

for _, row in df.iterrows():
    model = load_schur_model(row["seed_path"])
    if model is None:
        continue
    eigs, eigvecs, J = get_jacobian_eigen(model)

    W_out = model.W_out.weight.detach().cpu().numpy()
    W_in = model.rnn_step.W_in.weight.detach().cpu().numpy()

    config_path = os.path.join(row["seed_path"], "run_config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    n_bits = cfg.get("n_bits", 3)
    p_pulse = cfg.get("p_pulse", 0.05)

    eigen_records.append(dict(
        label=row["label"],
        seed=row["seed"],
        eigenvalues=eigs,
        eigenvectors=eigvecs,
        J=J,
        W_out=W_out,
        W_in=W_in,
        n_bits=n_bits,
        p_pulse=p_pulse,
    ))

print(f"Loaded eigenspectra for {len(eigen_records)} models")

# %% Plot 4: Eigenvalue spectra in the complex plane
if eigen_records:
    fig, axes = plt.subplots(1, len(CONDITION_LABELS), figsize=(16, 4),
                             sharex=True, sharey=True)

    for ax, (_, lab) in zip(axes, sorted(CONDITION_LABELS.items(),
                                          key=lambda x: list(COLORS.keys()).index(x[1])
                                          if x[1] in COLORS else 99)):
        recs = [r for r in eigen_records if r["label"] == lab]
        if not recs:
            ax.set_title(lab + "\n(no data)", fontsize=10)
            continue

        for r in recs:
            eigs = r["eigenvalues"]
            ax.scatter(eigs.real, eigs.imag, s=8, alpha=0.5,
                       color=COLORS.get(lab, "gray"), edgecolors="none")

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.5, alpha=0.4)
        ax.set_aspect("equal")
        ax.set_title(lab, fontsize=11)
        ax.set_xlabel("Re", fontsize=10)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Im", fontsize=10)
    fig.suptitle("Eigenvalues of Jacobian $J = Q T Q^\\top$ by Condition",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIG_DIR, "schur_eigenspectra.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% Plot 5: Effective timescale scree plot — tau_eff = -1/ln|lambda|
if eigen_records:
    first_cfg = df.iloc[0]["run_config"] if len(df) > 0 else {}
    _n_bits = first_cfg.get("n_bits", 3)
    _p_pulse = first_cfg.get("p_pulse", 0.05)
    _pp_list = _p_pulse if isinstance(_p_pulse, list) else [_p_pulse]
    _holds = [1.0 / p for p in _pp_list]

    condition_order_scree = [
        "T only (eigenvalues)",
        "Q only (basis)",
        "Both T & Q",
        "Frozen $W_\\mathrm{rec}$",
    ]
    active_conds = [c for c in condition_order_scree
                    if any(r["label"] == c for r in eigen_records)]
    n_panels = len(active_conds)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5),
                             squeeze=False, sharey=True)

    for col, cond in enumerate(active_conds):
        ax = axes[0, col]
        recs = [r for r in eigen_records if r["label"] == cond]
        if not recs:
            continue

        eigs = recs[0]["eigenvalues"]
        abs_sorted = np.sort(np.abs(eigs))[::-1]
        N = len(abs_sorted)
        ranks = np.arange(1, N + 1)

        log_abs = np.log(np.clip(abs_sorted, 1e-12, None))
        tau_eff = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)

        ax.scatter(ranks[_n_bits:], tau_eff[_n_bits:], s=20, color="#adb5bd",
                   edgecolors="none", alpha=0.6, zorder=3, label="Other modes")

        hold_colors = [f"C{k}" for k in range(len(_holds))]

        for ti in range(min(_n_bits, len(tau_eff))):
            c = hold_colors[ti % len(hold_colors)]
            ax.scatter(ranks[ti], tau_eff[ti], s=70, color=c,
                       edgecolors="white", linewidths=0.8, zorder=4)
            ax.annotate(f"$\\tau$ = {tau_eff[ti]:.0f}",
                        (ranks[ti], tau_eff[ti]),
                        textcoords="offset points", xytext=(10, 0),
                        fontsize=9, color=c, fontweight="bold", va="center",
                        arrowprops=dict(arrowstyle="-", color=c,
                                        lw=0.6, alpha=0.4))

        for hi, hold_val in enumerate(_holds):
            c = hold_colors[hi % len(hold_colors)]
            ax.axhline(hold_val, color=c, linewidth=0.9, linestyle=":",
                       alpha=0.5, zorder=1)
            ax.text(N * 0.92, hold_val * 1.15,
                    f"hold ≈ {hold_val:.0f}", fontsize=7.5,
                    color=c, ha="right", alpha=0.7)

        ax.set_xlabel("Eigenvalue rank", fontsize=11)
        if col == 0:
            ax.set_ylabel(
                r"$\tau_{\mathrm{eff}} = -1\,/\,\ln|\lambda|$  (steps)",
                fontsize=11)
        ax.set_title(cond, fontsize=12, color=COLORS.get(cond, "black"))
        ax.set_yscale("log")
        ax.grid(True, alpha=0.12, which="both")
        ax.legend(fontsize=8, loc="center right", framealpha=0.85,
                  edgecolor="none")
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Effective Timescales by Condition (ranked eigenvalues)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIG_DIR, "schur_tau_eff_scree.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% [markdown]
# ## Mode-to-Output Coupling
#
# $|W_\text{out} V|$ tells us how much each eigenvector mode contributes to
# each output bit.  We show a full heatmap across **all** modes (columns ranked
# by $|\lambda|$) for every condition, so differences in coupling structure
# between T-only, Q-only, Both, and Frozen are directly visible.

# %% Plot 6: Mode-to-output coupling heatmap — |W_out @ V| across all modes
if eigen_records:
    condition_order_coupling = [
        "T only (eigenvalues)",
        "Q only (basis)",
        "Both T & Q",
        "Frozen $W_\\mathrm{rec}$",
    ]
    active_conds = [c for c in condition_order_coupling
                    if any(r["label"] == c for r in eigen_records)]

    for cond in active_conds:
        recs = [r for r in eigen_records if r["label"] == cond]
        if not recs:
            continue
        rec = recs[0]

        eigs = rec["eigenvalues"]
        V = rec["eigenvectors"]
        W_out = rec["W_out"]
        n_bits = rec["n_bits"]
        pp = rec["p_pulse"]
        pp_list = pp if isinstance(pp, list) else [pp] * n_bits
        N = len(eigs)

        abs_rank_order = np.argsort(np.abs(eigs))[::-1]
        abs_ranked = np.abs(eigs[abs_rank_order])

        log_abs = np.log(np.clip(abs_ranked, 1e-12, None))
        tau_ranked = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)

        coupling_all = np.abs(W_out @ V)
        coupling_ranked = coupling_all[:, abs_rank_order]

        # --- Output coupling heatmap ---
        fig, ax = plt.subplots(figsize=(max(14, N * 0.12), max(n_bits * 0.7, 3)))
        im = ax.imshow(coupling_ranked, cmap="YlOrRd", aspect="auto",
                       interpolation="nearest")

        ax.set_yticks(range(n_bits))
        ax.set_yticklabels([f"Bit {i} (p={pp_list[i]})" for i in range(n_bits)],
                           fontsize=9)
        ax.set_xlabel("Eigenmode rank (by $|\\lambda|$)", fontsize=11)
        ax.set_ylabel("Output bit", fontsize=11)

        ax.axvline(n_bits - 0.5, color="white", linewidth=1.5, linestyle="--",
                   alpha=0.8)

        plt.colorbar(im, ax=ax, label="$|W_{\\mathrm{out}} V|$", shrink=0.8)
        fig.suptitle(
            f"Mode-to-Output Coupling — {cond}\n"
            f"$|W_{{\\mathrm{{out}}}} V|$ across all {N} modes "
            f"(dashed line = top {n_bits})",
            fontsize=12, fontweight="bold", y=1.04,
        )
        plt.tight_layout()
        if SAVE_FIGS:
            safe = cond.replace(" ", "_").replace("$", "").replace("\\", "")
            fig.savefig(os.path.join(FIG_DIR, f"schur_output_coupling_{safe}.pdf"),
                        bbox_inches="tight", dpi=150)
        plt.show()

        # --- Input coupling heatmap: |V^{-1} W_in| ---
        W_in = rec["W_in"]
        V_inv = np.linalg.pinv(V)
        input_coupling = np.abs(V_inv @ W_in)          # (N, n_bits)
        input_coupling_ranked = input_coupling[abs_rank_order]

        fig, ax = plt.subplots(figsize=(max(14, N * 0.12), max(n_bits * 0.7, 3)))
        im = ax.imshow(input_coupling_ranked.T, cmap="YlGnBu", aspect="auto",
                       interpolation="nearest")

        ax.set_yticks(range(n_bits))
        ax.set_yticklabels([f"Input {i} (p={pp_list[i]})" for i in range(n_bits)],
                           fontsize=9)
        ax.set_xlabel("Eigenmode rank (by $|\\lambda|$)", fontsize=11)
        ax.set_ylabel("Input bit", fontsize=11)

        ax.axvline(n_bits - 0.5, color="white", linewidth=1.5, linestyle="--",
                   alpha=0.8)

        plt.colorbar(im, ax=ax, label="$|V^{-1} W_{\\mathrm{in}}|$", shrink=0.8)
        fig.suptitle(
            f"Input-to-Mode Coupling — {cond}\n"
            f"$|V^{{-1}} W_{{\\mathrm{{in}}}}|$ across all {N} modes "
            f"(dashed line = top {n_bits})",
            fontsize=12, fontweight="bold", y=1.04,
        )
        plt.tight_layout()
        if SAVE_FIGS:
            safe = cond.replace(" ", "_").replace("$", "").replace("\\", "")
            fig.savefig(os.path.join(FIG_DIR, f"schur_input_coupling_{safe}.pdf"),
                        bbox_inches="tight", dpi=150)
        plt.show()

        # Dominant mode per bit (top-n_bits only)
        coupling_top = coupling_ranked[:, :n_bits]
        dominant = np.argmax(coupling_top, axis=1)
        print(f"\n{cond}: Dominant slow mode per output bit:")
        for bit_i in range(n_bits):
            mode_j = dominant[bit_i]
            print(f"  Bit {bit_i} (p_pulse={pp_list[bit_i]}, "
                  f"hold~{1.0/pp_list[bit_i]:.0f} steps)"
                  f"  <--  Mode {mode_j+1} "
                  f"(tau_eff={tau_ranked[mode_j]:.1f} steps)")

# %% [markdown]
# ## Alternative Memory Mechanisms
#
# Networks that don't learn slow eigenvalues (Frozen $W_\text{rec}$, Q-only)
# may still solve the task through other mechanisms:
#
# 1. **Transient amplification** — non-normal $J$ can sustain signals
#    longer than any single eigenvalue predicts ($\|J^k\| > \rho(J)^k$).
# 2. **Henrici non-normality** — $\|J\|_F^2 - \sum|\lambda_i|^2$ measures
#    how far $J$ is from being diagonalisable by a unitary.
# 3. **Direct pathway** — structured $W_\text{out} W_\text{in}$ can bypass
#    recurrent dynamics entirely.
# 4. **Complex-pair subspace coupling** — conjugate pairs $\lambda = r e^{\pm i\theta}$
#    create 2-D damped rotations; if $W_\text{in}$ and $W_\text{out}$ couple to the
#    right phase, the network encodes bits through oscillatory dynamics.

# %% Plot 8: Transient amplification — ||J^k||_2 vs k
if eigen_records:
    condition_order_alt = [
        "T only (eigenvalues)",
        "Q only (basis)",
        "Both T & Q",
        "Frozen $W_\\mathrm{rec}$",
    ]
    active_conds = [c for c in condition_order_alt
                    if any(r["label"] == c for r in eigen_records)]

    K_MAX = 300

    fig, ax = plt.subplots(figsize=(10, 5))
    for cond in active_conds:
        rec = next(r for r in eigen_records if r["label"] == cond)
        J = rec["J"]
        eigs = rec["eigenvalues"]
        rho = np.max(np.abs(eigs))

        Jk = np.eye(J.shape[0])
        norms = np.empty(K_MAX + 1)
        norms[0] = np.linalg.norm(Jk, ord=2)
        for k in range(1, K_MAX + 1):
            Jk = Jk @ J
            norms[k] = np.linalg.norm(Jk, ord=2)

        ks = np.arange(K_MAX + 1)
        ax.semilogy(ks, norms, linewidth=1.5, color=COLORS.get(cond, "gray"),
                    label=cond)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("$k$ (time steps)", fontsize=12)
    ax.set_ylabel("$\\|J^k\\|_2$ (operator norm)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, K_MAX)
    fig.suptitle("Transient Amplification — $\\|J^k\\|_2$ vs $k$\n"
                 "Non-normal matrices can amplify before decaying",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIG_DIR, "schur_transient_amplification.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% Plot 8b: Input-to-output transient gain — ||W_out J^k W_in||_F
# This is the quantity that actually matters for the task: how much does
# a pulse at time 0 still influence the output at time k?
if eigen_records:
    fig, axes = plt.subplots(1, len(active_conds),
                             figsize=(5 * len(active_conds), 4.5),
                             squeeze=False, sharey=True)

    for col, cond in enumerate(active_conds):
        ax = axes[0, col]
        rec = next(r for r in eigen_records if r["label"] == cond)
        J = rec["J"]
        W_out = rec["W_out"]
        W_in = rec["W_in"]
        n_bits = rec["n_bits"]
        pp = rec["p_pulse"]
        pp_list = pp if isinstance(pp, list) else [pp] * n_bits

        Jk = np.eye(J.shape[0])
        io_gain = np.empty((K_MAX + 1, n_bits, n_bits))
        for k in range(K_MAX + 1):
            G = W_out @ Jk @ W_in      # (n_bits, n_bits)
            io_gain[k] = np.abs(G)
            Jk = Jk @ J

        ks = np.arange(K_MAX + 1)
        for bit_i in range(n_bits):
            ax.semilogy(ks, io_gain[:, bit_i, bit_i], linewidth=1.4,
                        label=f"Bit {bit_i}→{bit_i} (p={pp_list[bit_i]})")

        ax.axhline(io_gain[0].max() * 0.01, color="gray", linestyle=":",
                   linewidth=0.6, alpha=0.5)
        ax.set_xlabel("$k$ (delay steps)", fontsize=11)
        if col == 0:
            ax.set_ylabel("$|W_{\\mathrm{out}} J^k W_{\\mathrm{in}}|_{ii}$",
                          fontsize=11)
        ax.set_title(cond, fontsize=11, color=COLORS.get(cond, "black"))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.15)
        ax.set_xlim(0, K_MAX)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Input→Output Transient Gain per Bit — $|W_{\\mathrm{out}} J^k W_{\\mathrm{in}}|$\n"
                 "How long does each bit's signal persist?",
                 fontsize=13, fontweight="bold", y=1.06)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIG_DIR, "schur_io_transient_gain.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% Non-normality & direct pathway summary table
if eigen_records:
    print("\n" + "=" * 75)
    print("Non-normality & Direct Pathway Analysis")
    print("=" * 75)

    for cond in active_conds:
        rec = next(r for r in eigen_records if r["label"] == cond)
        J = rec["J"]
        eigs = rec["eigenvalues"]
        W_out = rec["W_out"]
        W_in = rec["W_in"]
        n_bits = rec["n_bits"]
        N = J.shape[0]

        # Henrici non-normality: ||J||_F^2 - sum|lambda_i|^2
        frob_sq = np.linalg.norm(J, "fro") ** 2
        eig_sq = np.sum(np.abs(eigs) ** 2)
        henrici = np.sqrt(max(frob_sq - eig_sq, 0.0))
        henrici_normed = henrici / np.linalg.norm(J, "fro")

        # Spectral radius
        rho = np.max(np.abs(eigs))

        # Peak transient amplification
        Jk = np.eye(N)
        peak_norm = 1.0
        for k in range(1, K_MAX + 1):
            Jk = Jk @ J
            peak_norm = max(peak_norm, np.linalg.norm(Jk, ord=2))

        # Direct pathway
        direct = W_out @ W_in      # (n_bits, n_bits)

        print(f"\n  {cond}:")
        print(f"    spectral radius ρ(J) = {rho:.4f}")
        print(f"    Henrici non-normality = {henrici:.3f}  "
              f"(normalised = {henrici_normed:.3f})")
        print(f"    peak ||J^k||_2        = {peak_norm:.3f}  "
              f"(amplification ratio = {peak_norm / max(rho, 1e-12):.2f}×)")
        print(f"    direct pathway W_out @ W_in:")
        for i in range(n_bits):
            row_str = "  ".join(f"{direct[i, j]:+.4f}" for j in range(n_bits))
            print(f"      bit {i}: [{row_str}]")

# %% Plot 9: Direct pathway heatmap — W_out @ W_in
if eigen_records:
    n_conds = len(active_conds)
    first_rec = next(r for r in eigen_records if r["label"] == active_conds[0])
    n_bits = first_rec["n_bits"]
    pp = first_rec["p_pulse"]
    pp_list = pp if isinstance(pp, list) else [pp] * n_bits

    fig, axes = plt.subplots(1, n_conds, figsize=(3.5 * n_conds, 3.2),
                             squeeze=False)
    vmax_all = 0
    directs = {}
    for cond in active_conds:
        rec = next(r for r in eigen_records if r["label"] == cond)
        d = rec["W_out"] @ rec["W_in"]
        directs[cond] = d
        vmax_all = max(vmax_all, np.abs(d).max())

    for col, cond in enumerate(active_conds):
        ax = axes[0, col]
        d = directs[cond]
        im = ax.imshow(d, cmap="RdBu_r", aspect="equal",
                       vmin=-vmax_all, vmax=vmax_all)
        ax.set_xticks(range(n_bits))
        ax.set_xticklabels([f"In {i}" for i in range(n_bits)], fontsize=8)
        ax.set_yticks(range(n_bits))
        ax.set_yticklabels([f"Out {i}" for i in range(n_bits)], fontsize=8)
        for i in range(n_bits):
            for j in range(n_bits):
                ax.text(j, i, f"{d[i,j]:.3f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(d[i,j]) > 0.6 * vmax_all else "black")
        ax.set_title(cond, fontsize=10, color=COLORS.get(cond, "black"))

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8,
                 label="$W_{\\mathrm{out}} W_{\\mathrm{in}}$")
    fig.suptitle("Direct Pathway — $W_{\\mathrm{out}} W_{\\mathrm{in}}$\n"
                 "Instantaneous input→output coupling (bypass of recurrence)",
                 fontsize=12, fontweight="bold", y=1.06)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIG_DIR, "schur_direct_pathway.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% Plot 10: Complex conjugate pair subspace coupling
# For each complex pair λ, λ* near |λ|≈1, the 2D invariant subspace is
# spanned by Re(v) and Im(v).  We measure how strongly W_in and W_out
# project into each such subspace.
if eigen_records:
    for cond in active_conds:
        rec = next(r for r in eigen_records if r["label"] == cond)
        eigs = rec["eigenvalues"]
        V = rec["eigenvectors"]
        W_out = rec["W_out"]
        W_in = rec["W_in"]
        n_bits = rec["n_bits"]
        pp = rec["p_pulse"]
        pp_list = pp if isinstance(pp, list) else [pp] * n_bits
        N = len(eigs)

        # Find complex conjugate pairs (positive imaginary part)
        cpx_mask = np.abs(eigs.imag) > 1e-8
        pos_imag = cpx_mask & (eigs.imag > 0)
        pair_idx = np.where(pos_imag)[0]

        if len(pair_idx) == 0:
            print(f"{cond}: no complex pairs found")
            continue

        abs_pair = np.abs(eigs[pair_idx])
        sort_order = np.argsort(abs_pair)[::-1]
        pair_idx = pair_idx[sort_order]

        n_pairs = min(20, len(pair_idx))
        pair_idx = pair_idx[:n_pairs]

        # For each pair, build orthonormal basis of the 2D subspace
        in_coupling = np.zeros((n_pairs, n_bits))
        out_coupling = np.zeros((n_pairs, n_bits))
        pair_abs = np.zeros(n_pairs)
        pair_freq = np.zeros(n_pairs)

        for pi, idx in enumerate(pair_idx):
            v = V[:, idx]
            u1 = v.real.copy()
            u2 = v.imag.copy()
            # Gram-Schmidt
            u1 /= np.linalg.norm(u1) + 1e-15
            u2 -= u1 * np.dot(u1, u2)
            u2 /= np.linalg.norm(u2) + 1e-15
            P = np.outer(u1, u1) + np.outer(u2, u2)   # projector onto 2D subspace

            for b in range(n_bits):
                w_in_b = W_in[:, b]
                in_coupling[pi, b] = np.linalg.norm(P @ w_in_b)
                w_out_b = W_out[b, :]
                out_coupling[pi, b] = np.linalg.norm(P @ w_out_b)

            pair_abs[pi] = np.abs(eigs[idx])
            pair_freq[pi] = np.abs(eigs[idx].imag) / (2 * np.pi)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(n_pairs * 0.35, 3)))

        ylabels = [f"|λ|={pair_abs[i]:.3f}, f={pair_freq[i]:.3f}"
                   for i in range(n_pairs)]

        im1 = ax1.imshow(in_coupling, cmap="YlGnBu", aspect="auto")
        ax1.set_yticks(range(n_pairs))
        ax1.set_yticklabels(ylabels, fontsize=7)
        ax1.set_xticks(range(n_bits))
        ax1.set_xticklabels([f"In {i}" for i in range(n_bits)], fontsize=9)
        ax1.set_xlabel("Input bit", fontsize=11)
        ax1.set_ylabel("Complex pair (ranked by $|\\lambda|$)", fontsize=10)
        ax1.set_title("$\\|P_{\\lambda} \\, w_{\\mathrm{in}}^{(i)}\\|$",
                      fontsize=11)
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        im2 = ax2.imshow(out_coupling, cmap="YlOrRd", aspect="auto")
        ax2.set_yticks(range(n_pairs))
        ax2.set_yticklabels(ylabels, fontsize=7)
        ax2.set_xticks(range(n_bits))
        ax2.set_xticklabels([f"Out {i}" for i in range(n_bits)], fontsize=9)
        ax2.set_xlabel("Output bit", fontsize=11)
        ax2.set_title("$\\|P_{\\lambda} \\, w_{\\mathrm{out}}^{(i)}\\|$",
                      fontsize=11)
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        fig.suptitle(
            f"Complex Pair Subspace Coupling — {cond}\n"
            f"Top {n_pairs} conjugate pairs: how strongly does each 2D "
            f"invariant subspace couple to inputs/outputs?",
            fontsize=12, fontweight="bold", y=1.06,
        )
        plt.tight_layout()
        if SAVE_FIGS:
            safe = cond.replace(" ", "_").replace("$", "").replace("\\", "")
            fig.savefig(os.path.join(FIG_DIR,
                        f"schur_complex_pair_coupling_{safe}.pdf"),
                        bbox_inches="tight", dpi=150)
        plt.show()

# %% [markdown]
# ## Example Trajectories
#
# Generate flip-flop inputs and compare network outputs across conditions.

# %%
first_config = df.iloc[0]["run_config"] if len(df) > 0 else {}
n_bits = first_config.get("n_bits", 3)
p_pulse = first_config.get("p_pulse", 0.05)
num_time_steps = first_config.get("num_time_steps", 500)
dt = first_config.get("dt", 0.1)

dm = FlipFlopDataModule(
    n_bits=n_bits,
    p_pulse=p_pulse,
    num_time_steps=num_time_steps,
    batch_size=1,
    num_val_trajectories=1,
    num_workers=0,
)
dm.setup("test")

val_batch = next(iter(dm.val_dataloader()))
inputs_traj, aux_traj, targets_traj = val_batch
inputs_traj = inputs_traj.to(device)
targets_traj = targets_traj.to(device)

# %% Plot 7: Output trajectories per condition (one figure each)
SEQ_IDX = 0

condition_order = [
    "T only (eigenvalues)",
    "Q only (basis)",
    "Both T & Q",
    "Frozen $W_\\mathrm{rec}$",
]

p_pulse_list = p_pulse if isinstance(p_pulse, list) else [p_pulse] * n_bits
t_arr = np.arange(num_time_steps)

for _, row in df.iterrows():
    cond = row["label"]
    seed_path = row["seed_path"]

    model = load_schur_model(seed_path)
    if model is None:
        continue
    model = model.to(device)
    model.eval()

    dm_traj = FlipFlopDataModule(
        n_bits=n_bits,
        p_pulse=p_pulse,
        pulse_amplitude=first_config.get("pulse_amplitude", 1.0),
        num_time_steps=num_time_steps,
        num_val_trajectories=10,
        batch_size=10,
    )
    dm_traj.setup()
    inp, _, tgt = dm_traj.val_dataset.tensors

    with torch.no_grad():
        _, out = model(inp.to(device))
        out_prob = torch.sigmoid(out).cpu()

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
                color="black", linewidth=1.5,
                label="target" if bit == 0 else None)
        ax.plot(t_arr, out_prob[SEQ_IDX, :, bit].numpy(),
                color=COLORS.get(cond, "steelblue"), linewidth=1.2, alpha=0.9,
                label="output" if bit == 0 else None)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1])
        avg_hold = 1.0 / max(p_pulse_list[bit], 1e-8)
        ax.set_ylabel(f"Bit {bit}\n(p={p_pulse_list[bit]}, ~{avg_hold:.0f} steps)",
                       fontsize=10)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Schur RNN Trajectories — {cond} (seed {row['seed']})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        safe_name = cond.replace(" ", "_").replace("$", "").replace("\\", "")
        fig.savefig(os.path.join(FIG_DIR, f"schur_traj_{safe_name}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% [markdown]
# ## Summary Table

# %%
if len(df) > 0:
    summary_table = (
        df.groupby("label")
        .agg(
            seeds=("seed", "count"),
            mean_loss=("final_val_loss", "mean"),
            std_loss=("final_val_loss", "std"),
            min_loss=("final_val_loss", "min"),
        )
        .reindex([c for c in condition_order if c in df["label"].values])
    )
    print(summary_table.to_string(float_format="%.4f"))
