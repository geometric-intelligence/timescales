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


def get_jacobian_eigenvalues(model: SchurRNN) -> np.ndarray:
    """Compute eigenvalues of J = Q T Q^T."""
    with torch.no_grad():
        Q = model.rnn_step._effective_Q()
        T = model.rnn_step.T
        if model.rnn_step.train_q and model.rnn_step.q_parameterization == "free":
            I = torch.eye(Q.shape[0], device=Q.device)
            J = Q @ T @ torch.linalg.solve(Q, I)
        else:
            J = Q @ T @ Q.T
    return np.linalg.eigvals(J.cpu().numpy())


# %% Compute eigenspectra
eigen_records = []

for _, row in df.iterrows():
    model = load_schur_model(row["seed_path"])
    if model is None:
        continue
    eigs = get_jacobian_eigenvalues(model)
    eigen_records.append(dict(
        label=row["label"],
        seed=row["seed"],
        eigenvalues=eigs,
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

# %% Plot 5: Eigenvalue magnitude distribution (histogram)
if eigen_records:
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab in COLORS:
        recs = [r for r in eigen_records if r["label"] == lab]
        if not recs:
            continue
        all_mags = np.concatenate([np.abs(r["eigenvalues"]) for r in recs])
        ax.hist(all_mags, bins=50, alpha=0.5, density=True,
                color=COLORS[lab], label=lab, edgecolor="none")

    ax.axvline(1.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("|$\\lambda$|", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Eigenvalue Magnitude Distribution",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIG_DIR, "schur_eigval_magnitudes.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% Plot 6: Effective timescale scree plot — tau_eff = -1/ln|lambda|
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
