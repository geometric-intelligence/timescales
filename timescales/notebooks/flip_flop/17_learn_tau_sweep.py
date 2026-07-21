# %% [markdown]
# # Learnable vs Fixed Time Constants — 6-bit Flip-Flop (Identity)
#
# Sweep: `flip_flop_learn_tau_sweep`
#
# Two conditions (fixed τ=1 vs per-neuron learnable τ, both init at τ=1),
# crossed with gains g ∈ {0.6, 0.7, 0.8, 0.9}. Identity activation, N=512,
# normal-scaled W_rec init.
#
# **Key question**: does allowing τ to be learned change which Jacobian
# eigenvalues the network finds, and does it help it build slow modes that
# match the bit hold intervals?
#
# **Analyses**
# 1. Training curves (loss & accuracy)
# 2. Learned τ distributions (learn=True only)
# 3. Jacobian eigenspectra (complex plane) — fixed vs learned
# 4. Effective timescale scree — τ_eff = −1/ln|λ|, with hold intervals marked
#    and best-coupled modes highlighted per bit

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_FIGS = True
FIGS_DIR = os.path.join("notebooks", "flip_flop", "figs", "learn_tau")
os.makedirs(FIGS_DIR, exist_ok=True)

# %% ── CONFIG ────────────────────────────────────────────────────────────────
sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/flip_flop_learn_tau_sweep_20260425_214053"

TAU_VIS_MAX = 1000  # cap on τ_eff for scree plots; None = no cap
GAINS = [0.6, 0.7, 0.8, 0.9]
CONDITIONS = ["False", "True"]   # learn_time_constants

# Bit hold intervals from p_pulse (in timesteps, not physical time)
P_PULSE = [0.005, 0.007, 0.01, 0.02, 0.05, 0.1]
HOLDS = [1.0 / p for p in P_PULSE]   # [200, ~143, 100, 50, 20, 10]
bit_colors = [f"C{i}" for i in range(len(HOLDS))]   # one colour per bit

# %% ── HELPERS ───────────────────────────────────────────────────────────────

def _compute_jacobian(ckpt_state: dict, run_config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the Jacobian J from a checkpoint state dict.

    For Identity activation, J is state-independent:
        J = diag(1 − α) + diag(α) · g · W_rec
    where α is per-neuron when learn_time_constants=True,
    or a scalar broadcast to all neurons when False.

    Returns (J, alphas) where alphas has shape (N,).
    """
    g = float(run_config["recurrent_gain"])
    dt = float(run_config["dt"])
    learn_tau = run_config["learn_time_constants"]

    W_rec = None
    log_tau = None
    for key, val in ckpt_state.items():
        if "W_rec.weight" in key:
            W_rec = val.numpy()
        if "log_time_constants" in key:
            log_tau = val.numpy()

    if W_rec is None:
        raise RuntimeError("W_rec.weight not found in checkpoint")

    N = W_rec.shape[0]

    if learn_tau and log_tau is not None:
        taus = np.exp(log_tau)          # (N,)
        alphas = 1.0 - np.exp(-dt / taus)  # (N,)
    else:
        tau_val = float(run_config["time_constants_config"]["values"][0])
        alpha_scalar = 1.0 - np.exp(-dt / tau_val)
        alphas = np.full(N, alpha_scalar)

    J = np.diag(1.0 - alphas) + alphas[:, np.newaxis] * g * W_rec
    return J, alphas


def _tau_eff_from_eigs(eigs: np.ndarray, cap: float | None = None) -> np.ndarray:
    """Effective timescale in timesteps: τ_eff = −1 / ln|λ|."""
    log_abs = np.log(np.clip(np.abs(eigs), 1e-12, None))
    tau = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)
    if cap is not None:
        tau = np.clip(tau, 0.0, cap)
    return tau


def _load_ckpt_state(seed_path: str, which: str = "trained") -> dict | None:
    """Load checkpoint state dict. which='trained'|'untrained'."""
    if which == "untrained":
        p = os.path.join(seed_path, "checkpoints", "untrained.ckpt")
        if os.path.exists(p):
            return torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        return None
    best = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best:
        return None
    return torch.load(best[0], map_location="cpu", weights_only=False)["state_dict"]


# Colour palette: gains → colour, conditions → linestyle / marker
_GAIN_COLORS = {0.6: "#2a9d8f", 0.7: "#e9c46a", 0.8: "#f4a261", 0.9: "#e63946"}
_COND_STYLE = {"False": dict(ls="-",  marker="o"), "True": dict(ls="--", marker="s")}

# %% ── LOAD RECORDS ──────────────────────────────────────────────────────────
assert sweep_dir, "Set sweep_dir above before running!"

records = []

for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
        continue

    # Parse experiment name: g_{gain}_learn_tau_{condition}
    parts = exp_name.split("_")
    try:
        assert parts[0] == "g"
        gain = float(parts[1])
        assert parts[2] == "learn" and parts[3] == "tau"
        condition = parts[4]   # "True" or "False"
    except (IndexError, ValueError, AssertionError):
        continue

    for sdn in sorted(os.listdir(exp_dir)):
        if not sdn.startswith("seed_"):
            continue
        seed = int(sdn.split("_")[1])
        seed_path = os.path.join(exp_dir, sdn)

        config_file = os.path.join(seed_path, "run_config.yaml")
        if not os.path.exists(config_file):
            continue
        with open(config_file) as f:
            run_config = yaml.safe_load(f)

        fvl = None
        rf = os.path.join(seed_path, "job_result.yaml")
        if os.path.exists(rf):
            with open(rf) as f:
                fvl = yaml.safe_load(f).get("final_val_loss")

        val_losses, val_accs, steps = [], [], []
        val_accs_per_bit = {}
        lf = os.path.join(seed_path, "training_losses.json")
        if os.path.exists(lf):
            with open(lf) as f:
                ld = json.load(f)
            val_losses = ld.get("val_losses", [])
            val_accs = ld.get("val_accuracies", [])
            steps = ld.get("steps", [])
            val_accs_per_bit = ld.get("val_accuracies_per_bit", {})

        if fvl is None and val_losses:
            fvl = val_losses[-1]
        if fvl is None:
            continue

        final_acc = val_accs[-1] if val_accs else None

        records.append(dict(
            exp_name=exp_name, gain=gain, condition=condition, seed=seed,
            seed_path=seed_path, run_config=run_config,
            final_val_loss=fvl, final_val_acc=final_acc,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
            val_accs_per_bit=val_accs_per_bit,
        ))

df = pd.DataFrame(records)
print(f"Loaded {len(df)} runs")
print(df[["gain", "condition", "final_val_loss", "final_val_acc"]].to_string(index=False))

# %% ── PLOT 1: Training loss curves ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

for ax, cond in zip(axes, CONDITIONS):
    sub = df[df["condition"] == cond].sort_values("gain")
    for _, row in sub.iterrows():
        g = row["gain"]
        st = row["steps"]
        vl = row["val_losses"]
        steps_x = st[:len(vl)] if st else list(range(len(vl)))
        ax.plot(steps_x, vl, color=_GAIN_COLORS[g], linewidth=1.8,
                label=f"g = {g}")
    label_str = "Learnable τ" if cond == "True" else "Fixed τ = 1"
    ax.set_title(label_str, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Validation loss", fontsize=11)
fig.suptitle("Training Loss — Fixed vs Learnable τ (6-bit Hetero Flip-Flop)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "01_train_loss.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% ── PLOT 2: Training accuracy curves ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

for ax, cond in zip(axes, CONDITIONS):
    sub = df[df["condition"] == cond].sort_values("gain")
    for _, row in sub.iterrows():
        g = row["gain"]
        st = row["steps"]
        va = row["val_accs"]
        steps_x = st[:len(va)] if st else list(range(len(va)))
        ax.plot(steps_x, va, color=_GAIN_COLORS[g], linewidth=1.8,
                label=f"g = {g}")
    label_str = "Learnable τ" if cond == "True" else "Fixed τ = 1"
    ax.set_title(label_str, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylim(0.3, 1.02)
    ax.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

axes[0].set_ylabel("Validation accuracy", fontsize=11)
fig.suptitle("Training Accuracy — Fixed vs Learnable τ",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "02_train_acc.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% ── PLOT 3: Final accuracy vs gain — conditions overlaid ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, metric, ylabel in zip(
    axes,
    ["final_val_acc", "final_val_loss"],
    ["Final validation accuracy", "Final validation loss"],
):
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond].sort_values("gain")
        if sub.empty:
            continue
        kw = _COND_STYLE[cond]
        label = "Learnable τ" if cond == "True" else "Fixed τ = 1"
        color = "#264653" if cond == "True" else "#e76f51"
        ax.plot(sub["gain"], sub[metric],
                color=color, linewidth=1.8, label=label,
                marker=kw["marker"], markersize=8, ls=kw["ls"])

    ax.set_xlabel("Recurrent gain $g$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    if "loss" in metric:
        ax.set_yscale("log")

fig.suptitle("Fixed vs Learnable τ — Summary by Gain",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "03_summary_by_gain.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% ── PLOT 4: Learned τ distributions (learn=True only) ────────────────────
sub_true = df[df["condition"] == "True"].sort_values("gain")

if not sub_true.empty:
    n_g = len(sub_true)
    fig, axes = plt.subplots(1, n_g, figsize=(4.2 * n_g, 3.8), sharey=False)
    if n_g == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sub_true.iterrows()):
        g = row["gain"]
        seed_path = row["seed_path"]
        run_config = row["run_config"]

        state = _load_ckpt_state(seed_path, "trained")
        if state is None:
            ax.text(0.5, 0.5, "no ckpt", transform=ax.transAxes, ha="center")
            continue

        log_tau = None
        for key, val in state.items():
            if "log_time_constants" in key:
                log_tau = val.numpy()
                break

        if log_tau is None:
            ax.text(0.5, 0.5, "no log_tau", transform=ax.transAxes, ha="center")
            continue

        taus = np.exp(log_tau)
        ax.hist(taus, bins=50, color=_GAIN_COLORS[g], alpha=0.8, edgecolor="none")

        for bi, h in enumerate(HOLDS):
            dt = float(run_config["dt"])
            tau_for_hold = h * dt   # physical time of hold
            # The hold interval in steps is h = 1/p, in physical time = h*dt
            # For reference, draw the "ideal" tau for each bit's hold interval
            ax.axvline(h * dt, color=f"C{bi}", ls="--", lw=0.9, alpha=0.7,
                       label=f"bit {bi} τ*={h*dt:.1f}")

        ax.axvline(1.0, color="black", ls=":", lw=1.2, alpha=0.6, label="init τ=1")
        ax.set_xlabel("Learned τ", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"g = {g}\nacc = {row['final_val_acc']:.3f}", fontsize=11)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Learned τ Distribution (per-neuron) — Learnable τ Condition",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, "04_tau_distributions.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% ── COMPUTE JACOBIANS AND EIGENVALUES ─────────────────────────────────────
# Store eig_data[gain][condition] = dict with eigs, eigvecs, W_out, alphas, taus
eig_data = {g: {} for g in GAINS}

for _, row in df.iterrows():
    g = row["gain"]
    cond = row["condition"]
    seed_path = row["seed_path"]
    run_config = row["run_config"]

    state_trained = _load_ckpt_state(seed_path, "trained")
    state_untrained = _load_ckpt_state(seed_path, "untrained")

    for label, state in [("trained", state_trained), ("untrained", state_untrained)]:
        if state is None:
            continue
        try:
            J, alphas = _compute_jacobian(state, run_config)
        except RuntimeError:
            continue

        eigs, eigvecs = np.linalg.eig(J)

        W_out = None
        for key, val in state.items():
            if "W_out.weight" in key:
                W_out = val.numpy()
                break

        # Extract final learned taus (only meaningful for trained + learn=True)
        taus = None
        for key, val in state.items():
            if "log_time_constants" in key:
                taus = np.exp(val.numpy())
                break

        eig_data[g][f"{cond}_{label}"] = dict(
            eigs=eigs, eigvecs=eigvecs, W_out=W_out,
            alphas=alphas, taus=taus,
        )

print("Eigenvalue data computed for:")
for g in GAINS:
    keys = list(eig_data[g].keys())
    print(f"  g={g}: {keys}")

# %% ── COMPUTE: correlation-based dominant eigenvalue per bit (all gains) ──────
# For each trained network (all gains × conditions), Schur-decompose J, run
# an identity-RNN forward pass, compute Pearson r(Schur mode, target bit), and
# record which eigenvalue in the complex plane is "dominant" per bit.
#
# Results: _dom_eig_idx[g][cond] = {bi: [idx, ...]}  (indices into eig_data eigs)
# Used in Plot 5 to colour the spectrum by bit instead of by |λ| rank.

from scipy.linalg import schur as _sp_schur_p5
from datamodules.flip_flop import FlipFlopDataModule as _FFdm_p5

_N_TRAJ_P5 = 50

_dom_eig_idx = {g: {} for g in GAINS}

for _, _p5_row in df.iterrows():
    _p5_g    = _p5_row["gain"]
    _p5_cond = _p5_row["condition"]
    _p5_key  = f"{_p5_cond}_trained"
    _p5_rc   = _p5_row["run_config"]
    _p5_sp   = _p5_row["seed_path"]

    if _p5_key not in eig_data[_p5_g]:
        continue

    _p5_state = _load_ckpt_state(_p5_sp, "trained")
    if _p5_state is None:
        continue

    _p5_Wrec = _p5_Win = None
    for _k5, _v5 in _p5_state.items():
        if "W_rec.weight" in _k5: _p5_Wrec = _v5.numpy()
        if "W_in.weight"  in _k5: _p5_Win  = _v5.numpy()
    if _p5_Wrec is None or _p5_Win is None:
        continue

    _p5_alphas = eig_data[_p5_g][_p5_key]["alphas"]
    _p5_te     = eig_data[_p5_g][_p5_key]["eigs"]
    _p5_gval   = float(_p5_rc["recurrent_gain"])
    _p5_N      = _p5_Wrec.shape[0]
    _p5_nbits  = _p5_rc["n_bits"]

    _p5_J = np.diag(1.0 - _p5_alphas) + _p5_alphas[:, None] * _p5_gval * _p5_Wrec

    # Schur decomp; assign one eigenvalue to each column
    _p5_T5, _p5_Q5 = _sp_schur_p5(_p5_J, output="real")
    _p5_col_ev = np.zeros(_p5_N, dtype=complex)
    _ki5 = 0
    while _ki5 < _p5_N:
        if _ki5 + 1 < _p5_N and abs(_p5_T5[_ki5 + 1, _ki5]) > 1e-10:
            _ev2 = np.linalg.eigvals(_p5_T5[_ki5:_ki5+2, _ki5:_ki5+2])
            _p5_col_ev[_ki5]     = _ev2[0]
            _p5_col_ev[_ki5 + 1] = _ev2[1]
            _ki5 += 2
        else:
            _p5_col_ev[_ki5] = _p5_T5[_ki5, _ki5]
            _ki5 += 1

    _p5_srt   = np.argsort(np.abs(_p5_col_ev))[::-1]
    _p5_Q5_s  = _p5_Q5[:, _p5_srt]
    _p5_ev5_s = _p5_col_ev[_p5_srt]

    # Identity-RNN forward pass
    _p5_dm = _FFdm_p5(
        n_bits=_p5_nbits, p_pulse=_p5_rc["p_pulse"],
        pulse_amplitude=_p5_rc["pulse_amplitude"],
        num_time_steps=_p5_rc["num_time_steps"],
        num_val_trajectories=_N_TRAJ_P5, batch_size=_N_TRAJ_P5,
    )
    _p5_dm.setup()
    _p5_inp5, _, _p5_tgt5 = [t.numpy() for t in _p5_dm.val_dataset.tensors]
    _p5_B5, _p5_TL5, _ = _p5_inp5.shape
    _p5_AW5 = _p5_alphas[:, None] * _p5_Win
    _p5_hbuf5 = np.zeros((_p5_B5, _p5_TL5, _p5_N), dtype=np.float32)
    _p5_ht5   = np.zeros((_p5_B5, _p5_N), dtype=np.float32)
    for _tt5 in range(_p5_TL5):
        _p5_ht5 = _p5_ht5 @ _p5_J.T + _p5_inp5[:, _tt5, :] @ _p5_AW5.T
        _p5_hbuf5[:, _tt5, :] = _p5_ht5
    _p5_hf5 = _p5_hbuf5.reshape(-1, _p5_N)
    _p5_tf5 = _p5_tgt5.reshape(-1, _p5_nbits)

    # Pearson r: Schur projections vs targets
    _p5_Z5  = _p5_hf5 @ _p5_Q5_s
    _p5_Zc5 = _p5_Z5  - _p5_Z5.mean(0, keepdims=True)
    _p5_Yc5 = _p5_tf5 - _p5_tf5.mean(0, keepdims=True)
    _p5_Zs5 = np.where(_p5_Zc5.std(0) > 1e-12, _p5_Zc5.std(0), 1e-12)
    _p5_Ys5 = np.where(_p5_Yc5.std(0) > 1e-12, _p5_Yc5.std(0), 1e-12)
    _p5_R5  = (_p5_Zc5.T @ _p5_Yc5) / (_p5_hf5.shape[0] * _p5_Zs5[:, None] * _p5_Ys5[None, :])

    # Map dominant Schur column → index in eig_data eigs array
    _dom_eig_idx[_p5_g][_p5_cond] = {}
    for _bi5 in range(_p5_nbits):
        _dom_col5   = int(np.argmax(np.abs(_p5_R5[:, _bi5])))
        _target_ev5 = _p5_ev5_s[_dom_col5]
        _best5      = int(np.argmin(np.abs(_p5_te - _target_ev5)))
        _idxs5      = [_best5]
        if not np.isreal(_target_ev5):
            _conj5 = int(np.argmin(np.abs(_p5_te - np.conj(_target_ev5))))
            if _conj5 != _best5:
                _idxs5.append(_conj5)
        _dom_eig_idx[_p5_g][_p5_cond][_bi5] = _idxs5

    _rstr = [f"b{bi}:r={_p5_R5[np.argmax(np.abs(_p5_R5[:,bi])),bi]:+.2f}" for bi in range(_p5_nbits)]
    print(f"  g={_p5_g} {_p5_cond}: {_rstr}")

print("Correlation dominant eigenvalue mapping complete.")


# %% ── PLOT 5: Jacobian eigenspectra — complex plane ─────────────────────────
# Rings = dominant Schur mode per bit (by Pearson r). Bulk = salmon.
# Requires: eig_data, _dom_eig_idx, bit_colors, df, GAINS, CONDITIONS

theta = np.linspace(0, 2 * np.pi, 300)
n_g = len(GAINS)

fig, axes = plt.subplots(n_g, 2, figsize=(9, 3.8 * n_g), squeeze=False)

col_labels = ["Fixed τ = 1", "Learnable τ"]

for row_idx, g in enumerate(GAINS):
    for col_idx, cond in enumerate(CONDITIONS):
        ax = axes[row_idx, col_idx]
        key_trained = f"{cond}_trained"
        key_untrained = f"{cond}_untrained"

        ax.plot(np.cos(theta), np.sin(theta),
                color="#c0392b", lw=1.0, ls="--", alpha=0.5, zorder=1)
        ax.axhline(0, color="#ccc", lw=0.3)
        ax.axvline(0, color="#ccc", lw=0.3)

        if key_untrained in eig_data[g]:
            ue = eig_data[g][key_untrained]["eigs"]
            ax.scatter(ue.real, ue.imag, s=8, color="#c0d8e8",
                       alpha=0.5, edgecolors="none", zorder=2, label="untrained")

        if key_trained in eig_data[g]:
            te     = eig_data[g][key_trained]["eigs"]
            abs_te = np.abs(te)
            n_eig  = len(te)

            # Indices of dominant eigenvalues (one or two per bit for conj pairs)
            _corr_dom5 = _dom_eig_idx.get(g, {}).get(cond, {})
            _dom_set5  = set()
            _idx_to_bit5 = {}
            for _bi5p, _idxs5p in _corr_dom5.items():
                for _ii5p in _idxs5p:
                    _dom_set5.add(_ii5p)
                    _idx_to_bit5[_ii5p] = _bi5p

            _rest5 = np.array([i not in _dom_set5 for i in range(n_eig)])

            ax.scatter(te.real[_rest5], te.imag[_rest5], s=12,
                       color="#e76f51", alpha=0.6, edgecolors="none", zorder=3)

            _seen_bits5 = set()
            for _ii5p, _bi5p in _idx_to_bit5.items():
                _lbl5 = f"bit {_bi5p}" if _bi5p not in _seen_bits5 else "_nolegend_"
                _seen_bits5.add(_bi5p)
                ax.scatter(te.real[_ii5p], te.imag[_ii5p], s=40,
                           color=bit_colors[_bi5p],
                           edgecolors="white", linewidths=0.1, zorder=6,
                           label=_lbl5)

            if not _corr_dom5:
                _fb5 = np.argsort(abs_te)[-len(HOLDS):]
                ax.scatter(te.real[_fb5], te.imag[_fb5], s=35,
                           color="#e76f51", edgecolors="black", linewidths=0.5,
                           zorder=5, label=f"top {len(HOLDS)} |λ|")

            # ax.annotate(f"|λ|_max={np.max(abs_te):.3f}",
            #             xy=(0.97, 0.05), xycoords="axes fraction",
            #             ha="right", fontsize=8,
            #             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        row_data = df[(df["gain"] == g) & (df["condition"] == cond)]
        # if not row_data.empty:
            # acc = row_data.iloc[0]["final_val_acc"]
            # ax.annotate(f"acc={acc:.3f}" if acc is not None else "",
            #             xy=(0.03, 0.95), xycoords="axes fraction",
            #             ha="left", va="top", fontsize=8,
            #             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        ax.set_aspect("equal")
        ax.set_xlim(0.7, 1.02)
        ax.set_ylim(-0.2, 0.2)
        ax.grid(True, alpha=0.1)

        if row_idx == 0:
            ax.set_title(col_labels[col_idx], fontsize=12, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(f"g = {g}\nIm(λ)", fontsize=10)
        if row_idx == n_g - 1:
            ax.set_xlabel("Re(λ)", fontsize=10)
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=7, loc="lower left", ncol=2)

fig.suptitle(
    "Jacobian Eigenspectrum — Fixed vs Learnable τ\n"
    "Rings = dominant Schur mode per bit (Pearson r with target)",
    fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "05_eigenspectra.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% ── PLOT 6: τ_eff scree — rank vs effective timescale ────────────────────
# Grid: rows = gains, cols = conditions (Fixed | Learnable).
# Dots coloured by bit-coupling strength; horizontal dashed lines at hold intervals.

bit_colors = [f"C{i}" for i in range(len(HOLDS))]

fig, axes = plt.subplots(n_g, 2, figsize=(12, 3.5 * n_g), squeeze=False, sharey=True)

for row_idx, g in enumerate(GAINS):
    for col_idx, cond in enumerate(CONDITIONS):
        ax = axes[row_idx, col_idx]
        key = f"{cond}_trained"

        if key not in eig_data[g]:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue

        data = eig_data[g][key]
        eigs = data["eigs"]
        V = data["eigvecs"]
        W_out = data["W_out"]
        N = len(eigs)

        # Sort by |λ| descending
        abs_rank = np.argsort(np.abs(eigs))[::-1]
        eigs_sorted = eigs[abs_rank]
        tau_eff = _tau_eff_from_eigs(eigs_sorted, cap=TAU_VIS_MAX)
        ranks = np.arange(1, N + 1)

        # Per-bit coupling to identify best-matching mode
        if W_out is not None:
            coupling = np.abs(W_out @ V[:, abs_rank])   # (n_bits, N)
            best_rank_per_bit = {bi: int(np.argmax(coupling[bi])) for bi in range(len(HOLDS))}
        else:
            coupling = None
            best_rank_per_bit = {}

        highlighted = set(best_rank_per_bit.values())
        non_highlighted = [r for r in range(N) if r not in highlighted]

        ax.scatter(ranks[non_highlighted], tau_eff[non_highlighted],
                   s=10, color="#adb5bd", alpha=0.55, edgecolors="none", zorder=2)

        for bi, ri in best_rank_per_bit.items():
            c = bit_colors[bi]
            ax.scatter(ranks[ri], tau_eff[ri], s=20, color=c,
                       edgecolors="black", linewidths=0.6, zorder=5)
            ax.annotate(f"b{bi} τ={tau_eff[ri]:.0f}",
                        (ranks[ri], tau_eff[ri]),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=7, color=c, fontweight="bold", va="center",
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.4, alpha=0.4))

        for bi, h in enumerate(HOLDS):
            ax.axhline(h, color=bit_colors[bi], lw=0.9, ls=":", alpha=0.55)

        ax.set_yscale("log")
        ax.set_xlim(0, N + 1)
        ax.grid(True, alpha=0.15)

        row_data = df[(df["gain"] == g) & (df["condition"] == cond)]
        acc_str = ""
        if not row_data.empty and row_data.iloc[0]["final_val_acc"] is not None:
            acc_str = f" | acc={row_data.iloc[0]['final_val_acc']:.3f}"

        if row_idx == 0:
            cond_label = "Fixed τ = 1" if cond == "False" else "Learnable τ"
            ax.set_title(cond_label, fontsize=12, fontweight="bold")
        ax.set_title(
            (("Fixed τ = 1" if cond == "False" else "Learnable τ") if row_idx == 0 else "")
            + (f"\ng={g}{acc_str}" if row_idx > 0 else f" g={g}{acc_str}"),
            fontsize=10,
        )
        if col_idx == 0:
            ax.set_ylabel("τ_eff (steps)", fontsize=10)
        if row_idx == n_g - 1:
            ax.set_xlabel("Eigenmode rank (by |λ|)", fontsize=10)

# Add hold-interval legend (bit colours) once outside the grid
legend_lines = [Line2D([0], [0], color=bit_colors[bi], ls=":",
                        label=f"bit {bi} hold≈{HOLDS[bi]:.0f} steps")
                for bi in range(len(HOLDS))]
fig.legend(handles=legend_lines, fontsize=8, loc="upper right",
           bbox_to_anchor=(1.01, 0.98), framealpha=0.9)

fig.suptitle("τ_eff Scree — Fixed vs Learnable τ (colours = best-coupled mode per bit)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "06_tau_scree.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% ── PLOT 6b: Learned τ scree (learn=True only) ───────────────────────────
# Sorted learned per-neuron τ values vs neuron rank, with the "ideal" τ for
# each bit's hold interval marked as horizontal dashed lines.
# (Different from the τ_eff scree above: these are the raw learned time
# constants, not the effective timescales derived from Jacobian eigenvalues.)

sub_true = df[df["condition"] == "True"].sort_values("gain")

if not sub_true.empty:
    n_g_true = len(sub_true)
    fig, axes = plt.subplots(1, n_g_true, figsize=(4.5 * n_g_true, 4.0),
                             sharey=True, squeeze=False)

    for col, (_, row) in enumerate(sub_true.iterrows()):
        ax = axes[0, col]
        g = row["gain"]
        run_config = row["run_config"]
        dt = float(run_config["dt"])
        key = "True_trained"

        if key not in eig_data[g] or eig_data[g][key]["taus"] is None:
            ax.text(0.5, 0.5, "no learned τ", transform=ax.transAxes, ha="center")
            continue

        taus_learned = eig_data[g][key]["taus"]   # (N,) per-neuron learned τ
        taus_sorted = np.sort(taus_learned)[::-1]
        ranks = np.arange(1, len(taus_sorted) + 1)

        ax.scatter(ranks, taus_sorted, s=8, color=_GAIN_COLORS[g], alpha=0.6,
                   edgecolors="none", rasterized=True)

        # Ideal τ for each bit: hold_interval_in_steps * dt = physical hold time.
        # The network needs to integrate over the hold period; a rough heuristic
        # for the required τ is that alpha ≈ dt/τ → τ ≈ dt * hold_steps = hold_phys.
        for bi, h in enumerate(HOLDS):
            tau_ideal = h * dt   # physical time of hold interval
            ax.axhline(tau_ideal, color=bit_colors[bi], lw=0.9, ls="--", alpha=0.7,
                       label=f"bit {bi} hold×dt={tau_ideal:.2f}")

        ax.axhline(1.0, color="black", lw=1.1, ls=":", alpha=0.5, label="init τ=1")
        ax.set_xlabel("Neuron rank", fontsize=11)
        if col == 0:
            ax.set_ylabel("Learned τ", fontsize=11)
        ax.set_yscale("log")
        acc_str = f"  acc={row['final_val_acc']:.3f}" if row["final_val_acc"] is not None else ""
        ax.set_title(f"g = {g}{acc_str}", fontsize=10)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Learned τ Scree (per-neuron, learnable τ condition)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, "06b_learned_tau_scree.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% ── PLOT 7: Mode-to-output coupling ───────────────────────────────────────
# |W_out @ V|: coupling of each Jacobian eigenmode (ranked by |λ|) to each
# output bit. Reveals whether the output reads out from a sparse set of modes
# or is spread across the spectrum.
#
# Each line is one bit; the x-axis is eigenmode rank (1 = largest |λ|).

fig, axes = plt.subplots(n_g, 2, figsize=(12, 3.5 * n_g), squeeze=False)

for row_idx, g in enumerate(GAINS):
    for col_idx, cond in enumerate(CONDITIONS):
        ax = axes[row_idx, col_idx]
        key = f"{cond}_trained"

        if key not in eig_data[g]:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue

        data = eig_data[g][key]
        eigs = data["eigs"]
        V = data["eigvecs"]
        W_out = data["W_out"]
        N = len(eigs)

        if W_out is None:
            ax.text(0.5, 0.5, "no W_out", transform=ax.transAxes, ha="center")
            continue

        abs_rank = np.argsort(np.abs(eigs))[::-1]
        coupling = np.abs(W_out @ V[:, abs_rank])   # (n_bits, N)
        ranks = np.arange(1, N + 1)

        for bi in range(len(HOLDS)):
            ax.plot(ranks, np.exp(coupling[bi]), color=bit_colors[bi], lw=1.0,
                    alpha=0.8, label=f"bit {bi}", linestyle="None", marker="o")

        #ax.set_yscale("log")
        #ax.set_xlim(0, N + 1)
        ax.set_xlim(0, 20)
        ax.grid(True, alpha=0.15)

        cond_label = "Fixed τ = 1" if cond == "False" else "Learnable τ"
        row_data = df[(df["gain"] == g) & (df["condition"] == cond)]
        acc_str = ""
        if not row_data.empty and row_data.iloc[0]["final_val_acc"] is not None:
            acc_str = f" | acc={row_data.iloc[0]['final_val_acc']:.3f}"
        ax.set_title(f"{cond_label}  g={g}{acc_str}", fontsize=10)

        if col_idx == 0:
            ax.set_ylabel("|W_out @ v_k|", fontsize=10)
        if row_idx == n_g - 1:
            ax.set_xlabel("Eigenmode rank (by |λ|)", fontsize=10)
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

fig.suptitle("Mode-to-Output Coupling — Fixed vs Learnable τ",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "07_mode_coupling.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% ── PLOT 8: Dominant mode localisation (Participation Ratio) ──────────────
# For the best-coupled Jacobian eigenmode per output bit, measure how
# localised vs delocalised the corresponding eigenvector is across neurons.
#
#   PR(v) = 1 / Σ_i p_i²   where  p_i = |v_i|² / Σ_j |v_j|²
#
# PR = 1  → perfectly localised (all weight on one neuron).
# PR = N  → perfectly delocalised (equal weight on all neurons).
# Plotted as PR/N ∈ [1/N, 1] so it has a natural [0, 1] scale.


def participation_ratio(v: np.ndarray) -> float:
    """Participation ratio of a (possibly complex) vector v."""
    prob = np.abs(v) ** 2
    prob = prob / (prob.sum() + 1e-30)
    return float(1.0 / (np.sum(prob ** 2) + 1e-30))


pr_records = []
for g in GAINS:
    for cond in CONDITIONS:
        key = f"{cond}_trained"
        if key not in eig_data[g]:
            continue
        data = eig_data[g][key]
        eigs = data["eigs"]
        V = data["eigvecs"]
        W_out = data["W_out"]
        N = len(eigs)
        if W_out is None:
            continue
        abs_rank = np.argsort(np.abs(eigs))[::-1]
        coupling = np.abs(W_out @ V[:, abs_rank])   # (n_bits, N)
        for bi in range(len(HOLDS)):
            best_rank = int(np.argmax(coupling[bi]))
            v_best = V[:, abs_rank[best_rank]]
            pr = participation_ratio(v_best)
            tau_dom = _tau_eff_from_eigs(eigs[abs_rank[best_rank:best_rank + 1]])[0]
            pr_records.append(dict(
                gain=g, condition=cond, bit=bi,
                pr=pr, pr_norm=pr / N, tau_eff_dom=tau_dom,
            ))

df_pr = pd.DataFrame(pr_records)

# Bar chart: PR/N grouped by gain, one panel per condition
x = np.arange(len(GAINS))
bar_width = 0.12
n_bits_plot = len(HOLDS)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

for col_idx, cond in enumerate(CONDITIONS):
    ax = axes[col_idx]
    sub = df_pr[df_pr["condition"] == cond]
    if sub.empty:
        continue
    for bi in range(n_bits_plot):
        sub_b = sub[sub["bit"] == bi].sort_values("gain")
        offset = (bi - n_bits_plot / 2 + 0.5) * bar_width
        ax.bar(x + offset, sub_b["pr_norm"].values, width=bar_width,
               color=bit_colors[bi], alpha=0.85, label=f"bit {bi}",
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"g={g}" for g in GAINS], fontsize=10)
    ax.set_xlabel("Gain", fontsize=11)
    cond_label = "Fixed τ = 1" if cond == "False" else "Learnable τ"
    ax.set_title(cond_label, fontsize=12, fontweight="bold")
    ax.axhline(1.0, color="black", ls=":", lw=0.7, alpha=0.4, label="fully delocalised")
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 1.08)
    if col_idx == 0:
        ax.set_ylabel("PR / N  (0 = localised, 1 = delocalised)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Dominant Mode Localisation — Participation Ratio / N (per bit)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "08_mode_localization.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% ── PLOT 8b: Eigenvector weight profile for dominant modes ─────────────────
# For a representative gain, show the sorted |v_i|² profile of the dominant
# mode per bit. A steep drop-off = localised; a flat plateau = delocalised.

REP_GAIN = GAINS[-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)

for col_idx, cond in enumerate(CONDITIONS):
    ax = axes[col_idx]
    key = f"{cond}_trained"
    if key not in eig_data[REP_GAIN]:
        continue
    data = eig_data[REP_GAIN][key]
    eigs = data["eigs"]
    V = data["eigvecs"]
    W_out = data["W_out"]
    N = len(eigs)
    if W_out is None:
        continue
    abs_rank = np.argsort(np.abs(eigs))[::-1]
    coupling = np.abs(W_out @ V[:, abs_rank])
    for bi in range(len(HOLDS)):
        best_rank = int(np.argmax(coupling[bi]))
        v_best = V[:, abs_rank[best_rank]]
        weights = np.abs(v_best) ** 2
        weights_sorted = np.sort(weights)[::-1] / (weights.max() + 1e-30)
        pr = participation_ratio(v_best)
        ax.plot(np.arange(1, N + 1), weights_sorted,
                color=bit_colors[bi], lw=1.2, alpha=0.8,
                label=f"bit {bi}  PR={pr:.0f}")

    ax.set_xlabel("Neuron rank (by |v_i|²)", fontsize=11)
    cond_label = "Fixed τ = 1" if cond == "False" else "Learnable τ"
    ax.set_title(f"{cond_label}  g = {REP_GAIN}", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8)
    if col_idx == 0:
        ax.set_ylabel("|v_i|² / max_j |v_j|²", fontsize=10)

fig.suptitle(f"Eigenvector Weight Profile — Dominant Modes per Bit  (g = {REP_GAIN})",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "08b_eigvec_profile.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% ── PLOT 9: |λ| rank (spectral radius profile) ───────────────────────────
fig, axes = plt.subplots(n_g, 2, figsize=(12, 3.5 * n_g), squeeze=False, sharey=True)

for row_idx, g in enumerate(GAINS):
    for col_idx, cond in enumerate(CONDITIONS):
        ax = axes[row_idx, col_idx]
        key = f"{cond}_trained"
        key_un = f"{cond}_untrained"

        for k, style in [(key_un, dict(color="#c0d8e8", lw=1.0, ls="-", alpha=0.7, label="untrained")),
                         (key,    dict(color="#e76f51", lw=1.6, ls="-", alpha=0.9, label="trained"))]:
            if k not in eig_data[g]:
                continue
            abs_eigs = np.sort(np.abs(eig_data[g][k]["eigs"]))[::-1]
            ax.plot(np.arange(1, len(abs_eigs) + 1), abs_eigs, **style)

        ax.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.4, label="|λ|=1")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.15)

        cond_label = "Fixed τ = 1" if cond == "False" else "Learnable τ"
        ax.set_title(f"{cond_label}  g={g}", fontsize=10)

        if col_idx == 0:
            ax.set_ylabel("|λ|", fontsize=10)
        if row_idx == n_g - 1:
            ax.set_xlabel("Rank", fontsize=10)
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=8)

fig.suptitle("|λ| Rank Profile — Fixed vs Learnable τ",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "07_eigval_mag_rank.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% ── PRINT SUMMARY ─────────────────────────────────────────────────────────
print("\n=== Summary: final val accuracy ===")
print(f"{'Gain':>6}  {'Fixed τ':>10}  {'Learnable τ':>12}")
for g in GAINS:
    row_f = df[(df["gain"] == g) & (df["condition"] == "False")]
    row_t = df[(df["gain"] == g) & (df["condition"] == "True")]
    acc_f = f"{row_f.iloc[0]['final_val_acc']:.4f}" if not row_f.empty and row_f.iloc[0]["final_val_acc"] is not None else "n/a"
    acc_t = f"{row_t.iloc[0]['final_val_acc']:.4f}" if not row_t.empty and row_t.iloc[0]["final_val_acc"] is not None else "n/a"
    print(f"  {g:>4}  {acc_f:>10}  {acc_t:>12}")

print("\n=== Top-6 τ_eff vs required hold intervals (trained) ===")
print(f"Required holds (steps): {[f'{h:.0f}' for h in HOLDS]}")
for g in GAINS:
    for cond in CONDITIONS:
        key = f"{cond}_trained"
        if key not in eig_data[g]:
            continue
        eigs = eig_data[g][key]["eigs"]
        abs_rank = np.argsort(np.abs(eigs))[::-1]
        top_taus = _tau_eff_from_eigs(eigs[abs_rank[:len(HOLDS)]])
        cond_label = "Fixed" if cond == "False" else "Learn"
        print(f"  g={g} [{cond_label}]: {[f'{t:.1f}' for t in top_taus]}")

# %% [markdown]
# ## Theory: Untrained Jacobian Spectrum Under Different τ Distributions
#
# For **uniform** α all neurons share the same update rate, so J = (1−α)I + αgW
# commutes with W and its eigenvectors are identical to W's. The eigenspectrum
# is simply an affine rescaling of W's bulk disk.
#
# For **non-uniform** α (diagonal A = diag(α₁,…,α_N)), J = (I−A) + AgW does
# **not** in general commute with W, so J and W have different eigenvectors.
# The shape of the bulk in the complex plane and the τ_eff profile both change
# depending on the distribution of α (equivalently, of τ).
#
# Below we compare three τ distributions at initialisation, keeping g, N, dt,
# and the random W fixed so that any differences come purely from the τ choice.

# %% [markdown]
# ### Setup — shared random W

# %%
RNG_SEED_TH = 42
N_TH = 512
G_TH = 0.9
DT_TH = 0.1   # matches flip_flop base config

rng_th = np.random.default_rng(RNG_SEED_TH)
W_th = rng_th.normal(0.0, 1.0 / N_TH ** 0.5, size=(N_TH, N_TH))

tau_dists = {
    "Uniform  τ = 1": np.ones(N_TH),
    "Power-law τ\n(log-uniform [0.1, 100])": np.exp(
        rng_th.uniform(np.log(0.1), np.log(100.0), size=N_TH)
    ),
    "Gaussian τ\n(μ=5, σ=3, clip [0.1, 50])": np.clip(
        rng_th.normal(5.0, 3.0, size=N_TH), 0.1, 50.0
    ),
}

th_colors = ["#264653", "#2a9d8f", "#e76f51"]
dist_names = list(tau_dists.keys())
n_dists = len(dist_names)

th_eigs = {}
th_alphas = {}
for name, taus_th in tau_dists.items():
    alphas_th = 1.0 - np.exp(-DT_TH / taus_th)
    J_th = np.diag(1.0 - alphas_th) + alphas_th[:, np.newaxis] * G_TH * W_th
    th_eigs[name] = np.linalg.eigvals(J_th)
    th_alphas[name] = alphas_th

# %% [markdown]
# ### T1 — τ and α distributions

# %%
fig, axes = plt.subplots(2, n_dists, figsize=(4.5 * n_dists, 6.5))

for col, (name, color) in enumerate(zip(dist_names, th_colors)):
    taus_th = tau_dists[name]
    alphas_th = th_alphas[name]

    ax_tau = axes[0, col]
    ax_tau.hist(taus_th, bins=40, color=color, alpha=0.8, edgecolor="none")
    ax_tau.axvline(taus_th.mean(), color="black", ls="--", lw=1.0,
                   label=f"mean={taus_th.mean():.2f}")
    ax_tau.set_title(name, fontsize=10, fontweight="bold")
    ax_tau.set_xlabel("τ", fontsize=10)
    ax_tau.legend(fontsize=8)
    ax_tau.grid(True, alpha=0.2)
    if col == 0:
        ax_tau.set_ylabel("Count", fontsize=10)

    ax_a = axes[1, col]
    ax_a.hist(alphas_th, bins=40, color=color, alpha=0.8, edgecolor="none")
    ax_a.axvline(alphas_th.mean(), color="black", ls="--", lw=1.0,
                 label=f"mean={alphas_th.mean():.3f}")
    ax_a.set_xlabel("α = 1 − exp(−dt/τ)", fontsize=10)
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.2)
    if col == 0:
        ax_a.set_ylabel("Count", fontsize=10)

fig.suptitle(f"τ and α Distributions (untrained, N={N_TH}, dt={DT_TH})",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "T1_tau_alpha_dists.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ### T2 — Jacobian eigenspectra (complex plane)

# %%
theta_c = np.linspace(0, 2 * np.pi, 300)

fig, axes = plt.subplots(1, n_dists, figsize=(4.5 * n_dists, 4.5))

for ax, name, color in zip(axes, dist_names, th_colors):
    eigs_th = th_eigs[name]
    abs_th = np.abs(eigs_th)
    ax.plot(np.cos(theta_c), np.sin(theta_c), "k--", lw=0.8, alpha=0.4)
    ax.axhline(0, color="#ccc", lw=0.3)
    ax.axvline(0, color="#ccc", lw=0.3)
    ax.scatter(eigs_th.real, eigs_th.imag, s=8, color=color,
               alpha=0.6, edgecolors="none", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(name + f"\n|λ|_max = {abs_th.max():.4f}", fontsize=10)
    ax.set_xlabel("Re(λ)", fontsize=10)
    ax.grid(True, alpha=0.1)
    if ax is axes[0]:
        ax.set_ylabel("Im(λ)", fontsize=10)

fig.suptitle(f"Untrained Jacobian Eigenspectrum  (g={G_TH}, dt={DT_TH}, same W)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "T2_theory_eigenspectra.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ### T3 — τ_eff scree of untrained network

# %%
fig, axes = plt.subplots(1, n_dists, figsize=(4.5 * n_dists, 4.0), sharey=True)

for ax, name, color in zip(axes, dist_names, th_colors):
    eigs_th = th_eigs[name]
    abs_rank_th = np.argsort(np.abs(eigs_th))[::-1]
    tau_eff_th = _tau_eff_from_eigs(eigs_th[abs_rank_th], cap=TAU_VIS_MAX)
    ranks_th = np.arange(1, N_TH + 1)

    ax.scatter(ranks_th, tau_eff_th, s=6, color=color, alpha=0.6,
               edgecolors="none", rasterized=True)
    for bi, h in enumerate(HOLDS):
        ax.axhline(h, color=bit_colors[bi], lw=0.8, ls=":", alpha=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Mode rank (by |λ|)", fontsize=10)
    ax.set_title(name, fontsize=10)
    ax.grid(True, alpha=0.15)
    if ax is axes[0]:
        ax.set_ylabel("τ_eff (steps)", fontsize=10)

# hold-interval legend
leg_lines_th = [Line2D([0], [0], color=bit_colors[bi], ls=":",
                        label=f"bit {bi} hold≈{HOLDS[bi]:.0f}")
                for bi in range(len(HOLDS))]
axes[-1].legend(handles=leg_lines_th, fontsize=7, loc="upper right")

fig.suptitle(f"τ_eff Scree — Untrained (g={G_TH}, same W, different τ distributions)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "T3_theory_tau_scree.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# G_FOCUS = 0.6 deep-dive — coupling heatmaps + correlation scree
# (Fixed τ=1 and Learnable τ side by side)
# ══════════════════════════════════════════════════════════════════════════════

# %% G0.6 — imports & config
from scipy.linalg import schur as _scipy_schur
from datamodules.flip_flop import FlipFlopDataModule

G_FOCUS_17  = 0.6   # gain to analyse
N_TRAJ_17   = 50    # trajectories for correlation forward pass
N_HM_17     = 12    # heatmap x-axis width (top-k modes / neurons)

_COND_LABELS = {"False": "Fixed τ = 1", "True": "Learnable τ"}

# %% G0.6 — helpers

def _schur_sort_17(J: np.ndarray):
    """Real Schur decomp of J; columns sorted by |λ| descending."""
    T_mat, Q_mat = _scipy_schur(J, output="real")
    N = T_mat.shape[0]
    col_abs = np.zeros(N)
    blk_lbl = np.empty(N, dtype=object)
    k = 0
    while k < N:
        if k + 1 < N and abs(T_mat[k + 1, k]) > 1e-10:
            ae = np.abs(np.linalg.eigvals(T_mat[k:k+2, k:k+2])[0])
            col_abs[k] = col_abs[k+1] = ae
            blk_lbl[k] = "2x2-a"; blk_lbl[k+1] = "2x2-b"
            k += 2
        else:
            col_abs[k] = abs(T_mat[k, k])
            blk_lbl[k] = "1x1"
            k += 1
    idx = np.argsort(col_abs)[::-1]
    Q_s = Q_mat[:, idx]
    abs_s = col_abs[idx]
    log_a = np.log(np.clip(abs_s, 1e-12, None))
    tau_s = -1.0 / np.where(log_a < -1e-10, log_a, -1e-10)
    return Q_s, abs_s, tau_s, blk_lbl[idx]


def _pearson_r_17(Z, Y):
    """Pearson r: shape (K_z, K_y). Z:(M,Kz), Y:(M,Ky)."""
    n = Z.shape[0]
    Zc = Z - Z.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Zs = np.where(Zc.std(0) > 1e-12, Zc.std(0), 1e-12)
    Ys = np.where(Yc.std(0) > 1e-12, Yc.std(0), 1e-12)
    return (Zc.T @ Yc) / (n * Zs[:, None] * Ys[None, :])


# %% G0.6 — COMPUTE: Schur, coupling matrices, forward pass, correlations
# Produces _g06[cond] with everything needed for the plots below.

_g06 = {}   # {cond_str: dict}

for _cond in CONDITIONS:
    _row_g = df[(df["gain"] == G_FOCUS_17) & (df["condition"] == _cond)]
    if _row_g.empty:
        print(f"No data for g={G_FOCUS_17}, cond={_cond}")
        continue
    _row    = _row_g.iloc[0]
    _sp     = _row["seed_path"]
    _rc     = _row["run_config"]
    _nbits  = _rc["n_bits"]
    _ppl    = _rc["p_pulse"] if isinstance(_rc["p_pulse"], list) else [_rc["p_pulse"]] * _nbits

    _state = _load_ckpt_state(_sp, "trained")
    if _state is None:
        print(f"  No checkpoint for cond={_cond}"); continue

    # ── Weights ──────────────────────────────────────────────────────────────
    _W_rec = _W_out = _W_in = None
    for _k, _v in _state.items():
        if "W_rec.weight" in _k: _W_rec = _v.numpy()
        if "W_out.weight" in _k: _W_out = _v.numpy()
        if "W_in.weight"  in _k: _W_in  = _v.numpy()

    # ── Jacobian + Schur ─────────────────────────────────────────────────────
    _J, _alphas = _compute_jacobian(_state, _rc)
    _N = _J.shape[0]
    _Qf, _abs_ef, _tauf, _blkf = _schur_sort_17(_J)

    # ── Eigenmode decomposition ───────────────────────────────────────────────
    _ef_raw, _Vf_raw = np.linalg.eig(_J)                    # complex, original order
    _eig_srt   = np.argsort(np.abs(_ef_raw))[::-1]
    _ef_s      = _ef_raw[_eig_srt]                           # (N,) sorted eigenvalues
    _Vf_s      = _Vf_raw[:, _eig_srt]                       # sorted eigenvectors
    _log_ae    = np.log(np.clip(np.abs(_ef_s), 1e-12, None))
    _tau_eig   = -1.0 / np.where(_log_ae < -1e-10, _log_ae, -1e-10)   # (N,)
    _Vf_inv    = np.linalg.pinv(_Vf_s)                      # (N, N)

    # ── Per-neuron τ_eff via dominant Schur component ─────────────────────────
    # Neuron k's timescale = τ_eff of the Schur column it loads onto most.
    _dom_schur_k = np.argmax(np.abs(_Qf), axis=1)           # (N,) → Schur rank per neuron
    _tau_neu_all = _tauf[_dom_schur_k]                       # (N,) τ_eff per neuron

    # Learned τ per neuron (or None for fixed-τ run)
    _log_tau_raw = None
    for _kk, _vv in _state.items():
        if "log_time_constants" in _kk:
            _log_tau_raw = _vv.numpy()
    _learned_tau_all = np.exp(_log_tau_raw) if _log_tau_raw is not None else None

    # ── Connectivity coupling matrices ────────────────────────────────────────
    _coup_schur = np.abs(_W_out @ _Qf)                       # (n_bits, N) sorted by |λ|
    _coup_eig   = np.abs(_W_out @ _Vf_s).real                # (n_bits, N) sorted by |λ|
    _cout       = np.abs(_W_out)                              # (n_bits, H) original order
    _cout_nro   = np.argsort(np.linalg.norm(_cout, axis=0))[::-1]
    _cout_s     = _cout[:, _cout_nro]                        # sorted by ||

    _cin = _win_s = None
    if _W_in is not None:
        # Input → Schur: α|Q^T W_in|  (per-neuron α for learnable case)
        _cin    = np.abs(_Qf.T @ (_alphas[:, None] * _W_in)) # (N, n_bits)
        _win    = np.abs(_W_in)                               # (H, n_bits)
        _win_nro = np.argsort(np.linalg.norm(_win, axis=1))[::-1]
        _win_s  = _win[_win_nro, :]

    # ── Forward pass via manual integration (Identity RNN, no model object needed)
    # h[t+1] = J @ h[t] + diag(α) W_in u[t]   (exact for Identity activation)
    _dm = FlipFlopDataModule(
        n_bits=_nbits, p_pulse=_rc["p_pulse"],
        pulse_amplitude=_rc["pulse_amplitude"],
        num_time_steps=_rc["num_time_steps"],
        num_val_trajectories=N_TRAJ_17, batch_size=N_TRAJ_17,
    )
    _dm.setup()
    _inp_raw, _, _tgt_raw = _dm.val_dataset.tensors   # (B, T, nbits)
    _inp_np   = _inp_raw.numpy()
    _tgt_flat = _tgt_raw.numpy().reshape(-1, _nbits)

    if _W_in is not None:
        _B17, _T17, _ = _inp_np.shape
        _AW17 = _alphas[:, None] * _W_in               # (N, n_in)
        _h_np = np.zeros((_B17, _T17, _N), dtype=np.float32)
        _ht17 = np.zeros((_B17, _N), dtype=np.float32)
        for _t in range(_T17):
            _ht17 = _ht17 @ _J.T + _inp_np[:, _t, :] @ _AW17.T
            _h_np[:, _t, :] = _ht17
        _h_flat = _h_np.reshape(-1, _N)
    else:
        _h_flat = None
        print(f"  W_in not found for cond={_cond} — correlation skipped")

    # Schur / eigenmode / neuron projections + correlations
    if _h_flat is not None:
        _Z_schur     = _h_flat @ _Qf
        _r_schur_tgt = _pearson_r_17(_Z_schur, _tgt_flat)         # (N, n_bits) sorted

        _Z_eig       = np.real(_h_flat @ _Vf_inv.T)               # (n_traj*T, N) sorted
        _r_eig_tgt   = _pearson_r_17(_Z_eig, _tgt_flat)           # (N, n_bits) sorted

        _r_neu_tgt   = _pearson_r_17(_h_flat, _tgt_flat)          # (N, n_bits) orig. order

        _dom_r_17    = {bi: int(np.argmax(np.abs(_r_schur_tgt[:, bi])))
                        for bi in range(_nbits)}
        _bulk_mask   = np.ones(_N, dtype=bool)
        for _bi in _dom_r_17.values():
            _bulk_mask[_bi] = False

        # Output-only ablation coupling:
        #   abl[k, bi] = std(activity_k) * |effective_output_coupling_k→bi| / std(ŷ_bi)
        # Schur: Q orthogonal ⟹ effective coupling = |W_out Q| = coup_schur
        # Eig  : use coup_eig as proxy (exact only when V is unitary)
        # Neuron: effective coupling = |W_out|
        _y_hat_17  = _h_flat @ _W_out.T                              # (M, n_bits)
        _y_std_17  = _y_hat_17.std(0).clip(1e-12)                    # (n_bits,)
        _abl_schur = _Z_schur.std(0)[:, None] * _coup_schur.T / _y_std_17[None, :]  # (N, n_bits)
        _abl_eig   = _Z_eig.std(0)[:, None]   * _coup_eig.T   / _y_std_17[None, :]  # (N, n_bits)
        _abl_neu   = _h_flat.std(0)[:, None]   * np.abs(_W_out).T   / _y_std_17[None, :]  # (N, n_bits)
    else:
        _r_schur_tgt = _r_eig_tgt = _r_neu_tgt = None
        _abl_schur = _abl_eig = _abl_neu = None
        _dom_r_17    = {}
        _bulk_mask   = np.ones(_N, dtype=bool)

    _g06[_cond] = dict(
        Qf=_Qf, tauf=_tauf, blkf=_blkf, N=_N, nbits=_nbits, ppl=_ppl,
        dt=float(_rc["dt"]),
        # Schur
        coup_schur=_coup_schur, cin=_cin, r_schur_tgt=_r_schur_tgt,
        abl_schur=_abl_schur,
        dom_r=_dom_r_17, bulk_mask=_bulk_mask,
        # Eigenmode
        ef_s=_ef_s, tau_eig=_tau_eig, coup_eig=_coup_eig, r_eig_tgt=_r_eig_tgt,
        abl_eig=_abl_eig,
        # Neuron
        cout=_cout, cout_s=_cout_s, win_s=_win_s, r_neu_tgt=_r_neu_tgt,
        abl_neu=_abl_neu,
        tau_neu=_tau_neu_all,
        learned_tau=_learned_tau_all,
        acc=_row["final_val_acc"],
    )
    print(f"  g={G_FOCUS_17} cond={_cond}: Schur+coupling+corr done  "
          f"(N={_N}, acc={_row['final_val_acc']})")

# shared colormap scales across conditions (Schur output vs neuron output)
_vmax_schur_out_17 = max(
    _g06[c]["coup_schur"][:, :N_HM_17].max()
    for c in CONDITIONS if c in _g06)
_vmax_neu_out_17 = max(
    _g06[c]["cout_s"][:, :N_HM_17].max()
    for c in CONDITIONS if c in _g06)
_vmax_schur_in_17 = max(
    _g06[c]["cin"][:N_HM_17, :].max()
    for c in CONDITIONS if c in _g06 and _g06[c]["cin"] is not None) if any(
    _g06.get(c, {}).get("cin") is not None for c in CONDITIONS) else 1.0
_vmax_neu_in_17 = max(
    _g06[c]["win_s"][:N_HM_17, :].max()
    for c in CONDITIONS if c in _g06 and _g06[c]["win_s"] is not None) if any(
    _g06.get(c, {}).get("win_s") is not None for c in CONDITIONS) else 1.0

print(f"Shared vmaxes — schur_out={_vmax_schur_out_17:.4f}, "
      f"neu_out={_vmax_neu_out_17:.4f}, "
      f"schur_in={_vmax_schur_in_17:.4f}, neu_in={_vmax_neu_in_17:.4f}")

# Ablation coupling vmaxes (one per basis; ablation values are non-negative)
_vmax_abl_schur_17 = max(
    _g06[c]["abl_schur"][:N_HM_17, :].max()
    for c in CONDITIONS if c in _g06 and _g06[c]["abl_schur"] is not None
) if any(_g06.get(c, {}).get("abl_schur") is not None for c in CONDITIONS) else 1.0
_vmax_abl_eig_17 = max(
    _g06[c]["abl_eig"][:N_HM_17, :].max()
    for c in CONDITIONS if c in _g06 and _g06[c]["abl_eig"] is not None
) if any(_g06.get(c, {}).get("abl_eig") is not None for c in CONDITIONS) else 1.0
_vmax_abl_neu_17 = max(
    _g06[c]["abl_neu"][:N_HM_17, :].max()
    for c in CONDITIONS if c in _g06 and _g06[c]["abl_neu"] is not None
) if any(_g06.get(c, {}).get("abl_neu") is not None for c in CONDITIONS) else 1.0
print(f"Ablation vmaxes — schur={_vmax_abl_schur_17:.4f}, "
      f"eig={_vmax_abl_eig_17:.4f}, neu={_vmax_abl_neu_17:.4f}")


# %% G0.6 — PLOT 10: Schur → output coupling heatmap (Fixed τ | Learnable τ)
# Requires: _g06, N_HM_17, G_FOCUS_17, _vmax_schur_out_17, FIGS_DIR, SAVE_FIGS

fig, axes_10 = plt.subplots(1, 2, figsize=(max(N_HM_17 * 0.9, 6) * 2 + 1,
                                            max(6 * 0.65, 3.5)))
for col, cond in enumerate(CONDITIONS):
    ax = axes_10[col]
    if cond not in _g06:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue
    d = _g06[cond]
    _tk = min(N_HM_17, d["N"])
    _hm = d["coup_schur"][:, :_tk]       # (n_bits, top_k)
    _xlbls = [f"Schur {mi+1}\nτ={d['tauf'][mi]:.0f}\n"
              f"({'R' if d['blkf'][mi]=='1x1' else 'C'})"
              for mi in range(_tk)]

    im = ax.imshow(_hm, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=_vmax_schur_out_17)
    ax.set_yticks(range(d["nbits"]))
    ax.set_yticklabels([f"Bit {i} (p={d['ppl'][i]})" for i in range(d["nbits"])],
                        fontsize=8)
    ax.set_xticks(range(_tk))
    ax.set_xticklabels(_xlbls, fontsize=7)
    ax.set_xlabel(f"Schur mode rank (top {_tk})", fontsize=9)
    ax.set_ylabel("Output bit", fontsize=10)
    for i in range(d["nbits"]):
        for j in range(_tk):
            v = _hm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.5 * _vmax_schur_out_17 else "black")
    plt.colorbar(im, ax=ax, label="$|W_{out} Q|$", shrink=0.75)
    acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
    ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                 fontsize=11, fontweight="bold")

fig.suptitle(f"Schur → Output Coupling (connectivity)  —  g = {G_FOCUS_17}",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"10_schur_output_heatmap_g{G_FOCUS_17}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% G0.6 — PLOT 10b: Schur → output |r| heatmap (Fixed τ | Learnable τ)
# |Pearson r| between each Schur-mode activity trace and each output bit.
# |r| is used because the sign depends only on encoding convention; a large
# negative r is equally indicative of strong coupling as a large positive one.
# Requires: _g06, N_HM_17, G_FOCUS_17, FIGS_DIR, SAVE_FIGS

_has_r_schur_17 = any(
    _g06.get(c, {}).get("r_schur_tgt") is not None for c in CONDITIONS
)
if not _has_r_schur_17:
    print("PLOT 10b: No Schur correlation data — skipping.")
else:
    _vmax_r_schur_17 = max(
        np.abs(_g06[c]["r_schur_tgt"][:N_HM_17, :]).max()
        for c in CONDITIONS
        if c in _g06 and _g06[c]["r_schur_tgt"] is not None
    )

    fig, axes_10b = plt.subplots(1, 2, figsize=(max(N_HM_17 * 0.9, 6) * 2 + 1,
                                                max(6 * 0.65, 3.5)))
    for col, cond in enumerate(CONDITIONS):
        ax = axes_10b[col]
        if cond not in _g06 or _g06[cond]["r_schur_tgt"] is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue
        d = _g06[cond]
        _tk = min(N_HM_17, d["N"])
        _hm = np.abs(d["r_schur_tgt"][:_tk, :].T)    # (n_bits, top_k)
        _xlbls = [f"Schur {mi+1}\nτ={d['tauf'][mi]:.0f}\n"
                  f"({'R' if d['blkf'][mi]=='1x1' else 'C'})"
                  for mi in range(_tk)]

        im = ax.imshow(_hm, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=_vmax_r_schur_17)
        ax.set_yticks(range(d["nbits"]))
        ax.set_yticklabels([f"Bit {i} (p={d['ppl'][i]})" for i in range(d["nbits"])],
                            fontsize=8)
        ax.set_xticks(range(_tk))
        ax.set_xticklabels(_xlbls, fontsize=7)
        ax.set_xlabel(f"Schur mode rank (top {_tk})", fontsize=9)
        ax.set_ylabel("Output bit", fontsize=10)
        for i in range(d["nbits"]):
            for j in range(_tk):
                v = _hm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.5 * _vmax_r_schur_17 else "black")
        plt.colorbar(im, ax=ax, label=r"$|r|$", shrink=0.75)
        # Highlight the dominant mode per bit (highest |r|) with a coloured border
        if d["dom_r"]:
            for _bi, _dm in d["dom_r"].items():
                if _dm < _tk:
                    ax.add_patch(plt.Rectangle(
                        (_dm - 0.5, _bi - 0.5), 1, 1,
                        fill=False, edgecolor=f"C{_bi}", linewidth=2.0, zorder=5
                    ))
        acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
        ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                     fontsize=11, fontweight="bold")

    fig.suptitle(f"Schur → Output Correlation ($|r|$)  —  g = {G_FOCUS_17}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR,
                                 f"10b_schur_output_corr_heatmap_g{G_FOCUS_17}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% G0.6 — PLOT 10c: Eigenmode → output |r| heatmap (Fixed τ | Learnable τ)
# |Pearson r| between each eigenmode activity trace and each output bit.
# Requires: _g06, N_HM_17, G_FOCUS_17, FIGS_DIR, SAVE_FIGS

_has_r_eig_17 = any(
    _g06.get(c, {}).get("r_eig_tgt") is not None for c in CONDITIONS
)
if not _has_r_eig_17:
    print("PLOT 10c: No eigenmode correlation data — skipping.")
else:
    _vmax_r_eig_17 = max(
        np.abs(_g06[c]["r_eig_tgt"][:N_HM_17, :]).max()
        for c in CONDITIONS
        if c in _g06 and _g06[c]["r_eig_tgt"] is not None
    )

    fig, axes_10c = plt.subplots(1, 2, figsize=(max(N_HM_17 * 0.9, 6) * 2 + 1,
                                                max(6 * 0.65, 3.5)))
    for col, cond in enumerate(CONDITIONS):
        ax = axes_10c[col]
        if cond not in _g06 or _g06[cond]["r_eig_tgt"] is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue
        d = _g06[cond]
        _tk = min(N_HM_17, d["N"])
        _hm = np.abs(d["r_eig_tgt"][:_tk, :].T)    # (n_bits, top_k)
        _xlbls = [f"Eig {mi+1}\nτ={d['tau_eig'][mi]:.0f}" for mi in range(_tk)]

        im = ax.imshow(_hm, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=_vmax_r_eig_17)
        ax.set_yticks(range(d["nbits"]))
        ax.set_yticklabels([f"Bit {i} (p={d['ppl'][i]})" for i in range(d["nbits"])],
                            fontsize=8)
        ax.set_xticks(range(_tk))
        ax.set_xticklabels(_xlbls, fontsize=7)
        ax.set_xlabel(f"Eigenmode rank (top {_tk}, by |λ|)", fontsize=9)
        ax.set_ylabel("Output bit", fontsize=10)
        for i in range(d["nbits"]):
            for j in range(_tk):
                v = _hm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.5 * _vmax_r_eig_17 else "black")
        plt.colorbar(im, ax=ax, label=r"$|r|$", shrink=0.75)
        # Dominant eigenmode per bit (highest |r|)
        _dom_eig = {bi: int(np.argmax(_hm[bi])) for bi in range(d["nbits"])}
        for _bi, _dm in _dom_eig.items():
            if _dm < _tk:
                ax.add_patch(plt.Rectangle(
                    (_dm - 0.5, _bi - 0.5), 1, 1,
                    fill=False, edgecolor=f"C{_bi}", linewidth=2.0, zorder=5
                ))
        acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
        ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                     fontsize=11, fontweight="bold")

    fig.suptitle(f"Eigenmode → Output Correlation ($|r|$)  —  g = {G_FOCUS_17}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR,
                                 f"10c_eig_output_corr_heatmap_g{G_FOCUS_17}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% G0.6 — PLOT 10d: Schur → output ablation heatmap (Fixed τ | Learnable τ)
# Activity-weighted output coupling: std(z_k) * |W_out Q|_k / std(ŷ).
# Captures how much each Schur mode actually drives the output variance.
# Requires: _g06, N_HM_17, G_FOCUS_17, _vmax_abl_schur_17, FIGS_DIR, SAVE_FIGS

_has_abl_schur_17 = any(
    _g06.get(c, {}).get("abl_schur") is not None for c in CONDITIONS
)
if not _has_abl_schur_17:
    print("PLOT 10d: No Schur ablation data — skipping.")
else:
    fig, axes_10d = plt.subplots(1, 2, figsize=(max(N_HM_17 * 0.9, 6) * 2 + 1,
                                                max(6 * 0.65, 3.5)))
    for col, cond in enumerate(CONDITIONS):
        ax = axes_10d[col]
        if cond not in _g06 or _g06[cond]["abl_schur"] is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue
        d = _g06[cond]
        _tk = min(N_HM_17, d["N"])
        _hm = d["abl_schur"][:_tk, :].T    # (n_bits, top_k)
        _xlbls = [f"Schur {mi+1}\nτ={d['tauf'][mi]:.0f}\n"
                  f"({'R' if d['blkf'][mi]=='1x1' else 'C'})"
                  for mi in range(_tk)]

        im = ax.imshow(_hm, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=_vmax_abl_schur_17)
        ax.set_yticks(range(d["nbits"]))
        ax.set_yticklabels([f"Bit {i} (p={d['ppl'][i]})" for i in range(d["nbits"])],
                            fontsize=8)
        ax.set_xticks(range(_tk))
        ax.set_xticklabels(_xlbls, fontsize=7)
        ax.set_xlabel(f"Schur mode rank (top {_tk})", fontsize=9)
        ax.set_ylabel("Output bit", fontsize=10)
        for i in range(d["nbits"]):
            for j in range(_tk):
                v = _hm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.5 * _vmax_abl_schur_17 else "black")
        plt.colorbar(im, ax=ax,
                     label=r"$\sigma(z_k)\,|W_{\!out}Q|_k\,/\,\sigma(\hat{y})$",
                     shrink=0.75)
        _dom_abl = {bi: int(np.argmax(_hm[bi])) for bi in range(d["nbits"])}
        for _bi, _dm in _dom_abl.items():
            if _dm < _tk:
                ax.add_patch(plt.Rectangle(
                    (_dm - 0.5, _bi - 0.5), 1, 1,
                    fill=False, edgecolor=f"C{_bi}", linewidth=2.0, zorder=5
                ))
        acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
        ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                     fontsize=11, fontweight="bold")

    fig.suptitle(f"Schur → Output Ablation Coupling  —  g = {G_FOCUS_17}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR,
                                 f"10d_schur_output_abl_heatmap_g{G_FOCUS_17}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% G0.6 — PLOT 10e: Eigenmode → output ablation heatmap (Fixed τ | Learnable τ)
# Activity-weighted output coupling: std(z_k) * |W_out V|_k / std(ŷ).
# Requires: _g06, N_HM_17, G_FOCUS_17, _vmax_abl_eig_17, FIGS_DIR, SAVE_FIGS

_has_abl_eig_17 = any(
    _g06.get(c, {}).get("abl_eig") is not None for c in CONDITIONS
)
if not _has_abl_eig_17:
    print("PLOT 10e: No eigenmode ablation data — skipping.")
else:
    fig, axes_10e = plt.subplots(1, 2, figsize=(max(N_HM_17 * 0.9, 6) * 2 + 1,
                                                max(6 * 0.65, 3.5)))
    for col, cond in enumerate(CONDITIONS):
        ax = axes_10e[col]
        if cond not in _g06 or _g06[cond]["abl_eig"] is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue
        d = _g06[cond]
        _tk = min(N_HM_17, d["N"])
        _hm = d["abl_eig"][:_tk, :].T    # (n_bits, top_k)
        _xlbls = [f"Eig {mi+1}\nτ={d['tau_eig'][mi]:.0f}" for mi in range(_tk)]

        im = ax.imshow(_hm, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=_vmax_abl_eig_17)
        ax.set_yticks(range(d["nbits"]))
        ax.set_yticklabels([f"Bit {i} (p={d['ppl'][i]})" for i in range(d["nbits"])],
                            fontsize=8)
        ax.set_xticks(range(_tk))
        ax.set_xticklabels(_xlbls, fontsize=7)
        ax.set_xlabel(f"Eigenmode rank (top {_tk}, by |λ|)", fontsize=9)
        ax.set_ylabel("Output bit", fontsize=10)
        for i in range(d["nbits"]):
            for j in range(_tk):
                v = _hm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.5 * _vmax_abl_eig_17 else "black")
        plt.colorbar(im, ax=ax,
                     label=r"$\sigma(z_k)\,|W_{\!out}V|_k\,/\,\sigma(\hat{y})$",
                     shrink=0.75)
        _dom_abl_eig = {bi: int(np.argmax(_hm[bi])) for bi in range(d["nbits"])}
        for _bi, _dm in _dom_abl_eig.items():
            if _dm < _tk:
                ax.add_patch(plt.Rectangle(
                    (_dm - 0.5, _bi - 0.5), 1, 1,
                    fill=False, edgecolor=f"C{_bi}", linewidth=2.0, zorder=5
                ))
        acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
        ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                     fontsize=11, fontweight="bold")

    fig.suptitle(f"Eigenmode → Output Ablation Coupling  —  g = {G_FOCUS_17}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR,
                                 f"10e_eig_output_abl_heatmap_g{G_FOCUS_17}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% G0.6 — PLOT 11: Neuron → output coupling heatmap (Fixed τ | Learnable τ)
# Requires: _g06, N_HM_17, G_FOCUS_17, _vmax_neu_out_17, FIGS_DIR, SAVE_FIGS

fig, axes_11 = plt.subplots(1, 2, figsize=(max(N_HM_17 * 0.9, 6) * 2 + 1,
                                            max(6 * 0.65, 3.5)))
for col, cond in enumerate(CONDITIONS):
    ax = axes_11[col]
    if cond not in _g06:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue
    d = _g06[cond]
    _tk = min(N_HM_17, d["N"])
    _hm = d["cout_s"][:, :_tk]           # (n_bits, top_k)

    im = ax.imshow(_hm, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=_vmax_neu_out_17)
    ax.set_yticks(range(d["nbits"]))
    ax.set_yticklabels([f"Bit {i} (p={d['ppl'][i]})" for i in range(d["nbits"])],
                        fontsize=8)
    ax.set_xticks(range(_tk))
    ax.set_xticklabels([f"N{i+1}" for i in range(_tk)], fontsize=7)
    ax.set_xlabel(f"Neuron rank (top {_tk}, by ||w_out||)", fontsize=9)
    ax.set_ylabel("Output bit", fontsize=10)
    for i in range(d["nbits"]):
        for j in range(_tk):
            v = _hm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.5 * _vmax_neu_out_17 else "black")
    plt.colorbar(im, ax=ax, label="$|W_{out,ij}|$", shrink=0.75)
    acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
    ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                 fontsize=11, fontweight="bold")

fig.suptitle(f"Neuron → Output Coupling (connectivity)  —  g = {G_FOCUS_17}",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"11_neuron_output_heatmap_g{G_FOCUS_17}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% G0.6 — PLOT 11b: Neuron → output |r| heatmap (Fixed τ | Learnable τ)
# |Pearson r| between each neuron's activity trace and each output bit.
# Neurons ranked by ||w_out|| (same ordering as PLOT 11) so the two are directly
# comparable column-by-column.
# Requires: _g06, N_HM_17, G_FOCUS_17, FIGS_DIR, SAVE_FIGS

_has_r_neu_17 = any(
    _g06.get(c, {}).get("r_neu_tgt") is not None for c in CONDITIONS
)
if not _has_r_neu_17:
    print("PLOT 11b: No neuron correlation data — skipping.")
else:
    _vmax_r_neu_17 = max(
        np.abs(_g06[c]["r_neu_tgt"][:N_HM_17, :]).max()
        for c in CONDITIONS
        if c in _g06 and _g06[c]["r_neu_tgt"] is not None
    )

    fig, axes_11b = plt.subplots(1, 2, figsize=(max(N_HM_17 * 0.9, 6) * 2 + 1,
                                                max(6 * 0.65, 3.5)))
    for col, cond in enumerate(CONDITIONS):
        ax = axes_11b[col]
        if cond not in _g06 or _g06[cond]["r_neu_tgt"] is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue
        d = _g06[cond]
        _tk = min(N_HM_17, d["N"])
        # r_neu_tgt is in original neuron order; re-sort by ||w_out|| to match PLOT 11
        _cout_ord = np.argsort(-d["cout"].max(axis=0))   # descending by max |w_out|
        _hm = np.abs(d["r_neu_tgt"][_cout_ord[:_tk], :].T)   # (n_bits, top_k)

        im = ax.imshow(_hm, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=_vmax_r_neu_17)
        ax.set_yticks(range(d["nbits"]))
        ax.set_yticklabels([f"Bit {i} (p={d['ppl'][i]})" for i in range(d["nbits"])],
                            fontsize=8)
        ax.set_xticks(range(_tk))
        ax.set_xticklabels([f"N{i+1}" for i in range(_tk)], fontsize=7)
        ax.set_xlabel(f"Neuron rank (top {_tk}, by ||w_out||)", fontsize=9)
        ax.set_ylabel("Output bit", fontsize=10)
        for i in range(d["nbits"]):
            for j in range(_tk):
                v = _hm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.5 * _vmax_r_neu_17 else "black")
        plt.colorbar(im, ax=ax, label=r"$|r|$", shrink=0.75)
        # Dominant neuron per bit (highest |r| in this sorted order)
        _dom_neu = {bi: int(np.argmax(_hm[bi])) for bi in range(d["nbits"])}
        for _bi, _dm in _dom_neu.items():
            if _dm < _tk:
                ax.add_patch(plt.Rectangle(
                    (_dm - 0.5, _bi - 0.5), 1, 1,
                    fill=False, edgecolor=f"C{_bi}", linewidth=2.0, zorder=5
                ))
        acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
        ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                     fontsize=11, fontweight="bold")

    fig.suptitle(f"Neuron → Output Correlation ($|r|$)  —  g = {G_FOCUS_17}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR,
                                 f"11b_neuron_output_corr_heatmap_g{G_FOCUS_17}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% G0.6 — PLOT 11d: Neuron → output ablation heatmap (Fixed τ | Learnable τ)
# Activity-weighted output coupling: std(h_k) * |W_out|_k / std(ŷ).
# Neurons sorted by max |W_out| (same as PLOTs 11/11b) for direct comparison.
# Requires: _g06, N_HM_17, G_FOCUS_17, _vmax_abl_neu_17, FIGS_DIR, SAVE_FIGS

_has_abl_neu_17 = any(
    _g06.get(c, {}).get("abl_neu") is not None for c in CONDITIONS
)
if not _has_abl_neu_17:
    print("PLOT 11d: No neuron ablation data — skipping.")
else:
    fig, axes_11d = plt.subplots(1, 2, figsize=(max(N_HM_17 * 0.9, 6) * 2 + 1,
                                                max(6 * 0.65, 3.5)))
    for col, cond in enumerate(CONDITIONS):
        ax = axes_11d[col]
        if cond not in _g06 or _g06[cond]["abl_neu"] is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue
        d = _g06[cond]
        _tk = min(N_HM_17, d["N"])
        _cout_ord = np.argsort(-d["cout"].max(axis=0))
        _hm = d["abl_neu"][_cout_ord[:_tk], :].T    # (n_bits, top_k)

        im = ax.imshow(_hm, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=_vmax_abl_neu_17)
        ax.set_yticks(range(d["nbits"]))
        ax.set_yticklabels([f"Bit {i} (p={d['ppl'][i]})" for i in range(d["nbits"])],
                            fontsize=8)
        ax.set_xticks(range(_tk))
        ax.set_xticklabels([f"N{i+1}" for i in range(_tk)], fontsize=7)
        ax.set_xlabel(f"Neuron rank (top {_tk}, by max $|W_{{out}}|$)", fontsize=9)
        ax.set_ylabel("Output bit", fontsize=10)
        for i in range(d["nbits"]):
            for j in range(_tk):
                v = _hm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.5 * _vmax_abl_neu_17 else "black")
        plt.colorbar(im, ax=ax,
                     label=r"$\sigma(h_k)\,|W_{\!out}|_k\,/\,\sigma(\hat{y})$",
                     shrink=0.75)
        _dom_abl_neu = {bi: int(np.argmax(_hm[bi])) for bi in range(d["nbits"])}
        for _bi, _dm in _dom_abl_neu.items():
            if _dm < _tk:
                ax.add_patch(plt.Rectangle(
                    (_dm - 0.5, _bi - 0.5), 1, 1,
                    fill=False, edgecolor=f"C{_bi}", linewidth=2.0, zorder=5
                ))
        acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
        ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                     fontsize=11, fontweight="bold")

    fig.suptitle(f"Neuron → Output Ablation Coupling  —  g = {G_FOCUS_17}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR,
                                 f"11d_neuron_output_abl_heatmap_g{G_FOCUS_17}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% G0.6 — PLOT 11c: Full |r| trace — Schur / Eigenmode / Neuron (Fixed τ | Learnable τ)
# Shows |Pearson r| vs rank for every mode/neuron and every output bit.
# Sorting:  Schur & Eigenmode → by |λ| (largest first, same as heatmaps).
#           Neurons            → by max |w_out| across bits (same as PLOTs 11/11b).
# Dominant mode per bit (highest |r|) is marked with a filled circle.
# Requires: _g06, bit_colors, G_FOCUS_17, FIGS_DIR, SAVE_FIGS

_has_any_r_17 = any(
    any(_g06.get(c, {}).get(k) is not None for c in CONDITIONS)
    for k in ("r_schur_tgt", "r_eig_tgt", "r_neu_tgt")
)
if not _has_any_r_17:
    print("PLOT 11c: No correlation data available — skipping.")
else:
    _ROW_KEYS_11c  = ["r_schur_tgt", "r_eig_tgt", "r_neu_tgt"]
    _ROW_LBLS_11c  = ["Schur modes", "Eigenmodes", "Neurons"]
    _XLAB_11c      = [
        "Schur mode rank (sorted by |λ|, largest first)",
        "Eigenmode rank (sorted by |λ|, largest first)",
        "Neuron rank (sorted by max $|W_{out}|$)",
    ]

    fig, axes_11c = plt.subplots(3, 2, figsize=(14, 12), sharey="row")

    for col, cond in enumerate(CONDITIONS):
        if cond not in _g06:
            for ri in range(3):
                axes_11c[ri, col].text(0.5, 0.5, "no data",
                                       transform=axes_11c[ri, col].transAxes,
                                       ha="center"); continue
        d      = _g06[cond]
        _N17c  = d["N"]
        _nb17c = d["nbits"]
        _ranks = np.arange(1, _N17c + 1)

        # Neuron sort order (||w_out||, descending) — used for row 2 only
        _cout_ord_11c = np.argsort(-d["cout"].max(axis=0))

        for ri, (rkey, rlbl, xlab) in enumerate(
                zip(_ROW_KEYS_11c, _ROW_LBLS_11c, _XLAB_11c)):
            ax = axes_11c[ri, col]
            r_mat = d.get(rkey)   # (N, n_bits) or None

            if r_mat is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")
                continue

            # Re-sort neurons to match PLOT 11/11b column ordering
            if rkey == "r_neu_tgt":
                r_mat = r_mat[_cout_ord_11c, :]    # (N, n_bits)

            _abs_r = np.abs(r_mat)                 # (N, n_bits)

            for bi in range(_nb17c):
                col_r  = _abs_r[:, bi]
                dom_j  = int(np.argmax(col_r))
                clr    = bit_colors[bi] if bi < len(bit_colors) else f"C{bi}"

                # Full trace as a thin line
                ax.plot(_ranks, col_r, color=clr, lw=0.8, alpha=0.55,
                        label=f"Bit {bi} (p={d['ppl'][bi]})" if col == 0 else None)

                # Dominant mode: filled dot
                ax.scatter(_ranks[dom_j], col_r[dom_j],
                           s=60, color=clr, edgecolors="white",
                           linewidths=0.8, zorder=6)

            ax.set_xlim(1, _N17c)
            ax.set_ylim(0, 1.05)
            ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
            ax.grid(True, alpha=0.12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if ri == 2:
                ax.set_xlabel(xlab, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{rlbl}\n$|r|$", fontsize=9)

            acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
            if ri == 0:
                ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                             fontsize=11, fontweight="bold")

    # Single legend (bit colours) outside the figure
    from matplotlib.lines import Line2D as _LD11c
    _d_any = _g06[next(c for c in CONDITIONS if c in _g06)]
    _leg11c = [
        _LD11c([0], [0], color=bit_colors[i] if i < len(bit_colors) else f"C{i}",
               lw=2, label=f"Bit {i}  (p={_d_any['ppl'][i]})")
        for i in range(_d_any["nbits"])
    ]
    fig.legend(handles=_leg11c, fontsize=8, ncol=1,
               loc="upper left", bbox_to_anchor=(1.01, 0.99),
               framealpha=0.85, borderaxespad=0)

    fig.suptitle(
        f"Full $|r|$ trace — Schur / Eigenmode / Neuron  —  g = {G_FOCUS_17}\n"
        "Filled dot = dominant mode per bit (highest $|r|$)",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR,
                                 f"11c_full_r_trace_g{G_FOCUS_17}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% G0.6 — PLOT 11e: Full ablation trace — Schur / Eigenmode / Neuron (Fixed τ | Learnable τ)
# Mirrors PLOT 11c but uses activity-weighted output coupling (ablation metric)
# instead of |Pearson r|.  Sorting follows the same convention as 11c.
# Requires: _g06, bit_colors, G_FOCUS_17, FIGS_DIR, SAVE_FIGS

_has_any_abl_17 = any(
    any(_g06.get(c, {}).get(k) is not None for c in CONDITIONS)
    for k in ("abl_schur", "abl_eig", "abl_neu")
)
if not _has_any_abl_17:
    print("PLOT 11e: No ablation data available — skipping.")
else:
    _ROW_KEYS_11e = ["abl_schur", "abl_eig", "abl_neu"]
    _ROW_LBLS_11e = ["Schur modes", "Eigenmodes", "Neurons"]
    _XLAB_11e     = [
        "Schur mode rank (sorted by |λ|, largest first)",
        "Eigenmode rank (sorted by |λ|, largest first)",
        "Neuron rank (sorted by max $|W_{out}|$)",
    ]
    _YLAB_11e = r"$\sigma(\mathrm{activity})\,|\mathrm{coupling}|\,/\,\sigma(\hat{y})$"

    fig, axes_11e = plt.subplots(3, 2, figsize=(14, 12), sharey="row")

    for col, cond in enumerate(CONDITIONS):
        if cond not in _g06:
            for ri in range(3):
                axes_11e[ri, col].text(0.5, 0.5, "no data",
                                       transform=axes_11e[ri, col].transAxes,
                                       ha="center"); continue
        d      = _g06[cond]
        _N11e  = d["N"]
        _nb11e = d["nbits"]
        _ranks = np.arange(1, _N11e + 1)

        _cout_ord_11e = np.argsort(-d["cout"].max(axis=0))

        for ri, (rkey, rlbl, xlab) in enumerate(
                zip(_ROW_KEYS_11e, _ROW_LBLS_11e, _XLAB_11e)):
            ax = axes_11e[ri, col]
            abl_mat = d.get(rkey)    # (N, n_bits) or None

            if abl_mat is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")
                continue

            if rkey == "abl_neu":
                abl_mat = abl_mat[_cout_ord_11e, :]    # sort by ||w_out||

            for bi in range(_nb11e):
                col_a = abl_mat[:, bi]
                dom_j = int(np.argmax(col_a))
                clr   = bit_colors[bi] if bi < len(bit_colors) else f"C{bi}"

                ax.plot(_ranks, col_a, color=clr, lw=0.8, alpha=0.55,
                        label=f"Bit {bi} (p={d['ppl'][bi]})" if col == 0 else None)
                ax.scatter(_ranks[dom_j], col_a[dom_j],
                           s=60, color=clr, edgecolors="white",
                           linewidths=0.8, zorder=6)

            ax.set_xlim(1, _N11e)
            ax.set_ylim(bottom=0)
            ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
            ax.grid(True, alpha=0.12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if ri == 2:
                ax.set_xlabel(xlab, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{rlbl}\n{_YLAB_11e}", fontsize=8)

            acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
            if ri == 0:
                ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                             fontsize=11, fontweight="bold")

    from matplotlib.lines import Line2D as _LD11e
    _d_any_11e = _g06[next(c for c in CONDITIONS if c in _g06)]
    _leg11e = [
        _LD11e([0], [0], color=bit_colors[i] if i < len(bit_colors) else f"C{i}",
               lw=2, label=f"Bit {i}  (p={_d_any_11e['ppl'][i]})")
        for i in range(_d_any_11e["nbits"])
    ]
    fig.legend(handles=_leg11e, fontsize=8, ncol=1,
               loc="upper left", bbox_to_anchor=(1.01, 0.99),
               framealpha=0.85, borderaxespad=0)

    fig.suptitle(
        f"Full ablation trace — Schur / Eigenmode / Neuron  —  g = {G_FOCUS_17}\n"
        r"Filled dot = dominant mode per bit  $\cdot$  "
        r"metric $= \sigma(z_k)\,|\mathrm{coupling}_k|\,/\,\sigma(\hat{y})$",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR,
                                 f"11e_full_abl_trace_g{G_FOCUS_17}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% G0.6 — PLOT 12: Input → Schur coupling heatmap (Fixed τ | Learnable τ)
# Requires: _g06, N_HM_17, G_FOCUS_17, _vmax_schur_in_17, FIGS_DIR, SAVE_FIGS

_has_win = any(_g06.get(c, {}).get("cin") is not None for c in CONDITIONS)
if not _has_win:
    print("PLOT 12: W_in not found in checkpoints — skipping input coupling plots.")
else:
    fig, axes_12 = plt.subplots(1, 2, figsize=(max(N_HM_17 * 0.9, 6) * 2 + 1,
                                                max(6 * 0.65, 3.5)))
    for col, cond in enumerate(CONDITIONS):
        ax = axes_12[col]
        if cond not in _g06 or _g06[cond]["cin"] is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue
        d = _g06[cond]
        _tk = min(N_HM_17, d["N"])
        _hm = d["cin"][:_tk, :].T        # (n_bits, top_k) for imshow

        im = ax.imshow(_hm, cmap="Blues", aspect="auto",
                       vmin=0, vmax=_vmax_schur_in_17)
        ax.set_yticks(range(d["nbits"]))
        ax.set_yticklabels([f"In bit {i} (p={d['ppl'][i]})" for i in range(d["nbits"])],
                            fontsize=8)
        ax.set_xticks(range(_tk))
        ax.set_xticklabels([f"Schur {i+1}\nτ={d['tauf'][i]:.0f}" for i in range(_tk)],
                            fontsize=7)
        ax.set_xlabel(f"Schur mode rank (top {_tk})", fontsize=9)
        ax.set_ylabel("Input bit", fontsize=10)
        for i in range(d["nbits"]):
            for j in range(_tk):
                v = _hm[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.5 * _vmax_schur_in_17 else "black")
        plt.colorbar(im, ax=ax, label=r"$\alpha|Q^\top W_{in}|$", shrink=0.75)
        acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
        ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                     fontsize=11, fontweight="bold")

    fig.suptitle(f"Input → Schur Coupling (connectivity)  —  g = {G_FOCUS_17}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"12_input_schur_heatmap_g{G_FOCUS_17}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% G0.6 — PLOT 13: Input → Neuron coupling heatmap (Fixed τ | Learnable τ)
# Requires: _g06, N_HM_17, G_FOCUS_17, _vmax_neu_in_17, FIGS_DIR, SAVE_FIGS

if not _has_win:
    print("PLOT 13: W_in not found — skipping.")
else:
    fig, axes_13 = plt.subplots(1, 2, figsize=(max(N_HM_17 * 0.9, 6) * 2 + 1,
                                                max(6 * 0.65, 3.5)))
    for col, cond in enumerate(CONDITIONS):
        ax = axes_13[col]
        if cond not in _g06 or _g06[cond]["win_s"] is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue
        d = _g06[cond]
        _tk = min(N_HM_17, d["N"])
        _hm = d["win_s"][:_tk, :].T      # (n_bits, top_k) for imshow

        im = ax.imshow(_hm, cmap="Blues", aspect="auto",
                       vmin=0, vmax=_vmax_neu_in_17)
        ax.set_yticks(range(d["nbits"]))
        ax.set_yticklabels([f"In bit {i} (p={d['ppl'][i]})" for i in range(d["nbits"])],
                            fontsize=8)
        ax.set_xticks(range(_tk))
        ax.set_xticklabels([f"N{i+1}" for i in range(_tk)], fontsize=7)
        ax.set_xlabel(f"Neuron rank (top {_tk}, by ||w_in||)", fontsize=9)
        ax.set_ylabel("Input bit", fontsize=10)
        for i in range(d["nbits"]):
            for j in range(_tk):
                v = _hm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.5 * _vmax_neu_in_17 else "black")
        plt.colorbar(im, ax=ax, label=r"$|W_{in,ji}|$", shrink=0.75)
        acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
        ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                     fontsize=11, fontweight="bold")

    fig.suptitle(f"Input → Neuron Coupling (connectivity)  —  g = {G_FOCUS_17}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"13_input_neuron_heatmap_g{G_FOCUS_17}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% G0.6 — PLOT 14: Correlation scree (Fixed τ | Learnable τ)
# Bulk modes = solid blue;  dominant-by-correlation mode per bit = coloured ring.
# Hold-time reference segments drawn at their actual τ value in the middle of
# the x-axis for direct visual comparison.
# Requires: _g06, bit_colors, HOLDS, FIGS_DIR, SAVE_FIGS

fig, axes_14 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for col, cond in enumerate(CONDITIONS):
    ax = axes_14[col]
    if cond not in _g06:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); continue

    d = _g06[cond]
    _N17      = d["N"]
    _tauf17   = d["tauf"]
    _dom17    = d["dom_r"]
    _bulk17   = d["bulk_mask"]
    _r17      = d["r_schur_tgt"]
    _ppl17    = d["ppl"]
    _nb17     = d["nbits"]
    _bit_cls  = [f"C{i}" for i in range(_nb17)]
    _ranks17  = np.arange(1, _N17 + 1)

    # Bulk modes: solid blue
    ax.scatter(_ranks17[_bulk17], _tauf17[_bulk17],
               color="steelblue", s=18, alpha=0.5, edgecolors="none",
               rasterized=True, zorder=2)

    # Dominant modes: coloured rings + annotations
    for _bi in range(_nb17):
        _dm = _dom17[_bi]
        ax.scatter(_ranks17[_dm], _tauf17[_dm], s=180, color=_bit_cls[_bi],
                   edgecolors="white", linewidths=1.1, zorder=6)
        ax.annotate(
            f"Bit {_bi}  (p={_ppl17[_bi]})\n"
            f"τ={_tauf17[_dm]:.1f}  r={_r17[_dm,_bi]:+.2f}",
            xy=(_ranks17[_dm], _tauf17[_dm]),
            xytext=(18, 0), textcoords="offset points",
            fontsize=8, color=_bit_cls[_bi], fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-", color=_bit_cls[_bi], lw=0.5, alpha=0.4))

    ax.set_yscale("log")
    ax.set_xlim(0, _N17 + 10)
    ax.grid(True, alpha=0.1, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Hold-time segments in the middle of the (linear) x-axis
    _sx0 = 0.38 * (_N17 + 10)
    _sx1 = 0.62 * (_N17 + 10)
    for _i in range(_nb17):
        _h = 1.0 / _ppl17[_i]
        ax.plot([_sx0, _sx1], [_h, _h],
                color=_bit_cls[_i], linewidth=3.0, alpha=0.8,
                solid_capstyle="round", zorder=3)
        ax.text(_sx1 * 1.08, _h,
                f"hold≈{_h:.0f}  (Bit {_i})",
                color=_bit_cls[_i], fontsize=7.5, va="center")

    ax.set_xlabel("Schur mode rank (sorted by |λ|, largest first)", fontsize=11)
    if col == 0:
        ax.set_ylabel(r"$\tau_{\mathrm{eff}} = -1/\ln|\lambda|$  (steps)", fontsize=11)
    acc_str = f"  acc={d['acc']:.3f}" if d["acc"] else ""
    ax.set_title(f"{_COND_LABELS[cond]}  (g={G_FOCUS_17}{acc_str})",
                 fontsize=11, fontweight="bold")

    # Legend: dominant modes (only when correlation is available)
    if _dom17 and _r17 is not None:
        from matplotlib.lines import Line2D as _LD17
        _leg17 = [_LD17([0], [0], marker="o", color="w",
                         markerfacecolor=_bit_cls[i], markeredgecolor="white",
                         markersize=9,
                         label=f"Bit {i} dom  (r={_r17[_dom17[i],i]:+.2f},  "
                               f"τ={_tauf17[_dom17[i]]:.1f})")
                  for i in range(_nb17)]
        # ax.legend(handles=_leg17, fontsize=7.5, loc="upper right",
        #           framealpha=0.85, ncol=2)

fig.suptitle(
    f"Schur correlation scree  —  g = {G_FOCUS_17}\n"
    "Bulk = solid blue;  rings = dominant mode by Pearson r",
    fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"14_corr_scree_g{G_FOCUS_17}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% G0.6 — PLOT 15: τ_eff of dominant mode vs task hold time
# ─────────────────────────────────────────────────────────────────────────────
# Two separate figures — one per τ condition (Fixed / Learnable).
# Each figure: 3 rows (Schur | Eigenmode | Neuron) × 2 cols (Connectivity | Correlation)
#
# Notes on τ_eff per basis:
#   Schur    – τ_eff = –1/ln|λ| from Schur block eigenvalue
#   Eigenmode– τ_eff = –1/ln|λ|  (same eigenvalues, different sorting)
#   Neuron   – τ_eff = τ of the Schur mode most expressed in that neuron
#              (i.e., τ_eff[argmax_j |Q[k,j]|] for dominant neuron k)
# ─────────────────────────────────────────────────────────────────────────────
# Requires: _g06, bit_colors, HOLDS, _COND_LABELS, G_FOCUS_17, FIGS_DIR, SAVE_FIGS

_MODE_LABELS_15 = ["Schur modes", "Eigenmodes", "Neurons"]
_COUP_LABELS_15 = ["Connectivity", "Correlation"]

# Shared axis limits — ALL quantities in real time (multiply steps by dt).
# τ_eff (steps) × dt = real time;  HOLDS (steps) × dt = real time;
# learned_tau is already in real time.
_all_tau_vals_rt = []
for _c in CONDITIONS:
    if _c not in _g06:
        continue
    _d15 = _g06[_c]
    _dt15 = 0.1
    _all_tau_vals_rt.extend((_d15["tauf"]    * _dt15).tolist())
    _all_tau_vals_rt.extend((_d15["tau_eig"] * _dt15).tolist())
    _all_tau_vals_rt.extend((_d15["tau_neu"] * _dt15).tolist())
    if _d15["learned_tau"] is not None:
        _all_tau_vals_rt.extend(_d15["learned_tau"].tolist())

# _dt_ref    = _g06[next(c for c in CONDITIONS if c in _g06)]["dt"]
_dt_ref = 0.1
_holds_rt  = [h * _dt_ref for h in HOLDS]
_hold_max  = max(_holds_rt) * 1.6
_tau_max   = min(max(_all_tau_vals_rt) * 1.3, _hold_max * 1.5) if _all_tau_vals_rt else _hold_max
_lim_lo_15 = min(_holds_rt) * 0.4
_lim_hi_15 = max(_tau_max, _hold_max)   # shared upper limit → true square panels

from matplotlib.lines import Line2D as _LD15

_bit_leg_15 = [
    _LD15([0], [0], marker="o", color="w",
          markerfacecolor=bit_colors[i], markeredgecolor="white",
          markersize=9, label=f"Bit {i}  (hold\u2248{_holds_rt[i]:.1f})")
    for i in range(len(HOLDS))
]


def _make_tau_hold_fig(cond: str):
    """Draw the 3\u00d72 scatter for a single \u03c4 condition; return fig.
    All axes in real time units (τ_eff × dt for eigenvalue-based; τ_k directly
    for learned; HOLDS × dt for the y-axis)."""
    if cond not in _g06:
        print(f"No data for cond={cond}")
        return None

    d      = _g06[cond]
    dt     = 0.1
    n_bits = d["nbits"]
    holds_rt = np.array([1.0 / p for p in d["ppl"]]) * dt   # real time

    fig, axes = plt.subplots(3, 2, figsize=(8, 11), squeeze=False)

    for _ri, _mode in enumerate(["schur", "eig", "neu"]):
        for _ci, _meth in enumerate(["conn", "corr"]):
            ax = axes[_ri, _ci]

            # Diagonal reference line
            ax.plot([_lim_lo_15, _lim_hi_15], [_lim_lo_15, _lim_hi_15],
                    "k--", lw=1.2, alpha=0.45, zorder=0)

            # ── Select coupling matrix and τ_eff array (in real time) ──────
            if _mode == "schur":
                coup  = d["coup_schur"] if _meth == "conn" else (
                    np.abs(d["r_schur_tgt"]).T if d["r_schur_tgt"] is not None else None)
                tau_v = d["tauf"] * dt          # steps → real time

            elif _mode == "eig":
                coup  = d["coup_eig"] if _meth == "conn" else (
                    np.abs(d["r_eig_tgt"]).T if d["r_eig_tgt"] is not None else None)
                tau_v = d["tau_eig"] * dt       # steps → real time

            else:  # neuron
                coup  = d["cout"] if _meth == "conn" else (
                    np.abs(d["r_neu_tgt"]).T if d["r_neu_tgt"] is not None else None)
                # learned_tau already in real time;
                # Schur-projection fallback converted from steps
                tau_v = (d["learned_tau"] if d["learned_tau"] is not None
                         else d["tau_neu"] * dt)

            if coup is None:
                ax.text(0.5, 0.5, "no correlation data",
                        transform=ax.transAxes, ha="center",
                        va="center", fontsize=9, color="gray")
            else:
                for bi in range(n_bits):
                    dom_j   = int(np.argmax(coup[bi]))
                    tau_dom = float(tau_v[dom_j])
                    ax.scatter(float(holds_rt[bi]), tau_dom,
                               s=90, marker="o",
                               color=bit_colors[bi],
                               edgecolors="white", linewidths=0.9,
                               zorder=5, clip_on=False)

            # ax.set_xlim(_lim_lo_15, _lim_hi_15)
            # ax.set_ylim(_lim_lo_15, _lim_hi_15)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.15, which="both")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if _ri == 2:
                ax.set_xlabel("Task hold duration (real time)", fontsize=10)
            if _ci == 0:
                ax.set_ylabel(r"$\tau_{\mathrm{eff}}$ of dominant mode (real time)",
                              fontsize=10)

            # Annotate neuron row to clarify which τ is on the x-axis
            _mode_lbl = _MODE_LABELS_15[_ri]
            if _mode == "neu":
                _mode_lbl += (r"  ($\tau_k$ learned)"
                              if d["learned_tau"] is not None
                              else r"  ($\tau_{\mathrm{eff}}$ via Schur)")
            ax.set_title(f"{_mode_lbl} \u2014 {_COUP_LABELS_15[_ci]}",
                         fontsize=10, fontweight="bold")

    fig.legend(handles=_bit_leg_15, fontsize=8, ncol=1,
               loc="upper left", bbox_to_anchor=(1.01, 0.99),
               framealpha=0.85, borderaxespad=0)

    acc_str = f"  (acc={d['acc']:.3f})" if d["acc"] else ""
    fig.suptitle(
        f"Task hold vs dominant-mode $\\tau_{{\\mathrm{{eff}}}}$  "
        f"\u2014  {_COND_LABELS[cond]},  g = {G_FOCUS_17}{acc_str}",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# Figure 15a — Fixed τ
_fig15a = _make_tau_hold_fig("False")
if _fig15a is not None:
    if SAVE_FIGS:
        _fig15a.savefig(os.path.join(FIGS_DIR, f"15a_tau_vs_hold_fixed_g{G_FOCUS_17}.pdf"),
                        bbox_inches="tight", dpi=150)
    plt.show()

# Figure 15b — Learnable τ
_fig15b = _make_tau_hold_fig("True")
if _fig15b is not None:
    if SAVE_FIGS:
        _fig15b.savefig(os.path.join(FIGS_DIR, f"15b_tau_vs_hold_learnable_g{G_FOCUS_17}.pdf"),
                        bbox_inches="tight", dpi=150)
    plt.show()

# %%
