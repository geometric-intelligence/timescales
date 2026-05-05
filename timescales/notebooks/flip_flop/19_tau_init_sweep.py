# %% [markdown]
# # Tau Initialisation Sweep — 6-bit Flip-Flop (Identity, Learnable τ)
#
# Sweep: `flip_flop_tau_init_sweep`
#
# Linear (Identity) RNN, N=512, normal-scaled W_rec, per-neuron learnable τ.
# Two initialisation conditions crossed with gains g ∈ {0.6, 0.8}:
#
# | Condition        | τ init                                         |
# |------------------|------------------------------------------------|
# | `tau_init_uniform`   | all τ = 1                                  |
# | `tau_init_powerlaw`  | τ ~ τ⁻¹ log-uniform over [1, 200]          |
#
# **Key question**: does the initial spread of timescales bias which slow
# modes the network finds after training?
#
# ── NOTEBOOK STRUCTURE ───────────────────────────────────────────────────────
#   PART 0 : Imports & configuration  ← SET sweep_dir HERE
#   PART 1 : Load records
#   PART 2 : Plots
#             P1  Validation loss curves (all runs)
#             P2  Validation accuracy curves (all runs)
#             P3  Final loss & accuracy vs gain (conditions overlaid)

# %% Imports
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

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()
os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())

# ══════════════════════════════════════════════════════════════════════════════
# %% QUICK START — set sweep_dir before running
# ══════════════════════════════════════════════════════════════════════════════
sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/flip_flop_tau_init_sweep_20260501_180308"   # ← e.g. "logs/experiments/flip_flop_tau_init_sweep_20260501_120000"

SAVE_FIGS = True
FIGS_DIR  = os.path.join("notebooks", "flip_flop", "figs", "tau_init_sweep")
os.makedirs(FIGS_DIR, exist_ok=True)

# Expected gains and init conditions (must match the sweep config)
GAINS      = [0.6, 0.8]
INIT_TYPES = ["uniform", "powerlaw"]   # tau_init_{type}

# Pulse rates for the 6-bit task (used only for display)
P_PULSE = [0.005, 0.007, 0.01, 0.02, 0.05, 0.1]
HOLDS   = [1.0 / p for p in P_PULSE]

# Threshold for P4 "steps to criterion" plot
ACC_THRESHOLD = 0.9   # fraction; adjust as needed

# ── Cosmetics ────────────────────────────────────────────────────────────────
_GAIN_COLORS = {0.6: "#2a9d8f", 0.8: "#f4a261"}
_INIT_STYLE  = {
    "uniform":  dict(ls="-",  marker="o", label="τ init = 1 (uniform)"),
    "powerlaw": dict(ls="--", marker="s", label=r"τ init ~ τ⁻¹ (log-uniform)"),
}

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 1 — Load records
# ══════════════════════════════════════════════════════════════════════════════
assert sweep_dir, "Set sweep_dir above before running!"

records = []

for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
        continue

    # Parse experiment name: g_{gain}_tau_init_{init_type}
    parts = exp_name.split("_")
    try:
        assert parts[0] == "g"
        gain = float(parts[1])
        assert parts[2] == "tau" and parts[3] == "init"
        init_type = parts[4]   # "uniform" or "powerlaw"
    except (IndexError, ValueError, AssertionError):
        print(f"  Skipping unrecognised experiment: {exp_name}")
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

        # Final validation loss from job_result if available
        fvl = None
        rf  = os.path.join(seed_path, "job_result.yaml")
        if os.path.exists(rf):
            with open(rf) as f:
                fvl = yaml.safe_load(f).get("final_val_loss")

        # Training curves
        val_losses, val_accs, steps = [], [], []
        lf = os.path.join(seed_path, "training_losses.json")
        if os.path.exists(lf):
            with open(lf) as f:
                ld = json.load(f)
            val_losses = ld.get("val_losses", [])
            val_accs   = ld.get("val_accuracies", [])
            steps      = ld.get("steps", [])

        if fvl is None and val_losses:
            fvl = val_losses[-1]
        if fvl is None:
            print(f"  No loss data for {exp_name}/seed_{seed} — skipping")
            continue

        final_acc = val_accs[-1] if val_accs else None

        records.append(dict(
            exp_name=exp_name, gain=gain, init_type=init_type, seed=seed,
            seed_path=seed_path, run_config=run_config,
            final_val_loss=fvl, final_val_acc=final_acc,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
        ))

df = pd.DataFrame(records)
print(f"\nLoaded {len(df)} runs")
if not df.empty:
    print(df[["gain", "init_type", "final_val_loss", "final_val_acc"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 2 — Plots
# ══════════════════════════════════════════════════════════════════════════════

# %% P1  Validation loss curves
# One column per gain; both init conditions overlaid per panel.
fig, axes = plt.subplots(1, len(GAINS), figsize=(6 * len(GAINS), 4.5),
                          sharey=True, squeeze=False)
axes = axes[0]

for ax, g in zip(axes, GAINS):
    sub = df[df["gain"] == g]
    for init_type in INIT_TYPES:
        rows = sub[sub["init_type"] == init_type]
        sty  = _INIT_STYLE[init_type]
        for _, row in rows.iterrows():
            vl = row["val_losses"]
            st = row["steps"]
            xs = st[:len(vl)] if st else list(range(len(vl)))
            ax.plot(xs, vl,
                    color=_GAIN_COLORS.get(g, "gray"),
                    ls=sty["ls"], linewidth=1.8,
                    label=sty["label"])
    ax.set_title(f"g = {g}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    # Deduplicate legend entries
    _handles, _labels = ax.get_legend_handles_labels()
    _seen = {}
    for h, l in zip(_handles, _labels):
        if l not in _seen:
            _seen[l] = h
    ax.legend(_seen.values(), _seen.keys(), fontsize=9, loc="upper right")

axes[0].set_ylabel("Validation loss", fontsize=11)
fig.suptitle("Validation Loss — Tau Init Sweep  (6-bit Hetero Flip-Flop, Identity)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "P1_val_loss.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% P2  Validation accuracy curves
fig, axes = plt.subplots(1, len(GAINS), figsize=(6 * len(GAINS), 4.5),
                          sharey=True, squeeze=False)
axes = axes[0]

for ax, g in zip(axes, GAINS):
    sub = df[df["gain"] == g]
    for init_type in INIT_TYPES:
        rows = sub[sub["init_type"] == init_type]
        sty  = _INIT_STYLE[init_type]
        for _, row in rows.iterrows():
            va = row["val_accs"]
            st = row["steps"]
            xs = st[:len(va)] if st else list(range(len(va)))
            ax.plot(xs, va,
                    color=_GAIN_COLORS.get(g, "gray"),
                    ls=sty["ls"], linewidth=1.8,
                    label=sty["label"])
    ax.set_title(f"g = {g}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training step", fontsize=11)
    #ax.set_ylim(0.3, 1.02)
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.35)
    ax.axhline(ACC_THRESHOLD, color="red", ls="--", lw=1.8, alpha=0.85)
    ax.grid(True, alpha=0.25)
    _handles, _labels = ax.get_legend_handles_labels()
    _seen = {}
    for h, l in zip(_handles, _labels):
        if l not in _seen:
            _seen[l] = h
    ax.legend(_seen.values(), _seen.keys(), fontsize=9, loc="lower right")

axes[0].set_ylabel("Validation accuracy", fontsize=11)
fig.suptitle("Validation Accuracy — Tau Init Sweep  (6-bit Hetero Flip-Flop, Identity)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "P2_val_acc.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% P3  Final loss & accuracy vs gain — init conditions overlaid
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (metric, ylabel, do_log) in zip(axes, [
    ("final_val_loss", "Final validation loss",     True),
    ("final_val_acc",  "Final validation accuracy", False),
]):
    for init_type in INIT_TYPES:
        sub = df[df["init_type"] == init_type].sort_values("gain")
        if sub.empty:
            continue
        sty = _INIT_STYLE[init_type]
        colors = [_GAIN_COLORS.get(g, "gray") for g in sub["gain"]]
        ax.plot(sub["gain"], sub[metric],
                color="#333333", ls=sty["ls"], linewidth=1.8,
                marker=sty["marker"], markersize=9,
                label=sty["label"])
        # colour each marker by gain
        for g_val, y_val in zip(sub["gain"], sub[metric]):
            ax.scatter(g_val, y_val,
                       color=_GAIN_COLORS.get(g_val, "gray"),
                       s=80, zorder=5, edgecolors="white", linewidths=0.8)
    if do_log:
        ax.set_yscale("log")
    ax.set_xlabel("Recurrent gain $g$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(GAINS)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

# Gain colour legend
from matplotlib.lines import Line2D as _Line2D
_gain_handles = [
    _Line2D([0], [0], marker="o", color="w",
            markerfacecolor=_GAIN_COLORS[g], markeredgecolor="white",
            markersize=9, label=f"g = {g}")
    for g in GAINS
]
axes[1].legend(handles=_gain_handles, fontsize=9, loc="lower right",
               title="Gain (dot colour)")

fig.suptitle("Final Metrics vs Gain — Tau Init Sweep",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "P3_final_metrics.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% P4  Steps to criterion — strip plot
# x-axis: gain (one position per g).  y-axis: steps to criterion (log).
# Init condition encoded by marker + colour; no connecting lines.
# Open v = DNF (run placed at last recorded step as a lower bound).
# More seeds → point cloud per x-position; median + CI can be layered later.

def _steps_to_threshold(steps, val_accs, threshold):
    """Return the first step where val_acc >= threshold, or None."""
    for s, a in zip(steps, val_accs):
        if a is not None and a >= threshold:
            return s
    return None


# Build table
_p4_rows = []
for _, row in df.iterrows():
    st = row["steps"]
    va = row["val_accs"]
    xs = st[:len(va)] if st else list(range(len(va)))
    hit = _steps_to_threshold(xs, va, ACC_THRESHOLD)
    _p4_rows.append(dict(
        gain=row["gain"],
        init_type=row["init_type"],
        seed=row["seed"],
        step=hit if hit is not None else (xs[-1] if xs else None),
        reached=hit is not None,
    ))
_p4_df = pd.DataFrame(_p4_rows).dropna(subset=["step"])

_INIT_CLR = {"uniform": "#555555", "powerlaw": "#e76f51"}

fig, ax = plt.subplots(figsize=(3 + 1.5 * len(GAINS), 5))

for init_type in INIT_TYPES:
    sty = _INIT_STYLE[init_type]
    clr = _INIT_CLR[init_type]
    sub = _p4_df[_p4_df["init_type"] == init_type]
    for _, r in sub.iterrows():
        xp = r["gain"]
        yp = r["step"]
        if r["reached"]:
            ax.scatter(xp, yp, color=clr, marker=sty["marker"],
                       s=90, edgecolors="white", linewidths=0.9, zorder=4)
        else:
            ax.scatter(xp, yp, color=clr, marker="v",
                       s=80, facecolors="none", edgecolors=clr,
                       linewidths=1.5, zorder=4)

ax.set_yscale("log")
ax.set_xticks(GAINS)
ax.set_xticklabels([f"g = {g}" for g in GAINS], fontsize=11)
ax.set_xlim(min(GAINS) - 0.15, max(GAINS) + 0.15)
ax.set_ylabel(f"Steps to {ACC_THRESHOLD:.0%} accuracy", fontsize=11)
ax.grid(True, axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(bottom=False)

# Init-type legend outside
from matplotlib.lines import Line2D as _LD4
_leg4 = [
    _LD4([0], [0], marker=_INIT_STYLE[it]["marker"], color="w",
         markerfacecolor=_INIT_CLR[it], markeredgecolor="white",
         markersize=9, label=_INIT_STYLE[it]["label"])
    for it in INIT_TYPES
]
ax.legend(handles=_leg4, fontsize=9, ncol=1,
          loc="upper left", bbox_to_anchor=(1.02, 1.0),
          borderaxespad=0, framealpha=0.85)

ax.set_title(f"Steps to criterion  (acc ≥ {ACC_THRESHOLD:.0%})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "P4_steps_to_criterion.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — COUPLING COMPUTATIONS
# Run once after PART 1 to compute Jacobians, Schur/eigen decompositions,
# connectivity and correlation couplings.  Produces _coupling dict.
# ══════════════════════════════════════════════════════════════════════════════

# %% Helpers
from scipy.linalg import schur as _scipy_schur
from datamodules.flip_flop import FlipFlopDataModule as _FFdm

N_TRAJ_CORR = 50   # trajectories used for correlation forward pass


def _build_jacobian(state_dict: dict, run_config: dict):
    """
    Build J = diag(1-α) + diag(α)·g·W_rec and return (J, alphas).
    Works for both fixed-tau and learnable-tau runs.
    """
    g  = float(run_config["recurrent_gain"])
    dt = float(run_config["dt"])

    W_rec = log_tau = None
    for k, v in state_dict.items():
        if "W_rec.weight" in k:       W_rec   = v.numpy()
        if "log_time_constants" in k: log_tau = v.numpy()

    if W_rec is None:
        raise RuntimeError("W_rec.weight not found")
    N = W_rec.shape[0]

    if run_config.get("learn_time_constants") and log_tau is not None:
        taus   = np.exp(log_tau)
        alphas = 1.0 - np.exp(-dt / taus)
    else:
        tc_cfg = run_config.get("time_constants_config", {})
        tau_v  = float(tc_cfg.get("values", [1.0])[0])
        alphas = np.full(N, 1.0 - np.exp(-dt / tau_v))

    J = np.diag(1.0 - alphas) + alphas[:, None] * g * W_rec
    return J, alphas


def _schur_sort(J: np.ndarray):
    """Real Schur decomp, columns sorted by |λ| descending."""
    T_mat, Q_mat = _scipy_schur(J, output="real")
    N = T_mat.shape[0]
    col_abs = np.zeros(N);  blk_lbl = np.empty(N, dtype=object)
    k = 0
    while k < N:
        if k + 1 < N and abs(T_mat[k + 1, k]) > 1e-10:
            ae = np.abs(np.linalg.eigvals(T_mat[k:k+2, k:k+2])[0])
            col_abs[k] = col_abs[k+1] = ae
            blk_lbl[k] = "2x2-a";  blk_lbl[k+1] = "2x2-b"
            k += 2
        else:
            col_abs[k] = abs(T_mat[k, k])
            blk_lbl[k] = "1x1";  k += 1
    idx   = np.argsort(col_abs)[::-1]
    tau_s = -1.0 / np.where(np.log(np.clip(col_abs[idx], 1e-12, None)) < -1e-10,
                             np.log(np.clip(col_abs[idx], 1e-12, None)), -1e-10)
    return Q_mat[:, idx], col_abs[idx], tau_s, blk_lbl[idx]


def _pearson_r(Z, Y):
    """Pearson r → shape (Kz, Ky).  Z:(M, Kz), Y:(M, Ky)."""
    Zc = Z - Z.mean(0, keepdims=True);  Yc = Y - Y.mean(0, keepdims=True)
    Zs = np.where(Zc.std(0) > 1e-12, Zc.std(0), 1e-12)
    Ys = np.where(Yc.std(0) > 1e-12, Yc.std(0), 1e-12)
    return (Zc.T @ Yc) / (Z.shape[0] * Zs[:, None] * Ys[None, :])


def _load_state(seed_path: str, which: str = "trained"):
    """Load checkpoint state dict.  which = 'trained' | 'untrained'."""
    import glob as _glob
    if which == "untrained":
        p = os.path.join(seed_path, "checkpoints", "untrained.ckpt")
        if os.path.exists(p):
            return torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        return None
    best = _glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best:
        return None
    return torch.load(best[0], map_location="cpu", weights_only=False)["state_dict"]


# %% Compute couplings for all runs
# Produces _coupling[(gain, init_type)] with keys:
#   eigs_untrained, eigs_trained   – complex eigenvalue arrays
#   eig_abs_rank                   – indices into eigs_trained sorted by |λ|
#   dom_eig_conn  {bi: eig_idx}    – dominant eig by eigenmode connectivity
#   dom_schur_conn{bi: eig_idx}    – dominant eig by Schur connectivity
#   dom_eig_corr  {bi: eig_idx}    – dominant eig by eigenmode |r|
#   dom_schur_corr{bi: eig_idx}    – dominant eig by Schur |r|  (or None)
#   coup_eig      (n_bits, N)      – |W_out V|, sorted by |λ|
#   coup_schur    (n_bits, N)      – |W_out Q|, sorted by |λ|
#   r_eig_tgt     (N, n_bits)      – Pearson r, eig-basis  (or None)
#   r_schur_tgt   (N, n_bits)      – Pearson r, Schur basis (or None)
#   tau_eff       (N,)             – τ_eff for trained modes

_coupling = {}

for _, row in df.iterrows():
    g         = row["gain"]
    it        = row["init_type"]
    seed_path = row["seed_path"]
    rc        = row["run_config"]
    n_bits    = rc["n_bits"]

    print(f"\n── g={g}  init={it} ─────────────────────")

    # ── Load checkpoints ──────────────────────────────────────────────────────
    state_trn = _load_state(seed_path, "trained")
    state_unt = _load_state(seed_path, "untrained")
    if state_trn is None:
        print("  No trained checkpoint — skipped"); continue

    # ── Jacobians & eigenvalues ───────────────────────────────────────────────
    J_trn, alphas = _build_jacobian(state_trn, rc)
    eigs_trn, V_trn = np.linalg.eig(J_trn)            # complex; original order

    eigs_unt = None
    if state_unt is not None:
        J_unt    = _build_jacobian(state_unt, rc)[0]
        eigs_unt = np.linalg.eig(J_unt)[0]

    # Sort trained eigenvalues by |λ| descending
    eig_abs_rank = np.argsort(np.abs(eigs_trn))[::-1]  # indices into eigs_trn

    # ── Schur decomposition ───────────────────────────────────────────────────
    Q_s, abs_s, tau_s, blk_s = _schur_sort(J_trn)
    N = J_trn.shape[0]

    # ── Weights ───────────────────────────────────────────────────────────────
    W_out = W_in = None
    for k, v in state_trn.items():
        if "W_out.weight" in k: W_out = v.numpy()
        if "W_in.weight"  in k: W_in  = v.numpy()

    if W_out is None:
        print("  W_out not found — skipped"); continue

    # ── Connectivity couplings ────────────────────────────────────────────────
    # Eigenbasis (original order): |W_out @ V|
    coup_eig_raw = np.abs(W_out @ V_trn).real          # (n_bits, N) original order
    coup_eig     = coup_eig_raw[:, eig_abs_rank]        # sorted by |λ|

    # Schur basis (already sorted by |λ|): |W_out @ Q|
    coup_schur   = np.abs(W_out @ Q_s)                 # (n_bits, N) sorted

    # Dominant eigenvalue index per bit (connectivity)
    dom_eig_conn   = {bi: int(eig_abs_rank[int(np.argmax(coup_eig[bi]))])
                      for bi in range(n_bits)}
    dom_schur_conn = {bi: int(eig_abs_rank[int(np.argmax(coup_schur[bi]))])
                      for bi in range(n_bits)}

    # ── Forward pass for correlation (Identity dynamics: exact) ───────────────
    r_eig_tgt = r_schur_tgt = None
    dom_eig_corr = dom_schur_corr = None

    if W_in is not None:
        try:
            dm = _FFdm(
                n_bits=n_bits,
                p_pulse=rc["p_pulse"],
                pulse_amplitude=rc["pulse_amplitude"],
                num_time_steps=rc["num_time_steps"],
                num_val_trajectories=N_TRAJ_CORR,
                batch_size=N_TRAJ_CORR,
            )
            dm.setup()
            inp_np, _, tgt_np = dm.val_dataset.tensors
            inp_np  = inp_np.numpy()
            tgt_np  = tgt_np.numpy()
            B, T, _ = inp_np.shape
            tgt_flat = tgt_np.reshape(-1, n_bits)

            # h[t+1] = J h[t] + diag(α) W_in u[t]
            AW  = alphas[:, None] * W_in   # (N, n_in)
            h   = np.zeros((B, T, N), dtype=np.float32)
            ht  = np.zeros((B, N),    dtype=np.float32)
            for t in range(T):
                ht = ht @ J_trn.T + inp_np[:, t, :] @ AW.T
                h[:, t, :] = ht
            h_flat = h.reshape(-1, N)

            # Schur projections
            Z_schur     = h_flat @ Q_s
            r_schur_tgt = _pearson_r(Z_schur, tgt_flat)           # (N, n_bits)

            # Eigenvector projections (real part of V^{-T} h)
            V_inv    = np.linalg.pinv(V_trn)
            Z_eig    = np.real(h_flat @ V_inv.T)                   # (n_traj*T, N)
            r_eig_tgt = _pearson_r(Z_eig, tgt_flat)               # (N, n_bits)

            dom_schur_corr = {bi: int(eig_abs_rank[int(np.argmax(np.abs(r_schur_tgt[:, bi])))])
                              for bi in range(n_bits)}
            dom_eig_corr   = {bi: int(np.argmax(np.abs(r_eig_tgt[:, bi])))
                              for bi in range(n_bits)}
            print(f"  Correlation done.  dom_schur_corr = {dom_schur_corr}")
        except Exception as exc:
            print(f"  Forward pass failed: {exc}")
    else:
        print("  W_in not found — correlation skipped")

    print(f"  dom_schur_conn = {dom_schur_conn}")

    _coupling[(g, it)] = dict(
        eigs_trained=eigs_trn, eigs_untrained=eigs_unt,
        eig_abs_rank=eig_abs_rank,
        tau_eff=tau_s,
        coup_eig=coup_eig, coup_schur=coup_schur,
        r_eig_tgt=r_eig_tgt, r_schur_tgt=r_schur_tgt,
        dom_eig_conn=dom_eig_conn,     dom_schur_conn=dom_schur_conn,
        dom_eig_corr=dom_eig_corr,     dom_schur_corr=dom_schur_corr,
        n_bits=n_bits,
        final_val_acc=row["final_val_acc"],
    )

print("\nCoupling computation complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — SPECTRUM PLOTS
# Each plot is a grid: rows = gains, cols = init_types × {untrained, trained}
# ══════════════════════════════════════════════════════════════════════════════

# %% Spectrum plot constants
_UNTRAINED_COLOR   = "#8da0b5"   # light blue-grey
_BULK_COLOR        = "#e76f51"   # salmon
_UNIT_CIRCLE_COLOR = "#c0392b"

_bit_colors = [f"C{i}" for i in range(10)]   # one colour per bit

_theta = np.linspace(0, 2 * np.pi, 300)

# Zoom window — set to None for auto
ZOOM_RE = (0.5, 1.06)
ZOOM_IM = (-0.28, 0.28)


def _plot_spectrum_grid(criterion_label: str, dom_key: str, filename: str):
    """
    2 (gains) × 4 cols (uniform-untrained, uniform-trained,
                         powerlaw-untrained, powerlaw-trained) grid.
    dom_key selects which dominant-mode dict to colour-highlight.
    """
    col_order = [(it, phase) for it in INIT_TYPES for phase in ("untrained", "trained")]
    n_rows = len(GAINS);  n_cols = len(col_order)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 3.6 * n_rows),
                             squeeze=False)

    for ri, g in enumerate(GAINS):
        for ci, (it, phase) in enumerate(col_order):
            ax  = axes[ri, ci]
            key = (g, it)
            d   = _coupling.get(key)

            # Unit circle
            ax.plot(np.cos(_theta), np.sin(_theta),
                    color=_UNIT_CIRCLE_COLOR, lw=1.0, alpha=0.7, zorder=1)
            ax.axhline(0, color="#ccc", lw=0.3, zorder=0)
            ax.axvline(0, color="#ccc", lw=0.3, zorder=0)

            if d is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([]); continue

            if phase == "untrained":
                eigs = d["eigs_untrained"]
                if eigs is None:
                    ax.text(0.5, 0.5, "no untrained\ncheckpoint",
                            transform=ax.transAxes, ha="center", va="center", fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([]); continue
                ax.scatter(eigs.real, eigs.imag, s=12, alpha=0.55,
                           c=_UNTRAINED_COLOR, edgecolors="none", zorder=2)
            else:
                eigs   = d["eigs_trained"]
                dom    = d.get(dom_key) or {}
                n_bits = d["n_bits"]

                dom_eig_set = set(dom.values())
                rest = np.array([i for i in range(len(eigs)) if i not in dom_eig_set], dtype=int)
                if len(rest):
                    ax.scatter(eigs[rest].real, eigs[rest].imag,
                               s=12, alpha=0.50, c=_BULK_COLOR,
                               edgecolors="none", zorder=2)
                for bi, ei in dom.items():
                    ax.scatter(eigs[ei].real, eigs[ei].imag,
                               s=80, color=_bit_colors[bi],
                               edgecolors="white", linewidths=0.9, zorder=5,
                               label=f"Bit {bi}")
                    # Conjugate partner (if complex eigenvalue)
                    if abs(eigs[ei].imag) > 1e-6:
                        conj_dists = np.abs(eigs - np.conj(eigs[ei]))
                        for ci2 in np.where(conj_dists < 1e-7)[0]:
                            if ci2 != ei:
                                ax.scatter(eigs[ci2].real, eigs[ci2].imag, s=40,
                                           alpha=0.75, color=_bit_colors[bi],
                                           edgecolors="white", linewidths=0.6, zorder=4)

                _hs, _ls = ax.get_legend_handles_labels()
                if _hs and ci == 1:
                    ax.legend(fontsize=6.5, loc="upper left",
                              framealpha=0.8, ncol=1)

            if ZOOM_RE is not None:
                ax.set_xlim(*ZOOM_RE)
                ax.set_ylim(*ZOOM_IM)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.1)
            ax.set_xlabel("Re($\\lambda$)", fontsize=8)
            ax.set_ylabel("Im($\\lambda$)", fontsize=8)

            acc_str = f"  acc={d['final_val_acc']:.2f}" if phase == "trained" else ""
            ax.set_title(f"{it}  |  {'Untrained' if phase=='untrained' else 'Trained'}{acc_str}",
                         fontsize=7.5, fontweight="bold")

        axes[ri, 0].set_ylabel(f"g = {g}\nIm($\\lambda$)", fontsize=9)

    fig.suptitle(f"Jacobian Spectrum — {criterion_label}\n"
                 f"Coloured markers = dominant eigenvalue per bit",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, filename), bbox_inches="tight", dpi=150)
    plt.show()


# %% P4  Spectrum — Schur→output connectivity
_plot_spectrum_grid(
    criterion_label="Schur \u2192 output (connectivity  |W_out Q|)",
    dom_key="dom_schur_conn",
    filename="P4_spectrum_schur_conn.pdf",
)

# %% P5  Spectrum — Schur→output correlation
_plot_spectrum_grid(
    criterion_label="Schur \u2192 output (|correlation|  Pearson r)",
    dom_key="dom_schur_corr",
    filename="P5_spectrum_schur_corr.pdf",
)

# %% P6  Spectrum — Eigenmode→output connectivity
_plot_spectrum_grid(
    criterion_label="Eigenmode \u2192 output (connectivity  |W_out V|)",
    dom_key="dom_eig_conn",
    filename="P6_spectrum_eig_conn.pdf",
)

# %% P7  Spectrum — Eigenmode→output correlation
_plot_spectrum_grid(
    criterion_label="Eigenmode \u2192 output (|correlation|  Pearson r)",
    dom_key="dom_eig_corr",
    filename="P7_spectrum_eig_corr.pdf",
)

# %%
