# %% [markdown]
# # Tanh — Rate vs Voltage Dynamics, Fixed τ = 1 — 6-bit Flip-Flop
#
# Sweep: `flip_flop_tanh_dyn_sweep`
#
# **Scope of this notebook:** the `learn_time_constants = False` arm only
# (per-neuron τ fixed at 1).  Two axes remain:
#   * `g`              ∈ {0.5, 0.7, 0.9, 1.1, 1.3}
#   * `dynamics_type`  ∈ {rate, voltage}
#
# Activation: **Tanh**, `wrec_init = normal_scaled`, N=512.
# Gain range extends supercritical (g > 1) because Tanh saturation keeps
# trajectories bounded.
#
# ## Jacobian convention used throughout
#
# We linearize the discrete-time update at the origin h = 0.  Define
#
#       A = diag(α) ,   α_i = 1 − exp(−dt / τ_i) .
#
# With per-neuron α (here α = 1 − exp(−dt) is the same for every neuron because
# every τ_i = 1), the Jacobian is
#
#       **J = (I − A) + g · A · W_rec**
#
# i.e. `J_ij = (1 − α_i) δ_ij + α_i · g · W_rec[i, j]`.  This formula holds
# for both rate and voltage Tanh dynamics because tanh'(0) = 1, so the
# spectra differ only because the *trained* W_rec lands in different places
# under each formulation.
#
# **Headline question (this notebook):** what does this Jacobian spectrum
# look like for the **voltage-dynamics** trained networks, and how does it
# compare to the rate-dynamics counterpart?
#
# **Plot menu**
# 1.   Training loss curves                  (rate vs voltage)
# 2.   Training accuracy curves              (rate vs voltage)
# 3.   Final loss / accuracy vs gain         (rate-vs-voltage overlay)
# 4.   COMPUTE — Jacobians, eigvals, untrained baseline
# 5.   COMPUTE — correlation-based dominant eigvals per bit (nonlinear rollout)
# 6.   PLOT 5  — **Voltage Jacobian eigenspectra**, full plane and zoom
# 7.   PLOT 6  — Rate vs Voltage spectra side-by-side per gain
# 8.   PLOT 7  — τ_eff scree, voltage runs

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
from datamodules.flip_flop import FlipFlopDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_FIGS = True
FIGS_DIR = os.path.join("notebooks", "flip_flop", "figs", "tanh_dyn")
os.makedirs(FIGS_DIR, exist_ok=True)


# %% ── CONFIG ────────────────────────────────────────────────────────────────
sweep_dir = (
    "/home/facosta/timescales/timescales/logs/experiments/"
    "flip_flop_tanh_dyn_sweep_20260426_014801"
)

GAINS      = [0.5, 0.7, 0.9, 1.1, 1.3]
DYNAMICS   = ["rate", "voltage"]

# Fixed-τ arm only — see notebook header for rationale.
FIXED_COND = "False"
CONDITIONS = [FIXED_COND]

# Bit hold intervals from p_pulse (in *timesteps*, not physical time)
P_PULSE   = [0.005, 0.007, 0.01, 0.02, 0.05, 0.1]
HOLDS     = [1.0 / p for p in P_PULSE]         # ≈ [200, 143, 100, 50, 20, 10]
bit_colors = [f"C{i}" for i in range(len(HOLDS))]

# Number of validation trajectories used for the (nonlinear) Schur-correlation
# pass — keep modest for speed, increase if correlations look noisy.
N_TRAJ_CORR = 50

TAU_VIS_MAX = 1000      # cap on τ_eff for the scree plot

# Cosmetic palettes
_GAIN_COLORS = {
    0.5: "#2a9d8f",     # teal
    0.7: "#e9c46a",     # mustard
    0.9: "#f4a261",     # peach
    1.1: "#e76f51",     # salmon
    1.3: "#9b2226",     # crimson
}
_DYN_COLOR = {"rate": "#1d4ed8", "voltage": "#9b2226"}   # rate=blue, voltage=crimson
_DYN_LS    = {"rate": "-",       "voltage": "--"}
_DYN_MK    = {"rate": "o",       "voltage": "s"}
_DYN_LABEL = {"rate": "Rate · Fixed τ=1",
              "voltage": "Voltage · Fixed τ=1"}


# %% ── HELPERS ───────────────────────────────────────────────────────────────

def _parse_exp_name(name: str) -> tuple[float, str, str] | None:
    """
    Parse 'g_{gain}_dyn_{rate|voltage}_learn_tau_{True|False}'.

    Returns (gain, dynamics_type, condition) or None if the name doesn't match.
    """
    parts = name.split("_")
    try:
        if (parts[0] != "g" or parts[2] != "dyn" or
                parts[4] != "learn" or parts[5] != "tau"):
            return None
        gain     = float(parts[1])
        dyn      = parts[3]               # "rate" / "voltage"
        cond     = parts[6]               # "True" / "False"
        if dyn not in DYNAMICS or cond not in CONDITIONS:
            return None
        return gain, dyn, cond
    except (IndexError, ValueError):
        return None


def _load_ckpt_state(seed_path: str, which: str = "trained") -> dict | None:
    """Load checkpoint state dict.  which='trained'|'untrained'."""
    if which == "untrained":
        p = os.path.join(seed_path, "checkpoints", "untrained.ckpt")
        if os.path.exists(p):
            return torch.load(p, map_location="cpu",
                              weights_only=False)["state_dict"]
        return None
    best = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best:
        return None
    return torch.load(best[0], map_location="cpu",
                      weights_only=False)["state_dict"]


def _compute_jacobian(ckpt_state: dict,
                      run_config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Jacobian of the discrete-time update at the origin h = 0.

    With A = diag(α) and α_i = 1 − exp(−dt / τ_i),

        **J = (I − A) + g · A · W_rec**

    componentwise  J_ij = (1 − α_i) δ_ij + α_i · g · W_rec[i, j].

    Same formula for rate and voltage Tanh dynamics (tanh'(0) = 1).  The
    notebook is restricted to the fixed-τ arm, so α is a single scalar
    broadcast to every neuron (`learn_time_constants = False`); the
    `learn_tau=True` branch is kept here only for completeness.

    Returns (J, alphas) where alphas has shape (N,).
    """
    g         = float(run_config["recurrent_gain"])
    dt        = float(run_config["dt"])
    learn_tau = run_config["learn_time_constants"]

    W_rec   = None
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
        taus   = np.exp(log_tau)
        alphas = 1.0 - np.exp(-dt / taus)
    else:
        tau_val      = float(run_config["time_constants_config"]["values"][0])
        alpha_scalar = 1.0 - np.exp(-dt / tau_val)
        alphas       = np.full(N, alpha_scalar)

    # J = (I - A) + g A W_rec   with A = diag(alphas)
    J = np.diag(1.0 - alphas) + alphas[:, np.newaxis] * g * W_rec
    return J, alphas


def _tau_eff_from_eigs(eigs: np.ndarray, cap: float | None = None) -> np.ndarray:
    """Effective timescale in *steps*: τ_eff = −1 / ln|λ|."""
    log_abs = np.log(np.clip(np.abs(eigs), 1e-12, None))
    tau     = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)
    if cap is not None:
        tau = np.clip(tau, 0.0, cap)
    return tau


def _build_lit_model(run_config: dict, sd: dict) -> RNNLightning:
    """Re-instantiate the RNN + Lightning wrapper and load the state dict."""
    rc    = run_config
    model = RNN(
        input_size            = rc["n_bits"],
        hidden_size           = rc["hidden_size"],
        output_size           = rc["n_bits"],
        dt                    = rc["dt"],
        time_constants_config = rc.get("time_constants_config"),
        activation            = getattr(nn, rc["activation"]),
        learn_time_constants  = rc["learn_time_constants"],
        init_time_constant    = rc.get("init_time_constant"),
        shared_time_constant  = rc["shared_time_constant"],
        normalize_hidden      = rc["normalize_hidden"],
        zero_diag_wrec        = rc["zero_diag_wrec"],
        recurrent_gain        = rc["recurrent_gain"],
        noise_std             = 0.0,
        wrec_init             = rc["wrec_init"],
        alpha_parameterization = rc["alpha_parameterization"],
        dynamics_type         = rc["dynamics_type"],
    )
    lit = RNNLightning(
        model         = model,
        learning_rate = rc["learning_rate"],
        weight_decay  = rc["weight_decay"],
        step_size     = rc.get("lr_step_size", rc.get("step_size", 1000)),
        gamma         = rc["gamma"],
        task          = "flip_flop",
    )
    lit.load_state_dict(sd)
    lit.eval()
    return lit


# %% ── LOAD RECORDS ──────────────────────────────────────────────────────────
assert os.path.isdir(sweep_dir), f"sweep_dir not found: {sweep_dir}"

records = []
for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or exp_name == "configs":
        continue

    parsed = _parse_exp_name(exp_name)
    if parsed is None:
        continue
    gain, dyn, cond = parsed
    if cond != FIXED_COND:
        continue

    for sdn in sorted(os.listdir(exp_dir)):
        if not sdn.startswith("seed_"):
            continue
        seed = int(sdn.split("_")[1])
        seed_path = os.path.join(exp_dir, sdn)

        cf = os.path.join(seed_path, "run_config.yaml")
        if not os.path.exists(cf):
            continue
        with open(cf) as f:
            run_config = yaml.safe_load(f)

        fvl = None
        rf  = os.path.join(seed_path, "job_result.yaml")
        if os.path.exists(rf):
            with open(rf) as f:
                fvl = yaml.safe_load(f).get("final_val_loss")

        val_losses, val_accs, steps = [], [], []
        val_accs_per_bit = {}
        lf = os.path.join(seed_path, "training_losses.json")
        if os.path.exists(lf):
            with open(lf) as f:
                ld = json.load(f)
            val_losses        = ld.get("val_losses", [])
            val_accs          = ld.get("val_accuracies", [])
            steps             = ld.get("steps", [])
            val_accs_per_bit  = ld.get("val_accuracies_per_bit", {})

        if fvl is None and val_losses:
            fvl = val_losses[-1]
        if fvl is None:
            continue

        records.append(dict(
            exp_name=exp_name, gain=gain, dyn=dyn, condition=cond, seed=seed,
            seed_path=seed_path, run_config=run_config,
            final_val_loss=fvl,
            final_val_acc=val_accs[-1] if val_accs else None,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
            val_accs_per_bit=val_accs_per_bit,
        ))

df = pd.DataFrame(records)
print(f"Loaded {len(df)} runs")
print(df[["gain", "dyn", "condition", "final_val_loss", "final_val_acc"]]
      .sort_values(["dyn", "condition", "gain"]).to_string(index=False))


# %% ── PLOT 1: Training loss curves (1 × 2: rate | voltage) ────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True, sharey=True)

for ci, dyn in enumerate(DYNAMICS):
    ax  = axes[ci]
    sub = df[df["dyn"] == dyn].sort_values("gain")
    for _, row in sub.iterrows():
        g  = row["gain"]
        st = row["steps"]
        vl = row["val_losses"]
        steps_x = st[:len(vl)] if st else list(range(len(vl)))
        ax.plot(steps_x, vl,
                color=_GAIN_COLORS[g], linewidth=1.7, label=f"g={g}")
    ax.set_title(_DYN_LABEL[dyn], fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Training step", fontsize=10)
    if ci == 0:
        ax.set_ylabel("Validation loss", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="upper right")

fig.suptitle("Training loss — Tanh, rate vs voltage (fixed τ = 1)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "01_train_loss.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% ── PLOT 2: Training accuracy curves (1 × 2) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True, sharey=True)

for ci, dyn in enumerate(DYNAMICS):
    ax  = axes[ci]
    sub = df[df["dyn"] == dyn].sort_values("gain")
    for _, row in sub.iterrows():
        g  = row["gain"]
        st = row["steps"]
        va = row["val_accs"]
        steps_x = st[:len(va)] if st else list(range(len(va)))
        ax.plot(steps_x, va,
                color=_GAIN_COLORS[g], linewidth=1.7, label=f"g={g}")
    ax.set_title(_DYN_LABEL[dyn], fontsize=11, fontweight="bold")
    ax.set_ylim(0.3, 1.02)
    ax.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Training step", fontsize=10)
    if ci == 0:
        ax.set_ylabel("Validation accuracy", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="lower right")

fig.suptitle("Training accuracy — Tanh, rate vs voltage (fixed τ = 1)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "02_train_acc.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% ── PLOT 3: Final loss / accuracy vs gain (rate vs voltage) ────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, metric, ylabel in zip(
    axes,
    ["final_val_acc", "final_val_loss"],
    ["Final validation accuracy", "Final validation loss"],
):
    for dyn in DYNAMICS:
        sub = df[df["dyn"] == dyn].sort_values("gain")
        if sub.empty:
            continue
        ax.plot(sub["gain"], sub[metric],
                color=_DYN_COLOR[dyn], ls=_DYN_LS[dyn],
                marker=_DYN_MK[dyn], markersize=8, linewidth=1.8,
                label=_DYN_LABEL[dyn])
    ax.set_xlabel("Recurrent gain $g$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    if "loss" in metric:
        ax.set_yscale("log")

axes[0].legend(fontsize=10, loc="lower left")
fig.suptitle("Final loss / accuracy vs gain — rate vs voltage (fixed τ = 1)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "03_summary_by_gain.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% ── COMPUTE 5: Jacobians + eigenvalues, both trained and untrained ──────
# eig_data[g][dyn][cond] = {"trained": {...}, "untrained": {...}}
eig_data = {g: {d: {c: {} for c in CONDITIONS} for d in DYNAMICS}
            for g in GAINS}

for _, row in df.iterrows():
    g       = row["gain"]
    dyn     = row["dyn"]
    cond    = row["condition"]
    sp      = row["seed_path"]
    rc      = row["run_config"]

    for label in ("trained", "untrained"):
        sd = _load_ckpt_state(sp, label)
        if sd is None:
            continue
        try:
            J, alphas = _compute_jacobian(sd, rc)
        except RuntimeError:
            continue
        eigs, eigvecs = np.linalg.eig(J)
        W_out = None
        taus  = None
        for k, v in sd.items():
            if "W_out.weight" in k:
                W_out = v.numpy()
            if "log_time_constants" in k:
                taus = np.exp(v.numpy())
        eig_data[g][dyn][cond][label] = dict(
            J=J, eigs=eigs, eigvecs=eigvecs,
            W_out=W_out, alphas=alphas, taus=taus,
        )

print("Eigenvalue data computed for:")
for g in GAINS:
    have = []
    for d in DYNAMICS:
        for c in CONDITIONS:
            for lbl in ("trained", "untrained"):
                if lbl in eig_data[g][d][c]:
                    have.append(f"{d}/{c}/{lbl}")
    print(f"  g={g}:  {have}")


# %% ── COMPUTE 6: Correlation-based dominant eigvalue per bit ──────────────
# For each *trained* run we
#   1. instantiate the actual nonlinear RNN and run it on validation inputs;
#   2. real-Schur-decompose the (linearized) Jacobian and project hidden
#      states onto the Schur basis;
#   3. compute Pearson r(Schur projection, target bit) and pick the dominant
#      Schur column per bit;
#   4. map that Schur column back to the matching eigvalue index in
#      eig_data[g][dyn][cond]["trained"]["eigs"].
#
# Why nonlinear forward (not linearized)?  Supercritical g > 1 makes the
# linearized rollout blow up, so we need the actual Tanh dynamics for stable
# correlations.  The Schur basis itself is still computed from J (the Jacobian
# at the origin) — that is the spectrum we ultimately want to colour.
#
# Result: _dom_eig_idx[g][dyn][cond] = {bit_idx: [eig_idx, ...]}

from scipy.linalg import schur as _sp_schur


_dom_eig_idx = {g: {d: {c: None for c in CONDITIONS} for d in DYNAMICS}
                for g in GAINS}
_corr_summary = []

for _, row in df.iterrows():
    g       = row["gain"]
    dyn     = row["dyn"]
    cond    = row["condition"]
    sp      = row["seed_path"]
    rc      = row["run_config"]

    if "trained" not in eig_data[g][dyn][cond]:
        continue
    sd = _load_ckpt_state(sp, "trained")
    if sd is None:
        continue

    J        = eig_data[g][dyn][cond]["trained"]["J"]
    eigs_arr = eig_data[g][dyn][cond]["trained"]["eigs"]
    N        = J.shape[0]
    n_bits   = rc["n_bits"]

    # Real Schur decomposition of J; assign one (possibly complex) eigvalue
    # per Schur column, taking 2x2 conjugate blocks into account.
    Tr, Qr  = _sp_schur(J, output="real")
    col_ev  = np.zeros(N, dtype=complex)
    k = 0
    while k < N:
        if k + 1 < N and abs(Tr[k+1, k]) > 1e-10:
            ev2 = np.linalg.eigvals(Tr[k:k+2, k:k+2])
            col_ev[k]   = ev2[0]
            col_ev[k+1] = ev2[1]
            k += 2
        else:
            col_ev[k] = Tr[k, k]
            k += 1
    srt    = np.argsort(np.abs(col_ev))[::-1]
    Q_s    = Qr[:, srt]
    ev_s   = col_ev[srt]

    # Nonlinear forward pass on validation data
    try:
        lit = _build_lit_model(rc, sd).to(device)
    except Exception as e:
        print(f"  [skip] g={g} {dyn} {cond}: model build failed ({e})")
        continue

    dm = FlipFlopDataModule(
        n_bits           = n_bits,
        p_pulse          = rc["p_pulse"],
        pulse_amplitude  = rc["pulse_amplitude"],
        num_time_steps   = rc["num_time_steps"],
        num_val_trajectories = N_TRAJ_CORR,
        batch_size       = N_TRAJ_CORR,
    )
    dm.setup()
    inp_t, _, tgt_t = dm.val_dataset.tensors

    with torch.no_grad():
        h_seq, _ = lit.model(inp_t.to(device))
    h_arr = h_seq.cpu().numpy()             # (B, T, N)
    if not np.isfinite(h_arr).all():
        print(f"  [skip] g={g} {dyn} {cond}: nonlinear rollout had NaN/Inf")
        continue

    h_flat = h_arr.reshape(-1, N)           # (B*T, N)
    t_flat = tgt_t.numpy().reshape(-1, n_bits)
    Z      = h_flat @ Q_s                   # (B*T, N) — Schur projections
    Zc     = Z      - Z.mean(0, keepdims=True)
    Yc     = t_flat - t_flat.mean(0, keepdims=True)
    Zs     = np.where(Zc.std(0) > 1e-12, Zc.std(0), 1e-12)
    Ys     = np.where(Yc.std(0) > 1e-12, Yc.std(0), 1e-12)
    R      = (Zc.T @ Yc) / (h_flat.shape[0] * Zs[:, None] * Ys[None, :])

    bit_to_eigidx = {}
    for bi in range(n_bits):
        dom_col   = int(np.argmax(np.abs(R[:, bi])))
        target_ev = ev_s[dom_col]
        best      = int(np.argmin(np.abs(eigs_arr - target_ev)))
        idxs      = [best]
        if not np.isreal(target_ev):
            conj = int(np.argmin(np.abs(eigs_arr - np.conj(target_ev))))
            if conj != best:
                idxs.append(conj)
        bit_to_eigidx[bi] = idxs

    _dom_eig_idx[g][dyn][cond] = bit_to_eigidx

    rmax = [float(np.max(np.abs(R[:, bi]))) for bi in range(n_bits)]
    _corr_summary.append((g, dyn, cond, rmax))
    print(f"  g={g} {dyn:>7} {cond:>5}: |r|_max per bit = "
          + " ".join(f"{x:.2f}" for x in rmax))

print("Correlation dominant-eigenvalue mapping complete.")


# %% ── PLOT 5: VOLTAGE Jacobian eigenspectra (main figure) ────────────────
# Spectrum of   J = (I − A) + g · A · W_rec   (linearization of the voltage
# Tanh dynamics at h = 0).  One panel per gain (rows).  Light-blue dots =
# untrained baseline, salmon = trained bulk, coloured rings = dominant
# Schur mode per bit (Pearson r between the Schur projection of the *actual
# nonlinear* hidden trajectory and the corresponding target bit).
# Requires: eig_data, _dom_eig_idx, df, GAINS, bit_colors, FIXED_COND

theta = np.linspace(0, 2 * np.pi, 400)
n_g   = len(GAINS)

fig, axes = plt.subplots(n_g, 1, figsize=(6.2, 5.0 * n_g), squeeze=False)
for ri, g in enumerate(GAINS):
    ax   = axes[ri, 0]
    d_un = eig_data[g]["voltage"][FIXED_COND].get("untrained", {})
    d_tr = eig_data[g]["voltage"][FIXED_COND].get("trained",   {})

    ax.plot(np.cos(theta), np.sin(theta),
            color="#c0392b", lw=1.0, ls="-", alpha=0.7, zorder=2)
    ax.axhline(0, color="#ccc", lw=0.3)
    ax.axvline(0, color="#ccc", lw=0.3)

    if "eigs" in d_un:
        ue = d_un["eigs"]
        ax.scatter(ue.real, ue.imag, s=8, color="#c0d8e8",
                   alpha=0.55, edgecolors="none", zorder=2.5,
                   label="untrained" if ri == 0 else "_nolegend_")

    if "eigs" in d_tr:
        te       = d_tr["eigs"]
        n_eig    = len(te)
        corr_dom = _dom_eig_idx[g]["voltage"][FIXED_COND] or {}

        dom_set    = set()
        idx_to_bit = {}
        for bi, idxs in corr_dom.items():
            for ii in idxs:
                dom_set.add(ii)
                idx_to_bit[ii] = bi
        rest = np.array([i not in dom_set for i in range(n_eig)])

        ax.scatter(te.real[rest], te.imag[rest], s=12,
                   color="#e76f51", alpha=0.6, edgecolors="none", zorder=3,
                   label="trained (bulk)" if ri == 0 else "_nolegend_")

        seen = set()
        for ii, bi in idx_to_bit.items():
            lbl = (f"bit {bi}" if (bi not in seen and ri == 0)
                   else "_nolegend_")
            seen.add(bi)
            ax.scatter(te.real[ii], te.imag[ii], s=55,
                       color=bit_colors[bi],
                       edgecolors="white", linewidths=0.6, zorder=6,
                       label=lbl)

        rho = float(np.abs(te).max())
        ax.annotate(f"|λ|_max = {rho:.3f}",
                    xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", alpha=0.7))

    rrec = df[(df["gain"] == g) & (df["dyn"] == "voltage")]
    if not rrec.empty and rrec.iloc[0]["final_val_acc"] is not None:
        ax.annotate(f"acc = {rrec.iloc[0]['final_val_acc']:.3f}",
                    xy=(0.03, 0.95), xycoords="axes fraction",
                    ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", alpha=0.7))

    ax.set_aspect("equal")
    rmax = 1.05
    if "eigs" in d_tr:
        rmax = max(rmax, 1.05 * float(np.abs(d_tr["eigs"]).max()))
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.grid(True, alpha=0.1)
    ax.set_ylabel(f"g = {g}\nIm(λ)", fontsize=10)
    if ri == n_g - 1:
        ax.set_xlabel("Re(λ)", fontsize=10)

axes[0, 0].legend(fontsize=7, loc="lower left", ncol=2, framealpha=0.9)

fig.suptitle(
    "Voltage-dynamics Jacobian eigenspectra  —  J = (I − A) + g A W\n"
    "fixed τ = 1, A = diag(α), unit circle = stability boundary",
    fontsize=12, fontweight="bold", y=1.0)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "05_voltage_eigenspectra.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% ── PLOT 5b: VOLTAGE spectra — zoom near the unit circle ────────────────
# Same data as PLOT 5 zoomed to Re ∈ [0.7, max(ρ, 1.02)], Im ∈ [-0.4, 0.4]
# to inspect the slow modes the network builds.

fig, axes = plt.subplots(n_g, 1, figsize=(7, 3.4 * n_g), squeeze=False)
for ri, g in enumerate(GAINS):
    ax   = axes[ri, 0]
    d_un = eig_data[g]["voltage"][FIXED_COND].get("untrained", {})
    d_tr = eig_data[g]["voltage"][FIXED_COND].get("trained",   {})

    ax.plot(np.cos(theta), np.sin(theta),
            color="#c0392b", lw=1.0, ls="-", alpha=0.7, zorder=2)
    ax.axhline(0, color="#ccc", lw=0.3)
    ax.axvline(0, color="#ccc", lw=0.3)

    if "eigs" in d_un:
        ue = d_un["eigs"]
        ax.scatter(ue.real, ue.imag, s=8, color="#c0d8e8",
                   alpha=0.55, edgecolors="none", zorder=2.5)
    if "eigs" in d_tr:
        te       = d_tr["eigs"]
        n_eig    = len(te)
        corr_dom = _dom_eig_idx[g]["voltage"][FIXED_COND] or {}
        dom_set    = set()
        idx_to_bit = {}
        for bi, idxs in corr_dom.items():
            for ii in idxs:
                dom_set.add(ii); idx_to_bit[ii] = bi
        rest = np.array([i not in dom_set for i in range(n_eig)])
        ax.scatter(te.real[rest], te.imag[rest], s=12,
                   color="#e76f51", alpha=0.6, edgecolors="none", zorder=3)
        for ii, bi in idx_to_bit.items():
            ax.scatter(te.real[ii], te.imag[ii], s=55,
                       color=bit_colors[bi],
                       edgecolors="white", linewidths=0.6, zorder=6)

    #ax.set_aspect("equal")
    rho = 1.02
    if "eigs" in d_tr:
        rho = max(rho, 1.02 * float(np.abs(d_tr["eigs"]).max()))
    ax.set_xlim(0.8, rho)
    ax.set_ylim(-0.12, 0.12)
    ax.grid(True, alpha=0.1)
    ax.set_ylabel(f"g = {g}\nIm(λ)", fontsize=10)
    if ri == n_g - 1:
        ax.set_xlabel("Re(λ)", fontsize=10)

fig.suptitle(
    "Voltage-dynamics Jacobian eigenspectra — zoom near unit circle",
    fontsize=12, fontweight="bold", y=1.0)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "05b_voltage_eigenspectra_zoom.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% ── PLOT 6: Rate vs Voltage — spectra side-by-side per gain ──────────────
# Two-column layout: [Rate · Fixed τ | Voltage · Fixed τ], one row per gain.
# Useful for checking whether rate- and voltage-trained networks converge to
# qualitatively different spectra of  J = (I − A) + g A W.

n_cells = len(DYNAMICS)

fig, axes = plt.subplots(n_g, n_cells,
                          figsize=(4.5 * n_cells, 4.0 * n_g), squeeze=False)
for ri, g in enumerate(GAINS):
    for ci, dyn in enumerate(DYNAMICS):
        ax   = axes[ri, ci]
        d_un = eig_data[g][dyn][FIXED_COND].get("untrained", {})
        d_tr = eig_data[g][dyn][FIXED_COND].get("trained",   {})

        ax.plot(np.cos(theta), np.sin(theta),
                color="#c0392b", lw=1.0, ls="-", alpha=0.7, zorder=2)
        ax.axhline(0, color="#ccc", lw=0.3)
        ax.axvline(0, color="#ccc", lw=0.3)

        if "eigs" in d_un:
            ue = d_un["eigs"]
            ax.scatter(ue.real, ue.imag, s=6, color="#c0d8e8",
                       alpha=0.5, edgecolors="none", zorder=2.5)
        if "eigs" in d_tr:
            te       = d_tr["eigs"]
            n_eig    = len(te)
            corr_dom = _dom_eig_idx[g][dyn][FIXED_COND] or {}
            dom_set    = set()
            idx_to_bit = {}
            for bi, idxs in corr_dom.items():
                for ii in idxs:
                    dom_set.add(ii); idx_to_bit[ii] = bi
            rest = np.array([i not in dom_set for i in range(n_eig)])
            ax.scatter(te.real[rest], te.imag[rest], s=10,
                       color="#e76f51", alpha=0.6, edgecolors="none", zorder=3)
            for ii, bi in idx_to_bit.items():
                ax.scatter(te.real[ii], te.imag[ii], s=45,
                           color=bit_colors[bi],
                           edgecolors="white", linewidths=0.5, zorder=6)
            rho = float(np.abs(te).max())
            ax.text(0.97, 0.05, f"ρ={rho:.2f}",
                    transform=ax.transAxes, ha="right", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", alpha=0.8))
        rrec = df[(df["gain"] == g) & (df["dyn"] == dyn)]
        if not rrec.empty and rrec.iloc[0]["final_val_acc"] is not None:
            ax.text(0.03, 0.95, f"acc={rrec.iloc[0]['final_val_acc']:.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", alpha=0.8))

        ax.set_aspect("equal")
        rmax = 1.05
        if "eigs" in d_tr:
            rmax = max(rmax, 1.05 * float(np.abs(d_tr["eigs"]).max()))
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.grid(True, alpha=0.1)

        if ri == 0:
            ax.set_title(_DYN_LABEL[dyn], fontsize=11, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"g = {g}\nIm(λ)", fontsize=10)
        if ri == n_g - 1:
            ax.set_xlabel("Re(λ)", fontsize=10)

fig.suptitle(
    "Jacobian eigenspectra at the origin — Rate vs Voltage  (fixed τ = 1)\n"
    "J = (I − A) + g A W,   A = diag(α)",
    fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "06_spectra_grid.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% ── PLOT 7: τ_eff scree (voltage runs only) ──────────────────────────────
# One panel per gain.  Bulk eigenvalues = grey dots, bit-dominant Schur
# eigenvalues = coloured rings, hold intervals = dashed coloured horizontal
# lines.  τ_eff = -1/ln|λ|, in *steps*.

fig, axes = plt.subplots(n_g, 1, figsize=(8, 3.4 * n_g),
                          squeeze=False, sharey=True)
for ri, g in enumerate(GAINS):
    ax   = axes[ri, 0]
    d_tr = eig_data[g]["voltage"][FIXED_COND].get("trained", {})
    if "eigs" not in d_tr:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        continue
    eigs = d_tr["eigs"]
    N    = len(eigs)
    rank_by_abs = np.argsort(np.abs(eigs))[::-1]
    eigs_s   = eigs[rank_by_abs]
    tau_eff  = _tau_eff_from_eigs(eigs_s, cap=TAU_VIS_MAX)
    ranks    = np.arange(1, N + 1)

    corr_dom = _dom_eig_idx[g]["voltage"][FIXED_COND] or {}
    idx_to_bit = {ii: bi for bi, lst in corr_dom.items() for ii in lst}
    rank_to_bit = {}
    for r, ii in enumerate(rank_by_abs):
        if ii in idx_to_bit:
            rank_to_bit[r] = idx_to_bit[ii]

    highlighted = set(rank_to_bit.keys())
    non_high    = [r for r in range(N) if r not in highlighted]

    ax.scatter(ranks[non_high], tau_eff[non_high],
               s=10, color="#adb5bd", alpha=0.55, edgecolors="none",
               zorder=2)
    for r, bi in rank_to_bit.items():
        ax.scatter(ranks[r], tau_eff[r], s=32,
                   color=bit_colors[bi], edgecolors="black",
                   linewidths=0.6, zorder=5)
        ax.annotate(f"b{bi} τ={tau_eff[r]:.0f}",
                    (ranks[r], tau_eff[r]),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=7, color=bit_colors[bi], fontweight="bold",
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=bit_colors[bi],
                                    lw=0.4, alpha=0.4))
    for bi, h in enumerate(HOLDS):
        ax.axhline(h, color=bit_colors[bi], lw=0.9, ls=":", alpha=0.55)

    ax.set_yscale("log")
    ax.set_xlim(-1, N + 1)
    ax.grid(True, alpha=0.15)
    ax.set_ylabel(f"g = {g}\nτ_eff (steps)", fontsize=10)
    if ri == n_g - 1:
        ax.set_xlabel("Eigenmode rank (by |λ|)", fontsize=10)

legend_lines = [Line2D([0], [0], color=bit_colors[bi], ls=":",
                        label=f"bit {bi} hold≈{HOLDS[bi]:.0f} steps")
                for bi in range(len(HOLDS))]
fig.legend(handles=legend_lines, fontsize=8, loc="upper right",
           bbox_to_anchor=(1.01, 0.99), framealpha=0.9)

fig.suptitle(
    "τ_eff scree — voltage dynamics, fixed τ = 1 "
    "(rings = bit-dominant Schur mode)",
    fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "07_voltage_tau_scree.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()
