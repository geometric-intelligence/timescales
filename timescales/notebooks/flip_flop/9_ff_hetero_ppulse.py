# %% [markdown]
# # Flip-Flop with Heterogeneous Pulse Rates
#
# Each bit receives pulses at a different rate (p_pulse per bit),
# creating hold intervals of different lengths. The key question: does the
# trained network develop eigenvalues with **different** effective timescales
# matching the different hold intervals?
#
# ── NOTEBOOK STRUCTURE ────────────────────────────────────────────────────────
#   PART 0 : Imports & configuration  ← SET sweep_dir AND G_FOCUS HERE
#   PART 1 : Helper functions         (pure — define once, never re-run)
#   PART 2 : Data loading             (I/O heavy — run once per session)
#             2.1  Training records
#             2.2  Eigendecompositions (all gains)
#             2.3  G_FOCUS model + forward pass
#             [2.4 PCA — commented out]
#   PART 3 : Coupling computations    (numpy — run once per session)
#             3A   Schur decomposition for G_FOCUS
#             3A-ii  Non-normality of J
#             3B   Connectivity coupling matrices
#             3C   Correlation matrices (Pearson r)
#             3D   Correlation-derived quantities (dominant modes, PRs)
#             3E   Ablation sweeps (C1 / C3)
#   PART 4 : Plots                    (re-run individual cells freely)
#             P01-P03  Training / example trajectories
#             ── Coupling framework note ──
#             P-A1..6  Connectivity coupling        (G_FOCUS only)
#             P-B1..6  Correlation coupling         (G_FOCUS only)
#             P-C1,3,4 Ablation / causal coupling   (G_FOCUS only)
#             ── Spectrum & scree (after coupling) ──
#             P04  Non-normality / Schur factor T
#             P05  Jacobian eigenspectrum            (G_FOCUS only)
#             P05b All 4 coupling criteria spectra   (G_FOCUS only, 2x2 panel)
#             P06  tau_eff scree, detailed           (G_FOCUS only)
#             P07  Autocorrelation: top-N modes and neurons (exp fits)
#             [P-D1..2 PCA dimensionality — commented out]

# ══════════════════════════════════════════════════════════════════════════════
# PART 0 — IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

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
import torch.nn as nn
from scipy.linalg import schur as scipy_schur
from numpy.linalg import svd as _svd

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

# ══════════════════════════════════════════════════════════════════════════════
# %% QUICK START — edit only this block before running
# ══════════════════════════════════════════════════════════════════════════════

# ── Required: set these two before running anything ───────────────────────────
sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/flip_flop_hetero_ppulse_20260412_081857"
#             ^ path to the sweep output folder (contains g_0.5/, g_0.7/, ...)
G_FOCUS   = 0.5
#             ^ recurrent gain to analyse in detail (must be present in sweep_dir)

# ── Optional: secondary settings (safe to leave at defaults) ─────────────────
EXCLUDE_GAINS      = {0.95}        # gains to omit from loss/accuracy plots
N_HEATMAP_DIMS     = 12            # max number of modes/neurons shown on heatmap x-axis
SPECTRUM_HIGHLIGHT = 'correlation' # which criterion highlights "top" modes in spectrum
#                                    options: 'connectivity' | 'correlation'
SAVE_FIGS          = True          # write PDFs to figs/ sub-directory
N_TRAJ             = 100           # number of val trajectories for forward pass
#                                    (used for correlation and ablation computations)
# PCA_N_TRAJ = N_TRAJ              # (PCA is currently commented out)

FIGS_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

# colour helpers (populated after df is loaded)
UNTRAINED_COLOR = "#8da0b5"
TRAINED_COLOR   = "#e76f51"
theta = np.linspace(0, 2 * np.pi, 200)   # for unit-circle plots


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — HELPER FUNCTIONS  (pure functions — define once, never re-run)
# ══════════════════════════════════════════════════════════════════════════════

# %% Helper functions

def _schur_sort(J: np.ndarray):
    """
    Real Schur decomposition of J, columns of Q sorted by |λ| descending.

    Returns
    -------
    Q_sorted       : (N, N) real orthogonal
    abs_eig_sorted : (N,)   |λ| per Schur column
    tau_sorted     : (N,)   τ_eff = -1/ln|λ|
    block_label    : (N,)   '1x1' | '2x2-a' | '2x2-b'
    """
    T, Q = scipy_schur(J, output="real")
    N = T.shape[0]
    col_abs_eig = np.zeros(N)
    block_label = np.empty(N, dtype=object)
    k = 0
    while k < N:
        if k + 1 < N and abs(T[k + 1, k]) > 1e-10:
            abs_e = np.abs(np.linalg.eigvals(T[k:k + 2, k:k + 2])[0])
            col_abs_eig[k] = col_abs_eig[k + 1] = abs_e
            block_label[k] = "2x2-a"
            block_label[k + 1] = "2x2-b"
            k += 2
        else:
            col_abs_eig[k] = abs(T[k, k])
            block_label[k] = "1x1"
            k += 1
    sort_idx = np.argsort(col_abs_eig)[::-1]
    Q_s = Q[:, sort_idx]
    abs_s = col_abs_eig[sort_idx]
    log_abs = np.log(np.clip(abs_s, 1e-12, None))
    tau_s = -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)
    return Q_s, abs_s, tau_s, block_label[sort_idx]


def _pearson_r_matrix(Z, Y):
    """
    Pearson r between every column of Z (shape M×K_z) and every column of Y
    (shape M×K_y).  Returns r_mat of shape (K_z, K_y).
    """
    n = Z.shape[0]
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Z_std = np.where(Zc.std(axis=0) > 1e-12, Zc.std(axis=0), 1e-12)
    Y_std = np.where(Yc.std(axis=0) > 1e-12, Yc.std(axis=0), 1e-12)
    return (Zc.T @ Yc) / (n * Z_std[:, None] * Y_std[None, :])


def _pr_vec(coupling_vec):
    """Participation ratio of a coupling vector (PR = 1 → one mode, PR=N → uniform)."""
    c2 = coupling_vec ** 2
    c4 = coupling_vec ** 4
    return c2.sum() ** 2 / c4.sum() if c4.sum() > 1e-30 else float("nan")


def _silence(h, W_out, Q, mode_indices, bias):
    """Post-hoc readout ablation: remove mode_indices contribution from h → output."""
    y = h @ W_out.T + bias
    for j in mode_indices:
        z_j = h @ Q[:, j]
        c_j = W_out @ Q[:, j]
        y = y - z_j[..., None] * c_j
    return y


def _corr_heatmap_profile(r_mat, title_prefix, row_labels, col_labels,
                           xlabel_row, ylabel_col,
                           cmap="RdBu_r", top_k=None, save_path=None):
    """
    Correlation heatmap (top_k rows × K cols) with a max-|r| bar chart.
    Returns dom_idx dict: {col_idx → row_idx of dominant dimension}.
    """
    M, K = r_mat.shape
    top_k = top_k if top_k is not None else min(N_HEATMAP_DIMS, M)

    _rl = [row_labels(i) for i in range(top_k)] if callable(row_labels) else list(row_labels[:top_k])

    hm = r_mat[:top_k, :]
    max_abs = np.max(np.abs(r_mat), axis=1)

    fig_h = max(5.0, 0.45 * top_k + 1.5)
    fig, (ax_hm, ax_bar) = plt.subplots(1, 2, figsize=(14, fig_h),
                                          gridspec_kw=dict(width_ratios=[1, 0.38]))

    im = ax_hm.imshow(hm, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax_hm.set_yticks(range(top_k))
    ax_hm.set_yticklabels(_rl, fontsize=7)
    ax_hm.set_xticks(range(K))
    ax_hm.set_xticklabels(col_labels, fontsize=8)
    for _j in range(top_k):
        for _i in range(K):
            _v = hm[_j, _i]
            ax_hm.text(_i, _j, f"{_v:+.2f}", ha="center", va="center",
                       fontsize=6, color="white" if abs(_v) > 0.55 else "black")
    plt.colorbar(im, ax=ax_hm, label="Pearson $r$", shrink=0.55)
    ax_hm.set_xlabel(xlabel_row, fontsize=11)
    ax_hm.set_ylabel(ylabel_col, fontsize=11)
    ax_hm.set_title(title_prefix, fontsize=10)

    best_bit = np.argmax(np.abs(r_mat), axis=1)
    bit_cls = [f"C{i}" for i in range(K)]
    ax_bar.barh(range(top_k), max_abs[:top_k],
                color=[bit_cls[best_bit[j]] for j in range(top_k)],
                edgecolor="none", alpha=0.85)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("max $|r|$ across bits", fontsize=10)
    ax_bar.set_yticks(range(top_k))
    ax_bar.set_yticklabels(_rl, fontsize=7)
    ax_bar.grid(True, alpha=0.2, axis="x")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_title("Max $|r|$ per dim\n(color = dominant bit)", fontsize=9)
    plt.tight_layout()
    if save_path and SAVE_FIGS:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

    dom_idx = {_bi: int(np.argmax(np.abs(r_mat[:, _bi]))) for _bi in range(K)}
    print(f"  PR (correlation):")
    print(f"  {'Bit':<6} {'dominant dim':<16} {'max |r|':<10} {'PR'}")
    for _bi in range(K):
        c_vec = r_mat[:, _bi] ** 2
        pr = c_vec.sum() ** 2 / (c_vec ** 2).sum() if (c_vec ** 2).sum() > 1e-30 else float("nan")
        dom = dom_idx[_bi]
        print(f"  {_bi:<6} dim {dom+1:<11}  {abs(r_mat[dom, _bi]):<10.3f}  {pr:.2f}")
    return dom_idx


def _rnn_from_cfg(cfg, device="cpu"):
    """Instantiate an RNN + RNNLightning from a run config dict."""
    model = RNN(
        input_size=cfg["n_bits"],
        hidden_size=cfg["hidden_size"],
        output_size=cfg["n_bits"],
        dt=cfg["dt"],
        time_constants_config=cfg.get("time_constants_config"),
        activation=getattr(nn, cfg["activation"]),
        learn_time_constants=cfg["learn_time_constants"],
        init_time_constant=cfg.get("init_time_constant"),
        shared_time_constant=cfg["shared_time_constant"],
        normalize_hidden=cfg["normalize_hidden"],
        zero_diag_wrec=cfg["zero_diag_wrec"],
        recurrent_gain=cfg["recurrent_gain"],
        noise_std=0.0,
        wrec_init=cfg["wrec_init"],
        alpha_parameterization=cfg["alpha_parameterization"],
        dynamics_type=cfg["dynamics_type"],
    )
    lit = RNNLightning(
        model=model,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        step_size=cfg.get("lr_step_size", cfg.get("step_size", 1000)),
        gamma=cfg["gamma"],
        task="flip_flop",
    )
    return lit


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — DATA LOADING  (I/O heavy — run once per session)
# ══════════════════════════════════════════════════════════════════════════════

# %% 2.1  Load training records → df, gains
records = []

for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or not exp_name.startswith("g_"):
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

        val_losses, val_accs, steps = [], [], []
        lf = os.path.join(seed_path, "training_losses.json")
        if os.path.exists(lf):
            with open(lf) as f:
                ld = json.load(f)
            val_losses = ld.get("val_losses", ld.get("val_losses_epoch", []))
            val_accs   = ld.get("val_accuracies", [])
            steps      = ld.get("steps", [])

        if fvl is None and val_losses:
            fvl = val_losses[-1]
        if fvl is None:
            continue

        records.append(dict(
            exp_name=exp_name, gain=gain, seed=seed,
            seed_path=seed_path, final_val_loss=fvl,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
        ))

df    = pd.DataFrame(records)
gains = sorted(df["gain"].unique())
n_gains = len(gains)
print(f"Loaded {len(df)} runs, gains: {gains}")

_palette = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#f4a261", "#606c38", "#457b9d"]
COLORS   = {g: _palette[i % len(_palette)] for i, g in enumerate(gains)}

gains_plot = [g for g in gains if g not in EXCLUDE_GAINS]
_vir     = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(gains_plot) - 1, 1) + 1))
COLORS_V = {g: _vir[i] for i, g in enumerate(gains_plot)}

bit_colors = [f"C{i}" for i in range(10)]   # generic bit colours (re-used by scree)

# %% 2.2  Eigendecompositions (all gains) → eig_data
eig_data = {}

for g in gains:
    row_data  = df[df["gain"] == g].iloc[0]
    seed_path = row_data["seed_path"]
    with open(os.path.join(seed_path, "run_config.yaml")) as f:
        run_config = yaml.safe_load(f)

    n_bits_g    = run_config["n_bits"]
    p_pulse_cfg = run_config["p_pulse"]
    dt          = run_config["dt"]
    tau         = run_config["time_constants_config"]["values"][0]
    alpha       = 1.0 - np.exp(-dt / tau)

    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        continue

    ckpt  = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)
    W_rec = W_out = W_in = None
    for key, val in ckpt["state_dict"].items():
        if "W_rec.weight" in key:
            W_rec = val.numpy()
        elif "W_out.weight" in key:
            W_out = val.numpy()
        elif "W_in.weight" in key:
            W_in = val.numpy()

    if W_rec is None:
        continue

    N = W_rec.shape[0]
    J = (1.0 - alpha) * np.eye(N) + alpha * g * W_rec
    eigenvalues, eigvecs = np.linalg.eig(J)

    eig_data[g] = dict(
        J=J,                                          # stored directly — no need to reconstruct later
        eigs=eigenvalues, eigvecs=eigvecs, W_out=W_out, W_in=W_in,
        alpha=alpha, n_bits=n_bits_g, p_pulse=p_pulse_cfg,
    )

print(f"Eigendecompositions loaded for gains: {sorted(eig_data.keys())}")

# shared hold-time reference (from any gain, assumed identical across seeds)
_sample   = next(iter(eig_data.values()))
_pp_ref   = _sample["p_pulse"]
_pp_list  = _pp_ref if isinstance(_pp_ref, list) else [_pp_ref]
_holds    = [1.0 / p for p in _pp_list]

# %% 2.3  G_FOCUS model + forward pass → _pca_lit, _pca_h, _pca_tgt, _pca_inp_raw
assert G_FOCUS in eig_data, f"G_FOCUS={G_FOCUS} not found in eig_data"

_pca_row       = df[df["gain"] == G_FOCUS].iloc[0]
_pca_seed_path = _pca_row["seed_path"]
with open(os.path.join(_pca_seed_path, "run_config.yaml")) as f:
    _pca_cfg = yaml.safe_load(f)

_pca_ckpts = glob.glob(os.path.join(_pca_seed_path, "checkpoints", "best-model-*.ckpt"))
assert _pca_ckpts, f"No checkpoint found for g={G_FOCUS}"

_pca_lit = _rnn_from_cfg(_pca_cfg, device)
_sd = torch.load(_pca_ckpts[0], map_location=device, weights_only=False)
_pca_lit.load_state_dict(_sd["state_dict"])
_pca_lit.eval().to(device)

_pca_dm = FlipFlopDataModule(
    n_bits=_pca_cfg["n_bits"],
    p_pulse=_pca_cfg["p_pulse"],
    pulse_amplitude=_pca_cfg["pulse_amplitude"],
    num_time_steps=_pca_cfg["num_time_steps"],
    num_val_trajectories=N_TRAJ,
    batch_size=N_TRAJ,
)
_pca_dm.setup()
_pca_inp_raw, _, _pca_tgt = _pca_dm.val_dataset.tensors   # (n_traj, T, n_bits)

with torch.no_grad():
    _pca_h_seq, _ = _pca_lit.model(_pca_inp_raw.to(device), init_context=None)
    _pca_h = _pca_h_seq.cpu().numpy()   # (n_traj, T, H)

_pca_n_traj, _pca_T, _pca_H = _pca_h.shape
print(f"G_FOCUS forward pass: {N_TRAJ} trajectories,  shape h = {_pca_h.shape}")

# ── Small demo dataset for P03 (example trajectories) ────────────────────────
_demo_dm = FlipFlopDataModule(
    n_bits=_pca_cfg["n_bits"],
    p_pulse=_pca_cfg["p_pulse"],
    pulse_amplitude=_pca_cfg["pulse_amplitude"],
    num_time_steps=_pca_cfg["num_time_steps"],
    num_val_trajectories=10,
    batch_size=10,
)
_demo_dm.setup()
_demo_inp, _, _demo_tgt = _demo_dm.val_dataset.tensors

with torch.no_grad():
    _, _demo_out = _pca_lit.model(_demo_inp.to(device), init_context=None)
    _demo_out_prob = torch.sigmoid(_demo_out).cpu()

_p_list_demo = (_pca_cfg["p_pulse"] if isinstance(_pca_cfg["p_pulse"], list)
                else [_pca_cfg["p_pulse"]] * _pca_cfg["n_bits"])

# ── Untrained baseline for PCA comparison ─────────────────────────────────────
_unt_ckpt_path = os.path.join(_pca_seed_path, "checkpoints", "untrained.ckpt")
_has_unt = os.path.exists(_unt_ckpt_path)
if _has_unt:
    _unt_lit = _rnn_from_cfg(_pca_cfg, device)
    _unt_sd  = torch.load(_unt_ckpt_path, map_location=device, weights_only=False)
    _unt_lit.load_state_dict(_unt_sd["state_dict"])
    _unt_lit.eval().to(device)
    with torch.no_grad():
        _unt_hs, _ = _unt_lit.model(_pca_inp_raw.to(device), init_context=None)
        _unt_h = _unt_hs.cpu().numpy()
    # ── PCA (commented out) ──────────────────────────────────────────────────
    # _unt_data  = _unt_h.reshape(-1, _pca_cfg["hidden_size"])
    # _unt_data -= _unt_data.mean(axis=0, keepdims=True)
    # _, _unt_s2, _ = _svd(_unt_data, full_matrices=False)
    # _unt_var2 = _unt_s2 ** 2
    # _unt_cum  = np.cumsum(_unt_var2 / _unt_var2.sum())
    # print("Untrained baseline computed.")
    # ─────────────────────────────────────────────────────────────────────────
    print("Untrained ckpt found; PCA baseline skipped (commented out).")
else:
    print("No untrained.ckpt; baseline skipped.")

# %% 2.4  PCA of hidden states  [COMMENTED OUT — not needed for coupling/ablation]
# ── PCA (commented out) ──────────────────────────────────────────────────────
# The forward pass above (_pca_h, _pca_tgt, _pca_inp_raw) IS kept active
# because it is required by Section B (correlation) and Section C (ablation).
# Only the SVD decomposition and PCA-specific quantities are disabled here.
#
# _pca_data  = _pca_h.reshape(-1, _pca_H)
# _pca_data -= _pca_data.mean(axis=0, keepdims=True)
#
# _, _pca_s, _pca_Vt   = _svd(_pca_data, full_matrices=False)
# _pca_var      = _pca_s ** 2
# _pca_var_frac = _pca_var / _pca_var.sum()
# _pca_cum      = np.cumsum(_pca_var_frac)
#
# N_PCS_CORR   = min(_pca_cfg["n_bits"] + 6, _pca_data.shape[1])
# _n_bits_pca  = _pca_cfg["n_bits"]
# _pp_list_pca = (_pca_cfg["p_pulse"] if isinstance(_pca_cfg["p_pulse"], list)
#                 else [_pca_cfg["p_pulse"]] * _n_bits_pca)
#
# _tgt_flat = _pca_tgt.numpy().reshape(-1, _n_bits_pca)   # (n_traj*T, n_bits)
# _proj     = _pca_data @ _pca_Vt[:N_PCS_CORR].T          # (n_traj*T, N_PCS_CORR)
#
# _corr_mat = np.zeros((N_PCS_CORR, _n_bits_pca))
# for _j in range(N_PCS_CORR):
#     for _i in range(_n_bits_pca):
#         _c = np.corrcoef(_proj[:, _j], _tgt_flat[:, _i])[0, 1]
#         _corr_mat[_j, _i] = _c if np.isfinite(_c) else 0.0
#
# _pr_val = _pca_var.sum() ** 2 / np.sum(_pca_var ** 2)
#
# print(f"PCA done: {_pca_data.shape[0]} pts × {_pca_H} dims,  PR = {_pr_val:.2f}")
# for thresh in [0.80, 0.90, 0.95, 0.99]:
#     n_needed = int(np.searchsorted(_pca_cum, thresh)) + 1
#     print(f"  {thresh*100:.0f}% variance → {n_needed} PCs")
# ─────────────────────────────────────────────────────────────────────────────
print("2.4: PCA SVD skipped (commented out).  Forward pass is still active.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — COUPLING COMPUTATIONS  (numpy — run once per session)
# ══════════════════════════════════════════════════════════════════════════════

# %% 3A  Schur decomposition for G_FOCUS
# Produces: _ef, _Vf, _Wi, _Wo, _af, _nbf, _pplf, _Nf,
#           _Qf, _abs_ef, _tauf, _blkf

_d    = eig_data[G_FOCUS]
_Jf   = _d["J"]                                      # Jacobian built directly from weights
_ef   = _d["eigs"];  _Vf  = _d["eigvecs"]
_Wi   = _d["W_in"]; _Wo  = _d["W_out"]
_af   = _d["alpha"]; _nbf = _d["n_bits"]
_ppf  = _d["p_pulse"]
_pplf = _ppf if isinstance(_ppf, list) else [_ppf] * _nbf
_Nf   = len(_ef)

_Qf, _abs_ef, _tauf, _blkf = _schur_sort(_Jf)

print(f"3A: Schur done  N={_Nf},  n_bits={_nbf}")

# %% 3A-ii  Non-normality of J
# Produces:  _nn_idx, _T_offdiag_frac, _Tf  (Schur factor T)
#
# A normal matrix satisfies J^T J = J J^T and has a perfectly orthogonal
# eigenbasis.  Deviation from normality means eigenvectors can be nearly
# parallel, amplifying transient responses beyond what eigenvalues suggest.
# Three complementary diagnostics are computed below.

# -- (a) Scalar non-normality index: ||J^T J - J J^T||_F / ||J||_F^2 ----------
_comm      = _Jf.T @ _Jf - _Jf @ _Jf.T
_nn_idx    = np.linalg.norm(_comm, 'fro') / np.linalg.norm(_Jf, 'fro') ** 2
print(f"  ||J||_F                                  = {np.linalg.norm(_Jf, 'fro'):.4f}")
print(f"  Non-normality index (commutator / ||J||²) = {_nn_idx:.4f}")

# -- (b) Off-diagonal energy of Schur factor T --------------------------------
# J = Q T Q^T; T is quasi-upper-triangular.  Its strictly upper-triangular
# part encodes non-normal (inter-mode) coupling; the lower triangle is ~zero.
from scipy.linalg import schur as _scipy_schur_nn
_Tf, _Qf_nn = _scipy_schur_nn(_Jf, output='real')
# Sort columns of _Qf_nn by |λ| descending (same order as _Qf from _schur_sort)
_T_diagonly    = np.tril(_Tf)             # diagonal blocks + lower triangle
_T_offdiag     = _Tf - _T_diagonly        # strictly upper-triangular part
_T_offdiag_frac = (np.linalg.norm(_T_offdiag, 'fro') /
                   max(np.linalg.norm(_Tf, 'fro'), 1e-12))
print(f"  Off-diagonal energy of T (||T_upper|| / ||T||) = {_T_offdiag_frac:.4f}")

print("3A-ii: non-normality diagnostics done.  (_Tf, _nn_idx, _T_offdiag_frac available for P04)")

# %% 3B  Connectivity coupling matrices
# Eigenmode basis (A1, A2):
_eig_abs_rank   = np.argsort(np.abs(_ef))[::-1]      # (N,) rank order desc
_coup_eig_all   = np.abs(_Wo @ _Vf)                   # (n_bits, N) raw order
_coup_eig_ranked = _coup_eig_all[:, _eig_abs_rank]    # (n_bits, N) sorted by |λ|

# Schur basis (A3):
_coup_schur_all  = np.abs(_Wo @ _Qf)                  # (n_bits, N) sorted by |λ|

# Neuron → output (A5):
_cout       = np.abs(_Wo)                              # (n_bits, H)
_cout_nstr  = np.linalg.norm(_cout, axis=0)            # (H,)
_cout_nro   = np.argsort(_cout_nstr)[::-1]
_cout_s     = _cout[:, _cout_nro]                      # (n_bits, H) sorted by norm

if _Wi is not None:
    # Input → eigenmode (A2):
    _V_inv          = np.linalg.pinv(_Vf)              # (N, H)
    _in_eig_all     = np.abs(_V_inv @ _Wi)             # (N, n_bits)
    _in_eig_ranked  = _in_eig_all[_eig_abs_rank]       # sorted by |λ|

    # Input → Schur (A4):
    _cin = _af * np.abs(_Qf.T @ _Wi)                   # (N, n_bits) sorted by |λ|

    # Input → neuron (A6):
    _a6_win  = np.abs(_Wi)                             # (H, n_bits)
    _a6_nstr = np.linalg.norm(_a6_win, axis=1)         # (H,)
    _a6_ro   = np.argsort(_a6_nstr)[::-1]
    _a6_win_s = _a6_win[_a6_ro, :]                    # (H, n_bits) sorted by norm
    print("3B: Connectivity coupling matrices computed.")
else:
    print("3B: W_in not available — A2, A4, A6 will be skipped.")
# %%
# ── Shared colormap scales (recomputed here so plot cells are self-contained) ──
# Output heatmaps: Schur (A3) and neuron (A5) share one scale
_vmax_out_conn = max(
    _coup_schur_all[:, :N_HEATMAP_DIMS].max(),
    _cout_s[:, :N_HEATMAP_DIMS].max(),
)
# Input heatmaps: Schur (A4) and neuron (A6) share one scale
_vmax_in_conn = max(
    _cin[:N_HEATMAP_DIMS, :].max() if '_cin' in vars() else 0.0,
    _a6_win_s[:N_HEATMAP_DIMS, :].max() if '_a6_win_s' in vars() else 0.0,
)
print(f"  Shared output vmax = {_vmax_out_conn:.4f},  input vmax = {_vmax_in_conn:.4f}")

# %% 3C  Correlation matrices (Pearson r)
# Requires: _pca_h, _pca_tgt, _pca_inp_raw, _Qf, _V_inv (from 3B), _nbf

_b0_h   = _pca_h.reshape(-1, _pca_H)                  # (n_traj*T, H)
_b0_tgt = _pca_tgt.numpy().reshape(-1, _nbf)           # (n_traj*T, n_bits)
_b0_inp = _pca_inp_raw.numpy().reshape(-1, _nbf)        # (n_traj*T, n_bits)

_b0_Z_schur = _b0_h @ _Qf                              # (n_traj*T, N)
_b0_Z_eig   = np.real(_b0_h @ _V_inv.T)               # (n_traj*T, N)

_r_schur_tgt = _pearson_r_matrix(_b0_Z_schur, _b0_tgt)  # (N, n_bits)
_r_schur_inp = _pearson_r_matrix(_b0_Z_schur, _b0_inp)   # (N, n_bits)
_r_eig_tgt   = _pearson_r_matrix(_b0_Z_eig,   _b0_tgt)  # (N, n_bits)
_r_eig_inp   = _pearson_r_matrix(_b0_Z_eig,   _b0_inp)   # (N, n_bits)
_r_neu_tgt   = _pearson_r_matrix(_b0_h,        _b0_tgt)  # (H, n_bits)
_r_neu_inp   = _pearson_r_matrix(_b0_h,        _b0_inp)  # (H, n_bits)

print(f"3C: Correlation matrices computed.  Shapes:")
print(f"  schur_tgt {_r_schur_tgt.shape},  schur_inp {_r_schur_inp.shape}")
print(f"  eig_tgt   {_r_eig_tgt.shape},    eig_inp   {_r_eig_inp.shape}")
print(f"  neu_tgt   {_r_neu_tgt.shape},    neu_inp   {_r_neu_inp.shape}")

# %% 3D  Correlation-derived quantities (dominant modes, PRs, rankings)
# Dominant Schur mode per bit (by correlation) — used by SPECTRUM_HIGHLIGHT
_dom_r = {_bi: int(np.argmax(np.abs(_r_schur_tgt[:, _bi])))
          for _bi in range(_nbf)}

# B3 scree variables
_max_r_b3   = np.max(np.abs(_r_schur_tgt), axis=1)          # (N,)
_best_bit_b3 = np.argmax(np.abs(_r_schur_tgt), axis=1)       # (N,)
_bit_cls_b3  = [f"C{i}" for i in range(_nbf)]
_ranks_b3    = np.arange(1, _Nf + 1)

# Dominant Schur mode per bit — by correlation (used in B3 scree and localization)
_dom_b3  = {_bi: int(np.argmax(np.abs(_r_schur_tgt[:, _bi])))
             for _bi in range(_nbf)}
_bulk_b3 = np.ones(_Nf, dtype=bool)
for _bi in range(_nbf):
    _bulk_b3[_dom_b3[_bi]] = False

# B5/B6: neuron ranking by max |r|
_b5_max_r       = np.max(np.abs(_r_neu_tgt), axis=1)
_b5_ro          = np.argsort(_b5_max_r)[::-1]
_r_neu_tgt_ranked = _r_neu_tgt[_b5_ro, :]

_b6_max_r       = np.max(np.abs(_r_neu_inp), axis=1)
_b6_ro          = np.argsort(_b6_max_r)[::-1]
_r_neu_inp_ranked = _r_neu_inp[_b6_ro, :]

# B3 localization portrait PRs
_pr_mode_b3   = np.zeros(_nbf)
_pr_neuron_b3 = np.zeros(_nbf)
_r2_b3 = _r_schur_tgt ** 2
_r4_b3 = _r_schur_tgt ** 4
for _bi in range(_nbf):
    _r2 = _r2_b3[:, _bi]; _r4 = _r4_b3[:, _bi]
    _pr_mode_b3[_bi]   = (_r2.sum() ** 2) / _r4.sum() if _r4.sum() > 1e-30 else float("nan")
    _pr_neuron_b3[_bi] = 1.0 / np.sum(_Qf[:, _dom_b3[_bi]] ** 4)

print("3D: Derived quantities done.")
print(f"  _dom_r (by corr): {_dom_r}")
print(f"  Localization portrait  (g = {G_FOCUS}):")
print(f"  {'Bit':<6} {'PR_mode':<12} {'PR_neuron':<14} {'dom Schur':<12} {'tau_eff'}")
for _bi in range(_nbf):
    print(f"  {_bi:<6} {_pr_mode_b3[_bi]:<12.2f} {_pr_neuron_b3[_bi]:<14.2f} "
          f"Schur {_dom_b3[_bi]+1:<7} {_tauf[_dom_b3[_bi]]:.1f}")

# %% 3E  Ablation computations (C1 / C2 / C3) — loop-heavy, run once
# ── Shared readout weights & bias ─────────────────────────────────────────────
_abl_Wo   = _pca_lit.model.W_out.weight.data.cpu().numpy()   # (n_bits, H)
_abl_bias = (_pca_lit.model.W_out.bias.data.cpu().numpy()
             if _pca_lit.model.W_out.bias is not None else np.zeros(_nbf))

# ── C1: Systematic Schur mode sweep ──────────────────────────────────────────
_C1_K_ABL    = min(30, _Nf)
_c1_y        = _pca_h @ _abl_Wo.T + _abl_bias   # (n_traj, T, n_bits)
_c1_tgt_arr  = _pca_tgt.numpy()

def _acc(y, tgt):
    return ((y > 0.5).astype(float) == tgt).mean(axis=(0, 1))

_c1_acc_base = _acc(_c1_y, _c1_tgt_arr)
_c1_delta    = np.zeros((_C1_K_ABL, _nbf))
for _jj in range(_C1_K_ABL):
    _z_j = _pca_h @ _Qf[:, _jj]
    _c_j = _abl_Wo @ _Qf[:, _jj]
    _c1_delta[_jj, :] = _acc(_c1_y - _z_j[..., None] * _c_j, _c1_tgt_arr) - _c1_acc_base
_c1_tot = _c1_delta.sum(axis=1)

# ── C2: Detailed Schur ablation for bit-(n-1) ─────────────────────────────────
_abl_coup = np.abs(_abl_Wo @ _Qf)           # (n_bits, N)
_abl_dom  = np.argmax(_abl_coup, axis=1)     # dominant Schur mode per bit

BIT5  = _nbf - 1
dom5  = int(_abl_dom[BIT5])
if   _blkf[dom5] == "2x2-b": partner5 = dom5 - 1
elif _blkf[dom5] == "2x2-a": partner5 = dom5 + 1
else:                          partner5 = None

_mask5 = np.ones(_Nf, dtype=bool)
_mask5[dom5] = False
if partner5 is not None:
    _mask5[partner5] = False
sec5 = int(np.argmax(_abl_coup[BIT5] * _mask5))

_pair5 = [dom5] if partner5 is None else [partner5, dom5]
_conditions = [("Baseline", [])]
_conditions.append((f"Silence Schur {dom5+1} only\n(bit-{BIT5} dom, tau={_tauf[dom5]:.1f}, {_blkf[dom5]})", [dom5]))
if partner5 is not None:
    _conditions.append((f"Silence pair (Schur {partner5+1}+{dom5+1})", _pair5))
_conditions.append((f"Silence Schur {sec5+1}\n(bit-{BIT5} 2nd-dom, tau={_tauf[sec5]:.1f})", [sec5]))
if partner5 is not None:
    _conditions.append(("Silence pair + 2nd-dom", _pair5 + [sec5]))
for _bi in range(_nbf):
    _dm = int(_abl_dom[_bi])
    _conditions.append((f"Silence bit {_bi} dom\n(Schur {_dm+1}, tau={_tauf[_dm]:.1f})", [_dm]))

_abl_tgt_arr  = _pca_tgt.numpy()
_acc_table    = []
for _label, _modes in _conditions:
    _y_sil  = _silence(_pca_h, _abl_Wo, _Qf, _modes, _abl_bias)
    _acc_sil = _acc(_y_sil, _abl_tgt_arr)
    _acc_table.append((_label, _modes, _acc_sil))
_acc_base  = _acc_table[0][2]
_delta_arr = np.array([row[2] for row in _acc_table]) - _acc_base[None, :]

# ── C3: Neuron sweep ──────────────────────────────────────────────────────────
_C3_K_ABL      = min(50, _Nf)
_c3_coup_norm  = np.linalg.norm(_abl_Wo, axis=0)   # (H,)
_c3_rank_order = np.argsort(_c3_coup_norm)[::-1]
_c3_top_neurons = _c3_rank_order[:_C3_K_ABL]

_c3_y        = _pca_h @ _abl_Wo.T + _abl_bias
_c3_acc_base = _acc(_c3_y, _abl_tgt_arr)
_c3_delta    = np.zeros((_C3_K_ABL, _nbf))
for _ri, _kk in enumerate(_c3_top_neurons):
    _h_k   = _pca_h[:, :, _kk]
    _c_k   = _abl_Wo[:, _kk]
    _c3_delta[_ri, :] = _acc(_c3_y - _h_k[..., None] * _c_k, _abl_tgt_arr) - _c3_acc_base
_c3_tot = _c3_delta.sum(axis=1)

print("3E: Ablation sweeps done.")
print(f"  C1 K={_C1_K_ABL} modes,  C3 K={_C3_K_ABL} neurons,  C2 conds={len(_acc_table)}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — PLOTS  (re-run individual cells freely after Parts 2–3)
#
# Each cell lists its required variables in a leading comment.
# ══════════════════════════════════════════════════════════════════════════════

# ── Training & overview ───────────────────────────────────────────────────────

# %% P01  Training loss curves
# Requires: df, gains, EXCLUDE_GAINS, COLORS_V
fig, ax = plt.subplots(figsize=(8, 4))
_seen_g = set()
for _, row in df.iterrows():
    g  = row["gain"]
    if g in EXCLUDE_GAINS:
        continue
    vl = row["val_losses"]
    st = row["steps"][:len(vl)] if row["steps"] else list(range(1, len(vl) + 1))
    ax.plot(st, vl, linewidth=1.8, color=COLORS_V[g],
            label=f"g = {g}" if g not in _seen_g else None)
    _seen_g.add(g)
ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation loss", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.0, 0.82))
fig.suptitle("Hetero p_pulse Flip-Flop: Training Curves",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% P02  Training accuracy curves
# Requires: df, gains, COLORS
fig, ax = plt.subplots(figsize=(8, 4))
for _, row in df.iterrows():
    g  = row["gain"]
    va = row["val_accs"]
    st = row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1))
    if not va:
        continue
    ax.plot(st, va, linewidth=1.8, color=COLORS[g], label=f"g = {g}")
ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation accuracy", fontsize=12)
ax.set_ylim(0.4, 1.02)
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="lower right")
fig.suptitle("Hetero p_pulse Flip-Flop: Accuracy",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% P03  Example trajectories — G_FOCUS only
# Requires: _demo_inp, _demo_tgt, _demo_out_prob, _p_list_demo, _pca_cfg, G_FOCUS
SEQ_IDX = 0
n_bits_demo = _pca_cfg["n_bits"]
t_arr = np.arange(_pca_cfg["num_time_steps"])

fig, axes = plt.subplots(n_bits_demo, 1, figsize=(12, 2 * n_bits_demo), sharex=True)
if n_bits_demo == 1:
    axes = [axes]

for bit in range(n_bits_demo):
    ax  = axes[bit]
    pulse = _demo_inp[SEQ_IDX, :, bit].numpy()
    set_mask   = pulse > 0.5
    reset_mask = pulse < -0.5
    if set_mask.any():
        ax.scatter(t_arr[set_mask], np.full(set_mask.sum(), 0.65),
                   marker=6, s=25, color="C2", zorder=3,
                   label="set (+1)" if bit == 0 else None)
    if reset_mask.any():
        ax.scatter(t_arr[reset_mask], np.full(reset_mask.sum(), 0.35),
                   marker=7, s=25, color="C3", zorder=3,
                   label="reset (-1)" if bit == 0 else None)
    ax.step(t_arr, _demo_tgt[SEQ_IDX, :, bit].numpy(), where="post",
            color="black", linewidth=1.5, label="target" if bit == 0 else None)
    ax.plot(t_arr, _demo_out_prob[SEQ_IDX, :, bit].numpy(),
            color="steelblue", linewidth=1.2, alpha=0.9,
            label="output" if bit == 0 else None)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1])
    avg_hold = 1.0 / max(_p_list_demo[bit], 1e-8)
    ax.set_ylabel(f"Bit {bit}\n(p={_p_list_demo[bit]}, ~{avg_hold:.0f}steps)", fontsize=10)

axes[-1].set_xlabel("Timestep", fontsize=11)
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle(f"Hetero p_pulse Trajectories — g = {G_FOCUS}",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, f"P03_trajectories_g{G_FOCUS}.pdf"),
            bbox_inches="tight", dpi=150)
plt.show()

# %% P03b  Clean presentation — 2 selected bits, u(t) and y*(t) only
# ─────────────────────────────────────────────────────────────────────────────
# Two panels (one per selected bit).  Each panel shows:
#   • y*(t) — target state ∈ {0, 1}  as a step trace + light fill
#   • u(t)  — input pulses ∈ {-1, 0, +1} as step trace + light fill
# No network output shown.
# ─────────────────────────────────────────────────────────────────────────────
# Requires: _demo_inp, _demo_tgt, _p_list_demo, _pca_cfg, G_FOCUS

# ── Config ────────────────────────────────────────────────────────────────────
P03B_SEQ_IDX    = 0              # which trajectory in the batch to display
P03B_TARGET_P   = [0.01, 0.05]   # pick bits whose p_pulse is closest to these
P03B_T_START    = None           # first timestep to show  (None → 0)
P03B_T_END      = None           # last  timestep to show  (None → end)
P03B_SAVE       = True

_C_TARGET = "#DA627D"   # y*(t) — deep rose
_C_INPUT  = "#FFA5AB"   # u(t)  — salmon pink

# ── Select bits ───────────────────────────────────────────────────────────────
_p_arr_03b = np.array(_p_list_demo)
_sel_bits_03b = [int(np.argmin(np.abs(_p_arr_03b - pv))) for pv in P03B_TARGET_P]
# De-duplicate in case both targets map to the same bit
_sel_bits_03b = list(dict.fromkeys(_sel_bits_03b))

# ── Time window ───────────────────────────────────────────────────────────────
_T_total = _pca_cfg["num_time_steps"]
_t0 = P03B_T_START if P03B_T_START is not None else 0
_t1 = P03B_T_END   if P03B_T_END   is not None else _T_total
_t_plot = np.arange(_t0, _t1)

# ── Plot ──────────────────────────────────────────────────────────────────────
_n_panels = len(_sel_bits_03b)
fig03b, axes03b = plt.subplots(
    _n_panels, 1,
    figsize=(13, 2.6 * _n_panels),
    sharex=True,
)
if _n_panels == 1:
    axes03b = [axes03b]

for _row, _bit in enumerate(_sel_bits_03b):
    ax = axes03b[_row]

    _tgt_raw = _demo_tgt[P03B_SEQ_IDX, _t0:_t1, _bit].numpy()   # ∈ {0, 1}
    _inp     = _demo_inp[P03B_SEQ_IDX, _t0:_t1, _bit].numpy()   # ∈ {-1, 0, +1}

    # Remap target from {0, 1} → {-1, +1} to match input amplitude convention
    _tgt = 2.0 * _tgt_raw - 1.0

    # y*(t): step trace
    ax.step(_t_plot, _tgt, where="post",
            color=_C_TARGET, linewidth=1.8, label="$y^*(t)$", zorder=4)

    # u(t): step trace
    ax.step(_t_plot, _inp, where="post",
            color=_C_INPUT, linewidth=1.4, label="$u(t)$", zorder=3)

    # Horizontal baseline
    ax.axhline(0, color="#aaaaaa", linewidth=0.6, zorder=0)

    # Axis formatting
    _avg_hold = 1.0 / max(_p_list_demo[_bit], 1e-8)
    ax.set_ylim(-1.45, 1.45)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["-1", "0", "1"], fontsize=10)
    ax.set_ylabel(
        f"Amplitude\n$p={_p_list_demo[_bit]}$  "
        f"(hold $\\approx {_avg_hold:.0f}$)",
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.12, color="#bbbbbb")

axes03b[-1].set_xlabel("Time (steps)", fontsize=11)

# Legend — placed outside to the right
from matplotlib.lines import Line2D as _LD03b
fig03b.legend(
    handles=[
        _LD03b([0], [0], color=_C_TARGET, lw=2.2,
               label="$y^*(t)$ — target state"),
        _LD03b([0], [0], color=_C_INPUT,  lw=1.6,
               label="$u(t)$ — input pulse"),
    ],
    fontsize=10,
    loc="upper left",
    bbox_to_anchor=(1.01, 0.97),
    framealpha=0.9,
    borderaxespad=0,
)

fig03b.suptitle(
    f"Het-NBFF — Example Trajectories  (g = {G_FOCUS})",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
if P03B_SAVE:
    fig03b.savefig(
        os.path.join(FIGS_DIR, f"P03b_example_traj_g{G_FOCUS}.pdf"),
        bbox_inches="tight", dpi=150,
    )
plt.show()


# %% P05-COMMENTED-OUT  Effective timescale scree — all gains
# (commented out: superseded by the G_FOCUS detailed scree P06 below)
#
# from matplotlib.lines import Line2D as _Line2D
# fig, axes = plt.subplots(1, n_gains, figsize=(5 * n_gains, 4.5),
#                           squeeze=False, sharey=True)
# for col, g in enumerate(gains):
#     ax = axes[0, col]
#     if g not in eig_data:
#         continue
#     data_g  = eig_data[g]
#     eigs_g  = data_g["eigs"]
#     V_g     = data_g["eigvecs"]
#     W_out_g = data_g["W_out"]
#     n_bits_g = data_g["n_bits"]
#     N_g     = len(eigs_g)
#     abs_rank_g = np.argsort(np.abs(eigs_g))[::-1]
#     tau_eff_g  = -1.0 / np.where(
#         np.log(np.clip(np.abs(eigs_g[abs_rank_g]), 1e-12, None)) < -1e-10,
#         np.log(np.clip(np.abs(eigs_g[abs_rank_g]), 1e-12, None)),
#         -1e-10)
#     ranks_g = np.arange(1, N_g + 1)
#     coupling_g = np.abs(W_out_g @ V_g)[:, abs_rank_g] if W_out_g is not None else None
#     best_rank = ({bi: int(np.argmax(coupling_g[bi])) for bi in range(n_bits_g)}
#                  if coupling_g is not None else {})
#     non_hl = [r for r in range(N_g) if r not in best_rank.values()]
#     ax.scatter(ranks_g[non_hl], tau_eff_g[non_hl],
#                s=12, color=UNTRAINED_COLOR, edgecolors="none", alpha=0.5, zorder=3)
#     for bi, ri in best_rank.items():
#         c = bit_colors[bi]
#         ax.scatter(ranks_g[ri], tau_eff_g[ri], s=80, color=c,
#                    edgecolors="black", linewidths=0.8, zorder=5)
#         ax.annotate(f"bit {bi}\n$\\tau$={tau_eff_g[ri]:.0f}",
#                     (ranks_g[ri], tau_eff_g[ri]),
#                     textcoords="offset points", xytext=(12, 0),
#                     fontsize=7.5, color=c, fontweight="bold", va="center",
#                     arrowprops=dict(arrowstyle="-", color=c, lw=0.5, alpha=0.4))
#     for bi, h in enumerate(_holds):
#         ax.axhline(h, color=bit_colors[bi], linewidth=0.9,
#                    linestyle=":", alpha=0.5, zorder=1)
#         ax.text(N_g * 0.92, h * 1.15, f"hold≈{h:.0f}",
#                 fontsize=6.5, color=bit_colors[bi], ha="right", alpha=0.7)
#     ax.set_xlabel("Eigenvalue rank", fontsize=11)
#     if col == 0:
#         ax.set_ylabel(r"$\tau_{\mathrm{eff}} = -1\,/\,\ln|\lambda|$  (steps)", fontsize=11)
#     ax.set_title(f"$g = {g}$", fontsize=12)
#     ax.grid(True, alpha=0.12, which="both")
#     ax.set_yscale("log")
#     ax.tick_params(labelsize=9)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
# _leg_handles = [_Line2D([0], [0], marker="o", color="w",
#                          markerfacecolor=bit_colors[i], markeredgecolor="black",
#                          markeredgewidth=0.8, markersize=8,
#                          label=f"Bit {i} (hold≈{_holds[i]:.0f})")
#                 for i in range(len(_holds))]
# _leg_handles.append(_Line2D([0], [0], marker="o", color="w",
#                               markerfacecolor=UNTRAINED_COLOR,
#                               markersize=5, label="Other modes"))
# axes[0, -1].legend(handles=_leg_handles, fontsize=7, loc="center right", framealpha=0.85)
# fig.suptitle("Effective Timescales — best-coupling mode per bit highlighted",
#              fontsize=14, fontweight="bold")
# plt.tight_layout()
# if SAVE_FIGS:
#     fig.savefig(os.path.join(FIGS_DIR, "tau_eff_scree.pdf"), bbox_inches="tight", dpi=150)
# plt.show()

# %% [markdown]
# ## Coupling Analysis — Framework
#
# All coupling analyses in Sections A–C ask the same question —
# **how strongly is a given dimension (mode or neuron) linked to a given
# signal (input or output)?** — but differ along three independent axes.
#
# ### Axis 1 — Basis (what "dimension" means)
# | Label | Basis | Notes |
# |---|---|---|
# | **Neuron** | Raw hidden state h(t); no change of basis | Interpretable in terms of individual units |
# | **Eigenbasis** | Right eigenvectors V of J (J v = λ v) | Modes may be complex; eigenvalues give timescales directly |
# | **Schur modes Q** | Real Schur decomposition J = Q T Q^T | Always real; orthogonal; 2×2 blocks = oscillatory mode pairs |
#
# ### Axis 2 — Method (what kind of coupling)
# | Label | Formula | Data needed? | Interpretation |
# |---|---|---|---|
# | **Connectivity** | e.g. \|W_out Q\| | No | Structural — purely weight-based |
# | **Correlation** | Pearson r(z_j(t), signal(t)) | Yes (trajectories) | Observational/statistical — does the mode actually co-vary with the signal? |
# | **Ablation** *(causal)* | Δacc when mode j is silenced | Yes (trajectories) | Causal/interventional — "what if this mode's readout contribution were zeroed?" |
#
# > **Ablation = causal?** Yes. Silencing mode j post-hoc
# > (y^(-j) = y - z_j · c_j) is an interventional operation:
# > it tests the counterfactual "what would the output be if mode j had
# > zero readout weight?"  This is the closest analogue to a do-calculus
# > intervention available without re-training the network.
#
# ### Axis 3 — Direction
# - **→ Output**: how strongly does a mode/neuron *drive* the readout W_out?
# - **← Input**: how strongly is a mode/neuron *driven by* the input W_in?
#
# ---
#
# ### Note on complex conjugate pairs and 2×2 Schur blocks
#
# **Eigenbasis (complex pairs).**
# For a real matrix J, complex eigenvalues come in conjugate pairs (λ, λ*).
# The corresponding eigenvectors satisfy v* = conj(v), so the coupling
# magnitudes |W_out v| and |W_out v*| are identical.  The heatmap (Section A1)
# shows both columns of the pair — they carry redundant information.
# The label "(C)" flags complex modes.  When counting distinct modes or
# computing participation ratios, each conjugate pair should be counted once.
#
# **Schur basis (2×2 blocks).**
# The analogous structure in the Schur decomposition is a 2×2 diagonal block
# in T, which encodes a complex-conjugate eigenvalue pair.  The two corresponding
# columns of Q are real, but they are the quadrature components of one
# oscillatory mode (cos- and sin-like).  The heatmap labels "(C-a)" and "(C-b)"
# bracket these pairs.  Connectivity couplings |W_out Q_{:,a}| and
# |W_out Q_{:,b}| are generally *not* identical (unlike the eigenbasis case),
# because the two real columns do not have equal norm.  The participation ratio
# and dominant-mode computation in Section A3/B3 treat each 2×2 block as one
# logical mode.

# %% [markdown]
# ### Section A: Connectivity-based Coupling

# %% P-A1  Eigenmode → output (connectivity)
# Requires: _coup_eig_ranked, _eig_abs_rank, _ef, _nbf, _pplf, _Nf,
#           N_HEATMAP_DIMS, G_FOCUS, TRAINED_COLOR, FIGS_DIR, SAVE_FIGS
_a1_top_k = min(N_HEATMAP_DIMS, _Nf)
coupling_top_a1 = _coup_eig_ranked[:, :_a1_top_k]

# mode labels
_abs_top = np.abs(_ef[_eig_abs_rank[:_a1_top_k]])
_log_top = np.log(np.clip(_abs_top, 1e-12, None))
_tau_top  = -1.0 / np.where(_log_top < -1e-10, _log_top, -1e-10)
_top_eigs = _ef[_eig_abs_rank[:_a1_top_k]]
_is_cmplx = np.abs(_top_eigs.imag) > 1e-8
_mode_labels_a1 = []
for mi in range(_a1_top_k):
    lbl = f"Mode {mi+1}\n$\\tau$={_tau_top[mi]:.0f}"
    lbl += "\n(C)" if _is_cmplx[mi] else f"\n({'+'if _top_eigs[mi].real>=0 else chr(8722)}real)"
    _mode_labels_a1.append(lbl)

fig, ax = plt.subplots(figsize=(max(_a1_top_k * 0.9, 5), max(_nbf * 0.65, 3)))
im = ax.imshow(coupling_top_a1, cmap="YlOrRd", aspect="auto")
ax.set_yticks(range(_nbf))
ax.set_yticklabels([f"Bit {i} (p={_pplf[i]})" for i in range(_nbf)], fontsize=9)
ax.set_xticks(range(_a1_top_k))
ax.set_xticklabels(_mode_labels_a1, fontsize=7)
ax.set_xlabel(f"Eigenmode rank (by $|\\lambda|$, top {_a1_top_k})", fontsize=10)
ax.set_ylabel("Output bit", fontsize=11)
for i in range(_nbf):
    for j in range(_a1_top_k):
        v = coupling_top_a1[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > 0.5 * coupling_top_a1.max() else "black")
plt.colorbar(im, ax=ax, label="$|W_{\\mathrm{out}} V|$", shrink=0.8)
fig.suptitle(f"A1: Eigenmode → Output Coupling (connectivity) — g = {G_FOCUS}",
             fontsize=12, fontweight="bold", y=1.04)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"A1_mode_output_heatmap_g{G_FOCUS}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

fig, axs_a1 = plt.subplots(_nbf, 1, figsize=(12, 2.2 * _nbf), sharex=True)
if _nbf == 1: axs_a1 = [axs_a1]
for bi, ax_e in enumerate(axs_a1):
    ax_e.plot(np.arange(1, _Nf + 1), _coup_eig_ranked[bi],
              linewidth=0.8, color=TRAINED_COLOR, alpha=0.7)
    ax_e.axvspan(1, _a1_top_k + 0.5, alpha=0.08, color="C0",
                 label=f"top {_a1_top_k}" if bi == 0 else None)
    ax_e.set_ylabel(f"Bit {bi}\n(p={_pplf[bi]})", fontsize=9)
    ax_e.grid(True, alpha=0.1)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)
    if bi == 0: ax_e.legend(fontsize=8, loc="upper right")
axs_a1[-1].set_xlabel("Eigenmode rank (by $|\\lambda|$)", fontsize=11)
fig.suptitle(r"A1 profile: $|W_{\mathrm{out}}\,V|$ across all modes"
             + f" — g = {G_FOCUS}", fontsize=12, fontweight="bold")
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"A1_mode_output_profile_g{G_FOCUS}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

print(f"\nA1: Eigenmode → output PR  (g = {G_FOCUS}, connectivity):")
print(f"  {'Bit':<6} {'p_pulse':<10} {'dominant mode':<16} {'tau_eff':<10} {'PR'}")
for bi in range(_nbf):
    c_vec = _coup_eig_ranked[bi]
    pr = _pr_vec(c_vec)
    dom = int(np.argmax(c_vec))
    _log_dom = np.log(max(np.abs(_ef[_eig_abs_rank[dom]]), 1e-12))
    tau_dom = -1.0 / _log_dom if _log_dom < -1e-10 else float("inf")
    print(f"  {bi:<6} {_pplf[bi]:<10.3f} Mode {dom+1:<11}  {tau_dom:<10.1f}  {pr:.2f}")

# %% P-A2  Input → eigenmode (connectivity)
# Requires: _in_eig_ranked, _Nf, _nbf, _pplf, N_HEATMAP_DIMS, G_FOCUS, FIGS_DIR, SAVE_FIGS
if '_in_eig_ranked' not in vars():
    print("P-A2: W_in not available, skipping.")
else:
    _a2_top_k = min(N_HEATMAP_DIMS, _Nf)
    coupling_top_a2 = _in_eig_ranked[:_a2_top_k, :]   # (top_k, n_bits)

    fig, axs_a2 = plt.subplots(_nbf, 1, figsize=(12, 2.2 * _nbf), sharex=True)
    if _nbf == 1: axs_a2 = [axs_a2]
    for bi, ax_i in enumerate(axs_a2):
        ax_i.plot(np.arange(1, _Nf + 1), _in_eig_ranked[:, bi],
                  linewidth=0.8, color="#2a9d8f", alpha=0.8)
        ax_i.axvspan(1, _a2_top_k + 0.5, alpha=0.08, color="C0",
                     label=f"top {_a2_top_k}" if bi == 0 else None)
        ax_i.set_ylabel(f"Bit {bi}\n(p={_pplf[bi]})", fontsize=9)
        ax_i.grid(True, alpha=0.1)
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        if bi == 0: ax_i.legend(fontsize=8, loc="upper right")
    axs_a2[-1].set_xlabel("Eigenmode rank (by $|\\lambda|$)", fontsize=11)
    fig.suptitle(f"A2 profile: $|V^{{-1}} W_{{\\mathrm{{in}}}}|$ — Input → eigenmode — g = {G_FOCUS}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"A2_input_eig_profile_g{G_FOCUS}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(max(_a2_top_k * 0.9, 5), max(_nbf * 0.65, 3)))
    im2 = ax2.imshow(coupling_top_a2.T, cmap="Blues", aspect="auto")
    ax2.set_yticks(range(_nbf))
    ax2.set_yticklabels([f"In bit {i} (p={_pplf[i]})" for i in range(_nbf)], fontsize=9)
    ax2.set_xticks(range(_a2_top_k))
    ax2.set_xticklabels([f"Mode {i+1}" for i in range(_a2_top_k)], fontsize=7)
    ax2.set_xlabel(f"Eigenmode rank (top {_a2_top_k})", fontsize=11)
    ax2.set_ylabel("Input bit", fontsize=11)
    for i in range(_nbf):
        for j in range(_a2_top_k):
            v = coupling_top_a2[j, i]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if v > 0.5 * coupling_top_a2.max() else "black")
    plt.colorbar(im2, ax=ax2, label="$|V^{-1} W_{\\mathrm{in}}|$", shrink=0.8)
    fig2.suptitle(f"A2: Input → eigenmode (connectivity) — g = {G_FOCUS}",
                  fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig2.savefig(os.path.join(FIGS_DIR, f"A2_input_eig_heatmap_g{G_FOCUS}.pdf"),
                     bbox_inches="tight", dpi=150)
    plt.show()

    print(f"\nA2: Input → eigenmode PR  (g = {G_FOCUS}, connectivity):")
    print(f"  {'Bit':<6} {'p_pulse':<10} {'dominant mode':<16} {'PR'}")
    for bi in range(_nbf):
        c_vec = _in_eig_ranked[:, bi]
        dom = int(np.argmax(c_vec))
        print(f"  {bi:<6} {_pplf[bi]:<10.3f} Mode {dom+1:<11}  {_pr_vec(c_vec):.2f}")

# %% P-A3  Schur → output (connectivity)
# Requires: _coup_schur_all, _tauf, _blkf, _Nf, _nbf, _pplf, N_HEATMAP_DIMS, G_FOCUS
_a3_top_k = min(N_HEATMAP_DIMS, _Nf)
coupling_top_a3 = _coup_schur_all[:, :_a3_top_k]   # (n_bits, top_k)

_mode_labels_a3 = [
    f"Schur {mi+1}\n$\\tau$={_tauf[mi]:.0f}\n({'R' if _blkf[mi]=='1x1' else 'C'+_blkf[mi][-1]})"
    for mi in range(_a3_top_k)]

fig, ax = plt.subplots(figsize=(max(_a3_top_k * 0.9, 5), max(_nbf * 0.65, 3)))
im = ax.imshow(coupling_top_a3, cmap="YlOrRd", aspect="auto",
               vmin=0, vmax=_vmax_out_conn)
ax.set_yticks(range(_nbf))
ax.set_yticklabels([f"Bit {i} (p={_pplf[i]})" for i in range(_nbf)], fontsize=9)
ax.set_xticks(range(_a3_top_k))
ax.set_xticklabels(_mode_labels_a3, fontsize=7)
ax.set_xlabel(f"Schur mode rank (by $|\\lambda|$, top {_a3_top_k})", fontsize=9)
ax.set_ylabel("Output bit", fontsize=11)
for i in range(_nbf):
    for j in range(_a3_top_k):
        v = coupling_top_a3[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > 0.5 * _vmax_out_conn else "black")
plt.colorbar(im, ax=ax, label="$|W_{\\mathrm{out}}\\, Q|$", shrink=0.8)
fig.suptitle(f"A3: Schur → Output Coupling (connectivity) — g = {G_FOCUS}",
             fontsize=12, fontweight="bold", y=1.04)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"A3_schur_output_heatmap_g{G_FOCUS}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

fig, axs_a3 = plt.subplots(_nbf, 1, figsize=(12, 2.2 * _nbf), sharex=True)
if _nbf == 1: axs_a3 = [axs_a3]
for bi, ax_s in enumerate(axs_a3):
    ax_s.plot(np.arange(1, _Nf + 1), _coup_schur_all[bi],
              linewidth=0.8, color="#457b9d", alpha=0.8)
    ax_s.axvspan(1, _a3_top_k + 0.5, alpha=0.08, color="C0",
                 label=f"top {_a3_top_k}" if bi == 0 else None)
    ax_s.set_ylabel(f"Bit {bi}\n(p={_pplf[bi]})", fontsize=9)
    ax_s.grid(True, alpha=0.1)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    if bi == 0: ax_s.legend(fontsize=8, loc="upper right")
axs_a3[-1].set_xlabel("Schur mode rank (by $|\\lambda|$)", fontsize=11)
fig.suptitle(r"A3 profile: $|W_{\mathrm{out}}\,Q|$ across all Schur modes"
             + f" — g = {G_FOCUS}", fontsize=12, fontweight="bold")
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"A3_schur_output_profile_g{G_FOCUS}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

print(f"\nA3: Schur → output PR  (g = {G_FOCUS}, connectivity):")
print(f"  {'Bit':<6} {'p_pulse':<10} {'dominant Schur':<18} {'tau_eff':<10} {'block':<10} {'PR'}")
for bi in range(_nbf):
    c_vec = _coup_schur_all[bi]
    dom = int(np.argmax(c_vec))
    print(f"  {bi:<6} {_pplf[bi]:<10.3f} Schur {dom+1:<12}  {_tauf[dom]:<10.1f}  "
          f"{_blkf[dom]:<10}  {_pr_vec(c_vec):.2f}")

# %% P-A4  Input → Schur (connectivity)
# Requires: _cin, _Nf, _nbf, _pplf, _tauf, N_HEATMAP_DIMS, G_FOCUS
if '_cin' not in vars():
    print("P-A4: W_in not available, skipping.")
else:
    _a4_top_k = min(N_HEATMAP_DIMS, _Nf)
    _cin_top  = _cin[:_a4_top_k, :]       # (top_k, n_bits)

    fig, axs_a4 = plt.subplots(_nbf, 1, figsize=(12, 2.0 * _nbf), sharex=True)
    if _nbf == 1: axs_a4 = [axs_a4]
    for bi, ax_si in enumerate(axs_a4):
        ax_si.plot(np.arange(1, _Nf + 1), _cin[:, bi],
                   linewidth=0.8, color="#2a9d8f", alpha=0.85)
        ax_si.axvspan(1, _a4_top_k + 0.5, alpha=0.08, color="C0",
                      label=f"top {_a4_top_k}" if bi == 0 else None)
        ax_si.set_ylabel(f"Bit {bi}\n(p={_pplf[bi]})", fontsize=9)
        ax_si.grid(True, alpha=0.1)
        ax_si.spines["top"].set_visible(False)
        ax_si.spines["right"].set_visible(False)
    axs_a4[0].legend(fontsize=8, loc="upper right")
    axs_a4[-1].set_xlabel(r"Schur mode rank (by $|\lambda|$)", fontsize=11)
    fig.suptitle(r"A4 profile: $\alpha\,|Q^\top W_{\mathrm{in}}|$ — Input → Schur — "
                 f"g = {G_FOCUS}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(max(_a4_top_k * 0.9, 5), max(_nbf * 0.7, 3)))
    im2 = ax2.imshow(_cin_top.T, cmap="Blues", aspect="auto",
                     vmin=0, vmax=_vmax_in_conn)
    ax2.set_yticks(range(_nbf))
    ax2.set_yticklabels([f"In bit {i} (p={_pplf[i]})" for i in range(_nbf)], fontsize=9)
    ax2.set_xticks(range(_a4_top_k))
    ax2.set_xticklabels([f"Schur {i+1}\n$\\tau$={_tauf[i]:.0f}" for i in range(_a4_top_k)],
                         fontsize=7)
    for i in range(_nbf):
        for j in range(_a4_top_k):
            v = _cin_top[j, i]
            ax2.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                     color="white" if v > 0.5 * _vmax_in_conn else "black")
    plt.colorbar(im2, ax=ax2, label=r"$\alpha\,|Q^\top W_{\mathrm{in}}|$", shrink=0.8)
    fig2.suptitle(f"A4: Input → Schur coupling (connectivity) — g = {G_FOCUS}",
                  fontsize=11, fontweight="bold", y=1.06)
    plt.tight_layout()
    plt.show()

    print(f"\nA4: Input → Schur PR  (g = {G_FOCUS}, connectivity):")
    print(f"  {'Bit':<6} {'p_pulse':<10} {'dominant Schur':<18} {'tau_eff':<10} {'PR'}")
    for bi in range(_nbf):
        c_vec = _cin[:, bi]
        dom = int(np.argmax(c_vec))
        print(f"  {bi:<6} {_pplf[bi]:<10.3f} Schur {dom+1:<12}  {_tauf[dom]:<10.1f}  "
              f"{_pr_vec(c_vec):.2f}")

# %% P-A5  Neuron → output (connectivity)
# Requires: _cout_s, _Nf, _nbf, _pplf, N_HEATMAP_DIMS, G_FOCUS
_a5_top_k = min(N_HEATMAP_DIMS, _Nf)

fig, axs_a5 = plt.subplots(_nbf, 1, figsize=(12, 2.0 * _nbf), sharex=True)
if _nbf == 1: axs_a5 = [axs_a5]
for bi, ax_ne in enumerate(axs_a5):
    ax_ne.plot(np.arange(1, _Nf + 1), _cout_s[bi],
               linewidth=0.8, color="#e76f51", alpha=0.85)
    ax_ne.axvspan(1, _a5_top_k + 0.5, alpha=0.08, color="C0",
                  label=f"top {_a5_top_k}" if bi == 0 else None)
    ax_ne.set_ylabel(f"Bit {bi}\n(p={_pplf[bi]})", fontsize=9)
    ax_ne.grid(True, alpha=0.1)
    ax_ne.spines["top"].set_visible(False)
    ax_ne.spines["right"].set_visible(False)
axs_a5[0].legend(fontsize=8, loc="upper right")
axs_a5[-1].set_xlabel(r"Neuron rank (by $\|W_{\mathrm{out}}[:,j]\|_2$)", fontsize=11)
fig.suptitle(r"A5 profile: $|W_{\mathrm{out}}|$ — Neuron → output — "
             f"g = {G_FOCUS}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

fig2, ax2_ne = plt.subplots(figsize=(max(_a5_top_k * 0.9, 5), max(_nbf * 0.65, 3)))
im_ne = ax2_ne.imshow(_cout_s[:, :_a5_top_k], cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=_vmax_out_conn)
ax2_ne.set_yticks(range(_nbf))
ax2_ne.set_yticklabels([f"Bit {i} (p={_pplf[i]})" for i in range(_nbf)], fontsize=9)
ax2_ne.set_xticks(range(_a5_top_k))
ax2_ne.set_xticklabels([f"N{i+1}" for i in range(_a5_top_k)], fontsize=7)
ax2_ne.set_xlabel(f"Neuron rank (top {_a5_top_k})", fontsize=11)
ax2_ne.set_ylabel("Output bit", fontsize=11)
for bi in range(_nbf):
    for ni in range(_a5_top_k):
        v = _cout_s[bi, ni]
        ax2_ne.text(ni, bi, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 * _vmax_out_conn else "black")
plt.colorbar(im_ne, ax=ax2_ne, label=r"$|W_{\mathrm{out},ij}|$", shrink=0.7)
fig2.suptitle(f"A5: Neuron → output heatmap (connectivity) — g = {G_FOCUS}",
              fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print(f"\nA5: Neuron → output PR  (g = {G_FOCUS}, connectivity):")
print(f"  {'Bit':<6} {'p_pulse':<10} {'dominant neuron rank':<22} {'PR'}")
for bi in range(_nbf):
    c_vec = _cout_s[bi]
    dom = int(np.argmax(c_vec))
    print(f"  {bi:<6} {_pplf[bi]:<10.3f} rank {dom+1:<17}  {_pr_vec(c_vec):.2f}")

# %% P-A6  Input → neuron (connectivity)
# Requires: _a6_win_s, _Nf, _nbf, _pplf, N_HEATMAP_DIMS, G_FOCUS
if '_a6_win_s' not in vars():
    print("P-A6: W_in not available, skipping.")
else:
    _a6_top_k = min(N_HEATMAP_DIMS, _Nf)

    fig, axs_a6 = plt.subplots(_nbf, 1, figsize=(12, 2.0 * _nbf), sharex=True)
    if _nbf == 1: axs_a6 = [axs_a6]
    for bi, ax_a6 in enumerate(axs_a6):
        ax_a6.plot(np.arange(1, _Nf + 1), _a6_win_s[:, bi],
                   linewidth=0.8, color="#2a9d8f", alpha=0.85)
        ax_a6.axvspan(1, _a6_top_k + 0.5, alpha=0.08, color="C0",
                      label=f"top {_a6_top_k}" if bi == 0 else None)
        ax_a6.set_ylabel(f"Bit {bi}\n(p={_pplf[bi]})", fontsize=9)
        ax_a6.grid(True, alpha=0.1)
        ax_a6.spines["top"].set_visible(False)
        ax_a6.spines["right"].set_visible(False)
    axs_a6[0].legend(fontsize=8, loc="upper right")
    axs_a6[-1].set_xlabel(r"Neuron rank (by $\|W_{\mathrm{in}}[j,:]\|_2$)", fontsize=11)
    fig.suptitle(r"A6 profile: $|W_{\mathrm{in}}|$ — Input → neuron — "
                 f"g = {G_FOCUS}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    fig2, ax2_a6 = plt.subplots(figsize=(max(_a6_top_k * 0.9, 5), max(_nbf * 0.65, 3)))
    im_a6 = ax2_a6.imshow(_a6_win_s[:_a6_top_k, :].T, cmap="Blues", aspect="auto",
                           vmin=0, vmax=_vmax_in_conn)
    ax2_a6.set_yticks(range(_nbf))
    ax2_a6.set_yticklabels([f"In bit {i} (p={_pplf[i]})" for i in range(_nbf)], fontsize=9)
    ax2_a6.set_xticks(range(_a6_top_k))
    ax2_a6.set_xticklabels([f"N{i+1}" for i in range(_a6_top_k)], fontsize=7)
    ax2_a6.set_xlabel(f"Neuron rank (top {_a6_top_k})", fontsize=11)
    ax2_a6.set_ylabel("Input bit", fontsize=11)
    for bi in range(_nbf):
        for ni in range(_a6_top_k):
            v = _a6_win_s[ni, bi]
            ax2_a6.text(ni, bi, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 * _vmax_in_conn else "black")
    plt.colorbar(im_a6, ax=ax2_a6, label=r"$|W_{\mathrm{in},ji}|$", shrink=0.8)
    fig2.suptitle(f"A6: Input → neuron heatmap (connectivity) — g = {G_FOCUS}",
                  fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

    print(f"\nA6: Input → neuron PR  (g = {G_FOCUS}, connectivity):")
    print(f"  {'Bit':<6} {'p_pulse':<10} {'dominant neuron rank':<22} {'PR'}")
    for bi in range(_nbf):
        c_vec = _a6_win_s[:, bi]
        dom = int(np.argmax(c_vec))
        print(f"  {bi:<6} {_pplf[bi]:<10.3f} rank {dom+1:<17}  {_pr_vec(c_vec):.2f}")


# %% [markdown]
# ### Section B: Correlation-based Coupling

# %% P-B1  Eigenmode → output (correlation)
# Requires: _r_eig_tgt, _ef, _nbf, _pplf, G_FOCUS, FIGS_DIR, N_HEATMAP_DIMS
if '_r_eig_tgt' not in vars():
    print("P-B1: run Part 3C first.")
else:
    _b1_abs_rank = np.argsort(np.abs(_ef))[::-1]
    _b1_row_lbl  = (lambda i: f"Eig {i+1}  (|λ|={np.abs(_ef[_b1_abs_rank[i]]):.3f})")
    print(f"\nB1: Eigenmode → output correlation  (g = {G_FOCUS}):")
    _dom_b1 = _corr_heatmap_profile(
        _r_eig_tgt,
        title_prefix=f"corr(Re(V⁻¹h)_j, target_i) — g={G_FOCUS}",
        row_labels=_b1_row_lbl,
        col_labels=[f"Bit {i}\n(p={_pplf[i]})" for i in range(_nbf)],
        xlabel_row="Output bit",
        ylabel_col="Eigenmode rank (by |λ|)",
        save_path=os.path.join(FIGS_DIR, f"B1_eig_output_corr_g{G_FOCUS}.pdf"),
    )

# %% P-B2  Input → eigenmode (correlation)
# Requires: _r_eig_inp, _nbf, _pplf, G_FOCUS, FIGS_DIR, N_HEATMAP_DIMS
if '_r_eig_inp' not in vars():
    print("P-B2: run Part 3C first.")
else:
    print(f"\nB2: Input → eigenmode correlation  (g = {G_FOCUS}):")
    _dom_b2 = _corr_heatmap_profile(
        _r_eig_inp,
        title_prefix=f"corr(Re(V⁻¹h)_j, input_i) — g={G_FOCUS}",
        row_labels=(lambda i: f"Eig {i+1}"),
        col_labels=[f"In bit {i}\n(p={_pplf[i]})" for i in range(_nbf)],
        xlabel_row="Input bit",
        ylabel_col="Eigenmode rank (by |λ|)",
        save_path=os.path.join(FIGS_DIR, f"B2_eig_input_corr_g{G_FOCUS}.pdf"),
    )

# %% P-B3a  Schur → output correlation heatmap
# Requires: _r_schur_tgt, _nbf, _pplf, _tauf, _blkf, G_FOCUS, FIGS_DIR, N_HEATMAP_DIMS
if '_r_schur_tgt' not in vars():
    print("P-B3a: run Part 3C first.")
else:
    print(f"\nB3: Schur → output correlation  (g = {G_FOCUS}):")
    _dom_b3_plot = _corr_heatmap_profile(
        _r_schur_tgt,
        title_prefix=f"corr(Q[:,j]ᵀh, target_i) — g={G_FOCUS}",
        row_labels=(lambda i: f"Schur {i+1}  (τ={_tauf[i]:.1f},  "
                              f"{'R' if _blkf[i]=='1x1' else 'C'})"),
        col_labels=[f"Bit {i}\n(p={_pplf[i]})" for i in range(_nbf)],
        xlabel_row="Output bit",
        ylabel_col="Schur mode rank (by |λ|)",
        save_path=os.path.join(FIGS_DIR, f"B3_schur_output_corr_g{G_FOCUS}.pdf"),
    )

# %% P-B3b  Correlation-coloured scree
# Requires: _r_schur_tgt, _tauf, _Nf, _nbf, _pplf,
#           _dom_b3, _bulk_b3, _max_r_b3, _best_bit_b3, _bit_cls_b3, _ranks_b3
#           G_FOCUS, FIGS_DIR, SAVE_FIGS
# ── SCREE FIX: bulk = solid blue (no colormap/colorbar);
#               hold times drawn as thick segments at their actual τ value
#               in the middle of the x-axis for direct visual comparison ─────
if '_r_schur_tgt' not in vars():
    print("P-B3b: run Part 3C first.")
else:
    from matplotlib.lines import Line2D as _Line2D5

    fig2, ax2 = plt.subplots(figsize=(14, 7))

    # ── Bulk modes: solid blue, slightly larger points ────────────────────────
    ax2.scatter(_ranks_b3[_bulk_b3], _tauf[_bulk_b3],
                color="steelblue", s=22, alpha=0.55, edgecolors="none",
                rasterized=True, zorder=2)

    # ── Highlighted dominant modes (one per bit) ──────────────────────────────
    for _bi in range(_nbf):
        _dm = _dom_b3[_bi]
        ax2.scatter(_ranks_b3[_dm], _tauf[_dm], s=200, color=_bit_cls_b3[_bi],
                    edgecolors="white", linewidths=1.2, zorder=6)
        ax2.annotate(
            f"Bit {_bi}  (p={_pplf[_bi]})\n"
            f"τ={_tauf[_dm]:.1f}  r={_r_schur_tgt[_dm,_bi]:+.2f}",
            xy=(_ranks_b3[_dm], _tauf[_dm]),
            xytext=(20, 0), textcoords="offset points",
            fontsize=9, color=_bit_cls_b3[_bi], fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-", color=_bit_cls_b3[_bi], lw=0.6, alpha=0.5))

    ax2.set_xlabel("Schur mode rank (sorted by $|\\lambda|$, largest first)", fontsize=13)
    ax2.set_ylabel(r"$\tau_{\mathrm{eff}} = -1\,/\,\ln|\lambda|$  (steps)", fontsize=13)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlim(0, _Nf + 10)
    ax2.grid(True, alpha=0.1, which="both")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Dominant-modes legend (upper right) ───────────────────────────────────
    _leg5 = [_Line2D5([0], [0], marker="o", color="w",
                       markerfacecolor=_bit_cls_b3[i], markeredgecolor="white",
                       markersize=10,
                       label=f"Bit {i} dom-by-corr  "
                             f"(r={_r_schur_tgt[_dom_b3[i],i]:+.2f},  "
                             f"τ={_tauf[_dom_b3[i]]:.1f})")
             for i in range(_nbf)]
    #ax2.legend(handles=_leg5, fontsize=9, loc="upper right", framealpha=0.85, ncol=2)

    # ── Hold-time reference segments: drawn at actual τ_hold in the middle of
    #    the x-axis so they can be visually compared to the scattered modes ────
    _xlim = ax2.get_xlim()
    _xlog0, _xlog1 = np.log10(max(_xlim[0], 1)), np.log10(max(_xlim[1], 2))
    _seg_x0 = 10 ** (_xlog0 + 0.48 * (_xlog1 - _xlog0))
    _seg_x1 = 10 ** (_xlog0 + 0.62 * (_xlog1 - _xlog0))
    for _i in range(_nbf):
        _h = 1.0 / _pplf[_i]
        ax2.plot([_seg_x0, _seg_x1], [_h, _h],
                 color=_bit_cls_b3[_i], linewidth=3.5, alpha=0.85,
                 solid_capstyle="round", zorder=3)
        ax2.text(_seg_x1 * 1.08, _h,
                 f"hold≈{_h:.0f}  (Bit {_i}, p={_pplf[_i]})",
                 color=_bit_cls_b3[_i], fontsize=8.5, va="center")

    fig2.suptitle(
        f"B3 scree: Schur correlation scree — g = {G_FOCUS}\n"
        "Bulk modes = solid blue;  coloured rings = dominant-by-correlation",
        fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig2.savefig(os.path.join(FIGS_DIR, f"B3_schur_corr_scree_g{G_FOCUS}.pdf"),
                     bbox_inches="tight", dpi=150)
    plt.show()

# %% P-B3c  Localization portrait
# Requires: _pr_mode_b3, _pr_neuron_b3, _dom_b3, _tauf, _pplf, _nbf, _Nf, _bit_cls_b3
if '_pr_mode_b3' not in vars():
    print("P-B3c: run Part 3D first.")
else:
    print(f"\nB3: Localization portrait  (g = {G_FOCUS}):")
    print(f"  {'Bit':<6} {'PR_mode':<12} {'PR_neuron':<14} {'dom Schur':<12} {'tau_eff'}")
    for _bi in range(_nbf):
        print(f"  {_bi:<6} {_pr_mode_b3[_bi]:<12.2f} {_pr_neuron_b3[_bi]:<14.2f} "
              f"Schur {_dom_b3[_bi]+1:<7} {_tauf[_dom_b3[_bi]]:.1f}")

    fig3, ax3 = plt.subplots(figsize=(7, 6))
    for _bi in range(_nbf):
        _c3 = _bit_cls_b3[_bi]
        ax3.scatter(_pr_mode_b3[_bi], _pr_neuron_b3[_bi], s=200, color=_c3,
                    edgecolors="white", linewidths=1.2, zorder=4)
        ax3.annotate(
            f"Bit {_bi}  (p={_pplf[_bi]})\ndom Schur {_dom_b3[_bi]+1}  τ={_tauf[_dom_b3[_bi]]:.1f}",
            xy=(_pr_mode_b3[_bi], _pr_neuron_b3[_bi]),
            xytext=(10, 6), textcoords="offset points",
            fontsize=8.5, color=_c3, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=_c3, lw=0.5, alpha=0.4))
    ax3.axvline(1, color="#bbb", linewidth=0.8, linestyle="--", alpha=0.6, label="PR = 1")
    ax3.axhline(1, color="#bbb", linewidth=0.8, linestyle="--", alpha=0.6)
    ax3.axvline(_Nf, color="#e76f51", linewidth=0.7, linestyle=":", alpha=0.5,
                label=f"PR = N={_Nf}")
    ax3.axhline(_Nf, color="#e76f51", linewidth=0.7, linestyle=":", alpha=0.5)
    ax3.set_xlabel(
        r"$\mathrm{PR}_\mathrm{mode}$ " + "\n← sparse                 distributed →",
        fontsize=10)
    ax3.set_ylabel(
        r"$\mathrm{PR}_\mathrm{neuron}$ " + "\n← localized              extended →",
        fontsize=10)
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_xlim(0.8, _Nf * 1.5); ax3.set_ylim(0.8, _Nf * 1.5)
    ax3.grid(True, alpha=0.12, which="both")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    ax3.legend(fontsize=9, loc="lower right")
    fig3.suptitle(f"B3 localization portrait — g = {G_FOCUS}",
                  fontsize=11, fontweight="bold", y=1.03)
    plt.tight_layout()
    if SAVE_FIGS:
        fig3.savefig(os.path.join(FIGS_DIR, f"B3_localization_g{G_FOCUS}.pdf"),
                     bbox_inches="tight", dpi=150)
    plt.show()

# %% P-B4  Input → Schur (correlation)
# Requires: _r_schur_inp, _nbf, _pplf, _tauf, G_FOCUS, FIGS_DIR, N_HEATMAP_DIMS
if '_r_schur_inp' not in vars():
    print("P-B4: run Part 3C first.")
else:
    print(f"\nB4: Input → Schur correlation  (g = {G_FOCUS}):")
    _dom_b4 = _corr_heatmap_profile(
        _r_schur_inp,
        title_prefix=f"corr(Q[:,j]ᵀh, input_i) — g={G_FOCUS}",
        row_labels=(lambda i: f"Schur {i+1}  (τ={_tauf[i]:.1f})"),
        col_labels=[f"In bit {i}\n(p={_pplf[i]})" for i in range(_nbf)],
        xlabel_row="Input bit",
        ylabel_col="Schur mode rank (by |λ|)",
        save_path=os.path.join(FIGS_DIR, f"B4_schur_input_corr_g{G_FOCUS}.pdf"),
    )

# %% P-B5  Neuron → output (correlation)
# Requires: _r_neu_tgt_ranked, _b5_ro, _nbf, _pplf, G_FOCUS, FIGS_DIR, N_HEATMAP_DIMS
if '_r_neu_tgt_ranked' not in vars():
    print("P-B5: run Part 3D first.")
else:
    print(f"\nB5: Neuron → output correlation  (g = {G_FOCUS}):")
    _dom_b5 = _corr_heatmap_profile(
        _r_neu_tgt_ranked,
        title_prefix=f"corr(h_k, target_i) — g={G_FOCUS}",
        row_labels=(lambda i: f"Neu rank {i+1}"),
        col_labels=[f"Bit {i}\n(p={_pplf[i]})" for i in range(_nbf)],
        xlabel_row="Output bit",
        ylabel_col="Neuron rank (by max |r|)",
        save_path=os.path.join(FIGS_DIR, f"B5_neuron_output_corr_g{G_FOCUS}.pdf"),
    )

# %% P-B6  Input → neuron (correlation)
# Requires: _r_neu_inp_ranked, _b6_ro, _nbf, _pplf, G_FOCUS, FIGS_DIR, N_HEATMAP_DIMS
if '_r_neu_inp_ranked' not in vars():
    print("P-B6: run Part 3D first.")
else:
    print(f"\nB6: Input → neuron correlation  (g = {G_FOCUS}):")
    _dom_b6 = _corr_heatmap_profile(
        _r_neu_inp_ranked,
        title_prefix=f"corr(h_k, input_i) — g={G_FOCUS}",
        row_labels=(lambda i: f"Neu rank {i+1}"),
        col_labels=[f"In bit {i}\n(p={_pplf[i]})" for i in range(_nbf)],
        xlabel_row="Input bit",
        ylabel_col="Neuron rank (by max |r|)",
        save_path=os.path.join(FIGS_DIR, f"B6_neuron_input_corr_g{G_FOCUS}.pdf"),
    )


# %% [markdown]
# ### Section C: Ablation-based Coupling

# %% P-C1  Schur mode ablation sweep
# Requires: _c1_delta, _c1_tot, _C1_K_ABL, _tauf, _nbf, _pplf, G_FOCUS, FIGS_DIR, SAVE_FIGS
if '_c1_delta' not in vars():
    print("P-C1: run Part 3E first.")
else:
    fig, (ax_c1a, ax_c1b) = plt.subplots(
        1, 2, figsize=(14, max(4.5, 0.3 * _C1_K_ABL + 2)),
        gridspec_kw=dict(width_ratios=[1.4, 1]))

    im_c1 = ax_c1a.imshow(_c1_delta, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
    ax_c1a.set_xticks(range(_nbf))
    ax_c1a.set_xticklabels([f"Bit {i}\n(p={_pplf[i]})" for i in range(_nbf)], fontsize=8)
    ax_c1a.set_yticks(range(_C1_K_ABL))
    ax_c1a.set_yticklabels(
        [f"Schur {j+1}  tau={_tauf[j]:.1f}" for j in range(_C1_K_ABL)], fontsize=7)
    ax_c1a.set_xlabel("Output bit", fontsize=11)
    ax_c1a.set_ylabel("Schur mode silenced", fontsize=11)
    ax_c1a.set_title(f"C1: Δ Accuracy  (g = {G_FOCUS})", fontsize=10)
    for jj in range(_C1_K_ABL):
        for bi in range(_nbf):
            v = _c1_delta[jj, bi]
            ax_c1a.text(bi, jj, f"{v:+.3f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.25 else "black")
    plt.colorbar(im_c1, ax=ax_c1a, label="Δ Accuracy", shrink=0.6)

    ax_c1b.barh(range(_C1_K_ABL), _c1_tot,
                color=["#c0392b" if v < 0 else "#2a9d8f" for v in _c1_tot],
                edgecolor="none", alpha=0.85)
    ax_c1b.invert_yaxis()
    ax_c1b.set_yticks(range(_C1_K_ABL))
    ax_c1b.set_yticklabels([f"Schur {j+1}" for j in range(_C1_K_ABL)], fontsize=7)
    ax_c1b.set_xlabel("Total Δ Accuracy (sum over bits)", fontsize=10)
    ax_c1b.axvline(0, color="black", linewidth=0.7)
    ax_c1b.grid(True, alpha=0.2, axis="x")
    ax_c1b.spines["top"].set_visible(False)
    ax_c1b.spines["right"].set_visible(False)
    ax_c1b.set_title("Total drop per mode\n(negative = mode matters)", fontsize=9)

    fig.suptitle(f"C1: Schur mode ablation sweep  (g = {G_FOCUS},  top {_C1_K_ABL} modes)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"C1_schur_ablation_sweep_g{G_FOCUS}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% P-C3  Neuron ablation sweep
# Requires: _c3_delta, _c3_tot, _C3_K_ABL, _c3_coup_norm, _c3_top_neurons, _nbf, _pplf
if '_c3_delta' not in vars():
    print("P-C3: run Part 3E first.")
else:
    fig, (ax_c3a, ax_c3b) = plt.subplots(
        1, 2, figsize=(14, max(4.5, 0.3 * _C3_K_ABL + 2)),
        gridspec_kw=dict(width_ratios=[1.4, 1]))

    im_c3 = ax_c3a.imshow(_c3_delta, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
    ax_c3a.set_xticks(range(_nbf))
    ax_c3a.set_xticklabels([f"Bit {i}\n(p={_pplf[i]})" for i in range(_nbf)], fontsize=8)
    ax_c3a.set_yticks(range(_C3_K_ABL))
    ax_c3a.set_yticklabels(
        [f"Neu rank {r+1}  (||w||={_c3_coup_norm[_c3_top_neurons[r]]:.2f})"
         for r in range(_C3_K_ABL)], fontsize=7)
    ax_c3a.set_xlabel("Output bit", fontsize=11)
    ax_c3a.set_ylabel("Neuron rank (by output coupling norm)", fontsize=11)
    ax_c3a.set_title(f"C3: Δ Accuracy  (g = {G_FOCUS})", fontsize=10)
    for ri in range(_C3_K_ABL):
        for bi in range(_nbf):
            v = _c3_delta[ri, bi]
            ax_c3a.text(bi, ri, f"{v:+.3f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.25 else "black")
    plt.colorbar(im_c3, ax=ax_c3a, label="Δ Accuracy", shrink=0.6)

    ax_c3b.barh(range(_C3_K_ABL), _c3_tot,
                color=["#c0392b" if v < 0 else "#2a9d8f" for v in _c3_tot],
                edgecolor="none", alpha=0.85)
    ax_c3b.invert_yaxis()
    ax_c3b.set_yticks(range(_C3_K_ABL))
    ax_c3b.set_yticklabels([f"Neu rank {r+1}" for r in range(_C3_K_ABL)], fontsize=7)
    ax_c3b.set_xlabel("Total Δ Accuracy (sum over bits)", fontsize=10)
    ax_c3b.axvline(0, color="black", linewidth=0.7)
    ax_c3b.grid(True, alpha=0.2, axis="x")
    ax_c3b.spines["top"].set_visible(False)
    ax_c3b.spines["right"].set_visible(False)
    ax_c3b.set_title("Total drop per neuron\n(negative = neuron matters)", fontsize=9)

    fig.suptitle(
        f"C3: Neuron ablation sweep  (g = {G_FOCUS},  top {_C3_K_ABL} neurons)",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"C3_neuron_ablation_sweep_g{G_FOCUS}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% P-C4  Comparison: Schur mode vs neuron total ablation drop (sorted)
# Requires: _c1_tot, _c3_tot, _C1_K_ABL, _C3_K_ABL,
#           _tauf, _c3_coup_norm, _c3_top_neurons, G_FOCUS, FIGS_DIR, SAVE_FIGS
if '_c1_tot' not in vars() or '_c3_tot' not in vars():
    print("P-C4: run Part 3E first.")
else:
    # ── Sort each by largest total drop (most negative first) ─────────────────
    _c4_schur_order = np.argsort(_c1_tot)          # ascending = most negative first
    _c4_schur_tot   = _c1_tot[_c4_schur_order]
    _c4_schur_labels = [f"Schur {j+1}  (τ={_tauf[j]:.0f})"
                        for j in _c4_schur_order]

    _c4_neu_order  = np.argsort(_c3_tot)
    _c4_neu_tot    = _c3_tot[_c4_neu_order]
    _c4_neu_labels = [f"Neu rank {_c4_neu_order[r]+1}"
                      f"  (||w||={_c3_coup_norm[_c3_top_neurons[_c4_neu_order[r]]]:.2f})"
                      for r in range(_C3_K_ABL)]

    _n_show = max(_C1_K_ABL, _C3_K_ABL)
    fig, (ax_sm, ax_nm) = plt.subplots(
        1, 2, figsize=(14, max(5, 0.28 * _n_show + 2)),
        sharey=False, sharex=True)

    def _ablation_barh(ax, tots, labels, title):
        colors = ["#c0392b" if v < 0 else "#2a9d8f" for v in tots]
        ax.barh(range(len(tots)), tots, color=colors, edgecolor="none", alpha=0.85)
        ax.invert_yaxis()
        ax.set_yticks(range(len(tots)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Total Δ Accuracy (sum over bits)", fontsize=10)
        ax.grid(True, alpha=0.2, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, fontsize=10)

    _ablation_barh(ax_sm, _c4_schur_tot, _c4_schur_labels,
                   f"Schur modes  (top {_C1_K_ABL}, sorted by drop)")
    _ablation_barh(ax_nm, _c4_neu_tot,   _c4_neu_labels,
                   f"Neurons  (top {_C3_K_ABL}, sorted by drop)")

    fig.suptitle(
        f"C4: Ablation comparison — Schur modes vs neurons  (g = {G_FOCUS})\n"
        "Sorted by total accuracy drop; red = hurts performance",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, f"C4_ablation_comparison_g{G_FOCUS}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()


# %% [markdown]
# ---
# ## Spectrum & Scree (after coupling)
# These plots place the spectral picture *after* the coupling analysis, so
# that dominant modes already have meaning from Sections A–C before being
# highlighted in the spectrum.

# %% P04  Non-normality of J — Schur factor T
# Requires: _Tf, _nn_idx, _T_offdiag_frac  (computed in cell 3A-ii)
# Shows the Schur factor T on a log-colour scale; bright off-diagonal entries
# indicate strong non-normal (inter-mode) coupling.
_fig_nn, _ax_nn = plt.subplots(1, 1, figsize=(5, 4))
#_T_vis = np.log10(np.abs(_Tf) + 1e-12)
_T_vis = np.abs(_Tf)
_im_nn = _ax_nn.imshow(_T_vis[:20, :20], cmap='inferno', aspect='equal',
                        interpolation='nearest')
_cb_nn = _fig_nn.colorbar(_im_nn, ax=_ax_nn, fraction=0.046, pad=0.04)
_cb_nn.set_label(r'$\log_{10}|T_{ij}|$', fontsize=9)
_ax_nn.set_xlabel('Schur mode index  (sorted by $|\\lambda|$ $\\downarrow$)', fontsize=9)
_ax_nn.set_ylabel('Schur mode index', fontsize=9)
_ax_nn.set_title(
    f'Schur factor T   (g = {G_FOCUS})\n'
    f'off-diag energy = {_T_offdiag_frac:.3f}   |   '
    f'non-normality idx = {_nn_idx:.4f}',
    fontsize=10)
_fig_nn.tight_layout()
if SAVE_FIGS:
    _fig_nn.savefig(os.path.join(FIGS_DIR, f"P04_nonnorm_T_g{G_FOCUS}.pdf"),
                    bbox_inches="tight", dpi=150)
plt.show()

# %% P05  Eigenvalue spectrum (zoomed) — G_FOCUS only
# Requires: eig_data, G_FOCUS, _pca_seed_path, _pca_cfg, _dom_r, SPECTRUM_HIGHLIGHT,
#           ZOOM_XLIM, ZOOM_YLIM, theta, UNTRAINED_COLOR, TRAINED_COLOR
ZOOM_XLIM = (0.6, 1.05)
ZOOM_YLIM = (-0.225, 0.225)

_3b_alpha   = 1.0 - np.exp(-_pca_cfg["dt"] /
                             _pca_cfg["time_constants_config"]["values"][0])
_3b_n_bits  = _pca_cfg["n_bits"]
_unt_ckpt_3b = os.path.join(_pca_seed_path, "checkpoints", "untrained.ckpt")
_trn_ckpt_3b = _pca_ckpts[0]

fig, axes_3b = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=False)

for col_idx, (label, ckpt_path) in enumerate(
        [("Untrained", _unt_ckpt_3b), ("Trained", _trn_ckpt_3b)]):
    ax = axes_3b[0, col_idx]
    if not os.path.exists(ckpt_path):
        ax.text(0.5, 0.5, "no ckpt", transform=ax.transAxes, ha="center")
        continue

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _3b_Wrec = _3b_Wout = None
    for key, val in ckpt["state_dict"].items():
        if "W_rec.weight" in key: _3b_Wrec = val.numpy()
        if "W_out.weight" in key: _3b_Wout = val.numpy()
    if _3b_Wrec is None:
        continue

    N_3b  = _3b_Wrec.shape[0]
    J_3b  = (1.0 - _3b_alpha) * np.eye(N_3b) + _3b_alpha * G_FOCUS * _3b_Wrec
    eigs_3b     = np.linalg.eig(J_3b)[0]
    abs_eigs_3b = np.abs(eigs_3b)

    if col_idx == 1:
        if SPECTRUM_HIGHLIGHT == 'correlation' and '_dom_r' in vars():
            _3b_top_idx = np.array(sorted(_dom_r.values()))
            print("P05: highlighting modes by correlation (_dom_r).")
        else:
            if SPECTRUM_HIGHLIGHT == 'correlation':
                print("P05: _dom_r not available — falling back to connectivity.")
            if _3b_Wout is not None:
                _3b_coup    = np.abs(_3b_Wout @ eig_data[G_FOCUS]["eigvecs"])
                _3b_top_idx = np.unique(
                    [int(np.argmax(_3b_coup[bi])) for bi in range(_3b_n_bits)])
            else:
                _3b_top_idx = np.argsort(abs_eigs_3b)[-_3b_n_bits:]

    ax.plot(np.cos(theta), np.sin(theta), color="#c0392b",
            linewidth=1.5, alpha=0.85, linestyle="-", zorder=1, label="$|\\lambda|=1$")
    ax.axhline(0, color="#bbb", linewidth=0.3, zorder=0)
    ax.axvline(0, color="#bbb", linewidth=0.3, zorder=0)

    pt_color = UNTRAINED_COLOR if col_idx == 0 else TRAINED_COLOR
    if col_idx == 0:
        ax.scatter(eigs_3b.real, eigs_3b.imag, s=22, alpha=0.65,
                   c=pt_color, edgecolors="none", zorder=3)
    else:
        _3b_rest = np.ones(N_3b, dtype=bool)
        _3b_rest[_3b_top_idx] = False
        ax.scatter(eigs_3b.real[_3b_rest], eigs_3b.imag[_3b_rest],
                   s=22, alpha=0.65, c=pt_color, edgecolors="none", zorder=3)
        ax.scatter(eigs_3b.real[_3b_top_idx], eigs_3b.imag[_3b_top_idx],
                   s=45, alpha=0.95, c=TRAINED_COLOR,
                   edgecolors="black", linewidths=0.6, zorder=4,
                   label=f"Top {len(_3b_top_idx)} ({SPECTRUM_HIGHLIGHT})")

    ax.set_xlim(*ZOOM_XLIM)
    ax.set_ylim(*ZOOM_YLIM)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylabel(f"g = {G_FOCUS}\nIm($\\lambda$)", fontsize=11)
    ax.set_xlabel("Re($\\lambda$)", fontsize=11)

axes_3b[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.7)
axes_3b[0, 1].legend(fontsize=8, loc="upper left", framealpha=0.7)
fig.suptitle(f"Jacobian Eigenvalue Spectrum (zoom) — g = {G_FOCUS}  "
             f"(highlight: {SPECTRUM_HIGHLIGHT})",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"P05_spectrum_g{G_FOCUS}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% P06  Effective timescale scree — G_FOCUS detailed (correlation-highlighted)
# Requires: _dom_r, eig_data, G_FOCUS, _Nf, bit_colors, _holds, _pplf, _nbf, FIGS_DIR
from matplotlib.lines import Line2D as _Line2D4

_s4     = eig_data[G_FOCUS]
_eigs4  = _s4["eigs"]
_V4     = _s4["eigvecs"]
_Wo4    = _s4["W_out"]
_nb4    = _s4["n_bits"]
_ppl4   = _s4["p_pulse"] if isinstance(_s4["p_pulse"], list) else [_s4["p_pulse"]] * _nb4
_holds4 = [1.0 / p for p in _ppl4]
_N4     = len(_eigs4)

_abs_rank4  = np.argsort(np.abs(_eigs4))[::-1]
_tau4       = -1.0 / np.where(
    np.log(np.clip(np.abs(_eigs4[_abs_rank4]), 1e-12, None)) < -1e-10,
    np.log(np.clip(np.abs(_eigs4[_abs_rank4]), 1e-12, None)),
    -1e-10)
_ranks4 = np.arange(1, _N4 + 1)

# Use correlation-based dominant indices if available, else fall back to connectivity
if '_dom_r' in vars() and _dom_r:
    _dom4      = {bi: int(idx) for bi, idx in _dom_r.items()}
    _dom4_mode = 'correlation'
else:
    _coup4 = np.abs(_Wo4 @ _V4)[:, _abs_rank4] if _Wo4 is not None else None
    _dom4  = {}
    if _coup4 is not None:
        for _bi in range(_nb4):
            _dom4[_bi] = int(np.argmax(_coup4[_bi]))
    _dom4_mode = 'connectivity'
print(f"P06: dominant modes determined by {_dom4_mode}.")

# Separate dominant indices so we can compute 2nd-dominant for last bit
_dom4_idx_set = set(_dom4.values())
_tau4_ranked  = _tau4   # already in rank order

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 1]})

# -- left: scree --
_bulk_msk4 = np.ones(_N4, dtype=bool)
for _ri in _dom4_idx_set:
    _bulk_msk4[_ri] = False

ax.scatter(_ranks4[_bulk_msk4], _tau4[_bulk_msk4],
           color="steelblue", s=22, alpha=0.55,
           edgecolors="none", rasterized=True, zorder=2,
           label="Bulk modes")

for _bi, _ri in _dom4.items():
    _c4 = bit_colors[_bi]
    ax.scatter(_ranks4[_ri], _tau4[_ri], s=180, color=_c4,
               edgecolors="white", linewidths=1.0, zorder=6)
    ax.annotate(f"Bit {_bi}  (p={_ppl4[_bi]})\n"
                f"$\\tau$={_tau4[_ri]:.1f}  hold={_holds4[_bi]:.0f}",
                xy=(_ranks4[_ri], _tau4[_ri]),
                xytext=(18, 0), textcoords="offset points",
                fontsize=9, color=_c4, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=_c4, lw=0.6, alpha=0.5))

ax.set_xlabel("Eigenvalue rank  (sorted by $|\\lambda|$, largest first)", fontsize=13)
ax.set_ylabel(r"$\tau_{\mathrm{eff}} = -1\,/\,\ln|\lambda|$  (time steps)", fontsize=13)
ax.set_yscale("log")
ax.set_xlim(0, _N4 + 10)
ax.grid(True, alpha=0.1, which="both")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

_leg4 = [_Line2D4([0], [0], marker="o", color="w",
                   markerfacecolor=bit_colors[i], markeredgecolor="white",
                   markersize=10, label=f"Bit {i}  (hold≈{_holds4[i]:.0f})")
         for i in range(_nb4)]
_leg4.append(_Line2D4([0], [0], marker="o", color="w",
                       markerfacecolor="steelblue", markeredgecolor="none",
                       markersize=7, label="Bulk"))
ax.legend(handles=_leg4, fontsize=9, loc="upper right", framealpha=0.85, ncol=2)

# -- right: hold-time reference box --
ax2.set_xlim(0, 1)
_ylim_scree = ax.get_ylim()
ax2.set_ylim(_ylim_scree)
ax2.set_yscale("log")
ax2.set_axis_off()

_xlim2 = ax2.get_xlim()
_seg_x0 = _xlim2[0] + 0.15 * (_xlim2[1] - _xlim2[0])
_seg_x1 = _xlim2[0] + 0.55 * (_xlim2[1] - _xlim2[0])
for _bi in range(_nb4):
    _h = _holds4[_bi]
    ax2.plot([_seg_x0, _seg_x1], [_h, _h],
             color=bit_colors[_bi], linewidth=3.5, alpha=0.85,
             solid_capstyle="round", zorder=3)
    ax2.text(_seg_x1 + 0.05 * (_xlim2[1] - _xlim2[0]), _h,
             f"hold≈{_h:.0f}  (Bit {_bi}, p={_ppl4[_bi]})",
             color=bit_colors[_bi], fontsize=8.5, va="center")

fig.suptitle(f"Effective Timescale Scree  —  g = {G_FOCUS},  N = {_N4}  neurons  "
             f"(dominant: {_dom4_mode})",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, f"P06_tau_eff_scree_g{G_FOCUS}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% P05b  Jacobian spectrum — all 4 output-coupling criteria
# Requires: _ef, _Vf, _Wo, _Qf, _eig_abs_rank,
#           _coup_eig_all, _coup_schur_all, _r_eig_tgt, _r_schur_tgt, _dom_r,
#           ZOOM_XLIM, ZOOM_YLIM, theta, bit_colors, UNTRAINED_COLOR, TRAINED_COLOR
#
# 2 × 2 panel: rows = basis (eigenmode / Schur),
#              cols = method (connectivity / |correlation|)

# -- dominant eigenvalue indices per criterion (all index into _ef) ------------
_d5b_eigs = _ef.copy()     # eigenvalues in original np.linalg.eig order
_d5b_Vf   = _Vf.copy()

# eig-conn: argmax |W_out V| per bit → index into _ef
_dom5b_eig_conn = {bi: int(np.argmax(_coup_eig_all[bi])) for bi in range(_nbf)}

# eig-corr: argmax |r(eig-proj, target)| per bit → index into _ef
_dom5b_eig_corr = {bi: int(np.argmax(np.abs(_r_eig_tgt[:, bi]))) for bi in range(_nbf)}

# schur-conn: argmax |W_out Q| → Schur rank → eigenvalue index via _eig_abs_rank
_dom5b_schur_conn = {
    bi: int(_eig_abs_rank[int(np.argmax(_coup_schur_all[bi]))])
    for bi in range(_nbf)}

# schur-corr: _dom_r gives Schur rank → eigenvalue index via _eig_abs_rank
_dom5b_schur_corr = {bi: int(_eig_abs_rank[_dom_r[bi]]) for bi in range(_nbf)}

_criteria_5b = [
    ("Eig → output\n(connectivity)",   _dom5b_eig_conn),
    ("Eig → output\n(|correlation|)",  _dom5b_eig_corr),
    ("Schur → output\n(connectivity)", _dom5b_schur_conn),
    ("Schur → output\n(|correlation|)",_dom5b_schur_corr),
]

fig5b, axes5b = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
_ax5b_flat = axes5b.ravel()

for _pi, (_crit_name, _dom_dict) in enumerate(_criteria_5b):
    _ax5b = _ax5b_flat[_pi]

    # unit circle
    _ax5b.plot(np.cos(theta), np.sin(theta), color="#c0392b",
               linewidth=1.2, alpha=0.85, linestyle="-", zorder=1)
    _ax5b.axhline(0, color="#bbb", linewidth=0.3, zorder=0)
    _ax5b.axvline(0, color="#bbb", linewidth=0.3, zorder=0)

    # bulk eigenvalues (trained)
    _dom5b_set = set(_dom_dict.values())
    _rest5b = np.array([i for i in range(len(_d5b_eigs)) if i not in _dom5b_set])
    if len(_rest5b):
        _ax5b.scatter(_d5b_eigs[_rest5b].real, _d5b_eigs[_rest5b].imag,
                      s=16, alpha=0.55, c=TRAINED_COLOR, edgecolors="none", zorder=2)

    # highlighted dominant modes per bit
    for _bi, _ei in _dom_dict.items():
        _ax5b.scatter(_d5b_eigs[_ei].real, _d5b_eigs[_ei].imag,
                      s=70, alpha=1.0, color=bit_colors[_bi],
                      edgecolors="white", linewidths=0.8, zorder=5,
                      label=f"Bit {_bi}")
        # also mark conjugate if present
        _conj = np.conj(_d5b_eigs[_ei])
        if abs(_d5b_eigs[_ei].imag) > 1e-6:
            _conj_cands = np.where(np.abs(_d5b_eigs - _conj) < 1e-8)[0]
            for _cc in _conj_cands:
                if _cc != _ei:
                    _ax5b.scatter(_d5b_eigs[_cc].real, _d5b_eigs[_cc].imag,
                                  s=40, alpha=0.7, color=bit_colors[_bi],
                                  edgecolors="white", linewidths=0.6, zorder=4)

    _ax5b.set_xlim(*ZOOM_XLIM)
    _ax5b.set_ylim(*ZOOM_YLIM)
    _ax5b.set_aspect("equal")
    _ax5b.grid(True, alpha=0.12)
    _ax5b.set_title(_crit_name, fontsize=10, fontweight="bold")
    _ax5b.set_xlabel("Re($\\lambda$)", fontsize=9)
    _ax5b.set_ylabel("Im($\\lambda$)", fontsize=9)
    if _pi == 0:
        _ax5b.legend(fontsize=7.5, loc="upper left", framealpha=0.8)

fig5b.suptitle(f"Jacobian Eigenvalue Spectrum — g = {G_FOCUS}\n"
               f"(each panel highlights dominant eigenvalue per bit by the labelled criterion)",
               fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig5b.savefig(os.path.join(FIGS_DIR, f"P05b_spectrum_criteria_g{G_FOCUS}.pdf"),
                  bbox_inches="tight", dpi=150)
plt.show()

# %% P07  Autocorrelation of top-N Schur modes and neurons
# Requires: _pca_h, _Qf, _r_schur_tgt, _r_neu_tgt, _tauf, _nbf, G_FOCUS, FIGS_DIR
# Configuration
N_TOP_AC = 9     # number of modes/neurons to show  (best if a perfect square)
MAX_LAG  = 200   # autocorrelation lag in timesteps

from scipy.optimize import curve_fit as _curve_fit_ac

_n_side_ac = int(np.ceil(np.sqrt(N_TOP_AC)))

# -- select top modes and neurons by max |r_tgt| across bits --
_ac_mode_ranks = np.argsort(np.max(np.abs(_r_schur_tgt), axis=1))[::-1][:N_TOP_AC]
_ac_neu_ranks  = np.argsort(np.max(np.abs(_r_neu_tgt),   axis=1))[::-1][:N_TOP_AC]

# -- Schur-projected time series per trajectory ----------------------------
_Z_traj_ac = np.einsum('nth,hk->ntk', _pca_h, _Qf)   # (n_traj, T, N)

# -- vectorised autocorrelation (via direct sum; fast for small MAX_LAG) ---
def _autocorr_traj(x_traj, max_lag):
    """
    x_traj : (n_traj, T)
    Returns : (max_lag,) mean normalised autocorrelation across trajectories.
    """
    n_traj, T = x_traj.shape
    max_lag = min(max_lag, T - 1)
    x_c = x_traj - x_traj.mean(axis=1, keepdims=True)  # mean-centre per traj
    # variance per trajectory (avoid divide-by-zero)
    var = (x_c ** 2).mean(axis=1) + 1e-30               # (n_traj,)
    ac = np.zeros(max_lag)
    for lag in range(max_lag):
        ac[lag] = np.mean(
            (x_c[:, :T-lag] * x_c[:, lag:]).mean(axis=1) / var
        )
    return ac

def _exp_decay_ac(t, tau):
    return np.exp(-t / np.maximum(tau, 0.1))

_lags_ac = np.arange(MAX_LAG)

def _make_autocorr_grid(ax_grid, series_list, titles, grid_n, color, label_prefix):
    """Fill a grid of autocorrelation subplots, fit exponential to each."""
    for _k, (_ac_ts, _ttl) in enumerate(zip(series_list, titles)):
        _rr, _cc = _k // grid_n, _k % grid_n
        _ax = ax_grid[_rr, _cc]
        _ac = _autocorr_traj(_ac_ts, MAX_LAG)
        _ax.plot(_lags_ac, _ac, lw=1.2, color=color, alpha=0.85)
        _ax.axhline(0, color='k', lw=0.5, alpha=0.4, zorder=0)
        # fit exponential to first positive-correlation segment
        try:
            _pos = _ac > 0
            if _pos.sum() > 5:
                (tau_fit,), _ = _curve_fit_ac(
                    _exp_decay_ac, _lags_ac[_pos], _ac[_pos],
                    p0=[20.0], bounds=(0.1, 1e4), maxfev=5000)
                _ax.plot(_lags_ac, _exp_decay_ac(_lags_ac, tau_fit),
                         '--', lw=1.5, color="#e76f51", alpha=0.9,
                         label=f"$\\tau_{{\\rm fit}}$={tau_fit:.1f}")
                _ax.legend(fontsize=6.5, loc="upper right", framealpha=0.7)
        except Exception:
            pass
        _ax.set_title(_ttl, fontsize=7.5)
        _ax.set_ylim(-0.35, 1.05)
        _ax.set_xlabel("Lag (steps)", fontsize=7)
        _ax.set_ylabel("$C(\\tau)$", fontsize=7)
        _ax.tick_params(labelsize=6)
        _ax.grid(True, alpha=0.12)
    # hide unused cells
    for _k in range(len(series_list), grid_n ** 2):
        ax_grid[_k // grid_n, _k % grid_n].set_visible(False)

# ── Schur mode autocorrelations ──
_mode_series = [_Z_traj_ac[:, :, _mi] for _mi in _ac_mode_ranks]
_mode_titles = []
for _mi in _ac_mode_ranks:
    _r_str = "  ".join([f"B{bi}:{_r_schur_tgt[_mi, bi]:+.2f}" for bi in range(_nbf)])
    _mode_titles.append(f"Schur {_mi+1}  ($\\tau_{{\\rm eff}}$={_tauf[_mi]:.0f})\n{_r_str}")

fig_acm, axes_acm = plt.subplots(_n_side_ac, _n_side_ac,
                                  figsize=(_n_side_ac * 3.5, _n_side_ac * 2.5),
                                  squeeze=False)
_make_autocorr_grid(axes_acm, _mode_series, _mode_titles,
                    _n_side_ac, "#2563eb", "Schur")
fig_acm.suptitle(f"Autocorrelation — top-{N_TOP_AC} Schur modes  (by max |r_tgt|)  "
                 f"g = {G_FOCUS}\ndashed = exp fit",
                 fontsize=11, fontweight="bold")
plt.tight_layout()
if SAVE_FIGS:
    fig_acm.savefig(os.path.join(FIGS_DIR, f"P07a_autocorr_modes_g{G_FOCUS}.pdf"),
                    bbox_inches="tight", dpi=150)
plt.show()

# ── Neuron autocorrelations ──
_neu_series = [_pca_h[:, :, _ni] for _ni in _ac_neu_ranks]
_neu_titles = []
for _ni in _ac_neu_ranks:
    _r_str = "  ".join([f"B{bi}:{_r_neu_tgt[_ni, bi]:+.2f}" for bi in range(_nbf)])
    _neu_titles.append(f"Neuron {_ni+1}\n{_r_str}")

fig_acn, axes_acn = plt.subplots(_n_side_ac, _n_side_ac,
                                  figsize=(_n_side_ac * 3.5, _n_side_ac * 2.5),
                                  squeeze=False)
_make_autocorr_grid(axes_acn, _neu_series, _neu_titles,
                    _n_side_ac, "#059669", "Neuron")
fig_acn.suptitle(f"Autocorrelation — top-{N_TOP_AC} neurons  (by max |r_tgt|)  "
                 f"g = {G_FOCUS}\ndashed = exp fit",
                 fontsize=11, fontweight="bold")
plt.tight_layout()
if SAVE_FIGS:
    fig_acn.savefig(os.path.join(FIGS_DIR, f"P07b_autocorr_neurons_g{G_FOCUS}.pdf"),
                    bbox_inches="tight", dpi=150)
plt.show()


# %% [markdown]
# ### Section D: Dimensionality (PCA) — [commented out]
# Uncomment cell 2.4 and the two cells below to reactivate PCA analysis.

# %% P-D1  PCA dimensionality — cumulative variance  [COMMENTED OUT]
# ── PCA (commented out) ──────────────────────────────────────────────────────
# PCA_GAIN = G_FOCUS
# MAX_COMP_SHOW = min(60, len(_pca_cum))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
# if _has_unt:
#     ax1.plot(np.arange(1, MAX_COMP_SHOW + 1), _unt_cum[:MAX_COMP_SHOW],
#              "o--", markersize=3, linewidth=1.2, color=UNTRAINED_COLOR,
#              alpha=0.85, label="Untrained", zorder=2)
# ax1.plot(np.arange(1, MAX_COMP_SHOW + 1), _pca_cum[:MAX_COMP_SHOW],
#          "o-", markersize=4, linewidth=1.5, color="#264653",
#          label="Trained", zorder=3)
# for thresh in [0.90, 0.95, 0.99]:
#     n_needed = int(np.searchsorted(_pca_cum, thresh)) + 1
#     ax1.axhline(thresh, color="#adb5bd", linewidth=0.8, linestyle=":")
#     ax1.annotate(f"{thresh*100:.0f}% \u2192 {n_needed} PCs",
#                  xy=(n_needed, thresh), textcoords="offset points", xytext=(8, -10),
#                  fontsize=9, color="#e76f51", fontweight="bold",
#                  arrowprops=dict(arrowstyle="->", color="#e76f51", lw=0.8))
# ax1.set_xlabel("Number of principal components", fontsize=12)
# ax1.set_ylabel("Cumulative variance explained", fontsize=12)
# ax1.set_yscale("log"); ax1.set_xscale("log")
# ax1.grid(True, alpha=0.2)
# ax1.legend(fontsize=10, loc="lower right")
# ax1.set_title("Cumulative variance explained", fontsize=12)
# ax2.bar(np.arange(1, MAX_COMP_SHOW + 1), _pca_var_frac[:MAX_COMP_SHOW],
#         color="#2a9d8f", edgecolor="none", alpha=0.8)
# ax2.set_xlabel("Principal component", fontsize=12)
# ax2.set_ylabel("Fraction of variance", fontsize=12)
# ax2.set_xlim(0, MAX_COMP_SHOW + 1)
# ax2.grid(True, alpha=0.2, axis="y")
# ax2.set_title("Variance per component", fontsize=12)
# fig.suptitle(f"PCA Dimensionality \u2014 g={PCA_GAIN},  {N_TRAJ} traj,  "
#              f"N={_pca_H},  n_bits={_pca_cfg['n_bits']}",
#              fontsize=13, fontweight="bold", y=1.03)
# plt.tight_layout()
# if SAVE_FIGS:
#     fig.savefig(os.path.join(FIGS_DIR, f"pca_dimensionality_g{PCA_GAIN}.pdf"),
#                 bbox_inches="tight", dpi=150)
# plt.show()
# ─────────────────────────────────────────────────────────────────────────────

# %% P-D2  PC-to-bit correspondence  [COMMENTED OUT]
# ── PCA (commented out) ──────────────────────────────────────────────────────
# _fig_h = max(5.0, 0.52 * N_PCS_CORR + 1.2)
# fig, (ax_corr, ax_var) = plt.subplots(
#     1, 2, figsize=(14, _fig_h), gridspec_kw=dict(width_ratios=[1, 0.38]))
# im = ax_corr.imshow(_corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
# ax_corr.set_yticks(range(N_PCS_CORR))
# ax_corr.set_yticklabels(
#     [f"PC {j+1}  ({100*_pca_var_frac[j]:.1f}%)" for j in range(N_PCS_CORR)], fontsize=8)
# ax_corr.set_xticks(range(_n_bits_pca))
# ax_corr.set_xticklabels([f"Bit {i}\n(p={_pp_list_pca[i]})" for i in range(_n_bits_pca)],
#                          fontsize=8)
# for _j in range(N_PCS_CORR):
#     for _i in range(_n_bits_pca):
#         _v = _corr_mat[_j, _i]
#         ax_corr.text(_i, _j, f"{_v:+.2f}", ha="center", va="center",
#                      fontsize=6.5, color="white" if abs(_v) > 0.55 else "black")
# plt.colorbar(im, ax=ax_corr, label="Pearson $r$", shrink=0.6)
# ax_corr.set_xlabel("Output bit", fontsize=11)
# ax_corr.set_ylabel("Principal component", fontsize=11)
# ax_corr.set_title(f"corr(PC$_j$ proj, bit$_i$ state) \u2014 g = {PCA_GAIN}", fontsize=10)
# ax_var.barh(range(N_PCS_CORR), 100 * _pca_var_frac[:N_PCS_CORR],
#             color="#2a9d8f", edgecolor="none", alpha=0.85)
# ax_var.set_yticks(range(N_PCS_CORR))
# ax_var.set_yticklabels([f"PC {j+1}" for j in range(N_PCS_CORR)], fontsize=8)
# ax_var.invert_yaxis()
# ax_var.set_xlabel("Variance (%)", fontsize=10)
# ax_var.grid(True, alpha=0.2, axis="x")
# ax_var.spines["top"].set_visible(False)
# ax_var.spines["right"].set_visible(False)
# ax_var.set_title(f"Variance per PC\nPR = {_pr_val:.2f}", fontsize=10)
# fig.suptitle(f"PC\u2013bit correspondence \u2014 g = {PCA_GAIN},  n_bits = {_n_bits_pca},  "
#              f"N = {_pca_H},  PR = {_pr_val:.2f}",
#              fontsize=12, fontweight="bold", y=1.01)
# plt.tight_layout()
# if SAVE_FIGS:
#     fig.savefig(os.path.join(FIGS_DIR, f"pc_bit_corr_g{PCA_GAIN}.pdf"),
#                 bbox_inches="tight", dpi=150)
# plt.show()
# print(f"\nPC\u2013bit assignment  (g = {PCA_GAIN},  PR = {_pr_val:.2f}):")
# for _i in range(_n_bits_pca):
#     _best_j = int(np.argmax(np.abs(_corr_mat[:, _i])))
#     print(f"  Bit {_i}  (p={_pp_list_pca[_i]})  <->  PC {_best_j + 1}"
#           f"  (r = {_corr_mat[_best_j, _i]:+.3f},"
#           f"  var = {100*_pca_var_frac[_best_j]:.1f}%)")
# ─────────────────────────────────────────────────────────────────────────────

# %%
