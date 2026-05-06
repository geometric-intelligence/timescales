# %% [markdown]
# # 2. Coupling & Spectra Analysis — Schur-Init Grid
#
# For a single chosen (task, gain, init, theta) condition this notebook:
#   1. Loads the best seed (by a selectable metric) from best_seeds.csv
#   2. Runs a forward pass → Pearson correlations between eigenmodes /
#      neurons and output channels (results cached to disk)
#   3. Produces five publication-ready PDF figures:
#        Fig 1  Eigenmode coupling heatmap (top-K slow modes × n_ch)
#        Fig 2  Neuron coupling heatmap    (top-K slow neurons × n_ch)
#        Fig 3  Scatter: task τ vs dominant-eigenmode τ_eff
#        Fig 4  Scatter: task τ vs dominant-neuron τ
#        Fig 5  Jacobian spectrum (untrained vs trained, dominant modes highlighted)
#
# ── QUICK START ────────────────────────────────────────────────────────────────
#   1. Fill SWEEP_DIRS in PART 0 (same paths as notebook 1).
#   2. Ensure best_seeds.csv exists (run notebook 1, PART 1.5 first).
#   3. Set ANALYSIS_TASK / ANALYSIS_GAIN / ANALYSIS_INIT / ANALYSIS_THETA.
#   4. Run all cells top-to-bottom.


# %% Imports
import glob as _glob
import os
import re
import sys
import subprocess

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()
os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())

plt.rcParams["svg.fonttype"] = "path"


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 0 — Configuration  ← edit this section
# ══════════════════════════════════════════════════════════════════════════════

# ── Sweep directories (same as notebook 1) ────────────────────────────────────
SWEEP_DIRS = [
    # Server A (g = 0.5)
    "/home/facosta/timescales/timescales/logs/experiments/schur_init_grid_g05_20260505_061150",
    # Server B (g = 0.9)
    "/home/facosta/timescales/timescales/logs/experiments/schur_init_grid_g09_20260504_230709",
]

# ── Condition to analyse ──────────────────────────────────────────────────────

ANALYSIS_TASK   = "sine"          # "ff" | "sine"
ANALYSIS_GAIN   = 0.5           # 0.5 | 0.9
ANALYSIS_INIT   = "uniform"     # "uniform" | "powerlawtau" | "schurH"
ANALYSIS_THETA  = "full"      # "full" | "fixedA"

# Best-seed selection criterion (must match a column in best_seeds.csv)
SELECTION_METRIC = "min_val_loss"   # "min_val_loss" | "max_val_acc"

# ── Analysis parameters ───────────────────────────────────────────────────────
N_CORR    = 200   # number of trajectories for the correlation forward pass
N_HEATMAP = 6     # number of columns shown on each heatmap

# Heatmap column ordering (applies to both the eigenmode and neuron heatmaps)
#   "timescale" : columns = top-K slowest STABLE modes / highest-τ neurons
#   "coupling"  : columns = dominant mode per bit (bit 0 → slot 0, bit 1 → slot 1, …)
#                 deduplicating as we go; remaining slots filled with next-best couplings
HEATMAP_SORT = "timescale"   # "timescale" | "coupling"

# Coupling metric used for heatmaps, sorting, and scatter plots
#   "pearson"      : |Pearson r| between eigenmode/neuron activity and target output
#   "connectivity" : |W_out @ V| (eigenmode) and |W_out| (neuron) — purely structural,
#                    no forward pass needed beyond what is already cached
#   "ablation"     : σ(Z_k) × |W_out_k| / σ(ŷ) — activity-weighted structural coupling;
#                    proxy for how much each mode/neuron drives output fluctuations
COUPLING_METRIC = "connectivity"   # "pearson" | "connectivity" | "ablation"


# Timescale used in scatter plots (Fig 3 eigenmode axis)
#   "auto"        : flip-flop → decay, sine → oscillation
#   "decay"       : τ_eff = -dt / log|λ|  (how long the mode persists)
#   "oscillation" : τ_osc = 2π·dt / |arg(λ)|  (period of complex rotation)
SCATTER_TIMESCALE = "auto"   # "auto" | "decay" | "oscillation"

# Scatter plots (Figs 3 & 4)
LOG_LOG_SCATTER = False   # True = log-log axes

# For the flip-flop task: drop the last bit (bit N-1) from all displays.
# Set to False to keep all bits.
FF_DROP_LAST_BIT = True

# Scatter plots: aggregate over all seeds to show mean ± std error bars.
# For "pearson"/"ablation" this runs a forward pass per seed (a few seconds each).
# For "connectivity" it only reads spectral_final.pt (very fast).
SCATTER_MULTI_SEED = True

# x-axis of the log-polar spectrum (Fig 5b)
#   "decay_rate" : x = −log|λ|       (1/timestep)  — unit circle at x = 0
#   "tau_eff"    : x = −1/log|λ|     (timesteps)   — τ_eff directly; stable ↔ x > 0
LOGPOLAR_XAXIS = "tau_eff"   # "decay_rate" | "tau_eff"

# Complex-plane spectrum (Fig 5): plot ±SPECTRUM_ZOOM on Re and Im axes
SPECTRUM_XRANGE = (0.05, 1.05)#(0.7,1.02)#(0.5, 1.05)
SPECTRUM_YRANGE = (-0.45, 0.45)#(-0.16, 0.16)

# ── Explicit colours (edit freely) ───────────────────────────────────────────
C_SPEC_UNTRAINED = "#27d3f5"    # untrained eigenvalues  (light blue)
C_SPEC_TRAINED   = "#f54927"    # trained eigenvalues    (dark orange, ColorBrewer Dark2)
C_UNIT_CIRCLE    = "#222222"    # unit circle            (near-black)
C_HEATMAP        = "YlOrRd"     # heatmap colormap

# One colour per output channel (up to 6); used in scatter plots and spectrum.
# Uncomment exactly ONE of the options below.

# ── Option 0: ColorBrewer qualitative mixed palette (original)
# C_CHANNEL = [
#     "#e41a1c",   # channel 0  red
#     "#377eb8",   # channel 1  blue
#     "#4daf4a",   # channel 2  green
#     "#984ea3",   # channel 3  purple
#     "#ff7f00",   # channel 4  orange
#     "#a65628",   # channel 5  brown
# ]

# ── Option 1: plasma — purple → orange → yellow (warm, perceptually uniform)
C_CHANNEL = [plt.cm.plasma(x) for x in [0.12, 0.30, 0.50, 0.68, 0.82, 0.93]]

# ── Option 2: YlOrRd — yellow → orange → red (cohesive with heatmap C_HEATMAP)
#C_CHANNEL = [plt.cm.YlOrRd(x) for x in [0.25, 0.40, 0.55, 0.70, 0.83, 0.95]]

# ── Option 3: curated warm — deep crimson → orange → amber → straw
#C_CHANNEL = ["#8c0d10", "#d62728", "#f46d43", "#fdae61", "#fee08b", "#ffffbf"]

# ── I/O directories ───────────────────────────────────────────────────────────
BEST_SEEDS_CSV = os.path.join("notebooks", "schur_init_grid", "best_seeds.csv")
CACHE_DIR      = os.path.join("notebooks", "schur_init_grid", ".cache")
FIGS_DIR       = os.path.join("notebooks", "schur_init_grid", "figs", "coupling_spectra")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,  exist_ok=True)

SAVE_FIGS = False   # write PDFs to FIGS_DIR


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 1 — Locate best seed and load spectral data
# ══════════════════════════════════════════════════════════════════════════════

# ── 1a: read best_seeds.csv ───────────────────────────────────────────────────
if not os.path.exists(BEST_SEEDS_CSV):
    raise FileNotFoundError(
        f"best_seeds.csv not found at {BEST_SEEDS_CSV}.\n"
        "Run notebook 1, PART 1.5 first to generate it."
    )

bdf  = pd.read_csv(BEST_SEEDS_CSV)
cond = bdf[
    (bdf["task"]  == ANALYSIS_TASK)  &
    (bdf["gain"]  == ANALYSIS_GAIN)  &
    (bdf["init"]  == ANALYSIS_INIT)  &
    (bdf["theta"] == ANALYSIS_THETA)
]
if cond.empty:
    raise ValueError(
        f"No entry in best_seeds.csv for "
        f"task={ANALYSIS_TASK!r}, gain={ANALYSIS_GAIN}, "
        f"init={ANALYSIS_INIT!r}, theta={ANALYSIS_THETA!r}.\n"
        "Check that SWEEP_DIRS in notebook 1 were correct and that PART 1.5 ran."
    )

_seed_col = (
    "best_seed_min_loss" if SELECTION_METRIC == "min_val_loss"
    else "best_seed_max_acc"
)
BEST_SEED = int(cond.iloc[0][_seed_col])
print(f"Best seed ({SELECTION_METRIC}): {BEST_SEED}")

# ── 1b: locate seed_path by scanning SWEEP_DIRS ──────────────────────────────
# Experiment folder naming convention: g{gain}_{init}_{theta}_{task}
_gain_str = f"{ANALYSIS_GAIN:g}"     # 0.5 → "0.5",  0.9 → "0.9"
EXP_NAME  = f"g{_gain_str}_{ANALYSIS_INIT}_{ANALYSIS_THETA}_{ANALYSIS_TASK}"
SEED_PATH = None
for sd in SWEEP_DIRS:
    candidate = os.path.join(sd, EXP_NAME, f"seed_{BEST_SEED}")
    if os.path.isdir(candidate):
        SEED_PATH = candidate
        break
if SEED_PATH is None:
    raise FileNotFoundError(
        f"Could not find seed dir for experiment '{EXP_NAME}', "
        f"seed {BEST_SEED} in any of SWEEP_DIRS:\n"
        + "\n".join(f"  {sd}" for sd in SWEEP_DIRS)
    )
print(f"Seed path: {SEED_PATH}")

# ── 1c: load run_config ───────────────────────────────────────────────────────
with open(os.path.join(SEED_PATH, "run_config.yaml")) as _f:
    RC = yaml.safe_load(_f)
print(f"Task: {RC['task']},  dt={RC['dt']},  hidden_size={RC['hidden_size']}")

# ── 1d: load spectral snapshots ───────────────────────────────────────────────
def _load_pt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)

spec_init  = _load_pt(os.path.join(SEED_PATH, "spectral_init.pt"))
spec_final = _load_pt(os.path.join(SEED_PATH, "spectral_final.pt"))
print("Spectral snapshots loaded.")

# ── 1e: extract key arrays ────────────────────────────────────────────────────
W_out        = spec_final["W_out"]         # (n_ch, N)  output weight matrix
V            = spec_final["V"]             # (N, N)     eigenvector matrix (columns)
eigvals      = spec_final["eigvals_eig"]   # (N,)       complex eigenvalues (trained)
taus_neu     = spec_final["taus"]          # (N,)       neuron time constants (time units)
eigvals_init = spec_init["eigvals_eig"]    # (N,)       complex eigenvalues (untrained)
dt_val       = float(RC["dt"])

print(f"  W_out: {W_out.shape},  V: {V.shape},  eigvals: {eigvals.shape}")
print(f"  dt = {dt_val},  taus_neu range = [{taus_neu.min():.3f}, {taus_neu.max():.3f}]")


# ══════════════════════════════════════════════════════════════════════════════
# %% Module-level helpers (used by both the main analysis and multi-seed loop)
# ══════════════════════════════════════════════════════════════════════════════

def _pearson_r(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix.  (M, Kz), (M, Ky) → (Kz, Ky)."""
    n  = Z.shape[0]
    Zc = (Z - Z.mean(0)).astype(np.float64)
    Yc = (Y - Y.mean(0)).astype(np.float64)
    Zs = np.where(Zc.std(0) > 1e-12, Zc.std(0), 1e-12)
    Ys = np.where(Yc.std(0) > 1e-12, Yc.std(0), 1e-12)
    return ((Zc.T @ Yc) / (n * Zs[:, None] * Ys[None, :])).astype(np.float32)


def _build_model(rc: dict, seed_path: str):
    """Reconstruct the trained RNN from run_config + best checkpoint."""
    ckpts = _glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No best-model checkpoint found in {seed_path}/checkpoints/"
        )
    import torch.nn as _torch_nn
    from rnns.rnn import RNN, RNNLightning

    task_key = rc.get("task", "flip_flop")
    n_bits   = rc.get("n_bits", 6)
    n_pairs  = rc.get("n_pairs", 3)
    in_sz    = rc.get("input_size", 1) if task_key == "sine_wave" else n_bits
    out_sz   = 2 * n_pairs             if task_key == "sine_wave" else n_bits

    model = RNN(
        input_size=in_sz,
        hidden_size=rc["hidden_size"],
        output_size=out_sz,
        dt=rc["dt"],
        time_constants_config=rc.get("time_constants_config"),
        activation=getattr(_torch_nn, rc.get("activation", "Tanh")),
        learn_time_constants=rc.get("learn_time_constants", False),
        init_time_constant=rc.get("init_time_constant"),
        init_time_constants_config=rc.get("init_time_constants_config"),
        shared_time_constant=rc.get("shared_time_constant", False),
        normalize_hidden=rc.get("normalize_hidden", False),
        zero_diag_wrec=rc.get("zero_diag_wrec", False),
        recurrent_gain=rc["recurrent_gain"],
        noise_std=0.0,
        wrec_init="normal_scaled",
        wrec_init_config=None,
        alpha_parameterization=rc.get("alpha_parameterization", "exponential"),
        dynamics_type=rc.get("dynamics_type", "rate"),
    )
    lit = RNNLightning(
        model=model,
        learning_rate=rc["learning_rate"],
        weight_decay=rc["weight_decay"],
        step_size=rc.get("lr_step_size", rc.get("step_size", 1000)),
        gamma=rc["gamma"],
        task=task_key,
        init_hidden_value=rc.get("init_hidden_value"),
    )
    ckpt_data = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    lit.load_state_dict(ckpt_data["state_dict"])
    lit.eval()
    return lit.model


def _run_fwd(model, rc: dict, n_corr: int) -> tuple[np.ndarray, np.ndarray]:
    """Run n_corr validation trajectories; return (h_flat, tgt_flat) as float32."""
    _task = rc.get("task", "flip_flop")
    if _task == "flip_flop":
        from datamodules.flip_flop import FlipFlopDataModule
        _dm = FlipFlopDataModule(
            n_bits=rc["n_bits"], p_pulse=rc["p_pulse"],
            pulse_amplitude=rc.get("pulse_amplitude", 1.0),
            num_time_steps=rc["num_time_steps"],
            num_val_trajectories=n_corr, batch_size=n_corr,
        )
        _dm.setup()
        _inp, _, _tgt = _dm.val_dataset.tensors
        with torch.no_grad():
            _h_seq, _ = model(_inp)
    else:
        from datamodules.sine_wave import SineWaveDataModule
        _dm = SineWaveDataModule(
            n_pairs=rc["n_pairs"], periods=rc["periods"],
            num_time_steps=rc["num_time_steps"], dt=rc["dt"],
            num_val_trajectories=n_corr, batch_size=n_corr,
            init_hidden_value=rc.get("init_hidden_value", 1.0),
        )
        _dm.setup()
        _inp, _, _tgt = _dm.val_dataset.tensors
        _h0 = torch.full((n_corr, rc["hidden_size"]),
                         float(rc.get("init_hidden_value", 1.0)))
        with torch.no_grad():
            _h_seq, _ = model(_inp, init_hidden=_h0)
    _M = n_corr * rc["num_time_steps"]
    _hf = _h_seq.numpy().reshape(_M, rc["hidden_size"]).astype(np.float32)
    _tf = _tgt.numpy().reshape(_M, _tgt.shape[-1]).astype(np.float32)
    return _hf, _tf


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 2 — Compute correlations (with on-disk cache)
# ══════════════════════════════════════════════════════════════════════════════

_CACHE_KEY = (
    f"{ANALYSIS_TASK}_{_gain_str}_{ANALYSIS_INIT}_{ANALYSIS_THETA}"
    f"_seed{BEST_SEED}_ncorr{N_CORR}_cplxr"  # _cplxr: complex-aware Pearson (modulation amplitude)
)
_c_hidden  = os.path.join(CACHE_DIR, f"{_CACHE_KEY}_hidden.npy")
_c_targets = os.path.join(CACHE_DIR, f"{_CACHE_KEY}_targets.npy")
_c_reig    = os.path.join(CACHE_DIR, f"{_CACHE_KEY}_corr_eig.npy")
_c_rneu    = os.path.join(CACHE_DIR, f"{_CACHE_KEY}_corr_neu.npy")

_cache_hit = all(os.path.exists(p) for p in [_c_hidden, _c_targets, _c_reig, _c_rneu])

if _cache_hit:
    print(f"Cache hit — loading from {CACHE_DIR}")
    h_flat   = np.load(_c_hidden)
    tgt_flat = np.load(_c_targets)
    r_eig    = np.load(_c_reig)
    r_neu    = np.load(_c_rneu)

else:
    print("Cache miss — running forward pass …")

    # ── Load trained model ─────────────────────────────────────────────────────
    model = _build_model(RC, SEED_PATH)

    # ── Generate validation trajectories ──────────────────────────────────────
    task_key = RC.get("task", "flip_flop")

    if task_key == "flip_flop":
        from datamodules.flip_flop import FlipFlopDataModule
        dm = FlipFlopDataModule(
            n_bits=RC["n_bits"],
            p_pulse=RC["p_pulse"],
            pulse_amplitude=RC.get("pulse_amplitude", 1.0),
            num_time_steps=RC["num_time_steps"],
            num_val_trajectories=N_CORR,
            batch_size=N_CORR,
        )
        dm.setup()
        inp_raw, _, tgt_raw = dm.val_dataset.tensors   # (N_CORR, T, n_bits)
        with torch.no_grad():
            h_seq, _ = model(inp_raw)                  # h_seq: (N_CORR, T, N)

    else:  # sine_wave
        from datamodules.sine_wave import SineWaveDataModule
        dm = SineWaveDataModule(
            n_pairs=RC["n_pairs"],
            periods=RC["periods"],
            num_time_steps=RC["num_time_steps"],
            dt=RC["dt"],
            num_val_trajectories=N_CORR,
            batch_size=N_CORR,
            init_hidden_value=RC.get("init_hidden_value", 1.0),
        )
        dm.setup()
        inp_raw, _, tgt_raw = dm.val_dataset.tensors   # (N_CORR, T, 1), targets (N_CORR, T, 2*n_pairs)
        h0_val = float(RC.get("init_hidden_value", 1.0))
        h0 = torch.full((N_CORR, RC["hidden_size"]), h0_val)
        with torch.no_grad():
            h_seq, _ = model(inp_raw, init_hidden=h0)  # (N_CORR, T, N)

    # ── Flatten to (M, ·) ────────────────────────────────────────────────────
    M        = N_CORR * RC["num_time_steps"]
    h_flat   = h_seq.numpy().reshape(M, RC["hidden_size"]).astype(np.float32)
    tgt_flat = tgt_raw.numpy().reshape(M, tgt_raw.shape[-1]).astype(np.float32)
    print(f"  h_flat: {h_flat.shape},  tgt_flat: {tgt_flat.shape}")

    # ── Pearson correlation ────────────────────────────────────────────────────
    # Neuron–target correlation: r_neu[j, b] = Pearson(h_j, target_b)
    r_neu = _pearson_r(h_flat, tgt_flat)              # (N, n_ch)

    # Eigenmode–target correlation: project h onto eigenvector basis first.
    # J = V diag(λ) V^{-1}  ⟹  z = V^{-1} h  (column vector form)
    # Row-vector form:  Z = h_flat @ (V^{-1})^T
    # For oscillatory modes the projection is complex; we use modulation amplitude
    # sqrt(r_Re² + r_Im²) so that sine channels couple to their mode pair correctly.
    V_inv   = np.linalg.pinv(V.astype(np.complex128))          # (N, N) complex
    _Z_cplx = h_flat.astype(np.float64) @ V_inv.T              # (M, N) complex
    _r_re   = _pearson_r(_Z_cplx.real.astype(np.float32), tgt_flat)  # (N, n_ch)
    _r_im   = _pearson_r(_Z_cplx.imag.astype(np.float32), tgt_flat)  # (N, n_ch)
    r_eig   = np.sqrt(_r_re**2 + _r_im**2)                     # modulation amplitude

    # ── Cache results ─────────────────────────────────────────────────────────
    np.save(_c_hidden,  h_flat)
    np.save(_c_targets, tgt_flat)
    np.save(_c_reig,    r_eig)
    np.save(_c_rneu,    r_neu)
    print(f"  Saved to cache ({CACHE_DIR})")

print(f"r_eig (Pearson): {r_eig.shape},  r_neu (Pearson): {r_neu.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# %% Coupling metric selection  (fast — no new forward pass required)
# ══════════════════════════════════════════════════════════════════════════════
# r_eig / r_neu start as Pearson correlations (always cached above).
# For other metrics we overwrite them in-memory; the on-disk cache is untouched.
#
# spectral_final.pt already stores:
#   coup_eig_re  (n_ch, N)  = |W_out @ V|          (structural eigenmode coupling)
#   coup_neuron  (n_ch, N)  = |W_out|               (structural neuron coupling)
# These are transposed to (N, n_ch) to match r_eig / r_neu conventions.

if COUPLING_METRIC == "connectivity":
    r_eig = np.asarray(spec_final["coup_eig_re"]).T.astype(np.float32)  # (N, n_ch)
    r_neu = np.asarray(spec_final["coup_neuron"]).T.astype(np.float32)   # (N, n_ch)

elif COUPLING_METRIC == "ablation":
    # Ablation proxy: σ(Z_k) × |W_out_k→b| / σ(ŷ_b)
    # σ(Z_eig) = std of each eigenmode's projected activity over the validation set
    # |W_out_k→b| = connectivity magnitude (from spectral snapshot)
    # σ(ŷ) = std of model output (computed from h_flat and W_out)
    _conn_eig = np.asarray(spec_final["coup_eig_re"])   # (n_ch, N) — already |·|
    _conn_neu = np.asarray(spec_final["coup_neuron"])   # (n_ch, N) — already |·|
    _W_out_np = np.asarray(spec_final["W_out"])          # (n_ch, N)

    # Model output std (same normalisation as 17_learn_tau_sweep.py)
    _y_hat = h_flat @ _W_out_np.T                   # (M, n_ch)
    _y_std = _y_hat.std(0).clip(1e-12)              # (n_ch,)

    # Eigenmode activities  Z = h V^{-1}  (same projection as Pearson block)
    _V_inv = np.linalg.pinv(np.asarray(V).astype(np.complex128))
    _Z_eig = (h_flat.astype(np.float64) @ _V_inv.T).real.astype(np.float32)
    _z_std = _Z_eig.std(0).clip(1e-12)             # (N,) — std per eigenmode
    _h_std = h_flat.std(0).clip(1e-12)             # (N,) — std per neuron

    r_eig = (_z_std[:, None] * _conn_eig.T / _y_std[None, :]).astype(np.float32)  # (N, n_ch)
    r_neu = (_h_std[:, None] * _conn_neu.T / _y_std[None, :]).astype(np.float32)  # (N, n_ch)

# else "pearson" — r_eig, r_neu are already the Pearson matrices; no change.

_COUP_LABEL = {
    "pearson":      "|Pearson r|",
    "connectivity": "|W_out · v_k|",
    "ablation":     "σ(Z_k)·|W_out|/σ(ŷ)  (ablation)",
}[COUPLING_METRIC]

print(f"Coupling metric: {COUPLING_METRIC!r}  →  "
      f"r_eig range [{r_eig.min():.4f}, {r_eig.max():.4f}], "
      f"r_neu range [{r_neu.min():.4f}, {r_neu.max():.4f}]")


# ══════════════════════════════════════════════════════════════════════════════
# %% Derived quantities  (always recomputed — fast)
# ══════════════════════════════════════════════════════════════════════════════

n_ch = r_eig.shape[1]   # number of output channels (6 for both tasks)
N    = r_eig.shape[0]   # number of hidden units

task_key = RC.get("task", "flip_flop")

# ── Channel display filter ────────────────────────────────────────────────────
# For FF with FF_DROP_LAST_BIT=True: exclude the last bit from every plot.
_full_n_ch = n_ch
if task_key == "flip_flop" and FF_DROP_LAST_BIT:
    _ch_display = np.arange(n_ch - 1)
else:
    _ch_display = np.arange(n_ch)
r_eig = r_eig[:, _ch_display]
r_neu = r_neu[:, _ch_display]
n_ch  = len(_ch_display)

# ── Eigenmode effective timescales (in time units, i.e. "seconds") ────────────
# tau_eff = -1 / log|λ|  (in timesteps) × dt  (time units/step)  = time units.
#
# Only defined for STABLE modes (|λ| < 1).  For |λ| ≥ 1 (neutrally stable or
# growing modes), log|λ| ≥ 0 and the formula gives a non-positive result that
# is physically meaningless.  We set those to NaN so they are excluded from
# the "slowest modes" ranking.
_abs_eig   = np.abs(eigvals)                                           # (N,) real
_log_abs   = np.log(np.clip(_abs_eig, 1e-12, None))                   # (N,) real
_stable    = _abs_eig < 1.0                                            # boolean mask
# For stable modes use the formula; for very-near-unit-circle stable modes cap
# the denominator at -1e-10 to avoid division by ≈0.
_denom     = np.where(_stable & (_log_abs < -1e-10), _log_abs, -1e-10)
tau_eff_ts = np.where(_stable, -1.0 / _denom, np.nan)                 # timesteps
tau_eff_s  = tau_eff_ts * dt_val  # time units ("s")

# ── Eigenmode oscillation timescales ──────────────────────────────────────────
# τ_osc = 2π·dt / |arg(λ)|   — period of rotation in the complex plane.
# Only meaningful for modes with nonzero imaginary part; real eigenvalues
# (arg ≈ 0) get τ_osc = NaN (they don't oscillate).
_arg_abs     = np.abs(np.angle(eigvals))                                # (N,)
_has_imag    = _arg_abs > 1e-10                                         # boolean mask
tau_osc_ts   = np.where(_has_imag, 2 * np.pi / _arg_abs, np.nan)       # timesteps
tau_osc_s    = tau_osc_ts * dt_val                                      # time units

# ── Unified scatter timescale ─────────────────────────────────────────────────
_scatter_timescale = (
    "oscillation" if SCATTER_TIMESCALE == "auto" and task_key == "sine_wave"
    else "decay" if SCATTER_TIMESCALE == "auto"
    else SCATTER_TIMESCALE
)
if _scatter_timescale == "oscillation":
    tau_scatter_s   = tau_osc_s
    _scatter_ylabel = "Dominant eigenmode  τ_osc (s)"
    _scatter_title  = "Task τ  vs  eigenmode τ_osc"
elif _scatter_timescale == "decay":
    tau_scatter_s   = tau_eff_s
    _scatter_ylabel = "Dominant eigenmode  τ_eff (s)"
    _scatter_title  = "Task τ  vs  eigenmode τ_eff"
else:
    raise ValueError(
        f"SCATTER_TIMESCALE must be 'auto', 'decay', or 'oscillation', "
        f"got {SCATTER_TIMESCALE!r}"
    )





n_unstable = int((~_stable).sum())
if n_unstable:
    print(f"  ⚠ {n_unstable}/{len(eigvals)} eigenmodes have |λ| ≥ 1 "
          f"(unstable / neutrally stable); their τ_eff is set to NaN "
          f"and they are excluded from the slow-mode ranking.")

# ── Neuron time constants (already in time units) ─────────────────────────────
# taus_neu = current_time_constants (same units as dt), no dt multiplication
tau_neu_s = taus_neu   # (N,)  time units

# ── Dominant coupling per channel — STABLE modes only ────────────────────────
# Zero out unstable modes so argmax never picks them.
_r_eig_stable = np.where(_stable[:, None], np.abs(r_eig), 0.0)  # (N, n_ch)
dom_mode = np.argmax(_r_eig_stable, axis=0)   # (n_ch,) index into stable eigenmodes
dom_neu  = np.argmax(np.abs(r_neu),  axis=0)  # (n_ch,) index into neurons (all valid)

# Scatter-specific dominant modes. For oscillation-period scatter plots, discard
# stable but non-oscillatory real modes because τ_osc is undefined for arg(λ)≈0.
_scatter_valid = _stable & np.isfinite(tau_scatter_s)
if _scatter_valid.any():
    _r_eig_scatter = np.where(_scatter_valid[:, None], np.abs(r_eig), 0.0)
    dom_mode_scatter = np.argmax(_r_eig_scatter, axis=0)
else:
    dom_mode_scatter = dom_mode


# ── Heatmap column-ordering helper ────────────────────────────────────────────
def _coupling_sort_idx(abs_corr: np.ndarray, dom: np.ndarray,
                       n_slots: int, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Return N_HEATMAP column indices for the heatmap, sorted by coupling.

    Priority: channel 0's dominant mode → channel 1's → … → channel n_ch-1's,
    deduplicating as we go.  If fewer than n_slots unique modes are found in one
    pass, the next-best coupling rank for each channel is tried, round-robin style.

    abs_corr  : (N, n_ch) absolute Pearson correlation
    dom       : (n_ch,)   dominant mode/neuron index per channel
    n_slots   : number of columns to fill
    valid_mask: (N,) boolean — if given, invalid rows (e.g. unstable modes)
                are zeroed out so they are never selected.
    """
    n_modes, n_ch = abs_corr.shape
    if valid_mask is not None:
        abs_corr = abs_corr.copy()
        abs_corr[~valid_mask, :] = 0.0
    # Pre-sort each channel's modes by descending correlation (rank list)
    ranked = np.argsort(abs_corr, axis=0)[::-1]   # (N, n_ch)
    selected: list[int] = []
    seen: set[int] = set()
    rank = 0
    while len(selected) < n_slots and rank < n_modes:
        for b in range(n_ch):
            m = int(ranked[rank, b])
            if m not in seen:
                selected.append(m)
                seen.add(m)
                if len(selected) >= n_slots:
                    break
        rank += 1
    return np.array(selected[:n_slots], dtype=int)


# ── Sort indices ──────────────────────────────────────────────────────────────
if HEATMAP_SORT == "coupling":
    top_mode_idx = _coupling_sort_idx(
        np.abs(r_eig), dom_mode, N_HEATMAP, valid_mask=_stable)
    top_neu_idx  = _coupling_sort_idx(
        np.abs(r_neu), dom_neu,  N_HEATMAP, valid_mask=None)
else:  # "timescale" — slowest STABLE modes / highest-τ neurons
    _tau_for_sort = np.nan_to_num(tau_eff_s, nan=-np.inf)
    top_mode_idx  = np.argsort(_tau_for_sort)[::-1][:N_HEATMAP]
    top_neu_idx   = np.argsort(tau_neu_s)[::-1][:N_HEATMAP]

# ── Task timescales per channel (time units) ──────────────────────────────────
if task_key == "flip_flop":
    _p = RC["p_pulse"]
    # Build the full-length list first, then select only displayed channels.
    _p_full = _p if isinstance(_p, list) else [_p] * _full_n_ch
    task_tau_s = np.array([1.0 / _p_full[i] * dt_val for i in _ch_display])  # (n_ch,)
    ch_labels  = [f"bit {i}" for i in _ch_display]
else:  # sine_wave
    _periods    = RC["periods"]
    _per_list   = _periods if isinstance(_periods, list) else [_periods]
    # Two channels (cos, sin) per frequency pair — same task timescale for both
    task_tau_s  = np.array([_per_list[ch // 2] * dt_val for ch in range(n_ch)])
    ch_labels   = [f"ch {ch}" for ch in range(n_ch)]

# ── Scatter-plot channel reduction ────────────────────────────────────────────
# For sine, cos/sin pairs share the same task τ and (after the modulation-amplitude
# fix) map to the same dominant mode, so we show one point per frequency rather
# than one per channel.
if task_key == "sine_wave":
    _scatter_ch_idx    = np.arange(0, n_ch, 2)                         # even = cos channels
    _scatter_ch_labels = [f"freq {i+1}" for i in range(len(_scatter_ch_idx))]
else:
    _scatter_ch_idx    = np.arange(n_ch)
    _scatter_ch_labels = ch_labels

# ── Shared colour-scale max for heatmaps (Figs 1 & 2) ────────────────────────
_hm_eig_vals = np.abs(r_eig[top_mode_idx, :])   # (N_HEATMAP, n_ch)
_hm_neu_vals = np.abs(r_neu[top_neu_idx,  :])   # (N_HEATMAP, n_ch)
shared_vmax  = max(_hm_eig_vals.max(), _hm_neu_vals.max())

# ── Heatmap x-tick labels (reflect sort mode) ────────────────────────────────
if HEATMAP_SORT == "coupling":
    # Which bit "claimed" each mode/neuron index (first bit whose dom points there)
    _eig_claimant: dict[int, int] = {}
    for _b in range(n_ch):
        _m = int(dom_mode[_b])
        if _m not in _eig_claimant:
            _eig_claimant[_m] = _b
    _neu_claimant: dict[int, int] = {}
    for _b in range(n_ch):
        _n = int(dom_neu[_b])
        if _n not in _neu_claimant:
            _neu_claimant[_n] = _b

    _eig_xtick_labels = []
    for _i in top_mode_idx:
        _tau_str = f"{tau_eff_s[_i]:.2f}s" if np.isfinite(tau_eff_s[_i]) else "|λ|≥1"
        _claim   = f"dom {ch_labels[_eig_claimant[_i]]}" if _i in _eig_claimant else "rank≥2"
        _eig_xtick_labels.append(f"{_claim}\n{_tau_str}")
    _neu_xtick_labels = []
    for _i in top_neu_idx:
        _tau_str = f"{tau_neu_s[_i]:.2f}s"
        _claim   = f"dom {ch_labels[_neu_claimant[_i]]}" if _i in _neu_claimant else "rank≥2"
        _neu_xtick_labels.append(f"{_claim}\n{_tau_str}")

    _eig_xlabel = "Eigenmode  (sorted by coupling strength, per bit)"
    _neu_xlabel = "Neuron  (sorted by coupling strength, per bit)"
else:  # "timescale"
    _eig_xtick_labels = [
        f"{tau_eff_s[_i]:.2f}s" if np.isfinite(tau_eff_s[_i]) else "|λ|≥1"
        for _i in top_mode_idx
    ]
    _neu_xtick_labels = [f"{tau_neu_s[_i]:.2f}s" for _i in top_neu_idx]
    _eig_xlabel = "Eigenmode  (τ_eff, sorted slowest → fastest)"
    _neu_xlabel = "Neuron  (τ, sorted slowest → fastest)"

# ── Diagnostics ───────────────────────────────────────────────────────────────
_n_unstable = int(np.isnan(tau_eff_s).sum())
_top6_taus  = tau_eff_s[top_mode_idx]
print(f"\nTask timescales (s):       {task_tau_s.round(3)}")
print(f"Scatter timescale:         {_scatter_timescale}")
print(f"Top-{N_HEATMAP} stable eigenmode τ_eff (s): "
      f"{np.array2string(_top6_taus, precision=2, suppress_small=False)}")
print(f"Unstable modes (|λ|≥1):    {_n_unstable} / {len(eigvals)} (excluded from ranking)")
print(f"Dominant mode indices:     {dom_mode}")
print(f"Scatter mode indices:      {dom_mode_scatter}")
print(f"Dom-mode τ_eff (s):        {tau_eff_s[dom_mode].round(3)}")
print(f"Scatter-mode τ (s):        {tau_scatter_s[dom_mode_scatter].round(3)}")
print(f"Dom-neuron τ (s):          {tau_neu_s[dom_neu].round(3)}")
print(f"Shared heatmap vmax:       {shared_vmax:.3f}")

# ── Figure helper ─────────────────────────────────────────────────────────────
_FIG_KEY = (
    f"{ANALYSIS_TASK}_{_gain_str}_{ANALYSIS_INIT}_{ANALYSIS_THETA}"
    f"_seed{BEST_SEED}_{SELECTION_METRIC}"
    f"_{COUPLING_METRIC}_{HEATMAP_SORT}_{_scatter_timescale}"
)

def _savefig(fig: plt.Figure, name: str) -> None:
    if SAVE_FIGS:
        p = os.path.join(FIGS_DIR, f"{name}_{_FIG_KEY}.pdf")
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"  saved → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 2b — Multi-seed scatter aggregation
# ══════════════════════════════════════════════════════════════════════════════
# For each seed in the current condition, compute the dominant-mode timescale
# (y3) and dominant-neuron timescale (y4) per display channel using the
# coupling metric set by COUPLING_METRIC.  Cached as .npy files.

def _compute_seed_scatter(
    seed_dir: str,
    *,
    coupling_metric: str,
    scatter_timescale: str,
    dt_val: float,
    ch_display: np.ndarray,
    n_ch: int,
    rc: dict,
    n_corr: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (y3, y4) scatter y-values for one seed, or None on failure.

    y3 : (n_ch,)  scatter timescale of the per-channel dominant eigenmode
    y4 : (n_ch,)  neuron τ of the per-channel dominant neuron
    Both arrays may contain NaN for channels with no valid stable mode.
    """
    try:
        _snap = _load_pt(os.path.join(seed_dir, "spectral_final.pt"))
        _eigvals_s = np.asarray(_snap["eigvals_eig"])   # (N_s,) complex
        _taus_s    = np.asarray(_snap["taus"])           # (N_s,)
        _V_s       = np.asarray(_snap["V"])              # (N_s, N_s) complex

        # ── Scatter timescale per mode ─────────────────────────────────────
        _log_abs_s  = np.log(np.clip(np.abs(_eigvals_s), 1e-12, None))
        _stable_s   = np.abs(_eigvals_s) < 1.0
        _denom_s    = np.where(_stable_s & (_log_abs_s < -1e-10), _log_abs_s, -1e-10)
        _tau_eff_sv = np.where(_stable_s, -1.0 / _denom_s * dt_val, np.nan)
        _arg_s      = np.abs(np.angle(_eigvals_s))
        _tau_osc_sv = np.where(_arg_s > 1e-10, 2.0 * np.pi / _arg_s * dt_val, np.nan)
        _tau_sc_s   = _tau_osc_sv if scatter_timescale == "oscillation" else _tau_eff_sv
        _sc_valid_s = _stable_s & np.isfinite(_tau_sc_s)

        # ── Coupling matrices (N_s, n_ch_full) ────────────────────────────
        if coupling_metric == "connectivity":
            _r_eig_f = np.asarray(_snap["coup_eig_re"]).T   # (N_s, n_ch_full)
            _r_neu_f = np.asarray(_snap["coup_neuron"]).T    # (N_s, n_ch_full)
        else:
            _model_s     = _build_model(rc, seed_dir)
            _h_s, _tf_s  = _run_fwd(_model_s, rc, n_corr)
            _V_inv_s     = np.linalg.pinv(_V_s.astype(np.complex128))
            _Z_cplx_s    = _h_s.astype(np.float64) @ _V_inv_s.T   # (M, N_s) complex
            _r_re_s      = _pearson_r(_Z_cplx_s.real.astype(np.float32), _tf_s)
            _r_im_s      = _pearson_r(_Z_cplx_s.imag.astype(np.float32), _tf_s)
            _r_eig_f     = np.sqrt(_r_re_s**2 + _r_im_s**2)        # modulation amplitude
            _r_neu_f     = _pearson_r(_h_s, _tf_s)     # (N_s, n_ch_full)
            if coupling_metric == "ablation":
                _snap_c  = np.asarray(_snap["coup_eig_re"])  # (n_ch_full, N_s)
                _snap_cn = np.asarray(_snap["coup_neuron"])  # (n_ch_full, N_s)
                _W_out_s = np.asarray(_snap["W_out"])         # (n_ch_full, N_s)
                _y_hat_s = _h_s @ _W_out_s.T
                _y_std_s = _y_hat_s.std(0).clip(1e-12)
                _z_std_s = _Z_s.std(0).clip(1e-12)
                _h_std_s = _h_s.std(0).clip(1e-12)
                _r_eig_f = (_z_std_s[:, None] * _snap_c.T / _y_std_s[None, :]).astype(np.float32)
                _r_neu_f = (_h_std_s[:, None] * _snap_cn.T / _y_std_s[None, :]).astype(np.float32)

        # ── Apply display channel filter ───────────────────────────────────
        _r_eig_d = _r_eig_f[:, ch_display]   # (N_s, n_ch)
        _r_neu_d = _r_neu_f[:, ch_display]   # (N_s, n_ch)

        # ── Dominant mode / neuron per channel ────────────────────────────
        _r_eig_st_d = np.where(_sc_valid_s[:, None], np.abs(_r_eig_d), 0.0)
        _dom_m = np.argmax(_r_eig_st_d, axis=0)   # (n_ch,)
        _dom_n = np.argmax(np.abs(_r_neu_d), axis=0)  # (n_ch,)

        _y3 = np.array([
            _tau_sc_s[_dom_m[ch]] if _sc_valid_s[_dom_m[ch]] else np.nan
            for ch in range(n_ch)
        ])
        _y4 = _taus_s[_dom_n]
        return _y3, _y4

    except Exception as _exc:
        print(f"  WARNING: skipping {seed_dir}: {_exc}")
        return None


if SCATTER_MULTI_SEED:
    # ── Cache ──────────────────────────────────────────────────────────────────
    _gain_str_ms = f"g{ANALYSIS_GAIN}".replace(".", "p")
    # Use "pearsonmod" in cache key to distinguish from old real-only Pearson files.
    _coup_cache_lbl = "pearsonmod" if COUPLING_METRIC == "pearson" else COUPLING_METRIC
    _MS_KEY = (
        f"{ANALYSIS_TASK}_{_gain_str_ms}_{ANALYSIS_INIT}_{ANALYSIS_THETA}"
        f"_multiseed_{_coup_cache_lbl}_{_scatter_timescale}"
        f"_nch{n_ch}_ncorr{N_CORR}"
    )
    _c_ms_y3 = os.path.join(CACHE_DIR, f"{_MS_KEY}_y3.npy")
    _c_ms_y4 = os.path.join(CACHE_DIR, f"{_MS_KEY}_y4.npy")

    if os.path.exists(_c_ms_y3) and os.path.exists(_c_ms_y4):
        print(f"Multi-seed cache hit — loading from {CACHE_DIR}")
        _y3_arr = np.load(_c_ms_y3)   # (n_seeds, n_ch)
        _y4_arr = np.load(_c_ms_y4)
    else:
        print("Multi-seed cache miss — computing per-seed coupling …")
        # Collect all seed directories for this condition
        _all_seed_dirs: list[str] = []
        for _sd_base in SWEEP_DIRS:
            _exp_base = os.path.join(_sd_base, EXP_NAME)
            if not os.path.isdir(_exp_base):
                continue
            for _entry in sorted(os.listdir(_exp_base)):
                if _entry.startswith("seed_"):
                    _all_seed_dirs.append(os.path.join(_exp_base, _entry))

        _y3_list, _y4_list = [], []
        for _sdir in _all_seed_dirs:
            print(f"  {_sdir} …", end=" ", flush=True)
            _res = _compute_seed_scatter(
                _sdir,
                coupling_metric=COUPLING_METRIC,
                scatter_timescale=_scatter_timescale,
                dt_val=dt_val,
                ch_display=_ch_display,
                n_ch=n_ch,
                rc=RC,
                n_corr=N_CORR,
            )
            if _res is not None:
                _y3_list.append(_res[0])
                _y4_list.append(_res[1])
                print("ok")
            else:
                print("skipped")

        if not _y3_list:
            print("WARNING: no seeds succeeded — multi-seed scatter disabled.")
            SCATTER_MULTI_SEED = False
            _y3_arr = _y4_arr = None
        else:
            _y3_arr = np.array(_y3_list)   # (n_seeds, n_ch)
            _y4_arr = np.array(_y4_list)
            np.save(_c_ms_y3, _y3_arr)
            np.save(_c_ms_y4, _y4_arr)
            print(f"  saved → {_c_ms_y3}, {_c_ms_y4}")

    if SCATTER_MULTI_SEED and _y3_arr is not None:
        _y3_mean = np.nanmean(_y3_arr, axis=0)   # (n_ch,)
        _y3_std  = np.nanstd(_y3_arr, axis=0)
        _y4_mean = np.nanmean(_y4_arr, axis=0)
        _y4_std  = np.nanstd(_y4_arr, axis=0)
        _n_seeds_ok = _y3_arr.shape[0]
        print(f"Multi-seed aggregation: {_n_seeds_ok} seeds, {n_ch} channels")
        print(f"  y3 mean: {np.round(_y3_mean, 3)}")
        print(f"  y4 mean: {np.round(_y4_mean, 3)}")
else:
    _y3_arr = _y4_arr = None


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 3 — Fig 1: Eigenmode coupling heatmap
# ══════════════════════════════════════════════════════════════════════════════
# Rows = output channels;  columns = top-K slowest eigenmodes.
# Cell colour = selected coupling metric between that eigenmode and output channel.
# Coloured rectangle outlines the globally dominant eigenmode per channel
# (only drawn when the dominant mode falls within the top-K displayed).

_hm_data = np.abs(r_eig[top_mode_idx, :]).T    # (n_ch, N_HEATMAP)

fig1, ax1 = plt.subplots(figsize=(max(N_HEATMAP * 1.15 + 1.4, 6), n_ch * 0.85 + 1.0))
im1 = ax1.imshow(_hm_data, cmap=C_HEATMAP, vmin=0, vmax=shared_vmax, aspect="equal")

ax1.set_xticks(range(N_HEATMAP))
ax1.set_xticklabels(_eig_xtick_labels, rotation=40, ha="right", fontsize=9)
ax1.set_yticks(range(n_ch))
ax1.set_yticklabels(ch_labels, fontsize=10)
ax1.set_xlabel(_eig_xlabel, fontsize=11)
ax1.set_ylabel("Output channel", fontsize=11)
ax1.set_title(
    f"Eigenmode coupling  ({COUPLING_METRIC})  —  "
    f"{ANALYSIS_TASK}, g={ANALYSIS_GAIN}, {ANALYSIS_INIT}, {ANALYSIS_THETA}",
    fontsize=11, fontweight="bold",
)

# Subtle cell borders via minor-tick grid lines at half-integer positions
ax1.set_xticks(np.arange(-0.5, N_HEATMAP, 1), minor=True)
ax1.set_yticks(np.arange(-0.5, n_ch, 1), minor=True)
ax1.grid(which="minor", color="black", linewidth=1, alpha=0.45)
ax1.tick_params(which="minor", length=0)

cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.035, pad=0.03)
cbar1.set_label(_COUP_LABEL, fontsize=10)

plt.tight_layout()
_savefig(fig1, "fig1_heatmap_eig")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 4 — Fig 2: Neuron coupling heatmap
# ══════════════════════════════════════════════════════════════════════════════
# Same layout and colour scale as Fig 1.
# Columns = top-K slowest neurons (by τ_i); coloured outlines = dominant neuron.

_hm_neu = np.abs(r_neu[top_neu_idx, :]).T    # (n_ch, N_HEATMAP)

fig2, ax2 = plt.subplots(figsize=(max(N_HEATMAP * 1.15 + 1.4, 6), n_ch * 0.85 + 1.0))
im2 = ax2.imshow(_hm_neu, 
cmap=C_HEATMAP, 
vmin=0, 
vmax=None, 
aspect="equal",
)

ax2.set_xticks(range(N_HEATMAP))
ax2.set_xticklabels(_neu_xtick_labels, rotation=40, ha="right", fontsize=9)
ax2.set_yticks(range(n_ch))
ax2.set_yticklabels(ch_labels, fontsize=10)
ax2.set_xlabel(_neu_xlabel, fontsize=11)
ax2.set_ylabel("Output channel", fontsize=11)

# Subtle cell borders
ax2.set_xticks(np.arange(-0.5, N_HEATMAP, 1), minor=True)
ax2.set_yticks(np.arange(-0.5, n_ch, 1), minor=True)
ax2.grid(which="minor", color="black", linewidth=1, alpha=0.45)
ax2.tick_params(which="minor", length=0)
ax2.set_title(
    f"Neuron coupling  ({COUPLING_METRIC})  —  "
    f"{ANALYSIS_TASK}, g={ANALYSIS_GAIN}, {ANALYSIS_INIT}, {ANALYSIS_THETA}",
    fontsize=11, fontweight="bold",
)

cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.035, pad=0.03)
cbar2.set_label(_COUP_LABEL, fontsize=10)

plt.tight_layout()
_savefig(fig2, "fig2_heatmap_neu")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 5+6 — Fig 3: Combined scatter  (eigenmode ● and neuron ▲)
# ══════════════════════════════════════════════════════════════════════════════
# Circles (●) : dominant eigenmode timescale (decay or oscillation per task)
# Triangles (▲): dominant neuron τ  (learned τ_i)
# Color encodes output channel / frequency.
# Dashed diagonal = perfect prediction.

_n_sc = len(_scatter_ch_idx)

# ── Eigenmode data ────────────────────────────────────────────────────────────
_x3 = task_tau_s[_scatter_ch_idx]                               # (n_sc,)
_y3 = tau_scatter_s[dom_mode_scatter[_scatter_ch_idx]]           # (n_sc,)

_use_ms3 = SCATTER_MULTI_SEED and _y3_arr is not None
if _use_ms3:
    _y3_arr_sc = _y3_arr[:, _scatter_ch_idx]                    # (n_seeds, n_sc)
    _y3_ctr    = np.nanmean(_y3_arr_sc, axis=0)
    _valid3    = np.isfinite(_x3) & np.isfinite(_y3_ctr)
    _sv3       = _y3_arr_sc[:, _valid3].ravel()
else:
    _y3_ctr = _y3
    _valid3 = np.isfinite(_x3) & np.isfinite(_y3_ctr)
    _sv3    = np.array([])

# ── Neuron data ───────────────────────────────────────────────────────────────
_x4 = task_tau_s[_scatter_ch_idx]                               # (n_sc,)
_y4 = tau_neu_s[dom_neu[_scatter_ch_idx]]                       # (n_sc,)

_use_ms4 = SCATTER_MULTI_SEED and _y4_arr is not None
if _use_ms4:
    _y4_arr_sc = _y4_arr[:, _scatter_ch_idx]                    # (n_seeds, n_sc)
    _y4_ctr    = np.nanmean(_y4_arr_sc, axis=0)
    _valid4    = np.isfinite(_x4) & np.isfinite(_y4_ctr)
    _sv4       = _y4_arr_sc[:, _valid4].ravel()
else:
    _y4_ctr = _y4
    _valid4 = np.isfinite(_x4) & np.isfinite(_y4_ctr)
    _sv4    = np.array([])

# ── Shared axis limits (union of all data) ────────────────────────────────────
_cands_34 = np.concatenate([
    _x3[_valid3], _y3_ctr[_valid3],
    _x4[_valid4], _y4_ctr[_valid4],
    _sv3[np.isfinite(_sv3)] if _sv3.size > 0 else np.array([]),
    _sv4[np.isfinite(_sv4)] if _sv4.size > 0 else np.array([]),
])
if _cands_34.size == 0:
    _lo34, _hi34 = 0.1, 10.0
else:
    _lo34 = _cands_34[np.isfinite(_cands_34)].min() * 0.8
    _hi34 = _cands_34[np.isfinite(_cands_34)].max() * 1.25

# ── Plot ──────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6.0, 6.0))

for _i in range(_n_sc):
    _col = C_CHANNEL[_scatter_ch_idx[_i] % len(C_CHANNEL)]

    # Eigenmode — circles (○/●)
    if _valid3[_i]:
        if _use_ms3:
            for _si in range(_y3_arr_sc.shape[0]):
                if np.isfinite(_y3_arr_sc[_si, _i]):
                    ax3.scatter(
                        _x3[_i], _y3_arr_sc[_si, _i],
                        marker="o", color=_col, s=25, alpha=0.35, zorder=4, linewidths=0,
                    )
            ax3.scatter(
                _x3[_i], _y3_ctr[_i],
                marker="o", color=_col, s=110, zorder=5,
                edgecolors="white", linewidths=1.0,
            )
        else:
            ax3.scatter(
                _x3[_i], _y3_ctr[_i],
                marker="o", color=_col, s=120, zorder=5, edgecolors="white", linewidths=0.8,
            )

    # Neuron — upward triangles (△/▲)
    if _valid4[_i]:
        if _use_ms4:
            for _si in range(_y4_arr_sc.shape[0]):
                if np.isfinite(_y4_arr_sc[_si, _i]):
                    ax3.scatter(
                        _x4[_i], _y4_arr_sc[_si, _i],
                        marker="^", color=_col, s=28, alpha=0.35, zorder=4, linewidths=0,
                    )
            ax3.scatter(
                _x4[_i], _y4_ctr[_i],
                marker="^", color=_col, s=110, zorder=5,
                edgecolors="white", linewidths=1.0,
            )
        else:
            ax3.scatter(
                _x4[_i], _y4_ctr[_i],
                marker="^", color=_col, s=120, zorder=5, edgecolors="white", linewidths=0.8,
            )

ax3.plot([_lo34, _hi34], [_lo34, _hi34], "k--", lw=1.2, alpha=0.55, zorder=1)

ax3.set_xlim(_lo34, _hi34)
ax3.set_ylim(_lo34, _hi34)
ax3.set_box_aspect(1)
if LOG_LOG_SCATTER:
    ax3.set_xscale("log")
    ax3.set_yscale("log")

ax3.set_xlabel("Task timescale (s)", fontsize=11)
ax3.set_ylabel("Dominant timescale (s)", fontsize=11)
ax3.set_title(
    f"Task τ  vs  dominant timescale\n"
    f"{ANALYSIS_TASK}, g={ANALYSIS_GAIN}, {ANALYSIS_INIT}, {ANALYSIS_THETA}",
    fontsize=11, fontweight="bold",
)

# ── Legend: channel colours + marker-type key ─────────────────────────────────
_leg_ch = [
    mpatches.Patch(
        color=C_CHANNEL[_scatter_ch_idx[_i] % len(C_CHANNEL)],
        label=_scatter_ch_labels[_i],
    )
    for _i in range(_n_sc)
]
_leg_types = [
    plt.Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="0.25", markeredgecolor="none",
        markersize=8, label=f"eigenmode  ({_scatter_ylabel.split('(')[0].strip()})",
    ),
    plt.Line2D(
        [0], [0], marker="^", color="none",
        markerfacecolor="0.25", markeredgecolor="none",
        markersize=8, label="neuron  τ_i",
    ),
]
ax3.legend(
    handles=_leg_ch + [mpatches.Patch(color="none", label="")] + _leg_types,
    fontsize=9,
    loc="upper left", bbox_to_anchor=(1.02, 1.0),
    framealpha=0.85, borderaxespad=0,
)

plt.tight_layout()
_savefig(fig3, "fig34_scatter_combined")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 7 — Fig 5: Jacobian spectrum in the complex plane
# ══════════════════════════════════════════════════════════════════════════════
# Untrained eigenvalues: light-blue dots.
# Trained eigenvalues:   dark-orange dots.
# Dominant eigenvalue per scatter channel + conjugate: open-circle ring.
# Dashed unit circle marks the boundary of stability.


def _conj_idx(eigs: np.ndarray, idx: int) -> int:
    """Index of eigenvalue closest to conj(eigs[idx]).  Returns idx for real modes."""
    ev = eigs[idx]
    if np.abs(ev.imag) < 1e-8:
        return idx
    dists      = np.abs(eigs - np.conj(ev))
    dists[idx] = np.inf
    return int(np.argmin(dists))

fig5, ax5 = plt.subplots(figsize=(5.5, 5.5))

# Unit circle
_uc_th = np.linspace(0, 2 * np.pi, 512)
ax5.plot(
    np.cos(_uc_th), np.sin(_uc_th),
    color=C_UNIT_CIRCLE, ls="--", lw=0.9, alpha=0.70, zorder=1,
)
ax5.axhline(0, color=C_UNIT_CIRCLE, lw=0.30, alpha=0.25, zorder=0)
ax5.axvline(0, color=C_UNIT_CIRCLE, lw=0.30, alpha=0.25, zorder=0)

# Untrained eigenvalues
ax5.scatter(
    eigvals_init.real, eigvals_init.imag,
    s=7, c=C_SPEC_UNTRAINED, alpha=0.50, edgecolors="none",
    label="untrained", zorder=2,
)

# Trained eigenvalues
ax5.scatter(
    eigvals.real, eigvals.imag,
    s=9, c=C_SPEC_TRAINED, alpha=0.65, edgecolors="none",
    label="trained", zorder=3,
)

# Dominant mode per scatter channel (one per frequency for sine) + conjugate partner.
# Open-circle rings: clean on paper, and both poles of a conjugate pair are shown.
_seen_dom5: set[int] = set()
for _i in range(_n_sc):
    _ch5      = int(_scatter_ch_idx[_i])
    _eig5     = int(dom_mode[_ch5])
    _conj5    = _conj_idx(eigvals, _eig5)
    _col5     = C_CHANNEL[_scatter_ch_idx[_i] % len(C_CHANNEL)]
    for _eidx5 in {_eig5, _conj5}:
        if _eidx5 not in _seen_dom5:
            ax5.scatter(
                eigvals[_eidx5].real, eigvals[_eidx5].imag,
                s=200, facecolors="none", edgecolors=_col5,
                linewidths=2.0, zorder=6,
            )
            _seen_dom5.add(_eidx5)

ax5.set_xlim(*SPECTRUM_XRANGE)
ax5.set_ylim(*SPECTRUM_YRANGE)
ax5.set_box_aspect(1)
ax5.set_xlabel("Re(λ)", fontsize=11)
ax5.set_ylabel("Im(λ)", fontsize=11)
ax5.set_title(
    f"Jacobian spectrum\n"
    f"{ANALYSIS_TASK}, g={ANALYSIS_GAIN}, {ANALYSIS_INIT}, {ANALYSIS_THETA}",
    fontsize=11, fontweight="bold",
)

# Full legend outside the axes — use Line2D dots (matches scatter appearance better)
_base_handles = [
    plt.Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor=C_SPEC_UNTRAINED, markeredgecolor="none",
        markersize=7, label="untrained eigenvalues",
    ),
    plt.Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor=C_SPEC_TRAINED, markeredgecolor="none",
        markersize=8, label="trained eigenvalues",
    ),
    plt.Line2D([0], [0], color=C_UNIT_CIRCLE, ls="--", lw=0.9, label="unit circle"),
]
_dom_handles = [
    plt.Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="none",
        markeredgecolor=C_CHANNEL[_scatter_ch_idx[_i] % len(C_CHANNEL)],
        markeredgewidth=2.0,
        markersize=10, label=f"dom. mode — {_scatter_ch_labels[_i]}",
    )
    for _i in range(_n_sc)
]
ax5.legend(
    handles=_base_handles + _dom_handles,
    fontsize=8,
    loc="upper left", bbox_to_anchor=(1.02, 1.0),
    framealpha=0.90, borderaxespad=0,
)

plt.tight_layout()
_savefig(fig5, "fig5_spectrum")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 7b — Fig 5b: Jacobian spectrum in the log-polar plane
# ══════════════════════════════════════════════════════════════════════════════
# x = −log|λ|   (decay rate, units: 1/timestep)
# y =  arg(λ)   (angular frequency, units: rad/timestep)
#
# The unit circle maps to the vertical line x = 0.
# Slow-decaying modes live on the left; fast-decaying on the right.
# Conjugate pairs are symmetric about y = 0.

def _to_logpolar(eigs):
    """Return (x, angular_freq) arrays for an array of eigenvalues.

    x is controlled by LOGPOLAR_XAXIS:
      "decay_rate" : −log|λ|       (1/timestep); unit circle ↔ x = 0
      "tau_eff"    : −1/log|λ|     (timesteps);  stable ↔ x > 0,
                     unit circle ↔ x → ±∞
    """
    _log_abs = np.log(np.clip(np.abs(eigs), 1e-12, None))   # log|λ|
    freq = np.angle(eigs)                                     # arg(λ)
    if LOGPOLAR_XAXIS == "tau_eff":
        # τ_eff = −1/log|λ| ;  |log|λ|| ≈ 0 (unit circle) → clip to large finite
        x = np.where(np.abs(_log_abs) > 1e-10,
                     -1.0 / _log_abs,
                     np.sign(-_log_abs + 1e-300) * 1e10)
    else:  # "decay_rate"
        x = -_log_abs
    return x, freq

_dr_init, _af_init = _to_logpolar(eigvals_init)
_dr_trn,  _af_trn  = _to_logpolar(eigvals)

fig5b, ax5b = plt.subplots(figsize=(6.5, 5.0))

# Stability boundary.
# "decay_rate": unit circle maps to x = 0  → draw vertical dashed line.
# "tau_eff"   : unit circle maps to x → ±∞  → no finite line; stable region is x > 0.
if LOGPOLAR_XAXIS == "decay_rate":
    ax5b.axvline(0, color=C_UNIT_CIRCLE, ls="--", lw=0.9, alpha=0.70, zorder=1,
                 label="unit circle (|λ|=1)")
ax5b.axhline(0, color=C_UNIT_CIRCLE, lw=0.30, alpha=0.25, zorder=0)

# Untrained
ax5b.scatter(
    _dr_init, _af_init,
    s=7, c=C_SPEC_UNTRAINED, alpha=0.50, edgecolors="none",
    label="untrained", zorder=2,
)

# Trained
ax5b.scatter(
    _dr_trn, _af_trn,
    s=9, c=C_SPEC_TRAINED, alpha=0.65, edgecolors="none",
    label="trained", zorder=3,
)

# Task reference lines: one pair of dashed horizontals per frequency
if task_key == "sine_wave":
    _periods  = RC["periods"]
    _per_list = _periods if isinstance(_periods, list) else [_periods]
    for _i in range(_n_sc):
        _ch5b = int(_scatter_ch_idx[_i])
        per   = _per_list[_ch5b // 2]
        omega_task = 2 * np.pi * dt_val / per
        _col5b = C_CHANNEL[_scatter_ch_idx[_i] % len(C_CHANNEL)]
        ax5b.axhline(
             omega_task, color=_col5b, ls=":", lw=1.2, alpha=0.6, zorder=1,
             label=f"task ω (T={per})",
        )
        ax5b.axhline(-omega_task, color=_col5b, ls=":", lw=1.2, alpha=0.6, zorder=1)

# Dominant mode per scatter channel + conjugate — open-circle rings
_seen_dom5b: set[int] = set()
for _i in range(_n_sc):
    _ch5b  = int(_scatter_ch_idx[_i])
    _eig5b = int(dom_mode[_ch5b])
    _conj5b = _conj_idx(eigvals, _eig5b)
    _col5b  = C_CHANNEL[_scatter_ch_idx[_i] % len(C_CHANNEL)]
    _first5b = True
    for _eidx5b in {_eig5b, _conj5b}:
        if _eidx5b not in _seen_dom5b:
            _lbl5b = f"dom. mode — {_scatter_ch_labels[_i]}" if _first5b else None
            ax5b.scatter(
                _dr_trn[_eidx5b], _af_trn[_eidx5b],
                s=200, facecolors="none", edgecolors=_col5b,
                linewidths=2.0, zorder=6, label=_lbl5b,
            )
            _seen_dom5b.add(_eidx5b)
            _first5b = False

if LOGPOLAR_XAXIS == "tau_eff":
    ax5b.set_xlabel(
        r"$\tau_{\mathrm{eff}} = -1/\log|\lambda|$  (timesteps;  stable: $x > 0$)",
        fontsize=11,
    )
else:
    ax5b.set_xlabel(r"Decay rate  $-\log|\lambda|$  (1/timestep)", fontsize=11)
ax5b.set_ylabel(r"Angular frequency  $\arg(\lambda)$  (rad/timestep)", fontsize=11)
ax5b.set_title(
    f"Jacobian spectrum (log-polar plane)\n"
    f"{ANALYSIS_TASK}, g={ANALYSIS_GAIN}, {ANALYSIS_INIT}, {ANALYSIS_THETA}",
    fontsize=11, fontweight="bold",
)
#ax5b.grid(True, alpha=0.15)

# ── Secondary axes: timescales ────────────────────────────────────────────────
if LOGPOLAR_XAXIS == "tau_eff":
    # Primary x is already τ_eff in timesteps → secondary just scales to time units
    _secax_top = ax5b.secondary_xaxis(
        "top",
        functions=(
            lambda x: x * dt_val,   # timesteps → time units
            lambda t: t / dt_val,   # time units → timesteps
        ),
    )
    _secax_top.set_xlabel(r"$\tau_{\mathrm{eff}}$  (time units)", fontsize=10)
else:
    # Primary x is decay rate → secondary shows τ_eff in time units
    _secax_top = ax5b.secondary_xaxis(
        "top",
        functions=(
            # decay rate → τ_eff:  clamp denominator so we never divide by ≈0
            lambda x: dt_val / np.clip(np.abs(x), 1e-9, None),
            # τ_eff → decay rate
            lambda t: dt_val / np.clip(np.abs(t), 1e-9, None),
        ),
    )
    _secax_top.set_xlabel(r"$\tau_{\mathrm{eff}}$  (time units)", fontsize=10)

# Right axis: τ_osc = 2π·dt / |arg(λ)|  (only meaningful for |y| > 0)
_secax_right = ax5b.secondary_yaxis(
    "right",
    functions=(
        # angular freq → τ_osc: clamp denominator so we never divide by ≈0
        lambda y: 2 * np.pi * dt_val / np.clip(np.abs(y), 1e-9, None),
        # τ_osc → angular freq
        lambda t: 2 * np.pi * dt_val / np.clip(np.abs(t), 1e-9, None),
    ),
)
_secax_right.set_ylabel(r"$\tau_{\mathrm{osc}}$  (time units)", fontsize=10)

# Legend
ax5b.legend(
    fontsize=8,
    loc="upper left", bbox_to_anchor=(1.18, 1.0),
    framealpha=0.90, borderaxespad=0,
)

plt.tight_layout()
_savefig(fig5b, "fig5b_spectrum_logpolar")
plt.show()