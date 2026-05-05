# %% [markdown]
# # Sine-Wave Generation -- Training-Horizon (Slow-Period Cycles) Sweep
#
# Sweep over `num_time_steps` with everything else fixed.  With `dt=1.0`
# and `periods=[20, 50, 100]`, the meaningful quantity being varied is
# the **number of cycles of the slowest target period** seen during
# training:  `n_cycles = num_time_steps * dt / T_max`.
#
# Hypothesis: more cycles seen -> more constraint on periodicity ->
# the parsimonious solution shifts from a Prony-style memorisation
# (mode interference inside the training window) toward true eigenvalue
# placement on the unit circle.  We expect a sharp onset of
# generalisation around 5-10 slow-period cycles.
#
# Caveats this notebook handles:
# - Different runs have different `num_time_steps` -> the raw MSE val_loss
#   is not directly comparable; we report normalised metrics ($R^2$).
# - Long-BPTT runs (T_4000) sometimes diverge during training; we detect
#   non-finite outputs and flag them.
# - Extended rollout uses a fixed physical time so we can compare
#   waveforms across runs apples-to-apples.

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_FIGS = True
FIGS_DIR = os.path.join("notebooks", "figs", "sine_wave_cycles_sweep")
os.makedirs(FIGS_DIR, exist_ok=True)

# %% Specify sweep directory
SWEEP_DIR = "/home/facosta/timescales/timescales/logs/experiments/sine_wave_hetero_cycles_sweep_20260418_002520"

# Diagnostic hyperparameters
EXTRAP_SLOW_CYCLES = 10      # rollout extends EXTRAP_SLOW_CYCLES * T_max past training
MIN_ROLLOUT_SLOW_CYCLES = 15 # but at least this many slow cycles total
EIG_RADIAL_TOL = 0.05        # |lambda| within (1-tol, 1+tol)
EIG_ANGULAR_TOL_FRAC = 0.05  # angular tolerance = tol_frac * (2pi/T_k)
PR_THRESH_FRAC = 1e-1        # |W_out V|_k cutoff for PR
DIVERGENCE_THRESHOLD = 1e6   # |output| above this counts as divergent

# %% Load sweep records (parses num_time_steps from "T_<n>" experiment names)
records = []
for exp_name in sorted(os.listdir(SWEEP_DIR)):
    exp_dir = os.path.join(SWEEP_DIR, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
        continue
    if not exp_name.startswith("T_"):
        continue
    try:
        T_train = int(exp_name.split("_")[1])
    except (IndexError, ValueError):
        continue

    for sdn in sorted(os.listdir(exp_dir)):
        if not sdn.startswith("seed_"):
            continue
        seed = int(sdn.split("_")[1])
        seed_path = os.path.join(exp_dir, sdn)

        rc_path = os.path.join(seed_path, "run_config.yaml")
        if not os.path.exists(rc_path):
            continue
        with open(rc_path) as f:
            rc = yaml.safe_load(f)

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

        records.append(dict(
            exp_name=exp_name, T_train=T_train, seed=seed, seed_path=seed_path,
            final_val_loss=fvl,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
            run_config=rc,
        ))

df = pd.DataFrame(records)
T_trains = sorted(df["T_train"].unique())
print(f"Loaded {len(df)} runs, T_train values: {T_trains}")

# %% Extract task parameters (these are fixed across the sweep)
first_config = df.iloc[0]["run_config"] if len(df) > 0 else {}
n_pairs = first_config.get("n_pairs", 1)
periods_cfg = first_config.get("periods", first_config.get("period", 10.0))
if isinstance(periods_cfg, (int, float)):
    periods_list = [float(periods_cfg)] * n_pairs
else:
    periods_list = [float(p) for p in periods_cfg]
n_channels = 2 * n_pairs
init_hidden_value = first_config.get("init_hidden_value", 1.0)
dt = first_config.get("dt", 1.0)
g_fixed = first_config.get("recurrent_gain", 1.0)
hidden_size = first_config.get("hidden_size", 128)
T_slow = max(periods_list)
T_fast = min(periods_list)

channel_labels = []
for k in range(n_pairs):
    channel_labels.append(f"cos (T={periods_list[k]:.0f})")
    channel_labels.append(f"sin (T={periods_list[k]:.0f})")
ch_colors = [f"C{i}" for i in range(n_channels)]

# Annotate the dataframe with derived quantities
df["n_cycles_slow"] = df["T_train"] * dt / T_slow
df["n_cycles_fast"] = df["T_train"] * dt / T_fast

print(f"g={g_fixed}, N={hidden_size}, n_pairs={n_pairs}, "
      f"periods={periods_list}, dt={dt}, init_hidden={init_hidden_value}")
print(f"Slow period T_max={T_slow}; cycles seen during training:")
for _, row in df.drop_duplicates("T_train").sort_values("T_train").iterrows():
    print(f"  T_train={row['T_train']:>5d}  ->  {row['n_cycles_slow']:>5.1f} slow cycles  "
          f"({row['n_cycles_fast']:.1f} fast)")

# %% Color palette -- one color per training-horizon
palette = plt.cm.viridis(np.linspace(0.15, 0.95, len(T_trains)))
COLORS = {T: palette[i] for i, T in enumerate(T_trains)}

def cycles_label(T):
    return f"{T*dt/T_slow:.1f} slow cyc (T_train={T})"

# %% Helpers
def _load_model(row):
    rc = row["run_config"]
    seed_path = row["seed_path"]
    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        return None, rc
    ckpt = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    n_p = rc.get("n_pairs", 1)
    model = RNN(
        input_size=rc.get("input_size", 1),
        hidden_size=rc["hidden_size"],
        output_size=2 * n_p,
        dt=rc["dt"],
        time_constants_config=rc.get("time_constants_config"),
        activation=getattr(nn, rc["activation"]),
        learn_time_constants=rc["learn_time_constants"],
        init_time_constant=rc.get("init_time_constant"),
        shared_time_constant=rc["shared_time_constant"],
        normalize_hidden=rc["normalize_hidden"],
        zero_diag_wrec=rc["zero_diag_wrec"],
        recurrent_gain=rc["recurrent_gain"],
        noise_std=0.0,
        wrec_init=rc["wrec_init"],
        alpha_parameterization=rc["alpha_parameterization"],
        dynamics_type=rc["dynamics_type"],
    )
    lit = RNNLightning(
        model=model, learning_rate=rc["learning_rate"],
        weight_decay=rc["weight_decay"],
        step_size=rc.get("lr_step_size", rc.get("step_size", 1000)),
        gamma=rc["gamma"], task="sine_wave",
        init_hidden_value=rc.get("init_hidden_value", 1.0),
    )
    lit.load_state_dict(sd)
    lit.eval()
    return model, rc


def _extract_weights(seed_path, which="trained"):
    """
    Load (W_rec, W_out) from a checkpoint.

    `which` ∈ {"trained", "untrained"}.
    For "trained" we pick the first best-model-*.ckpt.
    For "untrained" we look for checkpoints/untrained.ckpt; returns (None, None)
    if it is not present (so callers can gracefully skip the baseline).
    """
    if which == "untrained":
        ckpt_path = os.path.join(seed_path, "checkpoints", "untrained.ckpt")
        if not os.path.exists(ckpt_path):
            return None, None
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            return None, None
    else:
        best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints",
                                            "best-model-*.ckpt"))
        if not best_ckpts:
            return None, None
        ckpt = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    W_rec = W_out = None
    for key, val in sd.items():
        if "W_rec.weight" in key:
            W_rec = val.numpy()
        elif "W_out.weight" in key or "readout.weight" in key:
            W_out = val.numpy()
    return W_rec, W_out


def _safe_run(model, T_run, h_init):
    """Run model autonomously; return numpy output and a 'diverged' flag."""
    dummy_inp = torch.zeros(1, T_run, 1)
    with torch.no_grad():
        _, out = model(dummy_inp, init_hidden=h_init)
    out_np = out[0].numpy()
    diverged = (not np.all(np.isfinite(out_np))) or (np.abs(out_np).max() > DIVERGENCE_THRESHOLD)
    return out_np, diverged


def _make_targets(T_run, dt_val):
    t_ax = np.arange(T_run) * dt_val
    targets = np.zeros((T_run, 2 * n_pairs))
    for k in range(n_pairs):
        phase = 2.0 * np.pi * t_ax / periods_list[k]
        targets[:, 2 * k] = np.cos(phase)
        targets[:, 2 * k + 1] = np.sin(phase)
    return t_ax, targets

# %% [markdown]
# ## 1. Training curves
#
# Note: the raw MSE val_loss is computed over the training window, whose
# length differs across runs.  We additionally plot the validation $R^2$
# (normalised) which IS comparable across runs.

# %% Plot 1: Validation loss vs training step
fig, ax = plt.subplots(figsize=(10, 5))
plotted = set()
for _, row in df.iterrows():
    T = row["T_train"]
    vl = row["val_losses"]
    st = row["steps"][:len(vl)] if row["steps"] else list(range(1, len(vl) + 1))
    label = cycles_label(T) if T not in plotted else None
    ax.plot(st, vl, linewidth=1.4, color=COLORS[T], alpha=0.8, label=label)
    plotted.add(T)
ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation loss (MSE, train window)", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="upper right")
fig.suptitle("Validation Loss by Training Horizon",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "val_loss.pdf"), bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 2: Validation R^2 vs training step
fig, ax = plt.subplots(figsize=(10, 5))
plotted = set()
for _, row in df.iterrows():
    T = row["T_train"]
    va = row["val_accs"]
    st = row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1))
    if not va:
        continue
    label = cycles_label(T) if T not in plotted else None
    ax.plot(st, va, linewidth=1.4, color=COLORS[T], alpha=0.8, label=label)
    plotted.add(T)
ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation $R^2$ (train window)", fontsize=12)
ax.set_ylim(-0.5, 1.05)
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="lower right")
fig.suptitle("Validation $R^2$ by Training Horizon",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "val_r2.pdf"), bbox_inches="tight", dpi=150)
plt.show()

# %% P_TRAJ  Example output trajectories — 2 selected frequencies
# ─────────────────────────────────────────────────────────────────────────────
# For a chosen trained network, show the network output y(t) and the target
# y*(t) for 2 frequency channels (one slower, one faster), over a short window
# of a few periods.  Analogue of the flip-flop P03b cell.
#
# No input signal (sine-wave task is autonomous after initial condition).
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ────────────────────────────────────────────────────────────────────
PTRAJ_T_TRAIN       = None    # which T_train to use (None → best-loss run)
PTRAJ_TARGET_PERIODS = None   # [T_fast, T_slow], or None → auto (min & max period)
PTRAJ_N_PERIODS_SHOW = 4      # how many cycles of the SLOWER channel to display
PTRAJ_SAVE          = True

_C_TARGET_SW = "#DA627D"   # y*(t) — deep rose
_C_OUTPUT_SW = "#FFA5AB"   # y(t)  — salmon pink

# ── Select run ────────────────────────────────────────────────────────────────
_ptraj_T = PTRAJ_T_TRAIN
if _ptraj_T is None or _ptraj_T not in T_trains:
    # Pick the T_train with best (lowest) validation loss
    _ptraj_T = df.loc[df["final_val_loss"].idxmin(), "T_train"]
print(f"P_TRAJ: using T_train = {_ptraj_T}")

_ptraj_row   = df[df["T_train"] == _ptraj_T]
_ptraj_row   = _ptraj_row.loc[_ptraj_row["final_val_loss"].idxmin()]
_ptraj_model, _ptraj_rc = _load_model(_ptraj_row)

# ── Select frequency channels ─────────────────────────────────────────────────
# Default: pick the pair with the longest period and the pair with the shortest
_target_periods_ptraj = (
    PTRAJ_TARGET_PERIODS if PTRAJ_TARGET_PERIODS is not None
    else [T_fast, T_slow]
)
_sel_pairs_ptraj = [
    int(np.argmin([abs(p - tp) for p in periods_list]))
    for tp in _target_periods_ptraj
]
_sel_pairs_ptraj = list(dict.fromkeys(_sel_pairs_ptraj))   # de-duplicate

# ── Generate target + network output ─────────────────────────────────────────
_dt_ptraj  = _ptraj_rc["dt"]
_T_show_ptraj = int(PTRAJ_N_PERIODS_SHOW * max(_target_periods_ptraj) / _dt_ptraj)
_t_ax_ptraj, _tgt_ptraj = _make_targets(_T_show_ptraj, _dt_ptraj)

if _ptraj_model is not None:
    _h0_ptraj   = torch.full((1, _ptraj_rc["hidden_size"]),
                             _ptraj_rc.get("init_hidden_value", 1.0))
    _out_ptraj, _div_ptraj = _safe_run(_ptraj_model, _T_show_ptraj, _h0_ptraj)
else:
    _out_ptraj, _div_ptraj = None, True

# ── Plot ──────────────────────────────────────────────────────────────────────
_n_panels_ptraj = len(_sel_pairs_ptraj)
figptraj, axesptraj = plt.subplots(
    _n_panels_ptraj, 1,
    figsize=(13, 2.6 * _n_panels_ptraj),
    sharex=True,
)
if _n_panels_ptraj == 1:
    axesptraj = [axesptraj]

for _row_i, _pair_k in enumerate(_sel_pairs_ptraj):
    ax = axesptraj[_row_i]
    _ic, _is = 2 * _pair_k, 2 * _pair_k + 1
    _T_k = periods_list[_pair_k]

    # Target: cosine component (representative of the pair)
    ax.plot(_t_ax_ptraj, _tgt_ptraj[:, _ic],
            color=_C_TARGET_SW, linewidth=1.8,
            label="$y^*(t)$ — target" if _row_i == 0 else None, zorder=4)

    # Network output
    if not _div_ptraj and _out_ptraj is not None:
        ax.plot(_t_ax_ptraj, _out_ptraj[:, _ic],
                color=_C_OUTPUT_SW, linewidth=1.4,
                label="$y(t)$ — network output" if _row_i == 0 else None,
                zorder=3)

    # Horizontal baseline
    ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)

    # Period markers: light vertical lines every T_k
    for _cyc in range(1, PTRAJ_N_PERIODS_SHOW + 1):
        ax.axvline(_cyc * _T_k, color="#dddddd", linewidth=0.8,
                   linestyle="--", zorder=0)

    ax.set_ylim(-1.45, 1.45)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["-1", "0", "1"], fontsize=10)
    ax.set_ylabel(
        f"Amplitude\n$T = {_T_k:.0f}$",
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.12, color="#bbbbbb")

axesptraj[-1].set_xlabel(f"Time  (dt = {_dt_ptraj})", fontsize=11)

# Legend — outside to the right
from matplotlib.lines import Line2D as _LDptraj
figptraj.legend(
    handles=[
        _LDptraj([0], [0], color=_C_TARGET_SW, lw=2.2,
                 label="$y^*(t)$ — target"),
        _LDptraj([0], [0], color=_C_OUTPUT_SW, lw=1.6,
                 label="$y(t)$ — network output"),
    ],
    fontsize=10,
    loc="upper left",
    bbox_to_anchor=(1.01, 0.97),
    framealpha=0.9,
    borderaxespad=0,
)

figptraj.suptitle(
    f"Sine-Wave Generation — Example Output  "
    f"(T_train = {_ptraj_T},  g = {g_fixed:.2f})",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
if PTRAJ_SAVE:
    figptraj.savefig(
        os.path.join(FIGS_DIR, f"P_traj_example_T{_ptraj_T}.pdf"),
        bbox_inches="tight", dpi=150,
    )
plt.show()


# %% [markdown]
# ## 2. Per-run diagnostics
#
# For each (T_train, seed) we compute:
# - Spectral radius `rho(J)` and # eigenvalues with `|lambda| > 1`
# - Modes-found scorecard at each target frequency
# - Readout participation ratio
# - Extrapolation $R^2$ on a fixed-physical-time rollout

# %% Diagnostic functions
def compute_eig_scorecard(eigs, periods, dt_val,
                          radial_tol=EIG_RADIAL_TOL,
                          angular_tol_frac=EIG_ANGULAR_TOL_FRAC):
    abs_eigs = np.abs(eigs)
    angles = np.angle(eigs)
    per_pair = np.zeros(len(periods), dtype=int)
    matched = np.zeros(len(eigs), dtype=bool)
    for k, T in enumerate(periods):
        omega_dt = 2.0 * np.pi * dt_val / T
        ang_tol = angular_tol_frac * omega_dt
        for sign in (+1, -1):
            target_ang = sign * omega_dt
            ang_diff = np.abs(((angles - target_ang + np.pi) % (2 * np.pi)) - np.pi)
            cand = ((~matched)
                    & (np.abs(abs_eigs - 1.0) < radial_tol)
                    & (ang_diff < ang_tol))
            if cand.any():
                cand_idx = np.where(cand)[0]
                d = np.sqrt((abs_eigs[cand_idx] - 1.0) ** 2
                            + ang_diff[cand_idx] ** 2)
                pick = cand_idx[np.argmin(d)]
                matched[pick] = True
                per_pair[k] += 1
    return per_pair, int(per_pair.sum())


def compute_readout_participation(W_out, V, thresh_frac=PR_THRESH_FRAC):
    M = W_out @ V
    s = np.linalg.norm(M, axis=0)
    if s.max() <= 0:
        return 0.0
    s = s.copy()
    s[s < thresh_frac * s.max()] = 0.0
    num = s.sum() ** 2
    den = (s ** 2).sum()
    return float(num / max(den, 1e-30))


def windowed_r2(targets, output, mask):
    if not mask.any():
        return np.nan
    t = targets[mask]
    p = output[mask]
    if not np.all(np.isfinite(p)):
        return np.nan
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean(axis=0)) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)

# %% Compute diagnostics
metrics = []
for _, row in df.iterrows():
    seed_path = row["seed_path"]
    rc = row["run_config"]
    T_train = row["T_train"]
    dt_rc = rc["dt"]

    # Rollout long enough to give EXTRAP_SLOW_CYCLES past training,
    # but at least MIN_ROLLOUT_SLOW_CYCLES total.
    extrap_steps = int(EXTRAP_SLOW_CYCLES * T_slow / dt_rc)
    rollout_steps = max(T_train + extrap_steps,
                        int(MIN_ROLLOUT_SLOW_CYCLES * T_slow / dt_rc))

    W_rec, W_out = _extract_weights(seed_path)
    if W_rec is None:
        continue
    N_dim = W_rec.shape[0]
    tau_val = rc["time_constants_config"]["values"][0]
    alpha = 1.0 - np.exp(-dt_rc / tau_val)
    g_rc = rc["recurrent_gain"]
    J = (1 - alpha) * np.eye(N_dim) + alpha * g_rc * W_rec
    eigs, V = np.linalg.eig(J)
    rho = float(np.abs(eigs).max())
    n_unstable = int((np.abs(eigs) > 1.0).sum())

    per_pair_found, n_found = compute_eig_scorecard(eigs, periods_list, dt_rc)
    pr = compute_readout_participation(W_out, V, thresh_frac=0.9) if W_out is not None else np.nan

    model, _ = _load_model(row)
    r2_in = r2_out = np.nan
    diverged = False
    if model is not None:
        h_init = torch.full((1, rc["hidden_size"]),
                            rc.get("init_hidden_value", 1.0))
        out_np, diverged = _safe_run(model, rollout_steps, h_init)
        _, targets = _make_targets(rollout_steps, dt_rc)
        in_mask = np.arange(rollout_steps) < T_train
        out_mask = np.arange(rollout_steps) >= T_train
        if not diverged:
            r2_in = windowed_r2(targets, out_np, in_mask)
            r2_out = windowed_r2(targets, out_np, out_mask)

    metrics.append(dict(
        T_train=T_train,
        n_cycles_slow=T_train * dt_rc / T_slow,
        seed=row["seed"],
        seed_path=seed_path,
        rho=rho, n_unstable=n_unstable,
        n_found=n_found, n_target=2 * n_pairs,
        per_pair_found=per_pair_found.tolist(),
        readout_pr=pr,
        r2_in=r2_in, r2_out=r2_out,
        diverged=diverged,
        final_val_loss=row["final_val_loss"],
    ))
    div_str = " [DIVERGED]" if diverged else ""
    print(f"T_train={T_train:>5d} ({T_train*dt_rc/T_slow:>5.1f} cyc): "
          f"rho={rho:.4f}, n_unstable={n_unstable}, "
          f"modes_found={n_found}/{2*n_pairs}, PR={pr:.1f}, "
          f"R2_in={r2_in:.3f}, R2_out={r2_out:.3f}{div_str}")

metrics_df = pd.DataFrame(metrics).sort_values("T_train").reset_index(drop=True)

# %% [markdown]
# ## 3. Summary panel: diagnostics vs slow-period cycles
#
# Primary x-axis = number of cycles of the slowest target period seen
# during training.  Secondary tick labels show the raw `num_time_steps`.
# The expected story (if the cycle hypothesis holds) is:
# - few cycles -> high in-window $R^2$, near-zero post-training $R^2$,
#   readout PR much greater than 2N_pairs, modes_found ~ 0 (memorisation).
# - many cycles -> in- and out-of-window $R^2$ both ~1, readout PR ~ 2N_pairs,
#   modes_found = 2N_pairs (placement).

# %%
def _cycles_axis_decoration(ax):
    """Add secondary x-axis showing raw num_time_steps."""
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([T * dt / T_slow for T in T_trains])
    ax2.set_xticklabels([str(T) for T in T_trains], fontsize=8)
    ax2.set_xlabel("num_time_steps", fontsize=9, color="gray")
    ax2.tick_params(axis="x", colors="gray")

x = metrics_df["n_cycles_slow"].values
divergent_mask = metrics_df["diverged"].values

fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

# --- final val loss (raw MSE, only meaningful within a row) ---
ax = axes[0, 0]
ax.plot(x[~divergent_mask], metrics_df["final_val_loss"].values[~divergent_mask],
        "o-", color="#264653")
if divergent_mask.any():
    ax.plot(x[divergent_mask], np.full(divergent_mask.sum(), np.nan), "x",
            color="red", markersize=12, label="diverged")
ax.set_ylabel("Final val loss (MSE)")
ax.set_yscale("log")
ax.set_title("Training-window loss (raw MSE)")
ax.grid(True, alpha=0.3)

# --- R^2 in/out of training window ---
ax = axes[0, 1]
ax.plot(x, metrics_df["r2_in"].values, "o-", color="#2a9d8f", label="train window")
ax.plot(x, metrics_df["r2_out"].values, "s--", color="#e76f51",
        label="post-training")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6)
ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.6)
ax.set_ylabel("$R^2$")
ax.set_ylim(-1.5, 1.1)
ax.set_title(f"Generalisation\n(extrapolate {EXTRAP_SLOW_CYCLES} slow cycles past train)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- spectral radius ---
ax = axes[0, 2]
ax.plot(x, metrics_df["rho"].values, "o-", color="#264653")
ax.axhline(1.0, color="red", linestyle=":", linewidth=1, alpha=0.7,
           label="unit circle")
ax.set_ylabel(r"$\rho(J) = \max_k |\lambda_k|$")
ax.set_title("Spectral radius")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- # unstable modes ---
ax = axes[1, 0]
ax.plot(x, metrics_df["n_unstable"].values, "o-", color="#c1121f")
ax.set_ylabel(r"# eigenvalues with $|\lambda| > 1$")
ax.set_title("Unstable modes (memorisation signature)")
ax.grid(True, alpha=0.3)

# --- modes-found scorecard ---
ax = axes[1, 1]
ax.plot(x, metrics_df["n_found"].values, "o-", color="#2a9d8f")
ax.axhline(2 * n_pairs, color="black", linestyle=":", linewidth=1,
           alpha=0.7, label=f"target = {2*n_pairs}")
ax.set_ylabel("# eigenvalues near targets")
ax.set_title(f"Eigenvalue scorecard (rad tol={EIG_RADIAL_TOL}, "
             f"ang frac={EIG_ANGULAR_TOL_FRAC})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- readout PR ---
ax = axes[1, 2]
ax.plot(x, metrics_df["readout_pr"].values, "o-", color="#264653")
ax.axhline(2 * n_pairs, color="green", linestyle=":", linewidth=1, alpha=0.7,
           label=f"placement = {2*n_pairs}")
ax.set_ylabel("Readout PR")
ax.set_yscale("log")
ax.set_title("Readout participation ratio (modes used)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel("Slow-period cycles in training window")
    ax.set_xscale("log")
    _cycles_axis_decoration(ax)

fig.suptitle(f"Cycles Sweep -- Diagnostics  (g={g_fixed}, N={hidden_size}, "
             f"dt={dt}, periods={periods_list})",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "diagnostics_summary.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Extended rollout -- waveforms on a common physical-time axis
#
# All runs are simulated for the same physical time horizon
# `max(T_train * dt, MIN_ROLLOUT_SLOW_CYCLES * T_slow)` so they can be
# compared apples-to-apples.
#
# Layout: **rows = output pair (frequency), columns = network (T_train)**.
# Set `ROLLOUT_T_FOCUS` to pick which networks to show (or leave as `None` to
# show them all).  When only one T_train is chosen, the figure collapses to a
# single column with one row per frequency, which makes per-pair behaviour
# easy to read.
#
# In each cell:
#   - dashed grey  = target  (cos and sin)
#   - solid color  = network output  (cos = solid pair colour, sin = dashed)
#   - vertical dotted black line = end of training horizon for that network

# %%
# Pick which T_train values to plot.  Set to None for "all".
ROLLOUT_T_FOCUS = None    # e.g. [500]   or   [100, 500, 2000]

# Per-pair colour palette consistent with the T_FOCUS section (Cn cycler)
_pair_colors_R = [f"C{k}" for k in range(n_pairs)]

_T_show = list(T_trains) if ROLLOUT_T_FOCUS is None else \
    [T for T in ROLLOUT_T_FOCUS if T in T_trains]
if not _T_show:
    print(f"None of ROLLOUT_T_FOCUS={ROLLOUT_T_FOCUS} found in T_trains={T_trains}.")
else:
    common_T_phys = max(max(_T_show) * dt,
                        MIN_ROLLOUT_SLOW_CYCLES * T_slow)
    print(f"Common physical rollout horizon: {common_T_phys:.0f} time units "
          f"({common_T_phys / T_slow:.1f} slow cycles)")
    print(f"Showing T_train ∈ {_T_show}, one row per output pair "
          f"({n_pairs} pairs)")

    n_rows_R, n_cols_R = n_pairs, len(_T_show)
    fig, axes_R = plt.subplots(
        n_rows_R, n_cols_R,
        figsize=(max(7.5 * n_cols_R, 7.5), 1.7 * n_rows_R + 0.6),
        squeeze=False, sharex=True, sharey=True,
    )

    for ci, T in enumerate(_T_show):
        sub = df[df["T_train"] == T]
        row = sub.loc[sub["final_val_loss"].idxmin()]
        model, rc = _load_model(row)

        if model is None:
            for ri in range(n_rows_R):
                ax = axes_R[ri, ci]
                ax.text(0.5, 0.5, "no checkpoint",
                        ha="center", va="center", transform=ax.transAxes)
            continue

        dt_rc = rc["dt"]
        rollout_steps = int(common_T_phys / dt_rc)
        h_init = torch.full((1, rc["hidden_size"]),
                            rc.get("init_hidden_value", 1.0))
        out_np, diverged = _safe_run(model, rollout_steps, h_init)
        t_ax, targets = _make_targets(rollout_steps, dt_rc)
        train_T_phys = T * dt_rc

        for k in range(n_pairs):
            ax  = axes_R[k, ci]
            col = _pair_colors_R[k]
            i_c, i_s = 2 * k, 2 * k + 1

            # Targets first (faint dashed grey, both cos & sin)
            ax.plot(t_ax, targets[:, i_c], color="#888", linestyle="--",
                    linewidth=0.7, alpha=0.7,
                    label="target cos" if (k == 0 and ci == 0) else None)
            ax.plot(t_ax, targets[:, i_s], color="#bbb", linestyle="--",
                    linewidth=0.7, alpha=0.7,
                    label="target sin" if (k == 0 and ci == 0) else None)

            # Network output: cos = solid pair colour, sin = dashed pair colour
            if not diverged:
                ax.plot(t_ax, out_np[:, i_c], color=col, linewidth=1.4,
                        alpha=0.95,
                        label="net cos" if (k == 0 and ci == 0) else None)
                ax.plot(t_ax, out_np[:, i_s], color=col, linewidth=1.4,
                        alpha=0.95, linestyle=(0, (4, 2)),
                        label="net sin" if (k == 0 and ci == 0) else None)

            # Training-horizon marker (per network)
            ax.axvline(train_T_phys, color="black", linestyle=":",
                       linewidth=1.3, alpha=0.7,
                       label="train horizon" if (k == 0 and ci == 0) else None)

            # Cosmetics
            ax.set_ylim(-1.6, 1.6)
            ax.grid(True, alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # y-axis label (leftmost column): pair frequency, coloured
            if ci == 0:
                ax.set_ylabel(f"T = {periods_list[k]:.0f}\n"
                              f"({common_T_phys/periods_list[k]:.1f} cyc visible)",
                              fontsize=9, fontweight="bold",
                              color=col)

            # Title at top of column (top row): network info
            if k == 0:
                title = f"T_train = {T}  ({T*dt/T_slow:.1f} slow cyc)"
                if diverged:
                    title += "   [DIVERGED]"
                ax.set_title(title, fontsize=10, fontweight="bold",
                             color="#c0392b" if diverged else "black")

            # Shade divergent runs for the whole column
            if diverged:
                ax.set_facecolor("#fff0f0")

        # Single legend in upper-right of the very first cell
        if ci == 0:
            axes_R[0, 0].legend(fontsize=7, loc="upper right",
                                 ncol=2, framealpha=0.85)

    for ax in axes_R[-1, :]:
        ax.set_xlabel(f"Time (slow period T_max = {T_slow:.0f})", fontsize=11)

    fig.suptitle(
        f"Extended Rollout — per-frequency view  "
        f"(common horizon = {common_T_phys/T_slow:.1f} slow cycles)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if SAVE_FIGS:
        suffix = "all" if ROLLOUT_T_FOCUS is None else \
            "_".join(str(T) for T in _T_show)
        fig.savefig(os.path.join(FIGS_DIR, f"extended_rollout_perpair_{suffix}.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()

# %% [markdown]
# ## 5. Eigenspectra
#
# Coloured "+" markers = target eigenvalues $e^{\pm i\, 2\pi\, dt/T_k}$
# on the unit circle.  Modes coloured by their dominant coupling to each
# (cos, sin) output pair.  Since `dt` is fixed across the sweep, the
# target marker positions are identical in every panel.

# %%
theta_circle = np.linspace(0, 2 * np.pi, 200)
pair_colors = ["#2a9d8f", "#e76f51", "#264653",
               "#a855f7", "#f59e0b", "#0ea5e9"][:n_pairs]

ncols_eig = min(3, len(T_trains))
nrows_eig = (len(T_trains) + ncols_eig - 1) // ncols_eig
fig, axes = plt.subplots(nrows_eig, ncols_eig,
                         figsize=(5.5 * ncols_eig, 5.5 * nrows_eig),
                         squeeze=False)

for si, T in enumerate(T_trains):
    ax = axes[si // ncols_eig, si % ncols_eig]
    sub = df[df["T_train"] == T]
    row = sub.loc[sub["final_val_loss"].idxmin()]
    seed_path = row["seed_path"]
    rc = row["run_config"]
    W_rec, W_out = _extract_weights(seed_path)
    if W_rec is None:
        ax.set_visible(False)
        continue
    N_dim = W_rec.shape[0]
    dt_rc = rc["dt"]
    tau_val = rc["time_constants_config"]["values"][0]
    alpha = 1.0 - np.exp(-dt_rc / tau_val)
    g_rc = rc["recurrent_gain"]
    J = (1 - alpha) * np.eye(N_dim) + alpha * g_rc * W_rec
    eigs, V = np.linalg.eig(J)

    ax.plot(np.cos(theta_circle), np.sin(theta_circle),
            "k-", linewidth=0.6, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)

    if W_out is not None:
        coupling_full = np.abs(W_out @ V)
        pair_coupling = np.zeros((n_pairs, len(eigs)))
        for k in range(n_pairs):
            pair_coupling[k] = coupling_full[2*k] + coupling_full[2*k + 1]

        highlighted = set()
        eig_pair_assignment = {}
        for k in range(n_pairs):
            top_for_pair = np.argsort(pair_coupling[k])[-2:]
            for idx in top_for_pair:
                if idx not in highlighted:
                    highlighted.add(idx)
                    eig_pair_assignment[idx] = k

        bulk_mask = np.array([i not in highlighted for i in range(len(eigs))])
        ax.scatter(eigs[bulk_mask].real, eigs[bulk_mask].imag,
                   s=6, color="#ccc", alpha=0.5, edgecolors="none",
                   rasterized=True, zorder=2)
        for k in range(n_pairs):
            idxs = np.array([i for i, pk in eig_pair_assignment.items()
                             if pk == k])
            if len(idxs) == 0:
                continue
            ax.scatter(eigs[idxs].real, eigs[idxs].imag, s=80,
                       color=pair_colors[k], alpha=0.85,
                       edgecolors="black", linewidths=0.4, zorder=5)
    else:
        ax.scatter(eigs.real, eigs.imag, s=6, color="#ccc",
                   alpha=0.5, edgecolors="none", rasterized=True)

    for k in range(n_pairs):
        omega_dt = 2.0 * np.pi * dt_rc / periods_list[k]
        for sign in (+1, -1):
            ax.plot(np.cos(sign * omega_dt), np.sin(sign * omega_dt), "+",
                    color=pair_colors[k], markersize=14, markeredgewidth=2.5,
                    zorder=10)

    rho = np.max(np.abs(eigs))
    n_unst = int((np.abs(eigs) > 1.0).sum())
    title = f"{cycles_label(T)}\nrho={rho:.3f}, |λ|>1: {n_unst}"
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.15)

for i in range(len(T_trains), nrows_eig * ncols_eig):
    axes[i // ncols_eig, i % ncols_eig].set_visible(False)

fig.suptitle("Global Jacobian Eigenspectra by Training Horizon",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "eigenspectra.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Eigenvalue coupling distribution (rows = T_train, cols = target pair)
#
# Each subplot shows ALL eigenvalues of $J$ in the complex plane, coloured
# by the pair-coupling strength
# $\;c_{k,m} = |W_\text{out} V|_{2k,m} + |W_\text{out} V|_{2k+1,m}$
# of mode $m$ to the column's target pair $k$.  Each panel is normalised
# to its own max (printed in the title) so we can read off the shape of
# the coupling distribution per cell:
# - **placement**: a few bright dots near the coloured "+" target on the
#   unit circle, and a dark/blue bulk everywhere else.
# - **memorisation**: many medium-bright dots scattered across the disk,
#   with no clear concentration near the target.
#
# Modes are drawn in ascending order of coupling so high-coupling dots sit
# on top of the bulk.  The coloured "+" in each column marks
# $e^{\pm i\, 2\pi\, dt/T_k}$, the *expected* eigenvalues for that pair.

# %%
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

COUPLING_DOT_SIZE = 22

# How to normalise per-panel coupling values c_m (one per mode m).  All
# normalisations are computed independently for each subplot.
#   "raw"      : c                                       sequential, vmin=0..max
#   "max"      : c / max(c)                              sequential, vmin=0..1
#   "log"      : c on a log axis                         sequential, COUPLING_LOG_DECADES
#   "centered" : c - mean(c)                             diverging,  centred at 0
#   "log_ratio": log10(c / median(c))                    diverging,  centred at 0
#                                                        ("decades above the typical mode")
#   "zscore"   : (c - mean(c)) / std(c)                  diverging,  centred at 0
#   "rank"     : argsort-rank(c) / (N-1)  in [0, 1]      sequential, best contrast
COUPLING_NORM = "zscore"

COUPLING_LOG_DECADES = 4         # only used when COUPLING_NORM == "log"

_SEQ_CMAP = "viridis"
_DIV_CMAP = "RdBu_r"


def _prep_color(c):
    """Return (color_values, cmap, norm, cbar_label) for the chosen normalisation."""
    cmax = float(np.max(c)) if c.size else 0.0
    if COUPLING_NORM == "raw":
        return c, _SEQ_CMAP, Normalize(vmin=0.0, vmax=max(cmax, 1e-12)), "coupling"
    if COUPLING_NORM == "max":
        return (c / max(cmax, 1e-12)), _SEQ_CMAP, Normalize(vmin=0.0, vmax=1.0), \
            "coupling / max"
    if COUPLING_NORM == "log":
        vmax_use = max(cmax, 1e-12)
        vmin_use = vmax_use * 10 ** (-COUPLING_LOG_DECADES)
        c_plot = np.clip(c, vmin_use, None)
        return c_plot, _SEQ_CMAP, LogNorm(vmin=vmin_use, vmax=vmax_use), "coupling (log)"
    if COUPLING_NORM == "centered":
        cc = c - np.mean(c)
        m = max(np.abs(cc).max(), 1e-12)
        return cc, _DIV_CMAP, TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m), \
            "coupling − mean"
    if COUPLING_NORM == "log_ratio":
        med = np.median(c)
        if med <= 0:
            med = max(np.mean(c), 1e-12)
        ratio = np.log10(np.clip(c, 1e-30, None) / max(med, 1e-30))
        m = max(np.abs(ratio).max(), 1e-3)
        return ratio, _DIV_CMAP, TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m), \
            "log10(coupling / median)"
    if COUPLING_NORM == "zscore":
        mu = np.mean(c)
        sd = np.std(c)
        z = (c - mu) / max(sd, 1e-12)
        m = max(np.abs(z).max(), 1e-3)
        return z, _DIV_CMAP, TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m), "z-score"
    if COUPLING_NORM == "rank":
        order = np.argsort(c)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(c))
        ranks = ranks / max(len(c) - 1, 1)
        return ranks, _SEQ_CMAP, Normalize(vmin=0.0, vmax=1.0), "rank fraction"
    raise ValueError(f"Unknown COUPLING_NORM={COUPLING_NORM}")

fig, axes = plt.subplots(len(T_trains), n_pairs,
                         figsize=(4.4 * n_pairs, 4.4 * len(T_trains)),
                         squeeze=False)

for ri, T in enumerate(T_trains):
    sub = df[df["T_train"] == T]
    row = sub.loc[sub["final_val_loss"].idxmin()]
    seed_path = row["seed_path"]
    rc = row["run_config"]
    W_rec, W_out = _extract_weights(seed_path)
    if W_rec is None or W_out is None:
        for ci in range(n_pairs):
            axes[ri, ci].set_visible(False)
        continue
    N_dim = W_rec.shape[0]
    dt_rc = rc["dt"]
    tau_val = rc["time_constants_config"]["values"][0]
    alpha = 1.0 - np.exp(-dt_rc / tau_val)
    g_rc = rc["recurrent_gain"]
    J = (1 - alpha) * np.eye(N_dim) + alpha * g_rc * W_rec
    eigs, V = np.linalg.eig(J)

    coupling_full = np.abs(W_out @ V)
    pair_coupling = np.zeros((n_pairs, len(eigs)))
    for k in range(n_pairs):
        pair_coupling[k] = coupling_full[2*k] + coupling_full[2*k + 1]

    for ci in range(n_pairs):
        ax = axes[ri, ci]
        c = pair_coupling[ci]
        cmax = float(c.max()) if c.size > 0 else 0.0

        ax.plot(np.cos(theta_circle), np.sin(theta_circle),
                "k-", linewidth=0.6, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0, color="gray", linewidth=0.3)

        c_vals, cmap, norm, cbar_label = _prep_color(c)
        # draw weakest first so strongest end up on top
        order = np.argsort(np.abs(c_vals))
        sc = ax.scatter(eigs[order].real, eigs[order].imag,
                        c=c_vals[order], cmap=cmap, norm=norm,
                        s=COUPLING_DOT_SIZE, edgecolors="none",
                        rasterized=True, zorder=4)

        omega_dt = 2.0 * np.pi * dt_rc / periods_list[ci]
        for sign in (+1, -1):
            tx = np.cos(sign * omega_dt)
            ty = np.sin(sign * omega_dt)
            ax.plot(tx, ty, "o", markerfacecolor="none",
                    markeredgecolor=pair_colors[ci], markersize=18,
                    markeredgewidth=2.0, zorder=10)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(cbar_label, fontsize=7)

        stats_str = f"max={cmax:.2g}, med={np.median(c):.2g}"
        if ri == 0:
            ax.set_title(f"pair {ci}: T={periods_list[ci]:.0f}\n{stats_str}",
                         fontsize=10, fontweight="bold",
                         color=pair_colors[ci])
        else:
            ax.set_title(stats_str, fontsize=9)

        if ci == 0:
            ax.set_ylabel(f"{cycles_label(T)}\nIm(λ)",
                          fontsize=9, fontweight="bold")
        else:
            ax.set_ylabel("Im(λ)", fontsize=9)
        if ri == len(T_trains) - 1:
            ax.set_xlabel("Re(λ)", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.grid(True, alpha=0.15)

fig.suptitle(f"Eigenvalue Coupling Distribution -- rows = T_train, "
             f"columns = target pair  (g={g_fixed}, N={hidden_size}, "
             f"norm={COUPLING_NORM})",
             fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "eigenspectra_coupling_grid.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Coupling vs. distance-to-target
#
# For each (network, target pair):
# - x = $\min_\pm |\lambda - e^{\pm i\, 2\pi\, dt/T_k}|$, the closest distance
#   in the complex plane between each eigenvalue $\lambda$ of $J$ and either
#   of the two target eigenvalues for the column's pair.
# - y = pair-coupling strength of that mode (log-scale).
#
# Eigenvalue placement looks like a **down-and-to-the-right** distribution:
# the few high-coupling modes sit at small distance (left), while the bulk
# is mid-distance with low coupling.  Memorisation looks scattered or even
# **anti-correlated** (high coupling at large distance).  Dots are coloured
# by $|\lambda|$ so you can see whether high-coupling modes are near the
# unit circle or inside the disk.

# %%
fig, axes = plt.subplots(len(T_trains), n_pairs,
                         figsize=(4.4 * n_pairs, 3.2 * len(T_trains)),
                         squeeze=False, sharex=True, sharey=True)

# Pre-compute global y-range so log axes are comparable per row
all_min_d = []

for ri, T in enumerate(T_trains):
    sub = df[df["T_train"] == T]
    row = sub.loc[sub["final_val_loss"].idxmin()]
    seed_path = row["seed_path"]
    rc = row["run_config"]
    W_rec, W_out = _extract_weights(seed_path)
    if W_rec is None or W_out is None:
        for ci in range(n_pairs):
            axes[ri, ci].set_visible(False)
        continue
    N_dim = W_rec.shape[0]
    dt_rc = rc["dt"]
    tau_val = rc["time_constants_config"]["values"][0]
    alpha = 1.0 - np.exp(-dt_rc / tau_val)
    g_rc = rc["recurrent_gain"]
    J = (1 - alpha) * np.eye(N_dim) + alpha * g_rc * W_rec
    eigs, V = np.linalg.eig(J)

    coupling_full = np.abs(W_out @ V)
    pair_coupling = np.zeros((n_pairs, len(eigs)))
    for k in range(n_pairs):
        pair_coupling[k] = coupling_full[2 * k] + coupling_full[2 * k + 1]

    abs_eigs = np.abs(eigs)

    for ci in range(n_pairs):
        ax = axes[ri, ci]
        omega_dt = 2.0 * np.pi * dt_rc / periods_list[ci]
        target_plus = np.exp(1j * omega_dt)
        target_minus = np.exp(-1j * omega_dt)
        dist = np.minimum(np.abs(eigs - target_plus),
                          np.abs(eigs - target_minus))

        c = np.clip(pair_coupling[ci], 1e-12, None)
        sc = ax.scatter(dist, c, c=abs_eigs, cmap="plasma",
                        vmin=0.0, vmax=max(abs_eigs.max(), 1.0),
                        s=18, alpha=0.75, edgecolors="none",
                        rasterized=True)

        ax.axvline(0, color=pair_colors[ci], linestyle=":", linewidth=1.2,
                   alpha=0.8, label="target")
        #ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.15)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(r"$|\lambda|$", fontsize=8)

        if ri == 0:
            ax.set_title(f"pair {ci}: T={periods_list[ci]:.0f}",
                         fontsize=10, fontweight="bold",
                         color=pair_colors[ci])
        if ci == 0:
            ax.set_ylabel(f"{cycles_label(T)}\ncoupling",
                          fontsize=9, fontweight="bold")
        else:
            ax.set_ylabel("coupling", fontsize=9)
        if ri == len(T_trains) - 1:
            ax.set_xlabel(r"min $|\lambda - \lambda_\mathrm{target}|$", fontsize=10)

fig.suptitle("Coupling vs. Distance-to-Target  "
             "(rows = T_train, cols = target pair)",
             fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "coupling_vs_distance.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Per-pair eigenvalue scorecard
#
# Did the network find the "right" eigenvalues for each *individual* target
# pair?  Each row is a target pair (period); each column is a training
# horizon.  Cell value = number of matching eigenvalues found (max = 2).

# %%
score_matrix = np.zeros((n_pairs, len(T_trains)), dtype=int)
for j, T in enumerate(T_trains):
    sub = metrics_df[metrics_df["T_train"] == T]
    if len(sub) == 0:
        continue
    row = sub.iloc[0]
    score_matrix[:, j] = row["per_pair_found"]

fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(T_trains) + 2), 2 + 0.5 * n_pairs))
im = ax.imshow(score_matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=2)
ax.set_xticks(range(len(T_trains)))
ax.set_xticklabels([f"{T}\n({T*dt/T_slow:.1f} cyc)" for T in T_trains], fontsize=9)
ax.set_yticks(range(n_pairs))
ax.set_yticklabels([f"T={p:.0f}" for p in periods_list], fontsize=10)
ax.set_xlabel("num_time_steps (slow-period cycles)")
ax.set_ylabel("Target pair")
for i in range(n_pairs):
    for j in range(len(T_trains)):
        v = score_matrix[i, j]
        ax.text(j, i, f"{v}/2", ha="center", va="center",
                color="white" if v >= 1 else "black", fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax, label="# eigenvalues found", shrink=0.8)
fig.suptitle("Per-Pair Eigenvalue Scorecard (max = 2 per pair)",
             fontsize=12, fontweight="bold", y=1.04)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "scorecard_per_pair.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Save metrics dataframe
if SAVE_FIGS:
    out_csv = os.path.join(FIGS_DIR, "metrics.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")

cols = ["T_train", "n_cycles_slow", "rho", "n_unstable",
        "n_found", "readout_pr", "r2_in", "r2_out", "diverged",
        "final_val_loss"]
print(metrics_df[cols].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# T_FOCUS deep-dive — Schur coupling heatmaps + spectrum (matches nb 9 / 17 style)
# ══════════════════════════════════════════════════════════════════════════════

# %% T_FOCUS — imports & config
from scipy.linalg import schur as _scipy_schur

T_FOCUS = [100, 500, 2000]                # which T_train values to deep-dive
T_FOCUS = [T for T in T_FOCUS if T in T_trains]   # keep only those in the sweep
N_HM    = 12                              # heatmap x-axis width (top-k Schur / neurons)

# Per-pair colour palette — curated, picked so each pair colour reads cleanly
# against the salmon "#e76f51" bulk eigenvalues and the red "#c0392b" unit
# circle.  Each (cos,sin) channel pair shares one colour with its dominant
# eigenmode and target tick.
pair_colors_TF = [
    "#2563eb",   # royal blue
    "#7c3aed",   # violet
    "#059669",   # emerald
    "#d97706",   # amber
    "#db2777",   # magenta
    "#0891b2",   # cyan
][:n_pairs]

# Channel labels: each pair contributes (cos, sin); both rows in a pair share
# the pair's colour.
channel_pair_idx = [k for k in range(n_pairs) for _ in range(2)]   # [0,0,1,1,...]


# %% T_FOCUS — helpers

def _schur_sort_full(J: np.ndarray):
    """
    Real Schur decomposition of J, columns sorted by |λ| descending.

    Returns:
        Q_s        (N, N)  orthogonal basis (columns sorted)
        col_ev     (N,)    complex eigenvalue of each Schur column
                           (members of a 2x2 block carry conj eigenvalues)
        blk_lbl    (N,)    one of {"1x1", "2x2-a", "2x2-b"}
        abs_s      (N,)    |λ| per column
        tau_s      (N,)    effective timescale −1/ln|λ|  (capped at TAU_VIS_MAX equiv.)
    """
    T_mat, Q_mat = _scipy_schur(J, output="real")
    N = T_mat.shape[0]
    col_ev  = np.zeros(N, dtype=complex)
    blk_lbl = np.empty(N, dtype=object)
    k = 0
    while k < N:
        if k + 1 < N and abs(T_mat[k + 1, k]) > 1e-10:
            ev_pair = np.linalg.eigvals(T_mat[k:k+2, k:k+2])
            col_ev[k]     = ev_pair[0]
            col_ev[k + 1] = ev_pair[1]
            blk_lbl[k]    = "2x2-a"
            blk_lbl[k + 1] = "2x2-b"
            k += 2
        else:
            col_ev[k]  = T_mat[k, k]
            blk_lbl[k] = "1x1"
            k += 1
    abs_s = np.abs(col_ev)
    idx   = np.argsort(abs_s)[::-1]
    Q_s    = Q_mat[:, idx]
    col_ev = col_ev[idx]
    blk_lbl = blk_lbl[idx]
    abs_s = abs_s[idx]
    log_a = np.log(np.clip(abs_s, 1e-12, None))
    tau_s = -1.0 / np.where(log_a < -1e-10, log_a, -1e-10)
    return Q_s, col_ev, blk_lbl, abs_s, tau_s


def _osc_label(eigval, blk_type=None):
    """
    x-tick label string based on the oscillation period

        T^osc = 2π / |arg(λ)|

    `eigval`   complex eigenvalue at this column / index.
    `blk_type` one of {"1x1", "2x2-a", "2x2-b"} for Schur columns (preferred);
               for raw eigenvectors pass None and we'll infer from |Im(λ)|.

    For a 2×2 Schur block (or a complex eigenvector) we print T^osc; for real
    eigenvalues we print "DC" (positive real, no oscillation) or "T=2"
    (negative real, the Nyquist flip).
    """
    is_complex = (
        blk_type in ("2x2-a", "2x2-b")
        if blk_type is not None
        else abs(eigval.imag) > 1e-8
    )
    if is_complex:
        ang = abs(np.angle(eigval))
        if ang > 1e-8:
            return f"$T^{{osc}}$={2 * np.pi / ang:.1f}"
        return "DC"
    return "DC (real$+$)" if eigval.real >= 0 else "$T^{osc}=2$ (real$-$)"


def _draw_block_brackets(ax, blk_lbl_s, n_rows,
                          color="dimgray", lw=2.0, label="2×2"):
    """
    Draw a thin horizontal bar (and a small label) just below the heatmap
    spanning the two columns of every 2×2 Schur block, so the viewer can
    tell at a glance which adjacent column pairs are conjugate-pair blocks.

    The bar sits in data coordinates at y = n_rows - 0.5 + 0.18, with
    `clip_on=False` so it draws below the imshow rectangle.
    """
    y_bar = n_rows - 0.5 + 0.18
    n = len(blk_lbl_s)
    i = 0
    while i < n:
        if blk_lbl_s[i] == "2x2-a" and i + 1 < n:
            ax.plot([i - 0.45, i + 1 + 0.45], [y_bar, y_bar],
                    color=color, lw=lw, clip_on=False, solid_capstyle="butt",
                    zorder=10)
            ax.text(i + 0.5, y_bar + 0.18, label,
                    ha="center", va="top", fontsize=6, color=color,
                    clip_on=False)
            i += 2
        else:
            i += 1


def _pearson_r_TF(Z, Y):
    """Pearson r matrix of shape (K_z, K_y); Z:(M,Kz), Y:(M,Ky)."""
    n = Z.shape[0]
    Zc = Z - Z.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Zs = np.where(Zc.std(0) > 1e-12, Zc.std(0), 1e-12)
    Ys = np.where(Yc.std(0) > 1e-12, Yc.std(0), 1e-12)
    return (Zc.T @ Yc) / (n * Zs[:, None] * Ys[None, :])


def _greedy_dom_blocks(pair_coup, blk_lbl, col_ev, eigs_full, n_pairs):
    """
    Greedy per-pair assignment of a dominant block in a (Schur or eigenvector)
    basis, mapped to indices in `eigs_full` for plotting on the spectrum.

    Args:
        pair_coup : (n_pairs, N)  per-pair coupling magnitudes (always ≥ 0)
        blk_lbl   : (N,)          block labels in {"1x1","2x2-a","2x2-b"}
        col_ev    : (N,)  complex eigenvalue carried by each column of the basis
                          (used to find the matching index in eigs_full)
        eigs_full : (N,)  eigenvalues from np.linalg.eig (for spectrum overlay)
        n_pairs   : int   number of output pairs

    Returns:
        dict { pair_k : [eig_full_idx, ...] }   (1 entry per real / 2 per complex pair)
    """
    N = len(col_ev)
    used_cols = set()
    used_eigs = set()
    dom = {}
    for k in range(n_pairs):
        masked = pair_coup[k].copy().astype(float)
        for c in used_cols:
            masked[c] = -np.inf
        best = int(np.argmax(masked))
        if blk_lbl[best] == "2x2-a" and best + 1 < N:
            cols = [best, best + 1]
        elif blk_lbl[best] == "2x2-b" and best - 1 >= 0:
            cols = [best - 1, best]
        else:
            cols = [best]
        used_cols.update(cols)

        eig_idxs = []
        for c in cols:
            target = col_ev[c]
            order = np.argsort(np.abs(eigs_full - target))
            for cand in order:
                ci = int(cand)
                if ci not in used_eigs:
                    eig_idxs.append(ci)
                    used_eigs.add(ci)
                    break
        dom[k] = eig_idxs
    return dom


def _eig_block_lbl(eigs_sorted, atol=1e-7):
    """
    Label each eigenvector column as "1x1" (real) or "2x2-a"/"2x2-b" if it
    forms an adjacent complex-conjugate pair with the next column.
    Used for the eigenmode heatmap so we can reuse `_draw_block_brackets`.
    """
    N = len(eigs_sorted)
    lbl = np.empty(N, dtype=object)
    i = 0
    while i < N:
        ev = eigs_sorted[i]
        if abs(ev.imag) > atol and i + 1 < N:
            nx = eigs_sorted[i + 1]
            if (abs(abs(ev) - abs(nx)) < atol
                    and abs(ev.real - nx.real) < atol
                    and ev.imag * nx.imag < 0):
                lbl[i] = "2x2-a"
                lbl[i + 1] = "2x2-b"
                i += 2
                continue
        lbl[i] = "1x1"
        i += 1
    return lbl


# %% T_FOCUS — COMPUTE: Schur, coupling matrices, dominant blocks, eig indices
# Produces _TF[T] with all data needed for the plot cells below.

_TF = {}

for _T in T_FOCUS:
    sub = df[df["T_train"] == _T]
    if sub.empty:
        print(f"No runs for T_train={_T}")
        continue
    _row = sub.loc[sub["final_val_loss"].idxmin()]
    _sp  = _row["seed_path"]
    _rc  = _row["run_config"]

    _W_rec, _W_out = _extract_weights(_sp)
    if _W_rec is None or _W_out is None:
        print(f"  T_train={_T}: missing weights, skipping.")
        continue

    _N      = _W_rec.shape[0]
    _dt     = _rc["dt"]
    _tauv   = _rc["time_constants_config"]["values"][0]
    _alpha  = 1.0 - np.exp(-_dt / _tauv)
    _g      = _rc["recurrent_gain"]
    _J      = (1.0 - _alpha) * np.eye(_N) + _alpha * _g * _W_rec

    # ── Untrained baseline eigenvalues (for the spectrum plots) ─────────────
    _W_rec_un, _ = _extract_weights(_sp, which="untrained")
    if _W_rec_un is not None and _W_rec_un.shape[0] == _N:
        _J_un = (1.0 - _alpha) * np.eye(_N) + _alpha * _g * _W_rec_un
        _eigs_untrained = np.linalg.eigvals(_J_un)
    else:
        _eigs_untrained = None

    # Eigendecomposition (for spectrum plot + eigenmode coupling)
    # Use lexsort so complex-conjugate pairs end up adjacent (sorted +imag, then -imag)
    _eigs_full_unsort, _V_full = np.linalg.eig(_J)
    _eig_abs_rank = np.lexsort((-_eigs_full_unsort.imag,
                                 -np.abs(_eigs_full_unsort)))
    _eigs_sorted  = _eigs_full_unsort[_eig_abs_rank]              # complex (N,)
    _V_sorted     = _V_full[:, _eig_abs_rank]                     # complex (N, N)
    _eig_blk_lbl  = _eig_block_lbl(_eigs_sorted)                  # like Schur blk_lbl

    # Backwards-compat alias for the spectrum plot
    _eigs_full = _eigs_full_unsort

    # Schur decomposition (orthogonal basis, sorted by |λ|)
    _Q_s, _col_ev_s, _blk_lbl_s, _abs_s, _tau_s = _schur_sort_full(_J)

    # Coupling: |W_out @ Q|  rows = output channels, cols = Schur modes
    _coup_schur = np.abs(_W_out @ _Q_s)               # (n_channels, N)
    _coup_eig   = np.abs(_W_out @ _V_sorted)          # (n_channels, N) real magn.

    # Coupling: |W_out|, columns sorted by ||W_out[:, j]||  (rows = channels)
    _cout       = np.abs(_W_out)
    _cout_nro   = np.argsort(np.linalg.norm(_cout, axis=0))[::-1]
    _cout_s     = _cout[:, _cout_nro]                  # (n_channels, N)

    # Per-pair coupling = sum over (cos, sin) for each pair k  (Schur basis)
    _pair_coup = np.zeros((n_pairs, _N))
    for _k in range(n_pairs):
        _pair_coup[_k] = _coup_schur[2 * _k] + _coup_schur[2 * _k + 1]

    # Per-pair coupling in eigenvector basis (used only for stats / sorting)
    _pair_eig_coup = np.zeros((n_pairs, _N))
    for _k in range(n_pairs):
        _pair_eig_coup[_k] = _coup_eig[2 * _k] + _coup_eig[2 * _k + 1]

    # ── Autonomous rollout for correlation (Identity activation, no input) ──
    # h[t+1] = J @ h[t],  h[0] = init_hidden_value * ones(N)
    _init_h_val = float(_rc.get("init_hidden_value", 1.0))
    _T_corr     = max(int(5 * T_slow / _dt), 500)
    _h_traj     = np.zeros((_T_corr, _N), dtype=np.float64)
    _h_traj[0]  = _init_h_val * np.ones(_N)
    for _ti in range(1, _T_corr):
        _h_traj[_ti] = _J @ _h_traj[_ti - 1]
    _, _tgt_corr = _make_targets(_T_corr, _dt)        # (T_corr, n_channels)

    # Drop transient (first T_slow steps) before computing correlations
    _t_skip  = min(int(T_slow / _dt), _T_corr // 4)
    _h_corr  = _h_traj[_t_skip:]
    _tg_corr = _tgt_corr[_t_skip:]

    if (not np.all(np.isfinite(_h_corr))) or (np.abs(_h_corr).max() > 1e10):
        _r_schur_tgt = None
        _r_eig_tgt   = None
        print(f"  T_train={_T}: rollout diverged ({np.abs(_h_corr).max():.2g}) — corr skipped")
    else:
        _Z_schur     = _h_corr @ _Q_s                                # (T_skip, N) real
        _r_schur_tgt = _pearson_r_TF(_Z_schur, _tg_corr)             # (N, n_channels)
        try:
            _V_inv  = np.linalg.pinv(_V_sorted)
            _Z_eig  = np.real(_h_corr @ _V_inv.T)                    # (T_skip, N) real
            _r_eig_tgt = _pearson_r_TF(_Z_eig, _tg_corr)             # (N, n_channels)
        except np.linalg.LinAlgError:
            _r_eig_tgt = None
            print(f"  T_train={_T}: V is rank-deficient — eig correlation skipped")

    # Dominant Schur block per pair (greedy, no double assignment)
    _dom_block_cols = {}
    _dom_eig_idx    = {}
    _used_cols      = set()
    _used_eigs      = set()
    for _k in range(n_pairs):
        _masked = _pair_coup[_k].copy()
        for _c in _used_cols:
            _masked[_c] = -np.inf
        _best_col = int(np.argmax(_masked))
        if _blk_lbl_s[_best_col] == "2x2-a" and _best_col + 1 < _N:
            _cols = [_best_col, _best_col + 1]
        elif _blk_lbl_s[_best_col] == "2x2-b" and _best_col - 1 >= 0:
            _cols = [_best_col - 1, _best_col]
        else:
            _cols = [_best_col]
        _dom_block_cols[_k] = _cols
        _used_cols.update(_cols)

        # Map each Schur column's eigenvalue to nearest np.linalg.eig index
        _eig_idxs = []
        for _c in _cols:
            _target = _col_ev_s[_c]
            _order  = np.argsort(np.abs(_eigs_full - _target))
            for _cand in _order:
                _ci = int(_cand)
                if _ci not in _used_eigs:
                    _eig_idxs.append(_ci)
                    _used_eigs.add(_ci)
                    break
        _dom_eig_idx[_k] = _eig_idxs

    # ── Dominant-mode mappings under all four coupling criteria ──────────────
    # 1) Schur · connectivity (already computed inline above)
    _dom_eig_idx_schur_conn = _dom_eig_idx

    # 2) Eigenmode · connectivity:  pair_eig_coup → block extension on V columns
    _dom_eig_idx_eig_conn = _greedy_dom_blocks(
        _pair_eig_coup, _eig_blk_lbl, _eigs_sorted, _eigs_full, n_pairs)

    # 3) Schur · correlation:  per-pair |r| summed over (cos, sin)
    if _r_schur_tgt is not None:
        _pair_r_schur = np.zeros((n_pairs, _N))
        for _k in range(n_pairs):
            _pair_r_schur[_k] = (np.abs(_r_schur_tgt[:, 2 * _k])
                                 + np.abs(_r_schur_tgt[:, 2 * _k + 1]))
        _dom_eig_idx_schur_corr = _greedy_dom_blocks(
            _pair_r_schur, _blk_lbl_s, _col_ev_s, _eigs_full, n_pairs)
    else:
        _pair_r_schur = None
        _dom_eig_idx_schur_corr = None

    # 4) Eigenmode · correlation
    if _r_eig_tgt is not None:
        _pair_r_eig = np.zeros((n_pairs, _N))
        for _k in range(n_pairs):
            _pair_r_eig[_k] = (np.abs(_r_eig_tgt[:, 2 * _k])
                               + np.abs(_r_eig_tgt[:, 2 * _k + 1]))
        _dom_eig_idx_eig_corr = _greedy_dom_blocks(
            _pair_r_eig, _eig_blk_lbl, _eigs_sorted, _eigs_full, n_pairs)
    else:
        _pair_r_eig = None
        _dom_eig_idx_eig_corr = None

    _TF[_T] = dict(
        N=_N, dt=_dt, alpha=_alpha, g=_g,
        eigs_full=_eigs_full, eigs_untrained=_eigs_untrained,
        # Schur basis
        Q_s=_Q_s, col_ev_s=_col_ev_s, blk_lbl_s=_blk_lbl_s,
        abs_s=_abs_s, tau_s=_tau_s,
        coup_schur=_coup_schur, cout_s=_cout_s,
        pair_coup=_pair_coup,
        dom_block_cols=_dom_block_cols,
        dom_eig_idx=_dom_eig_idx,                             # backward-compat alias
        # Eigenvector basis
        eigs_sorted=_eigs_sorted, V_sorted=_V_sorted,
        eig_blk_lbl=_eig_blk_lbl, coup_eig=_coup_eig,
        pair_eig_coup=_pair_eig_coup,
        # Correlations
        r_schur_tgt=_r_schur_tgt, r_eig_tgt=_r_eig_tgt,
        pair_r_schur=_pair_r_schur, pair_r_eig=_pair_r_eig,
        # Dominant-mode mappings under each criterion (for spectrum panels)
        dom_eig_idx_schur_conn=_dom_eig_idx_schur_conn,
        dom_eig_idx_eig_conn=_dom_eig_idx_eig_conn,
        dom_eig_idx_schur_corr=_dom_eig_idx_schur_corr,
        dom_eig_idx_eig_corr=_dom_eig_idx_eig_corr,
        final_val_loss=_row["final_val_loss"], seed=_row["seed"],
    )

    _r2o = metrics_df.loc[(metrics_df["T_train"] == _T)
                          & (metrics_df["seed"] == _row["seed"]),
                          "r2_out"]
    _r2_str = f"R²_out={float(_r2o.iloc[0]):+.2f}" if len(_r2o) else "R²_out=?"
    print(f"  T_train={_T:>5d}: Schur+coupling done   "
          f"(N={_N}, val_loss={_row['final_val_loss']:.2g}, {_r2_str})")

# Shared colormap scales across panels
_vmax_schur_out_TF = max(
    _TF[T]["coup_schur"][:, :N_HM].max() for T in T_FOCUS if T in _TF
) if _TF else 1.0
_vmax_neu_out_TF = max(
    _TF[T]["cout_s"][:, :N_HM].max() for T in T_FOCUS if T in _TF
) if _TF else 1.0
_vmax_eig_out_TF = max(
    _TF[T]["coup_eig"][:, :N_HM].max() for T in T_FOCUS if T in _TF
) if _TF else 1.0
print(f"Shared vmaxes — schur_out={_vmax_schur_out_TF:.4f}, "
      f"neuron_out={_vmax_neu_out_TF:.4f}, "
      f"eig_out={_vmax_eig_out_TF:.4f}")


# %% T_FOCUS — PLOT A: Schur → Output coupling heatmap
# Requires: _TF, T_FOCUS, N_HM, n_channels, channel_labels, _vmax_schur_out_TF

n_cols_TF = max(len(T_FOCUS), 1)
fig, axes_A = plt.subplots(
    1, n_cols_TF,
    figsize=(max(N_HM * 0.9, 6) * n_cols_TF + 1, max(n_channels * 0.55, 3.5)),
    squeeze=False,
)
axes_A = axes_A[0]

for col, T in enumerate(T_FOCUS):
    ax = axes_A[col]
    if T not in _TF:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        continue
    d   = _TF[T]
    tk  = min(N_HM, d["N"])
    hm  = d["coup_schur"][:, :tk]
    xlb = [f"Schur {mi+1}\n{_osc_label(d['col_ev_s'][mi], d['blk_lbl_s'][mi])}"
           for mi in range(tk)]

    im = ax.imshow(hm, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=_vmax_schur_out_TF)
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_labels, fontsize=8)
    # Colour each y-tick label by its pair colour
    for ti, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_color(pair_colors_TF[channel_pair_idx[ti]])
    ax.set_xticks(range(tk))
    ax.set_xticklabels(xlb, fontsize=7)
    ax.set_xlabel(f"Schur mode rank (top {tk})", fontsize=9)
    ax.set_ylabel("Output channel", fontsize=10)
    for i in range(n_channels):
        for j in range(tk):
            v = hm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.5 * _vmax_schur_out_TF else "black")
    _draw_block_brackets(ax, d["blk_lbl_s"][:tk], n_channels)
    plt.colorbar(im, ax=ax, label="$|W_{out}\\,Q|$", shrink=0.75)
    ax.set_title(f"T_train = {T}  ({T*dt/T_slow:.1f} cyc)\n"
                 f"val_loss={d['final_val_loss']:.2g}",
                 fontsize=10, fontweight="bold")

fig.suptitle(f"Schur → Output Coupling (connectivity)  —  "
             f"T_train ∈ {T_FOCUS}",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "schur_output_heatmap_TFOCUS.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% T_FOCUS — PLOT B: Neuron → Output coupling heatmap
# Requires: _TF, T_FOCUS, N_HM, n_channels, channel_labels, _vmax_neu_out_TF

fig, axes_B = plt.subplots(
    1, n_cols_TF,
    figsize=(max(N_HM * 0.9, 6) * n_cols_TF + 1, max(n_channels * 0.55, 3.5)),
    squeeze=False,
)
axes_B = axes_B[0]

for col, T in enumerate(T_FOCUS):
    ax = axes_B[col]
    if T not in _TF:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        continue
    d  = _TF[T]
    tk = min(N_HM, d["N"])
    hm = d["cout_s"][:, :tk]

    im = ax.imshow(hm, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=_vmax_neu_out_TF)
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_labels, fontsize=8)
    for ti, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_color(pair_colors_TF[channel_pair_idx[ti]])
    ax.set_xticks(range(tk))
    ax.set_xticklabels([f"N{i+1}" for i in range(tk)], fontsize=7)
    ax.set_xlabel(f"Neuron rank (top {tk}, by ||w_out||)", fontsize=9)
    ax.set_ylabel("Output channel", fontsize=10)
    for i in range(n_channels):
        for j in range(tk):
            v = hm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.5 * _vmax_neu_out_TF else "black")
    plt.colorbar(im, ax=ax, label="$|W_{out,ij}|$", shrink=0.75)
    ax.set_title(f"T_train = {T}  ({T*dt/T_slow:.1f} cyc)\n"
                 f"val_loss={d['final_val_loss']:.2g}",
                 fontsize=10, fontweight="bold")

fig.suptitle(f"Neuron → Output Coupling (connectivity)  —  "
             f"T_train ∈ {T_FOCUS}",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "neuron_output_heatmap_TFOCUS.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% T_FOCUS — PLOT C: Jacobian spectrum (color-matched per pair)
# Untrained eigenvalues = light blue "#c0d8e8"  (matches notebook 17 baseline)
# Bulk trained eigs     = salmon "#e76f51"      (matches notebooks 9 / 17)
# Dominant Schur block per pair k = pair_colors_TF[k] (matches heatmap rows)
# Target arg(λ) = ±2π dt / T_k drawn as short THICK RADIAL segments JUST OUTSIDE
# the unit circle, in the matching pair colour.  The radial gap between the
# unit circle and the inner tick is set by `_R_IN`, large enough that an
# eigenvalue sitting exactly on the target does not visually overlap the marker.
#
# Requires: _TF, T_FOCUS, n_pairs, periods_list, pair_colors_TF

theta_circ = np.linspace(0, 2 * np.pi, 400)
_R_IN, _R_OUT = 1.06, 1.14   # radial extent of the target tick segments
_AX_LIM       = 1.22         # axis half-width to leave room for the ticks

fig, axes_C = plt.subplots(1, n_cols_TF,
                            figsize=(5.6 * n_cols_TF, 5.6), squeeze=False)
axes_C = axes_C[0]

for col, T in enumerate(T_FOCUS):
    ax = axes_C[col]
    if T not in _TF:
        ax.set_visible(False); continue
    d     = _TF[T]
    eigs  = d["eigs_full"]
    n_eig = len(eigs)

    # Unit circle (solid red, like notebooks 9/17)
    ax.plot(np.cos(theta_circ), np.sin(theta_circ),
            color="#c0392b", lw=1.0, ls="-", alpha=0.7, zorder=2)
    ax.axhline(0, color="#ccc", lw=0.3)
    ax.axvline(0, color="#ccc", lw=0.3)

    # Untrained baseline (light blue dots, behind the trained bulk)
    eu = d.get("eigs_untrained")
    if eu is not None:
        ax.scatter(eu.real, eu.imag, s=10, color="#c0d8e8",
                   alpha=0.6, edgecolors="none", zorder=2.5,
                   label="untrained" if col == 0 else "_nolegend_")

    # Bulk trained eigenvalues
    dom_set     = set()
    idx_to_pair = {}
    for k_p, eig_idxs in d["dom_eig_idx"].items():
        for ii in eig_idxs:
            dom_set.add(ii)
            idx_to_pair[ii] = k_p
    rest_mask = np.array([i not in dom_set for i in range(n_eig)])

    ax.scatter(eigs[rest_mask].real, eigs[rest_mask].imag,
               s=12, color="#e76f51", alpha=0.6, edgecolors="none", zorder=3,
               label="trained (bulk)" if col == 0 else "_nolegend_")

    # Dominant Schur block per pair → colour-matched ring
    seen_pairs = set()
    for ii, k_p in idx_to_pair.items():
        lbl = (f"pair {k_p} (T={periods_list[k_p]:.0f})"
               if k_p not in seen_pairs else "_nolegend_")
        seen_pairs.add(k_p)
        ax.scatter(eigs[ii].real, eigs[ii].imag, s=80,
                   color=pair_colors_TF[k_p],
                   edgecolors="white", linewidths=0.8, zorder=6, label=lbl)

    # Target arg(λ): short thick RADIAL segments OUTSIDE the unit circle
    for k_p in range(n_pairs):
        omega_dt = 2.0 * np.pi * d["dt"] / periods_list[k_p]
        for sign in (+1, -1):
            theta_t = sign * omega_dt
            x0, y0 = _R_IN  * np.cos(theta_t), _R_IN  * np.sin(theta_t)
            x1, y1 = _R_OUT * np.cos(theta_t), _R_OUT * np.sin(theta_t)
            ax.plot([x0, x1], [y0, y1],
                    color=pair_colors_TF[k_p],
                    linewidth=3.2, alpha=0.95,
                    solid_capstyle="round", zorder=8)

    rho = float(np.abs(eigs).max())
    ax.set_title(f"T_train = {T}  ({T*d['dt']/T_slow:.1f} cyc)\n"
                 f"ρ={rho:.3f}   val_loss={d['final_val_loss']:.2g}",
                 fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-_AX_LIM, _AX_LIM)
    ax.set_ylim(-_AX_LIM, _AX_LIM)
    ax.set_xlabel("Re(λ)", fontsize=10)
    ax.set_ylabel("Im(λ)", fontsize=10)
    ax.grid(True, alpha=0.1)
    if col == 0:
        ax.legend(fontsize=7, loc="lower left", framealpha=0.85)

fig.suptitle(
    f"Jacobian Spectrum  —  T_train ∈ {T_FOCUS}\n"
    "Light blue = untrained · salmon = trained bulk · "
    "rings = dominant Schur block per pair · radial ticks = target arg(λ)",
    fontsize=12, fontweight="bold", y=1.03)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "spectrum_TFOCUS.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% T_FOCUS — PLOT D: Eigenmode → Output coupling heatmap (connectivity)
# Same layout as PLOT A but uses eigenvectors V instead of the Schur basis Q.
# x-tick labels show T^osc derived from the eigenvalue's argument.
# Adjacent complex-conjugate pairs are bracketed underneath, so it's clear
# which two columns share the same |λ|.
# Requires: _TF, T_FOCUS, N_HM, n_channels, channel_labels, _vmax_eig_out_TF

fig, axes_D = plt.subplots(
    1, n_cols_TF,
    figsize=(max(N_HM * 0.9, 6) * n_cols_TF + 1, max(n_channels * 0.55, 3.5)),
    squeeze=False,
)
axes_D = axes_D[0]

for col, T in enumerate(T_FOCUS):
    ax = axes_D[col]
    if T not in _TF:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        continue
    d   = _TF[T]
    tk  = min(N_HM, d["N"])
    hm  = d["coup_eig"][:, :tk]
    xlb = [f"Eig {mi+1}\n{_osc_label(d['eigs_sorted'][mi], d['eig_blk_lbl'][mi])}"
           for mi in range(tk)]

    im = ax.imshow(hm, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=_vmax_eig_out_TF)
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_labels, fontsize=8)
    for ti, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_color(pair_colors_TF[channel_pair_idx[ti]])
    ax.set_xticks(range(tk))
    ax.set_xticklabels(xlb, fontsize=7)
    ax.set_xlabel(f"Eigenmode rank (by $|\\lambda|$, top {tk})", fontsize=9)
    ax.set_ylabel("Output channel", fontsize=10)
    for i in range(n_channels):
        for j in range(tk):
            v = hm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.5 * _vmax_eig_out_TF else "black")
    _draw_block_brackets(ax, d["eig_blk_lbl"][:tk], n_channels,
                          label="conj-pair")
    plt.colorbar(im, ax=ax, label="$|W_{out}\\,V|$", shrink=0.75)
    ax.set_title(f"T_train = {T}  ({T*dt/T_slow:.1f} cyc)\n"
                 f"val_loss={d['final_val_loss']:.2g}",
                 fontsize=10, fontweight="bold")

fig.suptitle(f"Eigenmode → Output Coupling (connectivity)  —  "
             f"T_train ∈ {T_FOCUS}",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "eig_output_heatmap_TFOCUS.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% T_FOCUS — PLOT E: Schur ↔ Output correlation heatmap
# Pearson r between each Schur-mode trajectory (from autonomous rollout) and
# each output channel target signal. Rows = output channels, cols = top Schur
# modes. Diverging colormap centered at 0 (negative values = anti-correlation).
# Requires: _TF, T_FOCUS, N_HM, n_channels, channel_labels

fig, axes_E = plt.subplots(
    1, n_cols_TF,
    figsize=(max(N_HM * 0.9, 6) * n_cols_TF + 1, max(n_channels * 0.55, 3.5)),
    squeeze=False,
)
axes_E = axes_E[0]

for col, T in enumerate(T_FOCUS):
    ax = axes_E[col]
    if T not in _TF or _TF[T]["r_schur_tgt"] is None:
        ax.text(0.5, 0.5, "no correlation\n(rollout diverged?)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        continue
    d   = _TF[T]
    tk  = min(N_HM, d["N"])
    # r_schur_tgt: (N, n_channels)  → transpose to (n_channels, N) for imshow
    hm  = d["r_schur_tgt"][:tk, :].T   # (n_channels, tk)
    xlb = [f"Schur {mi+1}\n{_osc_label(d['col_ev_s'][mi], d['blk_lbl_s'][mi])}"
           for mi in range(tk)]

    im = ax.imshow(hm, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_labels, fontsize=8)
    for ti, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_color(pair_colors_TF[channel_pair_idx[ti]])
    ax.set_xticks(range(tk))
    ax.set_xticklabels(xlb, fontsize=7)
    ax.set_xlabel(f"Schur mode rank (top {tk})", fontsize=9)
    ax.set_ylabel("Output channel", fontsize=10)
    for i in range(n_channels):
        for j in range(tk):
            v = hm[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.55 else "black")
    _draw_block_brackets(ax, d["blk_lbl_s"][:tk], n_channels)
    plt.colorbar(im, ax=ax, label="Pearson $r$", shrink=0.75)
    ax.set_title(f"T_train = {T}  ({T*dt/T_slow:.1f} cyc)\n"
                 f"val_loss={d['final_val_loss']:.2g}",
                 fontsize=10, fontweight="bold")

fig.suptitle(f"Schur ↔ Output Correlation  (autonomous rollout, transient dropped)  "
             f"—  T_train ∈ {T_FOCUS}",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "schur_output_corr_TFOCUS.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% T_FOCUS — PLOT F: Eigenmode ↔ Output correlation heatmap
# Same as PLOT E but using eigenvector projections h(t) @ V^{-T} (real part).
# Requires: _TF, T_FOCUS, N_HM, n_channels, channel_labels

fig, axes_F = plt.subplots(
    1, n_cols_TF,
    figsize=(max(N_HM * 0.9, 6) * n_cols_TF + 1, max(n_channels * 0.55, 3.5)),
    squeeze=False,
)
axes_F = axes_F[0]

for col, T in enumerate(T_FOCUS):
    ax = axes_F[col]
    if T not in _TF or _TF[T]["r_eig_tgt"] is None:
        ax.text(0.5, 0.5, "no correlation\n(rollout diverged?)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        continue
    d   = _TF[T]
    tk  = min(N_HM, d["N"])
    hm  = d["r_eig_tgt"][:tk, :].T
    xlb = [f"Eig {mi+1}\n{_osc_label(d['eigs_sorted'][mi], d['eig_blk_lbl'][mi])}"
           for mi in range(tk)]

    im = ax.imshow(hm, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_labels, fontsize=8)
    for ti, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_color(pair_colors_TF[channel_pair_idx[ti]])
    ax.set_xticks(range(tk))
    ax.set_xticklabels(xlb, fontsize=7)
    ax.set_xlabel(f"Eigenmode rank (by $|\\lambda|$, top {tk})", fontsize=9)
    ax.set_ylabel("Output channel", fontsize=10)
    for i in range(n_channels):
        for j in range(tk):
            v = hm[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.55 else "black")
    _draw_block_brackets(ax, d["eig_blk_lbl"][:tk], n_channels,
                          label="conj-pair")
    plt.colorbar(im, ax=ax, label="Pearson $r$", shrink=0.75)
    ax.set_title(f"T_train = {T}  ({T*dt/T_slow:.1f} cyc)\n"
                 f"val_loss={d['final_val_loss']:.2g}",
                 fontsize=10, fontweight="bold")

fig.suptitle(f"Eigenmode ↔ Output Correlation  (autonomous rollout, transient dropped)  "
             f"—  T_train ∈ {T_FOCUS}",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "eig_output_corr_TFOCUS.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()


# %% T_FOCUS — PLOT G: Spectrum across all four coupling criteria
# 4 rows × len(T_FOCUS) columns. Each row picks the dominant eigenvalue per pair
# using a different criterion; everything else (bulk colour, ring style, target
# arg ticks, axis limits) matches PLOT C so the panels are visually comparable.
#
#   Row 1: Schur · connectivity      |W_out @ Q|
#   Row 2: Eigenmode · connectivity  |W_out @ V|
#   Row 3: Schur · |correlation|     |r(Z_schur, target)|
#   Row 4: Eigenmode · |correlation| |r(Z_eig,   target)|
#
# Requires: _TF, T_FOCUS, n_pairs, periods_list, pair_colors_TF, theta_circ,
#            _R_IN, _R_OUT, T_slow, FIGS_DIR, SAVE_FIGS

_CRITERIA = [
    ("Schur · connectivity",      "dom_eig_idx_schur_conn"),
    ("Eigenmode · connectivity",  "dom_eig_idx_eig_conn"),
    ("Schur · |correlation|",     "dom_eig_idx_schur_corr"),
    ("Eigenmode · |correlation|", "dom_eig_idx_eig_corr"),
]

n_rows_G = len(_CRITERIA)
fig, axes_G = plt.subplots(
    n_rows_G, n_cols_TF,
    figsize=(5.2 * n_cols_TF, 5.2 * n_rows_G),
    squeeze=False,
)

for ri, (label, attr) in enumerate(_CRITERIA):
    for col, T in enumerate(T_FOCUS):
        ax = axes_G[ri, col]
        if T not in _TF:
            ax.set_visible(False); continue
        d     = _TF[T]
        eigs  = d["eigs_full"]
        n_eig = len(eigs)
        dom_map = d.get(attr)

        # Unit circle + axes
        ax.plot(np.cos(theta_circ), np.sin(theta_circ),
                color="#c0392b", lw=1.0, ls="-", alpha=0.7, zorder=2)
        ax.axhline(0, color="#ccc", lw=0.3)
        ax.axvline(0, color="#ccc", lw=0.3)

        # Untrained baseline (light blue dots)
        eu = d.get("eigs_untrained")
        if eu is not None:
            ax.scatter(eu.real, eu.imag, s=10, color="#c0d8e8",
                       alpha=0.6, edgecolors="none", zorder=2.5,
                       label="untrained" if (ri == 0 and col == 0) else "_nolegend_")

        # Bulk + dominant
        if dom_map is None:
            # Correlation forward pass diverged; just show the bulk + a notice
            ax.scatter(eigs.real, eigs.imag, s=12, color="#e76f51",
                       alpha=0.6, edgecolors="none", zorder=3)
            ax.text(0.5, 0.95, "no correlation data",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, color="dimgray")
        else:
            dom_set     = set()
            idx_to_pair = {}
            for k_p, eig_idxs in dom_map.items():
                for ii in eig_idxs:
                    dom_set.add(ii)
                    idx_to_pair[ii] = k_p
            rest_mask = np.array([i not in dom_set for i in range(n_eig)])

            ax.scatter(eigs[rest_mask].real, eigs[rest_mask].imag, s=12,
                       color="#e76f51", alpha=0.6, edgecolors="none", zorder=3,
                       label="trained (bulk)" if (ri == 0 and col == 0) else "_nolegend_")

            seen_pairs = set()
            for ii, k_p in idx_to_pair.items():
                lbl_h = (f"pair {k_p} (T={periods_list[k_p]:.0f})"
                         if (k_p not in seen_pairs and ri == 0 and col == 0)
                         else "_nolegend_")
                seen_pairs.add(k_p)
                ax.scatter(eigs[ii].real, eigs[ii].imag, s=80,
                           color=pair_colors_TF[k_p],
                           edgecolors="white", linewidths=0.8, zorder=6,
                           label=lbl_h)

        # Target arg(λ) radial ticks (always shown for reference)
        for k_p in range(n_pairs):
            omega_dt = 2.0 * np.pi * d["dt"] / periods_list[k_p]
            for sign in (+1, -1):
                theta_t = sign * omega_dt
                x0, y0 = _R_IN  * np.cos(theta_t), _R_IN  * np.sin(theta_t)
                x1, y1 = _R_OUT * np.cos(theta_t), _R_OUT * np.sin(theta_t)
                ax.plot([x0, x1], [y0, y1],
                        color=pair_colors_TF[k_p],
                        linewidth=3.2, alpha=0.95,
                        solid_capstyle="round", zorder=8)

        rho = float(np.abs(eigs).max())
        if ri == 0:
            ax.set_title(f"T_train = {T}  ({T*d['dt']/T_slow:.1f} cyc)\n"
                         f"ρ={rho:.3f}   val_loss={d['final_val_loss']:.2g}",
                         fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-_AX_LIM, _AX_LIM)
        ax.set_ylim(-_AX_LIM, _AX_LIM)
        if ri == n_rows_G - 1:
            ax.set_xlabel("Re(λ)", fontsize=10)
        if col == 0:
            ax.set_ylabel(f"{label}\nIm(λ)",
                          fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.1)
        if ri == 0 and col == 0:
            ax.legend(fontsize=7, loc="lower left", framealpha=0.85)

fig.suptitle(
    f"Jacobian Spectrum — Dominant modes by 4 coupling criteria  —  "
    f"T_train ∈ {T_FOCUS}\n"
    "Light blue = untrained · salmon = trained bulk · "
    "rings = dominant block per pair · radial ticks = target arg(λ)",
    fontsize=12, fontweight="bold", y=1.005)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "spectrum_4criteria_TFOCUS.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()
