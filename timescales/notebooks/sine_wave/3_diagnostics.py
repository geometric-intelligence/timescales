# %% [markdown]
# # Sine-Wave Generation — Generic Sweep Diagnostics
#
# Loads any sweep that varies a single (or two) hyperparameter(s) and computes
# diagnostics that distinguish **eigenvalue placement** from **memorisation**:
#
# 1. Training/validation curves (loss + $R^2$) coloured by sweep variable.
# 2. Extrapolation $R^2$: train-window vs post-training-window quality on a
#    rollout several training horizons long.  This is the cleanest
#    "did the network find oscillatory eigenvalues?" signal.
# 3. Spectral radius $\rho(J) = \max_k |\lambda_k|$.  Memorisation solutions
#    often have modes with $|\lambda| > 1$ that conspire to cancel inside
#    $[0, T_\text{train}]$ and blow up outside.
# 4. **Modes-found scorecard**: for each target frequency $\omega_k = 2\pi/T_k$,
#    count how many eigenvalues of $J$ lie within $\epsilon$ of
#    $e^{\pm i\, \omega_k\, dt}$ (close to the unit circle, at the right angle).
#    Reported as `n_found / 2N`.
# 5. **Readout participation ratio**: $\mathrm{PR} = (\sum_k s_k)^2 / \sum_k s_k^2$
#    with $s_k = \|(W_\text{out} V)_{:,k}\|$.  Pure eigenvalue placement gives
#    $\mathrm{PR} \approx 2N$ (one per cos/sin per pair); memorisation spreads
#    over many modes and gives $\mathrm{PR} \gg 2N$.
#
# Configure `SWEEP_DIR`, `SWEEP_VAR`, and the prefix-extraction below for any
# 1-D sweep (cycles, dt, weight decay, gain, hidden size, ...).  For 2-D
# sweeps, set `SWEEP_VAR_2` and the heatmap section will activate.

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

# %% Sweep configuration -- EDIT THESE FOR YOUR SWEEP
SWEEP_DIR = "/home/facosta/timescales/timescales/logs/experiments/sine_wave_hetero_cycles_sweep_20260418_002520"

# The hyperparameter being swept (must be a key in run_config.yaml).
SWEEP_VAR = "hidden_size"
SWEEP_VAR_LABEL = "Hidden size $N$"

# Optional second sweep variable for 2-D heatmaps.  Set to None for 1-D.
SWEEP_VAR_2 = None
SWEEP_VAR_2_LABEL = None

SAVE_FIGS = True
FIGS_DIR = os.path.join("notebooks", "figs",
                       f"sine_wave_diag_{os.path.basename(SWEEP_DIR)}")
os.makedirs(FIGS_DIR, exist_ok=True)

# Diagnostic hyperparameters
ROLLOUT_MULTIPLIER = 5      # extended rollout = 5x training horizon
EIG_RADIAL_TOL = 0.05       # |lambda| within (1-tol, 1+tol)
EIG_ANGULAR_TOL_FRAC = 0.05 # angular tolerance = tol_frac * (2pi/T_k)
PR_THRESH_FRAC = 1e-3       # ignore modes with |W_out V|_k < frac * max for PR

# %% Load sweep records
records = []
for exp_name in sorted(os.listdir(SWEEP_DIR)):
    exp_dir = os.path.join(SWEEP_DIR, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
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

        if SWEEP_VAR not in rc:
            print(f"WARN: SWEEP_VAR={SWEEP_VAR} not in {rc_path}; skipping")
            continue

        sweep_val = rc[SWEEP_VAR]
        sweep_val_2 = rc.get(SWEEP_VAR_2) if SWEEP_VAR_2 else None

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
            exp_name=exp_name, seed=seed, seed_path=seed_path,
            sweep_val=sweep_val, sweep_val_2=sweep_val_2,
            final_val_loss=fvl,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
            run_config=rc,
        ))

df = pd.DataFrame(records)
sweep_vals = sorted(df["sweep_val"].unique())
print(f"Loaded {len(df)} runs over {SWEEP_VAR} = {sweep_vals}")
if SWEEP_VAR_2:
    sweep_vals_2 = sorted(df["sweep_val_2"].unique())
    print(f"  also varying {SWEEP_VAR_2} = {sweep_vals_2}")

# %% Extract task parameters from first config
first_config = df.iloc[0]["run_config"] if len(df) > 0 else {}
n_pairs = first_config.get("n_pairs", 1)
periods_cfg = first_config.get("periods", first_config.get("period", 10.0))
if isinstance(periods_cfg, (int, float)):
    periods_list = [float(periods_cfg)] * n_pairs
else:
    periods_list = [float(p) for p in periods_cfg]
n_channels = 2 * n_pairs
init_hidden_value = first_config.get("init_hidden_value", 1.0)

channel_labels = []
for k in range(n_pairs):
    channel_labels.append(f"cos (T={periods_list[k]:.0f})")
    channel_labels.append(f"sin (T={periods_list[k]:.0f})")
ch_colors = [f"C{i}" for i in range(n_channels)]

print(f"n_pairs={n_pairs}, periods={periods_list}, init_hidden={init_hidden_value}")

# %% Color palette - one color per (primary) sweep value
def _val_to_str(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)

palette = plt.cm.viridis(np.linspace(0.15, 0.95, len(sweep_vals)))
COLORS = {v: palette[i] for i, v in enumerate(sweep_vals)}

# %% Helper: load full model from a row
def _load_model(row):
    rc = row["run_config"]
    seed_path = row["seed_path"]
    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        return None, None, rc
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
        model=model,
        learning_rate=rc["learning_rate"],
        weight_decay=rc["weight_decay"],
        step_size=rc.get("lr_step_size", rc.get("step_size", 1000)),
        gamma=rc["gamma"],
        task="sine_wave",
        init_hidden_value=rc.get("init_hidden_value", 1.0),
    )
    lit.load_state_dict(sd)
    lit.eval()
    return model, lit, rc

# %% Helper: extract W_rec, W_out from checkpoint without rebuilding
def _extract_weights(seed_path):
    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
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

# %% [markdown]
# ## 1. Training curves

# %% Plot: Validation loss vs training step
fig, ax = plt.subplots(figsize=(10, 5))
plotted = set()
for _, row in df.iterrows():
    v = row["sweep_val"]
    vl = row["val_losses"]
    st = row["steps"][:len(vl)] if row["steps"] else list(range(1, len(vl) + 1))
    label = f"{SWEEP_VAR} = {_val_to_str(v)}" if v not in plotted else None
    ax.plot(st, vl, linewidth=1.4, color=COLORS[v], alpha=0.7, label=label)
    plotted.add(v)
ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation loss (MSE)", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2)
fig.suptitle(f"Validation Loss by {SWEEP_VAR_LABEL}",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "val_loss.pdf"), bbox_inches="tight", dpi=150)
plt.show()

# %% Plot: Validation R^2 vs training step
fig, ax = plt.subplots(figsize=(10, 5))
plotted = set()
for _, row in df.iterrows():
    v = row["sweep_val"]
    va = row["val_accs"]
    st = row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1))
    if not va:
        continue
    label = f"{SWEEP_VAR} = {_val_to_str(v)}" if v not in plotted else None
    ax.plot(st, va, linewidth=1.4, color=COLORS[v], alpha=0.7, label=label)
    plotted.add(v)
ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation $R^2$", fontsize=12)
ax.set_ylim(-0.1, 1.05)
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2, loc="lower right")
fig.suptitle(f"Validation $R^2$ by {SWEEP_VAR_LABEL}",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "val_r2.pdf"), bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Per-run diagnostics

# %% Diagnostic functions
def compute_eig_scorecard(eigs, periods, dt,
                          radial_tol=EIG_RADIAL_TOL,
                          angular_tol_frac=EIG_ANGULAR_TOL_FRAC):
    """For each target period, count eigenvalues near e^{+/- i omega dt}.

    For each (sign, target) pair we greedily match the closest
    not-yet-claimed eigenvalue that satisfies both the radial and angular
    tolerance.  Returns (per_pair_count, total).  Max per pair = 2.
    """
    abs_eigs = np.abs(eigs)
    angles = np.angle(eigs)
    per_pair = np.zeros(len(periods), dtype=int)
    matched = np.zeros(len(eigs), dtype=bool)
    for k, T in enumerate(periods):
        omega_dt = 2.0 * np.pi * dt / T
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
    """Effective number of eigenmodes the readout draws from.

    s_k = ||(W_out V)_{:,k}||_2; PR = (sum s_k)^2 / sum s_k^2.  Modes with
    s_k below thresh_frac * max(s) are zeroed out before computing PR.
    """
    M = W_out @ V
    s = np.linalg.norm(M, axis=0)
    if s.max() <= 0:
        return 0.0
    s = s.copy()
    s[s < thresh_frac * s.max()] = 0.0
    num = s.sum() ** 2
    den = (s ** 2).sum()
    return float(num / max(den, 1e-30))


def compute_extrapolation_r2(model, rc, periods, rollout_steps):
    """Run autonomous rollout and compute in/out R^2 vs sinusoidal targets."""
    n_p = rc.get("n_pairs", 1)
    dt = rc["dt"]
    dummy_inp = torch.zeros(1, rollout_steps, 1)
    h_init = torch.full((1, rc["hidden_size"]),
                       rc.get("init_hidden_value", 1.0))
    with torch.no_grad():
        _, out = model(dummy_inp, init_hidden=h_init)
    out_np = out[0].numpy()

    t_ax = np.arange(rollout_steps) * dt
    targets = np.zeros((rollout_steps, 2 * n_p))
    for k in range(n_p):
        phase = 2.0 * np.pi * t_ax / periods[k]
        targets[:, 2 * k] = np.cos(phase)
        targets[:, 2 * k + 1] = np.sin(phase)

    train_T = rc["num_time_steps"]
    in_mask = np.arange(rollout_steps) < train_T
    out_mask = np.arange(rollout_steps) >= train_T

    def _r2(tgt, pred, mask):
        if not mask.any():
            return np.nan
        t = tgt[mask]
        p = pred[mask]
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean(axis=0)) ** 2)
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    return (_r2(targets, out_np, in_mask),
            _r2(targets, out_np, out_mask),
            out_np, targets, train_T * dt)

# %% Compute diagnostics for every run
metrics = []
for _, row in df.iterrows():
    seed_path = row["seed_path"]
    rc = row["run_config"]
    n_p = rc.get("n_pairs", 1)
    dt_rc = rc["dt"]
    train_T = rc["num_time_steps"]
    rollout_steps = ROLLOUT_MULTIPLIER * train_T

    W_rec, W_out = _extract_weights(seed_path)
    if W_rec is None:
        continue
    N_dim = W_rec.shape[0]
    tau_val = rc["time_constants_config"]["values"][0]
    alpha = 1.0 - np.exp(-dt_rc / tau_val)
    g_rc = rc["recurrent_gain"]
    J = (1 - alpha) * np.eye(N_dim) + alpha * g_rc * W_rec
    eigs, V = np.linalg.eig(J)
    abs_eigs = np.abs(eigs)
    rho = float(abs_eigs.max())
    n_unstable = int((abs_eigs > 1.0).sum())

    per_pair_found, n_found = compute_eig_scorecard(eigs, periods_list, dt_rc)
    pr = compute_readout_participation(W_out, V) if W_out is not None else np.nan

    model, lit, _ = _load_model(row)
    r2_in = r2_out = np.nan
    if model is not None:
        r2_in, r2_out, _, _, _ = compute_extrapolation_r2(
            model, rc, periods_list, rollout_steps
        )

    metrics.append(dict(
        sweep_val=row["sweep_val"],
        sweep_val_2=row["sweep_val_2"],
        seed=row["seed"],
        seed_path=seed_path,
        N=N_dim,
        rho=rho,
        n_unstable=n_unstable,
        n_found=n_found,
        n_target=2 * n_p,
        per_pair_found=per_pair_found.tolist(),
        readout_pr=pr,
        r2_in=r2_in,
        r2_out=r2_out,
        train_T=train_T,
        dt=dt_rc,
        n_cycles_slow=train_T * dt_rc / max(periods_list),
        samples_per_period_fast=min(periods_list) / dt_rc,
    ))
    print(f"{row['exp_name']} seed={row['seed']}: "
          f"rho={rho:.4f}, n_unstable={n_unstable}, "
          f"modes_found={n_found}/{2*n_p}, PR={pr:.1f}, "
          f"R2_in={r2_in:.3f}, R2_out={r2_out:.3f}")

metrics_df = pd.DataFrame(metrics)

# %% [markdown]
# ## 3. Summary plots -- diagnostics vs sweep variable

# %% Aggregate by sweep value
agg = (
    metrics_df.groupby("sweep_val")
    .agg({"rho": ["mean", "std"],
          "n_unstable": ["mean"],
          "n_found": ["mean"],
          "readout_pr": ["mean", "std"],
          "r2_in": ["mean", "std"],
          "r2_out": ["mean", "std"]})
    .sort_index()
)
fvl_agg = df.groupby("sweep_val")["final_val_loss"].agg(["mean", "std"])

# %% Plot: 6-panel summary vs sweep variable
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

x = list(agg.index)
x_is_numeric = all(isinstance(v, (int, float)) for v in x)

ax = axes[0, 0]
ax.errorbar(x, fvl_agg["mean"].reindex(x), yerr=fvl_agg["std"].reindex(x),
            fmt="o-", capsize=4, color="#264653")
ax.set_ylabel("Final val loss (MSE)")
ax.set_yscale("log")
ax.set_title("Training-window loss")
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.errorbar(x, agg[("r2_in", "mean")], yerr=agg[("r2_in", "std")],
            fmt="o-", capsize=4, label="train window", color="#2a9d8f")
ax.errorbar(x, agg[("r2_out", "mean")], yerr=agg[("r2_out", "std")],
            fmt="s--", capsize=4, label="post-training", color="#e76f51")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6)
ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.6)
ax.set_ylabel("$R^2$")
ax.set_ylim(-1.5, 1.1)
ax.set_title(f"Generalisation (rollout = {ROLLOUT_MULTIPLIER}x train)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.errorbar(x, agg[("rho", "mean")], yerr=agg[("rho", "std")],
            fmt="o-", capsize=4, color="#264653")
ax.axhline(1.0, color="red", linestyle=":", linewidth=1, alpha=0.7,
           label="unit circle")
ax.set_ylabel(r"$\rho(J) = \max_k |\lambda_k|$")
ax.set_title("Spectral radius")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(x, agg[("n_unstable", "mean")], "o-", color="#c1121f")
ax.set_ylabel(r"# eigenvalues with $|\lambda| > 1$")
ax.set_title("Unstable modes (memorisation signature)")
ax.set_xlabel(SWEEP_VAR_LABEL)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(x, agg[("n_found", "mean")], "o-", color="#2a9d8f")
ax.axhline(2 * n_pairs, color="black", linestyle=":", linewidth=1,
           alpha=0.7, label=f"target = {2*n_pairs}")
ax.set_ylabel("# eigenvalues near targets")
ax.set_title(f"Eigenvalue scorecard "
             f"(rad tol={EIG_RADIAL_TOL}, ang frac={EIG_ANGULAR_TOL_FRAC})")
ax.set_xlabel(SWEEP_VAR_LABEL)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.errorbar(x, agg[("readout_pr", "mean")], yerr=agg[("readout_pr", "std")],
            fmt="o-", capsize=4, color="#264653")
ax.axhline(2 * n_pairs, color="green", linestyle=":", linewidth=1, alpha=0.7,
           label=f"placement = {2*n_pairs}")
ax.set_ylabel("Readout PR")
ax.set_yscale("log")
ax.set_title("Readout participation ratio (modes used)")
ax.set_xlabel(SWEEP_VAR_LABEL)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

for ax in axes[0, :]:
    ax.set_xlabel(SWEEP_VAR_LABEL)

fig.suptitle(f"Diagnostics vs {SWEEP_VAR_LABEL}",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "diagnostics_summary.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 4. 2-D heatmap (only if SWEEP_VAR_2 is set)

# %%
if SWEEP_VAR_2 is not None and metrics_df["sweep_val_2"].notna().any():
    pivot_in = (metrics_df.groupby(["sweep_val", "sweep_val_2"])["r2_in"]
                .mean().unstack("sweep_val_2")
                .sort_index(axis=0).sort_index(axis=1))
    pivot_out = (metrics_df.groupby(["sweep_val", "sweep_val_2"])["r2_out"]
                 .mean().unstack("sweep_val_2")
                 .sort_index(axis=0).sort_index(axis=1))
    pivot_pr = (metrics_df.groupby(["sweep_val", "sweep_val_2"])["readout_pr"]
                .mean().unstack("sweep_val_2")
                .sort_index(axis=0).sort_index(axis=1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    panels = [
        (axes[0], pivot_in, "$R^2$ (train window)", 0, 1, "viridis", False),
        (axes[1], pivot_out, "$R^2$ (post-training)", -0.5, 1, "RdBu_r", False),
        (axes[2], pivot_pr, "Readout PR (log)", None, None, "magma", True),
    ]

    for ax, mat, title, vmin, vmax, cmap, log_scale in panels:
        if log_scale:
            data = np.log10(np.clip(mat.values.astype(float), 1.0, None))
            im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
        else:
            im = ax.imshow(mat.values.astype(float), aspect="auto",
                           origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([_val_to_str(v) for v in mat.columns], rotation=45)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([_val_to_str(v) for v in mat.index])
        ax.set_xlabel(SWEEP_VAR_2_LABEL or SWEEP_VAR_2)
        ax.set_ylabel(SWEEP_VAR_LABEL)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if abs(v) < 0.5 else "black")

    fig.suptitle(f"2-D Sweep: {SWEEP_VAR_LABEL} x {SWEEP_VAR_2_LABEL or SWEEP_VAR_2}",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, "heatmap_2d.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()
else:
    print("(Skipping 2-D heatmap -- SWEEP_VAR_2 not set or all NaN)")

# %% [markdown]
# ## 5. Extended rollout -- best run per sweep value

# %%
fig, axes = plt.subplots(len(sweep_vals), 1,
                         figsize=(14, 2.6 * len(sweep_vals)),
                         squeeze=False, sharex=False)

for si, v in enumerate(sweep_vals):
    ax = axes[si, 0]
    sub = df[df["sweep_val"] == v]
    row = sub.loc[sub["final_val_loss"].idxmin()]
    model, lit, rc = _load_model(row)
    if model is None:
        ax.text(0.5, 0.5, "no checkpoint", ha="center", va="center",
                transform=ax.transAxes)
        continue

    train_T = rc["num_time_steps"]
    rollout_steps = ROLLOUT_MULTIPLIER * train_T
    n_p = rc.get("n_pairs", 1)
    dt_rc = rc["dt"]

    _, _, out_np, targets, train_T_t = compute_extrapolation_r2(
        model, rc, periods_list, rollout_steps
    )
    t_ax = np.arange(rollout_steps) * dt_rc

    for k in range(n_p):
        ci = 2 * k
        ax.plot(t_ax, targets[:, ci], "--", color=ch_colors[ci],
                linewidth=0.5, alpha=0.4)
        ax.plot(t_ax, out_np[:, ci], color=ch_colors[ci], linewidth=1.0,
                label=f"cos T={periods_list[k]:.0f}" if si == 0 else None)
        ax.plot(t_ax, targets[:, ci + 1], "--", color=ch_colors[ci + 1],
                linewidth=0.5, alpha=0.4)
        ax.plot(t_ax, out_np[:, ci + 1], color=ch_colors[ci + 1], linewidth=1.0,
                label=f"sin T={periods_list[k]:.0f}" if si == 0 else None)

    ax.axvline(train_T_t, color="black", linestyle=":", linewidth=1.5,
               alpha=0.7, label="train horizon" if si == 0 else None)
    ax.set_ylabel(f"{SWEEP_VAR}={_val_to_str(v)}", fontsize=10, fontweight="bold")
    ax.set_ylim(-2.0, 2.0)
    ax.grid(True, alpha=0.2)
    if si == 0:
        ax.legend(fontsize=7, ncol=n_p + 1, loc="upper right")

axes[-1, 0].set_xlabel("Time", fontsize=12)
fig.suptitle(f"Extended Rollout vs {SWEEP_VAR_LABEL} "
             f"(rollout = {ROLLOUT_MULTIPLIER}x train)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "extended_rollout.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Eigenspectra -- best run per sweep value

# %%
theta_circle = np.linspace(0, 2 * np.pi, 200)
pair_colors = ["#2a9d8f", "#e76f51", "#264653",
               "#a855f7", "#f59e0b", "#0ea5e9"][:n_pairs]

ncols_eig = min(3, len(sweep_vals))
nrows_eig = (len(sweep_vals) + ncols_eig - 1) // ncols_eig
fig, axes = plt.subplots(nrows_eig, ncols_eig,
                         figsize=(5.5 * ncols_eig, 5.5 * nrows_eig),
                         squeeze=False)

for si, v in enumerate(sweep_vals):
    ax = axes[si // ncols_eig, si % ncols_eig]
    sub = df[df["sweep_val"] == v]
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
    ax.set_title(f"{SWEEP_VAR}={_val_to_str(v)}  rho={rho:.3f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.15)

for i in range(len(sweep_vals), nrows_eig * ncols_eig):
    axes[i // ncols_eig, i % ncols_eig].set_visible(False)

fig.suptitle(f"Global Jacobian Eigenspectra by {SWEEP_VAR_LABEL}",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "eigenspectra.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Save metrics dataframe
if SAVE_FIGS:
    metrics_df.to_csv(os.path.join(FIGS_DIR, "metrics.csv"), index=False)
    print(f"Saved metrics to {os.path.join(FIGS_DIR, 'metrics.csv')}")
print(metrics_df.to_string())
