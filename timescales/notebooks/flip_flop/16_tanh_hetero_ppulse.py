# %% [markdown]
# # Tanh Heterogeneous Flip-Flop — Gain Sweep Analysis
#
# N-bit flip-flop with heterogeneous `p_pulse` and **Tanh** activation.
# We sweep the recurrent gain $g$ across values that span from overdamped
# ($g < 1$) to supercritical ($g > 1$); tanh saturation keeps dynamics
# bounded even for $g > 1$.
#
# **Main analyses:**
# 1. Training curves (loss / accuracy) and per-bit accuracy
# 2. Fixed-point finding via L-BFGS at each gain
# 3. Jacobian eigendecomposition at each fixed point
# 4. Effective timescale ($\tau_\text{eff}$) spectra across gains

# %%
import os
import sys
import subprocess
import json
import glob
import time

import yaml
import numpy as np
from scipy.optimize import minimize as sp_minimize
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
FIGS_DIR = os.path.join("notebooks", "figs", "tanh_hetero")
os.makedirs(FIGS_DIR, exist_ok=True)

# %% Specify sweep directory
sweep_dir = "/home/facosta/timescales/timescales/logs/experiments/flip_flop_hetero_ppulse_tanh_20260413_175256"

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

        val_losses, val_accs, steps = [], [], []
        val_losses_per_bit, val_accs_per_bit = {}, {}
        lf = os.path.join(seed_path, "training_losses.json")
        if os.path.exists(lf):
            with open(lf) as f:
                ld = json.load(f)
            val_losses = ld.get("val_losses", [])
            val_accs = ld.get("val_accuracies", [])
            steps = ld.get("steps", [])
            val_losses_per_bit = ld.get("val_losses_per_bit", {})
            val_accs_per_bit = ld.get("val_accuracies_per_bit", {})

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
            exp_name=exp_name, gain=gain, seed=seed,
            seed_path=seed_path, final_val_loss=fvl,
            val_losses=val_losses, val_accs=val_accs, steps=steps,
            val_losses_per_bit=val_losses_per_bit,
            val_accs_per_bit=val_accs_per_bit,
            run_config=run_config,
        ))

df = pd.DataFrame(records)
gains = sorted(df["gain"].unique())
print(f"Loaded {len(df)} runs, gains: {gains}")

# %% Color palette
palette = plt.cm.viridis(np.linspace(0.15, 0.95, len(gains)))
COLORS = {g: palette[i] for i, g in enumerate(gains)}

# %% Plot 1: Validation loss vs training step
fig, ax = plt.subplots(figsize=(10, 5))
plotted = set()
for _, row in df.iterrows():
    g = row["gain"]
    vl = row["val_losses"]
    st = row["steps"][:len(vl)] if row["steps"] else list(range(1, len(vl) + 1))
    label = f"g = {g}" if g not in plotted else None
    ax.plot(st, vl, linewidth=1.4, color=COLORS[g], alpha=0.7, label=label)
    plotted.add(g)

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation loss", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2)
fig.suptitle("Tanh RNN — Validation Loss by Gain",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "tanh_val_loss.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 2: Validation accuracy vs training step
fig, ax = plt.subplots(figsize=(10, 5))
plotted = set()
for _, row in df.iterrows():
    g = row["gain"]
    va = row["val_accs"]
    st = row["steps"][:len(va)] if row["steps"] else list(range(1, len(va) + 1))
    if not va:
        continue
    label = f"g = {g}" if g not in plotted else None
    ax.plot(st, va, linewidth=1.4, color=COLORS[g], alpha=0.7, label=label)
    plotted.add(g)

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Validation accuracy", fontsize=12)
ax.set_ylim(0.4, 1.02)
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2, loc="lower right")
fig.suptitle("Tanh RNN — Validation Accuracy by Gain",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "tanh_val_accuracy.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 3: Final val loss vs gain
summary = (
    df.groupby("gain")["final_val_loss"]
    .agg(["mean", "std", "count"])
    .sort_index()
)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(summary.index, summary["mean"], yerr=summary["std"],
            fmt="o-", capsize=4, linewidth=1.8, markersize=7,
            color="#264653", ecolor="#adb5bd")
ax.set_xlabel("Recurrent gain $g$", fontsize=12)
ax.set_ylabel("Final validation loss", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
           label="$g = 1$")
ax.legend(fontsize=10)
fig.suptitle("Tanh RNN — Final Loss vs Gain",
             fontsize=13, fontweight="bold", y=1.04)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "tanh_final_loss_vs_gain.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 4: Per-bit accuracy curves (one subplot per gain)
first_config = df.iloc[0]["run_config"] if len(df) > 0 else {}
n_bits = first_config.get("n_bits", 6)
p_pulse = first_config.get("p_pulse", 0.05)
pp_list = p_pulse if isinstance(p_pulse, list) else [p_pulse] * n_bits
holds = [1.0 / p for p in pp_list]

ncols_pb = min(4, len(gains))
nrows_pb = (len(gains) + ncols_pb - 1) // ncols_pb
fig, axes = plt.subplots(nrows_pb, ncols_pb,
                         figsize=(5 * ncols_pb, 3.5 * nrows_pb),
                         squeeze=False, sharey=True)
bit_colors = [f"C{i}" for i in range(n_bits)]

for gi, g in enumerate(gains):
    ax = axes[gi // ncols_pb, gi % ncols_pb]
    sub = df[df["gain"] == g]
    for _, row in sub.iterrows():
        apb = row["val_accs_per_bit"]
        if not apb:
            continue
        st = row["steps"]
        for ch_key, vals in apb.items():
            if not vals:
                continue
            bit_idx = int(ch_key.replace("channel_", ""))
            x = st[:len(vals)] if st else list(range(1, len(vals) + 1))
            ax.plot(x, vals, linewidth=1.2, color=bit_colors[bit_idx],
                    alpha=0.6)
    ax.set_title(f"g = {g}", fontsize=11, fontweight="bold")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.grid(True, alpha=0.2)
    if gi // ncols_pb == nrows_pb - 1:
        ax.set_xlabel("Step", fontsize=10)
    if gi % ncols_pb == 0:
        ax.set_ylabel("Per-bit accuracy", fontsize=10)

for i in range(len(gains), nrows_pb * ncols_pb):
    axes[i // ncols_pb, i % ncols_pb].set_visible(False)

legend_handles = [Line2D([0], [0], color=bit_colors[i], linewidth=2,
                         label=f"Bit {i} (hold≈{holds[i]:.0f})")
                  for i in range(n_bits)]
axes[0, -1].legend(handles=legend_handles, fontsize=7.5, loc="lower right")
fig.suptitle("Tanh RNN — Per-Bit Accuracy During Training",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR, "tanh_per_bit_accuracy_curves.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 5: Final per-bit accuracy vs gain (bar chart)
bit_acc_records = []
for _, row in df.iterrows():
    apb = row["val_accs_per_bit"]
    if not apb:
        continue
    for ch_key, vals in apb.items():
        if vals:
            bit_idx = int(ch_key.replace("channel_", ""))
            bit_acc_records.append(dict(
                gain=row["gain"], seed=row["seed"],
                bit=bit_idx, final_acc=vals[-1],
            ))

if bit_acc_records:
    df_bit = pd.DataFrame(bit_acc_records)
    fig, ax = plt.subplots(figsize=(9, 5))
    for bit_i in range(n_bits):
        sub = df_bit[df_bit["bit"] == bit_i]
        means = sub.groupby("gain")["final_acc"].mean()
        stds = sub.groupby("gain")["final_acc"].std().fillna(0)
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    fmt="o-", capsize=3, linewidth=1.4, markersize=5,
                    color=bit_colors[bit_i],
                    label=f"Bit {bit_i} (p={pp_list[bit_i]}, hold≈{holds[bit_i]:.0f})")

    ax.set_xlabel("Recurrent gain $g$", fontsize=12)
    ax.set_ylabel("Final validation accuracy", fontsize=12)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="chance (50%)")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.suptitle("Tanh RNN — Per-Bit Final Accuracy vs Gain",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIGS_DIR, "tanh_per_bit_final_accuracy.pdf"),
                    bbox_inches="tight", dpi=150)
    plt.show()
else:
    print("No per-bit accuracy data found (re-run sweep with updated callback)")

# %% [markdown]
# ## Fixed-Point Analysis
#
# For a Tanh rate network, the state update is:
# $$r_{t+1} = (1-\alpha) r_t + \alpha \tanh(g W_\text{rec} r_t + W_\text{in} u_t + b)$$
#
# At a fixed point $x^*$ with zero input ($u = 0$), we have:
# $$r^* = (1-\alpha) r^* + \alpha \tanh(g W_\text{rec} r^* + b)$$
# $$\Leftrightarrow \quad r^* = \tanh(g W_\text{rec} r^* + b)$$
#
# We find these by minimizing $q(r) = \|r - \tanh(g W_\text{rec} r + b)\|^2$.

# %% Fixed-point finding — all gains
N_FP_INITS = 500
N_RANDOM_INITS = 200       # additional random Gaussian inits to cover saddle points
FP_CLUSTER_THRESH = 1.0
FP_MAXITER = 5000
FP_FTOL = 1e-20
FP_GTOL = 1e-12
FP_Q_THRESH = 1e-10        # max q(r) to count as converged
GAINS_TO_FIND_FP = None    # None = all gains; or set e.g. [0.95, 1.0, 1.2, 1.5]

fp_data = {}

gains_to_find = gains if GAINS_TO_FIND_FP is None else [g for g in GAINS_TO_FIND_FP if g in set(df["gain"])]

for g in gains_to_find:
    sub = df[df["gain"] == g]
    if sub.empty:
        continue
    row = sub.iloc[0]
    seed_path = row["seed_path"]
    rc = row["run_config"]

    best_ckpts = glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not best_ckpts:
        print(f"g={g}: no checkpoint, skipping")
        continue

    ckpt = torch.load(best_ckpts[0], map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    W_rec = W_out = W_in = b_out = b_rec = b_in = None
    for key, val in sd.items():
        if "W_rec.weight" in key:
            W_rec = val.numpy()
        elif "W_out.weight" in key or "readout.weight" in key:
            W_out = val.numpy()
        elif "W_in.weight" in key:
            W_in = val.numpy()
        elif "W_out.bias" in key or "readout.bias" in key:
            b_out = val.numpy()
        elif "W_rec.bias" in key:
            b_rec = val.numpy()
        elif "W_in.bias" in key:
            b_in = val.numpy()

    if W_rec is None:
        continue

    N = W_rec.shape[0]
    dt = rc["dt"]
    tau_val = rc["time_constants_config"]["values"][0]
    alpha = 1.0 - np.exp(-dt / tau_val)

    gW = g * W_rec
    bias = np.zeros(N)
    if b_rec is not None:
        bias = bias + g * b_rec
    if b_in is not None:
        bias = bias + b_in

    # Generate initial conditions from trained model trajectories
    model = RNN(
        input_size=rc["n_bits"],
        hidden_size=rc["hidden_size"],
        output_size=rc["n_bits"],
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
        task="flip_flop",
    )
    lit.load_state_dict(sd)
    lit.eval()
    lit.to(device)

    dm = FlipFlopDataModule(
        n_bits=rc["n_bits"],
        p_pulse=rc["p_pulse"],
        pulse_amplitude=rc["pulse_amplitude"],
        num_time_steps=rc["num_time_steps"],
        num_val_trajectories=100,
        batch_size=100,
    )
    dm.setup()
    inp, _, tgt = dm.val_dataset.tensors

    print(f"g={g}: generating trajectory initial conditions on {device} ...")
    t_fwd = time.time()
    with torch.no_grad():
        h_seq, _ = lit.model(inp.to(device), init_context=None)
        hidden_all = h_seq.cpu().numpy().reshape(-1, N)
    print(f"  forward pass done in {time.time()-t_fwd:.1f}s")

    rng = np.random.RandomState(42)
    n_traj_inits = min(N_FP_INITS, len(hidden_all))
    init_idx = rng.choice(len(hidden_all), size=n_traj_inits, replace=False)
    traj_inits = hidden_all[init_idx]

    h_std = hidden_all.std()
    random_inits = rng.randn(N_RANDOM_INITS, N).astype(np.float64) * h_std

    init_points = np.vstack([np.zeros((1, N)), traj_inits, random_inits])
    print(f"  {len(init_points)} inits: 1 origin + {n_traj_inits} trajectory "
          f"+ {N_RANDOM_INITS} random")

    gW_T = gW.T.copy()

    def _fp_speed(r):
        res = r - np.tanh(gW @ r + bias)
        return 0.5 * np.dot(res, res)

    def _fp_grad(r):
        act = np.tanh(gW @ r + bias)
        res = r - act
        sech2 = 1.0 - act ** 2
        return res - gW_T @ (sech2 * res)

    print(f"g={g}: running {len(init_points)} independent L-BFGS inits "
          f"(N={N}, maxiter={FP_MAXITER}) ...")
    t0 = time.time()
    converged = []
    report_every = max(1, len(init_points) // 10)
    for ii, r0 in enumerate(init_points):
        res = sp_minimize(_fp_speed, r0, jac=_fp_grad, method="L-BFGS-B",
                          options={"maxiter": FP_MAXITER, "ftol": FP_FTOL,
                                   "gtol": FP_GTOL})
        if res.fun < FP_Q_THRESH:
            converged.append(res.x)
        if (ii + 1) % report_every == 0:
            elapsed = time.time() - t0
            print(f"  [{ii+1}/{len(init_points)}] {elapsed:.1f}s elapsed, "
                  f"{len(converged)} converged so far")

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s — {len(converged)}/{len(init_points)} converged")

    if not converged:
        print(f"g={g}: no fixed points converged")
        fp_data[g] = dict(fps=np.array([]), labels=[], W_out=W_out,
                          b_out=b_out, gW=gW, bias=bias, alpha=alpha, N=N,
                          W_in=W_in, rc=rc)
        continue

    converged = np.array(converged)

    unique_fps = [converged[0]]
    for r in converged[1:]:
        dists = [np.linalg.norm(r - u) for u in unique_fps]
        if min(dists) > FP_CLUSTER_THRESH:
            unique_fps.append(r)
    unique_fps = np.array(unique_fps)

    if len(unique_fps) > 1:
        from scipy.spatial.distance import pdist
        pw = pdist(unique_fps)
        print(f"  pairwise distances: min={pw.min():.3f}, "
              f"median={np.median(pw):.3f}, max={pw.max():.3f}")
        if pw.min() < 2 * FP_CLUSTER_THRESH:
            print(f"  ⚠ closest pair ({pw.min():.3f}) < 2×thresh "
                  f"({2*FP_CLUSTER_THRESH:.3f}) — consider raising threshold")

    labels = []
    if W_out is not None:
        for r in unique_fps:
            logit = W_out @ r + (b_out if b_out is not None else 0)
            out = 1.0 / (1.0 + np.exp(-logit))
            labels.append("".join(str(int(v > 0.5)) for v in out))
    else:
        labels = [str(i) for i in range(len(unique_fps))]

    is_stable = []
    for r in unique_fps:
        pre_r = gW @ r + bias
        sech2_r = 1.0 - np.tanh(pre_r) ** 2
        J_r = np.diag(np.full(N, 1 - alpha)) + alpha * np.diag(sech2_r) @ gW
        max_abs_eig = np.max(np.abs(np.linalg.eigvals(J_r)))
        is_stable.append(max_abs_eig < 1.0)

    n_stable = sum(is_stable)
    n_unstable = len(is_stable) - n_stable
    n_distinct_labels = len(set(labels))
    print(f"g={g}: {len(converged)}/{len(init_points)} converged → "
          f"{len(unique_fps)} unique FPs ({n_stable} stable, {n_unstable} unstable, "
          f"{n_distinct_labels} distinct labels)")

    fp_data[g] = dict(fps=unique_fps, labels=labels, is_stable=is_stable,
                      W_out=W_out, b_out=b_out, gW=gW, bias=bias,
                      alpha=alpha, N=N, W_in=W_in, rc=rc,
                      hidden_all=hidden_all)

# %% Print fixed-point summary table
print(f"\n{'g':>5s} {'#FPs':>5s} {'stable':>6s} {'unst.':>5s} {'#labels':>7s}  Labels")
print("-" * 80)
for g in gains_to_find:
    if g not in fp_data:
        continue
    d = fp_data[g]
    n_fp = len(d["fps"])
    n_st = sum(d["is_stable"])
    n_un = n_fp - n_st
    n_labels = len(set(d["labels"]))
    lab_parts = []
    for i, lab in enumerate(d["labels"][:30]):
        marker = "" if d["is_stable"][i] else "*"
        lab_parts.append(f"{lab}{marker}")
    suffix = " ..." if n_fp > 30 else ""
    print(f"{g:5.2f} {n_fp:5d} {n_st:6d} {n_un:5d} {n_labels:7d}  "
          f"{', '.join(lab_parts)}{suffix}")
print("(* = unstable)")

# %% Plot: 3D PCA projection of fixed points and trajectories
from sklearn.decomposition import PCA

FP_VIS_GAIN = gains_to_find[0]  # which gain to visualize; change as needed

assert FP_VIS_GAIN in fp_data and len(fp_data[FP_VIS_GAIN]["fps"]) > 0, \
    f"No FPs for g={FP_VIS_GAIN}"

d_vis = fp_data[FP_VIS_GAIN]
fps_vis = d_vis["fps"]
labels_vis = d_vis["labels"]
stable_vis = d_vis["is_stable"]
hidden_vis = d_vis["hidden_all"]

pca = PCA(n_components=3)
pca.fit(hidden_vis)
var_explained = pca.explained_variance_ratio_

fps_pca = pca.transform(fps_vis)
traj_pca = pca.transform(hidden_vis)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(traj_pca[::5, 0], traj_pca[::5, 1], traj_pca[::5, 2],
           s=0.3, alpha=0.04, c="#999", depthshade=False, rasterized=True)

stable_mask = np.array(stable_vis)
if stable_mask.any():
    ax.scatter(fps_pca[stable_mask, 0], fps_pca[stable_mask, 1],
               fps_pca[stable_mask, 2],
               s=120, c="#2a9d8f", edgecolors="black", linewidths=1,
               zorder=5, label="stable")
if (~stable_mask).any():
    ax.scatter(fps_pca[~stable_mask, 0], fps_pca[~stable_mask, 1],
               fps_pca[~stable_mask, 2],
               s=80, c="#e76f51", edgecolors="black", linewidths=0.8,
               marker="^", zorder=5, label="unstable")

for i, lab in enumerate(labels_vis):
    ax.text(fps_pca[i, 0], fps_pca[i, 1], fps_pca[i, 2],
            f"  {lab}", fontsize=7, zorder=6)

ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})", fontsize=10)
ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})", fontsize=10)
ax.set_zlabel(f"PC3 ({var_explained[2]:.1%})", fontsize=10)
total_var = var_explained[:3].sum()
ax.set_title(f"Fixed Points in PCA Space — g={FP_VIS_GAIN}\n"
             f"{len(fps_vis)} FPs ({sum(stable_vis)} stable, "
             f"{len(fps_vis)-sum(stable_vis)} unstable), "
             f"3 PCs explain {total_var:.1%} variance",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR,
                f"tanh_fp_pca_g{FP_VIS_GAIN}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Jacobian eigendecomposition at a selected fixed point
TAU_VIS_MAX = None  # upper cap on τ_eff for plots; None = no cap
FP_GAIN = 0.5   # which gain to analyze
FP_IDX = 0      # which fixed point (index into fp_data[FP_GAIN]["fps"])

assert FP_GAIN in fp_data and len(fp_data[FP_GAIN]["fps"]) > FP_IDX, \
    f"FP_GAIN={FP_GAIN}, FP_IDX={FP_IDX} not available. " \
    f"Available: {[(g, len(fp_data[g]['fps'])) for g in fp_data]}"

fp_info = fp_data[FP_GAIN]
r_star = fp_info["fps"][FP_IDX]
gW_fp = fp_info["gW"]
bias_fp = fp_info["bias"]
alpha_fp = fp_info["alpha"]
N_fp = fp_info["N"]
fp_label = fp_info["labels"][FP_IDX]

pre = gW_fp @ r_star + bias_fp
sech2 = 1.0 - np.tanh(pre) ** 2
J = np.diag(np.full(N_fp, 1 - alpha_fp)) + alpha_fp * np.diag(sech2) @ gW_fp
jac_eigs, jac_vecs = np.linalg.eig(J)

abs_eigs = np.abs(jac_eigs)
log_abs = np.log(np.clip(abs_eigs, 1e-12, None))
jac_tau_eff = -1.0 / np.clip(log_abs, None, -1e-10)

fp_output = None
if fp_info["W_out"] is not None:
    logit = fp_info["W_out"] @ r_star + (fp_info["b_out"] if fp_info["b_out"] is not None else 0)
    fp_output = 1.0 / (1.0 + np.exp(-logit))

print(f"=== Jacobian at FP {FP_IDX} (g={FP_GAIN}) ===")
print(f"  Label: {fp_label}")
if fp_output is not None:
    print(f"  Output (σ): [{', '.join(f'{v:.3f}' for v in fp_output)}]")
print(f"  ||r*|| = {np.linalg.norm(r_star):.4f}")
print(f"  q(r*) = {np.sum((r_star - np.tanh(gW_fp @ r_star + bias_fp))**2)/2:.2e}")
print(f"  Top-5 |λ|: {np.sort(abs_eigs)[-5:][::-1]}")
print(f"  Top-5 τ_eff: {np.sort(np.clip(jac_tau_eff, 0, TAU_VIS_MAX))[-5:][::-1]}")
print(f"  # eigs with |λ| > 0.99: {np.sum(abs_eigs > 0.99)}")
print(f"  # modes with τ ≥ longest hold ({holds[0]:.0f}): "
      f"{np.sum(np.clip(jac_tau_eff, 0, TAU_VIS_MAX) >= holds[0])}")

# %% Plot 6: Eigenspectrum on complex plane
hold_colors = plt.cm.tab10(np.linspace(0, 1, n_bits))
theta_circle = np.linspace(0, 2 * np.pi, 200)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.cos(theta_circle), np.sin(theta_circle),
        "k-", linewidth=0.6, alpha=0.3)
ax.axhline(0, color="gray", linewidth=0.3)
ax.axvline(0, color="gray", linewidth=0.3)

top_k = 20
top_idx = np.argsort(abs_eigs)[-top_k:]

ax.scatter(jac_eigs.real, jac_eigs.imag, s=8, color="#bbb",
           alpha=0.4, edgecolors="none", rasterized=True, label="all")
ax.scatter(jac_eigs[top_idx].real, jac_eigs[top_idx].imag,
           s=35, color="#e63946", edgecolors="black",
           linewidths=0.4, alpha=0.85, zorder=5,
           label=f"top-{top_k} by |λ|")

ax.set_title(f"Jacobian Eigenspectrum — g={FP_GAIN}, FP {FP_IDX}: "
             f"{fp_label}", fontsize=12, fontweight="bold")
ax.set_xlabel("Re(λ)", fontsize=11)
ax.set_ylabel("Im(λ)", fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.15)
ax.grid(True, alpha=0.15)
ax.legend(fontsize=9)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR,
                f"tanh_eigenspectrum_g{FP_GAIN}_fp{FP_IDX}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 7: Effective timescale scree
fig, ax = plt.subplots(figsize=(10, 5))
tau_sorted = np.sort(np.clip(jac_tau_eff, 0, TAU_VIS_MAX))[::-1]
ranks = np.arange(1, len(tau_sorted) + 1)
ax.plot(ranks, tau_sorted, linewidth=1.5, color="#264653")

for bi in range(n_bits):
    ax.axhline(holds[bi], color=hold_colors[bi], linestyle="--",
               linewidth=0.8, alpha=0.6,
               label=f"bit {bi} hold ≈ {holds[bi]:.0f}")

ax.set_xlabel("Eigenmode rank", fontsize=12)
ax.set_ylabel("$\\tau_{\\mathrm{eff}}$", fontsize=13)
ax.set_yscale("log")
cap_str = f" (capped at {TAU_VIS_MAX})" if TAU_VIS_MAX is not None else ""
ax.set_title(f"τ_eff Scree — g={FP_GAIN}, FP {FP_IDX}: {fp_label}{cap_str}",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.15)
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR,
                f"tanh_tau_scree_g{FP_GAIN}_fp{FP_IDX}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 8: |λ| rank
fig, ax = plt.subplots(figsize=(10, 4))
abs_sorted = np.sort(abs_eigs)[::-1]
ranks = np.arange(1, len(abs_sorted) + 1)
ax.plot(ranks, abs_sorted, linewidth=1.5, color="#264653")
ax.axhline(1.0, color="red", linestyle=":", linewidth=1, alpha=0.5,
           label="|λ| = 1")

ax.set_xlabel("Eigenmode rank", fontsize=12)
ax.set_ylabel("$|\\lambda|$", fontsize=13)
ax.set_title(f"|λ| Rank — g={FP_GAIN}, FP {FP_IDX}: {fp_label}",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.15)
ax.legend(fontsize=9)
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(os.path.join(FIGS_DIR,
                f"tanh_eigval_mag_g{FP_GAIN}_fp{FP_IDX}.pdf"),
                bbox_inches="tight", dpi=150)
plt.show()

# %% Plot 9: Cross-gain summary
if fp_data:
    summary_gains = []
    summary_n_fps = []
    summary_max_tau = []
    summary_n_above = {bi: [] for bi in range(n_bits)}

    for g in gains_to_find:
        if g not in fp_data or len(fp_data[g].get("jac_eigs", [])) == 0:
            continue
        d = fp_data[g]
        summary_gains.append(g)
        summary_n_fps.append(len(d["fps"]))

        all_taus = np.concatenate([np.clip(jd["tau_eff"], 0, TAU_VIS_MAX)
                                   for jd in d["jac_eigs"]])
        summary_max_tau.append(np.max(all_taus))

        for bi in range(n_bits):
            n_above = sum(
                np.sum(np.clip(jd["tau_eff"], 0, TAU_VIS_MAX) >= holds[bi])
                for jd in d["jac_eigs"]
            )
            summary_n_above[bi].append(n_above)

    if summary_gains:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Panel A: number of fixed points vs gain
        axes[0].bar(range(len(summary_gains)), summary_n_fps,
                    color="#457b9d", edgecolor="white")
        axes[0].set_xticks(range(len(summary_gains)))
        axes[0].set_xticklabels([f"{g}" for g in summary_gains], fontsize=9)
        axes[0].set_xlabel("Gain $g$", fontsize=11)
        axes[0].set_ylabel("Number of fixed points", fontsize=11)
        axes[0].grid(True, alpha=0.2, axis="y")

        # Panel B: max τ_eff vs gain, with hold-time reference lines
        axes[1].plot(summary_gains, summary_max_tau, "o-",
                     color="#e76f51", linewidth=2, markersize=7)
        for bi in range(n_bits):
            axes[1].axhline(holds[bi], color=hold_colors[bi], linestyle="--",
                            linewidth=0.8, alpha=0.6,
                            label=f"hold {bi} ≈ {holds[bi]:.0f}")
        axes[1].set_xlabel("Gain $g$", fontsize=11)
        axes[1].set_ylabel("Max $\\tau_{\\mathrm{eff}}$ (across FPs)",
                           fontsize=11)
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.2)
        axes[1].legend(fontsize=6.5, loc="lower right")

        # Panel C: modes matching each hold time
        x_pos = np.arange(len(summary_gains))
        width = 0.8 / n_bits
        for bi in range(n_bits):
            offset = (bi - n_bits / 2 + 0.5) * width
            axes[2].bar(x_pos + offset, summary_n_above[bi],
                        width=width, color=hold_colors[bi],
                        label=f"Bit {bi} (hold≈{holds[bi]:.0f})")
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels([f"{g}" for g in summary_gains], fontsize=9)
        axes[2].set_xlabel("Gain $g$", fontsize=11)
        axes[2].set_ylabel(f"# modes with $\\tau \\geq$ hold time", fontsize=10)
        axes[2].grid(True, alpha=0.2, axis="y")
        axes[2].legend(fontsize=6, loc="upper left", ncol=2)

        fig.suptitle("Tanh RNN — Fixed-Point & Timescale Summary Across Gains",
                     fontsize=14, fontweight="bold", y=1.04)
        plt.tight_layout()
        if SAVE_FIGS:
            fig.savefig(os.path.join(FIGS_DIR, "tanh_cross_gain_summary.pdf"),
                        bbox_inches="tight", dpi=150)
        plt.show()
