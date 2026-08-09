# %% [markdown]
# # Hidden-state trajectory geometry
#
# Where the task's temporal structure lives in state space, for **uniform** time-constant
# initialization, on both tasks and in both regimes (linear/rate and Tanh/voltage).
#
# The same trajectories are viewed in three coordinate systems:
#
# 1. **PCA** — the variance-ordered basis. What you see by default.
# 2. **Jacobian eigenmodes** — coefficients $c(t) = V^{-1}h(t)$. For a complex-conjugate
#    pair, $(\mathrm{Re}\,c_j, \mathrm{Im}\,c_j)$ rotates at that mode's own frequency.
# 3. **Output-coupled modes** — for each output channel, the single mode it reads from
#    most strongly, $\arg\max_j |W_{out}V|_{kj}$.
#
# The third is what makes "how are the different frequencies projected?" answerable: if
# each frequency has its own mode and each output reads one mode, the projections
# separate cleanly rather than mixing.
#
# Data comes from `extract_trajectories.py`, run where the checkpoints live.

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

for _cand in (os.getcwd(),
              os.path.abspath(os.path.join(os.getcwd(), "..")),
              os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))):
    if os.path.isdir(os.path.join(_cand, "timescales")) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

EXP = os.path.join(os.getcwd(), "logs", "experiments")
REGIMES = {"linear": os.path.join(EXP, "tau_init_grid", "trajectories_uniform.npz"),
           "tanh": os.path.join(EXP, "tau_init_grid_tanh_voltage",
                                "trajectories_uniform.npz")}
FIGS_DIR = os.environ.get(
    "FIGS_DIR", os.path.join(os.getcwd(), "notebooks", "tau_init_grid", "figs"))
os.makedirs(FIGS_DIR, exist_ok=True)

C_BLUE, C_ORANGE, C_AQUA, C_MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
TASK_NAME = {"ff": "flip-flop", "sine": "sine-wave"}
SINE_PERIODS = [20.0, 50.0, 100.0]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})

DATA = {}
for name, path in REGIMES.items():
    if os.path.exists(path):
        DATA[name] = np.load(path)
        print(f"{name:7s}: {len(DATA[name].files)} arrays")
    else:
        print(f"{name:7s}: MISSING {path}")


def get(regime, task, g, seed, field):
    d = DATA.get(regime)
    k = f"{task}_g{g:g}_seed{seed}_{field}"
    return d[k] if d is not None and k in d else None


def period_of(lam):
    """Oscillation period in timesteps for an eigenvalue; inf if not rotating."""
    th = abs(np.angle(lam))
    return 2 * np.pi / th if th > 1e-9 else np.inf


# %% [markdown]
# ## 1 — Trajectories in the leading principal components
#
# Colour runs from dark to light along the trajectory, so the direction of travel and
# whether the orbit closes are both visible.

# %%
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

G_SHOW = 0.9
# The sine-wave validation set uses a fixed initial state and no phase randomization,
# so every trajectory is identical; one is plotted. Flip-flop trajectories differ
# because the input pulse trains differ.
N_SHOW = {"ff": 3, "sine": 1}


def gradient_line(ax, P, cmap="viridis", lw=1.3):
    """Draw a 3D path coloured by time, which shows direction and closure."""
    pts = P[:, :3].reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=cmap, linewidths=lw)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection3d(lc)
    for dim, setlim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = P[:, dim].min(), P[:, dim].max()
        pad = 0.08 * max(hi - lo, 1e-9)
        setlim(lo - pad, hi + pad)


fig = plt.figure(figsize=(11.0, 8.0))
panel = 1
for task in ("ff", "sine"):
    for regime in ("linear", "tanh"):
        ax = fig.add_subplot(2, 2, panel, projection="3d")
        panel += 1
        proj = get(regime, task, G_SHOW, 0, "proj")
        evr = get(regime, task, G_SHOW, 0, "evr")
        if proj is None:
            ax.set_title(f"{TASK_NAME[task]} · {regime} — missing", fontsize=9)
            continue
        for t in range(min(N_SHOW[task], proj.shape[0])):
            P = proj[t]
            gradient_line(ax, P)
            ax.scatter(*P[0, :3], color="crimson", s=26, depthshade=False, zorder=5)
        ax.set_title(f"{TASK_NAME[task]} · {regime}\nPC1-3 = {100 * evr[:3].sum():.1f}% var",
                     fontsize=9.5, pad=0)
        for setter, lbl in ((ax.set_xlabel, "PC1"), (ax.set_ylabel, "PC2"),
                            (ax.set_zlabel, "PC3")):
            setter(lbl, fontsize=7)
        ax.tick_params(labelsize=6)
fig.suptitle("1 — Hidden-state trajectories in PC space (uniform init, g = 0.9, seed 0; "
             "red = start)", fontsize=11, y=0.97)
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03, wspace=0.10, hspace=0.22)
fig.savefig(os.path.join(FIGS_DIR, "traj1_pca_3d.png"))
plt.show()

# %% [markdown]
# ## 2 — Which principal axes carry which frequency?
#
# Rather than reading frequency off the PCs by Fourier transform — the sine-wave
# trajectory is only 100 steps, so the slowest component completes a single cycle and the
# transform has almost no resolution — we take it directly from the geometry. For each
# principal axis we compute its overlap $|\langle \mathrm{PC}_i, v_j\rangle|$ with the
# eigenmode each output channel reads from. The mode's period is exact, so this assigns a
# frequency to each axis without any spectral estimation.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
for ax, regime in zip(axes, ("linear", "tanh"), strict=False):
    overlap = get(regime, "sine", G_SHOW, 0, "overlap")
    top_idx = get(regime, "sine", G_SHOW, 0, "top_idx")
    top_lam = get(regime, "sine", G_SHOW, 0, "top_lam")
    if overlap is None:
        ax.set_title(f"{regime} — missing", fontsize=10)
        continue
    # one column per distinct output-coupled mode, labelled by its period
    uniq, first = np.unique(top_idx, return_index=True)
    order = np.argsort(first)
    cols = uniq[order]
    labels = [f"mode {m}\nT={period_of(top_lam[np.where(top_idx == m)[0][0]]):.0f}"
              for m in cols]
    M = overlap[:6][:, cols]
    im = ax.imshow(M, cmap="magma", aspect="auto", vmin=0, vmax=max(M.max(), 1e-9))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"PC{i + 1}" for i in range(6)], fontsize=8)
    # The two panels differ by more than an order of magnitude in scale, so state the
    # maximum rather than letting the shared colour vocabulary imply comparability.
    ax.set_title(f"sine-wave · {regime}   (max |overlap| = {M.max():.3f})", fontsize=9.5)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, label="|overlap|")
fig.suptitle("2 — Overlap between principal axes and the output-coupled eigenmodes",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "traj2_pc_mode_overlap.png"))
plt.show()

# %% [markdown]
# ## 3 — Projections onto the output-coupled eigenmodes
#
# For each output channel, the coefficient $c_{j^*}(t)$ of the mode that channel reads
# from most strongly, plotted in the complex plane. A mode on the unit circle traces a
# closed circle whose angular rate is its frequency; a decaying mode spirals inward.
#
# The sine-wave task has three cos/sine pairs, so six output channels. If the assignment
# is one-to-one, the six channels collapse onto three distinct modes.

# %%
for regime in ("linear", "tanh"):
    coef = get(regime, "sine", G_SHOW, 0, "coef_top")
    top_idx = get(regime, "sine", G_SHOW, 0, "top_idx")
    top_lam = get(regime, "sine", G_SHOW, 0, "top_lam")
    if coef is None:
        continue
    uniq, first = np.unique(top_idx, return_index=True)
    cols = uniq[np.argsort(first)]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.1 * len(cols), 3.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, cols, strict=False):
        chans = np.where(top_idx == m)[0]
        lam = top_lam[chans[0]]
        c = coef[0][:, chans[0]]
        ax.plot(c.real, c.imag, "-", color=C_BLUE, lw=1.4)
        ax.scatter(c.real[0], c.imag[0], color="crimson", s=28, zorder=5, label="start")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"mode {m}  |λ|={abs(lam):.4f}\nperiod {period_of(lam):.1f} steps\n"
                     f"read by outputs {chans.tolist()}", fontsize=8.5)
        ax.set_xlabel("Re c", fontsize=8)
        ax.set_ylabel("Im c", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"3 — Output-coupled mode coefficients, sine-wave · {regime} "
                 f"(uniform init, g = {G_SHOW}, seed 0)", fontsize=10.5, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"traj3_mode_coefficients_{regime}.png"))
    plt.show()

# %%
print("Output-coupled modes for the sine-wave task (uniform init, seed 0)")
print("Task target periods: 20, 50, 100 steps\n")
for regime in ("linear", "tanh"):
    for g in (0.3, 0.9):
        top_idx = get(regime, "sine", g, 0, "top_idx")
        top_lam = get(regime, "sine", g, 0, "top_lam")
        if top_idx is None:
            continue
        parts = [f"out{k}->m{top_idx[k]}(T={period_of(top_lam[k]):.1f},"
                 f"|λ|={abs(top_lam[k]):.3f})" for k in range(len(top_idx))]
        print(f"  {regime:7s} g={g:<5}")
        for p in parts:
            print(f"      {p}")
        print(f"      distinct modes used: {len(set(top_idx.tolist()))} of "
              f"{len(top_idx)} channels")

# %% [markdown]
# ## 4 — Does the flip-flop task organize the same way?
#
# The flip-flop task is decay-dominated rather than oscillatory, so its output-coupled
# modes should be slow and real rather than rotating. The same projection is applied, and
# the modes' decay timescales are compared against the per-bit task timescales.

# %%
print("Output-coupled modes for the flip-flop task (uniform init, seed 0)")
print("Per-bit task timescales: 200, 143, 100, 50, 20, 10 steps\n")
for regime in ("linear", "tanh"):
    for g in (0.3, 0.9):
        top_idx = get(regime, "ff", g, 0, "top_idx")
        top_lam = get(regime, "ff", g, 0, "top_lam")
        if top_idx is None:
            continue
        print(f"  {regime:7s} g={g}")
        for k, (m, lam) in enumerate(zip(top_idx, top_lam, strict=False)):
            mag = abs(lam)
            decay = -1.0 / np.log(mag) if 0 < mag < 1 else np.inf
            per = period_of(lam)
            kind = "real/decay" if per > 1e6 else f"osc T={per:.0f}"
            print(f"      bit {k}: mode {m:3d}  |λ|={mag:.4f}  "
                  f"decay={decay:8.1f} steps  {kind}")

# %% [markdown]
# ## 5 — Explained variance: how low-dimensional is the code?

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
for ax, task in zip(axes, ("ff", "sine"), strict=False):
    for regime, color in (("linear", C_BLUE), ("tanh", C_ORANGE)):
        for g, ls in ((0.3, ":"), (0.9, "-")):
            evrs = [get(regime, task, g, s, "evr") for s in (0, 1, 2)]
            evrs = [e for e in evrs if e is not None]
            if not evrs:
                continue
            cum = np.cumsum(np.mean(evrs, axis=0))
            ax.plot(range(1, len(cum) + 1), 100 * cum, ls, color=color, lw=1.8,
                    label=f"{regime}, g={g}")
    ax.axhline(90, color=C_MUTED, lw=1, ls="--")
    ax.set_xlabel("number of principal components")
    ax.set_title(TASK_NAME[task], fontsize=10)
    ax.set_xticks(range(1, 11))
axes[0].set_ylabel("cumulative variance (%)")
axes[0].legend(fontsize=7.5, loc="lower right")
fig.suptitle("5 — Cumulative explained variance (mean over 3 seeds; dashed line = 90%)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "traj5_explained_variance.png"))
plt.show()
