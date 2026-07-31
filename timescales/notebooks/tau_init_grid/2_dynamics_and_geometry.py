# %% [markdown]
# # Timescale distributions and hidden-state geometry
#
# Five visualisations requested off the back of the reviews:
#
# 1. **Per-neuron time constants** τ, before vs after training.
# 2. **Network decay timescales** ($-1/\log|\lambda|$), before vs after training.
# 3. **Network oscillation timescales** ($2\pi/|\arg\lambda|$), before vs after training.
# 4. **Dimensionality of the neural code** — PCs needed to reach 90% cumulative variance,
#    for trained and untrained networks.
# 5. **3D hidden-state trajectories** in the *trained* network's PCA basis, for the network
#    before and after training, with explained variance reported.
#
# (4) and (5) also answer R2's request to "plot trajectories in hidden-unit space".
#
# Everything is restricted to **Θ=full** as prioritised. Items 1–3 come from logged
# artefacts; 4–5 require forward passes from the checkpoints, extracted by
# `extract_dynamics.py` on the machine holding the sweep.

# %%
import csv
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

SWEEP_DIR = os.environ.get(
    "SWEEP_DIR", os.path.join(os.getcwd(), "logs", "experiments", "tau_init_grid"))
FIGS_DIR = os.environ.get(
    "FIGS_DIR", os.path.join(os.getcwd(), "notebooks", "tau_init_grid", "figs"))
os.makedirs(FIGS_DIR, exist_ok=True)

C_POWERLAW, C_UNIFORM, C_NULL, C_MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
SCHEME_COLOR = {"powerlaw": C_POWERLAW, "uniform": C_UNIFORM}
SCHEME_LABEL = {"powerlaw": "power-law (β=1)", "uniform": "uniform (standard)"}
TASK_NAME = {"ff": "flip-flop", "sine": "sine-wave"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def log_hist(ax, values, color, ls="-", alpha=1.0, label=None, bins=50, lo=None, hi=None):
    """Fraction-per-bin histogram on a log x-axis, drawn as a line.

    Fraction rather than `density=True`: with log-spaced bins a linear density is
    visually misleading (wide high-value bins get squashed), and "what share of modes
    sit here" is the quantity we actually want to compare between schemes.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 2:
        return
    lo = lo if lo is not None else v.min()
    hi = hi if hi is not None else v.max()
    edges = np.logspace(np.log10(lo), np.log10(hi), bins)
    h, e = np.histogram(v, bins=edges)
    ax.plot(0.5 * (e[1:] + e[:-1]), h / v.size, ls, color=color, lw=1.8,
            alpha=alpha, label=label)


# %% [markdown]
# ## 1 — Per-neuron time constants, before vs after training
#
# Pooled over 20 seeds × 512 units, in **timesteps** (τ/dt). Dotted = at initialisation,
# solid = after training. Θ=full, so τ is itself a trained parameter here.

# %%
tau_path = os.path.join(SWEEP_DIR, "tau_pooled_g0.9.npz")
if not os.path.exists(tau_path):
    print(f"missing {tau_path} — skipping")
else:
    tau = np.load(tau_path)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, task in zip(axes, ("ff", "sine"), strict=False):
        for scheme in ("uniform", "powerlaw"):
            for tag, ls, alpha in (("init", ":", 0.55), ("final", "-", 1.0)):
                key = f"{task}_{scheme}_{tag}"
                if key in tau:
                    log_hist(ax, tau[key], SCHEME_COLOR[scheme], ls, alpha,
                             f"{SCHEME_LABEL[scheme].split(' (')[0]} — {tag}")
        ax.set_xscale("log")
        ax.set_xlabel(r"per-neuron $\tau$ (timesteps)")
        ax.set_title(TASK_NAME[task], fontsize=10)
    axes[0].set_ylabel("fraction of units / modes")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("1 — Per-neuron τ: init (dotted) vs after training (solid), Θ=full, g=0.9",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "dyn1_tau_per_neuron.png"))
    plt.show()

# %% [markdown]
# ## 2 & 3 — Network decay and oscillation timescales, before vs after training
#
# From the Jacobian eigenvalues $\lambda$ of $J = (I-A) + gAW$:
#
# * decay timescale $= -1/\log|\lambda|$ over contracting modes ($|\lambda| < 1$)
# * oscillation period $= 2\pi/|\arg\lambda|$, keeping only periods $\le 2\times$ the
#   trajectory length (2000 steps for flip-flop, 200 for sine-wave)
#
# **The oscillation cutoff needs care.** A mode sitting essentially on the real axis has
# $\arg\lambda \approx 0$ and reports a "period" of $10^8$ timesteps — not an oscillation in
# any useful sense. But the obvious fix, reusing the $|\mathrm{Im}\,\lambda| \ge 0.1$
# pinching band, is *wrong here*: the slowest task frequency (T=100) needs only
# $|\mathrm{Im}\,\lambda| = \sin(2\pi/100) = 0.063$, so that band silently discards the very
# modes we are looking for. Capping the **period** at twice the trajectory length instead
# means "at least half a cycle is observable", which is task-relevant and keeps every
# frequency the task actually demands.
#
# These are the *network-level* timescales — the quantity the paper argues is the right
# level of analysis — as opposed to the per-neuron τ above.
#
# For reference, the task timescales in this sweep are: flip-flop, per-bit mean pulse
# intervals ≈ **[200, 143, 100, 50, 20, 10] steps** (heterogeneous `p_pulse`); sine-wave,
# periods **[20, 50, 100] steps**.

# %%
spec_path = os.path.join(SWEEP_DIR, "spectral_timescales.npz")
if not os.path.exists(spec_path):
    print(f"missing {spec_path} — run extract_dynamics.py first; skipping")
else:
    spec = np.load(spec_path)

    def n_total(task, scheme, tag, g="0.9"):
        k = f"{task}_{scheme}_g{g}_{tag}_total"
        return int(spec[k][0]) if k in spec else 512 * 20

    for kind, nice, fname, nbins in (
            ("decay", "decay timescale", "dyn2_network_decay", 50),
            ("osc", "oscillation period", "dyn3_network_oscillation", 24)):
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
        for ax, task in zip(axes, ("ff", "sine"), strict=False):
            for scheme in ("uniform", "powerlaw"):
                for tag, ls, alpha in (("init", ":", 0.55), ("final", "-", 1.0)):
                    key = f"{task}_{scheme}_g0.9_{tag}_{kind}"
                    if key not in spec:
                        continue
                    v = spec[key]
                    frac = v.size / n_total(task, scheme, tag)
                    lbl = f"{SCHEME_LABEL[scheme].split(' (')[0]} — {tag} ({frac:.0%})"
                    if v.size < 2:
                        # Nothing to plot: record the absence in the legend instead.
                        ax.plot([], [], ls, color=SCHEME_COLOR[scheme], lw=1.8,
                                alpha=alpha, label=lbl)
                        continue
                    log_hist(ax, v, SCHEME_COLOR[scheme], ls, alpha, lbl, bins=nbins)
            ax.set_xscale("log")
            ax.set_xlabel(f"network {nice} (timesteps)")
            ax.set_title(TASK_NAME[task], fontsize=10)
        axes[0].set_ylabel("fraction of modes\n(within each condition)")
        axes[0].legend(fontsize=6.5, loc="best")
        n = "2" if kind == "decay" else "3"
        extra = ("  ·  % = share of all modes that qualify"
                 if kind == "osc" else "")
        fig.suptitle(f"{n} — Network {nice}: init (dotted) vs after training (solid), "
                     f"Θ=full, g=0.9{extra}", fontsize=10.5, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, f"{fname}.png"))
        plt.show()

    print("Oscillatory modes available (period ≤ 2× trajectory length), g=0.9:\n")
    print(f"{'task':6s} {'scheme':9s} | {'at init':>22s} {'after training':>22s}")
    for task in ("ff", "sine"):
        for scheme in ("uniform", "powerlaw"):
            cells = []
            for tag in ("init", "final"):
                v = spec.get(f"{task}_{scheme}_g0.9_{tag}_osc", np.array([]))
                tot = n_total(task, scheme, tag)
                cells.append(f"{v.size / 20:8.1f}/seed ({100 * v.size / tot:4.1f}%)")
            print(f"{task:6s} {scheme:9s} | {cells[0]:>22s} {cells[1]:>22s}")
    print("\nTask target periods — flip-flop: 10-200 steps (6 bits); sine-wave: 20, 50, 100.")

# %% [markdown]
# ## 4 — Dimensionality of the neural code
#
# Number of principal components needed to explain 90% of the variance of the hidden states,
# computed on a validation batch. Reported for the **untrained** and **trained** network of
# every seed, so the change induced by training is visible.

# %%
dims_path = os.path.join(SWEEP_DIR, "pca_dims.csv")
if not os.path.exists(dims_path):
    print(f"missing {dims_path} — run extract_dynamics.py first; skipping")
else:
    with open(dims_path) as f:
        dims = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    for ax, task in zip(axes, ("ff", "sine"), strict=False):
        xt, xl = [], []
        for i, scheme in enumerate(("uniform", "powerlaw")):
            for j, g in enumerate((0.3, 0.9)):
                sub = [r for r in dims if r["task"] == task and r["scheme"] == scheme
                       and abs(float(r["g"]) - g) < 1e-9]
                if not sub:
                    continue
                base = i * 2.2 + j * 1.0
                for k, (col, c) in enumerate((
                        ("dim90_untrained", C_NULL),
                        ("dim90_trained", SCHEME_COLOR[scheme]))):
                    v = np.array([float(r[col]) for r in sub])
                    x = base + k * 0.34
                    ax.scatter(np.random.default_rng(0).normal(x, 0.05, v.size), v,
                               s=10, color=c, alpha=0.45, edgecolors="none", zorder=3)
                    ax.hlines(np.median(v), x - 0.13, x + 0.13, color=c, lw=2.4, zorder=4)
                xt.append(base + 0.17)
                xl.append(f"{scheme[:4]}\ng={g}")
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=7.5)
        ax.set_title(TASK_NAME[task], fontsize=10)
    axes[0].set_ylabel("PCs to reach 90% variance")
    handles = [plt.Line2D([], [], color=C_NULL, lw=2.4),
               plt.Line2D([], [], color=C_MUTED, lw=2.4)]
    axes[0].legend(handles, ["untrained", "trained (scheme colour)"], fontsize=7.5,
                   loc="best")
    fig.suptitle("4 — Dimensionality of the hidden-state code (20 seeds, Θ=full)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "dyn4_pca_dimensionality.png"))
    plt.show()

    print("Median PCs for 90% variance, and participation ratio:\n")
    print(f"{'task':6s} {'scheme':9s} {'g':>4s} | {'dim90 untr':>10s} {'dim90 train':>11s}"
          f" | {'PR untr':>8s} {'PR train':>9s}")
    for task in ("ff", "sine"):
        for scheme in ("uniform", "powerlaw"):
            for g in (0.3, 0.9):
                sub = [r for r in dims if r["task"] == task and r["scheme"] == scheme
                       and abs(float(r["g"]) - g) < 1e-9]
                if not sub:
                    continue
                def med(col, rows=sub):
                    return np.median([float(r[col]) for r in rows])
                print(f"{task:6s} {scheme:9s} {g:>4} | {med('dim90_untrained'):10.0f} "
                      f"{med('dim90_trained'):11.0f} | {med('pr_untrained'):8.1f} "
                      f"{med('pr_trained'):9.1f}")

# %% [markdown]
# ## 5 — Hidden-state trajectories in the trained network's PCA basis
#
# The PCA basis is fit on the **trained** network's hidden states; the untrained network's
# states are then projected into that *same* basis, so the two panels are directly
# comparable. Explained variance of the first three PCs is given per panel.

# %%
traj_path = os.path.join(SWEEP_DIR, "pca_traj.npz")
if not os.path.exists(traj_path):
    print(f"missing {traj_path} — run extract_dynamics.py first; skipping")
else:
    tj = np.load(traj_path)
    for task in ("ff", "sine"):
        fig = plt.figure(figsize=(11.5, 7.4))
        panel = 1
        for scheme in ("uniform", "powerlaw"):
            for tag in ("untrained", "trained"):
                key = f"{task}_{scheme}_full_g0.9_seed0_{tag}"
                ax = fig.add_subplot(2, 2, panel, projection="3d")
                panel += 1
                if key not in tj:
                    ax.set_title("missing", fontsize=9)
                    continue
                P = tj[key]                        # [n_traj, T, 3]
                evr = tj.get(f"{task}_{scheme}_full_g0.9_seed0_evr_{tag}")
                col = SCHEME_COLOR[scheme] if tag == "trained" else C_NULL
                for t in range(P.shape[0]):
                    ax.plot(P[t, :, 0], P[t, :, 1], P[t, :, 2], color=col, lw=0.9,
                            alpha=0.8)
                    ax.scatter(*P[t, 0], color=col, s=14, depthshade=False)
                pct = f"{100 * evr[:3].sum():.1f}%" if evr is not None else "?"
                ax.set_title(f"{SCHEME_LABEL[scheme].split(' (')[0]} — {tag}\n"
                             f"PC1-3 = {pct} var", fontsize=9, pad=0)
                ax.set_xlabel("PC1", fontsize=7)
                ax.set_ylabel("PC2", fontsize=7)
                ax.set_zlabel("PC3", fontsize=7)
                ax.tick_params(labelsize=6)
        fig.suptitle(f"5 — {TASK_NAME[task]}: hidden-state trajectories in the trained "
                     f"network's PCA basis (g=0.9, seed 0)", fontsize=11, y=0.98)
        # tight_layout mishandles 3D axes (titles land on the panel above); space manually.
        fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03,
                            wspace=0.10, hspace=0.30)
        fig.savefig(os.path.join(FIGS_DIR, f"dyn5_trajectories_{task}.png"))
        plt.show()
