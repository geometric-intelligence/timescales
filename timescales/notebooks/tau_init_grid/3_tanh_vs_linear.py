# %% [markdown]
# # Tanh/voltage vs linear: power-law vs uniform initialisation
#
# The tanh sweep is complete (400 runs: 2 tasks × 2 inits × 5 gains × 20 seeds, **Θ=full**).
# It differs from the linear sweep in two coupled settings:
#
# | | linear sweep | tanh sweep |
# |---|---|---|
# | `activation` | Identity | Tanh |
# | `dynamics_type` | rate | voltage |
# | grid | + fixedA, reservoir | Θ=full only |
#
# Everything else matches (N=512, LR=1e-3, 1500 steps, same tasks and gains), so the two are
# directly comparable on the Θ=full slice.
#
# Same analysis as before, kept to the essentials: **matched per-gain** comparison (not
# best-vs-best), Mann–Whitney U + Cliff's δ over 20 seeds, on final loss and on steps to
# threshold — plus the per-frequency check that explained the linear sine-wave failure.

# %%
import csv
import glob
import json
import os
import statistics as st
import sys

import matplotlib.pyplot as plt
import numpy as np

for _cand in (os.getcwd(),
              os.path.abspath(os.path.join(os.getcwd(), "..")),
              os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))):
    if os.path.isdir(os.path.join(_cand, "timescales")) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break
from timescales.stats import compare_samples  # noqa: E402

EXP = os.path.join(os.getcwd(), "logs", "experiments")
REGIMES = {"linear": os.path.join(EXP, "tau_init_grid"),
           "tanh": os.path.join(EXP, "tau_init_grid_tanh_voltage")}
FIGS_DIR = os.environ.get(
    "FIGS_DIR", os.path.join(os.getcwd(), "notebooks", "tau_init_grid", "figs"))
os.makedirs(FIGS_DIR, exist_ok=True)

C_POWERLAW, C_UNIFORM, C_MUTED = "#2a78d6", "#eb6834", "#898781"
SCHEME_COLOR = {"powerlaw": C_POWERLAW, "uniform": C_UNIFORM}
SCHEME_LABEL = {"powerlaw": "power-law (β=1)", "uniform": "uniform (standard)"}
G_GRID = [0.3, 0.5, 0.7, 0.9, 0.95]
PREFIX = {"flip_flop": "ff", "sine_wave": "sine"}
THRESHOLD, MAX_STEP, CENSORED = 0.9, 1500, 1510

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def fnum(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


RUNS = {}
for name, d in REGIMES.items():
    p = os.path.join(d, "run_table.csv")
    if os.path.exists(p):
        with open(p) as f:
            RUNS[name] = list(csv.DictReader(f))
        print(f"{name:7s}: {len(RUNS[name])} runs")
    else:
        print(f"{name:7s}: MISSING {p}")


def final_loss(regime, task, scheme, g):
    """Final val loss per seed, Θ=full only (the slice both sweeps share)."""
    return [v for v in (fnum(r, "final_val_loss") for r in RUNS.get(regime, [])
                        if r["cfg.task"] == task
                        and r["cfg.trainable"] == "full"
                        and r["cfg.tau_init.scheme"] == scheme
                        and abs((fnum(r, "cfg.recurrent_gain") or -1) - g) < 1e-9)
            if v is not None]


def steps_to_threshold(regime, task, scheme, g):
    """Per-seed steps to reach the threshold; never-reached ranked worst. Returns (arr, n)."""
    pat = os.path.join(REGIMES[regime], f"{PREFIX[task]}_{scheme}_full_g{g:g}",
                       "seed_*", "training_losses.json")
    out, reached = [], 0
    for path in sorted(glob.glob(pat)):
        with open(path) as f:
            d = json.load(f)
        hit = next((s for s, v in zip(d["steps"], d["val_accuracies"], strict=False)
                    if v >= THRESHOLD), None)
        out.append(hit if hit is not None else CENSORED)
        reached += hit is not None
    return np.asarray(out, dtype=float), reached


def verdict(d, p):
    if d is None or p is None or p >= 0.05:
        return "n.s."
    return "power-law" if d < 0 else "uniform"


# %% [markdown]
# ## Figure A — Final validation loss vs gain, both regimes
#
# Matched comparison: at each gain the only thing differing between the two curves is the
# initialisation.

# %%
fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), sharex=True)
for i, task in enumerate(("flip_flop", "sine_wave")):
    for j, regime in enumerate(("linear", "tanh")):
        ax = axes[i][j]
        for scheme in ("uniform", "powerlaw"):
            ys = [st.median(final_loss(regime, task, scheme, g)) or np.nan
                  if final_loss(regime, task, scheme, g) else np.nan for g in G_GRID]
            ax.plot(G_GRID, ys, "-o", color=SCHEME_COLOR[scheme], lw=2, ms=5,
                    label=SCHEME_LABEL[scheme])
        # mark gains where power-law is significantly better
        for g in G_GRID:
            a, b = final_loss(regime, task, "powerlaw", g), final_loss(regime, task,
                                                                      "uniform", g)
            if not a or not b:
                continue
            r = compare_samples(a, b)
            if verdict(r["cliffs_delta"], r["p_value"]) == "power-law":
                ax.plot(g, min(st.median(a), st.median(b)), "v", color=C_POWERLAW, ms=6,
                        alpha=0.6)
        ax.set_title(f"{task} · {regime}", fontsize=10)
        ax.set_xticks(G_GRID)
        ax.set_xticklabels([str(g) for g in G_GRID], rotation=45, ha="right", fontsize=8)
        if i == 1:
            ax.set_xlabel("recurrent gain $g$")
        if j == 0:
            ax.set_ylabel("median final val loss")
axes[0][0].legend(fontsize=8, loc="best")
fig.suptitle("A — Final loss vs gain (20 seeds; ▾ = power-law significantly better)",
             fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "tanh_A_loss_vs_g.png"))
plt.show()

# %%
print("Matched per-gain comparison — FINAL VAL LOSS (δ<0 ⇒ power-law better)\n")
for regime in ("linear", "tanh"):
    for task in ("flip_flop", "sine_wave"):
        print(f"--- {regime} / {task} ---")
        for g in G_GRID:
            a, b = final_loss(regime, task, "powerlaw", g), final_loss(regime, task,
                                                                      "uniform", g)
            if not a or not b:
                continue
            r = compare_samples(a, b)
            print(f"  g={g:<5} pl={st.median(a):.4f} uni={st.median(b):.4f}  "
                  f"δ={r['cliffs_delta']:+.2f} p={r['p_value']:8.2g}  "
                  f"-> {verdict(r['cliffs_delta'], r['p_value'])}")

# %% [markdown]
# ## Figure B — Steps to threshold vs gain, both regimes
#
# 90% accuracy (flip-flop) or R² ≥ 0.9 (sine-wave). Runs that never cross are ranked worse
# than every run that did; the reached-count is printed below.

# %%
fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), sharex=True)
for i, task in enumerate(("flip_flop", "sine_wave")):
    for j, regime in enumerate(("linear", "tanh")):
        ax = axes[i][j]
        for scheme in ("uniform", "powerlaw"):
            meds, los, his = [], [], []
            for g in G_GRID:
                v, _ = steps_to_threshold(regime, task, scheme, g)
                if v.size == 0:
                    meds.append(np.nan)
                    los.append(0)
                    his.append(0)
                    continue
                meds.append(np.median(v))
                los.append(np.median(v) - np.percentile(v, 25))
                his.append(np.percentile(v, 75) - np.median(v))
            ax.errorbar(G_GRID, meds, yerr=[los, his], fmt="-o",
                        color=SCHEME_COLOR[scheme], lw=2, ms=5, capsize=3,
                        label=SCHEME_LABEL[scheme])
        ax.axhline(CENSORED, color=C_MUTED, lw=1, ls=":")
        ax.set_title(f"{task} · {regime}", fontsize=10)
        ax.set_xticks(G_GRID)
        ax.set_xticklabels([str(g) for g in G_GRID], rotation=45, ha="right", fontsize=8)
        if i == 1:
            ax.set_xlabel("recurrent gain $g$")
        if j == 0:
            ax.set_ylabel(f"steps to {THRESHOLD:.0%}\n(median ± IQR)")
axes[0][0].legend(fontsize=8, loc="best")
fig.suptitle("B — Steps to threshold vs gain (dotted line = never reached)",
             fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "tanh_B_steps_vs_g.png"))
plt.show()

# %%
print(f"Matched per-gain comparison — STEPS TO {THRESHOLD:.0%} "
      f"(δ<0 ⇒ power-law faster; (n) = seeds reaching it)\n")
for regime in ("linear", "tanh"):
    for task in ("flip_flop", "sine_wave"):
        print(f"--- {regime} / {task} ---")
        for g in G_GRID:
            a, ra = steps_to_threshold(regime, task, "powerlaw", g)
            b, rb = steps_to_threshold(regime, task, "uniform", g)
            if a.size == 0 or b.size == 0:
                continue
            r = compare_samples(a, b)
            fa = ">1500" if np.median(a) >= CENSORED else f"{np.median(a):.0f}"
            fb = ">1500" if np.median(b) >= CENSORED else f"{np.median(b):.0f}"
            print(f"  g={g:<5} pl={fa:>6s} ({ra:2d}/20)  uni={fb:>6s} ({rb:2d}/20)  "
                  f"δ={r['cliffs_delta']:+.2f} p={r['p_value']:8.2g}  "
                  f"-> {verdict(r['cliffs_delta'], r['p_value'])}")

# %% [markdown]
# ## Figure C — Per-frequency R² on sine-wave: the linear failure does not carry over
#
# In the linear regime power-law failed on the fastest component (T=20) while fitting T=50
# and T=100 perfectly, because all its eigenvalues start pinned near the real axis. The
# obvious question is whether that survives the nonlinearity.

# %%
PERIODS = [20.0, 50.0, 100.0]


def per_frequency_r2(regime, scheme, g=0.7):
    """Median final R^2 for each (cos,sin) pair; channels 2k, 2k+1 <-> PERIODS[k]."""
    pat = os.path.join(REGIMES[regime], f"sine_{scheme}_full_g{g:g}",
                       "seed_*", "training_losses.json")
    per = {k: [] for k in range(len(PERIODS))}
    for path in sorted(glob.glob(pat)):
        with open(path) as f:
            chans = json.load(f).get("val_accuracies_per_bit", {})
        for k in range(len(PERIODS)):
            v = [chans[f"channel_{c}"][-1] for c in (2 * k, 2 * k + 1)
                 if f"channel_{c}" in chans]
            if v:
                per[k].append(float(np.mean(v)))
    return [np.median(per[k]) if per[k] else np.nan for k in range(len(PERIODS))]


fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
for ax, regime in zip(axes, ("linear", "tanh"), strict=False):
    width = 0.36
    for si, scheme in enumerate(("uniform", "powerlaw")):
        vals = per_frequency_r2(regime, scheme)
        x = np.arange(len(PERIODS)) + (si - 0.5) * width
        ax.bar(x, vals, width=width, color=SCHEME_COLOR[scheme],
               label=SCHEME_LABEL[scheme])
        for xi, v in zip(x, vals, strict=False):
            if np.isfinite(v) and v < 0.9:
                ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5,
                        color=C_MUTED)
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels([f"T={int(p)}" for p in PERIODS], fontsize=9)
    ax.set_title(f"{regime} (g=0.7)", fontsize=10)
    ax.axhline(0, color="#c3c2b7", lw=1)
axes[0].set_ylabel("final $R^2$ per frequency\n(median, 20 seeds)")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("C — Sine-wave per-frequency fit: the T=20 failure is specific to the "
             "linear regime", fontsize=10.5, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "tanh_C_per_frequency.png"))
plt.show()

# %%
print("Per-frequency final R² on sine-wave, g=0.7 (median over 20 seeds):\n")
print(f"{'regime':8s} {'scheme':9s} | {'T=20':>7s} {'T=50':>7s} {'T=100':>7s}")
for regime in ("linear", "tanh"):
    for scheme in ("uniform", "powerlaw"):
        v = per_frequency_r2(regime, scheme)
        print(f"{regime:8s} {scheme:9s} | {v[0]:+7.3f} {v[1]:+7.3f} {v[2]:+7.3f}")

# %% [markdown]
# ## Summary
#
# **Flip-flop — the two regimes agree.** Power-law is equivalent at low gain and
# significantly better from g≈0.5–0.7 upward, on both final loss and speed, and it is far
# less sensitive to the gain. The linear-regime conclusion carries over.
#
# **Sine-wave — the regimes disagree, and the nonlinearity removes the failure.**
#
# | | linear | tanh/voltage |
# |---|---|---|
# | power-law fits T=20? | **no** (R² ≈ 0.21) | **yes** (R² ≈ 0.999) |
# | seeds reaching R² ≥ 0.9 | 0–3 / 20 | **20 / 20** |
# | final loss vs uniform | uniform better everywhere | **power-law better at every g** (δ = −1.00) |
# | steps to threshold | power-law never gets there | uniform faster at every g |
#
# In the linear model the oscillation frequencies available to the network are exactly the
# eigenvalue angles of the origin Jacobian, and power-law init pins those onto the real
# axis — the fast component is simply not reachable. With tanh/voltage dynamics the
# effective linearisation changes with state amplitude, so the network is no longer confined
# to its initial spectrum and recovers the fast frequency.
#
# In short: **the "power-law hurts oscillatory tasks" result is
# a property of the linear model, not of power-law initialisation in general.** In the
# nonlinear regime power-law reaches a *better* solution on the oscillatory task than
# uniform at every gain, though it takes more steps to cross the threshold — it converges
# more slowly but lands better.
