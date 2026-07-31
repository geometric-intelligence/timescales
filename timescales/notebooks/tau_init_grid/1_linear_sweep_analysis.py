# %% [markdown]
# # Linear sweep — power-law vs uniform timescale initialization
#
# Walks through the timescale-initialization analyses one at a time, regenerating each
# figure from the aggregated tables.
#
# **Data**: `logs/experiments/tau_init_grid` — 60 conditions × 20 seeds = 1200 runs.
# Axes: task {flip_flop, sine_wave} × init {uniform, powerlaw β=1} ×
# trainable {full, fixedA, reservoir} × g {0.3, 0.5, 0.7, 0.9, 0.95}. LR fixed 1e-3,
# activation = Identity (**linear regime only** — the tanh sweep is incomplete).
#
# Reads only the aggregated tables (never checkpoints):
#
# | file | contents |
# |---|---|
# | `run_table.csv` | one row/run: config, final loss, convergence, init spectrum stats |
# | `per_g.csv` | medians per (task, trainable, scheme, g) |
# | `best_vs_best.csv` | best-g-vs-best-g head-to-head with effect sizes |
# | `mode_table_final.csv` | per-run matching + coupling metrics (trained network) |
# | `<exp>/seed_<n>/training_losses.json` | full validation curves |
#
# Tests are Mann–Whitney U + Cliff's delta via `timescales.stats.compare_samples`.

# %%
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Repo root on the path so `timescales.stats` is importable. Works whether the notebook
# is run from timescales/timescales/ or from its own directory.
for _cand in (os.getcwd(),
              os.path.abspath(os.path.join(os.getcwd(), "..")),
              os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))):
    if os.path.isdir(os.path.join(_cand, "timescales")) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from timescales.stats import compare_samples  # noqa: E402

# Override either with an env var to point at a different sweep (e.g. the tanh grid).
SWEEP_DIR = os.environ.get(
    "SWEEP_DIR",
    os.path.join(os.getcwd(), "logs", "experiments", "tau_init_grid"),
)
FIGS_DIR = os.environ.get(
    "FIGS_DIR", os.path.join(os.getcwd(), "notebooks", "tau_init_grid", "figs")
)
os.makedirs(FIGS_DIR, exist_ok=True)
print("sweep dir:", SWEEP_DIR)
print("figs dir :", FIGS_DIR)

# %% [markdown]
# ## Setup — loaders, palette, plot style
#
# Colours are fixed per init scheme and used identically in every figure, so the
# reader never has to re-learn the mapping. Blue = power-law (the proposed method),
# orange = uniform (the baseline), aqua = a null/control series.

# %%
C_POWERLAW = "#2a78d6"   # blue
C_UNIFORM = "#eb6834"    # orange
C_NULL = "#1baf7a"       # aqua — reservoir / shuffled controls
C_MUTED = "#898781"
SCHEME_COLOR = {"powerlaw": C_POWERLAW, "uniform": C_UNIFORM}
SCHEME_LABEL = {"powerlaw": "power-law (β=1)", "uniform": "uniform (standard)"}
G_GRID = [0.3, 0.5, 0.7, 0.9, 0.95]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def load_csv(name):
    with open(os.path.join(SWEEP_DIR, name)) as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    """Parse a cell as float; empty/non-numeric -> None (e.g. never-converged runs)."""
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def sel(rows, **kw):
    """Filter rows by exact column matches; float-valued keys compared numerically."""
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            cell = r.get(k)
            if isinstance(v, float):
                c = fnum(r, k)
                if c is None or abs(c - v) > 1e-9:
                    ok = False
                    break
            elif cell != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def vals(rows, key):
    """Numeric column values, dropping missing entries."""
    return [v for v in (fnum(r, key) for r in rows) if v is not None]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


run_table = load_csv("run_table.csv")
per_g = load_csv("per_g.csv")
best_vs_best = load_csv("best_vs_best.csv")
mode_final = load_csv("mode_table_final.csv")
print(f"run_table {len(run_table)} rows | per_g {len(per_g)} | "
      f"best_vs_best {len(best_vs_best)} | mode_table_final {len(mode_final)}")

# %% [markdown]
# ## Sanity check — the grid is complete and balanced
#
# Every one of the 60 conditions should carry exactly 20 seeds.

# %%
counts = defaultdict(int)
for r in run_table:
    counts[(r["cfg.task"], r["cfg.trainable"], r["cfg.tau_init.scheme"],
            round(fnum(r, "cfg.recurrent_gain"), 2))] += 1
bad = {k: v for k, v in counts.items() if v != 20}
print(f"conditions: {len(counts)}   all with 20 seeds: {not bad}")
if bad:
    print("UNEXPECTED counts:", bad)

# %% [markdown]
# ---
# # R3 — "compared only against the weakest baseline"
#
# R3's objection: power-law was only ever compared to a uniform baseline at g=0.9, and a
# well-chosen (low-g) uniform init might match it. This is the axis the whole sweep was
# built to settle.
#
# ## Figure 1 — Final loss vs g
#
# This is the single most informative panel in the sweep. Read it two ways: **where each
# curve bottoms out** (the best-vs-best question R3 asked) and **how much each curve
# moves across g** (robustness).

# %%
fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True)
for i, task in enumerate(["flip_flop", "sine_wave"]):
    for j, tr in enumerate(["full", "fixedA"]):
        ax = axes[i][j]
        for scheme in ("uniform", "powerlaw"):
            xs, ys = [], []
            for g in G_GRID:
                rows = sel(per_g, **{"cfg.task": task, "cfg.trainable": tr,
                                     "scheme": scheme, "g": float(g)})
                if rows:
                    xs.append(g)
                    ys.append(fnum(rows[0], "final_val_loss_median"))
            ax.plot(xs, ys, "-o", color=SCHEME_COLOR[scheme], lw=2, ms=5,
                    label=SCHEME_LABEL[scheme])
            # mark each scheme's own optimum — the best-vs-best comparison point
            if ys:
                k = int(np.argmin(ys))
                ax.plot(xs[k], ys[k], "o", ms=11, mfc="none",
                        mec=SCHEME_COLOR[scheme], mew=2)
        ax.set_title(f"{task} · Θ={tr}", fontsize=10)
        ax.set_xticks(G_GRID)
        # 0.90 and 0.95 sit close on a linear axis; rotate so labels never collide.
        ax.set_xticklabels([str(g) for g in G_GRID], rotation=45, ha="right", fontsize=8)
        if i == 1:
            ax.set_xlabel("recurrent gain $g$")
        if j == 0:
            ax.set_ylabel("median final val loss")
axes[0][0].legend(loc="upper left", fontsize=8)
fig.suptitle("Final validation loss vs recurrent gain (20 seeds; rings = each scheme's own best g)",
             fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig1_loss_vs_g.png"))
plt.show()

# %% [markdown]
# **What this shows.**
#
# * **R3 is right about the optimum.** Where the rings sit, uniform is at or below power-law
#   on flip-flop — the low-g uniform baseline they predicted does match/beat power-law.
# * **But power-law is nearly flat in g while uniform is not.** That gap is the one genuinely
#   favourable result in the sweep, and it is a *robustness* claim, not a "wins at the optimum"
#   claim.
# * Note uniform is still descending at g=0.3, the low edge of the grid — its true optimum may
#   lie below the range we swept, which would only strengthen R3's point.

# %%
print("Degradation across the g grid (median final loss, worst g / best g):\n")
for task in ("flip_flop", "sine_wave"):
    for tr in ("full", "fixedA"):
        for scheme in ("uniform", "powerlaw"):
            ys = []
            for g in G_GRID:
                rows = sel(per_g, **{"cfg.task": task, "cfg.trainable": tr,
                                     "scheme": scheme, "g": float(g)})
                if rows:
                    ys.append(fnum(rows[0], "final_val_loss_median"))
            if ys:
                print(f"  {task:10s} {tr:7s} {scheme:9s} "
                      f"best={min(ys):.4f} worst={max(ys):.4f} "
                      f"-> +{100*(max(ys)/min(ys)-1):5.1f}%")

# %% [markdown]
# ## Figure 2 — Matched per-g comparison (the primary contrast)
#
# At each g, power-law vs uniform with **everything else held fixed**, so only the init
# scheme varies. This is the controlled version of the question, and it is the analysis to
# lead with. Best-vs-best (Figure 2b) compares two *different* g values, which confounds
# init with gain and selects each arm's optimum on the same data used for the test.
#
# Reading the grid: blue = power-law significantly better, orange = uniform significantly
# better, grey = not significant (p ≥ 0.05). A row that changes colour is a **crossing** —
# exactly the structure a single best-vs-best number destroys.

# %%
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

h2h = load_csv("per_g_head_to_head.csv")
# Diverging map keeps the established semantics: blue pole = power-law, orange = uniform.
cmap = LinearSegmentedColormap.from_list(
    "pl_uni", [C_POWERLAW, "#f0efec", C_UNIFORM])

for metric, nice in (("final_val_loss", "final validation loss"),
                     ("conv_mse_threshold_steps", "steps to normalised MSE ≤ 0.1")):
    cells = [r for r in h2h if r["metric"] == metric]
    groups = sorted({(r["cfg.task"], r["cfg.trainable"]) for r in cells})
    fig, ax = plt.subplots(figsize=(7.4, 0.52 * len(groups) + 1.8))
    grid = np.full((len(groups), len(G_GRID)), np.nan)
    for gi, grp in enumerate(groups):
        for gj, g in enumerate(G_GRID):
            m = [r for r in cells if (r["cfg.task"], r["cfg.trainable"]) == grp
                 and abs((fnum(r, "g") or -99) - g) < 1e-9]
            if not m:
                continue
            r = m[0]
            d, p = fnum(r, "cliffs_delta"), fnum(r, "p_value")
            if d is None:
                continue
            grid[gi, gj] = d
            ax.text(gj, gi, f"{d:+.2f}\n{stars(p) if p is not None else ''}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(d) > 0.55 else "#0b0b0b")
    ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(G_GRID)))
    ax.set_xticklabels([str(g) for g in G_GRID])
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{t}\nΘ={tr}" for t, tr in groups], fontsize=8)
    ax.set_xlabel("recurrent gain $g$")
    ax.set_title(f"Matched per-g contrast — {nice}\n"
                 "Cliff's δ (blue = power-law better · orange = uniform better)",
                 fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"fig2_matched_per_g_{metric}.png"))
    plt.show()

for metric in ("final_val_loss", "conv_frac_of_final_steps", "conv_mse_threshold_steps"):
    tally = defaultdict(int)
    for r in h2h:
        if r["metric"] == metric:
            tally[r["winner"]] += 1
    print(f"{metric:26s} " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))

# %% [markdown]
# **This is the result that matters, and it reverses the flip-flop conclusion.**
#
# * On **flip-flop/Θ=full, power-law wins 4 of the 5 gains** (δ = −0.97 … −1.00 for
#   g ≥ 0.5), losing only at g=0.3. Best-vs-best reported "uniform wins" purely because
#   uniform's optimum sits at that one low-gain point.
# * On **absolute convergence speed** (steps to MSE ≤ 0.1) the matched tally is
#   **power-law 8, uniform 4** — the opposite of the best-vs-best verdict.
# * **Sine-wave stays negative for power-law** at every matched g on final loss, so that
#   conclusion is unchanged and is not an artefact of the comparison design.
#
# **Survivorship caveat on the speed panel.** `conv_mse_threshold_steps` only exists for
# runs that actually reached the threshold. On sine-wave only ~4% of power-law runs do
# (fig 3), so those sine-wave cells compare *the few power-law runs that converged*
# against most of the uniform ones — they are survivorship-biased and must not be read as
# a like-for-like speed claim. Blank cells are conditions where too few runs converged to
# compare at all. The flip-flop cells are unaffected: every flip-flop run converged.
#
# ## Figure 2b — Best-vs-best (secondary, kept for R3's specific challenge)
#
# Each scheme at *its own* best g. This is retained because R3 asked specifically whether
# the baseline was weak, and it is the least favourable framing for us — reporting it is
# the honest thing to do — but it should not be the headline for the reasons above.

# %%
rows_bvb = [r for r in best_vs_best if r.get("metric") == "final_val_loss"]
if not rows_bvb:  # column naming fallback
    rows_bvb = best_vs_best
labels, d_vals, p_vals, winners = [], [], [], []
for r in rows_bvb:
    task = r.get("task") or r.get("cfg.task", "")
    tr = r.get("trainable") or r.get("cfg.trainable", "")
    metric = r.get("metric", "")
    if metric and metric != "final_val_loss":
        continue
    d = fnum(r, "cliffs_delta")
    p = fnum(r, "p_value")
    if d is None:
        continue
    labels.append(f"{task}\nΘ={tr}")
    d_vals.append(d)
    p_vals.append(p if p is not None else float("nan"))
    winners.append(r.get("winner", ""))

fig, ax = plt.subplots(figsize=(7.2, 3.4))
colors = [C_UNIFORM if w == "uniform" else C_POWERLAW for w in winners]
bars = ax.bar(range(len(d_vals)), d_vals, color=colors, width=0.62)
ax.axhline(0, color="#c3c2b7", lw=1)
for b, d, p in zip(bars, d_vals, p_vals, strict=False):
    off = 0.05 if d >= 0 else -0.05
    ax.text(b.get_x() + b.get_width() / 2, d + off, stars(p),
            ha="center", va="bottom" if d >= 0 else "top", fontsize=8, color=C_MUTED)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Cliff's δ\n(+ = uniform better)")
ax.set_ylim(-0.35, 1.55)
ax.set_title("Best-vs-best on final loss: each scheme at its own optimal g", fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=C_UNIFORM),
           plt.Rectangle((0, 0), 1, 1, color=C_POWERLAW)]
ax.legend(handles, ["uniform wins", "power-law wins (none on this metric)"],
          fontsize=8, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig2b_best_vs_best.png"))
plt.show()

print("\nAll best-vs-best comparisons in the report "
      "(x = power-law, y = uniform; winner is the better of the two):\n")
for r in best_vs_best:
    print(f"  {r['cfg.task']:10s} {r['cfg.trainable']:9s} {r['metric']:24s} "
          f"pl(g={r['best_g_x']})={fnum(r,'median_x'):.4f}  "
          f"uni(g={r['best_g_y']})={fnum(r,'median_y'):.4f}  "
          f"δ={fnum(r,'cliffs_delta'):+.2f} p={fnum(r,'p_value'):.2g}  -> {r['winner']}")

# %% [markdown]
# ## Figure 2c — The curves behind the matched contrast
#
# The δ values above compressed to one number per cell. This is what they look like as full
# training curves: **Θ=full only**, one panel per g, power-law vs uniform, all 20 seeds
# aggregated.
#
# Line = median across seeds; shaded band = interquartile range; error bars are drawn at
# subsampled steps (the band and the bars carry the same IQR — the bars just make the
# spread legible at a glance). Median/IQR is used rather than mean/SD for consistency with
# the rank-based tests used everywhere else in this notebook.

# %%
def load_curves(task_prefix, scheme, g, trainable="full"):
    """Stack all seed curves for one condition -> (steps, {metric: array[n_seeds, n_pts]})."""
    pattern = os.path.join(SWEEP_DIR, f"{task_prefix}_{scheme}_{trainable}_g{g:g}",
                           "seed_*", "training_losses.json")
    steps, stacks = None, defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        if steps is None:
            steps = d["steps"]
        for m in ("val_losses", "val_accuracies"):
            stacks[m].append(d[m])
    return steps, {m: np.asarray(v, dtype=float) for m, v in stacks.items()}


CURVE_SPECS = (
    ("val_losses", "validation loss", None),
    ("val_accuracies", "validation accuracy / $R^2$", None),
)

for metric, nice, ylim in CURVE_SPECS:
    fig, axes = plt.subplots(2, len(G_GRID), figsize=(15.5, 5.8), sharex=True, sharey="row")
    for i, (prefix, task_name) in enumerate((("ff", "flip-flop"), ("sine", "sine-wave"))):
        for j, g in enumerate(G_GRID):
            ax = axes[i][j]
            for si, scheme in enumerate(("uniform", "powerlaw")):
                steps, stacks = load_curves(prefix, scheme, g)
                if metric not in stacks or steps is None:
                    continue
                arr = stacks[metric]
                med = np.median(arr, axis=0)
                q25, q75 = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)
                c = SCHEME_COLOR[scheme]
                ax.plot(steps, med, color=c, lw=1.8, zorder=3,
                        label=f"{SCHEME_LABEL[scheme]}  (n={arr.shape[0]})")
                ax.fill_between(steps, q25, q75, color=c, alpha=0.15, lw=0, zorder=2)
                # Explicit error bars at subsampled steps, offset so the two schemes
                # never overlap. Same IQR as the band, asymmetric about the median.
                # The small dot is only the bar's anchor, not a data point — where the
                # IQR is tiny (power-law) the bar collapses and just the anchor shows.
                idx = np.arange(7 + si * 6, len(steps), 18)
                ax.errorbar(np.asarray(steps)[idx], med[idx],
                            yerr=[med[idx] - q25[idx], q75[idx] - med[idx]],
                            fmt="o", ms=2, color=c, elinewidth=1, capsize=2,
                            zorder=4, linestyle="none")
            if i == 0:
                ax.set_title(f"g = {g}", fontsize=10)
            if i == 1:
                ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel(f"{task_name}\n{nice}", fontsize=9)
            if ylim:
                ax.set_ylim(*ylim)
    axes[0][0].legend(fontsize=7.5, loc="best")
    fig.suptitle(f"Θ=full · {nice} · median ± IQR over 20 seeds", fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"fig2c_curves_per_g_{metric}.png"))
    plt.show()

# %% [markdown]
# ## Figure 2d — Quantifying the seed-to-seed variability
#
# The bands in fig 2c are visibly wide for uniform and almost invisible for power-law. That
# is real, not a plotting artefact: it is the spread of *final validation loss* across the
# 20 seeds. Quantified two ways — the IQR itself, and a **Brown–Forsythe test**
# (Levene centred on the median), the standard test for a difference in spread.

# %%
from scipy import stats as sp_stats  # noqa: E402


def final_losses(task_prefix, scheme, g, trainable="full"):
    """Final validation loss for every seed of one condition."""
    pattern = os.path.join(SWEEP_DIR, f"{task_prefix}_{scheme}_{trainable}_g{g:g}",
                           "seed_*", "training_losses.json")
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            out.append(json.load(f)["val_losses"][-1])
    return np.asarray(out, dtype=float)


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
print("Spread of final val loss across 20 seeds (Θ=full).")
print("BF = Brown-Forsythe test for unequal spread.\n")
print(f"{'task':10s} {'g':>5s} | {'IQR uni':>9s} {'IQR pl':>9s} {'ratio':>7s} | "
      f"{'range uni':>10s} {'range pl':>10s} | {'BF p':>9s}")
for ax, (prefix, task_name) in zip(axes, (("ff", "flip-flop"), ("sine", "sine-wave")), strict=False):
    for scheme in ("uniform", "powerlaw"):
        iqrs = []
        for g in G_GRID:
            v = final_losses(prefix, scheme, g)
            iqrs.append(np.percentile(v, 75) - np.percentile(v, 25) if v.size else np.nan)
        ax.plot(G_GRID, iqrs, "-o", color=SCHEME_COLOR[scheme], lw=2, ms=5,
                label=SCHEME_LABEL[scheme])
    for g in G_GRID:
        u, p_ = final_losses(prefix, "uniform", g), final_losses(prefix, "powerlaw", g)
        iqr_u = np.percentile(u, 75) - np.percentile(u, 25)
        iqr_p = np.percentile(p_, 75) - np.percentile(p_, 25)
        bf = sp_stats.levene(u, p_, center="median")
        print(f"{task_name:10s} {g:5} | {iqr_u:9.4f} {iqr_p:9.4f} "
              f"{iqr_u / iqr_p:6.1f}x | {u.max() - u.min():10.4f} "
              f"{p_.max() - p_.min():10.4f} | {bf.pvalue:9.2g}")
    ax.set_yscale("log")
    ax.set_xticks(G_GRID)
    ax.set_xticklabels([str(g) for g in G_GRID], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("recurrent gain $g$")
    ax.set_title(task_name, fontsize=10)
axes[0].set_ylabel("IQR of final val loss\nacross 20 seeds (log)")
axes[0].legend(fontsize=8, loc="best")
fig.suptitle("Seed-to-seed variability vs gain (lower = more reproducible)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig2d_seed_variability.png"))
plt.show()

# %% [markdown]
# **Answer to "is the narrow band a mistake?" — no, it is real, but read it per task.**
#
# * **Flip-flop**: power-law is dramatically more reproducible, and the gap widens with $g$.
#   At g=0.95 uniform's final loss ranges 0.254–0.735 across seeds (some seeds barely learn
#   at all — 0.73 is near chance) while power-law spans only 0.210–0.221. That is a ~100×
#   difference in IQR. This is a *second*, independent robustness claim: not just a better
#   median, but far more reliable across seeds.
# * **Sine-wave**: the direction **reverses at high $g$**. There power-law has the *larger*
#   spread, because it is bimodal — a minority of seeds escape the R²≈0.7 plateau
#   (min final loss 0.031 at g=0.9) while most do not. Uniform is tightly clustered on the
#   good solution. So "power-law is more robust" is a flip-flop statement, not a universal one.
#
# ## Figure 2e — Steps to threshold, with significance
#
# The most standard summary for 20 seeds: **steps to reach 90% accuracy** (flip-flop) or
# **R² ≥ 0.9** (sine-wave) — the paper's own convergence metric — compared per $g$ with
# Mann–Whitney U and Cliff's δ.
#
# **Censoring is handled explicitly.** Runs that never cross the threshold within 1500 steps
# have no defined value; discarding them would bias the comparison toward whichever scheme
# fails more often. They are instead ranked as worse than every run that did converge
# (a finite sentinel just past the horizon). Since Mann–Whitney uses only ranks this is
# exact, and the reached-fraction is reported alongside so the censoring is never hidden.

# %%
THRESHOLD = 0.9
MAX_STEP = 1500
CENSORED = MAX_STEP + 10  # ranks worse than any completed run


def steps_to_threshold(task_prefix, scheme, g, thr=THRESHOLD, trainable="full"):
    """Per-seed steps to first reach `thr`; censored runs -> CENSORED sentinel."""
    pattern = os.path.join(SWEEP_DIR, f"{task_prefix}_{scheme}_{trainable}_g{g:g}",
                           "seed_*", "training_losses.json")
    out, reached = [], 0
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        hit = next((s for s, v in zip(d["steps"], d["val_accuracies"], strict=False)
                    if v >= thr), None)
        out.append(hit if hit is not None else CENSORED)
        reached += hit is not None
    return np.asarray(out, dtype=float), reached


def _fmt_steps(median_steps, n_reached):
    """Median steps, showing '>1500' when the median run never crossed the threshold."""
    shown = f">{MAX_STEP}" if median_steps >= CENSORED else f"{median_steps:.0f}"
    return f"{shown} ({n_reached}/20)"


fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
print(f"\nSteps to reach {THRESHOLD:.0%} accuracy / R² (Θ=full), 20 seeds per cell.")
print("'reached' = seeds crossing the threshold within 1500 steps.\n")
print(f"{'task':10s} {'g':>5s} | {'uniform':>18s} | {'power-law':>18s} | "
      f"{'δ':>6s} {'p':>9s}  winner")
for ax, (prefix, task_name) in zip(axes, (("ff", "flip-flop"), ("sine", "sine-wave")), strict=False):
    for scheme in ("uniform", "powerlaw"):
        meds, los, his = [], [], []
        for g in G_GRID:
            v, _ = steps_to_threshold(prefix, scheme, g)
            meds.append(np.median(v))
            los.append(np.median(v) - np.percentile(v, 25))
            his.append(np.percentile(v, 75) - np.median(v))
        ax.errorbar(G_GRID, meds, yerr=[los, his], fmt="-o", color=SCHEME_COLOR[scheme],
                    lw=2, ms=5, capsize=3, label=SCHEME_LABEL[scheme])
    for g in G_GRID:
        u, ru = steps_to_threshold(prefix, "uniform", g)
        p_, rp = steps_to_threshold(prefix, "powerlaw", g)
        cmp = compare_samples(p_, u)  # x = power-law, y = uniform
        d, pv = cmp["cliffs_delta"], cmp["p_value"]
        win = "power-law" if (d < 0 and pv < 0.05) else ("uniform" if (d > 0 and pv < 0.05)
                                                         else "n.s.")
        print(f"{task_name:10s} {g:5} | {_fmt_steps(np.median(u), ru):>18s} | "
              f"{_fmt_steps(np.median(p_), rp):>18s} | {d:+6.2f} {pv:9.2g}  {win}")
    ax.axhline(CENSORED, color=C_MUTED, lw=1, ls=":")
    ax.text(G_GRID[0], CENSORED, " never reached", fontsize=7, color=C_MUTED,
            va="bottom", ha="left")
    ax.set_xticks(G_GRID)
    ax.set_xticklabels([str(g) for g in G_GRID], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("recurrent gain $g$")
    ax.set_title(task_name, fontsize=10)
axes[0].set_ylabel(f"steps to reach {THRESHOLD:.0%}\n(median ± IQR, 20 seeds)")
axes[0].legend(fontsize=8, loc="best")
fig.suptitle(f"Steps to {THRESHOLD:.0%} threshold vs gain (lower = faster)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig2e_steps_to_threshold.png"))
plt.show()

# %% [markdown]
# ## Figure 3 — Sine-wave is a plateau, not slow convergence
#
# The sine-wave gap is large enough that it is worth checking it is not an artefact of the
# *relative* convergence metric (`frac_of_final` measures progress toward a run's own final
# level, so a badly-converged run can look "fast"). The absolute threshold settles it.

# %%
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
ax = axes[0]
bar_x, bar_h, bar_c, bar_l = [], [], [], []
for i, task in enumerate(("flip_flop", "sine_wave")):
    for j, scheme in enumerate(("uniform", "powerlaw")):
        rows = [r for r in run_table
                if r["cfg.task"] == task and r["cfg.tau_init.scheme"] == scheme
                and r["cfg.trainable"] in ("full", "fixedA")]
        never = sum(1 for r in rows if fnum(r, "conv_mse_threshold_steps") is None)
        bar_x.append(i * 2.4 + j * 0.9)
        bar_h.append(100 * never / len(rows))
        bar_c.append(SCHEME_COLOR[scheme])
        bar_l.append(f"{task}\n{scheme}")
ax.bar(bar_x, bar_h, color=bar_c, width=0.8)
for x, h in zip(bar_x, bar_h, strict=False):
    ax.text(x, h + 2, f"{h:.0f}%", ha="center", fontsize=8, color=C_MUTED)
ax.set_xticks(bar_x)
ax.set_xticklabels(bar_l, fontsize=7.5)
ax.set_ylabel("% runs never reaching\nnormalised MSE ≤ 0.1")
ax.set_ylim(0, 112)
ax.set_title("Absolute convergence failure rate", fontsize=10)

# validation curves for the two sine conditions at their best g
ax = axes[1]
for exp, scheme in (("sine_uniform_full_g0.3", "uniform"),
                    ("sine_powerlaw_full_g0.7", "powerlaw")):
    curves = []
    for seed in range(3):
        p = os.path.join(SWEEP_DIR, exp, f"seed_{seed}", "training_losses.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            curves.append((d["steps"], d["val_accuracies"]))
    for st, va in curves:
        ax.plot(st, va, color=SCHEME_COLOR[scheme], lw=0.9, alpha=0.3)
    if curves:
        n = min(len(c[1]) for c in curves)
        mean = np.mean([c[1][:n] for c in curves], axis=0)
        ax.plot(curves[0][0][:n], mean, color=SCHEME_COLOR[scheme], lw=2,
                label=SCHEME_LABEL[scheme])
ax.set_xlabel("training step")
ax.set_ylabel("validation $R^2$")
ax.set_ylim(0, 1.05)
ax.set_title("Sine-wave curves at each scheme's best g", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig3_sine_failure.png"))
plt.show()

# %% [markdown]
# **What this shows.** Power-law on sine-wave does not converge slowly — it *plateaus*
# at $R^2 \approx 0.7$ by roughly step 400 and stops improving, while uniform eventually
# reaches $R^2 \approx 1.0$ through a sharp, late transition (clearest in fig 2c).
#
# ## Figure 3b — *Which* frequency fails
#
# The plateau sits suspiciously close to 2/3. The task has three (cos, sin) pairs at
# T = 20, 50, 100 steps, and the datamodule emits them in order, so channels (0,1) are
# T=20, (2,3) are T=50, (4,5) are T=100. Resolving $R^2$ per channel says exactly which
# component is missing.

# %%
PERIODS = [20.0, 50.0, 100.0]

fig, axes = plt.subplots(1, len(G_GRID), figsize=(14.5, 3.0), sharey=True)
for j, g in enumerate(G_GRID):
    ax = axes[j]
    width = 0.36
    for si, scheme in enumerate(("uniform", "powerlaw")):
        med, lo, hi = [], [], []
        pattern = os.path.join(SWEEP_DIR, f"sine_{scheme}_full_g{g:g}",
                               "seed_*", "training_losses.json")
        per_pair = {k: [] for k in range(len(PERIODS))}
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                d = json.load(f)
            chans = d.get("val_accuracies_per_bit", {})
            for k in range(len(PERIODS)):
                vals_k = [chans[f"channel_{c}"][-1]
                          for c in (2 * k, 2 * k + 1) if f"channel_{c}" in chans]
                if vals_k:
                    per_pair[k].append(float(np.mean(vals_k)))
        for k in range(len(PERIODS)):
            v = per_pair[k] or [np.nan]
            med.append(np.median(v))
            lo.append(np.median(v) - np.percentile(v, 25))
            hi.append(np.percentile(v, 75) - np.median(v))
        x = np.arange(len(PERIODS)) + (si - 0.5) * width
        ax.bar(x, med, width=width, color=SCHEME_COLOR[scheme],
               label=SCHEME_LABEL[scheme] if j == 0 else None)
        ax.errorbar(x, med, yerr=[lo, hi], fmt="none", ecolor="#52514e",
                    elinewidth=1, capsize=2.5, zorder=4)
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels([f"T={int(p)}" for p in PERIODS], fontsize=8)
    ax.set_title(f"g = {g}", fontsize=10)
    ax.axhline(0, color="#c3c2b7", lw=1)
    if j == 0:
        ax.set_ylabel("final $R^2$ per frequency\n(median ± IQR, 20 seeds)", fontsize=8.5)
        ax.legend(fontsize=7.5, loc="lower right")
fig.suptitle("Sine-wave, Θ=full — power-law fails on the fastest component only",
             fontsize=11, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig3b_per_frequency_R2.png"))
plt.show()

# %% [markdown]
# **This converts the sine-wave negative into a precise, predictive statement.**
# Power-law fits T=50 and T=100 essentially perfectly ($R^2 \approx 1.00$) and fails
# almost completely on T=20 ($R^2 \approx 0.2$). Four of six channels perfect plus two
# near 0.2 gives an aggregate $R^2 \approx 0.74$ — the plateau seen in fig 2c and 3.
#
# It also matches the spectral prediction quantitatively. All power-law eigenvalues start
# within $|\mathrm{Im}\,\lambda| < 0.1$ (fig 6), and the rotation each target needs is
# $|\mathrm{Im}\,\lambda| \approx \sin(2\pi/T)$: **0.309 for T=20, 0.125 for T=50, 0.063
# for T=100**. The only component that fails is the one furthest outside the pinched band.
#
# So the honest claim is not "power-law is worse at oscillatory tasks" but the sharper and
# more useful: *power-law initialisation cannot reach oscillation frequencies faster than
# its spectral pinch allows, and which frequency fails is predictable from the init
# spectrum alone.*
#
# ---
# # R2 — "is timescale matching actually necessary?"
#
# ## Figure 4 — Matching: trained vs frozen (reservoir) network
#
# The reservoir preset freezes W, W_in and A at init and trains only the readout. If a
# frozen network matches the task timescales as well as a trained one, then the matching
# was **inherited from the initialisation, not learned**.

# %%
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
g_lo, g_hi = float("inf"), 0.0  # global data range (axes are shared)
for ax, task in zip(axes, ("flip_flop", "sine_wave"), strict=False):
    xt, xl, annots, ymax = [], [], [], 0.0
    for j, scheme in enumerate(("uniform", "powerlaw")):
        tr_vals = vals(sel(mode_final, task=task, trainable="full",
                           init_scheme=scheme, recurrent_gain=0.9),
                       "match_mean_abs_log_err")
        rs_vals = vals(sel(mode_final, task=task, trainable="reservoir",
                           init_scheme=scheme, recurrent_gain=0.9),
                       "match_mean_abs_log_err")
        base = j * 1.6
        for k, (v, c, lbl) in enumerate(((tr_vals, SCHEME_COLOR[scheme], "trained"),
                                         (rs_vals, C_NULL, "frozen"))):
            x = base + k * 0.62
            ax.scatter(np.random.default_rng(0).normal(x, 0.055, len(v)), v,
                       s=9, color=c, alpha=0.45, edgecolors="none", zorder=3)
            ax.hlines(np.median(v), x - 0.2, x + 0.2, color=c, lw=2.4, zorder=4)
            xt.append(x)
            xl.append(f"{SCHEME_LABEL[scheme].split(' (')[0]}\n{lbl}")
            if v:
                ymax = max(ymax, max(v))
                g_lo, g_hi = min(g_lo, min(v)), max(g_hi, max(v))
        if tr_vals and rs_vals:
            res = compare_samples(tr_vals, rs_vals)
            annots.append((base + 0.31, res))
    ax.set_yscale("log")
    for x, res in annots:
        ax.text(x, 0.96, f"δ={res['cliffs_delta']:+.2f} {stars(res['p_value'])}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color=C_MUTED)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7.5)
    ax.set_title(task, fontsize=10)
# Shared log axis: one range covering both panels, with headroom for the annotations.
axes[0].set_ylim(bottom=g_lo / 3, top=g_hi * 25)
axes[0].set_ylabel("mean |log(τ_net / τ_task)|\n(lower = better matched)")
fig.suptitle("Does training improve timescale matching, or is it inherited from init?",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig4_matching_trained_vs_frozen.png"))
plt.show()

# %%
print("Matching error — trained (Θ=full) vs frozen (reservoir), g=0.9:\n")
for task in ("flip_flop", "sine_wave"):
    for scheme in ("uniform", "powerlaw"):
        a = vals(sel(mode_final, task=task, trainable="full",
                     init_scheme=scheme, recurrent_gain=0.9), "match_mean_abs_log_err")
        b = vals(sel(mode_final, task=task, trainable="reservoir",
                     init_scheme=scheme, recurrent_gain=0.9), "match_mean_abs_log_err")
        res = compare_samples(a, b)
        verdict = ("NOT significant -> inherited from init"
                   if res["p_value"] >= 0.05 else "training improves matching")
        print(f"  {task:10s} {scheme:9s} trained={np.median(a):.4f} "
              f"frozen={np.median(b):.4f}  δ={res['cliffs_delta']:+.2f} "
              f"p={res['p_value']:.2g}   {verdict}")

# %% [markdown]
# **The critical result.** For flip-flop under power-law init, trained and frozen networks
# match the task timescales *equally well* (0.006 vs 0.007, p ≈ 0.41, not significant).
# Power-law τ densely tiles [1, 200]; the flip-flop task timescales lie inside that range;
# so some mode is always near any target regardless of training — a dartboard effect.
#
# By contrast **uniform-init networks demonstrably do learn to match** (flip-flop
# 0.226 → 0.099, δ=−0.81, p=1.3e-05). If the claim is "task-trained RNNs learn to match
# task timescales", the clean evidence comes from the homogeneous baseline, and the
# power-law evidence is confounded by construction.
#
# ## Figure 4b — Per-neuron τ distribution, before and after training
#
# Answers R1's *"are there long timescales among the individual neuron timescales?"*, R2's
# *"show the distribution of neural timescales after training"*, and R2's challenge that the
# Shi et al. distribution is measured **post**-training so may not be comparable.
#
# `TauTrajectoryCallback` logged per-neuron τ for all 1200 runs; these are the pooled values
# (20 seeds × 512 units) at init and after training, in **timesteps** (τ/dt), for Θ=full
# where τ is itself a trained parameter.

# %%
tau_npz = os.path.join(SWEEP_DIR, "tau_pooled_g0.9.npz")
if not os.path.exists(tau_npz):
    print(f"missing {tau_npz} — run the τ extraction step first; skipping.")
else:
    tau = np.load(tau_npz)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, (prefix, task_name) in zip(axes, (("ff", "flip-flop"), ("sine", "sine-wave")),
                                       strict=False):
        for scheme in ("uniform", "powerlaw"):
            for tag, ls, alpha in (("init", ":", 0.55), ("final", "-", 1.0)):
                key = f"{prefix}_{scheme}_{tag}"
                if key not in tau:
                    continue
                v = tau[key]
                v = v[v > 0]
                bins = np.logspace(np.log10(max(v.min(), 1e-3)), np.log10(v.max()), 60)
                h, edges = np.histogram(v, bins=bins, density=True)
                ax.plot(0.5 * (edges[1:] + edges[:-1]), h, ls, color=SCHEME_COLOR[scheme],
                        lw=1.8, alpha=alpha,
                        label=f"{SCHEME_LABEL[scheme].split(' (')[0]} — {tag}")
        ax.set_xscale("log")
        ax.set_xlabel(r"per-neuron $\tau$ (timesteps)")
        ax.set_title(task_name, fontsize=10)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Per-neuron τ at init (dotted) vs after training (solid) — Θ=full, g=0.9",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig4b_tau_distribution.png"))
    plt.show()

    print("\nPer-neuron τ in timesteps (pooled 20 seeds × 512 units, Θ=full, g=0.9):\n")
    print(f"{'task':6s} {'scheme':9s} {'when':6s} | {'p05':>7s} {'p50':>7s} {'p95':>7s} "
          f"{'max':>8s} | {'>100':>6s} {'>500':>6s}")
    with open(os.path.join(SWEEP_DIR, "tau_summary.csv")) as f:
        for r in csv.DictReader(f):
            if r["g"] != "0.9":
                continue
            print(f"{r['task']:6s} {r['scheme']:9s} {r['tag']:6s} | "
                  f"{float(r['p05']):7.1f} {float(r['p50']):7.1f} {float(r['p95']):7.1f} "
                  f"{float(r['max']):8.1f} | {100 * float(r['frac_gt_100']):5.1f}% "
                  f"{100 * float(r['frac_gt_500']):5.1f}%")

# %% [markdown]
# **Three things this settles.**
#
# 1. **Are there long per-neuron timescales under uniform init? No.** After training, flip-flop
#    uniform τ spans only 3–22 timesteps (median 8.5) — nothing remotely long — yet those
#    networks still reach 90% accuracy. The long task timescales are carried by the network's
#    eigenmodes, not by individual neurons. That is direct quantitative support for the paper's
#    central network-level claim.
# 2. **Training barely moves the per-neuron τ distribution**, even though τ is a free parameter
#    here. Power-law flip-flop: median 142 → 102 timesteps, fraction above 100 steps 56.8% →
#    50.2%. The network solves the task by reshaping W (and hence the spectrum), not by
#    re-tuning per-neuron time constants.
# 3. **Consequence for the Shi et al. comparison (R2).** Since the post-training distribution is
#    essentially the initialization, the model does not *predict* a power-law of per-neuron
#    timescales — it is handed one. The measured (post-learning) biological distribution
#    therefore cannot be read as a confirmation of the model; the honest framing is that a broad
#    τ prior is an assumption the model takes in, not a result it produces.
#
# ---
# # R1 — "matching and coupling are single examples, not 'across our experiments'"
#
# ## Figure 5 — Coupling sparsity and one-to-one-ness across 20 seeds
#
# Two nulls are used throughout: the **frozen reservoir** network, and a **shuffled
# readout** control computed per run.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

# --- panel A: sparsity (normalised participation ratio; lower = sparser)
ax = axes[0]
xt, xl = [], []
for i, task in enumerate(("flip_flop", "sine_wave")):
    for j, scheme in enumerate(("uniform", "powerlaw")):
        base = i * 2.6 + j * 1.15
        series = [
            (vals(sel(mode_final, task=task, trainable="full", init_scheme=scheme,
                      recurrent_gain=0.9), "coup_participation_ratio_norm"),
             SCHEME_COLOR[scheme]),
            (vals(sel(mode_final, task=task, trainable="reservoir", init_scheme=scheme,
                      recurrent_gain=0.9), "coup_participation_ratio_norm"), C_NULL),
            (vals(sel(mode_final, task=task, trainable="full", init_scheme=scheme,
                      recurrent_gain=0.9), "coup_shuffled_participation_ratio_norm"),
             C_MUTED),
        ]
        for k, (v, c) in enumerate(series):
            if not v:
                continue
            x = base + k * 0.33
            ax.hlines(np.median(v), x - 0.13, x + 0.13, color=c, lw=2.6, zorder=4)
            ax.vlines(x, np.percentile(v, 25), np.percentile(v, 75), color=c, lw=1.2)
        xt.append(base + 0.33)
        xl.append(f"{task.split('_')[0]}\n{scheme}")
ax.set_xticks(xt)
ax.set_xticklabels(xl, fontsize=7.5)
ax.set_ylabel("normalised participation ratio\n(lower = sparser)")
ax.set_title("Coupling sparsity", fontsize=10)

# --- panel B: one-to-one-ness (dominance ratio; higher = more one-to-one)
ax = axes[1]
xt, xl = [], []
for i, task in enumerate(("flip_flop", "sine_wave")):
    for j, scheme in enumerate(("uniform", "powerlaw")):
        base = i * 2.6 + j * 1.15
        series = [
            (vals(sel(mode_final, task=task, trainable="full", init_scheme=scheme,
                      recurrent_gain=0.9), "coup_dominance_ratio_median"),
             SCHEME_COLOR[scheme]),
            (vals(sel(mode_final, task=task, trainable="reservoir", init_scheme=scheme,
                      recurrent_gain=0.9), "coup_dominance_ratio_median"), C_NULL),
            (vals(sel(mode_final, task=task, trainable="full", init_scheme=scheme,
                      recurrent_gain=0.9), "coup_shuffled_dominance_ratio_median"),
             C_MUTED),
        ]
        for k, (v, c) in enumerate(series):
            if not v:
                continue
            x = base + k * 0.33
            ax.hlines(np.median(v), x - 0.13, x + 0.13, color=c, lw=2.6, zorder=4)
            ax.vlines(x, np.percentile(v, 25), np.percentile(v, 75), color=c, lw=1.2)
        xt.append(base + 0.33)
        xl.append(f"{task.split('_')[0]}\n{scheme}")
ax.set_xticks(xt)
ax.set_xticklabels(xl, fontsize=7.5)
ax.set_ylabel("dominance ratio\n(higher = more one-to-one)")
ax.set_title("Coupling one-to-one-ness", fontsize=10)

handles = [plt.Line2D([], [], color=C_POWERLAW, lw=2.6),
           plt.Line2D([], [], color=C_UNIFORM, lw=2.6),
           plt.Line2D([], [], color=C_NULL, lw=2.6),
           plt.Line2D([], [], color=C_MUTED, lw=2.6)]
fig.legend(handles, ["trained (power-law)", "trained (uniform)",
                     "frozen reservoir null", "shuffled readout"],
           fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.10))
fig.suptitle("Output-to-mode coupling across 20 seeds (median, IQR whiskers; g=0.9)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig5_coupling.png"))
plt.show()

# %%
print("One-to-one-ness (dominance ratio) — uniform vs power-law, both trained:\n")
for task in ("flip_flop", "sine_wave"):
    for tr in ("full", "fixedA"):
        a = vals(sel(mode_final, task=task, trainable=tr, init_scheme="uniform",
                     recurrent_gain=0.9), "coup_dominance_ratio_median")
        b = vals(sel(mode_final, task=task, trainable=tr, init_scheme="powerlaw",
                     recurrent_gain=0.9), "coup_dominance_ratio_median")
        res = compare_samples(a, b)
        print(f"  {task:10s} {tr:7s} uniform={np.median(a):.3f} "
              f"powerlaw={np.median(b):.3f}  δ={res['cliffs_delta']:+.2f} "
              f"p={res['p_value']:.2g}")

print("\nSparsity — trained vs frozen reservoir null:\n")
for task in ("flip_flop", "sine_wave"):
    for scheme in ("uniform", "powerlaw"):
        a = vals(sel(mode_final, task=task, trainable="full", init_scheme=scheme,
                     recurrent_gain=0.9), "coup_participation_ratio_norm")
        b = vals(sel(mode_final, task=task, trainable="reservoir", init_scheme=scheme,
                     recurrent_gain=0.9), "coup_participation_ratio_norm")
        res = compare_samples(a, b)
        flag = "" if res["cliffs_delta"] < 0 else "   <-- ANOMALY: frozen is sparser"
        print(f"  {task:10s} {scheme:9s} trained={np.median(a):.3f} "
              f"frozen={np.median(b):.3f}  δ={res['cliffs_delta']:+.2f} "
              f"p={res['p_value']:.2g}{flag}")

# %% [markdown]
# **What this shows.**
#
# * **Sparsity survives quantification**: trained networks are reliably sparser than both
#   nulls — *except* sine-wave/uniform, where the frozen network is sparser than the
#   trained one. That anomaly needs an explanation before this panel goes to a reviewer.
# * **One-to-one-ness is real but strongest in the baseline**: uniform is significantly
#   more one-to-one than power-law in all four trained conditions. The phenomenon R3
#   called the only non-trivial content is a property of trained RNNs generally, and is
#   *weaker* under the init we advocate.
#
# ### Metric caveat — `assignment_uniqueness` is unusable as defined

# %%
print("assignment_uniqueness (trained vs shuffled) — degenerate, do not report:\n")
for task in ("flip_flop", "sine_wave"):
    for scheme in ("uniform", "powerlaw"):
        a = vals(sel(mode_final, task=task, trainable="full", init_scheme=scheme,
                     recurrent_gain=0.9), "coup_assignment_uniqueness")
        b = vals(sel(mode_final, task=task, trainable="full", init_scheme=scheme,
                     recurrent_gain=0.9), "coup_shuffled_assignment_uniqueness")
        print(f"  {task:10s} {scheme:9s} trained={np.median(a):.2f} "
              f"shuffled={np.median(b):.2f}")
print("\n  flip_flop: saturates at 1.00 including the shuffled control -> no discrimination")
print("  sine_wave/uniform: 0.00 trained vs 1.00 shuffled -> worse than chance")
print("  => needs redefinition (design decision 5d) before it is reportable")

# %% [markdown]
# ---
# ## Figure 6 — Mechanism: what power-law init does to the spectrum
#
# Both convergence results (flip-flop robustness, sine-wave failure) point at the same
# cause. These statistics are computed from the Jacobian eigenvalues **at initialisation**,
# before any training.

# %%
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
for ax, key, ylab, title in (
    (axes[0], "spec_init_frac_near_real_axis",
     "fraction of eigenvalues\nwith |Im λ| < 0.1", "Spectral pinching toward the real axis"),
    (axes[1], "spec_init_decay_timescale_range",
     "decay-timescale range\n(timesteps, log scale)", "Range of available timescales"),
):
    for scheme in ("uniform", "powerlaw"):
        xs, ys, lo, hi = [], [], [], []
        for g in G_GRID:
            v = vals([r for r in run_table
                      if r["cfg.tau_init.scheme"] == scheme
                      and abs((fnum(r, "cfg.recurrent_gain") or -1) - g) < 1e-9
                      and r["cfg.trainable"] == "full"], key)
            if v:
                xs.append(g)
                ys.append(np.median(v))
                lo.append(np.percentile(v, 25))
                hi.append(np.percentile(v, 75))
        ax.plot(xs, ys, "-o", color=SCHEME_COLOR[scheme], lw=2, ms=5,
                label=SCHEME_LABEL[scheme])
        ax.fill_between(xs, lo, hi, color=SCHEME_COLOR[scheme], alpha=0.12, lw=0)
    ax.set_xlabel("recurrent gain $g$")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(G_GRID)
    ax.set_xticklabels([str(g) for g in G_GRID], rotation=45, ha="right", fontsize=8)
axes[1].set_yscale("log")
axes[0].legend(fontsize=8, loc="best")
fig.suptitle("Jacobian spectrum at initialisation (before training)", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig6_spectrum_at_init.png"))
plt.show()

# %% [markdown]
# **Interpretation.** Power-law init both pins a larger fraction of eigenvalues near the
# real axis and opens up a far wider range of decay timescales, and it does so almost
# independently of $g$ — which is exactly the signature that explains the two headline
# behaviours: gain-invariance (the spectrum barely depends on $g$) and sine-wave failure
# (few complex modes available for oscillation).
#
# Note the left panel reads **1.00 at every $g$** for power-law: *all* eigenvalues start
# within $|\mathrm{Im}\,\lambda| < 0.1$. It is worth checking that against the frequencies
# the sine-wave task actually demands.

# %%
print("Sine-wave targets vs the pinched band (eps_real_axis = 0.1):\n")
print(f"  {'period T':>9s}  {'arg λ = 2π/T':>13s}  {'≈|Im λ| at |λ|≈1':>17s}   inside band?")
for T in (20.0, 50.0, 100.0):
    arg = 2 * np.pi / T
    im = np.sin(arg)
    print(f"  {T:9.0f}  {arg:13.3f}  {im:17.3f}   "
          f"{'yes' if im < 0.1 else 'NO — outside'}")
print("\n  => 2 of the 3 target frequencies lie outside the band that contains 100% of")
print("     power-law eigenvalues at init. The fastest component (T=20) is furthest out.")
print("\n  Caveat: W is trainable in Θ=full/fixedA, so this is about how far the optimiser")
print("  must travel from its starting spectrum, not strict representability. The observed")
print("  plateau (fig 3) is what that difficulty looks like in practice.")

# %% [markdown]
# ---
# ## Summary of what each reviewer point looks like after the sweep

# %%
summary = [
    ("R3", "power-law only vs weak (high-g) baseline",
     "CONFIRMED — uniform wins 14/15 best-vs-best", "fig1, fig2"),
    ("R3", "final loss vs g must be first-class",
     "DONE — and it is where the one favourable result lives", "fig1"),
    ("R3", "empirical content thin / coupling is the non-trivial part",
     "quantified across 20 seeds; stronger in uniform than power-law", "fig5"),
    ("R2", "is timescale matching necessary?",
     "reservoir control: power-law matching is INHERITED, not learned", "fig4"),
    ("R2", "does a fixed network + trained readout solve the task?",
     "no — reservoir loss far above trained in the linear regime", "fig1 (Θ axis)"),
    ("R2", "Shi et al. comparability (post-training τ)",
     "NOT ANSWERABLE — needs log_tau_dist, not enabled", "-"),
    ("R1", "matching/coupling shown as single examples",
     "both now across 20 seeds with rank-based tests", "fig4, fig5"),
    ("R1", "results must be statistically tested",
     "Mann-Whitney U + Cliff's delta throughout", "all"),
]
w = max(len(s[1]) for s in summary)
print(f"{'rev':4s} {'point':{w}s}  verdict")
print("-" * (w + 60))
for rev, point, verdict, figs in summary:
    print(f"{rev:4s} {point:{w}s}  {verdict}  [{figs}]")

# %% [markdown]
# ### Scope limits, stated plainly
#
# 1. **Linear regime only** (activation = Identity). The tanh-voltage sweep stalled at
#    193/1200 runs; none of the above constrains the nonlinear case.
# 2. **Uniform's optimum may be below the grid** — it is still improving at g=0.3.
# 3. **Two loose ends** before any of this is reviewer-facing: the sine-wave/uniform
#    sparsity anomaly, and the degenerate `assignment_uniqueness` metric.
