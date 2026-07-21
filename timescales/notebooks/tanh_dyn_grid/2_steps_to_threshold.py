# %% [markdown]
# # Tanh-Voltage Initialization Speed Comparison
# Voltage-Tanh RNN  ·  uniform vs power-law τ  ·  fixed A vs trainable A
#
# Loads:
#   tanh_voltage_grid  (voltage-Tanh RNN,  g=0.5 and g=0.9,
#                       uniform + powerlawtau,  fixedA + full,  5 seeds each)
#
# Produces one figure with:
#   · Two loss-curve panels  (g = 0.5  |  g = 0.9)
#   · Two threshold strip panels  (one per gain)
# Plus a second figure filtered to trainable-A only.
#
# ── STRUCTURE ─────────────────────────────────────────────────────────────────
#   PART 0 : Configuration  ← edit paths / thresholds here
#   PART 1 : Load records
#   PART 2 : Helpers
#   PART 3 : Fig 1 — all conditions (fixed A + trainable A)
#   PART 4 : Fig 2 — trainable A only

# %% Imports
import json
import os
import re
import subprocess
import sys

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()
os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())

plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 0 — Configuration
# ══════════════════════════════════════════════════════════════════════════════

SWEEP_DIR = (
    "/home/facosta/timescales/timescales/logs/experiments"
    "/tanh_voltage_grid_20260506_100132"
)

SAVE_FIGS = True
FIGS_DIR  = os.path.join("notebooks", "tanh_dyn_grid", "figs", "init_speed")
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Experiment axes ───────────────────────────────────────────────────────────
GAINS       = [0.5, 0.9]
INITS_PLOT  = ["uniform", "powerlawtau"]
THETAS_PLOT = ["fixedA", "full"]

# ── Threshold / centering options ─────────────────────────────────────────────
THRESHOLD = 0.90    # accuracy threshold for "steps to threshold"
CENTER    = "mean"  # "mean" | "median"
LOG_LOSS  = True    # log-scale y-axis for loss curves
LOG_STEPS = False   # log-scale y-axis for the strip plot

# ── Color scheme: YlOrRd ──────────────────────────────────────────────────────
INIT_COLORS = {
    "uniform":     plt.cm.YlOrRd(0.42),   # amber-orange
    "powerlawtau": plt.cm.YlOrRd(0.80),   # deep red-orange
}
INIT_LABELS = {
    "uniform":     "Uniform  (τ = 1)",
    "powerlawtau": r"Power-law τ  (β = 1)",
}

# ── θ encoding ────────────────────────────────────────────────────────────────
THETA_MARKERS = {"fixedA": "o",  "full": "^"}
THETA_LS      = {"fixedA": "-",  "full": "--"}
THETA_LABELS  = {"fixedA": "Fixed A", "full": "Trainable A"}
_THETA_OFF    = {"fixedA": -0.15, "full": +0.15}


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 1 — Load records
# ══════════════════════════════════════════════════════════════════════════════

_RE = re.compile(
    r"^g(?P<gain>\d+(?:\.\d+)?)_tanh_voltage_"
    r"(?P<init>uniform|powerlawtau)_"
    r"(?P<theta>fixedA|full)_ff$"
)

_records: list[dict] = []
if not os.path.isdir(SWEEP_DIR):
    print(f"⚠ Sweep dir not found: {SWEEP_DIR}")
else:
    for _en in sorted(os.listdir(SWEEP_DIR)):
        _ed = os.path.join(SWEEP_DIR, _en)
        if not os.path.isdir(_ed):
            continue
        _m = _RE.match(_en)
        if _m is None:
            continue
        _gain  = float(_m.group("gain"))
        _init  = _m.group("init")
        _theta = _m.group("theta")
        for _sdn in sorted(os.listdir(_ed)):
            if not _sdn.startswith("seed_"):
                continue
            _lf = os.path.join(_ed, _sdn, "training_losses.json")
            if not os.path.exists(_lf):
                continue
            with open(_lf) as _f:
                _ld = json.load(_f)
            _steps    = np.asarray(_ld.get("steps",          []), dtype=float)
            _val_loss = np.asarray(_ld.get("val_losses",     []), dtype=float)
            _val_acc  = np.asarray(_ld.get("val_accuracies", []), dtype=float)
            if len(_steps) == 0 or len(_val_loss) == 0:
                continue
            _records.append(dict(
                gain=_gain, init=_init, theta=_theta,
                seed=int(_sdn.split("_")[1]),
                steps=_steps, val_losses=_val_loss, val_accs=_val_acc,
            ))

df = pd.DataFrame(_records)
print(f"Records loaded: {len(df)}")
if not df.empty:
    print(df.groupby(["gain", "init", "theta"]).size().rename("n_seeds").to_string())


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 2 — Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _steps_to_thr(val_accs, steps, thr: float = THRESHOLD) -> float:
    """First training step where val_acc >= thr; np.nan if never reached."""
    va = np.asarray(val_accs, dtype=float)
    st = np.asarray(steps,    dtype=float)
    n  = min(len(va), len(st))
    for i in range(n):
        if np.isfinite(va[i]) and va[i] >= thr:
            return float(st[i])
    return np.nan


def _center_val(arr: np.ndarray) -> float:
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
        return np.nan
    return float(np.nanmean(v) if CENTER == "mean" else np.nanmedian(v))


def _plot_loss_panel(ax: "plt.Axes", df_sub: pd.DataFrame, g: float) -> None:
    """Validation-loss curves coloured by init, linestyle by theta."""
    thetas = THETAS_PLOT if "theta" in df_sub.columns else ["fixedA"]
    rows   = df_sub[df_sub["gain"] == g]

    for init in INITS_PLOT:
        for theta in thetas:
            sel = rows[(rows["init"] == init) & (rows["theta"] == theta)]
            if sel.empty:
                continue
            col    = INIT_COLORS[init]
            ls     = THETA_LS.get(theta, "-")
            curves = []
            for _, row in sel.iterrows():
                st = np.asarray(row["steps"],      dtype=float)
                vl = np.asarray(row["val_losses"], dtype=float)
                n  = min(len(st), len(vl))
                if n < 3:
                    continue
                ax.plot(st[:n], vl[:n], color=col, lw=0.8, alpha=0.22,
                        ls=ls, zorder=2)
                curves.append((st[:n], vl[:n]))
            if not curves:
                continue
            L    = min(len(c[1]) for c in curves)
            mean = np.stack([c[1][:L] for c in curves]).mean(axis=0)
            lbl  = (f"{INIT_LABELS[init]}  [{THETA_LABELS[theta]}]"
                    if len(thetas) > 1 else INIT_LABELS[init])
            ax.plot(curves[0][0][:L], mean, color=col, lw=2.0, alpha=1.0,
                    ls=ls, zorder=3, label=lbl)

    if LOG_LOSS:
        ax.set_yscale("log")
    ax.set_title(f"g = {g}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel("Validation loss", fontsize=10)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )
    _h, _l = ax.get_legend_handles_labels()
    _seen, _uh, _ul = set(), [], []
    for h, l in zip(_h, _l):
        if l not in _seen:
            _seen.add(l); _uh.append(h); _ul.append(l)
    if _uh:
        ax.legend(_uh, _ul, fontsize=8, loc="upper right", framealpha=0.85,
                  borderaxespad=0.5)


def _plot_strip_panel(ax: "plt.Axes", df_sub: pd.DataFrame, g: float) -> None:
    """Strip plot for a single gain: x = init, y = steps to THRESHOLD.

    Color = init.  Marker shape = theta (○ fixedA, △ trainable A).
    Non-converging seeds shown as faint × near the top, with K/N annotation.
    """
    thetas    = THETAS_PLOT if "theta" in df_sub.columns else ["fixedA"]
    _x_pos    = {init: i for i, init in enumerate(INITS_PLOT)}
    _rng      = np.random.default_rng(seed=0)

    # Collect finite values to place non-converging markers consistently
    _all_finite: list[float] = []
    for init in INITS_PLOT:
        for theta in thetas:
            sel = df_sub[(df_sub["gain"] == g) & (df_sub["init"] == init)
                         & (df_sub["theta"] == theta)]
            for _, row in sel.iterrows():
                s = _steps_to_thr(row["val_accs"], row["steps"])
                if np.isfinite(s):
                    _all_finite.append(s)
    y_max_data = max(_all_finite) if _all_finite else 1.0

    for init in INITS_PLOT:
        col = INIT_COLORS[init]
        for theta in thetas:
            sel = df_sub[(df_sub["gain"] == g) & (df_sub["init"] == init)
                         & (df_sub["theta"] == theta)]
            if sel.empty:
                continue
            mkr    = THETA_MARKERS.get(theta, "o")
            x_off  = _THETA_OFF.get(theta, 0.0)
            x_base = float(_x_pos[init]) + x_off

            seed_steps = np.array([
                _steps_to_thr(row["val_accs"], row["steps"])
                for _, row in sel.iterrows()
            ])
            n_total  = len(seed_steps)
            n_finite = int(np.sum(np.isfinite(seed_steps)))
            finite_vals = seed_steps[np.isfinite(seed_steps)]
            inf_vals    = seed_steps[~np.isfinite(seed_steps)]

            if len(finite_vals) > 0:
                jitter = _rng.uniform(-0.05, 0.05, size=len(finite_vals))
                ax.scatter(x_base + jitter, finite_vals,
                           color=col, marker=mkr, s=46, alpha=0.55,
                           zorder=4, linewidths=0)
                ctr = _center_val(finite_vals)
                ax.scatter(x_base, ctr, color=col, marker=mkr,
                           s=200, alpha=1.0, zorder=5,
                           edgecolors="white", linewidths=1.4)

            if len(inf_vals) > 0:
                y_dnc = y_max_data * 1.08
                jitter_x = _rng.uniform(-0.05, 0.05, size=len(inf_vals))
                ax.scatter(x_base + jitter_x, np.full(len(inf_vals), y_dnc),
                           color=col, marker="x", s=50, alpha=0.50,
                           zorder=4, linewidths=1.2)

            ax.text(x_base, -0.08, f"{n_finite}/{n_total}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5,
                    color=col, alpha=0.85)

    ax.set_xticks(list(_x_pos.values()))
    ax.set_xticklabels([INIT_LABELS[i] for i in INITS_PLOT], fontsize=9.5)
    ax.set_xlim(-0.50, len(INITS_PLOT) - 0.50)
    ax.set_ylabel(f"Steps to ≥ {int(THRESHOLD * 100)} % accuracy", fontsize=10)
    if LOG_STEPS:
        ax.set_yscale("log")
    ax.set_title(f"g = {g}", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", length=0)

    if len(thetas) > 1:
        from matplotlib.lines import Line2D
        _leg = [
            Line2D([0], [0], marker=THETA_MARKERS[t], color="0.35",
                   ls="none", ms=7, label=THETA_LABELS[t])
            for t in thetas
        ]
        ax.legend(handles=_leg, fontsize=8, loc="upper right",
                  framealpha=0.80, borderaxespad=0.4)


def _make_fig(
    df_sub: pd.DataFrame,
    title: str,
    fname: str,
) -> None:
    """2×2 figure: loss curves (top) + threshold strip plots (bottom)."""
    if df_sub.empty:
        print(f"⚠ No data for {fname!r} — skipping.")
        return

    fig = plt.figure(figsize=(11.5, 8.2), layout="constrained")
    gs  = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0])
    ax_l05 = fig.add_subplot(gs[0, 0])
    ax_l09 = fig.add_subplot(gs[0, 1])
    ax_t05 = fig.add_subplot(gs[1, 0])
    ax_t09 = fig.add_subplot(gs[1, 1])

    _plot_loss_panel(ax_l05, df_sub, g=0.5)
    _plot_loss_panel(ax_l09, df_sub, g=0.9)
    _plot_strip_panel(ax_t05, df_sub, g=0.5)
    _plot_strip_panel(ax_t09, df_sub, g=0.9)

    # Share y-axis between the two strip panels
    _all_y = []
    for _ax in (ax_t05, ax_t09):
        for c in _ax.collections:
            oy = c.get_offsets()[:, 1]
            _all_y.extend(oy[np.isfinite(oy)].tolist())
    if _all_y:
        _ypad = (max(_all_y) - min(_all_y)) * 0.15 or max(_all_y) * 0.10 or 10
        _ylim = (max(0, min(_all_y) - _ypad * 0.5), max(_all_y) + _ypad)
        ax_t05.set_ylim(*_ylim)
        ax_t09.set_ylim(*_ylim)
    ax_t09.set_ylabel("")

    fig.suptitle(title, fontsize=14, fontweight="bold")

    if SAVE_FIGS and fname:
        p = os.path.join(FIGS_DIR, fname)
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"  saved → {p}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 3 — Fig 1: all conditions (fixed A + trainable A)
# ══════════════════════════════════════════════════════════════════════════════

_make_fig(
    df,
    title=(
        "Voltage-Tanh RNN  —  uniform vs power-law τ init  "
        "(fixed A  ·  trainable A,  6-bit flip-flop)"
    ),
    fname="fig1_init_speed_voltage_tanh.pdf",
)


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 4 — Fig 2: trainable A only
# ══════════════════════════════════════════════════════════════════════════════

_make_fig(
    df[df["theta"] == "full"].copy(),
    title=(
        "Voltage-Tanh RNN  —  uniform vs power-law τ init  "
        "(trainable A only,  6-bit flip-flop)"
    ),
    fname="fig2_init_speed_voltage_tanh_trainableA.pdf",
)
