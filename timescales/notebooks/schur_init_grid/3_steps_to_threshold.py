# %% [markdown]
# # Initialization Speed Comparison
# Linear RNN  ·  Rate-Tanh  ·  Voltage-Tanh
#
# Loads:
#   schur_init_grid  (linear RNN,  g=0.5 and g=0.9,  5 seeds per condition)
#   tanh_dyn_grid    (tanh RNN,    rate + voltage,    g=0.5 and g=0.9)
#
# Filters to **uniform** and **power-law τ** initializations (schurH excluded).
# For the linear network, uses θ = fixedA to match the tanh grid.
#
# Produces one figure per network type, each with:
#   · Two loss-curve panels  (g = 0.5  |  g = 0.9)
#   · One strip-plot panel   (steps to 90 % accuracy, all seeds + mean/median)
#
# ── STRUCTURE ─────────────────────────────────────────────────────────────────
#   PART 0 : Configuration  ← edit paths / thresholds here
#   PART 1 : Load linear records (schur_init_grid)
#   PART 2 : Load tanh records (tanh_dyn_grid)
#   PART 3 : Shared helpers
#   PART 4 : Fig 1 — Linear RNN
#   PART 5 : Fig 2 — Rate-Tanh RNN
#   PART 6 : Fig 3 — Voltage-Tanh RNN

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
import yaml

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

SWEEP_DIRS_LINEAR = [
    # Server A  (g = 0.5)
    "/home/facosta/timescales/timescales/logs/experiments/schur_init_grid_g05_20260505_061150",
    # Server B  (g = 0.9)
    "/home/facosta/timescales/timescales/logs/experiments/schur_init_grid_g09_20260504_230709",
]
SWEEP_DIR_TANH = (
    "/home/facosta/timescales/timescales/logs/experiments"
    "/tanh_dyn_grid_20260505_230622"
)

SAVE_FIGS = True
FIGS_DIR  = os.path.join("notebooks", "schur_init_grid", "figs", "init_speed")
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Experiment axes ───────────────────────────────────────────────────────────
GAINS        = [0.5, 0.9]
INITS_PLOT   = ["uniform", "powerlawtau"]        # schurH excluded throughout
THETAS_PLOT  = ["fixedA", "full"]                # fixedA = fixed A, full = trainable A

# ── Threshold / centering options ─────────────────────────────────────────────
THRESHOLD  = 0.90   # accuracy threshold for "steps to threshold" metric
CENTER     = "mean" # "mean" | "median"  (central marker in strip plot)
LOG_LOSS   = True   # log-scale y-axis for loss curves
LOG_STEPS  = False  # log-scale y-axis for the strip plot

# ── Color scheme: YlOrRd ──────────────────────────────────────────────────────
INIT_COLORS = {
    "uniform":     plt.cm.YlOrRd(0.42),   # amber-orange
    "powerlawtau": plt.cm.YlOrRd(0.80),   # deep red-orange
}
INIT_LABELS = {
    "uniform":     "Uniform  (τ = 1)",
    "powerlawtau": r"Power-law τ  (β = 1)",
}

# ── θ (A-matrix) encoding ─────────────────────────────────────────────────────
THETA_MARKERS = {"fixedA": "o",   "full": "^"}    # circle = fixed A, triangle = trainable A
THETA_LS      = {"fixedA": "-",   "full": "--"}   # solid = fixed A, dashed = trainable A
THETA_LABELS  = {"fixedA": "Fixed A",  "full": "Trainable A"}

# x-offset within each init position: each (init, theta) gets its own lane
_THETA_OFF    = {"fixedA": -0.15,  "full": +0.15}


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 1 — Load linear records (schur_init_grid, both tasks, fixedA only)
# ══════════════════════════════════════════════════════════════════════════════

_LIN_RE = re.compile(
    r"^g(?P<gain>\d+(?:\.\d+)?)_"
    r"(?P<init>uniform|powerlawtau|schurH)_"
    r"(?P<theta>full|fixedA)_"
    r"(?P<task>ff|sine)$"
)

_lin_records: list[dict] = []
for _sd in SWEEP_DIRS_LINEAR:
    if not os.path.isdir(_sd):
        print(f"⚠ Linear sweep dir not found — skipping: {_sd}")
        continue
    for _en in sorted(os.listdir(_sd)):
        _ed = os.path.join(_sd, _en)
        if not os.path.isdir(_ed):
            continue
        _m = _LIN_RE.match(_en)
        if _m is None:
            continue
        _gain, _init, _theta, _task = (
            float(_m.group("gain")), _m.group("init"),
            _m.group("theta"), _m.group("task"),
        )
        # Early filter: only the two inits we care about and the two thetas
        if _init not in INITS_PLOT or _theta not in THETAS_PLOT:
            continue
        for _sdn in sorted(os.listdir(_ed)):
            if not _sdn.startswith("seed_"):
                continue
            _sp = os.path.join(_ed, _sdn)
            _lf = os.path.join(_sp, "training_losses.json")
            if not os.path.exists(_lf):
                continue
            with open(_lf) as _f:
                _ld = json.load(_f)
            _steps    = _ld.get("steps", [])
            _val_loss = _ld.get("val_losses", [])
            _val_acc  = _ld.get("val_accuracies", [])
            if not _steps or not _val_loss:
                continue
            _lin_records.append(dict(
                gain=_gain, init=_init, theta=_theta, task=_task,
                seed=int(_sdn.split("_")[1]),
                steps=_steps, val_losses=_val_loss, val_accs=_val_acc,
            ))

df_lin = pd.DataFrame(_lin_records)
print(f"Linear records loaded: {len(df_lin)}")
if not df_lin.empty:
    print(df_lin.groupby(["task", "gain", "init", "theta"]).size().rename("n_seeds").to_string())

df_lin_ff   = df_lin[df_lin["task"] == "ff"].copy()   if not df_lin.empty else df_lin
df_lin_sine = df_lin[df_lin["task"] == "sine"].copy() if not df_lin.empty else df_lin


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 2 — Load tanh records (tanh_dyn_grid)
# ══════════════════════════════════════════════════════════════════════════════

_TANH_RE = re.compile(
    r"^g(?P<gain>\d+(?:\.\d+)?)_tanh_"
    r"(?P<dynamics>rate|voltage)_"
    r"(?P<init>uniform|powerlawtau|schurH)_"
    r"fixedA_ff$"
)

_tanh_records: list[dict] = []
if not os.path.isdir(SWEEP_DIR_TANH):
    print(f"⚠ Tanh sweep dir not found — skipping: {SWEEP_DIR_TANH}")
else:
    for _en in sorted(os.listdir(SWEEP_DIR_TANH)):
        _ed = os.path.join(SWEEP_DIR_TANH, _en)
        if not os.path.isdir(_ed):
            continue
        _m = _TANH_RE.match(_en)
        if _m is None:
            continue
        _gain, _dyn, _init = (
            float(_m.group("gain")), _m.group("dynamics"), _m.group("init"),
        )
        if _init not in INITS_PLOT:
            continue
        for _sdn in sorted(os.listdir(_ed)):
            if not _sdn.startswith("seed_"):
                continue
            _sp = os.path.join(_ed, _sdn)
            _lf = os.path.join(_sp, "training_losses.json")
            if not os.path.exists(_lf):
                continue
            with open(_lf) as _f:
                _ld = json.load(_f)
            _steps    = np.asarray(_ld.get("steps",           []), dtype=float)
            _val_loss = np.asarray(_ld.get("val_losses",      []), dtype=float)
            _val_acc  = np.asarray(_ld.get("val_accuracies",  []), dtype=float)
            if len(_steps) == 0 or len(_val_loss) == 0:
                continue
            _tanh_records.append(dict(
                gain=_gain, dynamics=_dyn, init=_init, theta="fixedA",
                seed=int(_sdn.split("_")[1]),
                steps=_steps, val_losses=_val_loss, val_accs=_val_acc,
            ))

df_tanh = pd.DataFrame(_tanh_records)
print(f"Tanh records loaded: {len(df_tanh)}")
if not df_tanh.empty:
    print(
        df_tanh.groupby(["gain", "dynamics", "init"])
        .size().rename("n_seeds").to_string()
    )


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 3 — Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _steps_to_thr(val_accs, steps, thr: float = THRESHOLD) -> float:
    """Return the first training step where val_acc >= thr; np.nan if never reached."""
    va = np.asarray(val_accs, dtype=float)
    st = np.asarray(steps,    dtype=float)
    n  = min(len(va), len(st))
    for i in range(n):
        if np.isfinite(va[i]) and va[i] >= thr:
            return float(st[i])
    return np.nan


def _center_val(arr: np.ndarray) -> float:
    """Mean or median of finite values (controlled by CENTER constant)."""
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
        return np.nan
    return float(np.nanmean(v) if CENTER == "mean" else np.nanmedian(v))


def _plot_loss_panel(ax: "plt.Axes", df_sub: pd.DataFrame, g: float) -> None:
    """Validation-loss curves for all seeds of a given gain, coloured by init.

    Color = init type.  Linestyle = theta (fixedA solid, full dashed).
    Individual seeds: thin semi-transparent.  Mean: bold opaque.
    """
    has_theta = "theta" in df_sub.columns
    thetas    = THETAS_PLOT if has_theta else ["fixedA"]
    rows      = df_sub[df_sub["gain"] == g]

    for init in INITS_PLOT:
        for theta in thetas:
            sel = rows[rows["init"] == init]
            if has_theta:
                sel = sel[sel["theta"] == theta]
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
    """Strip plot for a single gain: x = init type, y = steps to THRESHOLD.

    Color = init.  Marker shape = theta (○ fixedA, △ trainable A).
    Non-converging seeds are shown as faint 'x' markers at the top of the axis
    and annotated with "K/N converged".
    """
    has_theta = "theta" in df_sub.columns
    thetas    = THETAS_PLOT if has_theta else ["fixedA"]
    _x_pos    = {init: i for i, init in enumerate(INITS_PLOT)}
    _rng      = np.random.default_rng(seed=0)

    all_finite: list[float] = []
    # First pass: collect finite values to set y-limits for the "×" placement
    for init in INITS_PLOT:
        for theta in thetas:
            sel = df_sub[(df_sub["gain"] == g) & (df_sub["init"] == init)]
            if has_theta:
                sel = sel[sel["theta"] == theta]
            for _, row in sel.iterrows():
                s = _steps_to_thr(row["val_accs"], row["steps"])
                if np.isfinite(s):
                    all_finite.append(s)

    y_max_data = max(all_finite) if all_finite else 1.0

    for init in INITS_PLOT:
        col = INIT_COLORS[init]
        for theta in thetas:
            sel = df_sub[(df_sub["gain"] == g) & (df_sub["init"] == init)]
            if has_theta:
                sel = sel[sel["theta"] == theta]
            if sel.empty:
                continue
            mkr    = THETA_MARKERS.get(theta, "o")
            x_off  = _THETA_OFF.get(theta, 0.0)
            x_base = float(_x_pos[init]) + x_off

            seed_steps = np.array([
                _steps_to_thr(row["val_accs"], row["steps"])
                for _, row in sel.iterrows()
            ])
            n_total    = len(seed_steps)
            n_finite   = int(np.sum(np.isfinite(seed_steps)))
            finite_vals = seed_steps[np.isfinite(seed_steps)]
            inf_vals    = seed_steps[~np.isfinite(seed_steps)]

            # Converged seeds: jittered scatter
            if len(finite_vals) > 0:
                jitter = _rng.uniform(-0.05, 0.05, size=len(finite_vals))
                ax.scatter(x_base + jitter, finite_vals,
                           color=col, marker=mkr, s=46, alpha=0.55,
                           zorder=4, linewidths=0)
                ctr = _center_val(finite_vals)
                ax.scatter(x_base, ctr, color=col, marker=mkr,
                           s=200, alpha=1.0, zorder=5,
                           edgecolors="white", linewidths=1.4)

            # Non-converged seeds: faint × near top of data range
            if len(inf_vals) > 0:
                y_dnc = y_max_data * 1.08
                jitter_x = _rng.uniform(-0.05, 0.05, size=len(inf_vals))
                ax.scatter(x_base + jitter_x,
                           np.full(len(inf_vals), y_dnc),
                           color=col, marker="x", s=50, alpha=0.50,
                           zorder=4, linewidths=1.2)

            # Convergence annotation below each lane
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

    # Legend: theta marker shapes (only when multiple thetas present)
    if len(thetas) > 1:
        from matplotlib.lines import Line2D
        _leg = [
            Line2D([0], [0], marker=THETA_MARKERS[t], color="0.35",
                   ls="none", ms=7, label=THETA_LABELS[t])
            for t in thetas
        ]
        ax.legend(handles=_leg, fontsize=8, loc="upper right",
                  framealpha=0.80, borderaxespad=0.4)


def _make_network_fig(
    df_sub: pd.DataFrame,
    network_title: str,
    fname: str,
) -> None:
    """Full figure: 2×2 grid — loss curves (top) + threshold strip plots (bottom)."""
    fig = plt.figure(figsize=(11.5, 8.2), layout="constrained")
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.15, 1.0],
    )
    ax_l05  = fig.add_subplot(gs[0, 0])
    ax_l09  = fig.add_subplot(gs[0, 1])
    ax_t05  = fig.add_subplot(gs[1, 0])
    ax_t09  = fig.add_subplot(gs[1, 1])

    _plot_loss_panel(ax_l05, df_sub, g=0.5)
    _plot_loss_panel(ax_l09, df_sub, g=0.9)
    _plot_strip_panel(ax_t05, df_sub, g=0.5)
    _plot_strip_panel(ax_t09, df_sub, g=0.9)

    # Share y-axis between the two strip panels for direct comparison
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
    ax_t09.set_ylabel("")   # y-label only on left panel

    fig.suptitle(network_title, fontsize=14, fontweight="bold")

    if SAVE_FIGS and fname:
        p = os.path.join(FIGS_DIR, fname)
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"  saved → {p}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 4 — Fig 1: Linear RNN
# ══════════════════════════════════════════════════════════════════════════════

_make_network_fig(
    df_lin_ff,
    network_title=(
        "Linear RNN  —  uniform vs power-law τ init  "
        "(fixed A  ·  trainable A,  6-bit flip-flop)"
    ),
    fname="fig1_init_speed_linear_ff.pdf",
)

_make_network_fig(
    df_lin_ff[df_lin_ff["theta"] == "full"].copy(),
    network_title=(
        "Linear RNN  —  uniform vs power-law τ init  "
        "(trainable A only,  6-bit flip-flop)"
    ),
    fname="fig1b_init_speed_linear_ff_trainableA.pdf",
)


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 5 — Fig 2: Rate-Tanh RNN  (flip-flop)
# ══════════════════════════════════════════════════════════════════════════════

if not df_tanh.empty:
    _make_network_fig(
        df_tanh[df_tanh["dynamics"] == "rate"].copy(),
        network_title=(
            "Rate-Tanh RNN  —  uniform vs power-law τ init  "
            "(θ = fixedA,  6-bit flip-flop)"
        ),
        fname="fig2_init_speed_rate_tanh_ff.pdf",
    )
else:
    print("⚠ df_tanh is empty — check SWEEP_DIR_TANH.")


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 6 — Fig 3: Voltage-Tanh RNN  (flip-flop)
# ══════════════════════════════════════════════════════════════════════════════

if not df_tanh.empty:
    _make_network_fig(
        df_tanh[df_tanh["dynamics"] == "voltage"].copy(),
        network_title=(
            "Voltage-Tanh RNN  —  uniform vs power-law τ init  "
            "(θ = fixedA,  6-bit flip-flop)"
        ),
        fname="fig3_init_speed_voltage_tanh_ff.pdf",
    )


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 7 — Fig 4: Linear RNN  (sine wave)
# ══════════════════════════════════════════════════════════════════════════════

if not df_lin_sine.empty:
    _make_network_fig(
        df_lin_sine,
        network_title=(
            "Linear RNN  —  uniform vs power-law τ init  "
            "(fixed A  ·  trainable A,  sine-wave generation)"
        ),
        fname="fig4_init_speed_linear_sine.pdf",
    )

    _make_network_fig(
        df_lin_sine[df_lin_sine["theta"] == "full"].copy(),
        network_title=(
            "Linear RNN  —  uniform vs power-law τ init  "
            "(trainable A only,  sine-wave generation)"
        ),
        fname="fig4b_init_speed_linear_sine_trainableA.pdf",
    )
else:
    print("⚠ No linear sine-wave records found.")
