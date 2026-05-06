# %% [markdown]
# # Schur-Init Grid — Loss & Accuracy Curves
#
# Loads the two halves of the schur-init grid:
#   * `schur_init_grid_g05` (Server A — gain = 0.5)
#   * `schur_init_grid_g09` (Server B — gain = 0.9)
#
# Per task (flip-flop, sine-wave), produces:
#   * Validation loss curves through training
#   * Validation accuracy / R² curves through training
#
# Each panel is split by gain (g = 0.5 vs g = 0.9) and overlays the
# 6 (init × θ-set) conditions as mean ± std across 5 seeds.
#
# ── NOTEBOOK STRUCTURE ───────────────────────────────────────────────────────
#   PART 0 : Imports & sweep_dir configuration  ← SET PATHS HERE
#   PART 1 : Load records into one dataframe
#   PART 2 : Flip-flop loss & accuracy curves
#   PART 3 : Sine-wave loss & R² curves

# %% Imports
import os
import re
import sys
import json
import subprocess

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()
os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())

# Output SVG-as-paths (Adobe Illustrator-friendly)
plt.rcParams["svg.fonttype"] = "path"

# ══════════════════════════════════════════════════════════════════════════════
# %% QUICK START — set sweep paths before running
# ══════════════════════════════════════════════════════════════════════════════
SWEEP_DIRS = [
    # Server A (g = 0.5)
    "/home/facosta/timescales/timescales/logs/experiments/schur_init_grid_g05_20260505_061150",
    # Server B (g = 0.9)
    "/home/facosta/timescales/timescales/logs/experiments/schur_init_grid_g09_20260504_230709",
]

SAVE_FIGS = True
FIGS_DIR  = os.path.join("notebooks", "schur_init_grid", "figs", "loss_acc_curves")
os.makedirs(FIGS_DIR, exist_ok=True)



# Grid axes (must match the sweep configs)
GAINS  = [0.5, 0.9]
INITS  = ["uniform", "powerlawtau", "schurH"]
THETAS = ["full", "fixedA"]
TASKS  = ["ff", "sine"]

TASK_FULL_NAME = {"ff": "flip_flop", "sine": "sine_wave"}
TASK_TITLE     = {"ff": "Heterogeneous 6-bit Flip-Flop",
                  "sine": "Heterogeneous Sine-Wave (3 pairs, 6 channels)"}
ACC_LABEL      = {"ff": "Validation accuracy",
                  "sine": r"Validation $R^2$"}
ACC_FLOOR      = {"ff": 0.5, "sine": 0.0}    # baseline reference line

# Condition styling (init -> colour, theta -> linestyle)
INIT_COLORS = {
    "uniform":     "#2a5783",     # navy
    "powerlawtau": "#e76f51",     # orange
    "schurH":      "#2a9d8f",     # teal
}
THETA_LS = {
    "full":   "-",
    "fixedA": "--",
}
INIT_LABEL = {
    "uniform":     "uniform (τ=1)",
    "powerlawtau": r"powerlaw τ ($\beta_\tau{=}1$)",
    "schurH":      r"Schur-$H$ ($\beta_H{=}1$)",
}
THETA_LABEL = {
    "full":   r"$\Theta_{\rm full}$ (τ learnable)",
    "fixedA": r"$\Theta_{\rm fixedA}$ (τ frozen)",
}

# Regex for the new experiment naming convention used by schur_init_grid_g{05,09}
# e.g. "g0.5_uniform_full_ff", "g0.9_powerlawtau_fixedA_sine"
_EXP_RE = re.compile(
    r"^g(?P<gain>\d+(?:\.\d+)?)_"
    r"(?P<init>uniform|powerlawtau|schurH)_"
    r"(?P<theta>full|fixedA)_"
    r"(?P<task>ff|sine)$"
)

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 1 — Load records
# ══════════════════════════════════════════════════════════════════════════════
# Canonical record schema (used to seed an empty DataFrame so column access
# never KeyErrors even when no runs are loaded).
_RECORD_COLS = [
    "sweep_dir", "exp_name",
    "gain", "init", "theta", "task",
    "seed", "seed_path",
    "run_config",
    "steps",
    "val_losses", "val_accs",
    "train_losses", "train_accs",
    "final_val_loss", "final_val_acc",
    # Best-over-trajectory stats (filled in PART 1):
    "min_val_loss", "min_val_loss_step",
    "max_val_acc",  "max_val_acc_step",
]

records = []
load_diagnostics = {
    "sweep_dirs_missing": [],
    "exps_skipped":       [],
    "seeds_no_config":    [],
    "seeds_no_curves":    [],
}

for sweep_dir in SWEEP_DIRS:
    if not os.path.isdir(sweep_dir):
        print(f"⚠ Sweep directory not found, skipping: {sweep_dir}")
        load_diagnostics["sweep_dirs_missing"].append(sweep_dir)
        continue
    print(f"\n── Loading sweep: {sweep_dir}")
    for exp_name in sorted(os.listdir(sweep_dir)):
        exp_dir = os.path.join(sweep_dir, exp_name)
        if not os.path.isdir(exp_dir) or exp_name in ("configs",):
            continue

        m = _EXP_RE.match(exp_name)
        if m is None:
            print(f"  Skipping unrecognised experiment: {exp_name}")
            load_diagnostics["exps_skipped"].append((sweep_dir, exp_name))
            continue
        gain  = float(m.group("gain"))
        init  = m.group("init")
        theta = m.group("theta")
        task  = m.group("task")

        for sdn in sorted(os.listdir(exp_dir)):
            if not sdn.startswith("seed_"):
                continue
            seed = int(sdn.split("_")[1])
            seed_path = os.path.join(exp_dir, sdn)

            cfg_file = os.path.join(seed_path, "run_config.yaml")
            if not os.path.exists(cfg_file):
                load_diagnostics["seeds_no_config"].append(seed_path)
                continue
            with open(cfg_file) as f:
                run_config = yaml.safe_load(f)

            # Final val loss (job_result has the canonical value; fall back to last logged)
            fvl = None
            jr = os.path.join(seed_path, "job_result.yaml")
            if os.path.exists(jr):
                with open(jr) as f:
                    fvl = yaml.safe_load(f).get("final_val_loss")

            # Training curves
            steps, val_losses, val_accs = [], [], []
            train_losses, train_accs = [], []
            lf = os.path.join(seed_path, "training_losses.json")
            if os.path.exists(lf):
                with open(lf) as f:
                    ld = json.load(f)
                steps        = ld.get("steps", [])
                val_losses   = ld.get("val_losses", [])
                val_accs     = ld.get("val_accuracies", [])
                train_losses = ld.get("train_losses", [])
                train_accs   = ld.get("train_accuracies", [])

            if fvl is None and val_losses:
                fvl = val_losses[-1]
            if fvl is None:
                print(f"  No loss data for {exp_name}/seed_{seed} — skipping")
                load_diagnostics["seeds_no_curves"].append(seed_path)
                continue

            # Best-over-trajectory stats (None if curve missing). Steps array
            # may be shorter than the metric series if logging mis-aligned, so
            # we clip to the common prefix.
            min_vl, min_vl_step = None, None
            if val_losses:
                _vl = np.asarray(val_losses, dtype=float)
                _st = np.asarray(steps[:len(_vl)] if steps else
                                 list(range(len(_vl))), dtype=float)
                _i = int(np.argmin(_vl))
                min_vl = float(_vl[_i])
                min_vl_step = int(_st[_i]) if _i < len(_st) else None

            max_va, max_va_step = None, None
            if val_accs:
                _va = np.asarray(val_accs, dtype=float)
                _st = np.asarray(steps[:len(_va)] if steps else
                                 list(range(len(_va))), dtype=float)
                _i = int(np.argmax(_va))
                max_va = float(_va[_i])
                max_va_step = int(_st[_i]) if _i < len(_st) else None

            records.append(dict(
                sweep_dir=sweep_dir,
                exp_name=exp_name,
                gain=gain, init=init, theta=theta, task=task,
                seed=seed, seed_path=seed_path,
                run_config=run_config,
                steps=steps,
                val_losses=val_losses,   val_accs=val_accs,
                train_losses=train_losses, train_accs=train_accs,
                final_val_loss=fvl,
                final_val_acc=(val_accs[-1] if val_accs else None),
                min_val_loss=min_vl,     min_val_loss_step=min_vl_step,
                max_val_acc=max_va,      max_val_acc_step=max_va_step,
            ))

df = (pd.DataFrame(records, columns=_RECORD_COLS)
      if records else pd.DataFrame({c: [] for c in _RECORD_COLS}))

print(f"\nLoaded {len(df)} runs")
if df.empty:
    print("\n" + "=" * 70)
    print("NO DATA LOADED — diagnostics:")
    print("=" * 70)
    if load_diagnostics["sweep_dirs_missing"]:
        print(f"  Missing sweep dirs ({len(load_diagnostics['sweep_dirs_missing'])}):")
        for d in load_diagnostics["sweep_dirs_missing"]:
            print(f"    {d}")
    if load_diagnostics["exps_skipped"]:
        print(f"  Experiments skipped (regex mismatch) "
              f"({len(load_diagnostics['exps_skipped'])}):")
        for sd, en in load_diagnostics["exps_skipped"][:10]:
            print(f"    {sd}/{en}")
    if load_diagnostics["seeds_no_config"]:
        print(f"  Seeds with no run_config.yaml: "
              f"{len(load_diagnostics['seeds_no_config'])}")
    if load_diagnostics["seeds_no_curves"]:
        print(f"  Seeds with no training_losses.json: "
              f"{len(load_diagnostics['seeds_no_curves'])}")
    print("\n  Hint: edit SWEEP_DIRS at the top of this notebook with the "
          "real (timestamped) sweep directories you downloaded.")
    print("  Then re-run PART 1 before re-running the plot cells.")
    print("=" * 70)
else:
    summary = (
        df.groupby(["task", "gain", "init", "theta"])
          .agg(n_seeds=("seed", "nunique"),
               final_val_loss_mean=("final_val_loss", "mean"),
               final_val_acc_mean=("final_val_acc", "mean"),
               min_val_loss_mean=("min_val_loss", "mean"),
               max_val_acc_mean=("max_val_acc", "mean"))
          .round(4)
    )
    print(summary.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 1.5 — Best-seed lookup table (saved to disk)
# ══════════════════════════════════════════════════════════════════════════════
# For each (task, gain, init, theta) condition, identify:
#   - best_seed_min_loss : seed achieving the lowest validation loss EVER seen
#                          during training (across all val checkpoints).
#   - best_seed_max_acc  : seed achieving the highest validation accuracy / R²
#                          EVER seen during training.
#
# Saved to:   notebooks/schur_init_grid/best_seeds.csv
# Reusable downstream via `_lookup_best_seed_row(task, gain, init, theta, metric)`.

BEST_SEEDS_CSV = os.path.join("notebooks", "schur_init_grid", "best_seeds.csv")
SUPPORTED_SELECTION_METRICS = ("min_val_loss", "max_val_acc")


def _build_best_seeds_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (task, gain, init, theta) with the best-seed per metric."""
    if df.empty or "task" not in df.columns:
        return pd.DataFrame(columns=[
            "task", "gain", "init", "theta", "n_seeds",
            "best_seed_min_loss", "best_min_val_loss", "best_min_val_loss_step",
            "best_seed_max_acc",  "best_max_val_acc",  "best_max_val_acc_step",
        ])

    out_rows = []
    grouped = df.groupby(["task", "gain", "init", "theta"], sort=True)
    for (task, gain, init, theta), grp in grouped:
        row = dict(task=task, gain=gain, init=init, theta=theta,
                   n_seeds=int(grp["seed"].nunique()))
        # min loss
        sub_loss = grp.dropna(subset=["min_val_loss"])
        if not sub_loss.empty:
            best_loss_idx = sub_loss["min_val_loss"].idxmin()
            row["best_seed_min_loss"]      = int(sub_loss.loc[best_loss_idx, "seed"])
            row["best_min_val_loss"]       = float(sub_loss.loc[best_loss_idx, "min_val_loss"])
            row["best_min_val_loss_step"]  = sub_loss.loc[best_loss_idx, "min_val_loss_step"]
        else:
            row["best_seed_min_loss"]     = None
            row["best_min_val_loss"]      = None
            row["best_min_val_loss_step"] = None
        # max acc
        sub_acc = grp.dropna(subset=["max_val_acc"])
        if not sub_acc.empty:
            best_acc_idx = sub_acc["max_val_acc"].idxmax()
            row["best_seed_max_acc"]      = int(sub_acc.loc[best_acc_idx, "seed"])
            row["best_max_val_acc"]       = float(sub_acc.loc[best_acc_idx, "max_val_acc"])
            row["best_max_val_acc_step"]  = sub_acc.loc[best_acc_idx, "max_val_acc_step"]
        else:
            row["best_seed_max_acc"]     = None
            row["best_max_val_acc"]      = None
            row["best_max_val_acc_step"] = None
        out_rows.append(row)
    return pd.DataFrame(out_rows)


best_seeds_df = _build_best_seeds_table(df)

if not best_seeds_df.empty:
    os.makedirs(os.path.dirname(BEST_SEEDS_CSV), exist_ok=True)
    best_seeds_df.to_csv(BEST_SEEDS_CSV, index=False)
    print(f"\nBest-seed table saved -> {BEST_SEEDS_CSV}")
    # Pretty-print
    _disp = best_seeds_df.copy()
    for c in ("best_min_val_loss", "best_max_val_acc"):
        if c in _disp.columns:
            _disp[c] = _disp[c].astype(float).round(4)
    print(_disp.to_string(index=False))
else:
    print("\n(best_seeds_df is empty — load runs first.)")


def _lookup_best_seed_row(task: str, gain: float, init: str, theta: str,
                          metric: str) -> "pd.Series | None":
    """Return the row in `df` for the best seed under `metric` ∈ SUPPORTED_SELECTION_METRICS.

    Returns None if no record is available for that condition.
    """
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"metric must be one of {SUPPORTED_SELECTION_METRICS}, got {metric!r}")
    if best_seeds_df.empty:
        return None
    sel = best_seeds_df[(best_seeds_df["task"] == task) &
                        (best_seeds_df["gain"] == gain) &
                        (best_seeds_df["init"] == init) &
                        (best_seeds_df["theta"] == theta)]
    if sel.empty:
        return None
    seed_col = "best_seed_min_loss" if metric == "min_val_loss" else "best_seed_max_acc"
    seed_val = sel.iloc[0][seed_col]
    if seed_val is None or (isinstance(seed_val, float) and np.isnan(seed_val)):
        return None
    seed_int = int(seed_val)
    rows = df[(df["task"] == task) & (df["gain"] == gain) &
              (df["init"] == init) & (df["theta"] == theta) &
              (df["seed"] == seed_int)]
    if rows.empty:
        return None
    return rows.iloc[0]


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 2 — helpers (mean ± std across seeds on a common step grid)
# ══════════════════════════════════════════════════════════════════════════════
def _stack_curves(rows: pd.DataFrame, value_key: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Align all seeds to the LONGEST common step prefix and stack.

    Returns (steps, mean, std, n_seeds). Empty arrays if nothing to stack.
    """
    series = []
    step_lists = []
    for _, r in rows.iterrows():
        v = r[value_key]
        s = r["steps"]
        if not v or not s:
            continue
        L = min(len(v), len(s))
        if L == 0:
            continue
        series.append(np.asarray(v[:L], dtype=float))
        step_lists.append(np.asarray(s[:L], dtype=float))

    if not series:
        return np.array([]), np.array([]), np.array([]), 0

    L_min = min(len(a) for a in series)
    arr = np.stack([a[:L_min] for a in series], axis=0)
    steps = step_lists[0][:L_min]
    return steps, arr.mean(axis=0), arr.std(axis=0), len(series)


def _select_best_seed(rows: pd.DataFrame, value_key: str, direction: str
                      ) -> "pd.Series | None":
    """Pick the row whose seed has the best (min/max) value of `value_key`
    EVER ACHIEVED across the training trajectory (not just at the final step).

    Returns None if no row in `rows` has a usable curve.
    """
    candidates = []
    for idx, r in rows.iterrows():
        v = r[value_key]
        if v and len(v) > 0:
            arr = np.asarray(v, dtype=float)
            if direction == "min":
                candidates.append((idx, float(arr.min())))
            elif direction == "max":
                candidates.append((idx, float(arr.max())))
            else:
                raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")
    if not candidates:
        return None
    best_idx = (min if direction == "min" else max)(candidates, key=lambda x: x[1])[0]
    return rows.loc[best_idx]


def _plot_curves_for_task(task: str, value_key: str,
                          ylabel: str, title: str,
                          *, log_y: bool = False,
                          y_floor_line: float | None = None,
                          ylim: tuple[float, float] | None = None,
                          fname: str = "",
                          mode: str = "mean_band",
                          best_direction: str = "min"):
    """Two columns (one per gain), 6 (init × theta) conditions per panel.

    Parameters
    ----------
    mode
        - "mean_band":  per-condition mean ± std across all seeds
        - "best_seed":  per-condition single curve from the seed whose FINAL
                        value of `value_key` is best (min or max).
    best_direction
        Used only when mode == "best_seed". "min" for losses, "max" for
        accuracy / R^2.
    """
    if df.empty or "task" not in df.columns:
        print(f"⚠ df is empty — fix SWEEP_DIRS and re-run PART 1 before plotting "
              f"(task={task!r}).")
        return

    if mode not in ("mean_band", "best_seed"):
        raise ValueError(f"mode must be 'mean_band' or 'best_seed', got {mode!r}")

    fig, axes = plt.subplots(1, len(GAINS), figsize=(6.5 * len(GAINS), 4.6),
                              sharey=True, squeeze=False)
    axes = axes[0]

    sub_task = df[df["task"] == task]
    if sub_task.empty:
        print(f"⚠ No runs found for task={task!r}; skipping plot.")
        plt.close(fig)
        return

    for ax, g in zip(axes, GAINS):
        for init in INITS:
            for theta in THETAS:
                rows = sub_task[(sub_task["gain"] == g) &
                                (sub_task["init"] == init) &
                                (sub_task["theta"] == theta)]
                if rows.empty:
                    continue
                color = INIT_COLORS[init]
                ls    = THETA_LS[theta]

                if mode == "mean_band":
                    steps, mu, sd, n = _stack_curves(rows, value_key)
                    if n == 0:
                        continue
                    label = f"{INIT_LABEL[init]} | {THETA_LABEL[theta]} (n={n})"
                    ax.plot(steps, mu, color=color, ls=ls, lw=2.0, label=label,
                            zorder=3)
                    ax.fill_between(steps, mu - sd, mu + sd,
                                    color=color, alpha=0.18, lw=0, zorder=2)
                else:  # mode == "best_seed"
                    best = _select_best_seed(rows, value_key, best_direction)
                    if best is None:
                        continue
                    v = np.asarray(best[value_key], dtype=float)
                    s = np.asarray(best["steps"], dtype=float)
                    L = min(len(v), len(s))
                    label = (f"{INIT_LABEL[init]} | {THETA_LABEL[theta]} "
                             f"(seed {int(best['seed'])})")
                    ax.plot(s[:L], v[:L], color=color, ls=ls, lw=2.0,
                            label=label, zorder=3)

        ax.set_title(f"g = {g}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training step", fontsize=11)
        if log_y:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        if y_floor_line is not None:
            ax.axhline(y_floor_line, color="black", ls=":", lw=0.8, alpha=0.4)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(ylabel, fontsize=11)

    # Single shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=8, ncol=2,
                   loc="lower center", bbox_to_anchor=(0.5, -0.18),
                   frameon=False)

    if mode == "best_seed":
        n_seeds_max = int(df[df["task"] == task]
                          .groupby(["gain", "init", "theta"])["seed"]
                          .nunique().max())
        crit = "min" if best_direction == "min" else "max"
        title = (title + f"  (best of {n_seeds_max} seeds, "
                 f"by {crit} {value_key} ever achieved)")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if SAVE_FIGS and fname:
        out = os.path.join(FIGS_DIR, fname)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"  saved -> {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 3 — Flip-flop loss curves
# ══════════════════════════════════════════════════════════════════════════════
_plot_curves_for_task(
    task="ff", value_key="val_losses",
    ylabel="Validation loss (BCE)",
    title=f"Validation Loss — {TASK_TITLE['ff']}",
    log_y=True,
    fname="ff_val_loss.pdf",
)

# %% Flip-flop accuracy curves
_plot_curves_for_task(
    task="ff", value_key="val_accs",
    ylabel=ACC_LABEL["ff"],
    title=f"Validation Accuracy — {TASK_TITLE['ff']}",
    log_y=False,
    y_floor_line=ACC_FLOOR["ff"],
    ylim=(0.45, 1.02),
    fname="ff_val_acc.pdf",
)

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 4 — Sine-wave loss curves
# ══════════════════════════════════════════════════════════════════════════════
_plot_curves_for_task(
    task="sine", value_key="val_losses",
    ylabel="Validation loss (MSE)",
    title=f"Validation Loss — {TASK_TITLE['sine']}",
    log_y=True,
    fname="sine_val_loss.pdf",
)

# %% Sine-wave R² curves
# Note: `val_accuracies` for sine_wave is the R² metric (see RNNLightning._compute_accuracy).
_plot_curves_for_task(
    task="sine", value_key="val_accs",
    ylabel=ACC_LABEL["sine"],
    title=f"Validation $R^2$ — {TASK_TITLE['sine']}",
    log_y=False,
    y_floor_line=ACC_FLOOR["sine"],
    ylim=(-0.5, 1.05),
    fname="sine_val_r2.pdf",
)

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 5 — Best-seed-only versions
# ══════════════════════════════════════════════════════════════════════════════
# Same panels as PARTS 3–4, but per condition we plot the curve from the SEED
# whose final value of the displayed metric is best:
#   - losses        -> seed with min final loss
#   - accuracy / R² -> seed with max final value
#
# This gives a "ceiling" view of each condition: the best of 5 seeds.
# Loss-best and accuracy-best may be different seeds for the same condition.

# %% Flip-flop loss — best seed per condition
_plot_curves_for_task(
    task="ff", value_key="val_losses",
    ylabel="Validation loss (BCE)",
    title=f"Validation Loss — {TASK_TITLE['ff']}",
    log_y=True,
    fname="ff_val_loss_bestseed.pdf",
    mode="best_seed", best_direction="min",
)

# %% Flip-flop accuracy — best seed per condition
_plot_curves_for_task(
    task="ff", value_key="val_accs",
    ylabel=ACC_LABEL["ff"],
    title=f"Validation Accuracy — {TASK_TITLE['ff']}",
    log_y=False,
    y_floor_line=ACC_FLOOR["ff"],
    ylim=(0.45, 1.02),
    fname="ff_val_acc_bestseed.pdf",
    mode="best_seed", best_direction="max",
)

# %% Sine-wave loss — best seed per condition
_plot_curves_for_task(
    task="sine", value_key="val_losses",
    ylabel="Validation loss (MSE)",
    title=f"Validation Loss — {TASK_TITLE['sine']}",
    log_y=True,
    fname="sine_val_loss_bestseed.pdf",
    mode="best_seed", best_direction="min",
)

# %% Sine-wave R² — best seed per condition
_plot_curves_for_task(
    task="sine", value_key="val_accs",
    ylabel=ACC_LABEL["sine"],
    title=f"Validation $R^2$ — {TASK_TITLE['sine']}",
    log_y=False,
    y_floor_line=ACC_FLOOR["sine"],
    ylim=(-0.5, 1.05),
    fname="sine_val_r2_bestseed.pdf",
    mode="best_seed", best_direction="max",
)

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 6 — Jacobian spectra (best seed per condition)
# ══════════════════════════════════════════════════════════════════════════════
# For each (gain × init × theta) condition, pick the seed with the minimum
# final validation loss and overlay the Jacobian eigenvalues at INIT (gray)
# and at the END of training (init-colour) on the complex plane.
#
# Files produced (one per task): jacobian_spectra_{task}_bestseed.pdf
# Source data: `spectral_init.pt` / `spectral_final.pt` saved by
# `SpectralSnapshotCallback` in each seed dir (key: "eigvals_eig").

import torch  # local import; the global numpy/matplotlib imports are at PART 0


def _load_spectra(seed_path: str) -> "tuple[dict, dict] | None":
    """Load (init_blob, final_blob) for a seed; return None if either is missing."""
    init_p  = os.path.join(seed_path, "spectral_init.pt")
    final_p = os.path.join(seed_path, "spectral_final.pt")
    if not (os.path.exists(init_p) and os.path.exists(final_p)):
        return None
    init  = torch.load(init_p,  map_location="cpu", weights_only=False)
    final = torch.load(final_p, map_location="cpu", weights_only=False)
    return init, final


_METRIC_LABEL = {
    "min_val_loss": "min val_loss ever achieved",
    "max_val_acc":  "max val_acc/R² ever achieved",
}

# Which metric drives best-seed selection for the SPECTRUM panels (PART 6).
# One of: "min_val_loss"  (lowest validation loss ever achieved)
#         "max_val_acc"   (highest validation accuracy / R² ever achieved)
SPECTRA_SELECTION_METRIC = "max_val_acc"


def _plot_jacobian_spectra_for_task(task: str, *,
                                    fname: str = "",
                                    x_axis_lim: tuple[float, float] = (0, 1.1),
                                    y_axis_lim: tuple[float, float] = (-0.5, 0.5),
                                    point_size: float = 8.0,
                                    selection_metric: str = SPECTRA_SELECTION_METRIC):
    """4 rows × 3 cols figure of Jacobian spectra per (gain × theta) × init.

    `selection_metric` (one of SUPPORTED_SELECTION_METRICS) determines which
    seed is "best" per condition. Looked up via `_lookup_best_seed_row`.
    """
    if df.empty or "task" not in df.columns:
        print(f"⚠ df is empty — fix SWEEP_DIRS and re-run PART 1 before plotting.")
        return

    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {SUPPORTED_SELECTION_METRICS}, "
            f"got {selection_metric!r}")

    sub_task = df[df["task"] == task]
    if sub_task.empty:
        print(f"⚠ No runs found for task={task!r}; skipping spectra plot.")
        return

    row_keys = [(g, th) for g in GAINS for th in THETAS]   # 4 rows
    n_rows = len(row_keys)
    n_cols = len(INITS)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.1 * n_cols, 3.1 * n_rows),
                             squeeze=False)

    # Unit circle (precomputed for reuse)
    _theta = np.linspace(0, 2 * np.pi, 256)
    uc_x, uc_y = np.cos(_theta), np.sin(_theta)

    n_loaded, n_missing = 0, 0
    for r, (g, theta) in enumerate(row_keys):
        for c, init in enumerate(INITS):
            ax = axes[r, c]
            best = _lookup_best_seed_row(task, g, init, theta, selection_metric)
            if best is None:
                ax.set_title("(no runs)", fontsize=9, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_aspect("equal")
                ax.set_xlim(*x_axis_lim); ax.set_ylim(*y_axis_lim)
                continue

            spectra = _load_spectra(best["seed_path"])
            if spectra is None:
                ax.set_title(f"seed {int(best['seed'])} — no spectral snapshots",
                             fontsize=8, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_aspect("equal")
                ax.set_xlim(*x_axis_lim); ax.set_ylim(*y_axis_lim)
                n_missing += 1
                continue

            init_blob, final_blob = spectra
            eig_init  = np.asarray(init_blob["eigvals_eig"])
            eig_final = np.asarray(final_blob["eigvals_eig"])
            n_loaded += 1

            color = INIT_COLORS[init]

            # Unit circle + axes
            ax.plot(uc_x, uc_y, color="black", ls="--", lw=0.7, alpha=0.45,
                    zorder=1)
            ax.axhline(0, color="black", lw=0.4, alpha=0.3, zorder=1)
            ax.axvline(0, color="black", lw=0.4, alpha=0.3, zorder=1)

            # Eigenvalues
            ax.scatter(eig_init.real, eig_init.imag,
                       s=point_size, c="lightgray",
                       edgecolors="dimgray", linewidths=0.25,
                       alpha=0.8, label="init", zorder=2)
            ax.scatter(eig_final.real, eig_final.imag,
                       s=point_size, c=color,
                       edgecolors="white", linewidths=0.25,
                       alpha=0.85, label="final", zorder=3)

            ax.set_aspect("equal")
            ax.set_xlim(*x_axis_lim)
            ax.set_ylim(*y_axis_lim)
            ax.tick_params(labelsize=7)

            ax.set_title(f"seed {int(best['seed'])}", fontsize=8, loc="right",
                         color="gray")

            # Row labels (left edge)
            if c == 0:
                ax.set_ylabel(f"g={g}, {THETA_LABEL[theta]}\nIm(λ)", fontsize=9)
            else:
                ax.set_ylabel("")
            # Col headers (top edge)
            if r == 0:
                ax.text(0.5, 1.18, INIT_LABEL[init],
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
            # Bottom row x-label
            if r == n_rows - 1:
                ax.set_xlabel("Re(λ)", fontsize=9)

    # Shared legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="lightgray", markeredgecolor="dimgray",
                   markersize=7, label="init eigenvalues"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#444444", markeredgecolor="white",
                   markersize=7,
                   label="final eigenvalues (colour = init scheme)"),
        plt.Line2D([0], [0], color="black", ls="--", lw=0.7,
                   label="unit circle"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9, frameon=False)

    fig.suptitle(
        f"Jacobian spectrum before vs after training — {TASK_TITLE[task]}\n"
        f"(best seed per condition by {_METRIC_LABEL[selection_metric]})",
        fontsize=12, fontweight="bold", y=1.005)
    plt.tight_layout()

    print(f"  spectra loaded: {n_loaded}, missing snapshots: {n_missing}")

    if SAVE_FIGS and fname:
        out = os.path.join(FIGS_DIR, fname)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"  saved -> {out}")
    plt.show()


# %% Flip-flop spectra
_plot_jacobian_spectra_for_task(
    task="ff",
    x_axis_lim=(0.6, 1.05),
    y_axis_lim=(-0.2, 0.2),
    selection_metric=SPECTRA_SELECTION_METRIC,
    fname=f"jacobian_spectra_ff_by_{SPECTRA_SELECTION_METRIC}.pdf",
)

# %% Sine-wave spectra
_plot_jacobian_spectra_for_task(
    task="sine",
    selection_metric=SPECTRA_SELECTION_METRIC,
    fname=f"jacobian_spectra_sine_by_{SPECTRA_SELECTION_METRIC}.pdf",
)

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 7 — Example rollouts (best seed per condition)
# ══════════════════════════════════════════════════════════════════════════════
# Same 4-row × 3-col grid as PART 6 (gain × θ along rows, init scheme along
# cols).  For each panel the best-seed model (chosen by `selection_metric`) is
# loaded from its Lightning checkpoint and run forward.
#
#  Sine-wave:  autonomous rollout for `ROLLOUT_FACTOR × T_train` steps with
#              null input (zeros) and h₀ = init_hidden_value.  Each output
#              channel plotted in a distinct hue; a vertical dashed line marks
#              T_train.  Analytical targets overlaid as thin dashed curves.
#
#  Flip-flop:  one randomly-drawn validation trajectory of length T_train.
#              σ(output) overlaid on the {0,1} target per bit;  a horizontal
#              dash at 0.5 marks the decision boundary.
#
# Files produced (one per task):
#   rollouts_{task}_by_{metric}.pdf

import glob as _glob

# ── Config ───────────────────────────────────────────────────────────────────
ROLLOUT_FACTOR = 3      # sine-wave: rollout this many × T_train steps

# Channel colour palette (6 channels max)
_CH_PALETTE = ["#e41a1c", "#377eb8", "#4daf4a",
               "#984ea3", "#ff7f00", "#a65628"]


# ── Model loader ─────────────────────────────────────────────────────────────
def _load_trained_model(seed_path: str, run_config: dict):
    """Reconstruct RNN from run_config and load the best-val-loss checkpoint.

    Returns the (eval-mode) RNN model, or None if no checkpoint is found.
    """
    ckpts = _glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not ckpts:
        return None
    ckpt = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    rc = run_config

    task_key = rc.get("task", "flip_flop")
    n_bits   = rc.get("n_bits", 6)
    n_pairs  = rc.get("n_pairs", 3)
    if task_key == "sine_wave":
        in_sz  = rc.get("input_size", 1)
        out_sz = 2 * n_pairs
    else:
        in_sz  = n_bits
        out_sz = n_bits

    import torch.nn as _torch_nn
    from rnns.rnn import RNN, RNNLightning
    # Use "normal_scaled" regardless of the original wrec_init scheme: the
    # checkpoint immediately overwrites all weights, so running the full Schur
    # surgery during __init__ would be both wrong and expensive.
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
        wrec_init="normal_scaled",          # overridden: weights come from ckpt
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
    lit.load_state_dict(ckpt["state_dict"])
    lit.eval()
    return lit.model


# ── Rollout helpers ───────────────────────────────────────────────────────────
def _rollout_sine(model, run_config: dict, rollout_factor: int = ROLLOUT_FACTOR
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Autonomous sine-wave rollout.

    Returns (t_axis, outputs[T_long, out_sz], targets[T_long, out_sz], T_train).
    """
    rc       = run_config
    T_train  = rc["num_time_steps"]
    T_long   = T_train * rollout_factor
    dt_val   = rc["dt"]
    n_pairs  = rc.get("n_pairs", 3)
    periods  = rc.get("periods", [20.0, 50.0, 100.0])
    h0_val   = rc.get("init_hidden_value", 1.0)

    h0 = torch.full((1, rc["hidden_size"]), float(h0_val))
    inp = torch.zeros(1, T_long, rc.get("input_size", 1))
    with torch.no_grad():
        _, out = model(inp, init_hidden=h0)
    outputs = out[0].numpy()   # (T_long, out_sz)

    t_ax = np.arange(T_long) * dt_val
    targets = np.zeros((T_long, 2 * n_pairs))
    for k, period in enumerate(periods):
        phase = 2.0 * np.pi * t_ax / period
        targets[:, 2 * k]     = np.cos(phase)
        targets[:, 2 * k + 1] = np.sin(phase)

    return t_ax, outputs, targets, T_train


def _rollout_ff(model, run_config: dict
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one flip-flop trajectory and run the model on it.

    Returns (t_axis, sigmoid_outputs[T, n_bits], targets[T, n_bits]).
    """
    from datamodules.flip_flop import FlipFlopDataModule
    rc = run_config
    dm = FlipFlopDataModule(
        n_bits=rc["n_bits"],
        p_pulse=rc["p_pulse"],
        pulse_amplitude=rc.get("pulse_amplitude", 1.0),
        num_time_steps=rc["num_time_steps"],
        num_val_trajectories=1,
        batch_size=1,
    )
    dm.setup()
    inp_raw, _, tgt = dm.val_dataset.tensors   # (1, T, n_bits)
    with torch.no_grad():
        _, out = model(inp_raw)
    sig_out = torch.sigmoid(out)[0].numpy()
    tgt_np  = tgt[0].numpy()
    t_ax    = np.arange(rc["num_time_steps"]) * rc["dt"]
    return t_ax, sig_out, tgt_np


# ── Main plot function ────────────────────────────────────────────────────────
def _plot_rollouts_for_task(task: str, *,
                            selection_metric: str = SPECTRA_SELECTION_METRIC,
                            rollout_factor: int = ROLLOUT_FACTOR,
                            fname: str = ""):
    """4 rows × 3 cols figure of example model outputs for the best seed per
    (gain × theta × init) condition.

    Parameters
    ----------
    task
        "ff" or "sine".
    selection_metric
        One of SUPPORTED_SELECTION_METRICS — controls which seed is selected.
    rollout_factor
        (Sine only) rollout length as a multiple of T_train.
    """
    if df.empty or "task" not in df.columns:
        print(f"⚠ df is empty — fix SWEEP_DIRS and re-run PART 1.")
        return

    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {SUPPORTED_SELECTION_METRICS}, "
            f"got {selection_metric!r}")

    sub_task = df[df["task"] == task]
    if sub_task.empty:
        print(f"⚠ No runs found for task={task!r}.")
        return

    row_keys = [(g, th) for g in GAINS for th in THETAS]
    n_rows   = len(row_keys)
    n_cols   = len(INITS)

    # Panel height: enough for 6 stacked channels (or 6 bits)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    n_ok, n_miss = 0, 0
    for r, (g, theta) in enumerate(row_keys):
        for c, init in enumerate(INITS):
            ax  = axes[r, c]
            best = _lookup_best_seed_row(task, g, init, theta, selection_metric)

            def _placeholder(msg):
                ax.text(0.5, 0.5, msg, ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_xticks([]); ax.set_yticks([])

            if best is None:
                _placeholder("(no runs)")
                _label_axes(ax, r, c, g, theta, init, n_rows, n_cols, task)
                continue

            model = _load_trained_model(best["seed_path"], best["run_config"])
            if model is None:
                _placeholder(f"seed {int(best['seed'])}\nno checkpoint")
                n_miss += 1
                _label_axes(ax, r, c, g, theta, init, n_rows, n_cols, task)
                continue

            try:
                if task == "sine":
                    t_ax, outputs, targets, T_train = _rollout_sine(
                        model, best["run_config"], rollout_factor)
                    n_ch = outputs.shape[1]
                    for ch in range(n_ch):
                        col = _CH_PALETTE[ch % len(_CH_PALETTE)]
                        ax.plot(t_ax, targets[:, ch], color=col, lw=0.8,
                                ls="--", alpha=0.45, zorder=2)
                        ax.plot(t_ax, outputs[:, ch], color=col, lw=1.4,
                                alpha=0.9, zorder=3)
                    t_split = T_train * best["run_config"]["dt"]
                    ax.axvline(t_split, color="black", ls=":", lw=1.0,
                               alpha=0.7, zorder=4, label="T_train")
                    ax.set_ylim(-1.6, 1.6)
                    ax.set_xlabel("Time (steps × dt)", fontsize=8)

                else:  # ff
                    t_ax, sig_out, tgt_np = _rollout_ff(model, best["run_config"])
                    n_bits = sig_out.shape[1]
                    dt_val = best["run_config"]["dt"]
                    # Offset each bit by 2 so they stack vertically without overlap
                    for b in range(n_bits):
                        col    = _CH_PALETTE[b % len(_CH_PALETTE)]
                        offset = (n_bits - 1 - b) * 2.2
                        ax.step(t_ax, tgt_np[:, b] + offset, color=col,
                                lw=0.8, ls="--", alpha=0.5, where="post",
                                zorder=2)
                        ax.plot(t_ax, sig_out[:, b] + offset, color=col,
                                lw=1.3, alpha=0.9, zorder=3)
                        ax.axhline(0.5 + offset, color=col, lw=0.4,
                                   ls=":", alpha=0.3, zorder=1)
                    ax.set_yticks([])
                    ax.set_xlabel("Time (steps × dt)", fontsize=8)

                ax.set_title(f"seed {int(best['seed'])}", fontsize=8,
                             loc="right", color="gray")
                ax.grid(True, axis="x", alpha=0.2)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                n_ok += 1

            except Exception as exc:
                _placeholder(f"seed {int(best['seed'])}\nERROR: {exc}")
                n_miss += 1

            _label_axes(ax, r, c, g, theta, init, n_rows, n_cols, task)

    # Sine: add a T_train legend entry in the bottom-left panel
    if task == "sine":
        axes[-1, 0].plot([], [], color="black", ls=":", lw=1.0, label="T_train")
        axes[-1, 0].legend(fontsize=7, loc="lower left", framealpha=0.6)

    fig.suptitle(
        f"Example rollouts — {TASK_TITLE[task]}\n"
        f"(best seed per condition by {_METRIC_LABEL[selection_metric]})",
        fontsize=12, fontweight="bold", y=1.005)
    plt.tight_layout()
    print(f"  rollouts rendered: {n_ok}, missing/errors: {n_miss}")

    if SAVE_FIGS and fname:
        out = os.path.join(FIGS_DIR, fname)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"  saved -> {out}")
    plt.show()


def _label_axes(ax, r, c, g, theta, init, n_rows, n_cols, task):
    """Apply row/col annotations consistently across all panels."""
    if c == 0:
        ax.set_ylabel(f"g={g}, {THETA_LABEL[theta]}", fontsize=9)
    if r == 0:
        ax.text(0.5, 1.12, INIT_LABEL[init],
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, fontweight="bold")


# %% Flip-flop rollouts
_plot_rollouts_for_task(
    task="ff",
    selection_metric=SPECTRA_SELECTION_METRIC,
    fname=f"rollouts_ff_by_{SPECTRA_SELECTION_METRIC}.pdf",
)

# %% Sine-wave rollouts
_plot_rollouts_for_task(
    task="sine",
    selection_metric=SPECTRA_SELECTION_METRIC,
    rollout_factor=ROLLOUT_FACTOR,
    fname=f"rollouts_sine_by_{SPECTRA_SELECTION_METRIC}.pdf",
)
