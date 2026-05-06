# %% [markdown]
# # Tanh Dynamics Grid — Loss & Accuracy Curves
#
# Loads the tanh_dyn_grid sweep (12 conditions, 1 seed each):
#   * 2 gains   × 2 dynamics (rate / voltage) × 3 inits
#   * Task: 6-bit heterogeneous Flip-Flop, Θ_fixedA
#   * Activation: Tanh
#
# Produces:
#   * Validation loss curves through training     (per-condition overlay)
#   * Validation accuracy curves through training (per-condition overlay)
#   * Per-bit validation loss & accuracy breakdowns
#
# ── NOTEBOOK STRUCTURE ────────────────────────────────────────────────────────
#   PART 0 : Imports & sweep-dir configuration  ← SET PATH HERE
#   PART 1 : Load records into one DataFrame
#   PART 2 : Aggregate loss & accuracy curves
#   PART 3 : Per-bit breakdown curves

# %% Imports
import glob as _glob
import json
import os
import re
import subprocess
import sys

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

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 0 — Configuration  ← SET SWEEP_DIR HERE
# ══════════════════════════════════════════════════════════════════════════════

SWEEP_DIR = (
    "/home/facosta/timescales/timescales/logs/experiments"
    "/tanh_dyn_grid_20260505_230622"
)

SAVE_FIGS = True
FIGS_DIR  = os.path.join("notebooks", "tanh_dyn_grid", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Grid axes ─────────────────────────────────────────────────────────────────
GAINS    = [0.5, 0.9]
DYNAMICS = ["rate", "voltage"]
INITS    = ["uniform", "powerlawtau", "schurH"]
N_BITS   = 6

# ── Styling ───────────────────────────────────────────────────────────────────
# Init → colour (matched to notebook 1 for visual continuity)
INIT_COLORS = {
    "uniform":     "#2a5783",   # navy
    "powerlawtau": "#e76f51",   # orange
    "schurH":      "#2a9d8f",   # teal
}
INIT_LABELS = {
    "uniform":     r"Uniform (τ=1)",
    "powerlawtau": r"Power-law τ  (β=1)",
    "schurH":      r"Schur-H  (β=1)",
}

# Dynamics → linestyle
DYN_LS = {
    "rate":    "-",
    "voltage": "--",
}
DYN_LABELS = {
    "rate":    "Rate  (τ dr/dt = −r + tanh(·))",
    "voltage": "Voltage  (τ dx/dt = −x + tanh(·))",
}

# Per-bit colours (used in breakdown plots)
BIT_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a",
    "#984ea3", "#ff7f00", "#a65628",
]

# ── Experiment-name regex ─────────────────────────────────────────────────────
_EXP_RE = re.compile(
    r"^g(?P<gain>\d+(?:\.\d+)?)_tanh_"
    r"(?P<dynamics>rate|voltage)_"
    r"(?P<init>uniform|powerlawtau|schurH)_"
    r"fixedA_ff$"
)


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 1 — Load records
# ══════════════════════════════════════════════════════════════════════════════

records = []

if not os.path.isdir(SWEEP_DIR):
    print(f"⚠ SWEEP_DIR not found: {SWEEP_DIR}")
else:
    for exp_name in sorted(os.listdir(SWEEP_DIR)):
        exp_dir = os.path.join(SWEEP_DIR, exp_name)
        if not os.path.isdir(exp_dir) or exp_name in ("configs",):
            continue

        m = _EXP_RE.match(exp_name)
        if m is None:
            continue

        gain     = float(m.group("gain"))
        dynamics = m.group("dynamics")
        init     = m.group("init")

        # Only one seed per experiment
        seed_dirs = sorted(
            d for d in os.listdir(exp_dir) if d.startswith("seed_")
        )
        for sdn in seed_dirs:
            seed      = int(sdn.split("_")[1])
            seed_path = os.path.join(exp_dir, sdn)

            cfg_file   = os.path.join(seed_path, "run_config.yaml")
            curve_file = os.path.join(seed_path, "training_losses.json")
            if not os.path.exists(cfg_file) or not os.path.exists(curve_file):
                print(f"  ⚠ missing files in {seed_path}")
                continue

            with open(cfg_file) as f:
                run_config = yaml.safe_load(f)
            with open(curve_file) as f:
                curves = json.load(f)

            steps      = np.array(curves["steps"])
            val_losses = np.array(curves["val_losses"])
            val_accs   = np.array(curves["val_accuracies"])
            # train arrays may be one step shorter
            train_losses = np.array(curves.get("train_losses", []))
            train_accs   = np.array(curves.get("train_accuracies", []))

            # Per-bit arrays (dict channel_0 .. channel_5)
            vlpb = curves.get("val_losses_per_bit", {})
            vapb = curves.get("val_accuracies_per_bit", {})
            per_bit_val_loss = np.array([vlpb[f"channel_{b}"] for b in range(N_BITS)
                                         if f"channel_{b}" in vlpb])   # (n_bits, T)
            per_bit_val_acc  = np.array([vapb[f"channel_{b}"] for b in range(N_BITS)
                                         if f"channel_{b}" in vapb])   # (n_bits, T)

            records.append(dict(
                exp_name=exp_name,
                gain=gain, dynamics=dynamics, init=init, seed=seed,
                seed_path=seed_path, run_config=run_config,
                steps=steps,
                val_losses=val_losses, val_accs=val_accs,
                train_losses=train_losses, train_accs=train_accs,
                per_bit_val_loss=per_bit_val_loss,
                per_bit_val_acc=per_bit_val_acc,
                min_val_loss=float(val_losses.min()),
                max_val_acc=float(val_accs.max()),
            ))

df = pd.DataFrame(records)
print(f"\nLoaded {len(df)} records from {len(df['exp_name'].unique())} experiments.")
if not df.empty:
    print(df[["gain","dynamics","init","min_val_loss","max_val_acc"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 2A — Validation loss curves (2×2 grid: gain × dynamics)
# ══════════════════════════════════════════════════════════════════════════════
# Rows = gain, Columns = dynamics type.  Colour = init scheme.

def _plot_loss_acc(
    metric: str,        # "val_loss" | "val_acc"
    fname:  str = "",
) -> None:
    """2×2 grid of loss or accuracy curves, one panel per (gain, dynamics)."""
    if df.empty:
        print("⚠ DataFrame is empty — check SWEEP_DIR.")
        return

    col_key  = "val_losses" if metric == "val_loss" else "val_accs"
    y_label  = "Validation loss" if metric == "val_loss" else "Validation accuracy"
    title    = "Validation loss" if metric == "val_loss" else "Validation accuracy"

    fig, axes = plt.subplots(
        len(GAINS), len(DYNAMICS),
        figsize=(6.0 * len(DYNAMICS), 4.0 * len(GAINS)),
        squeeze=False, sharey=False,
    )

    for r, g in enumerate(GAINS):
        for c, dyn in enumerate(DYNAMICS):
            ax = axes[r, c]
            sub = df[(df["gain"] == g) & (df["dynamics"] == dyn)]

            for init in INITS:
                rows = sub[sub["init"] == init]
                if rows.empty:
                    continue
                for _, row in rows.iterrows():
                    steps_v = row["steps"]
                    vals    = row[col_key]
                    # align length (train arrays can be 1 shorter than steps)
                    n = min(len(steps_v), len(vals))
                    ax.plot(
                        steps_v[:n], vals[:n],
                        color=INIT_COLORS[init],
                        ls="-", lw=1.8, alpha=0.9,
                        label=INIT_LABELS[init],
                    )

            ax.set_title(
                f"g = {g},  {DYN_LABELS[dyn]}",
                fontsize=10, fontweight="bold",
            )
            ax.set_xlabel("Training step", fontsize=10)
            if c == 0:
                ax.set_ylabel(y_label, fontsize=10)
            ax.grid(True, alpha=0.20)

            # De-duplicate legend entries
            _handles, _labels = ax.get_legend_handles_labels()
            _seen, _h, _l = set(), [], []
            for h, lbl in zip(_handles, _labels):
                if lbl not in _seen:
                    _seen.add(lbl); _h.append(h); _l.append(lbl)
            ax.legend(_h, _l, fontsize=8, loc="best", framealpha=0.85)

    fig.suptitle(
        f"Tanh Dynamics Grid — {title}\n"
        "(6-bit Het. Flip-Flop,  Θ_fixedA,  1 seed per condition)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if SAVE_FIGS and fname:
        p = os.path.join(FIGS_DIR, fname)
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"  saved → {p}")
    plt.show()


_plot_loss_acc("val_loss", fname="val_loss_curves.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 2B — Validation accuracy curves
# ══════════════════════════════════════════════════════════════════════════════

_plot_loss_acc("val_acc", fname="val_acc_curves.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# %% PART 2C — Train vs Val curves (loss only; useful for checking overfitting)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_train_val_loss(
    gain: float,
    dynamics: str,
    fname: str = "",
) -> None:
    """Train and val loss overlay for a single (gain, dynamics) panel.

    Solid = val, dashed = train.
    """
    sub = df[(df["gain"] == gain) & (df["dynamics"] == dynamics)]
    if sub.empty:
        print(f"No data for g={gain}, dynamics={dynamics}")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for init in INITS:
        rows = sub[sub["init"] == init]
        if rows.empty:
            continue
        for _, row in rows.iterrows():
            n_val   = min(len(row["steps"]), len(row["val_losses"]))
            n_train = min(len(row["steps"]), len(row["train_losses"]))
            color   = INIT_COLORS[init]
            ax.plot(row["steps"][:n_val],   row["val_losses"][:n_val],
                    color=color, ls="-",  lw=1.8, alpha=0.9,
                    label=f"{INIT_LABELS[init]} (val)")
            if n_train > 0:
                ax.plot(row["steps"][:n_train], row["train_losses"][:n_train],
                        color=color, ls="--", lw=1.2, alpha=0.60,
                        label=f"{INIT_LABELS[init]} (train)")

    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title(
        f"Train vs Val loss  —  g={gain},  {DYN_LABELS[dynamics]}",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.20)
    _h, _l = ax.get_legend_handles_labels()
    _seen, _h2, _l2 = set(), [], []
    for h, lbl in zip(_h, _l):
        if lbl not in _seen:
            _seen.add(lbl); _h2.append(h); _l2.append(lbl)
    ax.legend(_h2, _l2, fontsize=8, loc="upper right", framealpha=0.85)
    plt.tight_layout()
    if SAVE_FIGS and fname:
        p = os.path.join(FIGS_DIR, fname)
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"  saved → {p}")
    plt.show()


for _g in GAINS:
    for _dyn in DYNAMICS:
        _plot_train_val_loss(
            _g, _dyn,
            fname=f"train_val_loss_g{_g:g}_{_dyn}.pdf",
        )


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 3A — Per-bit validation loss  (one figure per gain × dynamics)
# ══════════════════════════════════════════════════════════════════════════════
# Three panels per figure (one per init), each showing 6 per-bit loss curves.

def _plot_perbit(
    metric: str,         # "loss" | "acc"
    gain: float,
    dynamics: str,
    fname: str = "",
) -> None:
    """Per-bit loss or accuracy curves for one (gain, dynamics) pair."""
    sub = df[(df["gain"] == gain) & (df["dynamics"] == dynamics)]
    if sub.empty:
        return

    col_key  = "per_bit_val_loss" if metric == "loss" else "per_bit_val_acc"
    y_label  = "Per-bit val loss"   if metric == "loss" else "Per-bit val accuracy"

    fig, axes = plt.subplots(
        1, len(INITS),
        figsize=(5.5 * len(INITS), 4.0),
        sharey=False, squeeze=False,
    )
    for c, init in enumerate(INITS):
        ax   = axes[0, c]
        rows = sub[sub["init"] == init]
        if rows.empty:
            ax.set_title(f"{INIT_LABELS[init]}\n(no data)", fontsize=10)
            continue

        for _, row in rows.iterrows():
            pb = row[col_key]   # (n_bits, T) array
            if pb.size == 0:
                continue
            steps_v = row["steps"]
            n = min(pb.shape[1], len(steps_v))
            for b in range(min(pb.shape[0], N_BITS)):
                ax.plot(
                    steps_v[:n], pb[b, :n],
                    color=BIT_COLORS[b % len(BIT_COLORS)],
                    lw=1.6, alpha=0.85,
                    label=f"bit {b}",
                )

        ax.set_title(
            f"{INIT_LABELS[init]}",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Training step", fontsize=10)
        if c == 0:
            ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, alpha=0.20)

        _h, _l = ax.get_legend_handles_labels()
        _seen, _h2, _l2 = set(), [], []
        for h, lbl in zip(_h, _l):
            if lbl not in _seen:
                _seen.add(lbl); _h2.append(h); _l2.append(lbl)
        ax.legend(_h2, _l2, fontsize=8, loc="best",
                  bbox_to_anchor=(1.02, 1.0), framealpha=0.85, borderaxespad=0)

    fig.suptitle(
        f"Per-bit {y_label}  —  g={gain},  {DYN_LABELS[dynamics]}\n"
        "(6-bit Het. Flip-Flop,  Θ_fixedA)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if SAVE_FIGS and fname:
        p = os.path.join(FIGS_DIR, fname)
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"  saved → {p}")
    plt.show()


for _g in GAINS:
    for _dyn in DYNAMICS:
        _plot_perbit("loss", _g, _dyn, fname=f"perbit_loss_g{_g:g}_{_dyn}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 3B — Per-bit validation accuracy
# ══════════════════════════════════════════════════════════════════════════════

for _g in GAINS:
    for _dyn in DYNAMICS:
        _plot_perbit("acc", _g, _dyn, fname=f"perbit_acc_g{_g:g}_{_dyn}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# %% PART 4 — Summary table
# ══════════════════════════════════════════════════════════════════════════════
# Quick reference: best metrics achieved per condition.

if not df.empty:
    _summary = df[["gain", "dynamics", "init", "min_val_loss", "max_val_acc"]].copy()
    _summary["min_val_loss"] = _summary["min_val_loss"].round(4)
    _summary["max_val_acc"]  = _summary["max_val_acc"].round(4)
    _summary = _summary.sort_values(["gain", "dynamics", "init"]).reset_index(drop=True)
    print("\n── Best metrics per condition ────────────────────────────────")
    print(_summary.to_string(index=False))
