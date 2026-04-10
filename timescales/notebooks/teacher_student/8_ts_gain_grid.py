# %% [markdown]
# # Teacher-Student Gain Grid: Teacher g x Student g
#
# Deconfounding "higher student g is better" from "matching the teacher helps."
# 2D heatmaps of final MSE, R-squared, and steps to convergence.

# %%
import os, sys, subprocess, json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()
os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())

# %% Discover sweep
LOGS_DIR = "logs/experiments"
sweep_dirs = sorted(
    [d for d in os.listdir(LOGS_DIR) if d.startswith("teacher_student_gain_grid")],
    reverse=True,
)
if not sweep_dirs:
    raise RuntimeError("No teacher_student_gain_grid directories found")
sweep_dir = os.path.join(LOGS_DIR, sweep_dirs[0])
print(f"Using sweep: {sweep_dir}")

# %% Load data
records = []
for exp_name in sorted(os.listdir(sweep_dir)):
    exp_dir = os.path.join(sweep_dir, exp_name)
    if not os.path.isdir(exp_dir) or exp_name in ("configs",):
        continue
    if not exp_name.startswith("tg_"):
        continue
    parts = exp_name.split("_")
    try:
        teacher_g = float(parts[1])
        student_g = float(parts[3])
        activation = parts[4]
    except (IndexError, ValueError):
        continue

    for sdn in sorted(os.listdir(exp_dir)):
        if not sdn.startswith("seed_"):
            continue
        seed = int(sdn.split("_")[1])
        sp = os.path.join(exp_dir, sdn)

        fvl = None
        rf = os.path.join(sp, "job_result.yaml")
        if os.path.exists(rf):
            with open(rf) as f:
                fvl = yaml.safe_load(f).get("final_val_loss")
        if fvl is None:
            lf = os.path.join(sp, "training_losses.json")
            if os.path.exists(lf):
                with open(lf) as f:
                    ld = json.load(f)
                vl = ld.get("val_losses", ld.get("val_losses_epoch", []))
                if vl:
                    fvl = vl[-1]
        if fvl is None:
            continue

        va, vll, st = [], [], []
        lf = os.path.join(sp, "training_losses.json")
        if os.path.exists(lf):
            with open(lf) as f:
                ld = json.load(f)
            va = ld.get("val_accuracies", [])
            vll = ld.get("val_losses", ld.get("val_losses_epoch", []))
            st = ld.get("steps", [])

        records.append(dict(
            exp_name=exp_name, teacher_g=teacher_g, student_g=student_g,
            activation=activation, seed=seed, final_val_loss=fvl,
            final_r2=va[-1] if va else np.nan,
            val_accuracies=va, val_losses=vll, steps=st,
        ))

df = pd.DataFrame(records)
print(f"Loaded {len(df)} runs")
activations = sorted(df["activation"].unique())

# %% Heatmap helper
def plot_heatmap(pivot, ax, cmap, norm, cbar_label, fmt_fn, title):
    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v}" for v in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v}" for v in pivot.index], fontsize=9)
    ax.set_xlabel("Student $g$", fontsize=12)
    ax.set_ylabel("Teacher $g$", fontsize=12)
    ax.set_title(title, fontsize=13)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            txt, clr = fmt_fn(pivot.values[i, j], pivot.values)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=clr)
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

# %% Heatmap: Final MSE
fig, axes = plt.subplots(1, len(activations), figsize=(6*len(activations), 5))
if len(activations) == 1:
    axes = [axes]
for ax, act in zip(axes, activations):
    sub = df[df["activation"] == act]
    piv = sub.pivot_table(values="final_val_loss", index="teacher_g",
                          columns="student_g", aggfunc="mean").sort_index(ascending=False)
    pos = piv.values[piv.values > 0]
    norm = mcolors.LogNorm(vmin=pos.min(), vmax=pos.max()) if len(pos) else None
    med = np.median(pos) if len(pos) else 1.0
    def fmt(v, a, _m=med):
        if np.isnan(v): return ("", "black")
        t = f"{v:.1e}" if v < 0.01 else f"{v:.3f}"
        return (t, "white" if v > _m else "black")
    plot_heatmap(piv, ax, "viridis_r", norm, "MSE", fmt, act)
fig.suptitle("Teacher x Student Gain Grid: Final MSE", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout(); plt.show()

# %% Heatmap: Final R-squared
fig, axes = plt.subplots(1, len(activations), figsize=(6*len(activations), 5))
if len(activations) == 1:
    axes = [axes]
for ax, act in zip(axes, activations):
    sub = df[df["activation"] == act]
    piv = sub.pivot_table(values="final_r2", index="teacher_g",
                          columns="student_g", aggfunc="mean").sort_index(ascending=False)
    def fmt(v, a):
        if np.isnan(v): return ("", "black")
        return (f"{v:.3f}", "white" if v < 0.5 else "black")
    plot_heatmap(piv, ax, "RdYlGn", None, "$R^2$", fmt, act)
fig.suptitle("Teacher x Student Gain Grid: Final $R^2$", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout(); plt.show()

# %% Steps to convergence
R2_THRESHOLD = 0.99
conv_records = []
for _, row in df.iterrows():
    va = row.get("val_accuracies", [])
    if not va:
        conv_records.append(dict(teacher_g=row["teacher_g"], student_g=row["student_g"],
                                 activation=row["activation"], steps_to_convergence=np.nan))
        continue
    sarr = np.array(row["steps"][:len(va)]) if row["steps"] else np.arange(1, len(va)+1)
    va_arr = np.array(va)
    hit = np.where(va_arr >= R2_THRESHOLD)[0]
    stc = int(sarr[hit[0]]) if len(hit) else np.nan
    conv_records.append(dict(teacher_g=row["teacher_g"], student_g=row["student_g"],
                             activation=row["activation"], steps_to_convergence=stc))

df_conv = pd.DataFrame(conv_records)
fig, axes = plt.subplots(1, len(activations), figsize=(6*len(activations), 5))
if len(activations) == 1:
    axes = [axes]
for ax, act in zip(axes, activations):
    sub = df_conv[df_conv["activation"] == act]
    piv = sub.pivot_table(values="steps_to_convergence", index="teacher_g",
                          columns="student_g", aggfunc="mean").sort_index(ascending=False)
    def fmt(v, a):
        if np.isnan(v): return ("DNF", "red")
        return (f"{int(v)}", "white" if v > np.nanmedian(a) else "black")
    plot_heatmap(piv, ax, "viridis_r", None, "Steps", fmt, act)
fig.suptitle(f"Steps to Convergence ($R^2 \\geq {R2_THRESHOLD}$)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout(); plt.show()

# %% Diagonal analysis: is matching always best?
print("\n" + "="*60)
print("For each teacher_g, which student_g converges fastest?")
print("="*60)
for act in activations:
    sub = df_conv[df_conv["activation"] == act]
    print(f"\n{act}:")
    for tg in sorted(sub["teacher_g"].unique()):
        rs = sub[sub["teacher_g"] == tg].sort_values("steps_to_convergence")
        if not rs.empty and not np.isnan(rs.iloc[0]["steps_to_convergence"]):
            b = rs.iloc[0]
            tag = " (matched!)" if np.isclose(tg, b["student_g"]) else ""
            print(f"  tg={tg:.2f} -> best sg={b['student_g']:.2f}"
                  f"  ({int(b['steps_to_convergence'])} steps){tag}")
        else:
            print(f"  tg={tg:.2f} -> no run converged")
