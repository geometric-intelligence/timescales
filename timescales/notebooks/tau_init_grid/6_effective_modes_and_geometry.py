# %% [markdown]
# # An effective mode basis for the nonlinear network, and the geometry of the orbit
#
# Two questions left open by the previous notebook.
#
# **1. What is a good basis for the Tanh network?** The Jacobian eigenmodes give clean,
# interpretable planes for the linear network — each output pair reads one mode, and the
# trajectory traces a circle in it. For the Tanh network the same construction produces
# nothing: the coupled modes have periods unrelated to the task and explain none of the
# output. The reason is that the stored Jacobian is linearized **at the origin**, while a
# trained Tanh network runs on a limit cycle far away from it.
#
# The fix is to linearize where the network actually operates. **Dynamic mode
# decomposition** does this from data: fit the best linear operator $A$ with
# $h(t{+}1) \approx A\,h(t)$ along the observed trajectory, then use *its* eigenvectors.
# For a linear network DMD must recover the Jacobian, which makes it easy to validate;
# for the Tanh network it gives the effective dynamics on the attractor.
#
# **2. What is the overall geometry?** Each output pair traces a circle at its own
# frequency, so the orbit lies on a product of three circles — a 3-torus. Section 4 pins
# down which curve on that torus it is.

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

for _cand in (os.getcwd(), os.path.abspath(os.path.join(os.getcwd(), ".."))):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)
from train import _create_rnn_model  # noqa: E402

EXP = os.path.join(os.getcwd(), "logs", "experiments")
RUNS = {
    "linear": os.path.join(EXP, "tau_init_grid", "sine_uniform_full_g0.9", "seed_0"),
    "tanh": os.path.join(EXP, "tau_init_grid_tanh_voltage",
                         "sine_uniform_full_g0.9", "seed_0"),
}
FIGS_DIR = os.environ.get(
    "FIGS_DIR", os.path.join(os.getcwd(), "notebooks", "tau_init_grid", "figs"))
os.makedirs(FIGS_DIR, exist_ok=True)

T_ROLL, RANK = 600, 6
CMAP_PHASE = "twilight"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def rollout(run, T=T_ROLL):
    """Autonomous rollout of a trained network, plus its stored spectral quantities."""
    cfg = yaml.safe_load(open(os.path.join(run, "config_seed0.yaml")))
    model, _ = _create_rnn_model(cfg)
    model.load_state_dict(torch.load(os.path.join(run, "final_model_seed0.pth"),
                                     map_location="cpu", weights_only=False))
    model.eval()
    with torch.no_grad():
        H, Y = model(inputs=torch.zeros(1, T, cfg["input_size"]),
                     init_hidden=torch.full((1, cfg["hidden_size"]),
                                            float(cfg["init_hidden_value"])))
    blob = torch.load(os.path.join(run, "spectral_final.pt"), map_location="cpu",
                      weights_only=False)
    return (H[0].numpy(), Y[0].numpy(), np.asarray(blob["W_out"], float),
            np.asarray(blob["V"], complex), np.asarray(blob["eigvals_eig"], complex),
            [float(p) for p in cfg["periods"]], float(cfg["dt"]))


def dmd(H, r=RANK):
    """Exact DMD on a single trajectory, truncated to rank r.

    Returns eigenvalues, modes, per-step coefficients and the mean that was removed.
    The mean is subtracted so the rank budget is spent on the oscillation rather than on
    a constant offset, which would otherwise consume one mode at eigenvalue 1.
    """
    mu = H.mean(0, keepdims=True)
    X = H - mu
    A_, B_ = X[:-1].T, X[1:].T
    U, S, Vt = np.linalg.svd(A_, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    A_tilde = Ur.T @ B_ @ Vr @ np.diag(1 / Sr)
    ev, W = np.linalg.eig(A_tilde)
    Phi = B_ @ Vr @ np.diag(1 / Sr) @ W            # (N, r) DMD modes
    coef = (np.linalg.pinv(Phi) @ X.T).T           # (T, r)
    return ev, Phi, coef, mu


def period_of(z):
    th = abs(np.angle(z))
    return 2 * np.pi / th if th > 1e-9 else np.inf


DATA = {}
for name, run in RUNS.items():
    H, Y, W_out, V, lam, PERIODS, DT = rollout(run)
    ev, Phi, coef, mu = dmd(H)
    DATA[name] = dict(H=H, Y=Y, W_out=W_out, V=V, lam=lam, ev=ev, Phi=Phi,
                      coef=coef, mu=mu)
    print(f"{name}: H{H.shape}, DMD rank {RANK}")

t = np.arange(T_ROLL) * DT
phase = {k: (2 * np.pi * t / P) % (2 * np.pi) for k, P in enumerate(PERIODS)}
print(f"task periods: {PERIODS}")

# %% [markdown]
# ## 1 — Does DMD recover the task frequencies where the origin Jacobian does not?
#
# For each output pair, the mode it couples to most strongly, under each basis. The
# reconstruction column rebuilds that output channel from that single mode alone, which
# is the test of whether the basis is actually describing the computation.

# %%
def coupled_modes(W_out, basis, evals, Y, coef, offset=None):
    """Per output channel: most strongly coupled mode, its period, and single-mode R^2."""
    coup = np.abs(W_out @ basis)
    top = np.argmax(coup, axis=1)
    rows = []
    for ch in range(W_out.shape[0]):
        j = top[ch]
        rec = 2.0 * np.real((W_out[ch] @ basis[:, j]) * coef[:, j])
        if offset is not None:
            rec = rec + W_out[ch] @ offset[0]
        r2 = 1 - np.var(Y[:, ch] - rec) / max(np.var(Y[:, ch]), 1e-12)
        rows.append((j, period_of(evals[j]), abs(evals[j]), r2))
    return top, rows


print(f"{'regime':7s} {'basis':16s} {'pair':4s} {'target':>7s} | {'mode':>5s} "
      f"{'period':>8s} {'|z|':>7s} {'R² (ch a)':>10s} {'R² (ch b)':>10s}  paired?")
SUMMARY = {}
for name in ("linear", "tanh"):
    d = DATA[name]
    C_jac = (np.linalg.solve(d["V"], d["H"].T.astype(complex))).T
    for basis_name, basis, evals, coef, offset in (
            ("origin Jacobian", d["V"], d["lam"], C_jac, None),
            ("DMD (effective)", d["Phi"], d["ev"], d["coef"], d["mu"])):
        top, rows = coupled_modes(d["W_out"], basis, evals, d["Y"], coef, offset)
        SUMMARY[(name, basis_name)] = (top, rows)
        for k, P in enumerate(PERIODS):
            j, per, mag, r2a = rows[2 * k]
            _, _, _, r2b = rows[2 * k + 1]
            pd = "yes" if top[2 * k] == top[2 * k + 1] else "NO"
            ps = f"{per:8.2f}" if np.isfinite(per) else "     inf"
            print(f"{name:7s} {basis_name:16s} {k:<4d} {P:7.0f} | {j:5d} {ps} "
                  f"{mag:7.4f} {r2a:10.3f} {r2b:10.3f}  {pd}")

# %% [markdown]
# The Tanh rows are the point. Under the origin Jacobian the two channels of a pair select
# *different* modes, the periods are unrelated to the targets, and single-mode
# reconstruction explains essentially none of the output. Under DMD the pair collapses onto
# one mode at the right period, and that mode alone reconstructs the channel. The
# one-to-one structure was there all along; the origin Jacobian was simply the wrong place
# to look for it.
#
# For the linear network DMD agrees with the Jacobian, as it must — that is the control
# that says the method is not manufacturing structure.
#
# ## 2 — The trajectory in the DMD eigenplanes
#
# The same plot as before, now in the effective basis. Colour is the target phase of the
# corresponding pair.

# %%
for name in ("linear", "tanh"):
    d = DATA[name]
    top, rows = SUMMARY[(name, "DMD (effective)")]
    fig, axes = plt.subplots(1, len(PERIODS), figsize=(3.5 * len(PERIODS), 3.5))
    for k, P in enumerate(PERIODS):
        ax = axes[k]
        j = top[2 * k]
        c = d["coef"][:, j]
        sc = ax.scatter(c.real, c.imag, c=phase[k], cmap=CMAP_PHASE, s=5)
        ax.scatter(c.real[0], c.imag[0], color="crimson", s=36, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"pair {k}: target T={P:g}\nDMD mode {j}, period "
                     f"{period_of(d['ev'][j]):.1f}", fontsize=9)
        ax.set_xlabel("Re b", fontsize=8)
        ax.set_ylabel("Im b", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(sc, ax=ax, fraction=0.046, label=f"target phase T={P:g}")
    fig.suptitle(f"2 — Trajectory in the DMD eigenplanes, {name} network", fontsize=11,
                 y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"eff1_dmd_planes_{name}.png"))
    plt.show()

# %% [markdown]
# ## 3 — Overall geometry: which curve on the torus?
#
# Each pair contributes a circle, so the orbit lives on a 3-torus $S^1\times S^1\times S^1$
# embedded in the 6-dimensional subspace the trajectory occupies. Which curve it is
# depends on the frequency ratios. The task periods 20, 50 and 100 have least common
# multiple 100, so an exact solution closes after 100 steps, winding **5, 2 and 1** times
# around the three circles respectively — a closed $(5,2,1)$ curve rather than a
# space-filling one.
#
# Two checks: how nearly the state returns after 100 steps, and how many turns each DMD
# mode's phase actually accumulates.

# %%
print(f"{'regime':8s} {'rel. ||h(t+100)-h(t)||':>24s} | winding turns per 100 steps")
for name in ("linear", "tanh"):
    d = DATA[name]
    H = d["H"]
    closure = np.median([np.linalg.norm(H[i + 100] - H[i]) / np.linalg.norm(H[i])
                         for i in range(200, 400)])
    top, _ = SUMMARY[(name, "DMD (effective)")]
    turns = []
    for k in range(len(PERIODS)):
        j = top[2 * k]
        turns.append(100.0 / period_of(d["ev"][j]))
    print(f"{name:8s} {closure:24.4f} | " +
          "  ".join(f"pair{k}: {w:.2f}" for k, w in enumerate(turns)))

# %% [markdown]
# ## 4 — The torus, unrolled
#
# Plotting two of the three mode phases against each other flattens the torus into a
# square with periodic edges. A closed curve with rational winding shows up as a finite
# family of parallel lines; an irrational one would fill the square. Colour is the third
# phase, so the whole 3-torus is represented.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
for ax, name in zip(axes, ("linear", "tanh"), strict=False):
    d = DATA[name]
    top, _ = SUMMARY[(name, "DMD (effective)")]
    th = [np.angle(d["coef"][:, top[2 * k]]) % (2 * np.pi) for k in range(3)]
    sc = ax.scatter(th[0], th[1], c=th[2], cmap=CMAP_PHASE, s=6)
    ax.set_xlabel(r"phase of pair 0 mode (T=20)", fontsize=9)
    ax.set_ylabel(r"phase of pair 1 mode (T=50)", fontsize=9)
    ax.set_title(f"{name} network", fontsize=10)
    ax.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
    ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
    plt.colorbar(sc, ax=ax, fraction=0.046, label="phase of pair 2 mode (T=100)")
fig.suptitle("4 — The 3-torus unrolled: pairwise mode phases", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "eff2_torus_unrolled.png"))
plt.show()

# %% [markdown]
# ## Summary
#
# **A basis for the nonlinear network.** DMD on the trajectory supplies what the origin
# Jacobian cannot: modes at the task frequencies that each output pair reads one-to-one,
# and which reconstruct their channel on their own. It reduces to the Jacobian
# eigendecomposition in the linear case, so it is a strict generalization rather than a
# different measurement. The apparent absence of structure in the Tanh network was a
# property of the coordinate system, not of the network.
#
# **The geometry.** In both regimes the orbit is a closed curve winding 5, 2 and 1 times
# around three circles — one per frequency — carried by three effective modes spanning a
# 6-dimensional subspace. That is the same organization in both regimes; only the basis
# needed to see it differs.
