# %% [markdown]
# # Fixed points, attractors and perturbation response
#
# Whether the two regimes differ in their *stability* structure, not just their geometry.
#
# The linear network's dynamics are `h⁺ = (1−α)h + α·gWh`, exactly linear, so the Jacobian
# is the same everywhere. The Tanh network's are `h⁺ = (1−α)h + α·gW·tanh(h)`, so
#
# $$J(h) = \mathrm{diag}(1-\alpha) + \mathrm{diag}(\alpha)\,gW\,\mathrm{diag}(1-\tanh^2 h)$$
#
# depends on the state: away from the origin, saturation shrinks the effective recurrent
# gain. That single fact drives everything here.
#
# Findings are recorded in `docs/geometry_notes.md` (items 3, 6, 7, 10, 11).

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

T_ROLL, T_PERTURB, RANK, PERIOD = 900, 300, 6, 100
NONLIN = {"linear": False, "tanh": True}
COLOR = {"linear": "#2a78d6", "tanh": "#eb6834"}
C_MUTED = "#898781"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def load(name):
    run = RUNS[name]
    cfg = yaml.safe_load(open(os.path.join(run, "config_seed0.yaml")))
    model, _ = _create_rnn_model(cfg)
    model.load_state_dict(torch.load(os.path.join(run, "final_model_seed0.pth"),
                                     map_location="cpu", weights_only=False))
    model.eval()
    return model, cfg


def rollout(model, cfg, T=T_ROLL, h0=None, perturb=None):
    """Autonomous rollout, optionally adding a vector to the state at a given step."""
    h = (h0 if h0 is not None
         else torch.full((1, cfg["hidden_size"]), float(cfg["init_hidden_value"])))
    z = torch.zeros(1, model.rnn_step.W_in.in_features)
    out = []
    with torch.no_grad():
        for t in range(T):
            if perturb is not None and t == perturb[0]:
                h = h + perturb[1]
            h = model.rnn_step(z, h)
            out.append(h[0].numpy().copy())
    return np.asarray(out)


def jacobian(model, h, nonlinear):
    st = model.rnn_step
    a = st.current_alphas.detach().numpy()
    W = st.W_rec.weight.detach().numpy() * float(st.recurrent_weight_scale)
    d = (1 - np.tanh(h) ** 2) if nonlinear else np.ones_like(h)
    return np.diag(1 - a) + (a[:, None] * W) * d[None, :]


def dmd_basis(H, r=RANK):
    """Rank-r DMD. Returns eigenvalues, modes, and the mean that was removed."""
    mu = H.mean(0, keepdims=True)
    X = H - mu
    A_, B_ = X[:-1].T, X[1:].T
    U, S, Vt = np.linalg.svd(A_, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    ev, W = np.linalg.eig(Ur.T @ B_ @ Vr @ np.diag(1 / Sr))
    return ev, B_ @ Vr @ np.diag(1 / Sr) @ W, mu


MODELS = {n: load(n) for n in RUNS}
REF = {n: rollout(*MODELS[n]) for n in RUNS}
DMD = {n: dmd_basis(REF[n][T_PERTURB:]) for n in RUNS}
# oscillatory modes, one per conjugate pair, ordered fastest to slowest
OSC = {n: sorted([j for j in range(RANK) if np.angle(DMD[n][0][j]) > 1e-9],
                 key=lambda j, n=n: -abs(np.angle(DMD[n][0][j]))) for n in RUNS}
PERIODS = {n: [2 * np.pi / abs(np.angle(DMD[n][0][j])) for j in OSC[n]] for n in RUNS}
for n in RUNS:
    print(f"{n:7s} DMD periods: {[round(p, 2) for p in PERIODS[n]]}")


def coeffs(name, H):
    """Coefficients of states in that network's DMD basis."""
    ev, Phi, mu = DMD[name]
    return (np.linalg.pinv(Phi) @ (H - mu).T).T


# %% [markdown]
# ## 1 — Floquet multipliers: the correct stability object
#
# Instantaneous Jacobians are suggestive but do not establish stability: the J(h(t)) do not
# commute, and a product of matrices each with spectral radius below 1 can still grow. What
# governs stability around a cycle is the **monodromy matrix**, the product of the
# instantaneous Jacobians around one period. Its eigenvalues are the Floquet multipliers —
# modulus 1 is neutral, below 1 contracting, above 1 expanding.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
print(f"{'regime':8s} {'max|λ| origin':>14s} | top |Floquet multipliers| over one period")
for ax, n in zip(axes, RUNS, strict=False):
    model, cfg = MODELS[n]
    M = np.eye(cfg["hidden_size"])
    for t in range(400, 400 + PERIOD):
        M = jacobian(model, REF[n][t], NONLIN[n]) @ M
    mult = np.linalg.eigvals(M)
    mult = mult[np.argsort(-np.abs(mult))]
    m0 = np.abs(np.linalg.eigvals(
        jacobian(model, np.zeros(cfg["hidden_size"]), NONLIN[n]))).max()
    print(f"{n:8s} {m0:14.4f} | {np.round(np.abs(mult[:6]), 4).tolist()}")

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=C_MUTED, lw=1, ls="--", label="unit circle")
    ax.scatter(mult.real, mult.imag, s=26, color=COLOR[n], alpha=0.85, edgecolors="none")
    n_neutral = int((np.abs(mult) > 0.9).sum())
    ax.set_title(f"{n}\n{n_neutral} multiplier(s) with |μ| > 0.9", fontsize=10)
    ax.set_xlabel("Re μ")
    ax.set_ylabel("Im μ")
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.legend(fontsize=7.5, loc="upper right")
fig.suptitle("1 — Floquet multipliers of the monodromy matrix (one period = 100 steps)",
             fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "perturb1_floquet.png"))
plt.show()

# %% [markdown]
# The nonlinear network has **one** near-neutral multiplier with everything else strongly
# contracting — one neutral direction plus transverse contraction is the definition of an
# attracting limit cycle, and the neutral direction is the phase. The linear network has
# **six**: the whole task subspace is neutral, so nothing attracts.
#
# ## 2 — Amplitude per frequency from random initial states
#
# `‖h‖` is a poor observable — it oscillates as the state moves around the orbit. The
# amplitude of each frequency separately, |b_k| in the DMD basis, is constant on the orbit
# and shows the difference cleanly.

# %%
fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.8), sharex=True)
for row, n in enumerate(RUNS):
    model, cfg = MODELS[n]
    for s in range(5):
        h0 = torch.tensor(np.random.default_rng(s).normal(0, 0.5, (1, cfg["hidden_size"])),
                          dtype=torch.float32)
        b = coeffs(n, rollout(model, cfg, T=1500, h0=h0))
        for col, j in enumerate(OSC[n]):
            axes[row][col].plot(np.abs(b[:, j]), color=COLOR[n], lw=1.0, alpha=0.75)
    for col, per in enumerate(PERIODS[n]):
        axes[row][col].set_yscale("log")
        axes[row][col].set_title(f"{n} · mode T≈{per:.0f}", fontsize=9.5)
        if row == 1:
            axes[row][col].set_xlabel("step")
    axes[row][0].set_ylabel(r"amplitude $|b_k|$")
fig.suptitle("2 — Per-frequency amplitude from 5 random initial states "
             "(linear: set by the start; nonlinear: one attractor)", fontsize=10.5, y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "perturb2_amplitude_random_starts.png"))
plt.show()

# %%
print("Amplitude per frequency at t = 1400, from 5 random initial states:\n")
for n in RUNS:
    model, cfg = MODELS[n]
    print(f"  {n}")
    for s in range(5):
        h0 = torch.tensor(np.random.default_rng(s).normal(0, 0.5, (1, cfg["hidden_size"])),
                          dtype=torch.float32)
        b = coeffs(n, rollout(model, cfg, T=1500, h0=h0))
        amps = [np.abs(b[1350:1450, j]).mean() for j in OSC[n]]
        print(f"      start {s}: " + "  ".join(
            f"T≈{p:5.1f}: {a:8.4f}" for p, a in zip(PERIODS[n], amps, strict=False)))

# %% [markdown]
# ## 3 — Why a *random* perturbation appears to decay in both networks
#
# Before comparing recovery, one confound has to be removed. A random direction in 512
# dimensions barely intersects the 6-dimensional task subspace, so most of any random kick
# sits in fast-decaying directions and disappears regardless of regime.

# %%
print("Fraction of a random unit perturbation lying in the 6-D task subspace:\n")
for n in RUNS:
    H = REF[n][T_PERTURB:]
    U = np.linalg.svd(H - H.mean(0, keepdims=True), full_matrices=False)[2][:RANK]
    rng = np.random.default_rng(0)
    fr = []
    for _ in range(200):
        xi = rng.normal(0, 1, U.shape[1])
        fr.append(np.linalg.norm(U @ (xi / np.linalg.norm(xi))))
    print(f"  {n:8s} {np.mean(fr):.4f} ± {np.std(fr):.4f}")
print(f"  {'expected':8s} {np.sqrt(RANK / 512):.4f}   (random direction in 512-D)")
print("\nSo ~89% of a random kick was never in the persistent subspace. The linear network")
print("is not correcting it — it is shedding a component that could never have persisted.")

# %% [markdown]
# ## 4 — Perturbation confined to a single mode's eigenplane
#
# This removes the confound: the whole perturbation is placed in one mode's own 2D plane,
# so 100% of it lands in the persistent subspace. Any decay now is genuine correction.

# %%
def eigenplane_perturbation(name, j, frac=0.20):
    """A perturbation lying entirely in the real 2D plane of DMD mode j."""
    _, Phi, _ = DMD[name]
    v = Phi[:, j]
    e1 = np.real(v) / np.linalg.norm(np.real(v))
    e2 = np.imag(v) - (np.imag(v) @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return (e1 + e2) / np.sqrt(2) * np.linalg.norm(REF[name][T_PERTURB]) * frac


fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.8), sharex=True)
for row, n in enumerate(RUNS):
    model, cfg = MODELS[n]
    br = coeffs(n, REF[n])
    for col, (j, per) in enumerate(zip(OSC[n], PERIODS[n], strict=False)):
        xi = eigenplane_perturbation(n, j)
        P = rollout(model, cfg, perturb=(T_PERTURB,
                                         torch.tensor(xi[None], dtype=torch.float32)))
        bp = coeffs(n, P)
        ratio = np.abs(bp[:, j]) / np.abs(br[:, j])
        ax = axes[row][col]
        ax.plot(np.arange(len(ratio)) - T_PERTURB, ratio, color=COLOR[n], lw=1.6)
        ax.axhline(1.0, color=C_MUTED, lw=1, ls="--")
        ax.axvline(0, color=C_MUTED, lw=1, ls=":")
        ax.set_xlim(-40, 560)
        ax.set_ylim(0.45, 1.45)
        ax.set_title(f"{n} · mode T≈{per:.0f}", fontsize=9.5)
        if row == 1:
            ax.set_xlabel("steps since perturbation")
    axes[row][0].set_ylabel("amplitude ratio\nperturbed / reference")
fig.suptitle("4 — Recovery after a 20% perturbation placed inside one mode's eigenplane "
             "(dashed = full recovery)", fontsize=10.5, y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "perturb3_eigenplane_recovery.png"))
plt.show()

# %% [markdown]
# The linear traces are flat: a 20% amplitude error is still a 20% error 500 steps later.
# The nonlinear traces return to 1. This is the cleanest form of the result — with the
# geometric confound removed, the linear network shows *no* correction whatsoever.
#
# ## 5 — The same thing seen in the eigenplane itself
#
# Plotting the trajectory in the perturbed mode's own plane makes the mechanism visible:
# the nonlinear trajectory spirals back onto the reference circle, while the linear one
# simply continues on a circle of the wrong radius.

# %%
for n in RUNS:
    model, cfg = MODELS[n]
    br = coeffs(n, REF[n])
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    for ax, j, per in zip(axes, OSC[n], PERIODS[n], strict=False):
        xi = eigenplane_perturbation(n, j)
        bp = coeffs(n, rollout(model, cfg,
                               perturb=(T_PERTURB,
                                        torch.tensor(xi[None], dtype=torch.float32))))
        seg = slice(T_PERTURB, T_PERTURB + 520)
        ax.plot(br[seg, j].real, br[seg, j].imag, color=C_MUTED, lw=1.2,
                label="unperturbed")
        sc = ax.scatter(bp[seg, j].real, bp[seg, j].imag, s=3,
                        c=np.arange(520), cmap="viridis", label="perturbed")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"mode T≈{per:.0f}", fontsize=9.5)
        ax.set_xlabel(r"Re $b_k$", fontsize=8)
        ax.set_ylabel(r"Im $b_k$", fontsize=8)
        ax.tick_params(labelsize=7)
    plt.colorbar(sc, ax=axes[-1], fraction=0.046, label="steps since perturbation")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"5 — Perturbed vs unperturbed trajectory in the eigenplane · {n}",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"perturb4_eigenplane_{n}.png"))
    plt.show()

# %% [markdown]
# ## 6 — What is *not* recovered: phase
#
# A perturbation along the cycle changes *when* the network is on its orbit, not *where*
# the orbit is. The dynamics are autonomous — the update contains no reference to absolute
# time — so if h*(t) is a solution then so is h*(t+s), and nothing prefers one over the
# other. That is the neutral Floquet multiplier from section 1: its eigenvector is the
# tangent to the cycle, and perturbations along it are neither amplified nor damped.

# %%
print("Amplitude and phase of the perturbed mode, 500 steps after a 20% eigenplane kick:\n")
print(f"{'regime':8s} {'mode':>8s} | {'amplitude ratio':>16s} | {'phase offset':>13s}")
for n in RUNS:
    model, cfg = MODELS[n]
    br = coeffs(n, REF[n])
    for j, per in zip(OSC[n], PERIODS[n], strict=False):
        xi = eigenplane_perturbation(n, j)
        bp = coeffs(n, rollout(model, cfg,
                               perturb=(T_PERTURB,
                                        torch.tensor(xi[None], dtype=torch.float32))))
        k = T_PERTURB + 500
        amp = abs(bp[k, j]) / abs(br[k, j])
        dph = np.degrees(np.angle(bp[k, j] / br[k, j]))
        print(f"{n:8s} {f'T≈{per:.0f}':>8s} | {amp:16.4f} | {dph:+12.2f}°")
print("\nThe nonlinear network restores amplitude but keeps a phase offset, which is what")
print("should happen: phase is the neutral direction. It maintains the shape of its output")
print("and cannot re-lock its timing, since an autonomous task supplies no reference.")
