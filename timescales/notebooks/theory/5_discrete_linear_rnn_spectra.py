# %% [markdown]
# # Discrete-Time Linear RNN: Jacobian Spectrum vs g, tau-distribution, and W-distribution
#
# **Goal**: Understand how the Jacobian spectrum and effective timescale distribution
# of a *discrete-time* linear RNN change as we vary:
#
# 1. Recurrent gain g (fixed Gaussian W, uniform tau)
# 2. Time-constant distribution (fixed g, fixed Gaussian W)
# 3. Weight distribution (fixed g, uniform tau)
#
# **Discrete-time update rule** (Identity activation, rate dynamics):
#   h_{t+1} = (I - A) h_t + A g W h_t = J h_t
# where A = diag(alpha_1, ..., alpha_N),  alpha_i = 1 - exp(-dt/tau_i).
#
# **Jacobian**: J = (I - A) + A g W
#
# For *uniform* tau (scalar alpha) this simplifies to J = (1-alpha)I + alpha g W,
# whose eigenvalues are lambda_J = (1-alpha) + alpha g lambda_W -- an affine image
# of W's bulk disk, centred at (1-alpha, 0) with radius alpha g.
#
# For *heterogeneous* alpha_i, J and W no longer share eigenvectors; the spectrum
# must be computed numerically.
#
# **Effective timescale** (discrete-time):
#   tau_eff_k = -1 / ln|lambda_{J,k}|  (in units of timesteps)
# Defined and positive for stable modes (|lambda| < 1); diverges at |lambda| = 1.
#
# Contrast with notebook 1 (continuous-time): M = (gW-I)/tau,
# tau_eff^CT = -1/Re(lambda_M).
# Notebook 7 covers the continuous-time heterogeneous-tau case M = T^{-1}(gW-I).


# %% [markdown]
# ## Setup

# %%
import os
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from scipy.stats import levy_stable

_gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()
sys.path.insert(0, _gitroot)
sys.path.insert(0, os.path.join(_gitroot, "timescales"))

SEED = 42
rng = np.random.default_rng(SEED)

plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["font.size"] = 12

%config InlineBackend.figure_format = 'svg'
plt.rcParams["svg.fonttype"] = "path"   # text → outlines; renders identically in Illustrator

# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_J_eigs(W: np.ndarray, g: float, alphas) -> np.ndarray:
    """
    Eigenvalues of  J = diag(1-alpha) + diag(alpha) * g * W.

    `alphas` may be a scalar (uniform) or a length-N array (heterogeneous).
    For uniform alpha the eigenvalues are just an affine map of lambda_W
    (no matrix construction needed).
    """
    N = W.shape[0]
    a = np.broadcast_to(np.asarray(alphas, dtype=float), (N,)).copy()
    if a.min() == a.max():
        eigs_W = np.linalg.eigvals(W)
        return (1.0 - a[0]) + a[0] * g * eigs_W
    J = np.diag(1.0 - a) + a[:, np.newaxis] * (g * W)
    return np.linalg.eigvals(J)


def tau_eff_stable(eigs_J: np.ndarray, abs_tol: float = 1e-6) -> np.ndarray:
    """
    tau_eff = -1/ln|lambda| for stable modes (|lambda| < 1 - abs_tol).
    Returns only the stable subset (all values are positive).
    """
    abs_e = np.abs(eigs_J)
    stable = abs_e < (1.0 - abs_tol)
    log_abs = np.log(np.clip(abs_e[stable], 1e-12, None))
    return -1.0 / np.where(log_abs < -1e-10, log_abs, -1e-10)


def make_alphas(N: int, dist: str, dt: float, rng_: np.random.Generator, **kw):
    """
    Sample N time constants from `dist`, return (alphas, taus).

    dist="uniform"     : all tau = kw["tau"] (default 1.0)
    dist="gaussian"    : tau ~ N(mu, sigma), clipped to [lo, hi]
    dist="log_uniform" : tau ~ exp(Uniform(ln lo, ln hi))  -- p(tau) ~ 1/tau
    dist="powerlaw"    : p(tau) ~ tau^{-beta}, tau in [lo, hi], CDF inversion
    """
    if dist == "uniform":
        taus = np.full(N, kw.get("tau", 1.0))
    elif dist == "gaussian":
        mu, sigma = kw.get("mu", 5.0), kw.get("sigma", 2.0)
        lo, hi = kw.get("lo", 0.1), kw.get("hi", 20.0)
        taus = np.clip(rng_.normal(mu, sigma, N), lo, hi)
    elif dist == "log_uniform":
        lo, hi = kw.get("lo", 0.1), kw.get("hi", 20.0)
        taus = np.exp(rng_.uniform(np.log(lo), np.log(hi), N))
    elif dist == "powerlaw":
        lo, hi, beta = kw.get("lo", 0.1), kw.get("hi", 20.0), kw.get("beta", 2.0)
        u = rng_.uniform(0.0, 1.0, N)
        if abs(beta - 1.0) < 1e-6:
            taus = lo * (hi / lo) ** u
        else:
            exp = 1.0 - beta
            taus = (lo ** exp + u * (hi ** exp - lo ** exp)) ** (1.0 / exp)
    else:
        raise ValueError(f"Unknown distribution: {dist!r}")
    return 1.0 - np.exp(-dt / taus), taus


def generate_W_gauss(N: int, rng_: np.random.Generator) -> np.ndarray:
    return rng_.normal(0.0, 1.0 / N ** 0.5, size=(N, N)).astype(np.float32)


def generate_W_levy(N: int, alpha_stab: float, rng_: np.random.Generator,
                    normalization: str = "stable") -> np.ndarray:
    """
    W_ij ~ S(alpha_stab, beta=0, scale).

    normalization="stable"  -> scale = 1/N^{1/alpha_s} (spectral radius O(1) across alpha_s)
    normalization="rnn"     -> scale = 1/sqrt(N)        (matches wrec_init='levy_stable')
    """
    if normalization == "stable":
        scale = 1.0 / N ** (1.0 / alpha_stab)
        if alpha_stab == 2.0:
            scale /= 2.0 ** 0.5
    else:
        scale = 1.0 / N ** 0.5
    return levy_stable.rvs(
        alpha_stab, beta=0, loc=0, scale=scale, size=(N, N),
    ).astype(np.float32)


print("Helpers defined.")


# %% [markdown]
# ---
# ## Configuration

# %%
N_GAUSS  = 2000   # neurons for Gaussian W sections
N_LEVY   = 1000   # neurons for Levy W section (generation is slow for large N)
DT       = 0.5    # dt=1, tau=1 -> alpha = 1-exp(-1) ~ 0.632 (bulk clearly off-centre)
N_BINS   = 60
TAU_CAP  = 200    # cap tau_eff histograms to avoid inf near unit circle

# ---- Colormaps (change to customise all plots) ----------------------------
CMAP_SEC1 = "plasma"    # Section 1: gain sweep
CMAP_SEC2 = "viridis"   # Section 2: tau-distribution sweep
CMAP_SEC3 = "cividis"   # Section 3: W-distribution sweep

W_gauss = generate_W_gauss(N_GAUSS, rng)
_sr_Wg = np.abs(np.linalg.eigvals(W_gauss)).max()
print(f"Gaussian W: N={N_GAUSS}, spectral_radius(W)={_sr_Wg:.4f}  (expected ~1.0)")

_theta = np.linspace(0, 2 * np.pi, 400)   # unit circle arc, reused everywhere


# %% [markdown]
# ---
# ## Section 1 -- Gain Sweep
# Fixed: Gaussian W, uniform tau = 1.
#
# Because alpha is uniform, eigenvalues of J are just
#   lambda_J = (1-alpha) + alpha * g * lambda_W
# -- no matrix multiply needed.  The bulk fills a disk centred at (1-alpha, 0)
# with radius alpha*g.

# %%
G_VALUES    = [0.2, 0.5, 0.7, 0.9, 0.95, 1.0, 1.1]
TAU_FIXED   = 1.0
ALPHA_FIXED = 1.0 - np.exp(-DT / TAU_FIXED)

print(f"\nSection 1: tau={TAU_FIXED}, dt={DT}  =>  alpha={ALPHA_FIXED:.4f}")
print(f"J bulk: centre ({1 - ALPHA_FIXED:.3f}, 0),  radius = alpha*g")

results_g = {}
for g in G_VALUES:
    eigs_J = compute_J_eigs(W_gauss, g, ALPHA_FIXED)
    te = tau_eff_stable(eigs_J)
    n_stable = len(te)
    results_g[g] = dict(eigs_J=eigs_J, tau_eff=np.clip(te, None, TAU_CAP),
                        n_stable=n_stable, n_unstable=N_GAUSS - n_stable)
    msg = (f"tau_eff in [{te.min():.2f}, {min(te.max(), TAU_CAP):.1f}]"
           if n_stable else "no stable modes")
    print(f"  g={g}: {n_stable}/{N_GAUSS} stable  |  {msg}")

all_te_g  = np.concatenate([results_g[g]["tau_eff"] for g in G_VALUES
                             if results_g[g]["n_stable"] > 10])
ts_xmin_g = all_te_g.min() if len(all_te_g) else 0.5
ts_xmax_g = TAU_CAP

colors_g = plt.get_cmap(CMAP_SEC1)(np.linspace(0.15, 0.90, n_g := len(G_VALUES)))
xy_lim_g = 1.15   # always contains the unit circle

fig = plt.figure(figsize=(n_g * 3.0, 6.5))
gs1 = GridSpec(2, n_g, figure=fig,
               height_ratios=[1.0, 0.60],
               hspace=0.05, wspace=0.06,
               left=0.06, right=0.99, top=0.91, bottom=0.09)
axes_spec = [fig.add_subplot(gs1[0, c]) for c in range(n_g)]
axes_te   = [fig.add_subplot(gs1[1, c]) for c in range(n_g)]

for col, (g, color) in enumerate(zip(G_VALUES, colors_g)):
    res    = results_g[g]
    eigs_J = res["eigs_J"]
    te     = res["tau_eff"]

    # Row 0: J eigenspectrum
    ax = axes_spec[col]
    ax.scatter(eigs_J.real, eigs_J.imag, s=1, alpha=0.3, color=color, rasterized=True)
    ax.plot(np.cos(_theta), np.sin(_theta), "r--", lw=1.1, alpha=0.85)
    ax.axvline(0, color="gray", lw=0.4, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
    ax.set_xlim(-xy_lim_g, xy_lim_g)
    ax.set_ylim(-xy_lim_g, xy_lim_g)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"g = {g}", fontsize=10, color=color, pad=3)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("Im($\\lambda_J$)", fontsize=9)
    else:
        ax.set_yticklabels([])
    if res["n_unstable"] > 0:
        ax.annotate(f"{res['n_unstable']} unstable",
                    xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))

    # Row 1: tau_eff distribution
    ax = axes_te[col]
    if res["n_stable"] > 10:
        bins = np.logspace(np.log10(ts_xmin_g + 1e-3), np.log10(ts_xmax_g), N_BINS)
        ax.hist(te, bins=bins, density=True, alpha=0.8, color=color, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(ts_xmin_g * 0.9, ts_xmax_g * 1.1)
    ax.set_xlabel("$\\tau_{\\rm eff}$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel("Density", fontsize=9)
    else:
        ax.set_yticklabels([])

ylims_te = [axes_te[c].get_ylim() for c in range(n_g)
            if results_g[G_VALUES[c]]["n_stable"] > 10]
if ylims_te:
    for c in range(n_g):
        axes_te[c].set_ylim(min(y[0] for y in ylims_te), max(y[1] for y in ylims_te))

# Row labels on the left
axes_spec[0].set_ylabel("Im($\\lambda_J$)", fontsize=9)
fig.text(0.01, 0.73, "J spectrum", va="center", ha="left", rotation=90, fontsize=9, color="0.4")
fig.text(0.01, 0.28, "$\\tau_{\\rm eff}$", va="center", ha="left", rotation=90, fontsize=9, color="0.4")
fig.suptitle(
    f"Section 1: Gain sweep  --  Gaussian W, uniform tau={TAU_FIXED},"
    f"  dt={DT},  N={N_GAUSS}",
    fontsize=12, y=0.98)
plt.show()


# %% [markdown]
# ---
# ## Section 2 -- Time-Constant Distribution Sweep
# Fixed: Gaussian W (same realization as Section 1), g = 0.9.
#
# For *uniform* tau, J and W share eigenvectors (affine map of lambda_W).
# For *heterogeneous* tau_i, J = diag(1-alpha_i) + diag(alpha_i) g W is
# diagonalized numerically -- the eigenbasis is no longer that of W.

# %%
G_TAU_SWEEP = 0.9

TAU_CONFIGS = [
    dict(label="Uniform\nτ = 1",
         dist="uniform",  tau=1.0),
    dict(label="Power-law\nβ = 0.25",
         dist="powerlaw", lo=0.1, hi=20.0, beta=0.25),
    dict(label="Power-law\nβ = 0.5",
         dist="powerlaw", lo=0.1, hi=20.0, beta=0.5),
    dict(label="Power-law\nβ = 1.0",
         dist="powerlaw", lo=0.1, hi=20.0, beta=1.0),
    dict(label="Power-law\nβ = 1.5",
         dist="powerlaw", lo=0.1, hi=20.0, beta=1.5),
    dict(label="Power-law\nβ = 2.0",
         dist="powerlaw", lo=0.1, hi=20.0, beta=2.0),
    dict(label="Gaussian\nμ=5, σ=2",
         dist="gaussian", mu=5.0, sigma=2.0, lo=0.1, hi=20.0),
]

print(f"\nSection 2: g={G_TAU_SWEEP}, Gaussian W (same realization as Section 1), dt={DT}")

results_tau = []
for cfg in TAU_CONFIGS:
    kw = {k: v for k, v in cfg.items() if k not in ("label", "dist")}
    alphas, taus = make_alphas(N_GAUSS, cfg["dist"], DT, rng, **kw)
    eigs_J = compute_J_eigs(W_gauss, G_TAU_SWEEP, alphas)
    te = tau_eff_stable(eigs_J)
    n_stable = len(te)
    print(f"  {cfg['label'].split(chr(10))[0]:40s}: "
          f"alpha in [{alphas.min():.3f}, {alphas.max():.3f}]  "
          f"{n_stable}/{N_GAUSS} stable")
    results_tau.append(dict(cfg=cfg, alphas=alphas, taus=taus, eigs_J=eigs_J,
                            tau_eff=np.clip(te, None, TAU_CAP), n_stable=n_stable,
                            n_unstable=N_GAUSS - n_stable))

all_te_tau  = np.concatenate([r["tau_eff"] for r in results_tau if r["n_stable"] > 10])
ts_xmin_tau = all_te_tau.min()
ts_xmax_tau = TAU_CAP

n_tau    = len(TAU_CONFIGS)
colors_tau = plt.get_cmap(CMAP_SEC2)(np.linspace(0.15, 0.90, n_tau))

# Layout: 4 rows x n_tau cols
#   row 0: tau PDF  |  row 1: alpha PDF  |  row 2: J eigenspectrum  |  row 3: tau_eff
fig = plt.figure(figsize=(n_tau * 2.8, 12.5))
gs2 = GridSpec(4, n_tau, figure=fig,
               height_ratios=[0.40, 0.40, 1.00, 0.65],
               hspace=0.07, wspace=0.06,
               left=0.07, right=0.99, top=0.94, bottom=0.06)
axes_tpdf = [fig.add_subplot(gs2[0, c]) for c in range(n_tau)]
axes_apdf = [fig.add_subplot(gs2[1, c]) for c in range(n_tau)]
axes_spec = [fig.add_subplot(gs2[2, c]) for c in range(n_tau)]
axes_te   = [fig.add_subplot(gs2[3, c]) for c in range(n_tau)]

for col, (res, color) in enumerate(zip(results_tau, colors_tau)):
    cfg    = res["cfg"]
    taus_r = res["taus"]
    alp_r  = res["alphas"]
    eigs_J = res["eigs_J"]
    te     = res["tau_eff"]
    # Row 0: tau distribution
    ax = axes_tpdf[col]
    ax.hist(taus_r, bins=50, density=True, color=color, alpha=0.85, edgecolor="none")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_title(cfg["label"], fontsize=8, color=color, pad=3)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.25)
    ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("Density", fontsize=8)
    else:
        ax.set_yticklabels([])

    # Row 1: alpha distribution
    ax = axes_apdf[col]
    ax.hist(alp_r, bins=50, density=True, color=color, alpha=0.85, edgecolor="none")
    ax.set_yscale("log")
    ax.annotate(f"mean={alp_r.mean():.3f}", xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.25)
    ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("Density", fontsize=8)
    else:
        ax.set_yticklabels([])

    # Row 2: J eigenspectrum
    ax = axes_spec[col]
    ax.scatter(eigs_J.real, eigs_J.imag, s=1, alpha=0.3, color=color, rasterized=True)
    ax.plot(np.cos(_theta), np.sin(_theta), "r--", lw=1.1, alpha=0.85)
    ax.axvline(0, color="gray", lw=0.4, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
    ax.set_xlim(-xy_lim_g, xy_lim_g); ax.set_ylim(-xy_lim_g, xy_lim_g)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2)
    ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("Im($\\lambda_J$)", fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.annotate(f"{res['n_stable']}/{N_GAUSS} stable",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))

    # Row 3: tau_eff distribution
    ax = axes_te[col]
    if res["n_stable"] > 10:
        bins_t = np.logspace(np.log10(ts_xmin_tau + 1e-3), np.log10(ts_xmax_tau), N_BINS)
        ax.hist(te, bins=bins_t, density=True, alpha=0.8, color=color, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(ts_xmin_tau * 0.9, ts_xmax_tau * 1.1)
    ax.set_xlabel("$\\tau_{\\rm eff}$", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel("Density", fontsize=8)
    else:
        ax.set_yticklabels([])

# Sync y-limits for tau_eff row
ylims_t2 = [axes_te[c].get_ylim() for c in range(n_tau)
            if results_tau[c]["n_stable"] > 10]
if ylims_t2:
    for c in range(n_tau):
        axes_te[c].set_ylim(min(y[0] for y in ylims_t2), max(y[1] for y in ylims_t2))

# Row labels
for ax, label in zip([axes_tpdf[0], axes_apdf[0], axes_spec[0], axes_te[0]],
                     ["$\\tau$ dist.", "$\\alpha$ dist.", "J spectrum", "$\\tau_{\\rm eff}$"]):
    ax.set_ylabel(label, fontsize=8)

fig.suptitle(
    f"Section 2: tau-distribution sweep  --  Gaussian W, g={G_TAU_SWEEP},"
    f"  dt={DT},  N={N_GAUSS}",
    fontsize=12, y=0.975)
plt.show()


# %% [markdown]
# ---
# ## Section 2b -- Schur-Diagonal Spectral Surgery
#
# Start from the **same** J as Section 2 (Gaussian W, uniform τ=1, g=0.9).
# Real Schur decomposition: J = Q H Qᵀ, Q orthogonal, H upper quasi-triangular.
#
# For each power-law exponent β, draw N τ_eff values from p(τ) ∝ τ^{-β} on
# [τ_lo, τ_hi], convert to |λ_new| = exp(−1/τ), and *replace the eigenvalue
# magnitudes* of every diagonal block of H while keeping:
#   • the phases of complex eigenvalues (angular structure of the spectrum),
#   • all off-diagonal entries of H (upper-triangular part above the blocks),
#   • Q (the eigenbasis is completely unchanged).
#
# The resulting J̃ = Q H̃ Qᵀ is a valid real matrix with the prescribed
# spectral distribution, but the **same invariant subspaces** as the original J.
# This is fundamentally different from Section 2, where changing τ_i per-neuron
# reshuffles the entire eigenbasis.

# %%
from scipy.linalg import schur as _scipy_schur_s2b

TAU_LO_S2B  = 0.1
TAU_HI_S2B  = 20.0
BETAS_S2B   = [0.25, 0.5, 1.0, 1.5, 2.0]


def _sample_powerlaw_taus(N, beta, lo, hi, rng_):
    """CDF-inversion sample from p(τ) ∝ τ^{-β} on [lo, hi]."""
    u = rng_.uniform(0.0, 1.0, N)
    if abs(beta - 1.0) < 1e-6:
        return lo * (hi / lo) ** u
    exp = 1.0 - beta
    return (lo**exp + u * (hi**exp - lo**exp)) ** (1.0 / exp)


def _schur_spectral_surgery(J, taus_new):
    """
    Replace eigenvalue magnitudes of J's Schur diagonal blocks with
    exp(-1/tau) for each tau in taus_new (sorted descending to match
    Schur blocks sorted by |lambda|).

    Returns (J_tilde, H_tilde, Q, eigs_original, eigs_new).
    """
    H, Q = _scipy_schur_s2b(J, output="real")
    N    = H.shape[0]

    # Sort new taus descending so the largest tau_eff (slowest mode) maps to
    # the largest |lambda| block — matching scipy's default Schur ordering.
    mags_new = np.exp(-1.0 / np.sort(taus_new)[::-1])

    H_tilde  = H.copy()
    eigs_old, eigs_new_list = [], []
    i = k = 0   # k indexes into mags_new

    while i < N and k < len(mags_new):
        if i < N - 1 and abs(H[i + 1, i]) > 1e-12:
            # 2×2 conjugate-pair block
            a, b   = H[i, i], H[i, i + 1]
            mag_old = np.sqrt(a**2 + b**2)
            eigs_old.append(complex(a,  b))
            eigs_old.append(complex(a, -b))

            mag_new = mags_new[k]; k += 1
            s = mag_new / mag_old if mag_old > 1e-12 else 0.0
            H_tilde[i,     i    ] =  s * a
            H_tilde[i,     i + 1] =  s * b
            H_tilde[i + 1, i    ] =  s * H[i + 1, i]   # preserves sign of off-diag
            H_tilde[i + 1, i + 1] =  s * a
            eigs_new_list.extend([complex(s*a, s*b), complex(s*a, -s*b)])
            i += 2
        else:
            # 1×1 real eigenvalue block
            r = H[i, i]
            eigs_old.append(complex(r, 0))
            mag_new = mags_new[k]; k += 1
            r_new   = np.sign(r) * mag_new if abs(r) > 1e-12 else mag_new
            H_tilde[i, i] = r_new
            eigs_new_list.append(complex(r_new, 0))
            i += 1

    J_tilde = Q @ H_tilde @ Q.T
    return J_tilde, H_tilde, Q, np.array(eigs_old), np.array(eigs_new_list)


# ── Build baseline J (uniform tau=1, same W and g as Section 2) ──────────────
_alpha0_s2b = 1.0 - np.exp(-DT / 1.0)
_J0_s2b     = (1 - _alpha0_s2b) * np.eye(N_GAUSS) + _alpha0_s2b * G_TAU_SWEEP * W_gauss
_eigs0_s2b  = np.linalg.eigvals(_J0_s2b)
_te0_s2b    = tau_eff_stable(_eigs0_s2b)

# ── Run surgery for each β ────────────────────────────────────────────────────
_results_s2b = []
for beta in BETAS_S2B:
    _taus_new = _sample_powerlaw_taus(N_GAUSS, beta, TAU_LO_S2B, TAU_HI_S2B,
                                      np.random.default_rng(SEED))
    _Jt, _Ht, _Q, _eo, _en = _schur_spectral_surgery(_J0_s2b, _taus_new)
    _te_new = tau_eff_stable(_en)
    _results_s2b.append(dict(
        beta=beta, taus_new=_taus_new,
        J_tilde=_Jt, eigs_new=_en, tau_eff=_te_new,
    ))
    print(f"  β={beta:.2f}:  |λ| range [{np.abs(_en).min():.3f}, {np.abs(_en).max():.3f}]"
          f"  τ_eff range [{_te_new.min():.1f}, {_te_new.max():.1f}]  "
          f"({len(_te_new)}/{N_GAUSS} stable)")

# ── Plot ──────────────────────────────────────────────────────────────────────
_n_s2b     = len(BETAS_S2B)
_colors_s2b = plt.get_cmap(CMAP_SEC2)(np.linspace(0.15, 0.90, _n_s2b))
_all_te_s2b = np.concatenate([r["tau_eff"] for r in _results_s2b])
_te_lo_s2b  = _all_te_s2b.min()
_te_hi_s2b  = TAU_CAP

# 3 rows × n_beta cols:
#   row 0 — inserted τ_eff histogram (the power-law draws)
#   row 1 — J̃ eigenspectrum (grey = original J, colour = J̃)
#   row 2 — τ_eff distribution recovered from J̃ eigenvalues
fig_s2b = plt.figure(figsize=(_n_s2b * 2.8, 9))
gs_s2b  = GridSpec(3, _n_s2b, figure=fig_s2b,
                   height_ratios=[0.55, 1.0, 0.55],
                   hspace=0.08, wspace=0.06,
                   left=0.07, right=0.99, top=0.93, bottom=0.06)
_axes_tau  = [fig_s2b.add_subplot(gs_s2b[0, c]) for c in range(_n_s2b)]
_axes_spec = [fig_s2b.add_subplot(gs_s2b[1, c]) for c in range(_n_s2b)]
_axes_te   = [fig_s2b.add_subplot(gs_s2b[2, c]) for c in range(_n_s2b)]

_xy_s2b = xy_lim_g   # reuse spectral plot limit from Section 1

for col, (res, clr) in enumerate(zip(_results_s2b, _colors_s2b)):
    beta = res["beta"]

    # Row 0: inserted τ_eff distribution
    ax = _axes_tau[col]
    _bins_in = np.logspace(np.log10(TAU_LO_S2B * 0.9), np.log10(TAU_HI_S2B * 1.1), 50)
    ax.hist(res["taus_new"], bins=_bins_in, density=True,
            color=clr, alpha=0.85, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"β = {beta}", fontsize=9, color=clr, pad=3)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.25)
    ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("Inserted\n$\\tau$ dist.", fontsize=8)
    else:
        ax.set_yticklabels([])

    # Row 1: eigenspectrum — original J (grey) + J̃ (colour)
    ax = _axes_spec[col]
    ax.scatter(_eigs0_s2b.real, _eigs0_s2b.imag,
               s=1, alpha=0.2, color="gray", rasterized=True, zorder=1)
    ax.scatter(res["eigs_new"].real, res["eigs_new"].imag,
               s=1, alpha=0.35, color=clr, rasterized=True, zorder=2)
    ax.plot(np.cos(_theta), np.sin(_theta), "r--", lw=1.0, alpha=0.8)
    ax.axvline(0, color="gray", lw=0.4, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
    ax.set_xlim(-_xy_s2b, _xy_s2b); ax.set_ylim(-_xy_s2b, _xy_s2b)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    if col == 0:
        ax.set_ylabel("$\\lambda_{J̃}$ spectrum\n(grey = original J)", fontsize=8)
        ax.set_yticklabels([f"{v:.1f}" for v in
                            np.linspace(-_xy_s2b, _xy_s2b, 5)], fontsize=6)

    # Row 2: τ_eff of J̃
    ax = _axes_te[col]
    if len(res["tau_eff"]) > 5:
        _bins_te = np.logspace(np.log10(_te_lo_s2b + 1e-3),
                               np.log10(_te_hi_s2b), N_BINS)
        ax.hist(res["tau_eff"], bins=_bins_te, density=True,
                color=clr, alpha=0.8, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(_te_lo_s2b * 0.9, _te_hi_s2b * 1.1)
    ax.set_xlabel("$\\tau_{\\rm eff}$", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel("Recovered\n$\\tau_{\\rm eff}$ dist.", fontsize=8)
    else:
        ax.set_yticklabels([])

# Sync τ_eff y-limits across columns
_ylims_te_s2b = [_axes_te[c].get_ylim() for c in range(_n_s2b)
                 if len(_results_s2b[c]["tau_eff"]) > 5]
if _ylims_te_s2b:
    _yl_lo = min(y[0] for y in _ylims_te_s2b)
    _yl_hi = max(y[1] for y in _ylims_te_s2b)
    for c in range(_n_s2b):
        _axes_te[c].set_ylim(_yl_lo, _yl_hi)

fig_s2b.suptitle(
    f"Section 2b: Schur-diagonal spectral surgery  —  "
    f"Gaussian W, g={G_TAU_SWEEP}, dt={DT}, N={N_GAUSS}\n"
    r"J̃ = Q H̃ Qᵀ  |  same eigenbasis Q as uniform-τ J  |  "
    r"$|\lambda_k|$ replaced by exp(−1/τ),  τ ~ τ$^{-β}$",
    fontsize=11, y=0.975)
plt.show()


# %% [markdown]
# ---
# ## Section 3 -- Weight Distribution Sweep
# Fixed: g = 0.9, uniform tau = 1.
#
# Gaussian W (reference) vs Levy-stable W with varying stability index alpha_s.
# "Stable" normalization (scale = 1/N^{1/alpha_s}) keeps spectral radius O(1) across
# all alpha_s and recovers N(0, 1/N) in the Gaussian limit (alpha_s = 2).
# Because alpha is uniform, lambda_J = (1-alpha) + alpha*g*lambda_W -- only W is
# diagonalized, not J explicitly.
#
# See notebook 1 section 1.5 for the continuous-time analogue.

# %%
G_W_SWEEP   = 0.9
TAU_W       = 1.0
ALPHA_W     = 1.0 - np.exp(-DT / TAU_W)
LEVY_ALPHAS = [0.5, 1.0, 1.5, 2.0]   # 2.0 = Gaussian limit

print(f"\nSection 3: g={G_W_SWEEP}, tau={TAU_W}, dt={DT}  =>  alpha={ALPHA_W:.4f}")
print(f"Generating Levy-stable W matrices for alpha_s in {LEVY_ALPHAS}  (N={N_LEVY}) ...")

results_W = []

# Gaussian reference at N_LEVY for fair comparison
W_ref      = generate_W_gauss(N_LEVY, rng)
eigs_J_ref = compute_J_eigs(W_ref, G_W_SWEEP, ALPHA_W)
te_ref     = tau_eff_stable(eigs_J_ref)
results_W.append(dict(
    label="Gaussian  N(0, 1/N)",
    alpha_stab=None, W=W_ref,
    eigs_J=eigs_J_ref, tau_eff=np.clip(te_ref, None, TAU_CAP),
    n_stable=len(te_ref), n_unstable=N_LEVY - len(te_ref),
    spectral_radius=np.abs(np.linalg.eigvals(W_ref)).max(),
    w_hi=float(np.percentile(np.abs(W_ref.ravel()), 99)),
    scale=1.0 / N_LEVY ** 0.5,
))

for alpha_s in LEVY_ALPHAS:
    scale = 1.0 / N_LEVY ** (1.0 / alpha_s)
    if alpha_s == 2.0:
        scale /= 2.0 ** 0.5
    W_l      = generate_W_levy(N_LEVY, alpha_s, rng, normalization="stable")
    eigs_J_l = compute_J_eigs(W_l, G_W_SWEEP, ALPHA_W)
    te_l     = tau_eff_stable(eigs_J_l)
    sr_W     = np.abs(np.linalg.eigvals(W_l)).max()
    w_hi     = float(np.percentile(np.abs(W_l.ravel()), 99))
    print(f"  alpha_s={alpha_s}: sr(W)={sr_W:.3f},  {len(te_l)}/{N_LEVY} stable")
    results_W.append(dict(
        label=f"Levy  alpha_s={alpha_s}",
        alpha_stab=alpha_s, W=W_l,
        eigs_J=eigs_J_l, tau_eff=np.clip(te_l, None, TAU_CAP),
        n_stable=len(te_l), n_unstable=N_LEVY - len(te_l),
        spectral_radius=sr_W, w_hi=w_hi, scale=scale,
    ))

all_te_W  = np.concatenate([r["tau_eff"] for r in results_W if r["n_stable"] > 10])
ts_xmin_W = all_te_W.min()
ts_xmax_W = TAU_CAP

n_W      = len(results_W)
colors_W = plt.get_cmap(CMAP_SEC3)(np.linspace(0.10, 0.90, n_W))

# Layout: 3 rows x n_W cols
#   row 0: W entry PDF (log)  |  row 1: J eigenspectrum  |  row 2: tau_eff
fig = plt.figure(figsize=(n_W * 3.2, 9.5))
gs3 = GridSpec(3, n_W, figure=fig,
               height_ratios=[0.42, 1.00, 0.65],
               hspace=0.07, wspace=0.06,
               left=0.07, right=0.99, top=0.94, bottom=0.06)
axes_wpdf = [fig.add_subplot(gs3[0, c]) for c in range(n_W)]
axes_spec = [fig.add_subplot(gs3[1, c]) for c in range(n_W)]
axes_te   = [fig.add_subplot(gs3[2, c]) for c in range(n_W)]

for col, (res, color) in enumerate(zip(results_W, colors_W)):
    scale  = res["scale"]
    w_hi   = res["w_hi"]
    eigs_J = res["eigs_J"]
    te     = res["tau_eff"]

    # Row 0: W entry PDF (log-scale to show heavy tails)
    ax = axes_wpdf[col]
    x_w = np.linspace(-w_hi * 1.02, w_hi * 1.02, 800)
    if res["alpha_stab"] is None:
        y_w = stats.norm.pdf(x_w, 0, scale)
    else:
        y_w = levy_stable.pdf(x_w, res["alpha_stab"], beta=0, loc=0, scale=scale)
    valid = y_w > 1e-12
    ax.plot(x_w[valid], y_w[valid], color=color, lw=1.5)
    ax.fill_between(x_w[valid], y_w[valid], alpha=0.2, color=color)
    ax.set_yscale("log")
    ax.set_title(res["label"], fontsize=8, color=color, pad=3)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.25)
    ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("Density", fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.annotate(f"sr(W)={res['spectral_radius']:.3f}",
                xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))

    # Row 1: J eigenspectrum
    ax = axes_spec[col]
    ax.scatter(eigs_J.real, eigs_J.imag, s=2, alpha=0.35, color=color, rasterized=True)
    ax.plot(np.cos(_theta), np.sin(_theta), "r--", lw=1.1, alpha=0.85)
    ax.axvline(0, color="gray", lw=0.4, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
    ax.set_xlim(-xy_lim_g, xy_lim_g); ax.set_ylim(-xy_lim_g, xy_lim_g)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2)
    ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("Im($\\lambda_J$)", fontsize=8)
    else:
        ax.set_yticklabels([])
    if res["n_unstable"] > 0:
        ax.annotate(f"{res['n_unstable']} unstable",
                    xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))

    # Row 2: tau_eff distribution
    ax = axes_te[col]
    if res["n_stable"] > 10:
        bins_W = np.logspace(np.log10(ts_xmin_W + 1e-3), np.log10(ts_xmax_W), N_BINS)
        ax.hist(te, bins=bins_W, density=True, alpha=0.8, color=color, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(ts_xmin_W * 0.9, ts_xmax_W * 1.1)
    ax.set_xlabel("$\\tau_{\\rm eff}$", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel("Density", fontsize=8)
    else:
        ax.set_yticklabels([])

# Sync y-limits for tau_eff row
ylims_W3 = [axes_te[c].get_ylim() for c in range(n_W)
            if results_W[c]["n_stable"] > 10]
if ylims_W3:
    for c in range(n_W):
        axes_te[c].set_ylim(min(y[0] for y in ylims_W3), max(y[1] for y in ylims_W3))

fig.suptitle(
    f"Section 3: W-distribution sweep  --  g={G_W_SWEEP}, uniform tau={TAU_W},"
    f"  dt={DT},  N={N_LEVY}  (stable normalization)",
    fontsize=12, y=0.975)
plt.show()
