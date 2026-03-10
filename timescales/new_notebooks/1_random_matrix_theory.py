# %% [markdown]
# # Random Matrix Theory: Eigenvalue Spectra and Timescale Distributions
#
# **Goal**: Compute and visualize effective timescales of linear recurrent networks
# for different coupling strengths $g$, timescales $\tau$, and weight distributions
# (Gaussian, Levy-stable) — without simulation.
#
# Connectivity matrix $W_{ij} \sim \mathcal{N}(0, 1/N)$.  By the circular law,
# eigenvalues of $W$ fill the unit disk as $N \to \infty$.
#
# Linearized dynamics: $\tau \dot{x} = -x + gWx = (gW - I)x$
#
# Jacobian eigenvalues: $\lambda_M = \frac{1}{\tau}(g\lambda_W - 1)$,
# effective timescale: $\tau_{\rm eff} = \frac{-1}{\rm Re(\lambda_M)} = \frac{\tau}{1 - g\,{\rm Re}(\lambda_W)}$


# %% [markdown]
# ## Setup


# %%
import os
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import powerlaw
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from scipy.stats import levy_stable
from typing import Tuple
from numpy.typing import NDArray

import torch
import torch.nn as nn

# Add timescales package to path
_gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()
sys.path.insert(0, _gitroot)
sys.path.insert(0, os.path.join(_gitroot, "timescales"))

from rnns.multitimescale_rnn import MultiTimescaleRNN

SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
print(f"Random seed: {SEED}")


# %%
# --- Model infrastructure ---

def build_model(
    n_neurons: int, g: float, tau: float, dt: float,
    wrec_init: str = "normal_scaled",
    activation: type = nn.Identity,
    dynamics_type: str = "rate",
) -> MultiTimescaleRNN:
    """Build an untrained MultiTimescaleRNN for null-task analysis."""
    return MultiTimescaleRNN(
        input_size=1, hidden_size=n_neurons, output_size=1, dt=dt,
        timescales_config={"type": "discrete", "values": [tau]},
        learn_timescales=False, activation=activation, wrec_init=wrec_init,
        recurrent_gain=g, zero_diag_wrec=False, dynamics_type=dynamics_type,
        alpha_parameterization="linear", noise_std=0.0,
    )


def get_W(model: MultiTimescaleRNN) -> NDArray:
    """Extract W_rec as a numpy array (detached copy)."""
    return model.rnn_step.W_rec.weight.detach().numpy().copy()


def generate_W(n_neurons: int, wrec_init: str = "normal_scaled") -> NDArray:
    """Generate W via MultiTimescaleRNN initialization (default: Gaussian N(0,1/N))."""
    return get_W(build_model(n_neurons, g=1.0, tau=1.0, dt=0.1, wrec_init=wrec_init))


def generate_W_levy(
    n_neurons: int,
    alpha_stab: float,
    normalization: str = "rnn",
) -> NDArray:
    """
    W_ij ~ S(alpha_stab, beta=0, loc=0, scale)  (symmetric alpha-stable).

    normalization="rnn"    → scale = 1/sqrt(N)
                             Matches MultiTimescaleRNN wrec_init='levy_stable'.
                             Spectral radius grows as N^(1/alpha-1/2) for alpha < 2.
    normalization="stable" → scale = 1/N^(1/alpha), with √2 correction at alpha=2
                             so that the Gaussian limit gives exactly N(0, 1/N).
                             Canonical for alpha-stable random matrices.
                             Spectral radius stays ~O(1) across all alpha.
    """
    if normalization == "rnn":
        scale = 1.0 / n_neurons ** 0.5
    elif normalization == "stable":
        scale = 1.0 / n_neurons ** (1.0 / alpha_stab)
        if alpha_stab == 2.0:
            # levy_stable(alpha=2, scale=σ) ~ N(0, 2σ²); divide by √2 so variance = 1/N
            scale /= 2.0 ** 0.5
    else:
        raise ValueError(f"Unknown normalization: {normalization!r}. Use 'rnn' or 'stable'.")
    return levy_stable.rvs(
        alpha_stab, beta=0, loc=0, scale=scale, size=(n_neurons, n_neurons),
    ).astype(np.float32)


print("Model infrastructure defined.")


# %%
# --- Analysis helpers ---

def get_jacobian_eigenvalues(eigenvalues_W: NDArray, g: float, tau: float) -> NDArray:
    """lambda_M = (g * lambda_W - 1) / tau"""
    return (g * eigenvalues_W - 1.0) / tau


def get_effective_timescales(eigenvalues_M: NDArray) -> NDArray:
    """tau_eff = -1/Re(lambda_M) for stable modes (Re < 0)."""
    real_parts = np.real(eigenvalues_M)
    return -1.0 / real_parts[real_parts < 0]


def get_stable_indices(eigenvalues_M: NDArray) -> NDArray:
    return np.where(np.real(eigenvalues_M) < 0)[0]


def get_real_eigenvalue_indices(eigenvalues_M: NDArray, tol: float = 1e-8) -> NDArray:
    return np.where(np.abs(np.imag(eigenvalues_M)) < tol)[0]


def fit_powerlaw_slope(timescales: NDArray, n_bins: int = 50):
    log_tau = np.log10(timescales)
    bins = np.logspace(log_tau.min(), log_tau.max(), n_bins + 1)
    hist, bin_edges = np.histogram(timescales, bins=bins, density=True)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    valid = hist > 0
    slope, intercept, r_val, _, _ = stats.linregress(
        np.log10(bin_centers[valid]), np.log10(hist[valid])
    )
    return slope, intercept, bin_centers, hist, r_val**2


def find_conjugate(idx, eigs, tol=1e-8):
    target = np.conj(eigs[idx])
    diffs = np.abs(eigs - target)
    diffs[idx] = np.inf
    partner = np.argmin(diffs)
    return partner if diffs[partner] < tol else None


print("Analysis helpers defined.")


# %% [markdown]
# ## Configuration


# %%
config = {
    "n_neurons": 3000,
    "tau": 1.0,
    "g_values": [0.3, 0.6, 0.9, 0.99, 1.0, 1.2, 2],
    "n_bins": 50,
}


# %% [markdown]
# ---
# ## 1.1 Gaussian W and the Circular Law


# %%
N   = config["n_neurons"]
tau = config["tau"]
g_values = config["g_values"]

W = generate_W(N)
eigenvalues_W = np.linalg.eigvals(W)

print(f"W shape: {W.shape}")
print(f"Spectral radius: {np.abs(eigenvalues_W).max():.4f}  (expected ~1.0)")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(np.real(eigenvalues_W), np.imag(eigenvalues_W), s=2, alpha=0.4, color='steelblue')
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=1.5, label='Unit circle')
ax.set_xlabel(r'Re($\lambda_W$)'); ax.set_ylabel(r'Im($\lambda_W$)')
ax.set_title(f'W Eigenvalue Spectrum (N={N})\nCircular law: eigenvalues fill unit disk')
ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()


# %% [markdown]
# ---
# ## 1.2 Jacobian Eigenspectra and Timescales for Multiple g (Gaussian)


# %%
n_g    = len(g_values)
n_rows = n_g

# --- Gaussian results ---
results = {}
for g in g_values:
    eigenvalues_M = get_jacobian_eigenvalues(eigenvalues_W, g, tau)
    timescales    = get_effective_timescales(eigenvalues_M)
    n_stable, n_unstable = len(timescales), N - len(timescales)
    results[g] = dict(eigenvalues_M=eigenvalues_M, timescales=timescales,
                      n_stable=n_stable, n_unstable=n_unstable)
    if n_stable > 0:
        print(f"Gaussian  g={g}: {n_stable}/{N} stable, tau_eff in [{timescales.min():.3f}, {timescales.max():.3f}]")
    else:
        print(f"Gaussian  g={g}: 0/{N} stable")

# --- Shared limits ---
all_eig = np.concatenate([results[g]["eigenvalues_M"] for g in g_values])
xy_lim  = max(np.abs(np.real(all_eig)).max(), np.abs(np.imag(all_eig)).max()) * 1.08

stable_ts_list = [results[g]["timescales"] for g in g_values if len(results[g]["timescales"]) > 10]
if stable_ts_list:
    all_ts = np.concatenate(stable_ts_list)
    ts_xmin, ts_xmax = all_ts.min(), all_ts.max()
else:
    ts_xmin, ts_xmax = 0.1, 100.0

colors_gauss = plt.cm.plasma(np.linspace(0.1, 0.85, n_g))   

# Gaussian weight PDF: W_ij ~ N(0, 1/N)
sigma_W = 1.0 / np.sqrt(N)
x_gauss = np.linspace(-4.5 * sigma_W, 4.5 * sigma_W, 500)
y_gauss = stats.norm.pdf(x_gauss, 0, sigma_W)

# --- Layout: 3 columns (weight PDF | eigenspectrum | timescale histogram) ---
fig = plt.figure(figsize=(20, 4.5 * n_rows))
outer   = GridSpec(1, 3, figure=fig, width_ratios=[0.8, 1, 1.1], wspace=0.45,
                   left=0.06, right=0.97, top=0.95, bottom=0.03)
pdf_gs   = GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=outer[0], hspace=0.3)
left_gs  = GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=outer[1], hspace=0.3)
right_gs = GridSpecFromSubplotSpec(n_rows, 1, subplot_spec=outer[2], hspace=0.3)

axes_pdf = [fig.add_subplot(pdf_gs[row])   for row in range(n_rows)]
axes_eig = [fig.add_subplot(left_gs[row])  for row in range(n_rows)]
axes_ts  = [fig.add_subplot(right_gs[row]) for row in range(n_rows)]

# --- Gaussian rows ---
for row, (g, color) in enumerate(zip(g_values, colors_gauss)):
    eig_M      = results[g]["eigenvalues_M"]
    timescales = results[g]["timescales"]
    n_stable   = results[g]["n_stable"]
    n_unstable = results[g]["n_unstable"]

    ax = axes_pdf[row]
    ax.plot(x_gauss, y_gauss, color=color, linewidth=1.5)
    ax.fill_between(x_gauss, y_gauss, alpha=0.15, color=color)
    ax.set_ylabel('Density')
    ax.annotate(r'$\mathcal{N}(0,\,1/N)$', xy=(0.05, 0.97), xycoords='axes fraction',
                va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))
    ax.grid(True, alpha=0.3); ax.set_xticklabels([])

    ax = axes_eig[row]
    ax.scatter(np.real(eig_M), np.imag(eig_M), s=2, alpha=0.4, color=color, rasterized=True)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlim(-xy_lim, xy_lim); ax.set_ylim(-xy_lim, xy_lim)
    ax.set_aspect('equal', adjustable='box'); ax.set_ylabel(r'Im($\lambda_M$)')
    ax.annotate(f'Gaussian  ·  $g = {g}$  ·  {n_unstable} unstable',
                xy=(0.03, 0.97), xycoords='axes fraction', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.grid(True, alpha=0.3); ax.set_xticklabels([])

    ax = axes_ts[row]
    if n_stable > 10:
        bins = np.logspace(np.log10(ts_xmin), np.log10(ts_xmax), config["n_bins"])
        ax.hist(timescales, bins=bins, density=True, alpha=0.75, edgecolor='white', color=color)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(ts_xmin * 0.9, ts_xmax * 1.1); ax.set_ylabel('Density')
    ax.annotate(f'Gaussian  ·  $g = {g}$  ·  {n_stable} stable',
                xy=(0.03, 0.97), xycoords='axes fraction', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.grid(True, alpha=0.3); ax.set_xticklabels([])


# Sync timescale y-limits
ylims = [ax.get_ylim() for ax in axes_ts]
for ax in axes_ts:
    ax.set_ylim(min(y[0] for y in ylims), max(y[1] for y in ylims))

fig.suptitle(rf'$M = \frac{{1}}{{\tau}}(gW - I)$,  $\tau={tau}$,  $N={N}$', fontsize=14)
plt.show()


# %% [markdown]
# ---
# ## 1.3 Overlay Timescale Distributions and Power-Law Fits


# %%
fig, axes = plt.subplots(n_g, 1, figsize=(8, 3.5 * n_g), sharex=True, sharey=True)
colors = plt.cm.plasma(np.linspace(0.1, 0.85, n_g))

for ax, color, g in zip(axes, colors, g_values):
    timescales = results[g]["timescales"]
    if len(timescales) < 10:
        ax.annotate('insufficient stable modes', xy=(0.5, 0.5),
                    xycoords='axes fraction', ha='center', fontsize=11)
        continue

    fit   = powerlaw.Fit(timescales, xmax=None, discrete=False, verbose=False)
    gamma = fit.alpha
    theta = fit.xmin
    N_theta = (timescales >= theta).sum()

    log_tau = np.log10(timescales)
    bins = np.logspace(log_tau.min(), log_tau.max(), config["n_bins"] + 1)
    hist, bin_edges = np.histogram(timescales, bins=bins, density=True)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    valid = hist > 0
    ax.loglog(bin_centers[valid], hist[valid], 'o', color=color, markersize=5, alpha=0.65)

    tau_tail = np.logspace(np.log10(theta), np.log10(timescales.max()), 200)
    f_tail   = N_theta / len(timescales)
    ax.loglog(tau_tail, f_tail * fit.power_law.pdf(tau_tail), '--', color='black',
              linewidth=1, alpha=0.9,
              label=rf'$\gamma = {gamma:.2f}$,  $\theta = {theta:.2f}$,  $N_\theta = {N_theta}$')
    ax.axvline(theta, color=color, linewidth=1, linestyle=':', alpha=0.7)
    ax.set_ylabel('Density'); ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.annotate(f'$g = {g}$', xy=(0.02, 0.97), xycoords='axes fraction', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

axes[-1].set_xlabel(r'$\tau_{\mathrm{eff}}$', fontsize=13)
fig.suptitle(r'Timescale Distributions — Power-Law Fits (MLE + KS cutoff)', fontsize=13, y=1.01)
plt.tight_layout(); plt.show()


# %% [markdown]
# ---
# ## 1.4 Jacobian Eigenspectra for Multiple τ Values (fixed g)


# %%
g_fixed    = 0.9
tau_values = [0.2, 0.5, 1.0, 1.5]
n_tau      = len(tau_values)

results_tau = {}
for tau_val in tau_values:
    eigenvalues_M = get_jacobian_eigenvalues(eigenvalues_W, g_fixed, tau_val)
    timescales    = get_effective_timescales(eigenvalues_M)
    n_stable      = len(timescales)
    results_tau[tau_val] = dict(eigenvalues_M=eigenvalues_M, timescales=timescales,
                                n_stable=n_stable, n_unstable=N - n_stable)
    if n_stable > 0:
        print(f"tau={tau_val}: {n_stable}/{N} stable, tau_eff in [{timescales.min():.3f}, {timescales.max():.3f}]")
    else:
        print(f"tau={tau_val}: 0/{N} stable")

all_eig_tau = np.concatenate([results_tau[t]["eigenvalues_M"] for t in tau_values])
xy_lim_tau  = max(np.abs(np.real(all_eig_tau)).max(), np.abs(np.imag(all_eig_tau)).max()) * 1.08

stable_ts_list_tau = [results_tau[t]["timescales"] for t in tau_values if results_tau[t]["n_stable"] > 10]
if stable_ts_list_tau:
    all_ts_tau = np.concatenate(stable_ts_list_tau)
    ts_xmin_tau, ts_xmax_tau = all_ts_tau.min(), all_ts_tau.max()
else:
    ts_xmin_tau, ts_xmax_tau = 0.1, 100.0

colors_tau = plt.cm.magma(np.linspace(0.15, 0.85, n_tau))

# Gaussian weight PDF: W_ij ~ N(0, 1/N) (same W as above; tau doesn't change the weights)
sigma_W_tau = 1.0 / np.sqrt(N)
x_gauss_tau = np.linspace(-4.5 * sigma_W_tau, 4.5 * sigma_W_tau, 500)
y_gauss_tau = stats.norm.pdf(x_gauss_tau, 0, sigma_W_tau)

# --- Layout: 3 columns (weight PDF | eigenspectrum | timescale histogram) ---
fig = plt.figure(figsize=(20, 4.5 * n_tau))
outer_tau = GridSpec(1, 3, figure=fig, width_ratios=[0.8, 1, 1.1], wspace=0.45,
                     left=0.06, right=0.97, top=0.96, bottom=0.04)
pdf_gs_tau   = GridSpecFromSubplotSpec(n_tau, 1, subplot_spec=outer_tau[0], hspace=0.3)
left_gs_tau  = GridSpecFromSubplotSpec(n_tau, 1, subplot_spec=outer_tau[1], hspace=0.3)
right_gs_tau = GridSpecFromSubplotSpec(n_tau, 1, subplot_spec=outer_tau[2], hspace=0.3)
axes_pdf_tau = [fig.add_subplot(pdf_gs_tau[row])   for row in range(n_tau)]
axes_eig_tau = [fig.add_subplot(left_gs_tau[row])  for row in range(n_tau)]
axes_ts_tau  = [fig.add_subplot(right_gs_tau[row]) for row in range(n_tau)]

for row, (tau_val, color) in enumerate(zip(tau_values, colors_tau)):
    eig_M      = results_tau[tau_val]["eigenvalues_M"]
    timescales = results_tau[tau_val]["timescales"]
    n_stable   = results_tau[tau_val]["n_stable"]
    n_unstable = results_tau[tau_val]["n_unstable"]

    ax = axes_pdf_tau[row]
    ax.plot(x_gauss_tau, y_gauss_tau, color=color, linewidth=1.5)
    ax.fill_between(x_gauss_tau, y_gauss_tau, alpha=0.15, color=color)
    ax.set_ylabel('Density')
    ax.annotate(r'$\mathcal{N}(0,\,1/N)$', xy=(0.05, 0.97), xycoords='axes fraction',
                va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))
    ax.grid(True, alpha=0.3)
    if row < n_tau - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$W_{ij}$')

    ax = axes_eig_tau[row]
    ax.scatter(np.real(eig_M), np.imag(eig_M), s=2, alpha=0.4, color=color, rasterized=True)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlim(-xy_lim_tau, xy_lim_tau); ax.set_ylim(-xy_lim_tau, xy_lim_tau)
    ax.set_aspect('equal', adjustable='box'); ax.set_ylabel(r'Im($\lambda_M$)')
    ax.annotate(f'$\\tau = {tau_val}$  ·  {n_unstable} unstable',
                xy=(0.03, 0.97), xycoords='axes fraction', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.grid(True, alpha=0.3)
    if row < n_tau - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'Re($\lambda_M$)')

    ax = axes_ts_tau[row]
    if n_stable > 10:
        bins = np.logspace(np.log10(ts_xmin_tau), np.log10(ts_xmax_tau), config["n_bins"])
        ax.hist(timescales, bins=bins, density=True, alpha=0.75, edgecolor='white', color=color)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(ts_xmin_tau * 0.9, ts_xmax_tau * 1.1); ax.set_ylabel('Density')
    ax.annotate(f'$\\tau = {tau_val}$  ·  {n_stable} stable',
                xy=(0.68, 0.97), xycoords='axes fraction', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.grid(True, alpha=0.3)
    if row < n_tau - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$\tau_{\mathrm{eff}}$')

ylims_tau = [ax.get_ylim() for ax in axes_ts_tau]
for ax in axes_ts_tau:
    ax.set_ylim(min(y[0] for y in ylims_tau), max(y[1] for y in ylims_tau))

fig.suptitle(rf'$M = \frac{{1}}{{\tau}}(gW - I)$,  $g={g_fixed}$,  $N={N}$', fontsize=14)
plt.show()


# %% [markdown]
# ---
# ## 1.5 Levy-Stable Eigenspectra for Varying α
#
# For i.i.d. entries $W_{ij} \sim S(\alpha, 0, 0, \text{scale})$ (symmetric alpha-stable):
#
# **Two normalization conventions** (see `generate_W_levy`):
# - **`"rnn"`**: scale $= 1/\sqrt{N}$ — matches `MultiTimescaleRNN wrec_init='levy_stable'`.
#   Spectral radius grows as $N^{1/\alpha - 1/2}$ for $\alpha < 2$; eigenspectra are NOT
#   comparable across $\alpha$ values.
# - **`"stable"`**: scale $= 1/N^{1/\alpha}$ — canonical for $\alpha$-stable random matrices.
#   Spectral radius $\approx O(1)$ for all $\alpha$; eigenspectra are directly comparable.
#   At $\alpha=2$ this gives $\mathcal{N}(0, 2/N)$ rather than $\mathcal{N}(0, 1/N)$.


# %%
def _plot_levy_eigenspectra(alphas, g_levy, tau_levy, N_levy, normalization, n_bins):
    """
    3-column figure per normalization: weight PDF | Jacobian eigenspectrum (symlog) | timescale histogram.
    Horizontal colorbar inset on first eigenspectrum panel only.
    """
    norm_label = r"1/\sqrt{N}" if normalization == "rnn" else r"1/N^{1/\alpha}"
    print(f"\n─── normalization = '{normalization}' (scale = {norm_label}) ─────────────────────")

    levy_res = {}
    for alpha_s in alphas:
        scale = 1.0 / N_levy ** 0.5 if normalization == "rnn" else 1.0 / N_levy ** (1.0 / alpha_s)
        W_l    = generate_W_levy(N_levy, alpha_s, normalization=normalization)
        eigs_W = np.linalg.eigvals(W_l)
        eigs_M = get_jacobian_eigenvalues(eigs_W, g_levy, tau_levy)
        ts     = get_effective_timescales(eigs_M)
        sr     = np.abs(eigs_W).max()
        # Robust PDF x-range: 99th percentile of |entries| (avoids inf for heavy tails)
        w_hi   = float(np.percentile(np.abs(W_l.ravel()), 99))
        print(f"  alpha={alpha_s}: sr={sr:.3f}, {len(ts)}/{N_levy} stable, scale={scale:.4g}, |W|_99={w_hi:.4g}")
        levy_res[alpha_s] = dict(eigs_M=eigs_M, timescales=ts,
                                 n_stable=len(ts), spectral_radius=sr,
                                 n_unstable=N_levy - len(ts),
                                 scale=scale, w_hi=w_hi)

    n_rows_l  = len(alphas)
    all_eig_l = np.concatenate([levy_res[a]["eigs_M"] for a in alphas])
    # linthresh for symlog: 95th percentile of |Re(λ_M)| to keep the linear zone near zero tight
    linthresh_eig = float(np.percentile(np.abs(np.real(all_eig_l[np.real(all_eig_l) != 0])), 95))

    ts_lists = [levy_res[a]["timescales"] for a in alphas if levy_res[a]["n_stable"] > 10]
    if ts_lists:
        all_ts_l = np.concatenate(ts_lists)
        ts_xmin_l, ts_xmax_l = all_ts_l.min(), all_ts_l.max()
    else:
        ts_xmin_l, ts_xmax_l = 0.1, 100.0

    colors_l = plt.cm.viridis(np.linspace(0.1, 0.9, n_rows_l))

    # 3-column layout: weight PDF | eigenspectrum | timescale histogram
    fig = plt.figure(figsize=(20, 4.5 * n_rows_l))
    outer_l = GridSpec(1, 3, figure=fig, width_ratios=[0.8, 1, 1], wspace=0.45,
                       left=0.06, right=0.97, top=0.94, bottom=0.06)
    pdf_gs = GridSpecFromSubplotSpec(n_rows_l, 1, subplot_spec=outer_l[0], hspace=0.3)
    eig_gs = GridSpecFromSubplotSpec(n_rows_l, 1, subplot_spec=outer_l[1], hspace=0.3)
    ts_gs  = GridSpecFromSubplotSpec(n_rows_l, 1, subplot_spec=outer_l[2], hspace=0.3)
    axes_pdf_l = [fig.add_subplot(pdf_gs[row]) for row in range(n_rows_l)]
    axes_eig_l = [fig.add_subplot(eig_gs[row]) for row in range(n_rows_l)]
    axes_ts_l  = [fig.add_subplot(ts_gs[row])  for row in range(n_rows_l)]

    for row, (alpha_s, color) in enumerate(zip(alphas, colors_l)):
        res   = levy_res[alpha_s]
        eig_M = res["eigs_M"]
        ts    = res["timescales"]
        scale = res["scale"]
        w_hi  = res["w_hi"]

        # --- Weight distribution PDF ---
        ax = axes_pdf_l[row]
        x_pdf = np.linspace(-w_hi * 1.02, w_hi * 1.02, 800)
        y_pdf = levy_stable.pdf(x_pdf, alpha_s, beta=0, loc=0, scale=scale)
        valid_pdf = y_pdf > 1e-12
        ax.plot(x_pdf[valid_pdf], y_pdf[valid_pdf], color=color, linewidth=1.5)
        ax.fill_between(x_pdf[valid_pdf], y_pdf[valid_pdf], alpha=0.15, color=color)
        ax.set_yscale('log')
        ax.set_ylabel('Density')
        ax.annotate(rf'$\alpha={alpha_s}$,  scale$={scale:.3g}$',
                    xy=(0.05, 0.97), xycoords='axes fraction', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
        ax.grid(True, alpha=0.3)
        if row < n_rows_l - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'$W_{ij}$')

        # --- Jacobian eigenspectrum (symlog axes) ---
        ax = axes_eig_l[row]
        ax.scatter(np.real(eig_M), np.imag(eig_M), s=2, alpha=0.4, color=color, rasterized=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_xscale('symlog', linthresh=linthresh_eig)

        ax.set_yscale('symlog', linthresh=linthresh_eig)
        ax.set_ylabel(r'Im($\lambda_M$)')

        # # make x and y axes same scale
        # ax.set_xlim(-linthresh_eig, linthresh_eig)
        # ax.set_ylim(-linthresh_eig, linthresh_eig)
        ax.set_aspect('equal', adjustable='box')
        ax.annotate(rf'$\alpha={alpha_s}$  ·  sr$={res["spectral_radius"]:.2f}$  ·  {res["n_unstable"]} unstable',
                    xy=(0.03, 0.97), xycoords='axes fraction', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
        ax.grid(True, alpha=0.3)
        if row < n_rows_l - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'Re($\lambda_M$)')

        # --- Timescale histogram ---
        ax = axes_ts_l[row]
        if res["n_stable"] > 10:
            bins_l = np.logspace(np.log10(ts_xmin_l), np.log10(ts_xmax_l), n_bins)
            ax.hist(ts, bins=bins_l, density=True, alpha=0.75, edgecolor='white', color=color)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(ts_xmin_l * 0.9, ts_xmax_l * 1.1); ax.set_ylabel('Density')
        ax.annotate(rf'$\alpha={alpha_s}$  ·  {res["n_stable"]} stable',
                    xy=(0.03, 0.97), xycoords='axes fraction', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
        ax.grid(True, alpha=0.3)
        if row < n_rows_l - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'$\tau_{\mathrm{eff}}$')

    ylims_l = [ax.get_ylim() for ax in axes_ts_l]
    for ax in axes_ts_l:
        ax.set_ylim(min(y[0] for y in ylims_l), max(y[1] for y in ylims_l))

    fig.suptitle(
        rf"Levy-stable $W$,  $g={g_levy}$,  $N={N_levy}$,  scale $= {norm_label}$",
        fontsize=13,
    )
    plt.show()


# %%
levy_config = {
    "alpha_values": [0.5, 1.0, 1.5, 2.0],   # 2.0 = Gaussian limit
    "g_fixed": 0.9,
    "tau": 1.0,
    "N": 1000,    # smaller N for speed (eigendecomp only, no simulation)
    "n_bins": 50,
}

# Normalization "rnn": matches the RNN implementation; spectral radius varies with alpha
_plot_levy_eigenspectra(
    alphas=levy_config["alpha_values"],
    g_levy=levy_config["g_fixed"],
    tau_levy=levy_config["tau"],
    N_levy=levy_config["N"],
    normalization="rnn",
    n_bins=levy_config["n_bins"],
)

# Normalization "stable": spectral radius O(1) across all alpha; directly comparable spectra
_plot_levy_eigenspectra(
    alphas=levy_config["alpha_values"],
    g_levy=levy_config["g_fixed"],
    tau_levy=levy_config["tau"],
    N_levy=levy_config["N"],
    normalization="stable",
    n_bins=levy_config["n_bins"],
)

# %%
