# %% [markdown]
# # Linear Network Simulations
#
# Simulate $\tau \dot{x} = -x + gWx + \eta(t)$ (linear rate dynamics).
# Verify that fitted autocorrelation timescales match eigenvalue theory.
# Then sweep over $g$ values and $\alpha$-stable weight distributions.


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
from typing import Tuple
from numpy.typing import NDArray

import torch
import torch.nn as nn

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

def build_model(n_neurons, g, tau, dt, wrec_init="normal_scaled",
                activation=nn.Identity, dynamics_type="rate"):
    return MultiTimescaleRNN(
        input_size=1, hidden_size=n_neurons, output_size=1, dt=dt,
        timescales_config={"type": "discrete", "values": [tau]},
        learn_timescales=False, activation=activation, wrec_init=wrec_init,
        recurrent_gain=g, zero_diag_wrec=False, dynamics_type=dynamics_type,
        alpha_parameterization="linear", noise_std=0.0,
    )

def get_W(model):
    return model.rnn_step.W_rec.weight.detach().numpy().copy()

def generate_W(n_neurons, wrec_init="normal_scaled"):
    return get_W(build_model(n_neurons, g=1.0, tau=1.0, dt=0.1, wrec_init=wrec_init))

def generate_W_levy(n_neurons, alpha_stab, normalization="rnn"):
    """
    W_ij ~ S(alpha_stab, beta=0, loc=0, scale).
    normalization="rnn"    → scale = 1/sqrt(N)  [matches MultiTimescaleRNN levy_stable init]
    normalization="stable" → scale = 1/N^(1/alpha)  [spectral radius ~O(1) across alpha]
                             At alpha=2, includes √2 correction so Gaussian limit = N(0, 1/N).
    """
    if normalization == "rnn":
        scale = 1.0 / n_neurons ** 0.5
    elif normalization == "stable":
        scale = 1.0 / n_neurons ** (1.0 / alpha_stab)
        if alpha_stab == 2.0:
            scale /= 2.0 ** 0.5
    else:
        raise ValueError(f"Unknown normalization: {normalization!r}. Use 'rnn' or 'stable'.")
    return levy_stable.rvs(alpha_stab, beta=0, loc=0, scale=scale,
                           size=(n_neurons, n_neurons)).astype(np.float32)

def _inject_W(model, W):
    with torch.no_grad():
        model.rnn_step.W_rec.weight.copy_(torch.tensor(W, dtype=torch.float32))
        model.rnn_step.W_rec.bias.zero_()

def simulate_with_model(model, duration, noise_std):
    dt = model.dt
    n_steps = int(duration / dt)
    N = model.hidden_size
    model.rnn_step.noise_std = noise_std
    model.train()
    activity = np.zeros((n_steps, N), dtype=np.float32)
    hidden, zero_input = torch.randn(1, N) * 0.1, torch.zeros(1, 1)
    with torch.no_grad():
        for t in range(n_steps):
            hidden = model.rnn_step(zero_input, hidden)
            activity[t] = hidden[0].numpy()
            if t % 10000 == 0 and t > 0:
                print(f"  Step {t}/{n_steps}")
    return activity

def simulate_linear_ode(W, g, tau, dt, duration, noise_std):
    """tau*dx/dt = -x + g*W*x + noise  (linear rate)."""
    model = build_model(len(W), g=g, tau=tau, dt=dt, activation=nn.Identity, dynamics_type="rate")
    _inject_W(model, W)
    return simulate_with_model(model, duration, noise_std)

print("Model infrastructure defined.")


# %%
# --- Analysis helpers ---

def get_jacobian_eigenvalues(eigenvalues_W, g, tau):
    return (g * eigenvalues_W - 1.0) / tau

def get_effective_timescales(eigenvalues_M):
    real_parts = np.real(eigenvalues_M)
    return -1.0 / real_parts[real_parts < 0]

def get_stable_indices(eigenvalues_M):
    return np.where(np.real(eigenvalues_M) < 0)[0]

def get_real_eigenvalue_indices(eigenvalues_M, tol=1e-8):
    return np.where(np.abs(np.imag(eigenvalues_M)) < tol)[0]

def compute_autocorrelation(z_k, max_lag):
    n = len(z_k)
    is_cx = np.iscomplexobj(z_k) and np.abs(np.imag(z_k)).max() > 1e-10
    z_centered = z_k - np.mean(z_k)
    fft_len = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    Z = np.fft.fft(z_centered, fft_len)
    acf_full = np.fft.ifft(Z * np.conj(Z))
    acf = np.abs(acf_full[:max_lag + 1]) if is_cx else np.real(acf_full[:max_lag + 1])
    if acf[0] > 0:
        acf = acf / acf[0]
    return acf, is_cx

def fit_exponential_timescale(autocorr, dt, fit_range=(1, None)):
    start, end = fit_range[0], fit_range[1] or len(autocorr)
    lags = np.arange(start, end)
    ac = autocorr[start:end]
    positive_mask = ac > 0.01
    if positive_mask.sum() < 5:
        return np.nan
    slope, *_ = stats.linregress(lags[positive_mask] * dt, np.log(ac[positive_mask]))
    return np.nan if slope >= 0 else -1.0 / slope

def project_onto_eigenvectors(activity, eigenvectors):
    V_inv = np.linalg.inv(eigenvectors)
    return activity @ V_inv.T

def zscore_rows(arr):
    return (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-10)

print("Analysis helpers defined.")


# %%
# --- Sweep helper ---

def _fit_sweep_timescales(act, proj, stab_idx, N, dt, max_lag=100):
    """Fit exp timescales for all neurons + stable eigenmodes. Returns (tau_neu, tau_mode, valid_neu, valid_mode)."""
    tau_neu = np.full(N, np.nan)
    for i in range(N):
        acf, _ = compute_autocorrelation(act[:, i], max_lag)
        tau_neu[i] = fit_exponential_timescale(acf, dt, fit_range=(1, 30))
    tau_mode = np.full(len(stab_idx), np.nan)
    for j, midx in enumerate(stab_idx):
        acf, _ = compute_autocorrelation(proj[:, midx], max_lag)
        tau_mode[j] = fit_exponential_timescale(acf, dt, fit_range=(1, 30))
    valid_neu  = np.isfinite(tau_neu)  & (tau_neu  > 0)
    valid_mode = np.isfinite(tau_mode) & (tau_mode > 0)
    return tau_neu, tau_mode, valid_neu, valid_mode


def _plot_sweep_histograms(results, sweep_values, sweep_param_name, suptitle):
    """2-column histogram grid: theory vs neurons | theory vs eigenmodes, one row per sweep value."""
    n_rows = len(results)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(12, 3.5 * n_rows),
        sharex='col', sharey='col', constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    axes[0, 0].set_title('Theory vs Neurons')
    axes[0, 1].set_title('Theory vs Jacobian Eigenmodes')

    for row, (val, res) in enumerate(zip(sweep_values, results)):
        row_label = rf'${sweep_param_name}={val}$' + '\nDensity'
        if res.get('skipped'):
            for col in range(2):
                axes[row, col].text(0.5, 0.5, 'N/A\n(unstable)', ha='center', va='center',
                                    transform=axes[row, col].transAxes, fontsize=11, color='gray')
            axes[row, 0].set_ylabel(row_label)
            continue

        tau_th_paired  = res['tau_th_all'][res['valid_mode']]
        tau_neu_valid  = res['tau_neu'][res['valid_neu']]
        tau_mode_valid = res['tau_mode'][res['valid_mode']]

        all_vals = np.concatenate([tau_th_paired, tau_neu_valid, tau_mode_valid])
        if len(all_vals) < 2:
            axes[row, 0].set_ylabel(row_label)
            continue
        bins = np.logspace(
            np.log10(np.percentile(all_vals, 1)),
            np.log10(np.percentile(all_vals, 99)),
            40,
        )

        ax = axes[row, 0]
        ax.hist(tau_th_paired, bins=bins, alpha=0.55, density=True,
                color='tab:blue', edgecolor='white', label='Linear Theory')
        ax.hist(tau_neu_valid, bins=bins, alpha=0.55, density=True,
                color='tab:orange', edgecolor='white', label=f'Neurons (n={res["valid_neu"].sum()})')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_ylabel(row_label); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        ax.hist(tau_th_paired, bins=bins, alpha=0.55, density=True,
                color='tab:blue', edgecolor='white', label='Linear Theory')
        ax.hist(tau_mode_valid, bins=bins, alpha=0.55, density=True,
                color='tab:green', edgecolor='white', label=f'Eigenmodes (n={res["valid_mode"].sum()})')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for col in range(2):
        axes[-1, col].set_xlabel(r'$\tau_{\rm eff}$')
    fig.suptitle(suptitle, fontsize=13)
    plt.show()


print("Sweep helpers defined.")


# %% [markdown]
# ---
# ## Single-g Simulation and Validation
#
# Single W, fixed g: verify simulated ACF timescales match theory.


# %%
sim_config = {
    "n_neurons": 1000,
    "g":         0.99,
    "tau":       1.0,
    "dt":        0.1,
    "duration":  5000.0,
    "noise_std": 0.005,
}


# %%
n_neurons = sim_config["n_neurons"]
g_sim     = sim_config["g"]
tau_sim   = sim_config["tau"]
dt_sim    = sim_config["dt"]

_model_sim = build_model(n_neurons, g=g_sim, tau=tau_sim, dt=dt_sim,
                         activation=nn.Identity, dynamics_type="rate")
W_sim = get_W(_model_sim)

eigenvalues_W_sim, eigenvectors_sim = np.linalg.eig(W_sim)
eigenvalues_M_sim = get_jacobian_eigenvalues(eigenvalues_W_sim, g_sim, tau_sim)
tau_theory        = get_effective_timescales(eigenvalues_M_sim)
stable_indices    = get_stable_indices(eigenvalues_M_sim)
tau_theory_all    = tau_theory

print(f"W shape: {W_sim.shape}")
print(f"g = {g_sim},  tau = {tau_sim}")
print(f"Stable modes: {len(tau_theory)}/{n_neurons}")
print(f"Theoretical timescale range: [{tau_theory.min():.3f}, {tau_theory.max():.3f}]")


# %%
print("Running linear simulation...")
activity = simulate_linear_ode(W_sim, g_sim, tau_sim, dt_sim,
                                sim_config["duration"], sim_config["noise_std"])
print(f"Done. Activity shape: {activity.shape}")
print(f"Activity range: [{activity.min():.3f}, {activity.max():.3f}]")

discard_steps   = int(100 / dt_sim)
activity_steady = activity[discard_steps:]


# %%
print("Projecting onto eigenvectors of W...")
projections = project_onto_eigenvectors(activity_steady, eigenvectors_sim)
print(f"Projections shape: {projections.shape}")


# %%
max_lag = 100

print("Fitting neuron timescales...")
tau_neuron_fitted = np.full(n_neurons, np.nan)
for i in range(n_neurons):
    acf, _ = compute_autocorrelation(activity_steady[:, i], max_lag)
    tau_neuron_fitted[i] = fit_exponential_timescale(acf, dt_sim, fit_range=(1, 30))

print("Fitting eigenmode timescales...")
tau_mode_fitted = np.full(len(stable_indices), np.nan)
for j, mode_idx in enumerate(stable_indices):
    acf, _ = compute_autocorrelation(projections[:, mode_idx], max_lag)
    tau_mode_fitted[j] = fit_exponential_timescale(acf, dt_sim, fit_range=(1, 30))

valid_neuron = np.isfinite(tau_neuron_fitted) & (tau_neuron_fitted > 0)
valid_mode   = np.isfinite(tau_mode_fitted)   & (tau_mode_fitted   > 0)
print(f"Valid neuron fits:    {valid_neuron.sum()}/{n_neurons}")
print(f"Valid eigenmode fits: {valid_mode.sum()}/{len(stable_indices)}")


# %%
n_show       = 10
rng_display  = np.random.default_rng(0)
neuron_display_idx    = rng_display.choice(np.where(valid_neuron)[0], size=n_show, replace=False)
valid_mode_local      = np.where(valid_mode)[0]
mode_display_local_idx  = rng_display.choice(valid_mode_local, size=n_show, replace=False)
mode_display_global_idx = stable_indices[mode_display_local_idx]
_title_suffix = rf'$g={g_sim}$,  $\tau_0={tau_sim}$,  $N={n_neurons}$  [linear]'
eigs = eigenvalues_M_sim


# %% [markdown]
# ### Traces


# %%
mode_trace   = 'amplitude'
plot_steps   = min(int(500 / dt_sim), activity_steady.shape[0])
time_win     = np.arange(plot_steps) * dt_sim
colors_show  = plt.cm.tab10(np.arange(n_show) / 10)
offset_scale = 4.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for i in range(n_show):
    trace = activity_steady[:plot_steps, neuron_display_idx[i]]
    trace_norm = trace / (trace.std() + 1e-10)
    ax.plot(time_win, trace_norm + i * offset_scale, color=colors_show[i],
            linewidth=0.8, label=f'Neuron {neuron_display_idx[i]}')
ax.set_xlabel('Time'); ax.set_yticks([])
ax.set_title('Neural activity'); ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2)

ax = axes[1]
for i in range(n_show):
    gidx    = mode_display_global_idx[i]
    lidx    = mode_display_local_idx[i]
    tau_th  = tau_theory_all[lidx]
    eig_val = eigs[gidx]
    is_cx   = np.abs(np.imag(eig_val)) > 1e-8
    if mode_trace == 'amplitude':
        trace = np.sqrt(2) * np.abs(projections[:plot_steps, gidx]) if is_cx \
                else np.abs(np.real(projections[:plot_steps, gidx]))
        label = f'Mode {gidx}  (τ={tau_th:.1f})'
    else:
        trace  = np.real(projections[:plot_steps, gidx])
        suffix = 'ℝ' if not is_cx else 'ℂ'
        label  = f'[{suffix}] Mode {gidx}  (τ={tau_th:.1f})'
    trace_norm = trace / (trace.std() + 1e-10)
    ax.plot(time_win, trace_norm + i * offset_scale, color=colors_show[i], linewidth=0.8, label=label)

right_title = r'Eigenmode  $\sqrt{2}|z_k|$' if mode_trace == 'amplitude' else r'Eigenmode  Re$(z_k)$'
ax.set_xlabel('Time'); ax.set_yticks([]); ax.set_title(right_title)
ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2)

fig.suptitle(f'Sample traces  —  {_title_suffix}', fontsize=13)
plt.tight_layout(); plt.show()


# %% [markdown]
# ### Heatmaps


# %%
mode_display = 'real'
normalize    = True
n_rows_show  = 200

plot_steps = min(int(500 / dt_sim), activity_steady.shape[0])
time_win   = np.arange(plot_steps) * dt_sim

row_idx    = np.arange(n_rows_show)
N_rows     = len(row_idx)
raw_neurons = activity_steady[:plot_steps, row_idx].T

if mode_display == 'amplitude':
    raw_modes = np.abs(projections[:plot_steps, row_idx]).T
    mode_label = r'$|z_k|$'
else:
    raw_modes = np.real(projections[:plot_steps, row_idx]).T
    mode_label = r'Re$(z_k)$'

data_neurons = zscore_rows(raw_neurons) if normalize else raw_neurons
data_modes   = zscore_rows(raw_modes)   if normalize else raw_modes
cbar_label   = 'z-score' if normalize else 'raw'

vmin_shared, vmax_shared = (-2.5, 2.5) if normalize else (
    -max(np.abs(data_neurons).max(), np.abs(raw_modes).max()),
     max(np.abs(data_neurons).max(), np.abs(raw_modes).max()),
)
kw = dict(aspect='auto', vmin=vmin_shared, vmax=vmax_shared, cmap='RdBu_r',
          extent=[0, time_win[-1], N_rows - 0.5, -0.5], interpolation='none')

ytick_pos    = np.linspace(0, N_rows - 1, 6, dtype=int)
ytick_labels = [str(p) for p in ytick_pos]

fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
im = axes[0].imshow(data_neurons, **kw)
axes[0].set_xlabel('Time'); axes[0].set_ylabel('Neuron index')
axes[0].set_title(f'Neural activity  (showing {N_rows}/{n_neurons})')
axes[0].set_yticks(ytick_pos); axes[0].set_yticklabels(ytick_labels)
fig.colorbar(im, ax=axes[0], label=cbar_label, shrink=0.6)

im = axes[1].imshow(data_modes, **kw)
axes[1].set_xlabel('Time'); axes[1].set_ylabel('Mode index')
axes[1].set_title(f'Eigenmode projections  {mode_label}  (showing {N_rows}/{n_neurons})')
axes[1].set_yticks(ytick_pos); axes[1].set_yticklabels(ytick_labels)
fig.colorbar(im, ax=axes[1], label=cbar_label, shrink=0.6)

fig.suptitle(f'Full population heatmaps  —  {_title_suffix}', fontsize=13)
plt.show()


# %% [markdown]
# ### Per-unit Autocorrelations with Exponential Fits


# %%
n_show_ac = 8
lags = np.arange(max_lag + 1) * dt_sim

fig, axes = plt.subplots(n_show_ac, 2, figsize=(14, 3.5 * n_show_ac), sharex=True)
axes[0, 0].set_title('Neural activity', fontsize=12)
axes[0, 1].set_title('Eigenmode projections', fontsize=12)

for i in range(n_show_ac):
    ax   = axes[i, 0]
    nidx = neuron_display_idx[i]
    acf, _ = compute_autocorrelation(activity_steady[:, nidx], max_lag)
    tau_fit = tau_neuron_fitted[nidx]
    ax.semilogy(lags, np.maximum(acf, 1e-6), color='steelblue', linewidth=1.5, label='ACF')
    if np.isfinite(tau_fit):
        ax.semilogy(lags, np.exp(-lags / tau_fit), 'k--', linewidth=1.5, label=rf'Fitted  $\tau={tau_fit:.1f}$')
    ax.set_ylabel('Autocorrelation'); ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    ax.annotate(f'Neuron {nidx}', xy=(0.03, 0.05), xycoords='axes fraction',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.set_ylim([1e-2, 1.5])

    ax   = axes[i, 1]
    lidx = mode_display_local_idx[i]
    gidx = mode_display_global_idx[i]
    acf, is_cx = compute_autocorrelation(projections[:, gidx], max_lag)
    tau_th  = tau_theory_all[lidx]
    tau_fit = tau_mode_fitted[lidx]
    ax.semilogy(lags, np.maximum(acf, 1e-6), color='steelblue', linewidth=1.5,
                label=r'$|\mathrm{ACF}|$' if is_cx else 'ACF')
    if np.isfinite(tau_fit):
        ax.semilogy(lags, np.exp(-lags / tau_fit), 'k--', linewidth=1.5, label=rf'Fitted  $\tau={tau_fit:.1f}$')
    ax.semilogy(lags, np.exp(-lags / tau_th), 'r--', linewidth=1.5, label=rf'Theory  $\tau={tau_th:.1f}$')
    ax.set_ylabel('Autocorrelation'); ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    suffix = 'ℂ' if is_cx else 'ℝ'
    ax.annotate(f'[{suffix}] Mode {gidx}', xy=(0.03, 0.05), xycoords='axes fraction',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.set_ylim([1e-4, 1.5])

axes[-1, 0].set_xlabel('Lag (time units)'); axes[-1, 1].set_xlabel('Lag (time units)')
fig.suptitle(f'Autocorrelations  —  {_title_suffix}', fontsize=13)
plt.tight_layout(); plt.show()


# %% [markdown]
# ### Theory vs Simulation: Timescale Distribution Comparison


# %%
tau_theory_paired = tau_theory_all[valid_mode]
tau_mode_valid    = tau_mode_fitted[valid_mode]
tau_neuron_valid  = tau_neuron_fitted[valid_neuron]

all_vals = np.concatenate([tau_theory_paired, tau_neuron_valid, tau_mode_valid])
bins = np.logspace(np.log10(np.percentile(all_vals, 1)), np.log10(np.percentile(all_vals, 99)), 50)

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

ax = axes[0, 0]
ax.hist(tau_theory_paired, bins=bins, alpha=0.55, density=True, color='tab:blue',
        edgecolor='white', label=f'Theory  (n={len(tau_theory_paired)})')
ax.hist(tau_neuron_valid, bins=bins, alpha=0.55, density=True, color='tab:orange',
        edgecolor='white', label=f'Neurons  (n={len(tau_neuron_valid)})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\tau_{\rm eff}$'); ax.set_ylabel('Density')
ax.set_title('Distributions: Theory vs Neuron fits'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(tau_theory_paired, bins=bins, alpha=0.55, density=True, color='tab:blue',
        edgecolor='white', label=f'Theory  (n={len(tau_theory_paired)})')
ax.hist(tau_mode_valid, bins=bins, alpha=0.55, density=True, color='tab:green',
        edgecolor='white', label=f'Eigenmodes  (n={len(tau_mode_valid)})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\tau_{\rm eff}$'); ax.set_ylabel('Density')
ax.set_title('Distributions: Theory vs Eigenmode fits'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
n_min = min(len(tau_theory_paired), len(tau_neuron_valid))
ax.scatter(np.sort(tau_theory_paired)[:n_min], np.sort(tau_neuron_valid)[:n_min],
           s=8, alpha=0.3, color='tab:orange')
vmin_s, vmax_s = min(tau_theory_paired.min(), tau_neuron_valid.min()), \
                 max(tau_theory_paired.max(), tau_neuron_valid.max())
ax.plot([vmin_s, vmax_s], [vmin_s, vmax_s], 'r--', linewidth=2, label='Identity')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Theory $\tau$ (sorted)'); ax.set_ylabel(r'Fitted $\tau$ — neurons (sorted)')
ax.set_title('Rank-rank: Theory vs Neuron fits'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

ax = axes[1, 1]
ax.scatter(tau_theory_paired, tau_mode_valid, s=8, alpha=0.3, color='tab:green')
vmin_s, vmax_s = min(tau_theory_paired.min(), tau_mode_valid.min()), \
                 max(tau_theory_paired.max(), tau_mode_valid.max())
ax.plot([vmin_s, vmax_s], [vmin_s, vmax_s], 'r--', linewidth=2, label='Identity')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Theory $\tau$'); ax.set_ylabel(r'Fitted $\tau$ — eigenmodes')
ax.set_title('1-to-1: Theory vs Eigenmode fits'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

fig.suptitle(f'Theory vs simulation  —  {_title_suffix}', fontsize=13)
plt.tight_layout(); plt.show()


# %% [markdown]
# ---
# ## Sweep 1: Multiple g Values (Gaussian W, linear — g < 1 only)
#
# Linear dynamics are unstable for g ≥ 1. Shared W, varying g.


# %%
g_sweep_config = {
    "g_values":  [0.3, 0.5, 0.8, 0.95, 0.99],   # g < 1 only
    "n_neurons": 1000,
    "tau":       1.0,
    "dt":        0.1,
    "duration":  5000.0,
    "noise_std": 0.005,
}

torch.manual_seed(1)
N_gs  = g_sweep_config["n_neurons"]
W_gs  = generate_W(N_gs)
eigenvalues_W_gs, eigenvectors_gs = np.linalg.eig(W_gs)
print(f"W_gs spectral radius: {np.abs(eigenvalues_W_gs).max():.4f}")


# %%
discard_gs   = int(100 / g_sweep_config["dt"])
results_g_sw = []

for g in g_sweep_config["g_values"]:
    print(f"\n── g = {g} ──────────────────────────")
    tau_gs = g_sweep_config["tau"]
    dt_gs  = g_sweep_config["dt"]

    eigs_M_gs  = get_jacobian_eigenvalues(eigenvalues_W_gs, g, tau_gs)
    stab_idx_gs = get_stable_indices(eigs_M_gs)
    tau_th_gs  = get_effective_timescales(eigs_M_gs)
    n_unstable_gs = N_gs - len(stab_idx_gs)
    print(f"  Stable modes: {len(stab_idx_gs)}/{N_gs}")

    if n_unstable_gs > 0:
        print(f"  {n_unstable_gs} unstable modes — skipping linear simulation (would diverge).")
        results_g_sw.append(dict(g=g, skipped=True))
        continue

    print("  Simulating ...")
    act_gs  = simulate_linear_ode(W_gs, g, tau_gs, dt_gs, g_sweep_config["duration"],
                                   g_sweep_config["noise_std"])[discard_gs:]
    proj_gs = project_onto_eigenvectors(act_gs, eigenvectors_gs)

    print("  Fitting timescales ...")
    tau_neu_gs, tau_mode_gs, valid_neu_gs, valid_mode_gs = _fit_sweep_timescales(
        act_gs, proj_gs, stab_idx_gs, N_gs, dt_gs,
    )
    print(f"  Valid neuron: {valid_neu_gs.sum()}/{N_gs}, eigenmode: {valid_mode_gs.sum()}/{len(stab_idx_gs)}")

    results_g_sw.append(dict(
        g=g, tau_th_all=tau_th_gs, stab_idx=stab_idx_gs,
        tau_neu=tau_neu_gs, tau_mode=tau_mode_gs,
        valid_neu=valid_neu_gs, valid_mode=valid_mode_gs,
    ))
    print("  Done.")

print("\nSweep complete.")


# %%
_plot_sweep_histograms(
    results_g_sw,
    g_sweep_config["g_values"],
    sweep_param_name="g",
    suptitle=rf'Linear network — timescale distributions across $g$  ($N={N_gs}$, Gaussian $W$)',
)


# %% [markdown]
# ---
# ## Sweep 2: Multiple α Values (Levy-Stable W, linear, g < 1)
#
# Fresh W per α. Normalization controls whether eigenspectra are comparable across α.
#
# - `"rnn"`: scale = 1/√N — matches `MultiTimescaleRNN` init; spectral radius grows for small α.
# - `"stable"`: scale = 1/N^(1/α) — spectral radius ~O(1) across all α (recommended for comparison).


# %%
alpha_sweep_config = {
    "alpha_values": [0.5, 1.0, 1.5, 2.0],   # 2.0 = Gaussian limit
    "g_fixed":      0.9,
    "n_neurons":    1000,
    "tau":          1.0,
    "dt":           0.1,
    "duration":     5000.0,
    "noise_std":    0.005,
    # Choose normalization:
    #   "rnn"    → scale = 1/sqrt(N)    matches MultiTimescaleRNN levy_stable init
    #   "stable" → scale = 1/N^(1/alpha) spectral radius ~O(1), eigenspectra comparable
    "levy_normalization": "stable",
}

norm_str_as   = alpha_sweep_config["levy_normalization"]
norm_label_as = r"1/\sqrt{N}" if norm_str_as == "rnn" else r"1/N^{1/\alpha}"
print(f"levy_normalization = '{norm_str_as}' (scale = {norm_label_as})")


# %%
discard_as   = int(100 / alpha_sweep_config["dt"])
results_a_sw = []

for alpha_s in alpha_sweep_config["alpha_values"]:
    print(f"\n── α = {alpha_s} ──────────────────────────")
    N_as  = alpha_sweep_config["n_neurons"]
    g_as  = alpha_sweep_config["g_fixed"]
    tau_as = alpha_sweep_config["tau"]
    dt_as  = alpha_sweep_config["dt"]
    norm   = alpha_sweep_config["levy_normalization"]

    W_as       = generate_W_levy(N_as, alpha_s, normalization=norm)
    eigs_W_as, eigenvectors_as = np.linalg.eig(W_as)
    sr_as = np.abs(eigs_W_as).max()
    print(f"  Spectral radius: {sr_as:.3f}")

    eigs_M_as   = get_jacobian_eigenvalues(eigs_W_as, g_as, tau_as)
    stab_idx_as = get_stable_indices(eigs_M_as)
    tau_th_as   = get_effective_timescales(eigs_M_as)
    n_unstable_as = N_as - len(stab_idx_as)
    print(f"  Stable modes: {len(stab_idx_as)}/{N_as}")

    if n_unstable_as > 0:
        print(f"  {n_unstable_as} unstable modes — skipping linear simulation (would diverge).")
        results_a_sw.append(dict(alpha=alpha_s, skipped=True))
        continue

    print("  Simulating ...")
    act_as  = simulate_linear_ode(W_as, g_as, tau_as, dt_as, alpha_sweep_config["duration"],
                                   alpha_sweep_config["noise_std"])[discard_as:]
    proj_as = project_onto_eigenvectors(act_as, eigenvectors_as)

    print("  Fitting timescales ...")
    tau_neu_as, tau_mode_as, valid_neu_as, valid_mode_as = _fit_sweep_timescales(
        act_as, proj_as, stab_idx_as, N_as, dt_as,
    )
    print(f"  Valid neuron: {valid_neu_as.sum()}/{N_as}, eigenmode: {valid_mode_as.sum()}/{len(stab_idx_as)}")

    results_a_sw.append(dict(
        alpha=alpha_s, skipped=False, tau_th_all=tau_th_as, stab_idx=stab_idx_as,
        tau_neu=tau_neu_as, tau_mode=tau_mode_as,
        valid_neu=valid_neu_as, valid_mode=valid_mode_as,
    ))
    print("  Done.")

print("\nSweep complete.")


# %%
_plot_sweep_histograms(
    results_a_sw,
    alpha_sweep_config["alpha_values"],
    sweep_param_name=r"\alpha",
    suptitle=(
        rf'Linear network — timescale distributions across $\alpha$'
        rf'  ($g={alpha_sweep_config["g_fixed"]}$, $N={alpha_sweep_config["n_neurons"]}$,'
        rf'  scale $= {norm_label_as}$)'
    ),
)

# %%
