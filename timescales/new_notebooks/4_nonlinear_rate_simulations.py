# %% [markdown]
# # Nonlinear Rate-Based Network Simulations
#
# Simulate $\tau \dot{r} = -r + \tanh(g W r) + \eta(t)$  (rate dynamics).
#
# The nonlinearity acts on the **pre-synaptic input** $gWr$ rather than on the
# membrane potential. This is the "rate" formulation, distinct from the voltage
# model in File 3 ($\tau \dot{x} = -x + gW\tanh(x)$).
#
# **Theory (linearization at $r^*=0$):** $\tanh'(0) = 1$, so the Jacobian is
# still $M = \frac{1}{\tau}(gW - I)$. The same eigenvalue predictions apply
# in the subcritical regime, but the nonlinear regime differs from the voltage case.


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

def simulate_rate_tanh(W, g, tau, dt, duration, noise_std):
    """tau*dr/dt = -r + tanh(g*W*r) + noise  (nonlinear rate)."""
    model = build_model(len(W), g=g, tau=tau, dt=dt, activation=nn.Tanh, dynamics_type="rate")
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
# --- Sweep helpers ---

def _fit_sweep_timescales(act, proj, stab_idx, N, dt, max_lag=100, fit_range=(1, 30)):
    tau_neu = np.full(N, np.nan)
    for i in range(N):
        acf, _ = compute_autocorrelation(act[:, i], max_lag)
        tau_neu[i] = fit_exponential_timescale(acf, dt, fit_range=fit_range)
    tau_mode = np.full(len(stab_idx), np.nan)
    for j, midx in enumerate(stab_idx):
        acf, _ = compute_autocorrelation(proj[:, midx], max_lag)
        tau_mode[j] = fit_exponential_timescale(acf, dt, fit_range=fit_range)
    valid_neu  = np.isfinite(tau_neu)  & (tau_neu  > 0)
    valid_mode = np.isfinite(tau_mode) & (tau_mode > 0)
    return tau_neu, tau_mode, valid_neu, valid_mode


def _plot_sweep_histograms(results, sweep_values, sweep_param_name, suptitle):
    n_rows = len(results)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(12, 3.5 * n_rows),
        sharex='col', sharey='col', constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    axes[0, 0].set_title('Theory (linear) vs Neurons')
    axes[0, 1].set_title('Theory (linear) vs Jacobian Eigenmodes')

    for row, (val, res) in enumerate(zip(sweep_values, results)):
        row_label = rf'${sweep_param_name}={val}$' + '\nDensity'
        if res.get('skipped'):
            for col in range(2):
                axes[row, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
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
            np.log10(np.percentile(all_vals, 99)), 40,
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


# %%
rate_config = {
    "n_neurons": 1000,
    "g":         1.5,
    "tau":       1.0,
    "dt":        0.1,
    "duration":  5000.0,
    "noise_std": 0.005,
    "max_lag":   100,
    "fit_range": (1, 30),
}


# %%
torch.manual_seed(99)
n_neurons_rt = rate_config["n_neurons"]
g_sim_rt     = rate_config["g"]
tau_sim_rt   = rate_config["tau"]
dt_rt        = rate_config["dt"]

_model_rt = build_model(n_neurons_rt, g=g_sim_rt, tau=tau_sim_rt, dt=dt_rt,
                        activation=nn.Tanh, dynamics_type="rate")
W_rt = get_W(_model_rt)

eigenvalues_W_rt, eigenvectors_rt = np.linalg.eig(W_rt)
eigenvalues_M_rt  = get_jacobian_eigenvalues(eigenvalues_W_rt, g_sim_rt, tau_sim_rt)
stable_indices_rt = get_stable_indices(eigenvalues_M_rt)
tau_theory_all_rt = get_effective_timescales(eigenvalues_M_rt)

print(f"W shape: {W_rt.shape}")
print(f"g = {g_sim_rt},  tau = {tau_sim_rt}")
print(f"Stable modes: {len(stable_indices_rt)}/{n_neurons_rt}")
if len(tau_theory_all_rt) > 0:
    print(f"Theoretical timescale range: [{tau_theory_all_rt.min():.3f}, {tau_theory_all_rt.max():.3f}]")


# %%
print("Running rate-tanh simulation...")
activity_rt = simulate_rate_tanh(W_rt, g_sim_rt, tau_sim_rt, dt_rt,
                                  rate_config["duration"], rate_config["noise_std"])
print(f"Done. Shape: {activity_rt.shape}, range: [{activity_rt.min():.3f}, {activity_rt.max():.3f}]")

discard_rt       = int(100 / dt_rt)
activity_rt_steady = activity_rt[discard_rt:]


# %%
print("Projecting onto eigenvectors of W...")
projections_rt = project_onto_eigenvectors(activity_rt_steady, eigenvectors_rt)
print(f"Projections shape: {projections_rt.shape}")


# %%
max_lag_rt   = rate_config["max_lag"]
fit_range_rt = rate_config["fit_range"]

print("Fitting neuron timescales ...")
tau_neuron_fitted_rt = np.full(n_neurons_rt, np.nan)
for i in range(n_neurons_rt):
    acf, _ = compute_autocorrelation(activity_rt_steady[:, i], max_lag_rt)
    tau_neuron_fitted_rt[i] = fit_exponential_timescale(acf, dt_rt, fit_range=fit_range_rt)

print("Fitting eigenmode timescales ...")
tau_mode_fitted_rt = np.full(len(stable_indices_rt), np.nan)
for j, mode_idx in enumerate(stable_indices_rt):
    acf, _ = compute_autocorrelation(projections_rt[:, mode_idx], max_lag_rt)
    tau_mode_fitted_rt[j] = fit_exponential_timescale(acf, dt_rt, fit_range=fit_range_rt)

valid_neuron_rt = np.isfinite(tau_neuron_fitted_rt) & (tau_neuron_fitted_rt > 0)
valid_mode_rt   = np.isfinite(tau_mode_fitted_rt)   & (tau_mode_fitted_rt   > 0)
print(f"Valid neuron fits: {valid_neuron_rt.sum()}/{n_neurons_rt}")
print(f"Valid eigenmode fits: {valid_mode_rt.sum()}/{len(stable_indices_rt)}")


# %%
n_show_rt      = 10
rng_display_rt = np.random.default_rng(0)
neuron_display_idx_rt     = rng_display_rt.choice(np.where(valid_neuron_rt)[0], size=n_show_rt, replace=False)
valid_mode_local_rt       = np.where(valid_mode_rt)[0]
mode_display_local_idx_rt  = rng_display_rt.choice(valid_mode_local_rt, size=n_show_rt, replace=False)
mode_display_global_idx_rt = stable_indices_rt[mode_display_local_idx_rt]
_title_suffix_rt = rf'$g={g_sim_rt}$,  $\tau_0={tau_sim_rt}$,  $N={n_neurons_rt}$  [rate tanh]'


# %% [markdown]
# ### Traces


# %%
mode_trace_rt   = 'amplitude'
plot_steps_rt   = min(int(500 / dt_rt), activity_rt_steady.shape[0])
time_win_rt     = np.arange(plot_steps_rt) * dt_rt
colors_show_rt  = plt.cm.tab10(np.arange(n_show_rt) / 10)
offset_scale_rt = 4.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for i in range(n_show_rt):
    trace = activity_rt_steady[:plot_steps_rt, neuron_display_idx_rt[i]]
    trace_norm = trace / (trace.std() + 1e-10)
    ax.plot(time_win_rt, trace_norm + i * offset_scale_rt, color=colors_show_rt[i],
            linewidth=0.8, label=f'Neuron {neuron_display_idx_rt[i]}')
ax.set_xlabel('Time'); ax.set_yticks([])
ax.set_title('Neural activity'); ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2)

ax = axes[1]
for i in range(n_show_rt):
    gidx    = mode_display_global_idx_rt[i]
    lidx    = mode_display_local_idx_rt[i]
    tau_th  = tau_theory_all_rt[lidx]
    eig_val = eigenvalues_M_rt[gidx]
    is_cx   = np.abs(np.imag(eig_val)) > 1e-8
    if mode_trace_rt == 'amplitude':
        trace = np.sqrt(2) * np.abs(projections_rt[:plot_steps_rt, gidx]) if is_cx \
                else np.abs(np.real(projections_rt[:plot_steps_rt, gidx]))
        label = f'Mode {gidx}  (τ={tau_th:.1f})'
    else:
        trace  = np.real(projections_rt[:plot_steps_rt, gidx])
        suffix = 'ℝ' if not is_cx else 'ℂ'
        label  = f'[{suffix}] Mode {gidx}  (τ={tau_th:.1f})'
    trace_norm = trace / (trace.std() + 1e-10)
    ax.plot(time_win_rt, trace_norm + i * offset_scale_rt, color=colors_show_rt[i],
            linewidth=0.8, label=label)

right_title = r'Eigenmode  $\sqrt{2}|z_k|$' if mode_trace_rt == 'amplitude' else r'Eigenmode  Re$(z_k)$'
ax.set_xlabel('Time'); ax.set_yticks([]); ax.set_title(right_title)
ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2)
fig.suptitle(f'Sample traces  —  {_title_suffix_rt}', fontsize=13)
plt.tight_layout(); plt.show()


# %% [markdown]
# ### Heatmaps


# %%
mode_display_rt = 'real'
normalize_rt    = True
n_rows_show_rt  = 200

plot_steps_rt = min(int(500 / dt_rt), activity_rt_steady.shape[0])
time_win_rt   = np.arange(plot_steps_rt) * dt_rt

row_idx_rt  = np.arange(n_rows_show_rt)
N_rows_rt   = len(row_idx_rt)
raw_neurons_rt = activity_rt_steady[:plot_steps_rt, row_idx_rt].T

if mode_display_rt == 'amplitude':
    raw_modes_rt = np.abs(projections_rt[:plot_steps_rt, row_idx_rt]).T
    mode_label_rt = r'$|z_k|$'
else:
    raw_modes_rt = np.real(projections_rt[:plot_steps_rt, row_idx_rt]).T
    mode_label_rt = r'Re$(z_k)$'

data_neurons_rt = zscore_rows(raw_neurons_rt) if normalize_rt else raw_neurons_rt
data_modes_rt   = zscore_rows(raw_modes_rt)   if normalize_rt else raw_modes_rt
cbar_label_rt   = 'z-score' if normalize_rt else 'raw'

vmin_rt, vmax_rt = (-2.5, 2.5) if normalize_rt else (
    -max(np.abs(data_neurons_rt).max(), np.abs(raw_modes_rt).max()),
     max(np.abs(data_neurons_rt).max(), np.abs(raw_modes_rt).max()),
)
kw_rt = dict(aspect='auto', vmin=vmin_rt, vmax=vmax_rt, cmap='RdBu_r',
             extent=[0, time_win_rt[-1], N_rows_rt - 0.5, -0.5], interpolation='none')
ytick_pos_rt    = np.linspace(0, N_rows_rt - 1, 6, dtype=int)
ytick_labels_rt = [str(p) for p in ytick_pos_rt]

fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
im = axes[0].imshow(data_neurons_rt, **kw_rt)
axes[0].set_xlabel('Time'); axes[0].set_ylabel('Neuron index')
axes[0].set_title(f'Neural activity  (showing {N_rows_rt}/{n_neurons_rt})')
axes[0].set_yticks(ytick_pos_rt); axes[0].set_yticklabels(ytick_labels_rt)
fig.colorbar(im, ax=axes[0], label=cbar_label_rt, shrink=0.6)

im = axes[1].imshow(data_modes_rt, **kw_rt)
axes[1].set_xlabel('Time'); axes[1].set_ylabel('Mode index')
axes[1].set_title(f'Eigenmode projections  {mode_label_rt}  (showing {N_rows_rt}/{n_neurons_rt})')
axes[1].set_yticks(ytick_pos_rt); axes[1].set_yticklabels(ytick_labels_rt)
fig.colorbar(im, ax=axes[1], label=cbar_label_rt, shrink=0.6)
fig.suptitle(f'Full population heatmaps  —  {_title_suffix_rt}', fontsize=13)
plt.show()


# %% [markdown]
# ### Per-unit Autocorrelations with Exponential Fits


# %%
n_show_ac_rt = 8
lags_rt = np.arange(max_lag_rt + 1) * dt_rt

fig, axes = plt.subplots(n_show_ac_rt, 2, figsize=(14, 3.5 * n_show_ac_rt), sharex=True)
axes[0, 0].set_title('Neural activity', fontsize=12)
axes[0, 1].set_title('Eigenmode projections', fontsize=12)

for i in range(n_show_ac_rt):
    ax   = axes[i, 0]
    nidx = neuron_display_idx_rt[i]
    acf, _ = compute_autocorrelation(activity_rt_steady[:, nidx], max_lag_rt)
    tau_fit = tau_neuron_fitted_rt[nidx]
    ax.semilogy(lags_rt, np.maximum(acf, 1e-6), color='steelblue', linewidth=1.5, label='ACF')
    if np.isfinite(tau_fit):
        ax.semilogy(lags_rt, np.exp(-lags_rt / tau_fit), 'k--', linewidth=1.5,
                    label=rf'Fitted  $\tau={tau_fit:.1f}$')
    ax.set_ylabel('Autocorrelation'); ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    ax.annotate(f'Neuron {nidx}', xy=(0.03, 0.05), xycoords='axes fraction',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.set_ylim([1e-2, 1.5])

    ax   = axes[i, 1]
    lidx = mode_display_local_idx_rt[i]
    gidx = mode_display_global_idx_rt[i]
    acf, is_cx = compute_autocorrelation(projections_rt[:, gidx], max_lag_rt)
    tau_th  = tau_theory_all_rt[lidx]
    tau_fit = tau_mode_fitted_rt[lidx]
    ax.semilogy(lags_rt, np.maximum(acf, 1e-6), color='steelblue', linewidth=1.5,
                label=r'$|\mathrm{ACF}|$' if is_cx else 'ACF')
    if np.isfinite(tau_fit):
        ax.semilogy(lags_rt, np.exp(-lags_rt / tau_fit), 'k--', linewidth=1.5,
                    label=rf'Fitted  $\tau={tau_fit:.1f}$')
    ax.semilogy(lags_rt, np.exp(-lags_rt / tau_th), 'r--', linewidth=1.5,
                label=rf'Theory  $\tau={tau_th:.1f}$')
    ax.set_ylabel('Autocorrelation'); ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    suffix = 'ℂ' if is_cx else 'ℝ'
    ax.annotate(f'[{suffix}] Mode {gidx}', xy=(0.03, 0.05), xycoords='axes fraction',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.set_ylim([1e-4, 1.5])

axes[-1, 0].set_xlabel('Lag (time units)'); axes[-1, 1].set_xlabel('Lag (time units)')
fig.suptitle(f'Autocorrelations  —  {_title_suffix_rt}', fontsize=13)
plt.tight_layout(); plt.show()


# %% [markdown]
# ### Theory vs Simulation: Timescale Distribution Comparison


# %%
tau_theory_paired_rt = tau_theory_all_rt[valid_mode_rt]
tau_mode_valid_rt    = tau_mode_fitted_rt[valid_mode_rt]
tau_neuron_valid_rt  = tau_neuron_fitted_rt[valid_neuron_rt]

all_vals_rt = np.concatenate([tau_theory_paired_rt, tau_neuron_valid_rt, tau_mode_valid_rt])
bins_rt = np.logspace(
    np.log10(np.percentile(all_vals_rt, 1)),
    np.log10(np.percentile(all_vals_rt, 99)), 50,
)

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

ax = axes[0, 0]
ax.hist(tau_theory_paired_rt, bins=bins_rt, alpha=0.55, density=True, color='tab:blue',
        edgecolor='white', label=f'Linear Theory  (n={len(tau_theory_paired_rt)})')
ax.hist(tau_neuron_valid_rt, bins=bins_rt, alpha=0.55, density=True, color='tab:orange',
        edgecolor='white', label=f'Neurons  (n={len(tau_neuron_valid_rt)})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\tau_{\rm eff}$'); ax.set_ylabel('Density')
ax.set_title('Distributions: Linear Theory vs Neurons'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(tau_theory_paired_rt, bins=bins_rt, alpha=0.55, density=True, color='tab:blue',
        edgecolor='white', label=f'Linear Theory  (n={len(tau_theory_paired_rt)})')
ax.hist(tau_mode_valid_rt, bins=bins_rt, alpha=0.55, density=True, color='tab:green',
        edgecolor='white', label=f'Jacobian Eigenmodes  (n={len(tau_mode_valid_rt)})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\tau_{\rm eff}$'); ax.set_ylabel('Density')
ax.set_title('Distributions: Linear Theory vs Jacobian Eigenmodes'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
n_min_rt = min(len(tau_theory_paired_rt), len(tau_neuron_valid_rt))
ax.scatter(np.sort(tau_theory_paired_rt)[:n_min_rt], np.sort(tau_neuron_valid_rt)[:n_min_rt],
           s=8, alpha=0.3, color='tab:orange')
v0, v1 = min(tau_theory_paired_rt.min(), tau_neuron_valid_rt.min()), \
         max(tau_theory_paired_rt.max(), tau_neuron_valid_rt.max())
ax.plot([v0, v1], [v0, v1], 'r--', linewidth=2, label='Identity')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Theory $\tau$ (sorted)'); ax.set_ylabel(r'Fitted $\tau$ — neurons (sorted)')
ax.set_title('Rank-rank: Theory vs Neuron fits'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

ax = axes[1, 1]
ax.scatter(tau_theory_paired_rt, tau_mode_valid_rt, s=8, alpha=0.3, color='tab:green')
v0, v1 = min(tau_theory_paired_rt.min(), tau_mode_valid_rt.min()), \
         max(tau_theory_paired_rt.max(), tau_mode_valid_rt.max())
ax.plot([v0, v1], [v0, v1], 'r--', linewidth=2, label='Identity')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Theory $\tau$'); ax.set_ylabel(r'Fitted $\tau$ — eigenmodes')
ax.set_title('1-to-1: Theory vs Eigenmode fits'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

fig.suptitle(f'Theory vs simulation  —  {_title_suffix_rt}', fontsize=13)
plt.tight_layout(); plt.show()


# %% [markdown]
# ---
# ## Sweep 1: Multiple g Values (Gaussian W, rate tanh)


# %%
g_sweep_rt = {
    "g_values":  [0.5, 0.9, 1.2, 1.5, 2.0],
    "n_neurons": 1000,
    "tau":       1.0,
    "dt":        0.1,
    "duration":  5000.0,
    "noise_std": 0.005,
    "max_lag":   100,
    "fit_range": (1, 30),
}

torch.manual_seed(20)
N_grt = g_sweep_rt["n_neurons"]
W_grt = generate_W(N_grt)
eigenvalues_W_grt, eigenvectors_grt = np.linalg.eig(W_grt)
print(f"W_grt spectral radius: {np.abs(eigenvalues_W_grt).max():.4f}")


# %%
discard_grt  = int(100 / g_sweep_rt["dt"])
results_grt  = []

for g in g_sweep_rt["g_values"]:
    print(f"\n── g = {g} ──────────────────────────")
    tau_grt = g_sweep_rt["tau"]
    dt_grt  = g_sweep_rt["dt"]

    eigs_M_grt   = get_jacobian_eigenvalues(eigenvalues_W_grt, g, tau_grt)
    stab_idx_grt = get_stable_indices(eigs_M_grt)
    tau_th_grt   = get_effective_timescales(eigs_M_grt)
    print(f"  Stable (linear theory) modes: {len(stab_idx_grt)}/{N_grt}")

    print("  Simulating rate-tanh ...")
    act_grt  = simulate_rate_tanh(W_grt, g, tau_grt, dt_grt,
                                   g_sweep_rt["duration"], g_sweep_rt["noise_std"])[discard_grt:]
    proj_grt = project_onto_eigenvectors(act_grt, eigenvectors_grt)

    print("  Fitting timescales ...")
    tau_neu_grt, tau_mode_grt, valid_neu_grt, valid_mode_grt = _fit_sweep_timescales(
        act_grt, proj_grt, stab_idx_grt, N_grt, dt_grt,
        max_lag=g_sweep_rt["max_lag"], fit_range=g_sweep_rt["fit_range"],
    )
    print(f"  Valid neuron: {valid_neu_grt.sum()}/{N_grt}, eigenmode: {valid_mode_grt.sum()}/{len(stab_idx_grt)}")

    results_grt.append(dict(
        g=g, tau_th_all=tau_th_grt, stab_idx=stab_idx_grt,
        tau_neu=tau_neu_grt, tau_mode=tau_mode_grt,
        valid_neu=valid_neu_grt, valid_mode=valid_mode_grt,
    ))
    print("  Done.")

print("\nSweep complete.")


# %%
_plot_sweep_histograms(
    results_grt,
    g_sweep_rt["g_values"],
    sweep_param_name="g",
    suptitle=rf'Rate-tanh network — timescale distributions across $g$  ($N={N_grt}$, Gaussian $W$)',
)


# %% [markdown]
# ---
# ## Sweep 2: Multiple α Values (Levy-Stable W, rate tanh)


# %%
alpha_sweep_rt = {
    "alpha_values": [0.5, 1.0, 1.5, 2.0],
    "g_fixed":      1.5,
    "n_neurons":    1000,
    "tau":          1.0,
    "dt":           0.1,
    "duration":     5000.0,
    "noise_std":    0.005,
    "max_lag":      100,
    "fit_range":    (1, 30),
    # Choose normalization:
    #   "rnn"    → scale = 1/sqrt(N)    matches MultiTimescaleRNN levy_stable init
    #   "stable" → scale = 1/N^(1/alpha) spectral radius ~O(1), eigenspectra comparable
    "levy_normalization": "stable",
}

norm_str_rt   = alpha_sweep_rt["levy_normalization"]
norm_label_rt = r"1/\sqrt{N}" if norm_str_rt == "rnn" else r"1/N^{1/\alpha}"
print(f"levy_normalization = '{norm_str_rt}' (scale = {norm_label_rt})")


# %%
discard_art  = int(100 / alpha_sweep_rt["dt"])
results_art  = []

for alpha_s in alpha_sweep_rt["alpha_values"]:
    print(f"\n── α = {alpha_s} ──────────────────────────")
    N_art  = alpha_sweep_rt["n_neurons"]
    g_art  = alpha_sweep_rt["g_fixed"]
    tau_art = alpha_sweep_rt["tau"]
    dt_art  = alpha_sweep_rt["dt"]
    norm    = alpha_sweep_rt["levy_normalization"]

    W_art       = generate_W_levy(N_art, alpha_s, normalization=norm)
    eigs_W_art, eigenvectors_art = np.linalg.eig(W_art)
    print(f"  Spectral radius: {np.abs(eigs_W_art).max():.3f}")

    eigs_M_art   = get_jacobian_eigenvalues(eigs_W_art, g_art, tau_art)
    stab_idx_art = get_stable_indices(eigs_M_art)
    tau_th_art   = get_effective_timescales(eigs_M_art)
    print(f"  Stable (linear theory) modes: {len(stab_idx_art)}/{N_art}")

    print("  Simulating rate-tanh ...")
    act_art  = simulate_rate_tanh(W_art, g_art, tau_art, dt_art,
                                   alpha_sweep_rt["duration"], alpha_sweep_rt["noise_std"])[discard_art:]
    proj_art = project_onto_eigenvectors(act_art, eigenvectors_art)

    print("  Fitting timescales ...")
    tau_neu_art, tau_mode_art, valid_neu_art, valid_mode_art = _fit_sweep_timescales(
        act_art, proj_art, stab_idx_art, N_art, dt_art,
        max_lag=alpha_sweep_rt["max_lag"], fit_range=alpha_sweep_rt["fit_range"],
    )
    print(f"  Valid neuron: {valid_neu_art.sum()}/{N_art}, eigenmode: {valid_mode_art.sum()}/{len(stab_idx_art)}")

    results_art.append(dict(
        alpha=alpha_s, skipped=False, tau_th_all=tau_th_art, stab_idx=stab_idx_art,
        tau_neu=tau_neu_art, tau_mode=tau_mode_art,
        valid_neu=valid_neu_art, valid_mode=valid_mode_art,
    ))
    print("  Done.")

print("\nSweep complete.")


# %%
_plot_sweep_histograms(
    results_art,
    alpha_sweep_rt["alpha_values"],
    sweep_param_name=r"\alpha",
    suptitle=(
        rf'Rate-tanh network — timescale distributions across $\alpha$'
        rf'  ($g={alpha_sweep_rt["g_fixed"]}$, $N={alpha_sweep_rt["n_neurons"]}$,'
        rf'  scale $= {norm_label_rt}$)'
    ),
)
