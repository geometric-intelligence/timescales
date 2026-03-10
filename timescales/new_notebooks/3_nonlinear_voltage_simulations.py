# %% [markdown]
# # Nonlinear Voltage-Based Network Simulations
#
# Simulate $\tau \dot{x} = -x + g W \tanh(x) + \eta(t)$  (voltage dynamics).
#
# **Theory (linearization at $x^*=0$):** Since $\tanh'(0) = 1$, the Jacobian at the
# origin is $M = \frac{1}{\tau}(gW - I)$, identical to the linear case.
# Eigenvalue-based timescale predictions are valid in the subcritical regime ($g < 1$).
# For $g \geq 1$ the network can enter chaotic dynamics, but the simulation remains
# bounded (unlike the linear case), so we sweep all g values.


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

def simulate_voltage_tanh(W, g, tau, dt, duration, noise_std):
    """tau*dx/dt = -x + g*W*tanh(x) + noise  (nonlinear voltage)."""
    model = build_model(len(W), g=g, tau=tau, dt=dt, activation=nn.Tanh, dynamics_type="voltage")
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


def _plot_sweep_acf(act, proj, stab_idx, tau_th_all, tau_neu, tau_mode,
                    valid_neu, valid_mode, dt, title_suffix='', n_show=6, max_lag=100):
    """Per-unit ACF diagnostic: sample neurons (left) and eigenmodes (right)."""
    rng_acf = np.random.default_rng(0)
    lags = np.arange(max_lag + 1) * dt

    n_valid_neu  = valid_neu.sum()
    n_valid_mode = valid_mode.sum()
    n_show_neu   = min(n_show, n_valid_neu)
    n_show_mode  = min(n_show, n_valid_mode)
    n_rows_acf   = max(n_show_neu, n_show_mode, 1)

    if n_valid_neu == 0 and n_valid_mode == 0:
        print("  No valid fits — skipping ACF plot.")
        return

    neu_idx  = rng_acf.choice(np.where(valid_neu)[0],  size=n_show_neu, replace=False) if n_valid_neu  > 0 else []
    mode_loc = rng_acf.choice(np.where(valid_mode)[0], size=n_show_mode, replace=False) if n_valid_mode > 0 else []
    mode_gbl = stab_idx[mode_loc] if len(mode_loc) > 0 else []

    fig, axes = plt.subplots(n_rows_acf, 2, figsize=(14, 2.8 * n_rows_acf), sharex=True)
    if n_rows_acf == 1:
        axes = axes[np.newaxis, :]
    axes[0, 0].set_title('Neurons', fontsize=11)
    axes[0, 1].set_title('Eigenmodes', fontsize=11)

    for i in range(n_rows_acf):
        ax = axes[i, 0]
        if i < n_show_neu:
            nidx = neu_idx[i]
            acf, _ = compute_autocorrelation(act[:, nidx], max_lag)
            ax.semilogy(lags, np.maximum(acf, 1e-6), color='steelblue', linewidth=1.5, label='ACF')
            tf = tau_neu[nidx]
            if np.isfinite(tf):
                ax.semilogy(lags, np.exp(-lags / tf), 'k--', linewidth=1.5, label=rf'Fit $\tau={tf:.1f}$')
            ax.annotate(f'Neuron {nidx}', xy=(0.03, 0.05), xycoords='axes fraction', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        ax.set_ylabel('ACF'); ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim([1e-2, 1.5])

        ax = axes[i, 1]
        if i < n_show_mode:
            lidx = mode_loc[i]
            gidx = mode_gbl[i]
            acf, is_cx = compute_autocorrelation(proj[:, gidx], max_lag)
            tf = tau_mode[lidx]
            tth = tau_th_all[lidx]
            ax.semilogy(lags, np.maximum(acf, 1e-6), color='steelblue', linewidth=1.5,
                        label=r'$|\mathrm{ACF}|$' if is_cx else 'ACF')
            if np.isfinite(tf):
                ax.semilogy(lags, np.exp(-lags / tf), 'k--', linewidth=1.5, label=rf'Fit $\tau={tf:.1f}$')
            ax.semilogy(lags, np.exp(-lags / tth), 'r--', linewidth=1.5, label=rf'Theory $\tau={tth:.1f}$')
            suffix = 'ℂ' if is_cx else 'ℝ'
            ax.annotate(f'[{suffix}] Mode {gidx}', xy=(0.03, 0.05), xycoords='axes fraction', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        ax.set_ylabel('ACF'); ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim([1e-4, 1.5])

    axes[-1, 0].set_xlabel('Lag (time units)'); axes[-1, 1].set_xlabel('Lag (time units)')
    fig.suptitle(f'Sweep ACF diagnostics  —  {title_suffix}', fontsize=12)
    plt.tight_layout(); plt.show()


print("Sweep helpers defined.")


# %% [markdown]
# ---
# ## Single-g Simulation and Validation


# %%
nl_config = {
    "n_neurons": 1000,
    "g":         1.5,
    "tau":       1.0,
    "dt":        0.1,
    "duration":  5000.0,
    "noise_std": 0.005,
    "max_lag":   100,
    "fit_range": (1, 60),
}


# %%
torch.manual_seed(42)
n_neurons_nl = nl_config["n_neurons"]
g_sim_nl     = nl_config["g"]
tau_sim_nl   = nl_config["tau"]
dt_nl        = nl_config["dt"]

_model_nl = build_model(n_neurons_nl, g=g_sim_nl, tau=tau_sim_nl, dt=dt_nl,
                        activation=nn.Tanh, dynamics_type="voltage")
W_nl = get_W(_model_nl)

eigenvalues_W_nl, eigenvectors_nl = np.linalg.eig(W_nl)
eigenvalues_M_nl   = get_jacobian_eigenvalues(eigenvalues_W_nl, g_sim_nl, tau_sim_nl)
stable_indices_nl  = get_stable_indices(eigenvalues_M_nl)
tau_theory_all_nl  = get_effective_timescales(eigenvalues_M_nl)

print(f"W shape: {W_nl.shape}")
print(f"g = {g_sim_nl},  tau = {tau_sim_nl}")
print(f"Stable modes: {len(stable_indices_nl)}/{n_neurons_nl}")
if len(tau_theory_all_nl) > 0:
    print(f"Theoretical timescale range: [{tau_theory_all_nl.min():.3f}, {tau_theory_all_nl.max():.3f}]")


# %%
print("Running voltage-tanh simulation...")
activity_nl = simulate_voltage_tanh(W_nl, g_sim_nl, tau_sim_nl, dt_nl,
                                     nl_config["duration"], nl_config["noise_std"])
print(f"Done. Shape: {activity_nl.shape}, range: [{activity_nl.min():.3f}, {activity_nl.max():.3f}]")

discard_nl        = int(100 / dt_nl)
activity_nl_steady = activity_nl[discard_nl:]


# %%
print("Projecting onto eigenvectors of W...")
projections_nl = project_onto_eigenvectors(activity_nl_steady, eigenvectors_nl)
print(f"Projections shape: {projections_nl.shape}")


# %%
max_lag_nl   = nl_config["max_lag"]
fit_range_nl = nl_config["fit_range"]

print("Fitting neuron timescales ...")
tau_neuron_fitted_nl = np.full(n_neurons_nl, np.nan)
for i in range(n_neurons_nl):
    acf, _ = compute_autocorrelation(activity_nl_steady[:, i], max_lag_nl)
    tau_neuron_fitted_nl[i] = fit_exponential_timescale(acf, dt_nl, fit_range=fit_range_nl)

print("Fitting eigenmode timescales ...")
tau_mode_fitted_nl = np.full(len(stable_indices_nl), np.nan)
for j, mode_idx in enumerate(stable_indices_nl):
    acf, _ = compute_autocorrelation(projections_nl[:, mode_idx], max_lag_nl)
    tau_mode_fitted_nl[j] = fit_exponential_timescale(acf, dt_nl, fit_range=fit_range_nl)

valid_neuron_nl = np.isfinite(tau_neuron_fitted_nl) & (tau_neuron_fitted_nl > 0)
valid_mode_nl   = np.isfinite(tau_mode_fitted_nl)   & (tau_mode_fitted_nl   > 0)
print(f"Valid neuron fits: {valid_neuron_nl.sum()}/{n_neurons_nl}")
print(f"Valid eigenmode fits: {valid_mode_nl.sum()}/{len(stable_indices_nl)}")


# %%
n_show_nl      = 10
rng_display_nl = np.random.default_rng(0)
neuron_display_idx_nl     = rng_display_nl.choice(np.where(valid_neuron_nl)[0], size=n_show_nl, replace=False)
valid_mode_local_nl       = np.where(valid_mode_nl)[0]
mode_display_local_idx_nl  = rng_display_nl.choice(valid_mode_local_nl, size=n_show_nl, replace=False)
mode_display_global_idx_nl = stable_indices_nl[mode_display_local_idx_nl]
_title_suffix_nl = rf'$g={g_sim_nl}$,  $\tau_0={tau_sim_nl}$,  $N={n_neurons_nl}$  [voltage tanh]'


# %% [markdown]
# ### Traces


# %%
mode_trace_nl  = 'amplitude'
plot_steps_nl  = min(int(500 / dt_nl), activity_nl_steady.shape[0])
time_win_nl    = np.arange(plot_steps_nl) * dt_nl
colors_show_nl = plt.cm.tab10(np.arange(n_show_nl) / 10)
offset_scale_nl = 4.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for i in range(n_show_nl):
    trace = activity_nl_steady[:plot_steps_nl, neuron_display_idx_nl[i]]
    trace_norm = trace / (trace.std() + 1e-10)
    ax.plot(time_win_nl, trace_norm + i * offset_scale_nl, color=colors_show_nl[i],
            linewidth=0.8, label=f'Neuron {neuron_display_idx_nl[i]}')
ax.set_xlabel('Time'); ax.set_yticks([])
ax.set_title('Neural activity'); ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2)

ax = axes[1]
for i in range(n_show_nl):
    gidx    = mode_display_global_idx_nl[i]
    lidx    = mode_display_local_idx_nl[i]
    tau_th  = tau_theory_all_nl[lidx]
    eig_val = eigenvalues_M_nl[gidx]
    is_cx   = np.abs(np.imag(eig_val)) > 1e-8
    if mode_trace_nl == 'amplitude':
        trace = np.sqrt(2) * np.abs(projections_nl[:plot_steps_nl, gidx]) if is_cx \
                else np.abs(np.real(projections_nl[:plot_steps_nl, gidx]))
        label = f'Mode {gidx}  (τ={tau_th:.1f})'
    else:
        trace  = np.real(projections_nl[:plot_steps_nl, gidx])
        suffix = 'ℝ' if not is_cx else 'ℂ'
        label  = f'[{suffix}] Mode {gidx}  (τ={tau_th:.1f})'
    trace_norm = trace / (trace.std() + 1e-10)
    ax.plot(time_win_nl, trace_norm + i * offset_scale_nl, color=colors_show_nl[i],
            linewidth=0.8, label=label)

right_title = r'Eigenmode  $\sqrt{2}|z_k|$' if mode_trace_nl == 'amplitude' else r'Eigenmode  Re$(z_k)$'
ax.set_xlabel('Time'); ax.set_yticks([]); ax.set_title(right_title)
ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2)
fig.suptitle(f'Sample traces  —  {_title_suffix_nl}', fontsize=13)
plt.tight_layout(); plt.show()


# %% [markdown]
# ### Heatmaps


# %%
mode_display_nl = 'real'
normalize_nl    = True
n_rows_show_nl  = 200

plot_steps_nl = min(int(500 / dt_nl), activity_nl_steady.shape[0])
time_win_nl   = np.arange(plot_steps_nl) * dt_nl

row_idx_nl  = np.arange(n_rows_show_nl)
N_rows_nl   = len(row_idx_nl)
raw_neurons_nl = activity_nl_steady[:plot_steps_nl, row_idx_nl].T

if mode_display_nl == 'amplitude':
    raw_modes_nl = np.abs(projections_nl[:plot_steps_nl, row_idx_nl]).T
    mode_label_nl = r'$|z_k|$'
else:
    raw_modes_nl = np.real(projections_nl[:plot_steps_nl, row_idx_nl]).T
    mode_label_nl = r'Re$(z_k)$'

data_neurons_nl = zscore_rows(raw_neurons_nl) if normalize_nl else raw_neurons_nl
data_modes_nl   = zscore_rows(raw_modes_nl)   if normalize_nl else raw_modes_nl
cbar_label_nl   = 'z-score' if normalize_nl else 'raw'

vmin_nl, vmax_nl = (-2.5, 2.5) if normalize_nl else (
    -max(np.abs(data_neurons_nl).max(), np.abs(raw_modes_nl).max()),
     max(np.abs(data_neurons_nl).max(), np.abs(raw_modes_nl).max()),
)
kw_nl = dict(aspect='auto', vmin=vmin_nl, vmax=vmax_nl, cmap='RdBu_r',
             extent=[0, time_win_nl[-1], N_rows_nl - 0.5, -0.5], interpolation='none')
ytick_pos_nl    = np.linspace(0, N_rows_nl - 1, 6, dtype=int)
ytick_labels_nl = [str(p) for p in ytick_pos_nl]

fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
im = axes[0].imshow(data_neurons_nl, **kw_nl)
axes[0].set_xlabel('Time'); axes[0].set_ylabel('Neuron index')
axes[0].set_title(f'Neural activity  (showing {N_rows_nl}/{n_neurons_nl})')
axes[0].set_yticks(ytick_pos_nl); axes[0].set_yticklabels(ytick_labels_nl)
fig.colorbar(im, ax=axes[0], label=cbar_label_nl, shrink=0.6)

im = axes[1].imshow(data_modes_nl, **kw_nl)
axes[1].set_xlabel('Time'); axes[1].set_ylabel('Mode index')
axes[1].set_title(f'Eigenmode projections  {mode_label_nl}  (showing {N_rows_nl}/{n_neurons_nl})')
axes[1].set_yticks(ytick_pos_nl); axes[1].set_yticklabels(ytick_labels_nl)
fig.colorbar(im, ax=axes[1], label=cbar_label_nl, shrink=0.6)
fig.suptitle(f'Full population heatmaps  —  {_title_suffix_nl}', fontsize=13)
plt.show()


# %% [markdown]
# ### Per-unit Autocorrelations with Exponential Fits


# %%
n_show_ac_nl = 8
lags_nl = np.arange(max_lag_nl + 1) * dt_nl

fig, axes = plt.subplots(n_show_ac_nl, 2, figsize=(14, 3.5 * n_show_ac_nl), sharex=True)
axes[0, 0].set_title('Neural activity', fontsize=12)
axes[0, 1].set_title('Eigenmode projections', fontsize=12)

for i in range(n_show_ac_nl):
    ax   = axes[i, 0]
    nidx = neuron_display_idx_nl[i]
    acf, _ = compute_autocorrelation(activity_nl_steady[:, nidx], max_lag_nl)
    tau_fit = tau_neuron_fitted_nl[nidx]
    ax.semilogy(lags_nl, np.maximum(acf, 1e-6), color='steelblue', linewidth=1.5, label='ACF')
    if np.isfinite(tau_fit):
        ax.semilogy(lags_nl, np.exp(-lags_nl / tau_fit), 'k--', linewidth=1.5,
                    label=rf'Fitted  $\tau={tau_fit:.1f}$')
    ax.set_ylabel('Autocorrelation'); ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    ax.annotate(f'Neuron {nidx}', xy=(0.03, 0.05), xycoords='axes fraction',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.set_ylim([1e-2, 1.5])

    ax   = axes[i, 1]
    lidx = mode_display_local_idx_nl[i]
    gidx = mode_display_global_idx_nl[i]
    acf, is_cx = compute_autocorrelation(projections_nl[:, gidx], max_lag_nl)
    tau_th  = tau_theory_all_nl[lidx]
    tau_fit = tau_mode_fitted_nl[lidx]
    ax.semilogy(lags_nl, np.maximum(acf, 1e-6), color='steelblue', linewidth=1.5,
                label=r'$|\mathrm{ACF}|$' if is_cx else 'ACF')
    if np.isfinite(tau_fit):
        ax.semilogy(lags_nl, np.exp(-lags_nl / tau_fit), 'k--', linewidth=1.5,
                    label=rf'Fitted  $\tau={tau_fit:.1f}$')
    ax.semilogy(lags_nl, np.exp(-lags_nl / tau_th), 'r--', linewidth=1.5,
                label=rf'Theory  $\tau={tau_th:.1f}$')
    ax.set_ylabel('Autocorrelation'); ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    suffix = 'ℂ' if is_cx else 'ℝ'
    ax.annotate(f'[{suffix}] Mode {gidx}', xy=(0.03, 0.05), xycoords='axes fraction',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.set_ylim([1e-4, 1.5])

axes[-1, 0].set_xlabel('Lag (time units)'); axes[-1, 1].set_xlabel('Lag (time units)')
fig.suptitle(f'Autocorrelations  —  {_title_suffix_nl}', fontsize=13)
plt.tight_layout(); plt.show()


# %% [markdown]
# ### Theory vs Simulation: Timescale Distribution Comparison


# %%
tau_theory_paired_nl = tau_theory_all_nl[valid_mode_nl]
tau_mode_valid_nl    = tau_mode_fitted_nl[valid_mode_nl]
tau_neuron_valid_nl  = tau_neuron_fitted_nl[valid_neuron_nl]

all_vals_nl = np.concatenate([tau_theory_paired_nl, tau_neuron_valid_nl, tau_mode_valid_nl])
bins_nl = np.logspace(
    np.log10(np.percentile(all_vals_nl, 1)),
    np.log10(np.percentile(all_vals_nl, 99)), 50,
)

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

ax = axes[0, 0]
ax.hist(tau_theory_paired_nl, bins=bins_nl, alpha=0.55, density=True, color='tab:blue',
        edgecolor='white', label=f'Linear Theory  (n={len(tau_theory_paired_nl)})')
ax.hist(tau_neuron_valid_nl, bins=bins_nl, alpha=0.55, density=True, color='tab:orange',
        edgecolor='white', label=f'Neurons  (n={len(tau_neuron_valid_nl)})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\tau_{\rm eff}$'); ax.set_ylabel('Density')
ax.set_title('Distributions: Linear Theory vs Neurons'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(tau_theory_paired_nl, bins=bins_nl, alpha=0.55, density=True, color='tab:blue',
        edgecolor='white', label=f'Linear Theory  (n={len(tau_theory_paired_nl)})')
ax.hist(tau_mode_valid_nl, bins=bins_nl, alpha=0.55, density=True, color='tab:green',
        edgecolor='white', label=f'Jacobian Eigenmodes  (n={len(tau_mode_valid_nl)})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\tau_{\rm eff}$'); ax.set_ylabel('Density')
ax.set_title('Distributions: Linear Theory vs Jacobian Eigenmodes'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
n_min_nl = min(len(tau_theory_paired_nl), len(tau_neuron_valid_nl))
ax.scatter(np.sort(tau_theory_paired_nl)[:n_min_nl], np.sort(tau_neuron_valid_nl)[:n_min_nl],
           s=8, alpha=0.3, color='tab:orange')
v0, v1 = min(tau_theory_paired_nl.min(), tau_neuron_valid_nl.min()), \
         max(tau_theory_paired_nl.max(), tau_neuron_valid_nl.max())
ax.plot([v0, v1], [v0, v1], 'r--', linewidth=2, label='Identity')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Theory $\tau$ (sorted)'); ax.set_ylabel(r'Fitted $\tau$ — neurons (sorted)')
ax.set_title('Rank-rank: Theory vs Neuron fits'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

ax = axes[1, 1]
ax.scatter(tau_theory_paired_nl, tau_mode_valid_nl, s=8, alpha=0.3, color='tab:green')
v0, v1 = min(tau_theory_paired_nl.min(), tau_mode_valid_nl.min()), \
         max(tau_theory_paired_nl.max(), tau_mode_valid_nl.max())
ax.plot([v0, v1], [v0, v1], 'r--', linewidth=2, label='Identity')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Theory $\tau$'); ax.set_ylabel(r'Fitted $\tau$ — eigenmodes')
ax.set_title('1-to-1: Theory vs Eigenmode fits'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

fig.suptitle(f'Theory vs simulation  —  {_title_suffix_nl}', fontsize=13)
plt.tight_layout(); plt.show()


# %% [markdown]
# ---
# ## Sweep 1: Multiple g Values (Gaussian W, voltage tanh — all g)
#
# tanh nonlinearity keeps activity bounded even for g > 1. Shared W, varying g.
# Compare with Shi et al. 2025, Supplementary Figure 13.

# %%
g_sweep_vt = {
    "g_values":  [1.6, 1.8, 2.0, 2.2], #matching Shi et al. 2025, Supplementary Figure 13
    "n_neurons": 1000,
    "tau":       1.0,
    "dt":        0.1,
    "duration":  5000.0,
    "noise_std": 0.0,
    "max_lag":   500,
    "fit_range": (1, 60),
}

torch.manual_seed(10)
N_gvt = g_sweep_vt["n_neurons"]
W_gvt = generate_W(N_gvt)
eigenvalues_W_gvt, eigenvectors_gvt = np.linalg.eig(W_gvt)
print(f"W_gvt spectral radius: {np.abs(eigenvalues_W_gvt).max():.4f}")


# %%
discard_gvt   = int(100 / g_sweep_vt["dt"])
results_gvt   = []

for g in g_sweep_vt["g_values"]:
    print(f"\n── g = {g} ──────────────────────────")
    tau_gvt = g_sweep_vt["tau"]
    dt_gvt  = g_sweep_vt["dt"]

    eigs_M_gvt   = get_jacobian_eigenvalues(eigenvalues_W_gvt, g, tau_gvt)
    stab_idx_gvt = get_stable_indices(eigs_M_gvt)
    tau_th_gvt   = get_effective_timescales(eigs_M_gvt)
    print(f"  Stable (linear theory) modes: {len(stab_idx_gvt)}/{N_gvt}")

    print("  Simulating voltage-tanh ...")
    act_gvt  = simulate_voltage_tanh(W_gvt, g, tau_gvt, dt_gvt,
                                      g_sweep_vt["duration"], g_sweep_vt["noise_std"])[discard_gvt:]
    proj_gvt = project_onto_eigenvectors(act_gvt, eigenvectors_gvt)

    print("  Fitting timescales ...")
    tau_neu_gvt, tau_mode_gvt, valid_neu_gvt, valid_mode_gvt = _fit_sweep_timescales(
        act_gvt, proj_gvt, stab_idx_gvt, N_gvt, dt_gvt,
        max_lag=g_sweep_vt["max_lag"], fit_range=g_sweep_vt["fit_range"],
    )
    print(f"  Valid neuron: {valid_neu_gvt.sum()}/{N_gvt}, eigenmode: {valid_mode_gvt.sum()}/{len(stab_idx_gvt)}")

    _plot_sweep_acf(act_gvt, proj_gvt, stab_idx_gvt, tau_th_gvt,
                    tau_neu_gvt, tau_mode_gvt, valid_neu_gvt, valid_mode_gvt,
                    dt_gvt, title_suffix=rf'$g={g}$  [voltage tanh, Gaussian $W$]')

    results_gvt.append(dict(
        g=g, tau_th_all=tau_th_gvt, stab_idx=stab_idx_gvt,
        tau_neu=tau_neu_gvt, tau_mode=tau_mode_gvt,
        valid_neu=valid_neu_gvt, valid_mode=valid_mode_gvt,
    ))
    print("  Done.")

print("\nSweep complete.")


# %%
_plot_sweep_histograms(
    results_gvt,
    g_sweep_vt["g_values"],
    sweep_param_name="g",
    suptitle=rf'Voltage-tanh network — timescale distributions across $g$  ($N={N_gvt}$, Gaussian $W$)',
)


# %% [markdown]
# ---
# ## Sweep 2: Multiple g Values (Levy-Stable W, fixed α, voltage tanh)
#
# Same structure as Sweep 1 but with a Levy-stable W instead of Gaussian.
# Fresh single W drawn from S(α, 0, 0, scale), then sweep g.


# %%
g_sweep_levy_vt = {
    "g_values":  [0.5, 1.2, 1.5, 1.8],
    "alpha_fixed": 1.2,
    "n_neurons":   1000,
    "tau":         1.0,
    "dt":          0.1,
    "duration":    5000.0,
    "noise_std":   0.0,
    "max_lag":     100,
    "fit_range":   (1, 60),
    # Choose normalization:
    #   "rnn"    → scale = 1/sqrt(N)    matches MultiTimescaleRNN levy_stable init
    #   "stable" → scale = 1/N^(1/alpha) spectral radius ~O(1), eigenspectra comparable
    "levy_normalization": "stable",
}

alpha_glvt    = g_sweep_levy_vt["alpha_fixed"]
norm_glvt     = g_sweep_levy_vt["levy_normalization"]
norm_label_glvt = r"1/\sqrt{N}" if norm_glvt == "rnn" else r"1/N^{1/\alpha}"
N_glvt        = g_sweep_levy_vt["n_neurons"]
W_glvt        = generate_W_levy(N_glvt, alpha_glvt, normalization=norm_glvt)
eigenvalues_W_glvt, eigenvectors_glvt = np.linalg.eig(W_glvt)
print(f"Levy W (α={alpha_glvt}, norm={norm_glvt}): spectral radius = {np.abs(eigenvalues_W_glvt).max():.4f}")


# %%
discard_glvt  = int(100 / g_sweep_levy_vt["dt"])
results_glvt  = []

for g in g_sweep_levy_vt["g_values"]:
    print(f"\n── g = {g} ──────────────────────────")
    tau_glvt = g_sweep_levy_vt["tau"]
    dt_glvt  = g_sweep_levy_vt["dt"]

    eigs_M_glvt   = get_jacobian_eigenvalues(eigenvalues_W_glvt, g, tau_glvt)
    stab_idx_glvt = get_stable_indices(eigs_M_glvt)
    tau_th_glvt   = get_effective_timescales(eigs_M_glvt)
    print(f"  Stable (linear theory) modes: {len(stab_idx_glvt)}/{N_glvt}")

    print("  Simulating voltage-tanh ...")
    act_glvt  = simulate_voltage_tanh(W_glvt, g, tau_glvt, dt_glvt,
                                       g_sweep_levy_vt["duration"], g_sweep_levy_vt["noise_std"])[discard_glvt:]
    proj_glvt = project_onto_eigenvectors(act_glvt, eigenvectors_glvt)

    print("  Fitting timescales ...")
    tau_neu_glvt, tau_mode_glvt, valid_neu_glvt, valid_mode_glvt = _fit_sweep_timescales(
        act_glvt, proj_glvt, stab_idx_glvt, N_glvt, dt_glvt,
        max_lag=g_sweep_levy_vt["max_lag"], fit_range=g_sweep_levy_vt["fit_range"],
    )
    print(f"  Valid neuron: {valid_neu_glvt.sum()}/{N_glvt}, eigenmode: {valid_mode_glvt.sum()}/{len(stab_idx_glvt)}")

    _plot_sweep_acf(act_glvt, proj_glvt, stab_idx_glvt, tau_th_glvt,
                    tau_neu_glvt, tau_mode_glvt, valid_neu_glvt, valid_mode_glvt,
                    dt_glvt, title_suffix=rf'$g={g}$  [voltage tanh, Lévy $W$, $\alpha={alpha_glvt}$]')

    results_glvt.append(dict(
        g=g, tau_th_all=tau_th_glvt, stab_idx=stab_idx_glvt,
        tau_neu=tau_neu_glvt, tau_mode=tau_mode_glvt,
        valid_neu=valid_neu_glvt, valid_mode=valid_mode_glvt,
    ))
    print("  Done.")

print("\nSweep complete.")


# %%
_plot_sweep_histograms(
    results_glvt,
    g_sweep_levy_vt["g_values"],
    sweep_param_name="g",
    suptitle=(
        rf'Voltage-tanh network — timescale distributions across $g$'
        rf'  ($\alpha={alpha_glvt}$, $N={N_glvt}$, Lévy $W$, scale $= {norm_label_glvt}$)'
    ),
)


# %% [markdown]
# ---
# ## Sweep 3: Multiple α Values (Levy-Stable W, voltage tanh)


# %%
alpha_sweep_vt = {
    "alpha_values": [0.9, 1.2, 1.5],
    "g_fixed":      1.8,
    "n_neurons":    1000,
    "tau":          1.0,
    "dt":           0.1,
    "duration":     5000.0,
    "noise_std":    0.0,
    "max_lag":      100,
    "fit_range":    (1, 60),
    # Choose normalization:
    #   "rnn"    → scale = 1/sqrt(N)    matches MultiTimescaleRNN levy_stable init
    #   "stable" → scale = 1/N^(1/alpha) spectral radius ~O(1), eigenspectra comparable
    "levy_normalization": "rnn",
}

norm_str_vt   = alpha_sweep_vt["levy_normalization"]
norm_label_vt = r"1/\sqrt{N}" if norm_str_vt == "rnn" else r"1/N^{1/\alpha}"
print(f"levy_normalization = '{norm_str_vt}' (scale = {norm_label_vt})")


# %%
discard_avt   = int(100 / alpha_sweep_vt["dt"])
results_avt   = []

for alpha_s in alpha_sweep_vt["alpha_values"]:
    print(f"\n── α = {alpha_s} ──────────────────────────")
    N_avt  = alpha_sweep_vt["n_neurons"]
    g_avt  = alpha_sweep_vt["g_fixed"]
    tau_avt = alpha_sweep_vt["tau"]
    dt_avt  = alpha_sweep_vt["dt"]
    norm    = alpha_sweep_vt["levy_normalization"]

    W_avt       = generate_W_levy(N_avt, alpha_s, normalization=norm)
    eigs_W_avt, eigenvectors_avt = np.linalg.eig(W_avt)
    print(f"  Spectral radius: {np.abs(eigs_W_avt).max():.3f}")

    eigs_M_avt   = get_jacobian_eigenvalues(eigs_W_avt, g_avt, tau_avt)
    stab_idx_avt = get_stable_indices(eigs_M_avt)
    tau_th_avt   = get_effective_timescales(eigs_M_avt)
    print(f"  Stable (linear theory) modes: {len(stab_idx_avt)}/{N_avt}")

    print("  Simulating voltage-tanh ...")
    act_avt  = simulate_voltage_tanh(W_avt, g_avt, tau_avt, dt_avt,
                                      alpha_sweep_vt["duration"], alpha_sweep_vt["noise_std"])[discard_avt:]
    proj_avt = project_onto_eigenvectors(act_avt, eigenvectors_avt)

    print("  Fitting timescales ...")
    tau_neu_avt, tau_mode_avt, valid_neu_avt, valid_mode_avt = _fit_sweep_timescales(
        act_avt, proj_avt, stab_idx_avt, N_avt, dt_avt,
        max_lag=alpha_sweep_vt["max_lag"], fit_range=alpha_sweep_vt["fit_range"],
    )
    print(f"  Valid neuron: {valid_neu_avt.sum()}/{N_avt}, eigenmode: {valid_mode_avt.sum()}/{len(stab_idx_avt)}")

    _plot_sweep_acf(act_avt, proj_avt, stab_idx_avt, tau_th_avt,
                    tau_neu_avt, tau_mode_avt, valid_neu_avt, valid_mode_avt,
                    dt_avt, title_suffix=rf'$\alpha={alpha_s}$  [voltage tanh, Lévy $W$]')

    results_avt.append(dict(
        alpha=alpha_s, skipped=False, tau_th_all=tau_th_avt, stab_idx=stab_idx_avt,
        tau_neu=tau_neu_avt, tau_mode=tau_mode_avt,
        valid_neu=valid_neu_avt, valid_mode=valid_mode_avt,
    ))
    print("  Done.")

print("\nSweep complete.")


# %%
_plot_sweep_histograms(
    results_avt,
    alpha_sweep_vt["alpha_values"],
    sweep_param_name=r"\alpha",
    suptitle=(
        rf'Voltage-tanh network — timescale distributions across $\alpha$'
        rf'  ($g={alpha_sweep_vt["g_fixed"]}$, $N={alpha_sweep_vt["n_neurons"]}$,'
        rf'  scale $= {norm_label_vt}$)'
    ),
)
