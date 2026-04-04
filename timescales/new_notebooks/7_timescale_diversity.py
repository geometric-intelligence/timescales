# %% [markdown]
# # Timescale Diversity: How Heterogeneous Intrinsic Timescales Shape Effective Dynamics
#
# **Goal**: Understand how varying the distribution of intrinsic timescales
# $\tau_i$ across neurons affects the effective timescales of a linear network.
#
# Dynamics: $T \dot{r} = -r + gWr$, where $T = \text{diag}(\tau_1, \ldots, \tau_N)$.
#
# Rearranging: $\dot{r} = T^{-1}(gW - I)r = Mr$
#
# The Jacobian is $M = T^{-1}(gW - I)$, and its eigenvalues determine the
# effective timescales: $\tau_{\text{eff}} = -1 / \text{Re}(\lambda_M)$.
#
# **Key question**: When all neurons share a single $\tau$, the Jacobian is just
# $(gW - I)/\tau$ — a simple rescaling. But when $\tau_i$ vary, $T^{-1}$ is a
# non-trivial diagonal matrix that *mixes* with $gW - I$ in a way that can
# create new effective timescales not present in either factor alone.


# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

SEED = 42
rng = np.random.default_rng(SEED)

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12


# %%
# --- Core functions ---

def generate_W(N: int, rng: np.random.Generator) -> np.ndarray:
    """Random Gaussian connectivity W_ij ~ N(0, 1/N)."""
    return rng.normal(0, 1.0 / np.sqrt(N), size=(N, N)).astype(np.float64)


def compute_jacobian_homogeneous(W: np.ndarray, g: float, tau: float) -> np.ndarray:
    """M = (gW - I) / tau — all neurons share the same timescale."""
    N = W.shape[0]
    return (g * W - np.eye(N)) / tau


def compute_jacobian_heterogeneous(
    W: np.ndarray, g: float, taus: np.ndarray
) -> np.ndarray:
    """M = T^{-1}(gW - I), where T = diag(tau_1, ..., tau_N)."""
    N = W.shape[0]
    T_inv = np.diag(1.0 / taus)
    return T_inv @ (g * W - np.eye(N))


def effective_timescales(M: np.ndarray) -> np.ndarray:
    """tau_eff = -1/Re(lambda) for stable modes (Re(lambda) < 0)."""
    eigs = np.linalg.eigvals(M)
    real_parts = eigs.real
    stable = real_parts < 0
    return -1.0 / real_parts[stable]


# %% [markdown]
# ## 1. Baseline: Homogeneous Timescale
#
# When all $\tau_i = \tau$, the Jacobian is $(gW - I)/\tau$.
# Eigenvalues of $M$ are $(g\lambda_W - 1)/\tau$, so the effective
# timescales are $\tau / (1 - g\,\text{Re}(\lambda_W))$.
#
# This is a simple rescaling of the connectivity spectrum.

# %%
N = 500
g_values = [0.5, 0.9, 0.99]
tau_homo = 1.0

W = generate_W(N, rng)
eigs_W = np.linalg.eigvals(W)

fig, axes = plt.subplots(1, len(g_values), figsize=(5 * len(g_values), 4), sharey=True)

for ax, g in zip(axes, g_values):
    M = compute_jacobian_homogeneous(W, g, tau_homo)
    tau_effs = effective_timescales(M)

    ax.hist(tau_effs, bins=60, color="steelblue", alpha=0.7, edgecolor="white",
            density=True)
    ax.axvline(tau_homo, color="red", linestyle="--", linewidth=1.2,
               label=f"$\\tau$ = {tau_homo}")
    ax.set_xlabel("$\\tau_{\\mathrm{eff}}$", fontsize=13)
    if ax == axes[0]:
        ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"g = {g}", fontsize=13)
    ax.set_xlim(0, min(10, np.percentile(tau_effs, 99) * 1.5))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

fig.suptitle(f"Effective Timescales — Homogeneous $\\tau = {tau_homo}$, N = {N}",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 2. Heterogeneous Timescales: Uniform Distribution
#
# Now let $\tau_i \sim \text{Uniform}[\tau_{\min}, \tau_{\max}]$.
# $T^{-1}$ no longer commutes with $gW$, so the effective timescales
# are not simply a rescaling of the connectivity spectrum.

# %%
tau_ranges = [
    (1.0, 1.0),    # homogeneous (control)
    (0.5, 1.5),    # mild spread
    (0.1, 2.0),    # moderate spread
    (0.1, 10.0),   # wide spread
]
g = 0.9

fig, axes = plt.subplots(1, len(tau_ranges), figsize=(4.5 * len(tau_ranges), 4),
                          sharey=True)

for ax, (tau_lo, tau_hi) in zip(axes, tau_ranges):
    if tau_lo == tau_hi:
        taus = np.full(N, tau_lo)
        label = f"$\\tau$ = {tau_lo}"
    else:
        taus = rng.uniform(tau_lo, tau_hi, size=N)
        label = f"$\\tau \\sim$ U[{tau_lo}, {tau_hi}]"

    M = compute_jacobian_heterogeneous(W, g, taus)
    tau_effs = effective_timescales(M)

    ax.hist(tau_effs, bins=60, color="steelblue", alpha=0.7, edgecolor="white",
            density=True)
    ax.axvline(np.mean(taus), color="red", linestyle="--", linewidth=1.2,
               label=f"mean $\\tau$ = {np.mean(taus):.1f}")
    ax.set_xlabel("$\\tau_{\\mathrm{eff}}$", fontsize=13)
    if ax == axes[0]:
        ax.set_ylabel("Density", fontsize=13)
    ax.set_title(label, fontsize=11)
    ax.set_xlim(0, min(20, np.percentile(tau_effs, 99) * 1.3))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

fig.suptitle(f"Effect of Timescale Spread — g = {g}, N = {N}",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 3. Log-Normal Timescale Distribution
#
# Biological neural circuits show log-normally distributed timescales.
# $\tau_i \sim \text{LogNormal}(\mu, \sigma)$

# %%
sigma_values = [0.0, 0.3, 0.7, 1.0, 1.5]
mu_log = 0.0  # median tau = exp(mu) = 1.0
g = 0.9

fig, axes = plt.subplots(2, len(sigma_values),
                          figsize=(3.5 * len(sigma_values), 7))

for col, sigma in enumerate(sigma_values):
    if sigma == 0:
        taus = np.ones(N)
    else:
        taus = rng.lognormal(mu_log, sigma, size=N)

    M = compute_jacobian_heterogeneous(W, g, taus)
    tau_effs = effective_timescales(M)

    # Top row: tau distribution
    ax_top = axes[0, col]
    if sigma == 0:
        ax_top.axvline(1.0, color="steelblue", linewidth=2)
        ax_top.set_xlim(0, 3)
    else:
        ax_top.hist(taus, bins=50, color="#8da0b5", alpha=0.7, edgecolor="white",
                     density=True)
        ax_top.set_xlim(0, min(np.percentile(taus, 99) * 1.3, 30))
    ax_top.set_title(f"$\\sigma$ = {sigma}", fontsize=12)
    if col == 0:
        ax_top.set_ylabel("Intrinsic $\\tau_i$\ndensity", fontsize=11)
    ax_top.grid(True, alpha=0.15)

    # Bottom row: effective timescale distribution
    ax_bot = axes[1, col]
    ax_bot.hist(tau_effs, bins=60, color="#e76f51", alpha=0.7, edgecolor="white",
                density=True)
    ax_bot.set_xlabel("$\\tau_{\\mathrm{eff}}$", fontsize=12)
    if col == 0:
        ax_bot.set_ylabel("Effective $\\tau_{\\mathrm{eff}}$\ndensity", fontsize=11)
    ax_bot.set_xlim(0, min(20, np.percentile(tau_effs, 99) * 1.3))
    ax_bot.grid(True, alpha=0.15)

    ax_bot.annotate(f"median = {np.median(tau_effs):.2f}\nmax = {np.max(tau_effs):.1f}",
                    xy=(0.95, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.suptitle(f"Log-Normal Intrinsic Timescales → Effective Timescales — g = {g}, N = {N}",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. Gain × Timescale Spread Interaction
#
# How do $g$ and the spread of $\tau_i$ interact?
# At low $g$, the network is dominated by the leak term $-T^{-1}r$,
# so effective timescales ≈ intrinsic timescales.
# At high $g$, connectivity dominates and the interaction with $T^{-1}$
# creates emergent slow modes.

# %%
g_values = [0.3, 0.6, 0.9, 0.99]
sigma_values_2d = [0.0, 0.5, 1.0, 1.5]

fig, axes = plt.subplots(len(sigma_values_2d), len(g_values),
                          figsize=(3.5 * len(g_values), 3 * len(sigma_values_2d)),
                          sharex=True)

for row, sigma in enumerate(sigma_values_2d):
    if sigma == 0:
        taus = np.ones(N)
    else:
        taus = rng.lognormal(mu_log, sigma, size=N)

    for col, g in enumerate(g_values):
        ax = axes[row, col]
        M = compute_jacobian_heterogeneous(W, g, taus)
        tau_effs = effective_timescales(M)

        ax.hist(tau_effs, bins=50, color="#e76f51", alpha=0.7,
                edgecolor="white", density=True)
        ax.set_xlim(0, min(30, np.percentile(tau_effs, 99) * 1.3))
        ax.grid(True, alpha=0.15)

        if row == 0:
            ax.set_title(f"g = {g}", fontsize=12)
        if col == 0:
            lbl = "homo" if sigma == 0 else f"$\\sigma$ = {sigma}"
            ax.set_ylabel(f"{lbl}\ndensity", fontsize=10)
        if row == len(sigma_values_2d) - 1:
            ax.set_xlabel("$\\tau_{\\mathrm{eff}}$", fontsize=11)

        ax.annotate(f"max={np.max(tau_effs):.1f}",
                    xy=(0.95, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.suptitle("Gain × Timescale Diversity Interaction",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5. Summary Statistics: Max & Spread of Effective Timescales
#
# Collapse the distributions into scalar summaries to see the trends clearly.

# %%
g_sweep = np.linspace(0.1, 0.99, 30)
sigma_sweep = [0.0, 0.3, 0.7, 1.0, 1.5]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(sigma_sweep)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for sigma, color in zip(sigma_sweep, colors):
    max_taus = []
    median_taus = []
    iqr_taus = []

    for g in g_sweep:
        if sigma == 0:
            taus = np.ones(N)
        else:
            taus = rng.lognormal(mu_log, sigma, size=N)

        M = compute_jacobian_heterogeneous(W, g, taus)
        tau_effs = effective_timescales(M)

        max_taus.append(np.max(tau_effs))
        median_taus.append(np.median(tau_effs))
        iqr_taus.append(np.percentile(tau_effs, 90) - np.percentile(tau_effs, 10))

    label = "homo" if sigma == 0 else f"$\\sigma$ = {sigma}"
    axes[0].plot(g_sweep, max_taus, color=color, linewidth=2, label=label)
    axes[1].plot(g_sweep, median_taus, color=color, linewidth=2, label=label)
    axes[2].plot(g_sweep, iqr_taus, color=color, linewidth=2, label=label)

for ax, ylabel, title in zip(axes,
    ["$\\max(\\tau_{\\mathrm{eff}})$", "median($\\tau_{\\mathrm{eff}}$)", "90th–10th percentile"],
    ["Slowest Mode", "Typical Timescale", "Timescale Spread"]):
    ax.set_xlabel("Recurrent gain $g$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.15, which="both")
    ax.legend(fontsize=8)

fig.suptitle(f"Effect of Intrinsic Timescale Diversity on Network Dynamics (N = {N})",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Eigenvalue Spectra in the Complex Plane
#
# Visualize how $T^{-1}$ reshapes the eigenvalue cloud of $gW - I$.

# %%
g = 0.9
sigma_vals_plane = [0.0, 0.5, 1.0, 1.5]

fig, axes = plt.subplots(1, len(sigma_vals_plane),
                          figsize=(4.5 * len(sigma_vals_plane), 4.5))

for ax, sigma in zip(axes, sigma_vals_plane):
    if sigma == 0:
        taus = np.ones(N)
    else:
        taus = rng.lognormal(mu_log, sigma, size=N)

    M = compute_jacobian_heterogeneous(W, g, taus)
    eigs = np.linalg.eigvals(M)

    ax.scatter(eigs.real, eigs.imag, s=4, alpha=0.5, c="#e76f51")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0, color="#bbb", linewidth=0.3)
    ax.set_xlabel("Re($\\lambda$)", fontsize=12)
    if ax == axes[0]:
        ax.set_ylabel("Im($\\lambda$)", fontsize=12)
    lbl = "homo" if sigma == 0 else f"$\\sigma$ = {sigma}"
    ax.set_title(lbl, fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

    n_unstable = np.sum(eigs.real > 0)
    ax.annotate(f"unstable: {n_unstable}/{N}",
                xy=(0.95, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.suptitle(f"Jacobian Eigenvalue Spectra — g = {g}, N = {N}",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Discrete Timescale Groups
#
# What if instead of a continuous distribution, we have K discrete groups
# of neurons with distinct timescales? E.g., fast ($\tau=0.5$) and slow ($\tau=5$).

# %%
g = 0.9
group_configs = [
    {"label": "1 group: $\\tau$=1", "taus": [1.0], "fracs": [1.0]},
    {"label": "2 groups: 0.5, 5", "taus": [0.5, 5.0], "fracs": [0.5, 0.5]},
    {"label": "2 groups: 0.1, 10", "taus": [0.1, 10.0], "fracs": [0.5, 0.5]},
    {"label": "3 groups: 0.1, 1, 10", "taus": [0.1, 1.0, 10.0], "fracs": [0.33, 0.34, 0.33]},
    {"label": "2 groups (90/10): 1, 10", "taus": [1.0, 10.0], "fracs": [0.9, 0.1]},
]

fig, axes = plt.subplots(2, len(group_configs),
                          figsize=(3.5 * len(group_configs), 7))

for col, cfg in enumerate(group_configs):
    tau_list = cfg["taus"]
    frac_list = cfg["fracs"]

    taus = np.zeros(N)
    idx = 0
    for tau_val, frac in zip(tau_list, frac_list):
        n_group = int(round(frac * N))
        taus[idx:idx + n_group] = tau_val
        idx += n_group
    taus[idx:] = tau_list[-1]

    M = compute_jacobian_heterogeneous(W, g, taus)
    tau_effs = effective_timescales(M)
    eigs = np.linalg.eigvals(M)

    # Top: effective timescale histogram
    ax_top = axes[0, col]
    ax_top.hist(tau_effs, bins=50, color="#e76f51", alpha=0.7, edgecolor="white",
                density=True)
    for tv in tau_list:
        ax_top.axvline(tv, color="#3b82f6", linestyle=":", linewidth=1.2, alpha=0.6)
    ax_top.set_xlim(0, min(30, np.percentile(tau_effs, 99) * 1.5))
    ax_top.set_title(cfg["label"], fontsize=10)
    if col == 0:
        ax_top.set_ylabel("$\\tau_{\\mathrm{eff}}$ density", fontsize=11)
    ax_top.grid(True, alpha=0.15)

    # Bottom: complex plane
    ax_bot = axes[1, col]
    ax_bot.scatter(eigs.real, eigs.imag, s=4, alpha=0.5, c="#e76f51")
    ax_bot.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_bot.set_aspect("equal")
    if col == 0:
        ax_bot.set_ylabel("Im($\\lambda$)", fontsize=11)
    ax_bot.set_xlabel("Re($\\lambda$)", fontsize=11)
    ax_bot.grid(True, alpha=0.15)

fig.suptitle(f"Discrete Timescale Groups — g = {g}, N = {N}",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
