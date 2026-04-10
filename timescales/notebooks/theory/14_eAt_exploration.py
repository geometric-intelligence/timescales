# %% [markdown]
# # Fitting a Simple RNN to $e^{At}$
#
# A minimal exploration of how a linear RNN learns to approximate
# continuous-time dynamics $f(t) = e^{At}$.
#
# The model is a discrete-time linear RNN:
# $$\begin{align}
# r' &= W r \\
# y  &= C r
# \end{align}$$
#
# This is nearly trivial, but useful because we can define exactly the
# timescales we want the RNN to learn and study the loss landscape,
# gradient flow, and eigenvalue dynamics during training.

# %%
import os
import sys
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()
os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())

# %% [markdown]
# ## Step 1: Define model and target
#
# Ground truth function: $f(t) = e^{At}$
#
# For scalar $A$, this is just exponential decay/growth.
# For matrix $A$, it is the matrix exponential applied to an initial
# condition $v_0$.

# %%
def fn_rollout(T, A, v0=None):
    """Roll out the ground-truth continuous dynamics f(t) = e^{At} v0."""
    if np.ndim(A) == 0:
        return np.array([np.exp(A * t) for t in range(T)])
    else:
        if v0 is None:
            v0 = np.ones(A.shape[0])
        return np.array([expm(A * t) @ v0 for t in range(T)])


def rand_init(n_neurons, n_output, rs=4):
    """Random initialization for W, C, r0."""
    rng = np.random.default_rng(rs)
    W = rng.normal(0, 0.1, (n_neurons, n_neurons))
    C = rng.normal(0, 0.1, (n_output, n_neurons))
    r0 = rng.normal(1, 0.1, (n_neurons, 1))
    return W, C, r0


def rnn_rollout(T, W, C, r0):
    """
    Discrete linear RNN: r' = W r,  y = C r.

    Returns (T, n_out) outputs and (T, N) hidden states.
    """
    r = r0.flatten()
    rs, ys = [], []
    for _ in range(T):
        ys.append(C @ r)
        rs.append(r.copy())
        r = W @ r
    return np.array(ys), np.array(rs)


# %% [markdown]
# **Sanity check**: plot $e^{At}$ for several scalar $A$ values alongside
# the output of a randomly initialized 1-neuron RNN.

# %%
As = [np.array(-0.01), np.array(-0.1), np.array(-5)]
n_neurons, n_output = 1, 1
T = 50

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for i, A_true in enumerate(As):
    ax = axes[i]
    ts = list(range(T))
    ys_true = fn_rollout(T, A=A_true)
    ax.plot(ts, ys_true, label=f"$A = {A_true}$")

    W, C, r0 = rand_init(n_neurons, n_output, rs=i)
    ys, rs = rnn_rollout(T, W, C, r0)
    ax.plot(ts, ys, label="random RNN")
    ax.legend(fontsize=8)

plt.suptitle("$f(t) = e^{At}$")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2: Loss landscape
#
# $$\text{Loss} = f = \frac{1}{2T}\sum_t (y - \hat{y})^2
#   = \frac{1}{2T}\sum_t (C\,W^t\,r_0 - \hat{y})^2$$
#
# The loss contour over $(W, C)$ has a huge cliff at $|W| > 1$
# because the dynamics blow up. Hence we clip / log the loss to
# visualize the contour.

# %%
def loss(ys_pred, ys_true):
    n_out = ys_pred.shape[-1] if ys_pred.ndim > 1 else 1
    return np.mean(
        0.5 * (ys_pred.reshape(-1, n_out) - ys_true.reshape(-1, n_out)) ** 2
    )


# %%
As = [np.array(-0.01), np.array(-0.1), np.array(-5)]
T = 50
r0 = np.array([[1]])

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

for i, A_true in enumerate(As):
    ax = axes[i]
    ys_true = fn_rollout(T, A=A_true)

    x1_grid = np.linspace(-1.2, 1.2, 70)
    x2_grid = np.linspace(-10, 10, 70)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Loss = np.zeros_like(X1)
    for ii in range(X1.shape[0]):
        for jj in range(X1.shape[1]):
            W = np.array([[X1[ii, jj]]])
            C = np.array([[X2[ii, jj]]])
            ys_pred, _ = rnn_rollout(T, W, C, r0)
            Loss[ii, jj] = loss(ys_pred, ys_true)
    ax.contour(X1, X2, np.clip(Loss, 0, 0.7), 30, cmap="PiYG", alpha=0.3)

    true_W = np.exp(A_true)
    true_C = 1 / r0.item()
    ax.scatter([true_W], [true_C], marker="*", color="y",
               edgecolors="k", linewidths=0.5, s=50)
    ax.set_title(f"$A = {A_true}$")

plt.suptitle("Loss contour over $(W, C)$", fontweight="bold")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Step 3: Gradient descent
#
# $$\frac{\partial f}{\partial C} = \frac{1}{T} \sum_t (y - \hat{y})\, r(t)^\top$$
#
# $\frac{\partial f}{\partial W}$ requires tracking all the ways $r$
# affects the loss from each timestep (backpropagation through time).
#
# Gradient descent with step size $\alpha > 0$:
# $$x_{k+1} = x_k - \alpha\,\nabla f(x_k)$$

# %%
def grad_loss(T, W, C, r0, ys_true):
    """Compute analytic gradients dL/dW and dL/dC via BPTT."""
    ys_pred, rs_pred = rnn_rollout(T, W, C, r0)

    n_out = C.shape[0]
    ys_pred = ys_pred.reshape(T, n_out)
    ys_true_r = ys_true.reshape(T, n_out)

    e = (1 / T) * (ys_pred - ys_true_r)

    dC = e.T @ rs_pred

    g = C.T @ e[T - 1]
    dW = np.zeros_like(W)
    for t in range(T - 2, -1, -1):
        dW += np.outer(g, rs_pred[t])
        g = C.T @ e[t] + W.T @ g

    return dW, dC


def grad_descent(T, W, C, r0, ys_true, alpha=0.1):
    dW, dC = grad_loss(T, W, C, r0, ys_true)
    W = W - alpha * dW
    C = C - alpha * dC
    return W, C


# %% Loss contour with gradient descent trajectories
As = [np.array(-0.01), np.array(-0.1), np.array(-5)]
T = 10
r0 = np.array([[1]])

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

for idx, A_true in enumerate(As):
    ax = axes[idx]
    ys_true = fn_rollout(T, A=A_true)

    x1_grid = np.linspace(-1.2, 1.2, 70)
    x2_grid = np.linspace(-5, 5, 70)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Loss = np.zeros_like(X1)
    for ii in range(X1.shape[0]):
        for jj in range(X1.shape[1]):
            W = np.array([[X1[ii, jj]]])
            C = np.array([[X2[ii, jj]]])
            ys_pred, _ = rnn_rollout(T, W, C, r0)
            Loss[ii, jj] = loss(ys_pred, ys_true)
    ct = ax.contour(X1, X2, np.clip(Loss, 0, 1), 30, cmap="PiYG", alpha=0.3)

    true_W = np.exp(A_true)
    true_C = 1 / r0.item()
    ax.scatter([true_W], [true_C], marker="*", color="y",
               edgecolors="k", linewidths=0.5, s=50)

    n_gd_steps = 1000
    xs = np.linspace(-0.9, 0.9, 4)
    for x in xs:
        for y in xs:
            Xs = [[x, y]]
            W_gd, C_gd = np.array([[x]]), np.array([[y]])
            for _ in range(n_gd_steps - 1):
                W_gd, C_gd = grad_descent(T, W_gd, C_gd, r0, ys_true, alpha=0.01)
                Xs.append([W_gd.item(), C_gd.item()])
            Xs = np.array(Xs)
            ax.plot(Xs[:, 0], Xs[:, 1], c="k", alpha=0.5)

    ax.set_title(f"$A = {A_true}$")

plt.suptitle("Loss contour + GD trajectories", fontweight="bold")
plt.tight_layout()
fig.colorbar(ct, ax=axes[-1], label="Loss")
plt.show()

# %% [markdown]
# **Observations:**
# - Fast timescales (large $|A|$) are harder to learn because the signal
#   is concentrated near $t = 0$. Sometimes the optimizer diverges toward
#   the $y$-axis ($W \to 0$), learning a trivial zero-output solution.

# %% [markdown]
# ## Step 4: Learning
#
# Train W and C via gradient descent and track:
# 1. The fit (target vs learned output)
# 2. Loss over training steps
# 3. Eigenvalue trajectory of W in the complex plane

# %%
def do_learning_plots(A_true, T, n_neurons, n_steps=30000, alpha=1e-2, rs=6):
    """Train a linear RNN to fit e^{At} and visualize the learning dynamics."""
    ys_true = fn_rollout(T, A_true)
    n_output = 1 if len(ys_true.shape) == 1 else ys_true.shape[1]
    W, C, r0 = rand_init(n_neurons, n_output, rs=rs)

    losses, eigs_traj = [], []
    for step in range(n_steps):
        ys_pred, _ = rnn_rollout(T, W, C, r0)
        losses.append(loss(ys_pred, ys_true))
        W, C = grad_descent(T, W, C, r0, ys_true, alpha=alpha)
        eigs_traj.append(np.linalg.eigvals(W))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ts = np.arange(T, dtype=float)
    ys_pred, _ = rnn_rollout(T, W, C, r0)
    if ys_true.ndim == 1:
        axes[0].plot(ts, ys_true, color="pink", label="target")
        axes[0].plot(ts, ys_pred, "--", color="green", label="RNN")
    else:
        for dim in range(n_output):
            axes[0].plot(ts, ys_true[:, dim], color="pink",
                         label=f"target dim {dim}")
            axes[0].plot(ts, ys_pred[:, dim], "--", color="green",
                         label=f"RNN dim {dim}")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Fit")

    axes[1].semilogy(losses)
    axes[1].set_xlabel("Step")
    axes[1].set_title("Loss over training")

    eigs_W = np.linalg.eigvals(W)
    if A_true.ndim == 0:
        eigs_A = np.array([np.exp(A_true)])
    else:
        eigs_A = np.exp(np.linalg.eigvals(A_true))

    eigs_traj = np.array(eigs_traj)
    subsample = 100
    if eigs_W.shape[0] == 1:
        sc = axes[2].scatter(
            eigs_traj[::subsample].real, eigs_traj[::subsample].imag,
            c=list(range(0, n_steps, subsample)), alpha=0.2, cmap="PiYG", s=5,
        )
    else:
        for i in range(eigs_W.shape[0]):
            sc = axes[2].scatter(
                eigs_traj[::subsample, i].real,
                eigs_traj[::subsample, i].imag,
                c=list(range(0, n_steps, subsample)),
                alpha=0.2, cmap="PiYG", s=5,
            )

    theta = np.linspace(0, 2 * np.pi, 300)
    axes[2].plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2)
    axes[2].scatter(eigs_W.real, eigs_W.imag, color="green", s=100, zorder=10,
                    label="learned eigs(W)")
    axes[2].scatter(eigs_A.real, eigs_A.imag, marker="*", s=500, color="pink",
                    zorder=6, label="$e^{\\lambda_A}$")
    axes[2].set_aspect("equal")
    axes[2].set_title("Eigenvalues of $W$")
    axes[2].legend(fontsize=8, loc="lower left")
    plt.colorbar(sc, ax=axes[2], label="training step")

    print("True A eigenvalues (mapped to discrete):", eigs_A)
    print("Learned W eigenvalues:", eigs_W)

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### Scalar $A$

# %%
A_true = np.array(-0.1)
T = 10
n_neurons = 1
do_learning_plots(A_true, T, n_neurons, n_steps=30000, alpha=1e-2, rs=6)

# %% [markdown]
# ### Matrix $A$

# %%
A_true = np.array([[-0.1, 0.0], [0.0, -1.0]])
T = 10
n_neurons = 2
do_learning_plots(A_true, T, n_neurons, n_steps=30000, alpha=1e-2, rs=6)

# %% [markdown]
# ### Mismatch between $A$ and $W$ (fewer neurons than modes)
#
# What happens when the RNN has fewer neurons than the dimensionality
# of $A$? It must approximate a 2D system with a 1D RNN.

# %%
A_true = np.array([[-0.1, 0.0], [0.0, -1.0]])
T = 10
n_neurons = 1
do_learning_plots(A_true, T, n_neurons, n_steps=30000, alpha=1e-2, rs=6)

# %% [markdown]
# ### Rotation matrix $A$
#
# $A$ has purely imaginary eigenvalues ($\pm i$), producing oscillatory
# dynamics. Can the RNN learn to rotate?

# %%
A_true = np.array([[0, -1], [1, 0]])
T = 10
n_neurons = 2
do_learning_plots(A_true, T, n_neurons, n_steps=30000, alpha=1e-2, rs=6)
