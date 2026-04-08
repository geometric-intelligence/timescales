# %% [markdown]
# # Coupled RNN: nonlinear r-network + linear s-network
#
# Two coupled networks:
# $$\tau_r \dot{r} = -r + \phi(W^{rec} r + W^{in} u + V s)$$
# $$\tau_s \dot{s} = -s + W^s s + U r$$
# $$\hat{y} = W^{out} r$$
#
# Trainable: $W^{rec}, W^{in}, V, U, W^{out}$.  Fixed: $W^s, \tau_r, \tau_s$.
#
# This notebook:
# 1. Instantiates a CoupledRNN + flip-flop data
# 2. Sanity-checks the forward pass
# 3. Inspects the fixed $W^s$ eigenspectrum
# 4. Runs a short training loop and plots the loss curve

# %%
import os
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

gitroot = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
).strip()

os.chdir(os.path.join(gitroot, "timescales"))
sys.path.append(gitroot)
sys.path.append(os.getcwd())
print("Working directory:", os.getcwd())

from rnns.coupled_rnn import CoupledRNN, CoupledRNNLightning
from datamodules.flip_flop import FlipFlopDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# %% [markdown]
# ## 1  Instantiate model and data

# %%
N_BITS = 3

model = CoupledRNN(
    input_size=N_BITS,
    r_hidden_size=256,
    s_hidden_size=64,
    output_size=N_BITS,
    dt=0.1,
    tau_r=0.5,
    tau_s=5.0,
    activation=nn.Tanh,
    zero_diag_wrec=True,
    recurrent_gain=1.0,
    noise_std=0.0,
    wrec_init="orthogonal",
    w_s_gain=1.0,
).to(device)

dm = FlipFlopDataModule(
    n_bits=N_BITS,
    p_pulse=0.05,
    pulse_amplitude=1.0,
    num_time_steps=500,
    num_val_trajectories=200,
    batch_size=64,
)
dm.setup()

print(f"r_hidden_size = {model.r_hidden_size}")
print(f"s_hidden_size = {model.s_hidden_size}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# ## 2  Forward-pass sanity check

# %%
inp, _, tgt = dm.val_dataset.tensors
inp_batch = inp[:4].to(device)

with torch.no_grad():
    r_states, s_states, outputs = model(inp_batch)

print(f"r_states : {r_states.shape}")    # (4, 500, 256)
print(f"s_states : {s_states.shape}")    # (4, 500, 64)
print(f"outputs  : {outputs.shape}")     # (4, 500, 3)

# Quick plot: output sigmoids vs targets for the first sequence
t_arr = np.arange(inp.shape[1])
out_prob = torch.sigmoid(outputs[0].cpu()).numpy()
tgt_np = tgt[0].numpy()
inp_np = inp[0].numpy()

fig, axes = plt.subplots(N_BITS, 1, figsize=(12, 2 * N_BITS), sharex=True)
for bit in range(N_BITS):
    ax = axes[bit]
    ax.step(t_arr, tgt_np[:, bit], where="post", color="black", lw=1.5, label="target")
    ax.plot(t_arr, out_prob[:, bit], color="steelblue", lw=1.2, alpha=0.8, label="output (untrained)")
    pulse = inp_np[:, bit]
    set_mask = pulse > 0.5
    reset_mask = pulse < -0.5
    if set_mask.any():
        ax.scatter(t_arr[set_mask], np.full(set_mask.sum(), 0.65),
                   marker=6, s=20, color="C2", zorder=3)
    if reset_mask.any():
        ax.scatter(t_arr[reset_mask], np.full(reset_mask.sum(), 0.35),
                   marker=7, s=20, color="C3", zorder=3)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel(f"Bit {bit}")
axes[-1].set_xlabel("Timestep")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Untrained CoupledRNN — Flip-Flop output", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3  $W^s$ eigenspectrum (fixed)

# %%
W_s_np = model.rnn_step.W_s.cpu().numpy()
eigs_s = np.linalg.eigvals(W_s_np)

dt = model.dt
tau_s = model.tau_s
alpha_s = 1.0 - np.exp(-dt / tau_s)
J_s = (1.0 - alpha_s) * np.eye(W_s_np.shape[0]) + alpha_s * W_s_np
eigs_J = np.linalg.eigvals(J_s)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
theta = np.linspace(0, 2 * np.pi, 200)

ax = axes[0]
ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3)
ax.scatter(eigs_s.real, eigs_s.imag, s=18, alpha=0.7, c="#2a9d8f")
ax.set_aspect("equal")
ax.set_title("$W^s$ eigenvalues")
ax.set_xlabel("Re"); ax.set_ylabel("Im")
ax.grid(True, alpha=0.15)

ax = axes[1]
ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="$|\\lambda|=1$")
ax.scatter(eigs_J.real, eigs_J.imag, s=18, alpha=0.7, c="#e76f51")
ax.set_aspect("equal")
ax.set_title("Discrete-time Jacobian $J_s = (1-\\alpha_s)I + \\alpha_s W^s$")
ax.set_xlabel("Re"); ax.set_ylabel("Im")
ax.grid(True, alpha=0.15)
ax.legend(fontsize=8)

max_abs = np.max(np.abs(eigs_J))
ax.annotate(f"$|\\lambda|_{{max}}$ = {max_abs:.3f}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.suptitle("Linear s-network spectrum", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4  Short training loop (no WandB, pure PyTorch)

# %%
from lightning import Trainer

lit = CoupledRNNLightning(
    model=model,
    learning_rate=1e-3,
    weight_decay=1e-4,
    step_size=500,
    gamma=0.5,
    task="flip_flop",
    lr_interval="step",
)

trainer = Trainer(
    max_epochs=50,
    limit_train_batches=10,
    enable_checkpointing=False,
    logger=False,
    accelerator="auto",
    devices=1,
)

trainer.fit(lit, dm.train_dataloader(), dm.val_dataloader())

# %% Plot training loss from trainer logged metrics
val_losses = []
for cb in trainer.callbacks:
    if hasattr(cb, "val_losses"):
        val_losses = cb.val_losses
        break

if val_losses:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(val_losses, marker="o", markersize=3, linewidth=1.5, color="#264653")
    ax.set_xlabel("Validation epoch")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title("CoupledRNN training — Flip-Flop", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
else:
    print("No val losses recorded (expected when logger=False and no LossLoggerCallback)")

# %% [markdown]
# ## 5  Post-training: output trajectories

# %%
lit.eval()
inp_eval = inp[:4].to(device)
with torch.no_grad():
    _, _, out_eval = lit.model(inp_eval)
out_prob_eval = torch.sigmoid(out_eval[0].cpu()).numpy()

fig, axes = plt.subplots(N_BITS, 1, figsize=(12, 2 * N_BITS), sharex=True)
for bit in range(N_BITS):
    ax = axes[bit]
    ax.step(t_arr, tgt_np[:, bit], where="post", color="black", lw=1.5, label="target")
    ax.plot(t_arr, out_prob_eval[:, bit], color="steelblue", lw=1.2, alpha=0.9, label="output (trained)")
    pulse = inp_np[:, bit]
    set_mask = pulse > 0.5
    reset_mask = pulse < -0.5
    if set_mask.any():
        ax.scatter(t_arr[set_mask], np.full(set_mask.sum(), 0.65),
                   marker=6, s=20, color="C2", zorder=3)
    if reset_mask.any():
        ax.scatter(t_arr[reset_mask], np.full(reset_mask.sum(), 0.35),
                   marker=7, s=20, color="C3", zorder=3)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel(f"Bit {bit}")
axes[-1].set_xlabel("Timestep")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Trained CoupledRNN — Flip-Flop output", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
