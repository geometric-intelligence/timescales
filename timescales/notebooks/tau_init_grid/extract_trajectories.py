"""Extract hidden-state trajectories and their projections for a completed sweep.

Runs where the checkpoints live and emits one compact .npz so only small files have
to be copied back. Consumed by 4_trajectory_geometry.

Three coordinate systems are produced for the same trajectories, so they can be
compared directly:

* **PCA** — the usual variance-ordered basis of the hidden states.
* **Jacobian eigenmodes** — the coefficients ``c(t) = V^-1 h(t)``. Note the inverse:
  ``V`` is not orthogonal, so projecting with ``V.T`` would *not* give the coefficient
  of ``h`` in the eigenbasis. For a complex-conjugate pair, ``(Re c_j, Im c_j)`` traces
  a rotation at the mode's own frequency, which is what makes "how is each frequency
  represented" answerable.
* **Output-coupled modes** — for each output channel, the single mode it reads from
  most strongly, ``argmax_j |W_out V|_{kj}``. This is the same selection rule used by
  the matching metric in ``timescales.mode_analysis``.

It also records the overlap between the leading principal axes and the leading
eigenmodes, which says whether PCA is recovering the dynamical modes or mixing them.

    SWEEP_DIR=logs/experiments/tau_init_grid OUT=/tmp/traj.npz \\
        python extract_trajectories.py

Defaults assume it is run from the inner ``timescales`` package directory.
"""

import os
import sys

import numpy as np
import torch
import yaml

SWEEP = os.environ.get(
    "SWEEP_DIR", os.path.join(os.getcwd(), "logs", "experiments", "tau_init_grid"))
OUT = os.environ.get("OUT", "/tmp/traj.npz")
_here = os.getcwd()
for _cand in (_here, os.path.abspath(os.path.join(_here, ".."))):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

SCHEME = os.environ.get("SCHEME", "uniform")   # focus of this pass
TASKS = ("ff", "sine")
GS = (0.3, 0.9)
SEEDS = (0, 1, 2)
N_TRAJ = 4        # trajectories kept per condition
N_PC = 10         # principal axes kept
BATCH = 32        # trajectories per forward pass
TIME_SUB = {"ff": 2, "sine": 1}   # subsampling for the raw states kept for UMAP


def _load_model(config, state, strip_prefix):
    from train import _create_rnn_model
    model, _ = _create_rnn_model(config)
    if strip_prefix:
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    return model.to(DEV).eval()


def _forward(model, batch, config):
    """Hidden states and outputs for one validation batch."""
    inputs = batch[0][:BATCH].to(DEV)
    kw = {}
    ihv = config.get("init_hidden_value")
    if ihv is not None:
        kw["init_hidden"] = torch.full((inputs.shape[0], config["hidden_size"]),
                                       float(ihv), device=DEV)
    with torch.no_grad():
        h, y = model(inputs=inputs, **kw)
    return h.cpu().numpy(), y.cpu().numpy(), batch[2][:BATCH].cpu().numpy()


def _pca(H, n_pc):
    """Variance-ordered basis of [B,T,N] states. Returns (basis[n_pc,N], evr, mean)."""
    X = H.reshape(-1, H.shape[-1])
    mu = X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = S ** 2
    return Vt[:n_pc], (var / var.sum())[:n_pc], mu


def _mode_coefficients(H, V):
    """c(t) = V^-1 h(t): the coefficient of each state in the eigenbasis.

    Solved rather than inverted explicitly, for conditioning.
    """
    X = H.reshape(-1, H.shape[-1]).T                 # (N, B*T)
    C = np.linalg.solve(V, X.astype(complex))        # (N modes, B*T)
    return C.T.reshape(H.shape[0], H.shape[1], -1)   # [B, T, N] complex


def run():
    out = {}
    for task in TASKS:
        for g in GS:
            for seed in SEEDS:
                cond = f"{task}_{SCHEME}_full_g{g:g}"
                d = f"{SWEEP}/{cond}/seed_{seed}"
                cfg_p = f"{d}/config_seed{seed}.yaml"
                fin_p = f"{d}/final_model_seed{seed}.pth"
                spec_p = f"{d}/spectral_final.pt"
                if not all(os.path.exists(p) for p in (cfg_p, fin_p, spec_p)):
                    continue
                with open(cfg_p) as f:
                    config = yaml.safe_load(f)
                config["num_workers"] = 0

                from train import create_datamodule
                dm = create_datamodule(config)
                dm.prepare_data()
                dm.setup()
                batch = next(iter(dm.val_dataloader()))

                model = _load_model(
                    config, torch.load(fin_p, map_location="cpu", weights_only=False), False)
                H, Y, T = _forward(model, batch, config)

                blob = torch.load(spec_p, map_location="cpu", weights_only=False)
                V = np.asarray(blob["V"], dtype=complex)
                lam = np.asarray(blob["eigvals_eig"], dtype=complex)
                W_out = np.asarray(blob["W_out"], dtype=float)

                basis, evr, mu = _pca(H, N_PC)
                proj = ((H.reshape(-1, H.shape[-1]) - mu) @ basis.T).reshape(
                    H.shape[0], H.shape[1], N_PC)
                C = _mode_coefficients(H, V)

                # Which mode each output channel reads from most strongly.
                coup = np.abs(W_out @ V)                   # (n_out, N)
                top = np.argmax(coup, axis=1)              # (n_out,)

                # Overlap between principal axes and eigenmodes: |<pc_i, v_j>| with
                # both normalized, so 1 means the axis is that mode.
                Vn = V / np.linalg.norm(V, axis=0, keepdims=True)
                overlap = np.abs(basis.astype(complex) @ Vn)   # (n_pc, N)

                k = f"{task}_g{g:g}_seed{seed}"
                out[f"{k}_evr"] = evr.astype(np.float32)
                out[f"{k}_proj"] = proj[:N_TRAJ].astype(np.float32)
                out[f"{k}_coef_top"] = C[:N_TRAJ][:, :, top].astype(np.complex64)
                out[f"{k}_top_idx"] = top.astype(np.int32)
                out[f"{k}_top_lam"] = lam[top].astype(np.complex64)
                out[f"{k}_targets"] = T[:N_TRAJ].astype(np.float32)
                out[f"{k}_outputs"] = Y[:N_TRAJ].astype(np.float32)
                out[f"{k}_overlap"] = overlap[:, :].astype(np.float32)
                out[f"{k}_lam"] = lam.astype(np.complex64)
                if seed == 0:
                    sub = TIME_SUB[task]
                    out[f"{k}_H"] = H[:N_TRAJ, ::sub].astype(np.float32)
                print(f"  {cond} seed{seed}: H{H.shape} "
                      f"PC1-3={100 * evr[:3].sum():.1f}% top_modes={top.tolist()}",
                      flush=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}: {len(out)} arrays")


if __name__ == "__main__":
    print(f"device: {DEV}  sweep: {SWEEP}  scheme: {SCHEME}")
    run()
