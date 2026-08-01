"""Extract hidden-state geometry and spectral timescales from a completed sweep.

Runs where the sweep lives (it needs the checkpoints), and emits three compact
artefacts so only small files have to be copied back:
  spectral_timescales.npz  - network decay / oscillation timescale distributions,
                             from the Jacobian eigenvalues at init and after training
  pca_dims.csv             - PCA dimensionality of the hidden-state code per run
  pca_traj.npz             - example hidden-state trajectories projected into the
                             TRAINED network's PCA basis (trained + untrained)

Consumed by 2_dynamics_and_geometry, which reads the three files from the sweep
directory. Paths are taken from the environment so this is not tied to one machine:

    SWEEP_DIR=logs/experiments/tau_init_grid OUT_DIR=/tmp/dyn_out \
        python extract_dynamics.py

Defaults assume it is run from the inner ``timescales`` package directory.
"""
import csv
import glob
import os
import sys

import numpy as np
import torch
import yaml

SWEEP = os.environ.get("SWEEP_DIR",
                       os.path.join(os.getcwd(), "logs", "experiments", "tau_init_grid"))
OUT = os.environ.get("OUT_DIR", SWEEP)
# Repo root and inner package dir on the path, so `train` and `timescales.*` import.
_here = os.getcwd()
for _cand in (_here, os.path.abspath(os.path.join(_here, ".."))):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

TASKS = ("ff", "sine")
SCHEMES = ("uniform", "powerlaw")
GS = (0.3, 0.9)                 # representative low / high gain
SEEDS_SCALAR = range(20)        # dimensionality stats over all seeds
SEEDS_TRAJ = (0, 1, 2)          # saved example trajectories
N_TRAJ_SAVE = 4                 # trajectories kept per run
BATCH = 32                      # val trajectories per forward pass
VAR_THRESH = 0.90
# An oscillation only counts if a cycle is observable in the trajectory, so cap the
# period at 2x the trajectory length. A |Im lambda| band is the wrong cutoff here: the
# slowest task frequency (period 100) needs only |Im| = sin(2*pi/100) = 0.063 and a
# 0.1 band would silently discard exactly the modes being looked for.
TRAJ_STEPS = {"ff": 1000, "sine": 100}


# ---------------------------------------------------------------- spectra
def spectral_timescales():
    """Decay (-1/log|lambda|) and oscillation (2pi/|arg lambda|) timescales, in timesteps."""
    out = {}
    for task in TASKS:
        for scheme in SCHEMES:
            for g in GS:
                for tag in ("init", "final"):
                    dec, osc = [], []
                    pat = f"{SWEEP}/{task}_{scheme}_full_g{g:g}/seed_*/spectral_{tag}.pt"
                    for p in sorted(glob.glob(pat)):
                        lam = torch.load(p, map_location="cpu",
                                         weights_only=False)["eigvals_eig"]
                        lam = np.asarray(lam, dtype=complex)
                        mag = np.abs(lam)
                        m = mag < 1.0                     # contracting modes only
                        with np.errstate(divide="ignore"):
                            d = -1.0 / np.log(mag[m])
                        dec.append(d[np.isfinite(d)])
                        # Only genuinely rotating modes. A mode whose imaginary part is
                        # numerically ~0 has arg(lambda)~0 and would report an absurd
                        # "period" (1e8 steps); use the same |Im| band as the pinching
                        # statistic so "oscillatory" means the same thing throughout.
                        th = np.abs(np.angle(lam))
                        with np.errstate(divide="ignore"):
                            per = 2.0 * np.pi / th
                        cap = 2 * TRAJ_STEPS[task]
                        osc.append(per[np.isfinite(per) & (per <= cap)])
                    if dec:
                        key = f"{task}_{scheme}_g{g:g}_{tag}"
                        out[f"{key}_decay"] = np.concatenate(dec).astype(np.float32)
                        out[f"{key}_osc"] = np.concatenate(osc).astype(np.float32)
    np.savez_compressed(f"{OUT}/spectral_timescales.npz", **out)
    print(f"spectral_timescales.npz: {len(out)} arrays")


# ------------------------------------------------------------------- PCA
def _load_model(config, state, strip_prefix):
    from train import _create_rnn_model
    model, _ = _create_rnn_model(config)
    if strip_prefix:
        state = {k[len("model."):]: v for k, v in state.items()
                 if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    return model.to(DEV).eval()


def _hidden_states(model, batch, config):
    inputs = batch[0][:BATCH].to(DEV)
    kw = {}
    ihv = config.get("init_hidden_value")
    if ihv is not None:
        kw["init_hidden"] = torch.full((inputs.shape[0], config["hidden_size"]),
                                       float(ihv), device=DEV)
    with torch.no_grad():
        h, _ = model(inputs=inputs, **kw)
    return h.cpu().numpy()          # [B, T, N]


def _pca_stats(H, basis=None):
    """PCA on [B,T,N] hidden states. Returns evr, dim@thresh, participation ratio, basis."""
    X = H.reshape(-1, H.shape[-1])
    X = X - X.mean(0, keepdims=True)
    if basis is None:
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        var = S ** 2
        evr = var / var.sum()
        basis = Vt
    else:
        proj = X @ basis.T
        var = proj.var(0)
        evr = var / var.sum()
    dim = int(np.searchsorted(np.cumsum(evr), VAR_THRESH) + 1)
    pr = float(var.sum() ** 2 / (var ** 2).sum())
    return evr, dim, pr, basis


def pca_and_trajectories():
    from train import create_datamodule
    rows, trajs = [], {}
    for task in TASKS:
        for scheme in SCHEMES:
            for g in GS:
                cond = f"{task}_{scheme}_full_g{g:g}"
                for seed in SEEDS_SCALAR:
                    d = f"{SWEEP}/{cond}/seed_{seed}"
                    cfg_p = f"{d}/config_seed{seed}.yaml"
                    fin_p = f"{d}/final_model_seed{seed}.pth"
                    unt_p = f"{d}/checkpoints/untrained.ckpt"
                    if not (os.path.exists(cfg_p) and os.path.exists(fin_p)
                            and os.path.exists(unt_p)):
                        continue
                    with open(cfg_p) as f:
                        config = yaml.safe_load(f)
                    config["num_workers"] = 0
                    dm = create_datamodule(config)
                    dm.prepare_data()
                    dm.setup()
                    batch = next(iter(dm.val_dataloader()))

                    m_tr = _load_model(config, torch.load(fin_p, map_location="cpu",
                                                          weights_only=False), False)
                    m_un = _load_model(config, torch.load(unt_p, map_location="cpu",
                                                          weights_only=False)["state_dict"],
                                       True)
                    H_tr = _hidden_states(m_tr, batch, config)
                    H_un = _hidden_states(m_un, batch, config)

                    evr_tr, dim_tr, pr_tr, basis = _pca_stats(H_tr)
                    evr_un, dim_un, pr_un, _ = _pca_stats(H_un)
                    rows.append(dict(task=task, scheme=scheme, g=g, seed=seed,
                                     dim90_trained=dim_tr, dim90_untrained=dim_un,
                                     pr_trained=round(pr_tr, 3),
                                     pr_untrained=round(pr_un, 3),
                                     evr1_trained=round(float(evr_tr[0]), 4),
                                     evr3_trained=round(float(evr_tr[:3].sum()), 4),
                                     evr3_untrained_in_trained_basis=None))

                    if seed in SEEDS_TRAJ:
                        # Both networks projected into the TRAINED basis, as requested.
                        def proj(H, B=basis):
                            X = H.reshape(-1, H.shape[-1])
                            X = X - X.mean(0, keepdims=True)
                            return (X @ B[:3].T).reshape(H.shape[0], H.shape[1], 3)
                        P_tr, P_un = proj(H_tr), proj(H_un)
                        k = f"{cond}_seed{seed}"
                        trajs[f"{k}_trained"] = P_tr[:N_TRAJ_SAVE].astype(np.float32)
                        trajs[f"{k}_untrained"] = P_un[:N_TRAJ_SAVE].astype(np.float32)
                        trajs[f"{k}_evr_trained"] = evr_tr[:50].astype(np.float32)
                        trajs[f"{k}_evr_untrained"] = evr_un[:50].astype(np.float32)
                    print(f"  {cond} seed{seed}: dim90 trained={dim_tr} untrained={dim_un}",
                          flush=True)
    with open(f"{OUT}/pca_dims.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(f"{OUT}/pca_traj.npz", **trajs)
    print(f"pca_dims.csv: {len(rows)} rows | pca_traj.npz: {len(trajs)} arrays")


if __name__ == "__main__":
    print(f"device: {DEV}")
    spectral_timescales()
    pca_and_trajectories()
