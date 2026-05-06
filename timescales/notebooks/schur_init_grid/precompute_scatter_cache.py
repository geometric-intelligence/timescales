"""Batch pre-computation of ALL notebook-2 caches — GPU-accelerated.

Computes two cache layers per condition so the notebook opens with zero
forward passes:

  PART 2  (best-seed)  : h_flat, tgt_flat, r_eig, r_neu  (used by heatmaps)
  PART 2b (multi-seed) : y3_arr, y4_arr                  (used by scatter error bars)

Usage (from the timescales/ working directory):
    # all four GPUs in parallel  ← recommended
    poetry run python notebooks/schur_init_grid/precompute_scatter_cache.py --gpus 0,1,2,3

    # single GPU
    poetry run python notebooks/schur_init_grid/precompute_scatter_cache.py --gpus 0

    # CPU only
    poetry run python notebooks/schur_init_grid/precompute_scatter_cache.py

    # overwrite existing cache
    ... --overwrite

Optional flags:
    --coupling          pearson | connectivity | ablation  (default: pearson)
    --n-corr            N                                  (default: 200)
    --scatter           auto | decay | oscillation         (default: auto)
    --selection-metric  min_val_loss | max_val_acc         (default: min_val_loss)
    --no-drop-last-bit  keep last bit for FF
    --overwrite         ignore existing cache
    --gpus              comma-separated GPU indices, e.g. "0,1,2,3"
    --skip-main-cache   only compute multi-seed scatter cache (skip PART 2)
"""

from __future__ import annotations

import argparse
import glob as _glob
import multiprocessing as mp
import os
import re
import sys
import subprocess
import traceback

import numpy as np
import torch
import yaml


# ── Bootstrap: cd to timescales/ and add it to sys.path ─────────────────────
def _bootstrap() -> str:
    gitroot = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
    ).strip()
    target = os.path.join(gitroot, "timescales")
    os.chdir(target)
    for p in (gitroot, target):
        if p not in sys.path:
            sys.path.insert(0, p)
    return gitroot


_GITROOT = _bootstrap()


# ══════════════════════════════════════════════════════════════════════════════
# Configuration — edit SWEEP_DIRS / BEST_SEEDS_CSV to match your setup
# ══════════════════════════════════════════════════════════════════════════════

SWEEP_DIRS = [
    # Server A (g = 0.5)
    "/home/facosta/timescales/timescales/logs/experiments/schur_init_grid_g05_20260505_061150",
    # Server B (g = 0.9)
    "/home/facosta/timescales/timescales/logs/experiments/schur_init_grid_g09_20260504_230709",
]

CACHE_DIR      = os.path.join("notebooks", "schur_init_grid", ".cache")
BEST_SEEDS_CSV = os.path.join("notebooks", "schur_init_grid", "best_seeds.csv")
os.makedirs(CACHE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers — defined at module level so they pickle for spawn
# ══════════════════════════════════════════════════════════════════════════════

def _load_pt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _pearson_r_torch(Z: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
    """GPU Pearson r.  (M, Kz), (M, Ky) → (Kz, Ky) float32 numpy array."""
    n  = Z.shape[0]
    Zc = Z - Z.mean(0)
    Yc = Y - Y.mean(0)
    Zs = Zc.std(0).clamp(min=1e-12)
    Ys = Yc.std(0).clamp(min=1e-12)
    r  = (Zc.T @ Yc) / (n * Zs[:, None] * Ys[None, :])
    return r.float().cpu().numpy()


def _build_model(rc: dict, seed_path: str):
    """Reconstruct the trained RNN from run_config + best checkpoint."""
    ckpts = _glob.glob(os.path.join(seed_path, "checkpoints", "best-model-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No best-model checkpoint in {seed_path}/checkpoints/"
        )
    import torch.nn as _nn
    from rnns.rnn import RNN, RNNLightning

    task_key = rc.get("task", "flip_flop")
    n_bits   = rc.get("n_bits", 6)
    n_pairs  = rc.get("n_pairs", 3)
    in_sz    = rc.get("input_size", 1) if task_key == "sine_wave" else n_bits
    out_sz   = 2 * n_pairs             if task_key == "sine_wave" else n_bits

    model = RNN(
        input_size=in_sz,
        hidden_size=rc["hidden_size"],
        output_size=out_sz,
        dt=rc["dt"],
        time_constants_config=rc.get("time_constants_config"),
        activation=getattr(_nn, rc.get("activation", "Tanh")),
        learn_time_constants=rc.get("learn_time_constants", False),
        init_time_constant=rc.get("init_time_constant"),
        init_time_constants_config=rc.get("init_time_constants_config"),
        shared_time_constant=rc.get("shared_time_constant", False),
        normalize_hidden=rc.get("normalize_hidden", False),
        zero_diag_wrec=rc.get("zero_diag_wrec", False),
        recurrent_gain=rc["recurrent_gain"],
        noise_std=0.0,
        wrec_init="normal_scaled",
        wrec_init_config=None,
        alpha_parameterization=rc.get("alpha_parameterization", "exponential"),
        dynamics_type=rc.get("dynamics_type", "rate"),
    )
    lit = RNNLightning(
        model=model,
        learning_rate=rc["learning_rate"],
        weight_decay=rc["weight_decay"],
        step_size=rc.get("lr_step_size", rc.get("step_size", 1000)),
        gamma=rc["gamma"],
        task=task_key,
        init_hidden_value=rc.get("init_hidden_value"),
    )
    ckpt = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    lit.load_state_dict(ckpt["state_dict"])
    lit.eval()
    return lit.model


def _generate_inputs(
    rc: dict, n_corr: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Generate n_corr validation inputs; move to device.  Returns (inp, tgt, h0|None)."""
    task = rc.get("task", "flip_flop")
    if task == "flip_flop":
        from datamodules.flip_flop import FlipFlopDataModule
        dm = FlipFlopDataModule(
            n_bits=rc["n_bits"], p_pulse=rc["p_pulse"],
            pulse_amplitude=rc.get("pulse_amplitude", 1.0),
            num_time_steps=rc["num_time_steps"],
            num_val_trajectories=n_corr, batch_size=n_corr,
        )
        dm.setup()
        inp, _, tgt = dm.val_dataset.tensors
        return inp.to(device), tgt.to(device), None
    else:
        from datamodules.sine_wave import SineWaveDataModule
        dm = SineWaveDataModule(
            n_pairs=rc["n_pairs"], periods=rc["periods"],
            num_time_steps=rc["num_time_steps"], dt=rc["dt"],
            num_val_trajectories=n_corr, batch_size=n_corr,
            init_hidden_value=rc.get("init_hidden_value", 1.0),
        )
        dm.setup()
        inp, _, tgt = dm.val_dataset.tensors
        h0 = torch.full(
            (n_corr, rc["hidden_size"]),
            float(rc.get("init_hidden_value", 1.0)),
        )
        return inp.to(device), tgt.to(device), h0.to(device)


def _fwd(model, inp_gpu, h0_gpu, rc, n_corr):
    """Run forward pass; return (h_flat, tgt_flat) as GPU float tensors."""
    with torch.no_grad():
        if h0_gpu is not None:
            h_seq, _ = model(inp_gpu, init_hidden=h0_gpu)
        else:
            h_seq, _ = model(inp_gpu)
    M = n_corr * rc["num_time_steps"]
    return h_seq.reshape(M, rc["hidden_size"]).float()


def _z_eig_parts_gpu(
    h_gpu: torch.Tensor, V_s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project h onto the (complex) eigenmode basis; return (Z_re, Z_im) float32.

    For real h: (h @ V_inv.T).real = h @ V_inv.real.T
                (h @ V_inv.T).imag = h @ V_inv.imag.T
    Computed this way to avoid the real-double × complex-double dtype mismatch.
    """
    V_inv = torch.linalg.pinv(V_s)                          # (N, N) complex
    Z_re  = (h_gpu.double() @ V_inv.real.T).float()         # (M, N)
    Z_im  = (h_gpu.double() @ V_inv.imag.T).float()         # (M, N)
    return Z_re, Z_im


def _pearson_mod_torch(
    Z_re: torch.Tensor, Z_im: torch.Tensor, Y: torch.Tensor
) -> np.ndarray:
    """Modulation-amplitude Pearson: sqrt(r_Re² + r_Im²).

    Correctly accounts for the complex eigenmode projection so that
    oscillatory modes (e.g., sine-wave task) couple to their output channels.
    (N, Ky) float32 numpy array.
    """
    r_re = _pearson_r_torch(Z_re, Y)
    r_im = _pearson_r_torch(Z_im, Y)
    return np.sqrt(r_re**2 + r_im**2)


# ── PART 2 (best-seed) cache ─────────────────────────────────────────────────

def _precompute_best_seed(
    task: str,
    gain: float,
    init: str,
    theta: str,
    seed_dirs: list[str],
    best_seed: int,
    device: torch.device,
    n_corr: int,
    cache_dir: str,
    overwrite: bool,
    rc: dict,
) -> str:
    """Compute and save h_flat / tgt_flat / r_eig / r_neu for the best seed.

    Cache key format matches notebook 2's _CACHE_KEY exactly:
        {task}_{gain:g}_{init}_{theta}_seed{best_seed}_ncorr{n_corr}
    """
    label     = f"{task} g={gain} {init}/{theta} seed{best_seed}"
    gain_str  = f"{gain:g}"    # "0.5", not "g0p5"
    # _cplxr suffix matches notebook's _CACHE_KEY (complex-aware modulation amplitude Pearson)
    key       = f"{task}_{gain_str}_{init}_{theta}_seed{best_seed}_ncorr{n_corr}_cplxr"
    c_hidden  = os.path.join(cache_dir, f"{key}_hidden.npy")
    c_targets = os.path.join(cache_dir, f"{key}_targets.npy")
    c_reig    = os.path.join(cache_dir, f"{key}_corr_eig.npy")
    c_rneu    = os.path.join(cache_dir, f"{key}_corr_neu.npy")

    if not overwrite and all(os.path.exists(p) for p in [c_hidden, c_targets, c_reig, c_rneu]):
        return f"CACHED  [main]  {label}"

    # Find the seed directory
    seed_dir = next(
        (sd for sd in seed_dirs if os.path.basename(sd) == f"seed_{best_seed}"),
        None,
    )
    if seed_dir is None:
        return f"SKIP    [main]  {label} — seed_{best_seed} dir not found"

    try:
        snap  = _load_pt(os.path.join(seed_dir, "spectral_final.pt"))
        V_s   = torch.tensor(np.asarray(snap["V"]), dtype=torch.complex128, device=device)

        inp_gpu, tgt_gpu, h0_gpu = _generate_inputs(rc, n_corr, device)
        M = n_corr * rc["num_time_steps"]

        model = _build_model(rc, seed_dir).to(device)
        h_gpu  = _fwd(model, inp_gpu, h0_gpu, rc, n_corr)
        tg_gpu = tgt_gpu.reshape(M, -1).float()

        r_neu_np       = _pearson_r_torch(h_gpu, tg_gpu)
        Z_re, Z_im     = _z_eig_parts_gpu(h_gpu, V_s)
        r_eig_np       = _pearson_mod_torch(Z_re, Z_im, tg_gpu)

        np.save(c_hidden,  h_gpu.cpu().numpy())
        np.save(c_targets, tg_gpu.cpu().numpy())
        np.save(c_reig,    r_eig_np)
        np.save(c_rneu,    r_neu_np)
        return f"SAVED   [main]  {label}"

    except Exception:
        return f"FAILED  [main]  {label}\n{traceback.format_exc()}"


# ── PART 2b (multi-seed scatter) cache ───────────────────────────────────────

def _precompute_scatter(
    task: str,
    gain: float,
    init: str,
    theta: str,
    seed_dirs: list[str],
    device: torch.device,
    coupling_metric: str,
    n_corr: int,
    ff_drop_last_bit: bool,
    scatter_timescale: str,
    cache_dir: str,
    overwrite: bool,
    rc: dict,
    inp_gpu: torch.Tensor,
    tgt_gpu: torch.Tensor,
    h0_gpu: torch.Tensor | None,
) -> str:
    """Compute y3_arr / y4_arr across all seeds for scatter error bars."""
    label = f"{task} g={gain} {init}/{theta}"

    dt_val    = float(rc["dt"])
    task_key  = rc.get("task", "flip_flop")
    n_ch_full = rc.get("n_bits", 6) if task_key == "flip_flop" else 2 * rc.get("n_pairs", 3)
    ch_display = (
        np.arange(n_ch_full - 1)
        if (task_key == "flip_flop" and ff_drop_last_bit)
        else np.arange(n_ch_full)
    )
    n_ch = len(ch_display)

    if scatter_timescale == "auto":
        scatter_ts = "oscillation" if task_key == "sine_wave" else "decay"
    else:
        scatter_ts = scatter_timescale

    # Cache key matches notebook 2's _MS_KEY exactly.
    # "pearson" → "pearsonmod" to distinguish from old real-only Pearson files.
    gain_str_ms = f"g{gain}".replace(".", "p")   # "g0p5"
    _coup_lbl   = "pearsonmod" if coupling_metric == "pearson" else coupling_metric
    ms_key = (
        f"{task}_{gain_str_ms}_{init}_{theta}"
        f"_multiseed_{_coup_lbl}_{scatter_ts}"
        f"_nch{n_ch}_ncorr{n_corr}"
    )
    c_y3 = os.path.join(cache_dir, f"{ms_key}_y3.npy")
    c_y4 = os.path.join(cache_dir, f"{ms_key}_y4.npy")

    if not overwrite and os.path.exists(c_y3) and os.path.exists(c_y4):
        return f"CACHED  [scatter]  {label}"

    M = n_corr * rc["num_time_steps"]
    y3_list: list[np.ndarray] = []
    y4_list: list[np.ndarray] = []

    for seed_dir in seed_dirs:
        seed_label = os.path.basename(seed_dir)
        try:
            snap      = _load_pt(os.path.join(seed_dir, "spectral_final.pt"))
            eigvals_s = np.asarray(snap["eigvals_eig"])
            taus_s    = np.asarray(snap["taus"])

            # Per-mode scatter timescale
            log_abs_s  = np.log(np.clip(np.abs(eigvals_s), 1e-12, None))
            stable_s   = np.abs(eigvals_s) < 1.0
            denom_s    = np.where(stable_s & (log_abs_s < -1e-10), log_abs_s, -1e-10)
            tau_eff_sv = np.where(stable_s, -1.0 / denom_s * dt_val, np.nan)
            arg_s      = np.abs(np.angle(eigvals_s))
            tau_osc_sv = np.where(arg_s > 1e-10, 2.0 * np.pi / arg_s * dt_val, np.nan)
            tau_sc_s   = tau_osc_sv if scatter_ts == "oscillation" else tau_eff_sv
            sc_valid_s = stable_s & np.isfinite(tau_sc_s)

            # Coupling
            if coupling_metric == "connectivity":
                r_eig_f = np.asarray(snap["coup_eig_re"]).T
                r_neu_f = np.asarray(snap["coup_neuron"]).T
            else:
                V_s       = torch.tensor(
                    np.asarray(snap["V"]), dtype=torch.complex128, device=device
                )
                model_s   = _build_model(rc, seed_dir).to(device)
                h_gpu     = _fwd(model_s, inp_gpu, h0_gpu, rc, n_corr)
                tg_gpu    = tgt_gpu.reshape(M, -1).float()
                Z_re, Z_im = _z_eig_parts_gpu(h_gpu, V_s)
                r_eig_f   = _pearson_mod_torch(Z_re, Z_im, tg_gpu)
                r_neu_f   = _pearson_r_torch(h_gpu, tg_gpu)
                if coupling_metric == "ablation":
                    h_np     = h_gpu.cpu().numpy()
                    Z_np     = Z_re.cpu().numpy()   # real part; std ≈ RMS amplitude
                    W_out_s  = np.asarray(snap["W_out"])
                    snap_c   = np.asarray(snap["coup_eig_re"])
                    snap_cn  = np.asarray(snap["coup_neuron"])
                    y_hat    = h_np @ W_out_s.T
                    y_std    = y_hat.std(0).clip(1e-12)
                    z_std    = Z_np.std(0).clip(1e-12)
                    h_std    = h_np.std(0).clip(1e-12)
                    r_eig_f  = (z_std[:, None] * snap_c.T  / y_std[None, :]).astype(np.float32)
                    r_neu_f  = (h_std[:, None] * snap_cn.T / y_std[None, :]).astype(np.float32)

            r_eig_d    = r_eig_f[:, ch_display]
            r_neu_d    = r_neu_f[:, ch_display]
            r_eig_st_d = np.where(sc_valid_s[:, None], np.abs(r_eig_d), 0.0)
            dom_m      = np.argmax(r_eig_st_d, axis=0)
            dom_n      = np.argmax(np.abs(r_neu_d), axis=0)

            y3 = np.array([
                tau_sc_s[dom_m[ch]] if sc_valid_s[dom_m[ch]] else np.nan
                for ch in range(n_ch)
            ])
            y3_list.append(y3)
            y4_list.append(taus_s[dom_n])
            print(f"  [{label}] {seed_label} ok", flush=True)

        except Exception:
            print(f"  [{label}] {seed_label} FAILED:\n{traceback.format_exc()}", flush=True)

    if not y3_list:
        return f"FAILED  [scatter]  {label} — all seeds errored"

    np.save(c_y3, np.array(y3_list))
    np.save(c_y4, np.array(y4_list))
    return f"SAVED   [scatter]  {label} ({len(y3_list)}/{len(seed_dirs)} seeds)"


# ── Main per-condition entry point ────────────────────────────────────────────

def _process_condition(
    task: str,
    gain: float,
    init: str,
    theta: str,
    seed_dirs: list[str],
    best_seed: int | None,
    device: torch.device,
    coupling_metric: str,
    n_corr: int,
    ff_drop_last_bit: bool,
    scatter_timescale: str,
    cache_dir: str,
    overwrite: bool,
    skip_main_cache: bool,
) -> list[str]:
    """Run both cache passes for one condition.  Returns list of status strings."""
    # Load RC from first available seed
    rc = None
    for sd in seed_dirs:
        p = os.path.join(sd, "run_config.yaml")
        if os.path.exists(p):
            with open(p) as f:
                rc = yaml.safe_load(f)
            break
    if rc is None:
        msg = f"SKIP (no run_config.yaml): {task} g={gain} {init}/{theta}"
        return [msg, msg]

    # Generate inputs ONCE, shared across all seeds of this condition
    inp_gpu, tgt_gpu, h0_gpu = _generate_inputs(rc, n_corr, device)

    results = []

    # ── PART 2: best-seed heatmap cache ──────────────────────────────────────
    if not skip_main_cache and best_seed is not None:
        status = _precompute_best_seed(
            task, gain, init, theta, seed_dirs, best_seed,
            device, n_corr, cache_dir, overwrite, rc,
        )
        print(f"  → {status}", flush=True)
        results.append(status)

    # ── PART 2b: multi-seed scatter cache ─────────────────────────────────────
    status = _precompute_scatter(
        task, gain, init, theta, seed_dirs,
        device, coupling_metric, n_corr, ff_drop_last_bit,
        scatter_timescale, cache_dir, overwrite, rc,
        inp_gpu, tgt_gpu, h0_gpu,
    )
    print(f"  → {status}", flush=True)
    results.append(status)

    return results


# ── Worker entry point for spawned processes ──────────────────────────────────

def _worker(args: tuple) -> list[str]:
    conditions_chunk, gpu_id, cfg = args
    _bootstrap()

    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        print(f"Worker GPU {gpu_id}: {len(conditions_chunk)} conditions", flush=True)
    else:
        device = torch.device("cpu")
        print(f"Worker CPU: {len(conditions_chunk)} conditions", flush=True)

    results: list[str] = []
    for task, gain, init, theta, seed_dirs, best_seed in conditions_chunk:
        for s in _process_condition(
            task, gain, init, theta, seed_dirs, best_seed, device, **cfg
        ):
            results.append(s)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--coupling", default="pearson",
                        choices=["pearson", "connectivity", "ablation"])
    parser.add_argument("--n-corr", type=int, default=200)
    parser.add_argument("--scatter", default="auto",
                        choices=["auto", "decay", "oscillation"])
    parser.add_argument("--selection-metric", default="min_val_loss",
                        choices=["min_val_loss", "max_val_acc"])
    parser.add_argument("--no-drop-last-bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-main-cache", action="store_true",
                        help="Only compute multi-seed scatter cache, skip PART-2 cache.")
    parser.add_argument(
        "--gpus", default="",
        help='Comma-separated GPU indices, e.g. "0,1,2,3".  Omit for CPU.',
    )
    args = parser.parse_args()

    COUPLING_METRIC   = args.coupling
    N_CORR            = args.n_corr
    SCATTER_TIMESCALE = args.scatter
    SELECTION_METRIC  = args.selection_metric
    FF_DROP_LAST_BIT  = not args.no_drop_last_bit
    OVERWRITE         = args.overwrite
    SKIP_MAIN         = args.skip_main_cache
    GPUS: list[int]   = (
        [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
        if args.gpus.strip() else []
    )

    print("=" * 70)
    print(f"Coupling metric   : {COUPLING_METRIC}")
    print(f"N_CORR            : {N_CORR}")
    print(f"Scatter timescale : {SCATTER_TIMESCALE}")
    print(f"Selection metric  : {SELECTION_METRIC}")
    print(f"FF_DROP_LAST_BIT  : {FF_DROP_LAST_BIT}")
    print(f"Overwrite cache   : {OVERWRITE}")
    print(f"Skip main cache   : {SKIP_MAIN}")
    print(f"Device(s)         : {('GPU ' + args.gpus) if GPUS else 'CPU'}")
    print("=" * 70)

    # ── Load best seeds CSV ───────────────────────────────────────────────────
    import pandas as pd
    _seed_col = (
        "best_seed_min_loss" if SELECTION_METRIC == "min_val_loss"
        else "best_seed_max_acc"
    )
    best_seeds_map: dict[tuple, int] = {}
    if not SKIP_MAIN:
        if not os.path.exists(BEST_SEEDS_CSV):
            print(f"WARNING: {BEST_SEEDS_CSV} not found — skipping PART-2 (main) cache.")
            SKIP_MAIN = True
        else:
            bdf = pd.read_csv(BEST_SEEDS_CSV)
            for _, row in bdf.iterrows():
                key = (str(row["task"]), float(row["gain"]),
                       str(row["init"]), str(row["theta"]))
                if _seed_col in row and not pd.isna(row[_seed_col]):
                    best_seeds_map[key] = int(row[_seed_col])

    # ── Discover conditions ───────────────────────────────────────────────────
    _COND_RE = re.compile(
        r"^g(?P<gain>[\d.]+)_(?P<init>[^_]+)_(?P<theta>[^_]+)_(?P<task>.+)$"
    )
    conditions: dict[tuple, list[str]] = {}
    for sw in SWEEP_DIRS:
        if not os.path.isdir(sw):
            print(f"WARNING: sweep dir not found: {sw}")
            continue
        for exp in sorted(os.listdir(sw)):
            m = _COND_RE.match(exp)
            if m is None:
                continue
            key = (m["task"], float(m["gain"]), m["init"], m["theta"])
            exp_path = os.path.join(sw, exp)
            seeds = sorted(
                os.path.join(exp_path, s)
                for s in os.listdir(exp_path)
                if s.startswith("seed_") and os.path.isdir(os.path.join(exp_path, s))
            )
            if seeds:
                conditions.setdefault(key, []).extend(seeds)

    if not conditions:
        print("No conditions found — check SWEEP_DIRS.")
        sys.exit(1)

    print(f"\nFound {len(conditions)} conditions:")
    for k, dirs in sorted(conditions.items()):
        bs = best_seeds_map.get(k, "?")
        print(f"  task={k[0]}, g={k[1]}, init={k[2]}, theta={k[3]}"
              f"  →  {len(dirs)} seed(s)  best={bs}")
    print()

    _cfg = dict(
        coupling_metric=COUPLING_METRIC,
        n_corr=N_CORR,
        ff_drop_last_bit=FF_DROP_LAST_BIT,
        scatter_timescale=SCATTER_TIMESCALE,
        cache_dir=CACHE_DIR,
        overwrite=OVERWRITE,
        skip_main_cache=SKIP_MAIN,
    )

    all_conditions = [
        (task, gain, init, theta, seed_dirs, best_seeds_map.get((task, gain, init, theta)))
        for (task, gain, init, theta), seed_dirs in sorted(conditions.items())
    ]

    # ── Dispatch ──────────────────────────────────────────────────────────────
    all_results: list[str] = []

    if len(GPUS) <= 1:
        device = (
            torch.device(f"cuda:{GPUS[0]}")
            if (GPUS and torch.cuda.is_available())
            else torch.device("cpu")
        )
        print(f"Running on {device}\n")
        for task, gain, init, theta, seed_dirs, best_seed in all_conditions:
            for s in _process_condition(
                task, gain, init, theta, seed_dirs, best_seed, device, **_cfg
            ):
                all_results.append(s)
    else:
        n_gpus = len(GPUS)
        chunks = [all_conditions[i::n_gpus] for i in range(n_gpus)]
        worker_args = [
            (chunk, gpu_id, _cfg)
            for gpu_id, chunk in zip(GPUS, chunks)
            if chunk
        ]
        ctx = mp.get_context("spawn")
        print(f"Spawning {len(worker_args)} worker(s) across GPUs {GPUS}\n")
        with ctx.Pool(len(worker_args)) as pool:
            for results in pool.map(_worker, worker_args):
                all_results.extend(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_saved  = sum(1 for r in all_results if r.startswith("SAVED"))
    n_cached = sum(1 for r in all_results if r.startswith("CACHED"))
    n_failed = sum(1 for r in all_results if "FAILED" in r or "SKIP" in r)

    print()
    print("=" * 70)
    print(f"Done.  saved={n_saved}  already_cached={n_cached}  failed={n_failed}")
