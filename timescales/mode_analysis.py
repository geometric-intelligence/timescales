"""Matching & coupling quantification across seeds (post-review delta, Section 5).

Turns the paper's single-example Fig. 2B (timescale matching) and Fig. 2C (readout-
to-mode coupling) into metrics computed for every run, so they can be aggregated
across seeds and statistically tested. Runs post-hoc on the spectral snapshots the
harness already saves (``spectral_{init,final}.pt``) — no re-training needed.

Definitions (defaults; flagged as design decisions in delta Section 10):

- **Task timescales** (in discrete steps, from the run config):
  flip-flop -> per-bit mean inter-pulse interval ``1 / p_pulse``  (decay kind);
  sine-wave -> per-pair period ``T_i / dt``                        (oscillation kind).

- **Mode timescales** from Jacobian eigenvalues: decay ``-1/ln|lambda|`` for
  contracting modes; oscillation period ``2*pi/|arg(lambda)|`` for rotating modes.

- **Matching error** (5b): for each task timescale, the log-ratio to the nearest
  network mode of the matching kind: ``min_j | ln(tau_net_j / tau_task_i) |``.

- **Coupling** (5c): complex coupling ``C = W_out @ V``, magnitudes aggregated over
  conjugate pairs (a mode = a real eigenvalue or a conjugate pair). Per output row:
  * sparsity: participation ratio ``PR = (sum c^2)^2 / sum c^4`` over pair-level
    couplings, normalized by the number of pairs (lower = sparser);
  * one-to-one-ness: dominance ratio ``c_(1) / c_(2)`` (largest over second-largest);
  * assignment uniqueness: fraction of outputs whose argmax pair is not claimed by
    another output.
  A **shuffled-readout control** (entries of W_out permuted globally, preserving the
  weight distribution while destroying alignment with modes) is computed with the
  same metrics; the reservoir runs in the grid provide the trained-network null.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Task timescales (steps) from a run config
# ---------------------------------------------------------------------------

def task_timescales_steps(config: dict) -> dict[str, list[float]]:
    """Return {"decay": [...], "oscillation": [...]} in discrete-step units."""
    task = config.get("task", "flip_flop")
    if task in ("flip_flop", "signed_flip_flop"):
        p = config.get("p_pulse", 0.05)
        n_bits = int(config.get("n_bits", 1))
        ps = list(np.broadcast_to(np.asarray(p, dtype=float), (n_bits,)))
        return {"decay": [1.0 / max(pi, 1e-8) for pi in ps], "oscillation": []}
    if task == "sine_wave":
        dt = float(config.get("dt", 1.0))
        periods = config.get("periods", config.get("period", []))
        if isinstance(periods, (int, float)):
            periods = [periods]
        return {"decay": [], "oscillation": [float(T) / dt for T in periods]}
    return {"decay": [], "oscillation": []}


# ---------------------------------------------------------------------------
# Modes: timescales and conjugate-pair grouping
# ---------------------------------------------------------------------------

def mode_timescales(eigvals) -> dict[str, np.ndarray]:
    """Decay timescales (contracting modes) and oscillation periods (rotating)."""
    lam = np.asarray(eigvals, dtype=complex).ravel()
    abs_lam = np.abs(lam)
    contracting = (abs_lam > 0) & (abs_lam < 1)
    with np.errstate(divide="ignore"):
        decay = -1.0 / np.log(abs_lam[contracting])
    decay = decay[np.isfinite(decay)]
    theta = np.abs(np.angle(lam))
    rotating = theta > 1e-9
    osc = 2.0 * np.pi / theta[rotating]
    return {"decay": decay, "oscillation": osc}


def pair_indices(eigvals, tol: float = 1e-9) -> list[list[int]]:
    """Group eigenvalue indices into modes: conjugate pairs + real singletons.

    Each eigenvalue with Im > tol is matched to the unused nearest-conjugate
    eigenvalue with Im < -tol; near-real eigenvalues are singleton modes.
    """
    lam = np.asarray(eigvals, dtype=complex).ravel()
    pos = [i for i in range(lam.size) if lam[i].imag > tol]
    neg = [i for i in range(lam.size) if lam[i].imag < -tol]
    real = [i for i in range(lam.size) if abs(lam[i].imag) <= tol]

    groups: list[list[int]] = [[i] for i in real]
    unused = set(neg)
    for i in pos:
        if unused:
            j = min(unused, key=lambda k: abs(np.conj(lam[i]) - lam[k]))
            unused.discard(j)
            groups.append([i, j])
        else:  # numerically unpaired; keep as singleton
            groups.append([i])
    groups.extend([j] for j in sorted(unused))
    return groups


# ---------------------------------------------------------------------------
# 5b — matching errors
# ---------------------------------------------------------------------------

def matching_errors(eigvals, task_ts: dict[str, list[float]]) -> dict:
    """Per task timescale: |log-ratio| to the nearest mode of the matching kind."""
    net = mode_timescales(eigvals)
    out: dict = {}
    for kind in ("decay", "oscillation"):
        targets = task_ts.get(kind) or []
        modes = net[kind]
        errs = []
        for t in targets:
            if modes.size == 0 or t <= 0:
                errs.append(np.nan)
                continue
            errs.append(float(np.min(np.abs(np.log(modes / t)))))
        out[kind] = errs
    all_errs = [e for k in out.values() for e in k if np.isfinite(e)]
    out["mean_abs_log_err"] = float(np.mean(all_errs)) if all_errs else None
    out["max_abs_log_err"] = float(np.max(all_errs)) if all_errs else None
    return out


# ---------------------------------------------------------------------------
# 5c — coupling metrics
# ---------------------------------------------------------------------------

def pair_couplings(W_out, V, groups: list[list[int]]) -> np.ndarray:
    """|W_out @ V| aggregated over conjugate pairs: (n_outputs, n_modes)."""
    W_out = np.asarray(W_out, dtype=float)
    V = np.asarray(V, dtype=complex)
    C = np.abs(W_out @ V)  # (n_out, N)
    agg = np.stack(
        [np.sqrt(np.sum(C[:, idx] ** 2, axis=1)) for idx in groups], axis=1
    )
    return agg


def participation_ratio(row) -> float:
    """(sum c^2)^2 / sum c^4 — number of effectively-coupled modes (>=1)."""
    c2 = np.asarray(row, dtype=float) ** 2
    s2 = c2.sum()
    if s2 <= 0:
        return float("nan")
    return float(s2**2 / np.sum(c2**2))


def dominance_ratio(row) -> float:
    """Largest / second-largest coupling (one-to-one-ness; higher = more 1:1)."""
    c = np.sort(np.asarray(row, dtype=float))[::-1]
    if c.size < 2 or c[1] <= 0:
        return float("inf") if c.size and c[0] > 0 else float("nan")
    return float(c[0] / c[1])


def assignment_uniqueness(C_pairs: np.ndarray) -> float:
    """Fraction of outputs whose argmax mode is not the argmax of another output."""
    tops = np.argmax(C_pairs, axis=1)
    unique, counts = np.unique(tops, return_counts=True)
    n_unique_outputs = int(np.sum(counts == 1))
    return float(n_unique_outputs / len(tops)) if len(tops) else float("nan")


def coupling_metrics(W_out, V, eigvals, rng: np.random.Generator | None = None) -> dict:
    """Sparsity / one-to-one-ness for the readout, plus a shuffled-readout control."""
    groups = pair_indices(eigvals)
    C = pair_couplings(W_out, V, groups)

    def _summarize(Cp: np.ndarray, prefix: str) -> dict:
        prs = [participation_ratio(r) for r in Cp]
        doms = [dominance_ratio(r) for r in Cp]
        finite_doms = [d for d in doms if np.isfinite(d)]
        return {
            f"{prefix}participation_ratio_mean": float(np.nanmean(prs)),
            f"{prefix}participation_ratio_norm": float(np.nanmean(prs) / Cp.shape[1]),
            f"{prefix}dominance_ratio_median": (
                float(np.median(finite_doms)) if finite_doms else None
            ),
            f"{prefix}assignment_uniqueness": assignment_uniqueness(Cp),
        }

    out = _summarize(C, "coup_")
    out["n_modes"] = C.shape[1]
    out["n_outputs"] = C.shape[0]

    rng = rng or np.random.default_rng(0)
    W_shuf = np.asarray(W_out, dtype=float).copy()
    flat = W_shuf.ravel()
    rng.shuffle(flat)
    W_shuf = flat.reshape(W_shuf.shape)
    out.update(_summarize(pair_couplings(W_shuf, V, groups), "coup_shuffled_"))
    return out


# ---------------------------------------------------------------------------
# Per-run analysis + sweep-level table
# ---------------------------------------------------------------------------

def analyze_seed_dir(seed_dir: str, tag: str = "final") -> dict | None:
    """One flat metrics row for a run, from its spectral blob + config. None if absent."""
    import torch

    blob_path = os.path.join(seed_dir, f"spectral_{tag}.pt")
    cfg_matches = glob.glob(os.path.join(seed_dir, "config_seed*.yaml"))
    if not os.path.exists(blob_path) or not cfg_matches:
        return None
    blob = torch.load(blob_path, map_location="cpu", weights_only=False)
    with open(cfg_matches[0]) as f:
        config = yaml.safe_load(f) or {}

    eigvals = np.asarray(blob["eigvals_eig"])
    row: dict = {
        "seed_dir": seed_dir,
        "snapshot": tag,
        "experiment_name": os.path.basename(os.path.dirname(seed_dir)),
        "seed": config.get("seed"),
        "task": config.get("task"),
        "trainable": config.get("trainable"),
        "recurrent_gain": config.get("recurrent_gain"),
        "init_scheme": (config.get("tau_init") or {}).get("scheme"),
    }

    match = matching_errors(eigvals, task_timescales_steps(config))
    row["match_mean_abs_log_err"] = match["mean_abs_log_err"]
    row["match_max_abs_log_err"] = match["max_abs_log_err"]
    for kind in ("decay", "oscillation"):
        for i, e in enumerate(match[kind]):
            row[f"match_{kind}_{i}_abs_log_err"] = e

    seed = config.get("seed") or 0
    row.update(coupling_metrics(
        blob["W_out"], blob["V"], eigvals,
        rng=np.random.default_rng(int(seed)),
    ))
    return row


def collect_mode_rows(sweep_dir: str, tag: str = "final") -> list[dict]:
    rows = []
    for blob_path in sorted(glob.glob(os.path.join(sweep_dir, "*", "seed_*",
                                                   f"spectral_{tag}.pt"))):
        row = analyze_seed_dir(os.path.dirname(blob_path), tag=tag)
        if row is not None:
            rows.append(row)
    return rows


def write_mode_table(sweep_dir: str, out_path: str | None = None,
                     tag: str = "final") -> str:
    rows = collect_mode_rows(sweep_dir, tag=tag)
    if out_path is None:
        out_path = os.path.join(sweep_dir, f"mode_table_{tag}.csv")
    fieldnames = sorted({k for r in rows for k in r})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} mode-analysis rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Matching & coupling metrics over a sweep (delta Section 5)")
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--tag", default="final", choices=["init", "final"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    write_mode_table(args.sweep_dir, args.out, tag=args.tag)


if __name__ == "__main__":
    main()
