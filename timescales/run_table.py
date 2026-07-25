"""Per-run columnar record: one machine-readable row per run (spec Workstream C1).

Walks a sweep directory and, for each completed run, assembles a flat row from the
completion marker (``job_result.yaml``), the resolved config (``config_seed*.yaml``),
and the spectral-pinching sidecars (``spectral_stats_{init,final}.json``). Full
validation curves and spectrum blobs stay as separate files — the row carries only
scalars plus *handles* (paths) to them, so aggregation never re-opens checkpoints.

Serialization prefers Parquet (via pandas); if pandas/pyarrow are unavailable it
falls back to CSV so the table is always produced. Row collection itself is pure
stdlib and independently testable.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import yaml

from timescales.run_ids import COMPLETION_MARKER


def _flatten(obj, prefix: str, out: dict) -> None:
    """Flatten nested dicts/lists into ``prefix``-joined scalar columns."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, (list, tuple)):
        # Keep short lists as a string; they are usually small config values.
        out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj


def _read_yaml(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _row_for_seed_dir(seed_dir: str) -> dict:
    marker = _read_yaml(os.path.join(seed_dir, COMPLETION_MARKER))
    row: dict = {
        "experiment_name": os.path.basename(os.path.dirname(seed_dir)),
        "seed_dir": seed_dir,
        "run_id": marker.get("run_id"),
        "seed": marker.get("seed"),
        "fingerprint": marker.get("fingerprint"),
        "final_val_loss": marker.get("final_val_loss"),
        "steps_to_convergence": marker.get("steps_to_convergence"),
        "completed_at": marker.get("completed_at"),
    }

    # Convergence: one column per candidate method's step count.
    by_method = (marker.get("convergence") or {}).get("by_method") or {}
    for method, rec in by_method.items():
        row[f"conv_{method}_steps"] = (rec or {}).get("steps")

    # Provenance.
    prov = marker.get("provenance") or {}
    row["git_commit"] = prov.get("git_commit")
    row["git_dirty"] = prov.get("git_dirty")
    for pkg, ver in (prov.get("packages") or {}).items():
        row[f"pkg_{pkg}"] = ver

    # Resolved config (flattened under cfg.*).
    cfg_matches = glob.glob(os.path.join(seed_dir, "config_seed*.yaml"))
    if cfg_matches:
        _flatten(_read_yaml(cfg_matches[0]), "cfg", row)

    # Spectral-pinching stats (init + final), flattened under spec_<tag>_*.
    for tag in ("init", "final"):
        stats = _read_json(os.path.join(seed_dir, f"spectral_stats_{tag}.json"))
        for k, v in stats.items():
            row[f"spec_{tag}_{k}"] = v

    # Handles to the heavy artifacts (not inlined).
    for name, fname in (
        ("curve_path", "training_losses.json"),
        ("spectrum_init_path", "spectral_init.pt"),
        ("spectrum_final_path", "spectral_final.pt"),
    ):
        p = os.path.join(seed_dir, fname)
        row[name] = p if os.path.exists(p) else None

    return row


def collect_run_rows(sweep_dir: str) -> list[dict]:
    """One row per completed run (a seed dir carrying a completion marker)."""
    rows = []
    pattern = os.path.join(sweep_dir, "*", "seed_*", COMPLETION_MARKER)
    for marker_path in sorted(glob.glob(pattern)):
        rows.append(_row_for_seed_dir(os.path.dirname(marker_path)))
    return rows


def _write_csv(rows: list[dict], path: str) -> str:
    fieldnames = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_run_table(sweep_dir: str, out_path: str | None = None) -> str:
    """Build and serialize the run table. Returns the path actually written.

    Prefers Parquet; falls back to CSV when pandas/pyarrow are unavailable.
    """
    rows = collect_run_rows(sweep_dir)
    if out_path is None:
        out_path = os.path.join(sweep_dir, "run_table.parquet")

    if out_path.endswith(".parquet"):
        try:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(out_path)
            return out_path
        except Exception as e:  # pandas or pyarrow missing / write failure
            csv_path = out_path[: -len(".parquet")] + ".csv"
            print(f"[run_table] parquet unavailable ({e}); writing CSV: {csv_path}")
            return _write_csv(rows, csv_path)

    return _write_csv(rows, out_path)


def main():
    parser = argparse.ArgumentParser(description="Build the per-run columnar table")
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--out", default=None, help="Output path (.parquet or .csv)")
    args = parser.parse_args()
    path = write_run_table(args.sweep_dir, args.out)
    n = len(collect_run_rows(args.sweep_dir))
    print(f"Wrote {n} run rows to {path}")


if __name__ == "__main__":
    main()
