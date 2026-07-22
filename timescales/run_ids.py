"""Deterministic run identity, so reruns are idempotent and sweeps resumable.

A run's identity is a stable fingerprint of its *resolved* config with volatile
runtime keys stripped, combined with the seed. Two invocations that would train
the same network land on the same ``run_id`` and the same output directory; a run
that already carries a completion marker can therefore be skipped on a rerun.

Design notes:
- The fingerprint is computed from the config as *authored* (base + sweep
  overrides + fixed overrides), before ``create_datamodule`` injects derived keys
  like ``input_size`` — so those are absent at fingerprint time and need no special
  handling. ``VOLATILE_KEYS`` covers keys that may still be present but describe
  *where/how* a run executes rather than *what* it computes.
- Completion is recorded by ``COMPLETION_MARKER`` (``job_result.yaml``), written
  only after a run finishes. The marker carries the fingerprint, so a resume skips
  a seed dir only when its recorded fingerprint matches the job about to run —
  changing any identity-bearing field forces a fresh run even if the dir exists.
"""

from __future__ import annotations

import hashlib
import json
import os

import yaml

# Keys describing where/how a run executes (or derived at runtime), not what it
# computes. Excluded from the fingerprint so re-pointing GPUs, sweep dirs, or
# wandb labels does not change a run's identity.
VOLATILE_KEYS = frozenset(
    {
        "seed",
        "sweep_dir",
        "experiment_name",
        "devices",
        "accelerator",
        "strategy",
        "device",
        "run_id",
        "project_name",
        "force_rerun",
        # Derived by create_datamodule at runtime; never identity-bearing.
        "input_size",
        "output_size",
    }
)

COMPLETION_MARKER = "job_result.yaml"


def _canonical(obj):
    """Recursively drop volatile keys and sort dicts for a stable serialization."""
    if isinstance(obj, dict):
        return {
            k: _canonical(v)
            for k, v in sorted(obj.items())
            if k not in VOLATILE_KEYS
        }
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def config_fingerprint(config: dict, length: int = 10) -> str:
    """Stable short hash of the identity-bearing parts of ``config``."""
    blob = json.dumps(_canonical(dict(config)), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


def run_id_for(config: dict, seed: int) -> str:
    """Deterministic run id: ``<fingerprint>_seed<seed>``."""
    return f"{config_fingerprint(config)}_seed{seed}"


def marker_path(run_dir: str) -> str:
    return os.path.join(run_dir, COMPLETION_MARKER)


def load_marker(run_dir: str) -> dict | None:
    """Return the completion marker dict for ``run_dir``, or None if absent."""
    path = marker_path(run_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None


def is_complete(run_dir: str, fingerprint: str | None = None) -> bool:
    """True if ``run_dir`` holds a completion marker.

    When ``fingerprint`` is given, the marker must also record a matching
    fingerprint (guards against reusing a dir whose config has since changed).
    """
    marker = load_marker(run_dir)
    if marker is None:
        return False
    if fingerprint is None:
        return True
    return marker.get("fingerprint") == fingerprint
