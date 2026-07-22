"""Run provenance: git commit + key library versions (spec Workstream C3).

Recorded in every run's completion marker so a run is reproducible from its logged
config + provenance alone. RNG state is captured separately by train.py
(``rng_state_{init,final}.pt``).
"""

from __future__ import annotations

import os
import subprocess
from importlib import metadata

_TRACKED_PACKAGES = ("torch", "numpy", "scipy", "lightning", "pandas")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def collect_provenance() -> dict:
    """Return a JSON-serializable provenance record for the current run."""
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "packages": _package_versions(),
    }
