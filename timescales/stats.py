"""Cross-seed statistics for any scalar metric (delta Section 7, amends v2/D2).

Rank-based defaults — Mann-Whitney U and Cliff's delta — because
steps-to-convergence, matching errors, and coupling scores are typically
non-normal and possibly heavy-tailed. Applies identically to every metric
(speed, final loss, matching error, coupling sparsity / one-to-one-ness).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sp_stats


def _clean(values) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    return x[np.isfinite(x)]


def compare_samples(x, y) -> dict:
    """Mann-Whitney U + Cliff's delta between two seed samples.

    Cliff's delta from the U statistic: ``delta = 2U/(n*m) - 1``, in [-1, 1];
    positive means x tends larger than y. Cohen's d included as secondary.
    """
    x, y = _clean(x), _clean(y)
    out: dict = {
        "n_x": int(x.size), "n_y": int(y.size),
        "median_x": float(np.median(x)) if x.size else None,
        "median_y": float(np.median(y)) if y.size else None,
        "U": None, "p_value": None, "cliffs_delta": None, "cohens_d": None,
    }
    if x.size == 0 or y.size == 0:
        return out

    U, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
    out["U"] = float(U)
    out["p_value"] = float(p)
    out["cliffs_delta"] = float(2.0 * U / (x.size * y.size) - 1.0)

    # Cohen's d (secondary): pooled-SD standardized mean difference.
    if x.size > 1 and y.size > 1:
        sp = math.sqrt(
            ((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1))
            / (x.size + y.size - 2)
        )
        out["cohens_d"] = float((x.mean() - y.mean()) / sp) if sp > 0 else None
    return out


def group_metric(rows: list[dict], metric: str, by: tuple[str, ...]) -> dict[tuple, list[float]]:
    """Group rows by the `by` keys and collect finite metric values per group."""
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        key = tuple(r.get(k) for k in by)
        groups.setdefault(key, []).append(v)
    return groups
