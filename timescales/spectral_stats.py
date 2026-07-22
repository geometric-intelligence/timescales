"""Named spectral-pinching statistics from a Jacobian's eigenvalues.

Quantifies the "power-law init expands the range of available timescales" claim so
it is measured, not only shown (spec Workstream C2). Operates on the complex
eigenvalues of the one-step Jacobian ``J = (I - A) + g A W_rec`` captured at init by
``SpectralSnapshotCallback``.

Discrete-time reading: ``r_{t+1} = J r_t``, so an eigenvalue ``lambda`` is a mode
with per-step gain ``|lambda|`` and rotation ``arg(lambda)``. Timescales are in
*timesteps* (matching the tau range [1, 200] used for the power-law init)::

    decay timescale      tau_decay = -1 / ln|lambda|        for |lambda| < 1
    oscillation period   T_osc     = 2*pi / |arg(lambda)|   for arg(lambda) != 0
"""

from __future__ import annotations

import numpy as np

DEFAULT_EPS_REAL_AXIS = 0.1

# Keys always present in the returned dict (None when undefined), so rows from
# different runs share a schema for the columnar run-table.
_DECAY_KEYS = (
    "decay_timescale_min",
    "decay_timescale_max",
    "decay_timescale_range",
    "decay_timescale_p05",
    "decay_timescale_p50",
    "decay_timescale_p95",
)
_OSC_KEYS = ("osc_period_min", "osc_period_max")


def spectral_pinching_stats(
    eigvals, eps_real_axis: float = DEFAULT_EPS_REAL_AXIS
) -> dict:
    """Compute pinching statistics from complex eigenvalues of ``J``.

    Returns (all timescales in timesteps):
    - ``max_abs_lambda``       : max |lambda| (closeness to the stability boundary).
    - ``gap_to_unit_circle``   : ``1 - max|lambda|``.
    - ``frac_near_real_axis``  : fraction with ``|Im(lambda)| < eps_real_axis``.
    - ``frac_oscillatory``     : fraction with a non-trivial rotation.
    - ``n_unstable``           : count with ``|lambda| >= 1``.
    - ``decay_timescale_*``    : spread of decay timescales over contracting modes.
    - ``osc_period_*``         : spread of oscillation periods over rotating modes.
    """
    lam = np.asarray(eigvals, dtype=complex).ravel()
    stats: dict = {
        "n_eigs": int(lam.size),
        "eps_real_axis": float(eps_real_axis),
        "max_abs_lambda": None,
        "gap_to_unit_circle": None,
        "frac_near_real_axis": None,
        "frac_oscillatory": None,
        "n_unstable": None,
    }
    for k in _DECAY_KEYS + _OSC_KEYS:
        stats[k] = None
    if lam.size == 0:
        return stats

    abs_lam = np.abs(lam)
    im = np.abs(lam.imag)
    max_abs = float(abs_lam.max())

    stats["max_abs_lambda"] = max_abs
    stats["gap_to_unit_circle"] = float(1.0 - max_abs)
    stats["frac_near_real_axis"] = float(np.mean(im < eps_real_axis))
    stats["n_unstable"] = int(np.count_nonzero(abs_lam >= 1.0))

    # Decay timescales for contracting modes (|lambda| < 1); log(1)=0 -> exclude.
    contracting = abs_lam < 1.0
    with np.errstate(divide="ignore"):
        tau_decay = -1.0 / np.log(abs_lam[contracting])
    tau_decay = tau_decay[np.isfinite(tau_decay)]
    if tau_decay.size:
        stats["decay_timescale_min"] = float(tau_decay.min())
        stats["decay_timescale_max"] = float(tau_decay.max())
        stats["decay_timescale_range"] = float(tau_decay.max() - tau_decay.min())
        stats["decay_timescale_p05"] = float(np.percentile(tau_decay, 5))
        stats["decay_timescale_p50"] = float(np.percentile(tau_decay, 50))
        stats["decay_timescale_p95"] = float(np.percentile(tau_decay, 95))

    # Oscillation periods for rotating modes (non-zero imaginary angle).
    theta = np.abs(np.angle(lam))
    rotating = theta > 1e-9
    stats["frac_oscillatory"] = float(np.mean(rotating))
    if np.any(rotating):
        t_osc = 2.0 * np.pi / theta[rotating]
        stats["osc_period_min"] = float(t_osc.min())
        stats["osc_period_max"] = float(t_osc.max())

    return stats
