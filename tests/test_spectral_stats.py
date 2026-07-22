"""Tests for timescales.spectral_stats — named pinching statistics."""

import numpy as np

from timescales import spectral_stats as ss


def test_empty_returns_schema_with_nones():
    stats = ss.spectral_pinching_stats([])
    assert stats["n_eigs"] == 0
    assert stats["max_abs_lambda"] is None
    assert stats["decay_timescale_range"] is None
    assert stats["osc_period_min"] is None


def test_max_abs_and_gap():
    lam = [0.5, 0.9, 0.2 + 0.1j]
    stats = ss.spectral_pinching_stats(lam)
    assert abs(stats["max_abs_lambda"] - 0.9) < 1e-12
    assert abs(stats["gap_to_unit_circle"] - 0.1) < 1e-12


def test_frac_near_real_axis_uses_eps():
    lam = [0.5 + 0.0j, 0.5 + 0.05j, 0.5 + 0.5j]  # |Im| = 0, 0.05, 0.5
    assert ss.spectral_pinching_stats(lam, eps_real_axis=0.1)["frac_near_real_axis"] == 2 / 3
    assert ss.spectral_pinching_stats(lam, eps_real_axis=0.01)["frac_near_real_axis"] == 1 / 3


def test_decay_timescale_from_magnitude():
    # |lambda| = exp(-1/tau) -> tau_decay = -1/ln|lambda|.
    taus = np.array([2.0, 50.0])
    lam = np.exp(-1.0 / taus)  # real, contracting
    stats = ss.spectral_pinching_stats(lam)
    assert abs(stats["decay_timescale_min"] - 2.0) < 1e-6
    assert abs(stats["decay_timescale_max"] - 50.0) < 1e-6
    assert abs(stats["decay_timescale_range"] - 48.0) < 1e-6


def test_n_unstable_and_contracting_excluded():
    lam = [0.5, 1.2, 1.0]  # one clearly unstable, one on the circle
    stats = ss.spectral_pinching_stats(lam)
    assert stats["n_unstable"] == 2  # |lambda| >= 1 counts 1.2 and 1.0
    # only 0.5 is contracting -> a single decay timescale, range 0
    assert stats["decay_timescale_range"] == 0.0


def test_oscillatory_fraction_and_period():
    # A conjugate pair at angle +/- pi/2 -> period 4 timesteps; plus a real mode.
    lam = [0.8j, -0.8j, 0.5]
    stats = ss.spectral_pinching_stats(lam)
    assert abs(stats["frac_oscillatory"] - 2 / 3) < 1e-12
    assert abs(stats["osc_period_min"] - 4.0) < 1e-6
