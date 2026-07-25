"""Tests for timescales.mode_analysis — matching & coupling metrics (delta S5)."""

import math

import numpy as np

from timescales import mode_analysis as ma


# ------------------------------------------------------------- task timescales

def test_task_timescales_flip_flop_scalar_and_list():
    ts = ma.task_timescales_steps({"task": "flip_flop", "p_pulse": 0.05, "n_bits": 2})
    assert ts["decay"] == [20.0, 20.0] and ts["oscillation"] == []
    ts = ma.task_timescales_steps({"task": "flip_flop", "p_pulse": [0.1, 0.02], "n_bits": 2})
    assert ts["decay"] == [10.0, 50.0]


def test_task_timescales_sine_converts_dt():
    ts = ma.task_timescales_steps(
        {"task": "sine_wave", "periods": [20.0, 50.0], "dt": 0.5})
    assert ts["oscillation"] == [40.0, 100.0] and ts["decay"] == []


# ------------------------------------------------------------------ mode math

def test_mode_timescales_decay_and_osc():
    lam = [math.exp(-1.0 / 20.0), 0.9 * np.exp(1j * 2 * np.pi / 50.0)]
    out = ma.mode_timescales(lam)
    assert np.isclose(out["decay"].max(), 20.0)          # real mode decay
    assert np.isclose(out["oscillation"][0], 50.0)       # rotating mode period


def test_pair_indices_groups_conjugates():
    lam = np.array([0.5, 0.3 + 0.4j, 0.3 - 0.4j, -0.2])
    groups = ma.pair_indices(lam)
    assert sorted(len(g) for g in groups) == [1, 1, 2]
    pair = next(g for g in groups if len(g) == 2)
    assert np.isclose(lam[pair[0]], np.conj(lam[pair[1]]))


def test_matching_error_zero_when_exact():
    lam = [math.exp(-1.0 / 20.0), 0.5]
    m = ma.matching_errors(lam, {"decay": [20.0], "oscillation": []})
    assert np.isclose(m["decay"][0], 0.0)
    assert np.isclose(m["mean_abs_log_err"], 0.0)


def test_matching_error_log_ratio():
    lam = [math.exp(-1.0 / 10.0)]  # only a tau=10 mode for a tau=20 target
    m = ma.matching_errors(lam, {"decay": [20.0], "oscillation": []})
    assert np.isclose(m["decay"][0], abs(math.log(10.0 / 20.0)))


def test_matching_error_nan_when_no_modes_of_kind():
    m = ma.matching_errors([0.5], {"decay": [], "oscillation": [50.0]})
    assert np.isnan(m["oscillation"][0])
    assert m["mean_abs_log_err"] is None


# ------------------------------------------------------------------- coupling

def test_coupling_one_to_one_readout():
    # Real diagonal system: V = I, each output reads exactly one mode.
    N = 4
    eigvals = np.array([0.9, 0.8, 0.7, 0.6])
    V = np.eye(N)
    W_out = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    groups = ma.pair_indices(eigvals)
    C = ma.pair_couplings(W_out, V, groups)
    assert C.shape == (2, 4)
    assert np.isclose(ma.participation_ratio(C[0]), 1.0)
    assert ma.dominance_ratio(C[0]) == float("inf")
    assert ma.assignment_uniqueness(C) == 1.0


def test_coupling_dense_readout_high_pr():
    N = 6
    eigvals = np.linspace(0.4, 0.9, N)
    V = np.eye(N)
    W_out = np.ones((1, N))  # equally coupled to all modes
    C = ma.pair_couplings(W_out, V, ma.pair_indices(eigvals))
    assert np.isclose(ma.participation_ratio(C[0]), N)
    assert np.isclose(ma.dominance_ratio(C[0]), 1.0)


def test_assignment_uniqueness_collision():
    C = np.array([[1.0, 0.1], [0.9, 0.2]])  # both outputs argmax mode 0
    assert ma.assignment_uniqueness(C) == 0.0


def test_conjugate_pair_aggregation():
    # One conjugate pair: coupling should combine the two columns in quadrature.
    eigvals = np.array([0.3 + 0.4j, 0.3 - 0.4j])
    V = np.eye(2, dtype=complex)
    W_out = np.array([[3.0, 4.0]])
    C = ma.pair_couplings(W_out, V, ma.pair_indices(eigvals))
    assert C.shape == (1, 1)
    assert np.isclose(C[0, 0], 5.0)  # sqrt(3^2 + 4^2)


def test_coupling_metrics_includes_shuffled_control():
    rng = np.random.default_rng(1)
    N = 20
    eigvals = rng.uniform(0.2, 0.95, N)  # all real
    V = np.eye(N)
    W_out = np.zeros((3, N))
    W_out[0, 2] = W_out[1, 5] = W_out[2, 11] = 1.0
    out = ma.coupling_metrics(W_out, V, eigvals, rng=rng)
    assert np.isclose(out["coup_participation_ratio_mean"], 1.0)
    assert out["coup_assignment_uniqueness"] == 1.0
    assert "coup_shuffled_participation_ratio_mean" in out
    assert out["n_modes"] == N and out["n_outputs"] == 3
