"""Tests for timescales.presets (trainable / tau_init) and the lognormal tau draw."""

import math

import pytest
import torch

from timescales import presets
from timescales.rnns.rnn import RNN


# ---------------------------------------------------------------- trainable

def test_trainable_full():
    cfg = {"trainable": "full"}
    presets.resolve_trainable(cfg)
    assert cfg["learn_time_constants"] is True
    assert "freeze_wrec" not in cfg


def test_trainable_fixedA():
    cfg = {"trainable": "fixedA", "learn_time_constants": True}
    presets.resolve_trainable(cfg)
    assert cfg["learn_time_constants"] is False
    assert "freeze_wrec" not in cfg


def test_trainable_reservoir():
    cfg = {"trainable": "reservoir"}
    presets.resolve_trainable(cfg)
    assert cfg["learn_time_constants"] is False
    assert cfg["freeze_wrec"] is True
    assert cfg["freeze_win"] is True


def test_trainable_unknown_raises():
    with pytest.raises(ValueError, match="trainable preset"):
        presets.resolve_trainable({"trainable": "bogus"})


def test_trainable_absent_is_noop():
    cfg = {"learn_time_constants": True}
    presets.resolve_trainable(cfg)
    assert cfg == {"learn_time_constants": True}


# ---------------------------------------------------------------- tau_init

def test_tau_init_uniform_fixed():
    cfg = {"learn_time_constants": False, "tau_init": {"scheme": "uniform", "value": 1.0}}
    presets.resolve_tau_init(cfg)
    assert cfg["time_constants_config"] == {"type": "discrete", "values": [1.0]}


def test_tau_init_uniform_learnable():
    cfg = {
        "learn_time_constants": True,
        "init_time_constants_config": {"stale": True},
        "tau_init": {"scheme": "uniform", "value": 2.0},
    }
    presets.resolve_tau_init(cfg)
    assert cfg["init_time_constant"] == 2.0
    assert "init_time_constants_config" not in cfg  # stale key cleared


def test_tau_init_powerlaw_fixed_range_fallback():
    # tau range comes from top-level keys (set by the task variant).
    cfg = {
        "learn_time_constants": False,
        "tau_min": 1.0, "tau_max": 200.0,
        "tau_init": {"scheme": "powerlaw", "exponent": 1.0},
    }
    presets.resolve_tau_init(cfg)
    tc = cfg["time_constants_config"]
    assert tc["distribution"] == "powerlaw"
    assert tc["exponent"] == 1.0
    assert tc["min_time_constant"] == 1.0
    assert tc["max_time_constant"] == 200.0


def test_tau_init_powerlaw_learnable_routes_to_init_config():
    cfg = {
        "learn_time_constants": True,
        "init_time_constant": 1.0,
        "tau_init": {"scheme": "powerlaw", "tau_min": 10.0, "tau_max": 200.0},
    }
    presets.resolve_tau_init(cfg)
    assert cfg["init_time_constants_config"]["distribution"] == "powerlaw"
    assert "init_time_constant" not in cfg


def test_tau_init_lognormal_defaults_match_range():
    cfg = {
        "learn_time_constants": False,
        "tau_min": 1.0, "tau_max": 200.0,
        "tau_init": {"scheme": "lognormal"},
    }
    presets.resolve_tau_init(cfg)
    tc = cfg["time_constants_config"]
    assert tc["distribution"] == "lognormal"
    assert abs(tc["median"] - math.sqrt(200.0)) < 1e-9
    assert abs(tc["sigma"] - math.log(200.0) / 4.0) < 1e-9
    assert tc["min_time_constant"] == 1.0
    assert tc["max_time_constant"] == 200.0


def test_tau_init_lognormal_explicit_params_win():
    cfg = {
        "learn_time_constants": False,
        "tau_min": 1.0, "tau_max": 200.0,
        "tau_init": {"scheme": "lognormal", "median": 30.0, "sigma": 0.5},
    }
    presets.resolve_tau_init(cfg)
    assert cfg["time_constants_config"]["median"] == 30.0
    assert cfg["time_constants_config"]["sigma"] == 0.5


def test_tau_init_missing_range_raises():
    with pytest.raises(ValueError, match="tau_min/tau_max"):
        presets.resolve_tau_init({"tau_init": {"scheme": "powerlaw"}})


def test_tau_init_unknown_scheme_raises():
    with pytest.raises(ValueError, match="tau_init scheme"):
        presets.resolve_tau_init({"tau_init": {"scheme": "bogus"}})


def test_resolve_presets_order_trainable_first():
    # trainable=full must flip learn_time_constants BEFORE tau_init routes.
    cfg = {
        "trainable": "full",
        "learn_time_constants": False,
        "tau_min": 1.0, "tau_max": 200.0,
        "tau_init": {"scheme": "powerlaw"},
    }
    presets.resolve_presets(cfg)
    assert cfg["learn_time_constants"] is True
    assert "init_time_constants_config" in cfg
    assert "powerlaw" == cfg["init_time_constants_config"]["distribution"]


# ------------------------------------------------------- lognormal sampling

def test_generate_lognormal_taus_clipped_and_centered():
    torch.manual_seed(0)
    cfg = {
        "type": "continuous",
        "distribution": "lognormal",
        "median": math.sqrt(200.0),
        "sigma": math.log(200.0) / 4.0,
        "min_time_constant": 1.0,
        "max_time_constant": 200.0,
    }
    taus = RNN._generate_time_constants(None, 2000, cfg)
    assert taus.shape == (2000,)
    assert taus.min() >= 1.0
    assert taus.max() <= 200.0
    # Median of the draw should sit near the configured median (log-scale check).
    med = taus.median().item()
    assert 0.5 * math.sqrt(200.0) < med < 2.0 * math.sqrt(200.0)
