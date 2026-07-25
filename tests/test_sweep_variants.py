"""Tests for the sweep `variants` generator (Cartesian named-variant crossing)."""

from timescales.sweep import generate_experiment_configs


def _sweep_config():
    return {
        "_base_config": {"a": 1, "recurrent_gain": 0.0, "nested": {"x": 0}},
        "fixed_overrides": {"a": 2},
        "naming": {"format": "{task}_{init}_g{recurrent_gain}"},
        "variants": {
            "task": {
                "ff": {"task": "flip_flop"},
                "sine": {"task": "sine_wave", "nested": {"x": 9}},
            },
            "init": {
                "uniform": {"tau_init": {"scheme": "uniform"}},
                "powerlaw": {"tau_init": {"scheme": "powerlaw"}},
            },
        },
        "grid": {"recurrent_gain": [0.5, 0.9]},
    }


def test_variant_count_and_names():
    configs = generate_experiment_configs(_sweep_config())
    assert len(configs) == 2 * 2 * 2
    names = [n for n, _ in configs]
    assert "ff_uniform_g0.5" in names
    assert "sine_powerlaw_g0.9" in names
    assert len(set(names)) == 8  # all unique


def test_variant_merge_semantics():
    configs = dict(generate_experiment_configs(_sweep_config()))
    c = configs["sine_powerlaw_g0.9"]
    assert c["a"] == 2                       # fixed override applied
    assert c["task"] == "sine_wave"          # task variant
    assert c["nested"]["x"] == 9             # deep merge from variant
    assert c["tau_init"]["scheme"] == "powerlaw"
    assert c["recurrent_gain"] == 0.9        # grid override on top


def test_variants_without_grid():
    sc = _sweep_config()
    del sc["grid"]
    sc["naming"] = {}
    configs = generate_experiment_configs(sc)
    assert len(configs) == 4
    assert configs[0][0] == "ff_uniform"  # default underscore naming


def test_base_not_mutated_across_experiments():
    configs = dict(generate_experiment_configs(_sweep_config()))
    # ff experiments must not see sine's nested override.
    assert configs["ff_uniform_g0.5"]["nested"]["x"] == 0
    assert configs["ff_uniform_g0.5"]["task"] == "flip_flop"
