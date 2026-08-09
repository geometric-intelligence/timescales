import json

import pytest

from timescales.rich_learning_analysis import (
    analyze_run,
    compute_component_timing,
    compute_curve_shape,
)


def test_abrupt_drop_is_more_concentrated_than_gradual_improvement():
    steps = list(range(0, 200, 10))
    gradual = [10 ** (-i / 19) for i in range(20)]
    abrupt = [1.0] * 10 + [0.1] * 10

    gradual_metrics = compute_curve_shape(steps, gradual)
    abrupt_metrics = compute_curve_shape(steps, abrupt)

    assert gradual_metrics["log10_improvement"] == pytest.approx(1.0, abs=0.12)
    assert abrupt_metrics["log10_improvement"] == pytest.approx(1.0)
    assert (
        abrupt_metrics["drop_concentration_top_10pct"]
        > gradual_metrics["drop_concentration_top_10pct"]
    )
    assert abrupt_metrics["transition_width_steps"] < gradual_metrics[
        "transition_width_steps"
    ]


def test_flat_curve_does_not_receive_shape_scores():
    metrics = compute_curve_shape([10, 20, 30, 40], [1.0, 0.999, 1.001, 1.0])
    assert metrics["meaningful_improvement"] is False
    assert metrics["drop_concentration_top_10pct"] is None
    assert metrics["t50_steps"] is None


def test_sine_components_are_adjacent_output_pairs():
    curve = {
        "steps": [10, 20, 30, 40, 50, 60, 70, 80],
        "val_losses_per_bit": {
            "channel_0": [1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "channel_1": [1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "channel_2": [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1],
            "channel_3": [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1],
        },
    }
    metrics = compute_component_timing(curve, "sine_wave")
    assert metrics["n_components"] == 2
    assert metrics["component_0_t50_steps"] < metrics["component_1_t50_steps"]
    assert metrics["component_t50_spread_steps"] > 0


def test_analyze_run_includes_config_and_restructuring(tmp_path):
    run_dir = tmp_path / "example" / "seed_0"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.yaml").write_text(
        "task: flip_flop\n"
        "activation: Identity\n"
        "optimizer_name: adam\n"
        "wrec_init_scale: 0.1\n"
        "output_coupling_gamma: 0.3\n"
        "seed: 0\n"
    )
    curve = {
        "steps": [10, 20, 30, 40, 50, 60],
        "val_losses": [1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
        "val_losses_per_bit": {"channel_0": [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]},
    }
    (run_dir / "training_losses.json").write_text(json.dumps(curve))
    (run_dir / "recurrent_restructuring.json").write_text(
        json.dumps({"records": [{"effective_wrec_delta_rms": 0.012}]})
    )

    row = analyze_run(run_dir)
    assert row["task"] == "flip_flop"
    assert row["optimizer_name"] == "adam"
    assert row["output_coupling_gamma"] == pytest.approx(0.3)
    assert row["final_effective_wrec_delta_rms"] == pytest.approx(0.012)
