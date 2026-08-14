"""Sweep and factory integration tests for transformer experiments."""

from pathlib import Path

from timescales.sweep import generate_experiment_configs, load_sweep_config


def test_transformer_architecture_smoke_sweep_has_all_five_variants():
    root = Path(__file__).parents[1]
    path = root / "timescales/sweep_configs/transformer/multiscale_hmm_architecture_smoke.yaml"
    sweep = load_sweep_config(str(path))
    experiments = generate_experiment_configs(sweep)
    architectures = {config["architecture"] for _, config in experiments}
    assert architectures == {
        "linear_fir",
        "static_linear_attention",
        "linear_attention",
        "softmax_attention",
        "softmax_tanh",
    }


def test_rich_learning_pilot_contains_27_conditions():
    root = Path(__file__).parents[1]
    path = root / "timescales/sweep_configs/transformer/multiscale_hmm_rich_learning_pilot.yaml"
    sweep = load_sweep_config(str(path))
    experiments = generate_experiment_configs(sweep)
    assert len(experiments) == 27
    assert {config["model_type"] for _, config in experiments} == {"transformer"}
