"""Focused tests for the rich-learning parameterization and sweep."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

from timescales.callbacks import (
    RecurrentRestructuringCallback,
    SpectralSnapshotCallback,
)
from timescales.optimizers import SGLD
from timescales.rnns.rnn import RNN, RNNLightning
from timescales.sweep import generate_experiment_configs


def _model(**overrides) -> RNN:
    kwargs = {
        "input_size": 2,
        "hidden_size": 64,
        "output_size": 2,
        "dt": 0.1,
        "time_constants_config": {"type": "discrete", "values": [1.0]},
        "activation": nn.Tanh,
        "wrec_init": "normal_scaled",
        "dynamics_type": "voltage",
        "zero_diag_wrec": False,
    }
    kwargs.update(overrides)
    return RNN(**kwargs)


def _lightning(model: RNN, **overrides) -> RNNLightning:
    kwargs = {
        "model": model,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "step_size": 1000,
        "gamma": 0.5,
        "task": "sine_wave",
        "use_lr_scheduler": False,
    }
    kwargs.update(overrides)
    return RNNLightning(**kwargs)


def test_voltage_readout_uses_activation_but_rate_readout_does_not():
    hidden = torch.tensor([[0.5, -1.0]])
    voltage = _model(hidden_size=2, input_size=1, output_size=1)
    assert torch.allclose(voltage.readout_features(hidden), torch.tanh(hidden))

    rate = _model(
        hidden_size=2,
        input_size=1,
        output_size=1,
        dynamics_type="rate",
    )
    assert torch.equal(rate.readout_features(hidden), hidden)


def test_clark_parameterization_scales_raw_recurrence_and_readout():
    torch.manual_seed(0)
    n = 256
    init_scale = 0.3
    gamma = 0.5
    model = _model(
        hidden_size=n,
        recurrent_gain=0.5,
        recurrent_parameterization="clark",
        output_coupling_gamma=gamma,
        wrec_init_scale=init_scale,
        use_biases=False,
    )

    assert model.rnn_step.recurrent_weight_scale == pytest.approx(0.5 / n**0.5)
    assert model.rnn_step.W_rec.weight.std().item() == pytest.approx(
        init_scale, rel=0.03
    )
    assert model.W_out.weight.std().item() == pytest.approx(1.0, rel=0.15)
    assert model.readout_scale == pytest.approx(1.0 / (n * gamma))
    assert torch.allclose(
        model.effective_readout_weight,
        model.W_out.weight / (n * gamma),
    )
    assert model.rnn_step.W_rec.bias is None
    assert model.rnn_step.W_in.bias is None


def test_svd_low_rank_matches_paired_dense_draw_and_preserves_frobenius_norm():
    n = 32
    rank = 5
    init_scale = 0.4

    torch.manual_seed(123)
    dense = _model(
        hidden_size=n,
        recurrent_parameterization="clark",
        wrec_init="normal_scaled",
        wrec_init_scale=init_scale,
        use_biases=False,
    )
    dense_weight = dense.rnn_step.W_rec.weight.detach()

    torch.manual_seed(123)
    low_rank = _model(
        hidden_size=n,
        recurrent_parameterization="clark",
        wrec_init="svd_low_rank",
        wrec_init_config={"rank": rank, "normalization": "frobenius"},
        wrec_init_scale=init_scale,
        use_biases=False,
    )
    low_rank_weight = low_rank.rnn_step.W_rec.weight.detach()

    left, singular_values, right_h = torch.linalg.svd(
        dense_weight, full_matrices=False
    )
    expected = (
        left[:, :rank] * singular_values[:rank]
    ) @ right_h[:rank, :]
    expected *= torch.linalg.vector_norm(dense_weight) / torch.linalg.vector_norm(
        expected
    )

    torch.testing.assert_close(low_rank_weight, expected, rtol=2e-4, atol=2e-5)
    assert torch.linalg.vector_norm(low_rank_weight).item() == pytest.approx(
        torch.linalg.vector_norm(dense_weight).item(), rel=1e-6
    )
    numerical_rank = torch.linalg.matrix_rank(
        low_rank_weight,
        atol=0.0,
        rtol=n * torch.finfo(low_rank_weight.dtype).eps,
    )
    assert numerical_rank.item() == rank


def test_svd_low_rank_full_rank_recovers_dense_baseline():
    n = 24
    torch.manual_seed(456)
    dense = _model(
        hidden_size=n,
        recurrent_parameterization="clark",
        wrec_init="normal_scaled",
        use_biases=False,
    )
    torch.manual_seed(456)
    full_rank = _model(
        hidden_size=n,
        recurrent_parameterization="clark",
        wrec_init="svd_low_rank",
        wrec_init_config={"rank": n},
        use_biases=False,
    )
    torch.testing.assert_close(
        full_rank.rnn_step.W_rec.weight,
        dense.rnn_step.W_rec.weight,
        rtol=2e-4,
        atol=2e-5,
    )


def test_svd_low_rank_initialization_does_not_constrain_training_rank():
    n = 16
    model = _model(
        hidden_size=n,
        recurrent_parameterization="clark",
        wrec_init="svd_low_rank",
        wrec_init_config={"rank": 2},
        use_biases=False,
    )
    weight = model.rnn_step.W_rec.weight
    assert isinstance(weight, nn.Parameter)

    optimizer = torch.optim.SGD([weight], lr=0.1)
    optimizer.zero_grad()
    weight.grad = torch.randn_like(weight)
    optimizer.step()
    assert torch.linalg.matrix_rank(weight).item() == n


def test_spectral_snapshot_records_low_rank_diagnostics(tmp_path):
    model = _model(
        hidden_size=8,
        recurrent_gain=0.5,
        recurrent_parameterization="clark",
        wrec_init="svd_low_rank",
        wrec_init_config={"rank": 3},
        use_biases=False,
    )
    callback = SpectralSnapshotCallback(str(tmp_path))
    callback._dump(model, tag="init")

    snapshot = torch.load(tmp_path / "spectral_init.pt", weights_only=False)
    assert snapshot["wrec_init"] == "svd_low_rank"
    assert snapshot["wrec_init_config"]["rank"] == 3
    assert snapshot["wrec_numerical_rank"] == 3
    assert snapshot["singular_values_wrec"].shape == (8,)
    assert 1.0 <= snapshot["wrec_stable_rank"] <= 3.0 + 1e-5
    assert snapshot["wrec_spectral_radius"] >= 0.0

    with (tmp_path / "spectral_stats_init.json").open() as f:
        stats = json.load(f)
    assert stats["wrec_numerical_rank"] == 3
    assert stats["wrec_stable_rank"] == pytest.approx(
        snapshot["wrec_stable_rank"]
    )


@pytest.mark.parametrize("rank", [None, True, 0, 17, 1.5])
def test_svd_low_rank_rejects_invalid_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        _model(
            hidden_size=16,
            recurrent_parameterization="clark",
            wrec_init="svd_low_rank",
            wrec_init_config={"rank": rank},
            use_biases=False,
        )


def test_svd_low_rank_rejects_zero_diagonal_and_other_normalizations():
    with pytest.raises(ValueError, match="zero_diag_wrec=False"):
        _model(
            hidden_size=16,
            wrec_init="svd_low_rank",
            wrec_init_config={"rank": 3},
            zero_diag_wrec=True,
        )
    with pytest.raises(ValueError, match="normalization='frobenius'"):
        _model(
            hidden_size=16,
            wrec_init="svd_low_rank",
            wrec_init_config={"rank": 3, "normalization": "spectral"},
        )


def test_clark_loss_scaling_changes_objective_not_reported_task_loss():
    model = _model(
        hidden_size=32,
        recurrent_parameterization="clark",
        output_coupling_gamma=0.25,
        use_biases=False,
    )
    module = _lightning(model)
    task_loss = torch.tensor(2.0)
    expected_scale = 0.5 * 32 * 0.25**2
    assert module.task_loss_scale == pytest.approx(expected_scale)
    assert module._compute_objective(task_loss).item() == pytest.approx(
        expected_scale * task_loss.item()
    )


def test_optimizer_selection_has_no_scheduler_when_disabled():
    adam_module = _lightning(_model(), optimizer_name="adam")
    assert isinstance(adam_module.configure_optimizers(), torch.optim.Adam)

    sgld_module = _lightning(
        _model(),
        optimizer_name="sgld",
        sgld_beta=2000.0,
        sgld_add_noise=False,
    )
    assert isinstance(sgld_module.configure_optimizers(), SGLD)


def test_sgld_without_noise_is_gradient_descent():
    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([2.0])
    optimizer = SGLD([parameter], lr=0.1, beta=10.0, add_noise=False)
    optimizer.step()
    assert parameter.item() == pytest.approx(0.8)


def test_restructuring_callback_records_absolute_relative_and_directional_metrics(
    tmp_path,
):
    model = _model(hidden_size=4, recurrent_gain=0.5)

    class Module:
        def __init__(self, wrapped_model):
            self.model = wrapped_model

        def log(self, *args, **kwargs):
            pass

    trainer = SimpleNamespace(global_step=0, current_epoch=0, sanity_checking=False)
    module = Module(model)
    callback = RecurrentRestructuringCallback(str(tmp_path))
    callback.on_fit_start(trainer, module)

    with torch.no_grad():
        model.rnn_step.W_rec.weight.add_(0.1)
    trainer.global_step = 10
    trainer.current_epoch = 1
    callback.on_validation_epoch_end(trainer, module)

    with (tmp_path / "recurrent_restructuring.json").open() as f:
        records = json.load(f)["records"]
    assert [record["tag"] for record in records] == ["init", "validation"]
    final = records[-1]
    assert final["wrec_delta_fro"] > 0.0
    assert final["wrec_relative_delta_fro"] > 0.0
    assert final["wrec_delta_orthogonal_fro"] >= 0.0
    assert final["effective_wrec_delta_fro"] == pytest.approx(
        0.5 * final["wrec_delta_fro"]
    )


def test_rich_learning_sweep_has_expected_cartesian_product():
    project_root = Path(__file__).parents[1]
    sweep_path = (
        project_root
        / "timescales"
        / "sweep_configs"
        / "rnn"
        / "rich_learning_grid.yaml"
    )
    with sweep_path.open() as f:
        sweep = yaml.safe_load(f)
    with (project_root / "timescales" / sweep["base_config"]).open() as f:
        sweep["_base_config"] = yaml.safe_load(f)

    configs = generate_experiment_configs(sweep)
    assert len(configs) == 2 * 2 * 2 * 4 * 4
    assert len({name for name, _ in configs}) == len(configs)

    config = dict(configs)["sine_tanh_sgld_s0.03_gamma1"]
    assert config["task"] == "sine_wave"
    assert config["activation"] == "Tanh"
    assert config["optimizer_name"] == "sgld"
    assert config["learning_rate"] == pytest.approx(0.05)
    assert config["use_lr_scheduler"] is False
    assert config["weight_decay"] == 0.0
    assert config["wrec_init_scale"] == pytest.approx(0.03)
    assert config["output_coupling_gamma"] == pytest.approx(1.0)


def test_rich_learning_adam_discovery_sweep_has_expected_grid_and_horizon():
    project_root = Path(__file__).parents[1]
    sweep_path = (
        project_root
        / "timescales"
        / "sweep_configs"
        / "rnn"
        / "rich_learning_adam_discovery_4k.yaml"
    )
    with sweep_path.open() as f:
        sweep = yaml.safe_load(f)
    with (project_root / "timescales" / sweep["base_config"]).open() as f:
        sweep["_base_config"] = yaml.safe_load(f)

    configs = generate_experiment_configs(sweep)
    assert len(configs) == 2 * 2 * 8 * 8
    assert len({name for name, _ in configs}) == len(configs)
    assert sweep["n_seeds"] == 1

    config = dict(configs)["sine_tanh_adam_s0.01_gamma2"]
    assert config["task"] == "sine_wave"
    assert config["activation"] == "Tanh"
    assert config["optimizer_name"] == "adam"
    assert config["learning_rate"] == pytest.approx(0.001)
    assert config["max_steps"] == 4000
    assert config["val_every_n_steps"] == 10
    assert config["save_checkpoint_every_n_epochs"] == 100
    assert (
        config["save_checkpoint_every_n_epochs"]
        * config["val_every_n_steps"]
        == 1000
    )
    assert config["use_lr_scheduler"] is False
    assert config["weight_decay"] == 0.0
    assert config["recurrent_gain"] == pytest.approx(0.5)
    assert config["wrec_init_scale"] == pytest.approx(0.01)
    assert config["output_coupling_gamma"] == pytest.approx(2.0)


def test_low_rank_discovery_sweep_has_48_paired_rank_conditions():
    project_root = Path(__file__).parents[1]
    sweep_path = (
        project_root
        / "timescales"
        / "sweep_configs"
        / "rnn"
        / "rich_learning_low_rank_discovery_4k.yaml"
    )
    with sweep_path.open() as f:
        sweep = yaml.safe_load(f)
    with (project_root / "timescales" / sweep["base_config"]).open() as f:
        sweep["_base_config"] = yaml.safe_load(f)

    configs = generate_experiment_configs(sweep)
    assert len(configs) == 6 * 8
    assert len({name for name, _ in configs}) == len(configs)
    assert sweep["n_seeds"] == 1

    by_name = dict(configs)
    sine = by_name["sine_linear_s0.64_gamma0.25_rank16"]
    assert sine["task"] == "sine_wave"
    assert sine["activation"] == "Identity"
    assert sine["wrec_init_scale"] == pytest.approx(0.64)
    assert sine["output_coupling_gamma"] == pytest.approx(0.25)
    assert sine["wrec_init"] == "svd_low_rank"
    assert sine["wrec_init_config"] == {
        "normalization": "frobenius",
        "rank": 16,
    }

    flip_flop = by_name["ff_tanh_s0.08_gamma0.03_rank512"]
    assert flip_flop["task"] == "flip_flop"
    assert flip_flop["activation"] == "Tanh"
    assert flip_flop["n_bits"] == 3
    assert flip_flop["p_pulse"] == [0.005, 0.02, 0.1]
    assert flip_flop["wrec_init_config"]["rank"] == 512

    for _, config in configs:
        assert config["hidden_size"] == 512
        assert config["recurrent_gain"] == pytest.approx(0.5)
        assert config["optimizer_name"] == "adam"
        assert config["max_steps"] == 4000
        assert config["use_lr_scheduler"] is False
        assert config["weight_decay"] == 0.0
        assert config["zero_diag_wrec"] is False


def test_forced_pulse_pair_sweep_has_two_full_rank_runs_and_dense_checkpoints():
    project_root = Path(__file__).parents[1]
    sweep_path = (
        project_root
        / "timescales"
        / "sweep_configs"
        / "rnn"
        / "rich_learning_forced_pulse_pair_4k.yaml"
    )
    with sweep_path.open() as f:
        sweep = yaml.safe_load(f)
    with (project_root / "timescales" / sweep["base_config"]).open() as f:
        sweep["_base_config"] = yaml.safe_load(f)

    configs = generate_experiment_configs(sweep)
    assert len(configs) == 2
    assert sweep["n_seeds"] == 1
    by_name = dict(configs)

    flip_flop = by_name[
        "ff_tanh_forced_tau100_200_400_s0.08_gamma0.03"
    ]
    assert flip_flop["task"] == "flip_flop"
    assert flip_flop["activation"] == "Tanh"
    assert flip_flop["p_pulse"] == pytest.approx([0.01, 0.005, 0.0025])
    assert flip_flop["force_initial_pulse"] is True
    assert flip_flop["num_time_steps"] == 2000
    assert flip_flop["init_hidden_value"] is None

    sine = by_name["sine_linear_s0.08_gamma1"]
    assert sine["task"] == "sine_wave"
    assert sine["activation"] == "Identity"
    assert sine["periods"] == pytest.approx([20.0, 50.0, 100.0])

    for config in by_name.values():
        assert config["wrec_init"] == "normal_scaled"
        assert config["wrec_init_scale"] == pytest.approx(0.08)
        assert config["max_steps"] == 4000
        assert (
            config["save_checkpoint_every_n_epochs"]
            * config["val_every_n_steps"]
            == 100
        )
