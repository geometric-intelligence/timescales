import numpy as np
import pytest
import torch
import torch.nn as nn

from timescales.datamodules import CumulativeVectorAdditionDataModule
from timescales.datamodules.cumulative_vector_addition import (
    generate_cumulative_vector_addition_trajectories,
)
from timescales.rnns.rnn import RNN, RNNLightning


def test_targets_are_inclusive_cumulative_sums():
    inputs, targets, states = generate_cumulative_vector_addition_trajectories(
        num_trajectories=5,
        num_time_steps=12,
        vector_size=3,
        increment_std=0.2,
        rng=np.random.RandomState(7),
    )

    assert inputs.shape == (5, 12, 3)
    np.testing.assert_allclose(targets, np.cumsum(inputs, axis=1), rtol=1e-6)
    np.testing.assert_array_equal(states, targets)
    np.testing.assert_array_equal(targets[:, 0, :], inputs[:, 0, :])


def test_sparse_setting_can_disable_all_addition_events():
    inputs, targets, states = generate_cumulative_vector_addition_trajectories(
        num_trajectories=4,
        num_time_steps=8,
        vector_size=2,
        increment_std=1.0,
        increment_probability=0.0,
        rng=np.random.RandomState(3),
    )

    np.testing.assert_array_equal(inputs, 0.0)
    np.testing.assert_array_equal(targets, 0.0)
    np.testing.assert_array_equal(states, 0.0)


def test_rademacher_distribution_produces_signed_discrete_steps():
    inputs, targets, _ = generate_cumulative_vector_addition_trajectories(
        num_trajectories=10,
        num_time_steps=20,
        vector_size=2,
        increment_distribution="rademacher",
        increment_std=2.0,
        rng=np.random.RandomState(5),
    )

    assert set(np.unique(inputs)) == {-2.0, 2.0}
    np.testing.assert_array_equal(targets, np.cumsum(inputs, axis=1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vector_size", 0),
        ("increment_std", -0.1),
        ("increment_probability", 1.1),
        ("increment_distribution", "uniform"),
    ],
)
def test_invalid_task_parameters_are_rejected(field, value):
    kwargs = {
        "num_trajectories": 2,
        "num_time_steps": 4,
        "vector_size": 2,
        "increment_distribution": "gaussian",
        "increment_std": 0.1,
        "increment_probability": 1.0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        generate_cumulative_vector_addition_trajectories(**kwargs)


def test_datamodule_exposes_matching_input_and_output_dimensions():
    datamodule = CumulativeVectorAdditionDataModule(
        vector_size=4,
        increment_std=0.05,
        num_time_steps=10,
        num_val_trajectories=6,
        batch_size=3,
        num_workers=0,
    )
    datamodule.setup()

    assert datamodule.input_size == 4
    assert datamodule.output_size == 4
    val_inputs, val_states, val_targets = next(iter(datamodule.val_dataloader()))
    assert val_inputs.shape == (3, 10, 4)
    assert val_states.shape == val_targets.shape == val_inputs.shape


def test_rnn_lightning_uses_mse_and_r_squared_for_integration_task():
    model = RNN(
        input_size=2,
        hidden_size=8,
        output_size=2,
        dt=1.0,
        time_constants_config={"type": "discrete", "values": [1.0]},
        activation=nn.Identity,
        wrec_init="normal_scaled",
        zero_diag_wrec=False,
        dynamics_type="voltage",
    )
    lightning = RNNLightning(
        model=model,
        learning_rate=1e-3,
        weight_decay=0.0,
        step_size=100,
        gamma=1.0,
        task="cumulative_vector_addition",
        use_lr_scheduler=False,
    )

    inputs, targets, _ = generate_cumulative_vector_addition_trajectories(
        num_trajectories=3,
        num_time_steps=5,
        vector_size=2,
        rng=np.random.RandomState(11),
    )
    _, predictions = model(torch.from_numpy(inputs))
    training_loss, _ = lightning._compute_loss(
        predictions, torch.from_numpy(targets)
    )
    training_loss.backward()
    assert torch.isfinite(training_loss)
    assert model.rnn_step.W_rec.weight.grad is not None

    target_tensor = torch.from_numpy(targets)
    loss, per_channel = lightning._compute_loss(target_tensor, target_tensor)
    score, per_channel_score = lightning._compute_accuracy(
        target_tensor, target_tensor
    )

    assert loss.item() == pytest.approx(0.0)
    assert set(per_channel) == {"channel_0", "channel_1"}
    assert score.item() == pytest.approx(1.0)
    assert per_channel_score == pytest.approx({"channel_0": 1.0, "channel_1": 1.0})
