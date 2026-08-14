"""Tests for the multiscale noisy-symbol task and its theory references."""

import numpy as np
import pytest

from timescales.datamodules.multiscale_hmm import (
    MultiscaleHMMDataModule,
    estimate_bayes_mse,
    generate_multiscale_hmm_sequences,
    optimal_linear_predictor,
    timescale_to_pole,
)


def test_generator_shapes_values_and_alignment():
    inputs, latent_targets, targets = generate_multiscale_hmm_sequences(
        num_trajectories=7,
        num_time_steps=19,
        timescales=[4.0, 8.0, 16.0],
        observation_flip_probability=0.0,
        rng=np.random.default_rng(3),
    )
    assert inputs.shape == latent_targets.shape == targets.shape == (7, 19, 3)
    assert set(np.unique(inputs)) <= {-1.0, 1.0}
    assert set(np.unique(targets)) <= {-1.0, 1.0}
    np.testing.assert_array_equal(latent_targets, targets)


def test_fixed_validation_data_are_deterministic():
    kwargs = dict(
        n_components=3,
        timescales=[4.0, 8.0, 16.0],
        observation_flip_probability=0.15,
        num_time_steps=24,
        num_val_trajectories=10,
        batch_size=5,
        num_workers=0,
        validation_seed=17,
    )
    first = MultiscaleHMMDataModule(**kwargs)
    second = MultiscaleHMMDataModule(**kwargs)
    first.setup()
    second.setup()
    for first_tensor, second_tensor in zip(
        first.val_dataset.tensors, second.val_dataset.tensors, strict=True
    ):
        np.testing.assert_array_equal(first_tensor.numpy(), second_tensor.numpy())


def test_empirical_autocorrelation_matches_requested_pole():
    timescale = 12.0
    epsilon = 0.15
    inputs, _, _ = generate_multiscale_hmm_sequences(
        num_trajectories=4096,
        num_time_steps=64,
        timescales=[timescale],
        observation_flip_probability=epsilon,
        rng=np.random.default_rng(4),
    )
    empirical_lag_one = float(np.mean(inputs[:, 1:, 0] * inputs[:, :-1, 0]))
    expected = (1.0 - 2.0 * epsilon) ** 2 * timescale_to_pole(timescale)
    assert empirical_lag_one == pytest.approx(expected, abs=0.015)


def test_bayes_reference_is_no_worse_than_linear_reference():
    for timescale in (4.0, 12.0, 32.0):
        _, linear_mse = optimal_linear_predictor(timescale, 0.15, 32)
        bayes_mse = estimate_bayes_mse(
            timescale,
            0.15,
            context_length=32,
            n_trajectories=2048,
        )
        assert 0.0 <= bayes_mse <= linear_mse + 0.01 <= 1.01


def test_uninformative_observations_have_unit_reference_loss():
    _, linear_mse = optimal_linear_predictor(16.0, 0.5, 64)
    bayes_mse = estimate_bayes_mse(
        16.0,
        0.5,
        context_length=16,
        n_trajectories=256,
    )
    assert linear_mse == pytest.approx(1.0)
    assert bayes_mse == pytest.approx(1.0)


def test_metadata_names_components_and_reference_levels():
    datamodule = MultiscaleHMMDataModule(
        n_components=3,
        timescales=[4.0, 8.0, 16.0],
        observation_flip_probability=0.15,
        num_time_steps=16,
        num_val_trajectories=8,
        batch_size=4,
        num_workers=0,
    )
    metadata = datamodule.analysis_metadata()
    assert metadata["component_names"] == ["tau_4", "tau_8", "tau_16"]
    assert metadata["component_groups"] == [[0], [1], [2]]
    assert len(metadata["linear_reference_losses"]) == 3
    assert len(metadata["bayes_reference_losses"]) == 3
