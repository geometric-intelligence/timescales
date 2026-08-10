import numpy as np

from timescales.datamodules.flip_flop import simulate_flip_flop_trajectories


def test_forced_initial_pulse_sets_each_bit_and_persists_without_later_pulses():
    inputs, targets, states = simulate_flip_flop_trajectories(
        num_trajectories=12,
        num_time_steps=8,
        n_bits=3,
        p_pulse=0.0,
        pulse_amplitude=2.0,
        force_initial_pulse=True,
    )

    np.testing.assert_array_equal(np.abs(inputs[:, 0, :]), 2.0)
    np.testing.assert_array_equal(inputs[:, 1:, :], 0.0)
    expected_initial_state = (inputs[:, 0, :] > 0).astype(np.float32)
    np.testing.assert_array_equal(targets[:, 0, :], expected_initial_state)
    np.testing.assert_array_equal(
        targets,
        np.repeat(expected_initial_state[:, None, :], repeats=8, axis=1),
    )
    np.testing.assert_array_equal(states, targets)


def test_initial_state_remains_zero_by_default_when_no_pulses_arrive():
    inputs, targets, states = simulate_flip_flop_trajectories(
        num_trajectories=4,
        num_time_steps=5,
        n_bits=2,
        p_pulse=0.0,
    )

    np.testing.assert_array_equal(inputs, 0.0)
    np.testing.assert_array_equal(targets, 0.0)
    np.testing.assert_array_equal(states, 0.0)


def test_forced_initial_pulse_handles_empty_trajectories():
    inputs, targets, states = simulate_flip_flop_trajectories(
        num_trajectories=3,
        num_time_steps=0,
        n_bits=2,
        p_pulse=[0.1, 0.2],
        force_initial_pulse=True,
    )

    assert inputs.shape == (3, 0, 2)
    assert targets.shape == (3, 0, 2)
    assert states.shape == (3, 0, 2)
