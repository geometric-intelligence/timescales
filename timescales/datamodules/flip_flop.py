"""
N-bit Flip-Flop DataModule.

A classic memory task with N independent 1-bit flip-flops
(Sussillo & Barak 2013):
- At each timestep, the network receives an N-dimensional input vector.
  For each bit i, input_i is +1 (set to 1), -1 (set to 0), or 0 (hold).
- Pulses arrive independently with probability p_pulse per bit per timestep.
  When a pulse fires for bit i, it is +1 or -1 with equal probability.
- Between pulses, inputs are zero.
- The network must output the current state of all N bits at every timestep.

This task requires the network to maintain N independent binary memories over
variable-length intervals, testing its ability to sustain stable fixed points.

Training data is generated online (fresh batch of trajectories every step).
Validation data is fixed for consistent metrics across training.
"""

import lightning as L
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import torch
import numpy as np


def simulate_flip_flop_trajectories(
    num_trajectories: int,
    num_time_steps: int,
    n_bits: int,
    p_pulse: float | list[float],
    pulse_amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate flip-flop trajectories.

    Args:
        p_pulse: Pulse probability per timestep. A scalar applies to all bits;
                 a list of length n_bits sets a different rate per bit.

    Returns:
        inputs:  [B, T, n_bits]  pulse inputs (+1 set, -1 reset, 0 hold)
        targets: [B, T, n_bits]  current state of each bit
        states:  [B, T, n_bits]  same as targets (aux_info)
    """
    B, T, N = num_trajectories, num_time_steps, n_bits
    amp = pulse_amplitude
    p_arr = np.broadcast_to(np.asarray(p_pulse, dtype=np.float32), (N,))

    inputs = np.zeros((B, T, N), dtype=np.float32)
    states = np.zeros((B, T, N), dtype=np.float32)

    current_state = np.zeros((B, N), dtype=np.float32)

    for t in range(T):
        pulse_mask = np.random.random((B, N)) < p_arr[np.newaxis, :]
        sign = (2 * np.random.randint(0, 2, size=(B, N)) - 1).astype(np.float32)

        inputs[:, t, :] = pulse_mask * sign * amp

        current_state[pulse_mask & (sign > 0)] = 1.0
        current_state[pulse_mask & (sign < 0)] = 0.0

        states[:, t, :] = current_state

    targets = states.copy()
    return inputs, targets, states


class FlipFlopOnlineDataset(IterableDataset):
    """Generates a fresh batch of flip-flop trajectories every training step."""

    def __init__(
        self,
        n_bits: int,
        p_pulse: float | list[float],
        pulse_amplitude: float,
        num_time_steps: int,
        batch_size: int,
    ):
        self.n_bits = n_bits
        self.p_pulse = p_pulse
        self.pulse_amplitude = pulse_amplitude
        self.num_time_steps = num_time_steps
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            inputs, targets, states = simulate_flip_flop_trajectories(
                num_trajectories=self.batch_size,
                num_time_steps=self.num_time_steps,
                n_bits=self.n_bits,
                p_pulse=self.p_pulse,
                pulse_amplitude=self.pulse_amplitude,
            )
            yield (
                torch.from_numpy(inputs),
                torch.from_numpy(states),
                torch.from_numpy(targets),
            )


class FlipFlopDataModule(L.LightningDataModule):
    def __init__(
        self,
        n_bits: int = 3,
        p_pulse: float | list[float] = 0.05,
        pulse_amplitude: float = 1.0,
        num_time_steps: int = 500,
        num_val_trajectories: int = 2000,
        batch_size: int = 200,
        num_workers: int = 4,
        # Legacy params (ignored, kept for config backward compat)
        num_trajectories: int | None = None,
        train_val_split: float | None = None,
    ) -> None:
        """
        :param n_bits: Number of independent flip-flop bits.
        :param p_pulse: Pulse probability per timestep. Scalar (all bits equal)
                        or list of length n_bits (per-bit rates).
        :param pulse_amplitude: Amplitude of the input pulse.
        :param num_time_steps: Sequence length per trajectory.
        :param num_val_trajectories: Number of fixed validation trajectories.
        :param batch_size: Training batch size (fresh trajectories generated per step).
        :param num_workers: Number of workers for validation data loading.
        """
        super().__init__()
        self.n_bits = n_bits
        self.p_pulse = p_pulse
        self.pulse_amplitude = pulse_amplitude
        self.num_time_steps = num_time_steps
        self.num_val_trajectories = num_val_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers

        if isinstance(p_pulse, (list, tuple)):
            intervals = [f"{1.0/max(p,1e-8):.0f}" for p in p_pulse]
            print(f"Flip-flop task: {n_bits} bits, p_pulse={p_pulse} "
                  f"(avg intervals ~{intervals} steps)")
        else:
            avg_interval = 1.0 / max(p_pulse, 1e-8)
            print(f"Flip-flop task: {n_bits} bits, p_pulse={p_pulse} "
                  f"(avg interval ~{avg_interval:.0f} steps)")

    def setup(self, stage=None) -> None:
        inputs, targets, states = simulate_flip_flop_trajectories(
            num_trajectories=self.num_val_trajectories,
            num_time_steps=self.num_time_steps,
            n_bits=self.n_bits,
            p_pulse=self.p_pulse,
            pulse_amplitude=self.pulse_amplitude,
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(inputs),
            torch.from_numpy(states),
            torch.from_numpy(targets),
        )

        self.train_dataset = FlipFlopOnlineDataset(
            n_bits=self.n_bits,
            p_pulse=self.p_pulse,
            pulse_amplitude=self.pulse_amplitude,
            num_time_steps=self.num_time_steps,
            batch_size=self.batch_size,
        )
        print(f"Online training: {self.batch_size} fresh trajectories/step, "
              f"{self.num_val_trajectories} fixed val trajectories")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )

    @property
    def input_size(self) -> int:
        return self.n_bits

    @property
    def output_size(self) -> int:
        return self.n_bits
