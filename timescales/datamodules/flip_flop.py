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
"""

import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import numpy as np


class FlipFlopDataModule(L.LightningDataModule):
    def __init__(
        self,
        n_bits: int = 3,
        p_pulse: float = 0.05,
        pulse_amplitude: float = 1.0,
        num_time_steps: int = 500,
        num_trajectories: int = 10000,
        batch_size: int = 200,
        num_workers: int = 4,
        train_val_split: float = 0.8,
    ) -> None:
        """
        :param n_bits: Number of independent flip-flop bits.
        :param p_pulse: Probability that a pulse arrives for any given bit at each step.
        :param pulse_amplitude: Amplitude of the input pulse.
        :param num_time_steps: Sequence length per trajectory.
        :param num_trajectories: Total number of trajectories.
        :param batch_size: Batch size for dataloaders.
        :param num_workers: Number of workers for data loading.
        :param train_val_split: Fraction of data used for training.
        """
        super().__init__()
        self.n_bits = n_bits
        self.p_pulse = p_pulse
        self.pulse_amplitude = pulse_amplitude
        self.num_time_steps = num_time_steps
        self.num_trajectories = num_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        avg_interval = 1.0 / max(p_pulse, 1e-8)
        print(f"Flip-flop task: {n_bits} bits, p_pulse={p_pulse} "
              f"(avg interval ~{avg_interval:.0f} steps)")

    def simulate_trajectories(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate flip-flop trajectories.

        Returns:
            inputs:  [B, T, n_bits]  pulse inputs (+1 set, -1 reset, 0 hold)
            targets: [B, T, n_bits]  current state of each bit
            states:  [B, T, n_bits]  same as targets (aux_info)
        """
        B, T, N = self.num_trajectories, self.num_time_steps, self.n_bits
        amp = self.pulse_amplitude

        inputs = np.zeros((B, T, N), dtype=np.float32)
        states = np.zeros((B, T, N), dtype=np.float32)

        current_state = np.random.randint(0, 2, size=(B, N)).astype(np.float32)

        for t in range(T):
            pulse_mask = np.random.random((B, N)) < self.p_pulse

            # +1 or -1 with equal probability
            sign = (2 * np.random.randint(0, 2, size=(B, N)) - 1).astype(np.float32)

            inputs[:, t, :] = pulse_mask * sign * amp

            # Update state: +1 pulse -> bit=1, -1 pulse -> bit=0
            current_state[pulse_mask & (sign > 0)] = 1.0
            current_state[pulse_mask & (sign < 0)] = 0.0

            states[:, t, :] = current_state

        targets = states.copy()
        return inputs, targets, states

    def setup(self, stage=None) -> None:
        inputs, targets, states = self.simulate_trajectories()

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        states_tensor = torch.tensor(states, dtype=torch.float32)

        full_dataset = TensorDataset(inputs_tensor, states_tensor, targets_tensor)

        train_size = int(self.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        print(f"Dataset created: {train_size} train, {val_size} val trajectories")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
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
