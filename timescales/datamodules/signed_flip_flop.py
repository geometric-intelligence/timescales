"""
Signed N-bit flip-flop DataModule.

This is the demo-style variant of the flip-flop memory task:
- pulse inputs are signed (-1, 0, +1)
- persistent memory targets are signed (-1, +1)
- the initial memory state is explicitly randomized at t=0

The existing FlipFlopDataModule keeps the repository's binary 0/1 target
convention; this module is intentionally separate to avoid changing that
behavior for older experiments.
"""

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset


def simulate_signed_flip_flop_trajectories(
    num_trajectories: int,
    num_time_steps: int,
    n_bits: int,
    p_pulse: float | list[float],
    pulse_amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate signed flip-flop trajectories.

    Args:
        p_pulse: Pulse probability per timestep. A scalar applies to all bits;
                 a list of length n_bits sets a different rate per bit.

    Returns:
        inputs:  [B, T, n_bits] pulse inputs (-amp, 0, +amp)
        targets: [B, T, n_bits] signed memory state (-1, +1)
        states:  [B, T, n_bits] same as targets (aux_info)
    """
    B, T, N = num_trajectories, num_time_steps, n_bits
    p_arr = np.broadcast_to(np.asarray(p_pulse, dtype=np.float32), (N,))

    inputs = np.zeros((B, T, N), dtype=np.float32)
    states = np.zeros((B, T, N), dtype=np.float32)

    current_state = (2 * np.random.randint(0, 2, size=(B, N)) - 1).astype(np.float32)
    inputs[:, 0, :] = current_state * pulse_amplitude
    states[:, 0, :] = current_state

    for t in range(1, T):
        pulse_mask = np.random.random((B, N)) < p_arr[np.newaxis, :]
        sign = (2 * np.random.randint(0, 2, size=(B, N)) - 1).astype(np.float32)

        inputs[:, t, :] = pulse_mask * sign * pulse_amplitude
        current_state[pulse_mask] = sign[pulse_mask]
        states[:, t, :] = current_state

    targets = states.copy()
    return inputs, targets, states


class SignedFlipFlopOnlineDataset(IterableDataset):
    """Generates a fresh batch of signed flip-flop trajectories every step."""

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
            inputs, targets, states = simulate_signed_flip_flop_trajectories(
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


class SignedFlipFlopDataModule(L.LightningDataModule):
    def __init__(
        self,
        n_bits: int = 2,
        p_pulse: float | list[float] = 0.05,
        pulse_amplitude: float = 1.0,
        num_time_steps: int = 100,
        num_val_trajectories: int = 100,
        batch_size: int = 100,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.p_pulse = p_pulse
        self.pulse_amplitude = pulse_amplitude
        self.num_time_steps = num_time_steps
        self.num_val_trajectories = num_val_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers

        if isinstance(p_pulse, (list, tuple)):
            intervals = [f"{1.0 / max(p, 1e-8):.0f}" for p in p_pulse]
            print(
                f"Signed flip-flop task: {n_bits} bits, p_pulse={p_pulse} "
                f"(avg intervals ~{intervals} steps)"
            )
        else:
            avg_interval = 1.0 / max(p_pulse, 1e-8)
            print(
                f"Signed flip-flop task: {n_bits} bits, p_pulse={p_pulse} "
                f"(avg interval ~{avg_interval:.0f} steps)"
            )

    def setup(self, stage=None) -> None:
        inputs, targets, states = simulate_signed_flip_flop_trajectories(
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

        self.train_dataset = SignedFlipFlopOnlineDataset(
            n_bits=self.n_bits,
            p_pulse=self.p_pulse,
            pulse_amplitude=self.pulse_amplitude,
            num_time_steps=self.num_time_steps,
            batch_size=self.batch_size,
        )
        print(
            f"Online signed training: {self.batch_size} fresh trajectories/step, "
            f"{self.num_val_trajectories} fixed val trajectories"
        )

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
