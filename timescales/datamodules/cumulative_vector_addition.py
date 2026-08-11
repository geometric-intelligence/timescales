"""Cumulative vector-addition task.

At every timestep the network receives a vector increment and must output the
running sum, independently in every channel:

    target[t] = input[0] + ... + input[t]

This is the simplest discrete integration task. Training trajectories are
generated online; validation trajectories are fixed for consistent metrics.
"""

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset


def generate_cumulative_vector_addition_trajectories(
    num_trajectories: int,
    num_time_steps: int,
    vector_size: int,
    increment_distribution: str = "gaussian",
    increment_std: float = 0.1,
    increment_probability: float = 1.0,
    rng=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate vector increments and their inclusive cumulative sums.

    Args:
        num_trajectories: Number of independent trajectories.
        num_time_steps: Number of increments in each trajectory.
        vector_size: Number of independently accumulated coordinates.
        increment_distribution: ``"gaussian"`` or ``"rademacher"``. The
            latter samples signed increments with values ``-increment_std`` and
            ``+increment_std`` and is useful for integer-style addition.
        increment_std: Standard deviation of each nonzero increment.
        increment_probability: Probability that an increment is present in each
            coordinate and timestep. ``1`` gives dense integration; values below
            one produce sparse addition events.
        rng: Optional NumPy random-state object. Defaults to ``np.random``.

    Returns:
        inputs: [B, T, D] vector increments.
        targets: [B, T, D] inclusive cumulative sums.
        states: [B, T, D] same as targets (auxiliary ground-truth state).
    """
    if num_trajectories < 0:
        raise ValueError("num_trajectories must be nonnegative")
    if num_time_steps < 0:
        raise ValueError("num_time_steps must be nonnegative")
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")
    if increment_std < 0.0:
        raise ValueError("increment_std must be nonnegative")
    if increment_distribution not in ("gaussian", "rademacher"):
        raise ValueError(
            "increment_distribution must be 'gaussian' or 'rademacher'"
        )
    if not 0.0 <= increment_probability <= 1.0:
        raise ValueError("increment_probability must be between 0 and 1")

    rng = np.random if rng is None else rng
    shape = (num_trajectories, num_time_steps, vector_size)
    if increment_distribution == "gaussian":
        inputs = rng.normal(loc=0.0, scale=increment_std, size=shape).astype(
            np.float32
        )
    else:
        signs = 2 * rng.randint(0, 2, size=shape) - 1
        inputs = (increment_std * signs).astype(np.float32)
    if increment_probability < 1.0:
        mask = rng.uniform(size=shape) < increment_probability
        inputs *= mask

    targets = np.cumsum(inputs, axis=1, dtype=np.float32)
    states = targets.copy()
    return inputs, targets, states


class CumulativeVectorAdditionOnlineDataset(IterableDataset):
    """Generate a fresh batch of integration trajectories every training step."""

    def __init__(
        self,
        vector_size: int,
        increment_distribution: str,
        increment_std: float,
        increment_probability: float,
        num_time_steps: int,
        batch_size: int,
    ) -> None:
        self.vector_size = vector_size
        self.increment_distribution = increment_distribution
        self.increment_std = increment_std
        self.increment_probability = increment_probability
        self.num_time_steps = num_time_steps
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            inputs, targets, states = (
                generate_cumulative_vector_addition_trajectories(
                    num_trajectories=self.batch_size,
                    num_time_steps=self.num_time_steps,
                    vector_size=self.vector_size,
                    increment_distribution=self.increment_distribution,
                    increment_std=self.increment_std,
                    increment_probability=self.increment_probability,
                )
            )
            yield (
                torch.from_numpy(inputs),
                torch.from_numpy(states),
                torch.from_numpy(targets),
            )


class CumulativeVectorAdditionDataModule(L.LightningDataModule):
    def __init__(
        self,
        vector_size: int = 2,
        increment_distribution: str = "gaussian",
        increment_std: float = 0.1,
        increment_probability: float = 1.0,
        num_time_steps: int = 100,
        num_val_trajectories: int = 500,
        batch_size: int = 200,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.vector_size = vector_size
        self.increment_distribution = increment_distribution
        self.increment_std = increment_std
        self.increment_probability = increment_probability
        self.num_time_steps = num_time_steps
        self.num_val_trajectories = num_val_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers

        expected_additions = num_time_steps * increment_probability
        final_std = increment_std * expected_additions**0.5
        print(
            f"Cumulative vector addition: D={vector_size}, "
            f"distribution={increment_distribution}, "
            f"increment_std={increment_std}, "
            f"increment_probability={increment_probability}, "
            f"T={num_time_steps} steps "
            f"(expected final std ~{final_std:.3g})"
        )

    def setup(self, stage=None) -> None:
        inputs, targets, states = (
            generate_cumulative_vector_addition_trajectories(
                num_trajectories=self.num_val_trajectories,
                num_time_steps=self.num_time_steps,
                vector_size=self.vector_size,
                increment_distribution=self.increment_distribution,
                increment_std=self.increment_std,
                increment_probability=self.increment_probability,
            )
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(inputs),
            torch.from_numpy(states),
            torch.from_numpy(targets),
        )
        self.train_dataset = CumulativeVectorAdditionOnlineDataset(
            vector_size=self.vector_size,
            increment_distribution=self.increment_distribution,
            increment_std=self.increment_std,
            increment_probability=self.increment_probability,
            num_time_steps=self.num_time_steps,
            batch_size=self.batch_size,
        )
        print(
            f"Online integration training: {self.batch_size} fresh "
            f"trajectories/step, {self.num_val_trajectories} fixed validation "
            "trajectories"
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
        return self.vector_size

    @property
    def output_size(self) -> int:
        return self.vector_size
