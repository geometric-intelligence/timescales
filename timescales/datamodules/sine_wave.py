"""
Sine-Wave Generation DataModule.

Autonomous oscillation task: the network must generate a multi-frequency
sinusoidal target with no external input, starting from a uniform initial
hidden state r_0 = init_value for all neurons.

The target consists of N (cos, sin) pairs, each oscillating at its own period:
    y(t) = (cos(2π t/T_1), sin(2π t/T_1), ..., cos(2π t/T_N), sin(2π t/T_N))

When N = 1, this reduces to the classic single-frequency task with period T.

The heterogeneous variant (N > 1 with distinct periods) is the oscillatory
analogue of the heterogeneous flip-flop: it probes whether the network can
sustain limit cycles at multiple timescales simultaneously.
"""

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset


def _normalize_periods(periods: float | list[float], n_pairs: int) -> list[float]:
    """Expand a scalar period to a list, or validate a list matches n_pairs."""
    if isinstance(periods, (int, float)):
        return [float(periods)] * n_pairs
    periods = [float(p) for p in periods]
    if len(periods) != n_pairs:
        raise ValueError(
            f"len(periods)={len(periods)} does not match n_pairs={n_pairs}"
        )
    return periods


def generate_sine_wave_targets(
    num_trajectories: int,
    num_time_steps: int,
    periods: list[float],
    dt: float = 1.0,
    random_phase: bool = False,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate multi-frequency sine-wave targets.

    Args:
        num_trajectories: batch size.
        num_time_steps: sequence length.
        periods: list of oscillation periods (one per cos/sin pair).
        dt: integration timestep.
        random_phase: if True, each trajectory gets a uniform random
            phase offset per pair in [0, 2π).
        rng: random state for reproducibility (used only when random_phase=True).

    Returns:
        inputs:  [B, T, 1]             all-zero dummy inputs.
        targets: [B, T, 2 * n_pairs]   interleaved (cos_i, sin_i) per pair.
    """
    if rng is None:
        rng = np.random.RandomState()

    B, T = num_trajectories, num_time_steps
    n_pairs = len(periods)
    t = np.arange(T) * dt

    channels = []
    for k, period in enumerate(periods):
        phase = 2.0 * np.pi * t / period

        if random_phase:
            offset = rng.uniform(0, 2.0 * np.pi, size=(B, 1))
        else:
            offset = np.zeros((B, 1))

        phase_shifted = phase[np.newaxis, :] + offset
        channels.append(np.cos(phase_shifted))
        channels.append(np.sin(phase_shifted))

    targets = np.stack(channels, axis=-1).astype(np.float32)
    inputs = np.zeros((B, T, 1), dtype=np.float32)

    return inputs, targets


class SineWaveOnlineDataset(IterableDataset):
    """Yields sine-wave batches; supports optional random-phase augmentation."""

    def __init__(
        self,
        num_time_steps: int,
        batch_size: int,
        periods: list[float],
        dt: float = 1.0,
        random_phase: bool = False,
    ):
        self.num_time_steps = num_time_steps
        self.batch_size = batch_size
        self.periods = periods
        self.dt = dt
        self.random_phase = random_phase

    def __iter__(self):
        rng = np.random.RandomState()
        while True:
            inputs, targets = generate_sine_wave_targets(
                num_trajectories=self.batch_size,
                num_time_steps=self.num_time_steps,
                periods=self.periods,
                dt=self.dt,
                random_phase=self.random_phase,
                rng=rng,
            )
            aux = np.zeros_like(inputs)
            yield (
                torch.from_numpy(inputs),
                torch.from_numpy(aux),
                torch.from_numpy(targets),
            )


class SineWaveDataModule(L.LightningDataModule):
    def __init__(
        self,
        n_pairs: int = 1,
        periods: float | list[float] = 10.0,
        dt: float = 1.0,
        num_time_steps: int = 200,
        init_hidden_value: float = 1.0,
        random_phase: bool = False,
        num_val_trajectories: int = 200,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        """
        :param n_pairs: number of (cos, sin) output pairs.
        :param periods: oscillation period(s). A scalar applies to all pairs;
            a list of length n_pairs sets a different period per pair.
        :param dt: integration timestep (should match the RNN's dt).
        :param num_time_steps: sequence length per trajectory.
        :param init_hidden_value: initial hidden-state value for all neurons.
        :param random_phase: if True, add a random phase offset per trajectory.
        :param num_val_trajectories: fixed validation set size.
        :param batch_size: training batch size.
        :param num_workers: DataLoader workers for validation.
        """
        super().__init__()
        self.n_pairs = n_pairs
        self.periods = _normalize_periods(periods, n_pairs)
        self.dt = dt
        self.num_time_steps = num_time_steps
        self.init_hidden_value = init_hidden_value
        self.random_phase = random_phase
        self.num_val_trajectories = num_val_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers

        n_cycles = [num_time_steps * dt / p for p in self.periods]
        cycles_str = ", ".join(f"{c:.1f}" for c in n_cycles)
        print(f"Sine-wave task: {n_pairs} pair(s), "
              f"periods={self.periods}, dt={dt}, "
              f"T={num_time_steps} steps (cycles: [{cycles_str}]), "
              f"init_hidden={init_hidden_value}, "
              f"random_phase={random_phase}")

    def setup(self, stage=None) -> None:
        val_rng = np.random.RandomState(0)
        inputs, targets = generate_sine_wave_targets(
            num_trajectories=self.num_val_trajectories,
            num_time_steps=self.num_time_steps,
            periods=self.periods,
            dt=self.dt,
            random_phase=self.random_phase,
            rng=val_rng,
        )
        aux = np.zeros_like(inputs)
        self.val_dataset = TensorDataset(
            torch.from_numpy(inputs),
            torch.from_numpy(aux),
            torch.from_numpy(targets),
        )

        self.train_dataset = SineWaveOnlineDataset(
            num_time_steps=self.num_time_steps,
            batch_size=self.batch_size,
            periods=self.periods,
            dt=self.dt,
            random_phase=self.random_phase,
        )
        print(f"Sine-wave: {self.batch_size} trajectories/step, "
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
        return 1

    @property
    def output_size(self) -> int:
        return 2 * self.n_pairs
