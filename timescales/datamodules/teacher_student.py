"""
Teacher-Student DataModule.

A teacher-student setup where a fixed random teacher RNN generates target
outputs from smoothed random inputs. The student RNN trains to reproduce
the teacher's input-output mapping via MSE loss.

The teacher is an RNN with:
- Fixed random connectivity (W_rec ~ N(0, 1/N))
- Configurable recurrent gain and intrinsic timescale
- Initialized with a dedicated seed independent of the training seed

Inputs are 2D signals drawn from Uniform(0, 1) and smoothed with a
Savitzky-Golay filter to create temporally correlated driving signals.

Dataset is pre-generated: 500 input-output sequence pairs (default),
split 400 train / 100 validation, each unfolding over 20 time steps.
"""

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, TensorDataset

from timescales.rnns.rnn import RNN


def generate_smooth_inputs(
    num_sequences: int,
    num_time_steps: int,
    input_dim: int,
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Generate smooth random input sequences.

    Draws from Uniform(0, 1) and smooths with a Savitzky-Golay filter.

    Returns:
        inputs: [B, T, input_dim] float32
    """
    if rng is None:
        rng = np.random.RandomState()

    raw = rng.uniform(0, 1, size=(num_sequences, num_time_steps, input_dim)).astype(
        np.float32
    )

    smoothed = np.empty_like(raw)
    for dim in range(input_dim):
        smoothed[:, :, dim] = savgol_filter(
            raw[:, :, dim], window_length=savgol_window, polyorder=savgol_polyorder, axis=1
        )

    smoothed = np.clip(smoothed, 0.0, 1.0)
    return smoothed


def create_teacher(
    input_size: int,
    hidden_size: int,
    output_size: int,
    dt: float,
    timescale: float,
    recurrent_gain: float,
    activation: str,
    wrec_init: str,
    teacher_seed: int,
) -> RNN:
    """
    Create a deterministic teacher RNN using a dedicated seed.

    Saves and restores the global RNG state so the caller's random
    state is not affected.
    """
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        [torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    np_state = np.random.get_state()

    torch.manual_seed(teacher_seed)
    np.random.seed(teacher_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(teacher_seed)

    teacher = RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dt=dt,
        time_constants_config={"type": "discrete", "values": [timescale]},
        activation=getattr(nn, activation),
        learn_time_constants=False,
        shared_time_constant=False,
        normalize_hidden=False,
        zero_diag_wrec=False,
        recurrent_gain=recurrent_gain,
        noise_std=0.0,
        wrec_init=wrec_init,
        alpha_parameterization="exponential",
        dynamics_type="rate",
    )

    # Restore global RNG state
    torch.random.set_rng_state(torch_state)
    np.random.set_state(np_state)
    if torch.cuda.is_available():
        for d, state in enumerate(cuda_states):
            torch.cuda.set_rng_state(state, d)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    return teacher


class TeacherStudentDataModule(L.LightningDataModule):
    def __init__(
        self,
        teacher_hidden_size: int = 10,
        teacher_recurrent_gain: float = 0.99,
        teacher_timescale: float = 1.0,
        teacher_activation: str = "Tanh",
        teacher_wrec_init: str = "normal_scaled",
        teacher_seed: int = 42,
        input_dim: int = 2,
        output_dim: int = 2,
        num_time_steps: int = 20,
        dt: float = 1.0,
        num_sequences: int = 500,
        train_val_split: float = 0.8,
        savgol_window: int = 5,
        savgol_polyorder: int = 2,
        batch_size: int = 100,
        num_workers: int = 4,
    ) -> None:
        """
        :param teacher_hidden_size: Number of hidden units in the teacher.
        :param teacher_recurrent_gain: Recurrent gain g of the teacher.
        :param teacher_timescale: Intrinsic timescale tau of the teacher.
        :param teacher_activation: Activation function name for the teacher.
        :param teacher_wrec_init: W_rec initialization for the teacher.
        :param teacher_seed: Fixed seed for teacher RNN initialization.
        :param input_dim: Dimension of the input signal.
        :param output_dim: Dimension of the output signal.
        :param num_time_steps: Length of each sequence.
        :param dt: Discrete time step.
        :param num_sequences: Total number of input-output pairs.
        :param train_val_split: Fraction of sequences for training.
        :param savgol_window: Savitzky-Golay filter window length (must be odd).
        :param savgol_polyorder: Savitzky-Golay polynomial order.
        :param batch_size: Batch size for data loaders.
        :param num_workers: Number of data loader workers.
        """
        super().__init__()
        self.teacher_hidden_size = teacher_hidden_size
        self.teacher_recurrent_gain = teacher_recurrent_gain
        self.teacher_timescale = teacher_timescale
        self.teacher_activation = teacher_activation
        self.teacher_wrec_init = teacher_wrec_init
        self.teacher_seed = teacher_seed
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_time_steps = num_time_steps
        self.dt = dt
        self.num_sequences = num_sequences
        self.train_val_split = train_val_split
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.teacher = None

        n_train = int(num_sequences * train_val_split)
        n_val = num_sequences - n_train
        print(
            f"Teacher-student task: teacher N={teacher_hidden_size}, "
            f"g={teacher_recurrent_gain}, tau={teacher_timescale}, "
            f"activation={teacher_activation}"
        )
        print(
            f"Data: {num_sequences} sequences ({n_train} train, {n_val} val), "
            f"T={num_time_steps}, input_dim={input_dim}, output_dim={output_dim}"
        )

    def setup(self, stage=None) -> None:
        # Create teacher with dedicated seed
        self.teacher = create_teacher(
            input_size=self.input_dim,
            hidden_size=self.teacher_hidden_size,
            output_size=self.output_dim,
            dt=self.dt,
            timescale=self.teacher_timescale,
            recurrent_gain=self.teacher_recurrent_gain,
            activation=self.teacher_activation,
            wrec_init=self.teacher_wrec_init,
            teacher_seed=self.teacher_seed,
        )

        # Generate inputs with a dedicated data seed (teacher_seed + 1)
        data_rng = np.random.RandomState(self.teacher_seed + 1)
        inputs_np = generate_smooth_inputs(
            num_sequences=self.num_sequences,
            num_time_steps=self.num_time_steps,
            input_dim=self.input_dim,
            savgol_window=self.savgol_window,
            savgol_polyorder=self.savgol_polyorder,
            rng=data_rng,
        )

        inputs_t = torch.from_numpy(inputs_np)

        # Run teacher to generate targets
        with torch.no_grad():
            _, targets_t = self.teacher(inputs_t, init_context=None)

        # Split into train / val
        n_train = int(self.num_sequences * self.train_val_split)

        self.train_dataset = TensorDataset(
            inputs_t[:n_train],
            torch.zeros(n_train, 1),  # aux_info placeholder
            targets_t[:n_train],
        )
        self.val_dataset = TensorDataset(
            inputs_t[n_train:],
            torch.zeros(self.num_sequences - n_train, 1),
            targets_t[n_train:],
        )

        print(
            f"Teacher-student data generated: "
            f"{n_train} train, {self.num_sequences - n_train} val sequences"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    @property
    def input_size(self) -> int:
        return self.input_dim

    @property
    def output_size(self) -> int:
        return self.output_dim
