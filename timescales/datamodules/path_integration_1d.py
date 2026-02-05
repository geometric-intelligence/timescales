"""
1D Path Integration DataModule.

Simplified version of path integration for theoretical validation:
- 1D position (scalar x)
- 1D velocity input (scalar v, can be positive or negative)
- 1D place cells (Gaussians on a line)
"""

import lightning as L
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split


class PathIntegration1DDataModule(L.LightningDataModule):
    def __init__(
        self,
        dt: float,
        num_time_steps: int,
        arena_size: float,
        # Place cell parameters
        num_place_cells: int,
        place_cell_rf: float,
        DoG: bool = False,
        surround_scale: float = 2.0,
        place_cell_layout: str = "uniform",  # "random" | "uniform"
        # Velocity parameters (signed velocity with OU dynamics)
        velocity_mean: float = 0.0,  # Mean velocity (m/s), typically 0 for unbiased
        velocity_std: float = 0.5,   # Std of velocity (m/s)
        velocity_tau: float = 1.0,   # OU autocorrelation time (s)
        # DataLoader parameters
        num_trajectories: int = 10000,
        batch_size: int = 200,
        num_workers: int = 4,
        train_val_split: float = 0.8,
    ) -> None:
        """
        Initialize the 1D PathIntegrationDataModule.

        This is a simplified 1D version for theoretical validation experiments.
        The agent moves along a 1D line with reflective boundaries.

        :param dt: Simulation time step size (s)
        :param num_time_steps: Number of time steps to simulate
        :param arena_size: Size of the arena (m), agent moves in [-arena_size/2, arena_size/2]
        :param num_place_cells: Number of place cells
        :param place_cell_rf: Place cell receptive field radius (m)
        :param DoG: Whether to use Difference-of-Gaussians place cell activation
        :param surround_scale: Surround scale for DoG place cell activation
        :param place_cell_layout: Place cell layout ("random" | "uniform")
        :param velocity_mean: Mean velocity (m/s), use 0 for symmetric random walk
        :param velocity_std: Standard deviation of velocity for OU process (m/s)
        :param velocity_tau: Autocorrelation time for velocity OU process (s)
        :param num_trajectories: Number of trajectories to simulate
        :param batch_size: Batch size
        :param num_workers: Number of workers for data loading
        :param train_val_split: Train/val split ratio
        """
        super().__init__()

        self.dt = dt
        self.num_time_steps = num_time_steps
        self.arena_size = arena_size

        # Place cell parameters
        self.num_place_cells = num_place_cells
        self.place_cell_rf = place_cell_rf
        self.DoG = DoG
        self.surround_scale = surround_scale
        self.place_cell_layout = place_cell_layout

        # Velocity parameters
        self.velocity_mean = velocity_mean
        self.velocity_std = velocity_std
        self.velocity_tau = velocity_tau

        # DataLoader parameters
        self.num_trajectories = num_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        # Initialize place cell centers (1D)
        if place_cell_layout == "random":
            centers = np.random.uniform(
                -arena_size / 2, arena_size / 2, (num_place_cells,)
            )
        elif place_cell_layout == "uniform":
            centers = np.linspace(
                -arena_size / 2, arena_size / 2, num_place_cells
            )
        else:
            raise ValueError(f"Unknown place_cell_layout: {place_cell_layout}")

        self.place_cell_centers = torch.tensor(centers, dtype=torch.float32)
        self.softmax = torch.nn.Softmax(dim=-1)

    def _get_place_cell_activations(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute place cell activations for given 1D positions.

        :param pos: Positions of shape [batch_size, num_time_steps, 1]
        :return: Place cell activations [batch_size, num_time_steps, num_place_cells]
        """
        # Move centers to same device as pos
        centers = self.place_cell_centers.to(pos.device)

        # Compute distances: pos is [B, T, 1], centers is [Np]
        # Squeeze pos to [B, T] for broadcasting
        pos_squeezed = pos.squeeze(-1)  # [B, T]
        d = pos_squeezed[:, :, None] - centers[None, None, :]  # [B, T, Np]

        # Compute squared distance
        norm2 = d**2  # [B, T, Np]

        # Compute place cell activations with softmax normalization
        outputs = self.softmax(-norm2 / (2 * self.place_cell_rf**2))

        if self.DoG:
            # Subtract surround (larger width)
            surround = self.softmax(
                -norm2 / (2 * self.surround_scale * self.place_cell_rf**2)
            )
            outputs = outputs - surround

            # Shift and scale to [0,1]
            min_output, _ = outputs.min(-1, keepdim=True)
            outputs = outputs + torch.abs(min_output)
            outputs = outputs / outputs.sum(-1, keepdim=True)

        return outputs

    def _simulate_1d_ou(self, device: str = "cpu"):
        """
        Generate (B, T) 1D trajectories using OU dynamics on velocity.
        
        Returns:
            inputs: [B, T, 1] - velocity at each timestep
            positions: [B, T, 1] - position at each timestep
            place_cell_activations: [B, T, Np] - place cell activations
        """
        B, T, dt = self.num_trajectories, self.num_time_steps, self.dt
        R = self.arena_size / 2.0

        # OU parameters
        mu_v = float(self.velocity_mean)
        s_v = float(self.velocity_std)
        tau_v = float(self.velocity_tau)

        # OU diffusion scale: Var = (sigma^2 * tau) / 2
        sig_v = np.float32(np.sqrt(2.0 * (s_v**2) / max(tau_v, 1e-6)))
        sqrt_dt = np.float32(np.sqrt(dt))

        # Initialize state
        pos = np.random.uniform(-R, R, size=(B,)).astype(np.float32)
        v = np.random.normal(mu_v, s_v, size=(B,)).astype(np.float32)

        pos_list = [pos.copy()]
        v_list = [v.copy()]

        for _ in range(T - 1):
            # OU update for velocity
            v += (-(v - mu_v) / tau_v) * dt + sig_v * sqrt_dt * np.random.randn(B).astype(np.float32)

            # Kinematics: x(t+1) = x(t) + v(t) * dt
            pos = pos + v * dt

            # Reflective boundaries (handle multiple reflections for large steps)
            for _ in range(2):
                over = pos > R
                under = pos < -R
                if np.any(over):
                    pos[over] = 2 * R - pos[over]
                    v[over] = -v[over]  # Reverse velocity on reflection
                if np.any(under):
                    pos[under] = -2 * R - pos[under]
                    v[under] = -v[under]  # Reverse velocity on reflection

            pos_list.append(pos.copy())
            v_list.append(v.copy())

        pos_all = np.stack(pos_list, axis=1).astype(np.float32)  # (B, T)
        v_all = np.stack(v_list, axis=1).astype(np.float32)  # (B, T)

        # Reshape for consistency: [B, T, 1]
        inputs = v_all[:, :, None]  # velocity as input
        positions = pos_all[:, :, None]

        # Compute place cell activations
        pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
        pc = self._get_place_cell_activations(pos_tensor).cpu().numpy()

        return inputs, positions, pc

    def simulate_trajectories(
        self, device: str = "cpu"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates 1D trajectories.

        Returns:
            inputs: numpy array of shape (batch, T, 1) - velocity
            positions: numpy array of shape (batch, T, 1)
            place_cell_activations: numpy array of shape (batch, T, num_place_cells)
        """
        return self._simulate_1d_ou(device)

    def setup(self, stage=None) -> None:
        # Get numpy arrays from trajectory generation
        inputs_np, positions_np, place_cells_np = self.simulate_trajectories(
            device="cpu"
        )

        # Convert to tensors
        inputs = torch.tensor(inputs_np, dtype=torch.float32)
        positions = torch.tensor(positions_np, dtype=torch.float32)
        place_cell_activations = torch.tensor(place_cells_np, dtype=torch.float32)

        full_dataset = TensorDataset(inputs, positions, place_cell_activations)

        # Split into train and val
        train_size = int(self.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )
