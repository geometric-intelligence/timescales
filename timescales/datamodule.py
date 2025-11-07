import lightning as L
from torch.utils.data import DataLoader
import torch
import math
import numpy as np
from torch.utils.data import TensorDataset, random_split


class PathIntegrationDataModule(L.LightningDataModule):
    def __init__(
        self,
        num_trajectories: int,
        batch_size: int,
        num_workers: int,
        train_val_split: float,
        velocity_representation: str,
        dt: float,  # Use dt directly instead of trajectory_duration
        num_time_steps: int,
        arena_size: float,
        speed_scale: float,
        sigma_speed: float,
        tau_vel: float,
        sigma_rotation: float,
        border_region: float,
        # Place cell parameters
        num_place_cells: int,
        place_cell_rf: float,
        surround_scale: float,
        DoG: bool,
        # Trajectory generation
        trajectory_type: str = "ornstein_uhlenbeck",
        place_cell_layout: str = "random",
    ) -> None:
        super().__init__()
        self.num_trajectories = num_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.velocity_representation = velocity_representation
        self.dt = dt  # Use dt directly
        self.num_time_steps = num_time_steps

        self.arena_size = arena_size
        self.speed_scale = speed_scale  # typical speed (m/sec)
        self.sigma_speed = sigma_speed
        self.tau_vel = tau_vel
        self.sigma_rotation = sigma_rotation  # stdev rotation velocity (rads/sec)
        self.border_region = border_region  # meters

        # Trajectory generation type
        self.trajectory_type = trajectory_type

        # Place cell parameters
        self.num_place_cells = num_place_cells
        self.place_cell_rf = place_cell_rf
        self.surround_scale = surround_scale
        self.DoG = DoG

        self.place_cell_layout = place_cell_layout

        # Initialize place cell centers based on layout
        if place_cell_layout == "random":
            centers_x = np.random.uniform(
                -arena_size / 2, arena_size / 2, (num_place_cells,)
            )
            centers_y = np.random.uniform(
                -arena_size / 2, arena_size / 2, (num_place_cells,)
            )
        elif place_cell_layout == "uniform":
            centers_x, centers_y = self._create_uniform_place_cells(
                num_place_cells, arena_size
            )
        else:
            raise ValueError(f"Unknown place_cell_layout: {place_cell_layout}")

        self.place_cell_centers = torch.tensor(
            np.vstack([centers_x, centers_y]).T, dtype=torch.float32
        )

        self.softmax = torch.nn.Softmax(dim=-1)

    def _create_uniform_place_cells(self, num_place_cells: int, arena_size: float):
        """Create uniformly spaced place cell centers on a grid."""
        # Find grid dimensions that fit num_place_cells
        grid_size = int(np.ceil(np.sqrt(num_place_cells)))

        # Create uniform grid
        x_coords = np.linspace(-arena_size / 2, arena_size / 2, grid_size)
        y_coords = np.linspace(-arena_size / 2, arena_size / 2, grid_size)

        # Create meshgrid and flatten
        xx, yy = np.meshgrid(x_coords, y_coords)
        centers_x = xx.flatten()
        centers_y = yy.flatten()

        # Take only the first num_place_cells (in case grid_size^2 > num_place_cells)
        centers_x = centers_x[:num_place_cells]
        centers_y = centers_y[:num_place_cells]

        print(
            f"Created {len(centers_x)} uniform place cells in {grid_size}x{grid_size} grid"
        )

        return centers_x, centers_y

    def get_place_cell_activations(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute place cell activations for given positions.

        :param pos: Positions of shape [batch_size, num_time_steps, 2]
        :return: Place cell activations [batch_size, num_time_steps, num_place_cells]
        """
        # Move centers to same device as pos
        centers = self.place_cell_centers.to(pos.device)

        # Compute distances: pos is [B, T, 2], centers is [Np, 2]
        d = torch.abs(pos[:, :, None, :] - centers[None, None, ...]).float()

        # Compute squared distance
        norm2 = (d**2).sum(-1)  # [B, T, Np]

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

    def _simulate_ornstein_uhlenbeck_trajectories(
        self,
        device: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates trajectories using Ornstein-Uhlenbeck process.
        """
        # Use numpy from the start
        pos = (
            np.random.uniform(0, 1, (self.num_trajectories, 2)) - 0.5
        ) * self.arena_size

        # Sample initial heading and speed
        hd0 = np.random.uniform(0, 2 * np.pi, self.num_trajectories)
        spd0 = np.clip(
            np.random.normal(self.speed_scale, self.sigma_speed, self.num_trajectories),
            a_min=0.0,
            a_max=None,
        )
        vel = np.stack([np.cos(hd0), np.sin(hd0)], axis=-1) * spd0[:, None]

        pos_list, vel_list = [pos.copy()], [vel.copy()]

        sqrt_2dt_over_tau = math.sqrt(2 * self.dt / self.tau_vel)

        for _ in range(self.num_time_steps - 1):
            # OU velocity update (momentum) - all in numpy
            noise = np.random.randn(*vel.shape)
            vel = (
                vel
                + (self.dt / self.tau_vel) * (-vel)
                + self.sigma_speed * sqrt_2dt_over_tau * noise
            )

            # position update
            pos = pos + vel * self.dt

            # Reflective boundaries - numpy version
            out_left = pos[:, 0] < -self.arena_size / 2
            out_right = pos[:, 0] > self.arena_size / 2
            out_bottom = pos[:, 1] < -self.arena_size / 2
            out_top = pos[:, 1] > self.arena_size / 2

            # Reflect positions and flip velocity components
            if np.any(out_left):
                pos[out_left, 0] = -self.arena_size - pos[out_left, 0]
                vel[out_left, 0] *= -1
            if np.any(out_right):
                pos[out_right, 0] = self.arena_size - pos[out_right, 0]
                vel[out_right, 0] *= -1
            if np.any(out_bottom):
                pos[out_bottom, 1] = -self.arena_size - pos[out_bottom, 1]
                vel[out_bottom, 1] *= -1
            if np.any(out_top):
                pos[out_top, 1] = self.arena_size - pos[out_top, 1]
                vel[out_top, 1] *= -1

            pos_list.append(pos.copy())
            vel_list.append(vel.copy())

        # Convert to numpy arrays
        vel_all = np.stack(vel_list, axis=1)  # (batch, T, 2)
        pos_all = np.stack(pos_list, axis=1)  # (batch, T, 2)

        # Choose velocity representation
        if self.velocity_representation == "cartesian":
            inputs = vel_all
        elif self.velocity_representation == "polar":
            speeds = np.linalg.norm(vel_all, axis=-1)
            headings = np.arctan2(vel_all[..., 1], vel_all[..., 0]) % (2 * np.pi)
            inputs = np.stack([headings, speeds], axis=-1)  # (batch, T, 2)
        else:
            raise ValueError(
                f"Invalid velocity representation: {self.velocity_representation}"
            )

        # Only convert to torch temporarily for place cell computation
        pos_tensor = torch.tensor(pos_all, dtype=torch.float32, device=device)
        place_cell_activations = self.get_place_cell_activations(pos_tensor)
        place_cell_activations_np = place_cell_activations.cpu().numpy()

        return inputs, pos_all, place_cell_activations_np

    def _simulate_random_walk_trajectories(self, device: str = "cpu"):
        """
        Simulates trajectories using random walk with wall avoidance (from original grid cell paper).

        Returns numpy arrays instead of tensors for efficiency.
        """
        # Use configurable time parameters
        dt = self.dt

        # Initialize variables
        batch_size = self.num_trajectories
        position = np.zeros([batch_size, self.num_time_steps + 2, 2])
        head_dir = np.zeros([batch_size, self.num_time_steps + 2])
        position[:, 0, 0] = np.random.uniform(
            -self.arena_size / 2, self.arena_size / 2, batch_size
        )
        position[:, 0, 1] = np.random.uniform(
            -self.arena_size / 2, self.arena_size / 2, batch_size
        )
        head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size)
        velocity = np.zeros([batch_size, self.num_time_steps + 2])

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(
            0, self.sigma_rotation, [batch_size, self.num_time_steps + 1]
        )
        random_vel = np.random.rayleigh(
            self.speed_scale, [batch_size, self.num_time_steps + 1]
        )
        v = np.abs(np.random.normal(0, self.speed_scale * np.pi / 2, batch_size))

        def avoid_wall(position, hd, box_width, box_height):
            """Wall avoidance from old code"""
            x = position[:, 0]
            y = position[:, 1]
            dists = [
                box_width / 2 - x,
                box_height / 2 - y,
                box_width / 2 + x,
                box_height / 2 + y,
            ]
            d_wall = np.min(dists, axis=0)
            angles = np.arange(4) * np.pi / 2
            theta = angles[np.argmin(dists, axis=0)]
            hd = np.mod(hd, 2 * np.pi)
            a_wall = hd - theta
            a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

            is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
            turn_angle = np.zeros_like(hd)
            turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (
                np.pi / 2 - np.abs(a_wall[is_near_wall])
            )

            return is_near_wall, turn_angle

        for t in range(self.num_time_steps + 1):
            # Update velocity
            v = random_vel[:, t]
            turn_angle = np.zeros(batch_size)

            # Wall avoidance (not periodic boundaries)
            is_near_wall, turn_angle = avoid_wall(
                position[:, t], head_dir[:, t], self.arena_size, self.arena_size
            )
            v[is_near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt * random_turn[:, t]

            # Take a step
            velocity[:, t] = v * dt
            update = velocity[:, t, None] * np.stack(
                [np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1
            )
            position[:, t + 1] = position[:, t] + update

            # Rotate head direction
            head_dir[:, t + 1] = head_dir[:, t] + turn_angle

        head_dir = np.mod(head_dir + np.pi, 2 * np.pi) - np.pi  # Periodic variable

        # Extract trajectories (like old code)
        # init_pos = position[:, 1, :]  # Use position at t=1 as initial
        traj_pos = position[:, 2:, :]  # Positions from t=2 onwards
        ego_v = velocity[:, 1:-1]  # Ego velocities
        target_hd = head_dir[:, 1:-1]  # Head directions

        # Choose velocity representation (stay in numpy):
        if self.velocity_representation == "cartesian":
            # Convert to cartesian velocities (like old code does)
            v_inputs = np.stack(
                [ego_v * np.cos(target_hd), ego_v * np.sin(target_hd)], axis=-1
            )
        elif self.velocity_representation == "polar":
            # Keep as polar [heading, speed]
            v_inputs = np.stack([target_hd, ego_v], axis=-1)
        else:
            raise ValueError(
                f"Invalid velocity representation: {self.velocity_representation}"
            )

        # Compute place cell activations (need tensors for this, then convert back)
        pos_tensor = torch.tensor(traj_pos, dtype=torch.float32, device=device)
        place_cell_activations = self.get_place_cell_activations(pos_tensor)
        place_cell_activations_np = place_cell_activations.cpu().numpy()

        # Return all numpy arrays
        return v_inputs, traj_pos, place_cell_activations_np

    def simulate_trajectories(
        self, device: str = "cpu"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates trajectories based on the specified trajectory type.

        Returns:
            inputs: numpy array of shape (batch, T, 2)
            positions: numpy array of shape (batch, T, 2)
            place_cell_activations: numpy array of shape (batch, T, num_place_cells)
        """
        if self.trajectory_type == "ornstein_uhlenbeck":
            return self._simulate_ornstein_uhlenbeck_trajectories(device)
        elif self.trajectory_type == "random_walk":
            return self._simulate_random_walk_trajectories(device)
        else:
            raise NotImplementedError(
                f"Trajectory type '{self.trajectory_type}' is not implemented. "
                f"Currently supported: ['ornstein_uhlenbeck', 'random_walk']"
            )

    def setup(self, stage=None) -> None:
        # Get numpy arrays from trajectory generation
        inputs_np, positions_np, place_cells_np = self.simulate_trajectories(
            device="cpu"
        )

        # Convert to tensors ONCE
        inputs = torch.tensor(inputs_np, dtype=torch.float32)
        positions = torch.tensor(positions_np, dtype=torch.float32)
        place_cell_activations = torch.tensor(place_cells_np, dtype=torch.float32)

        full_dataset = TensorDataset(inputs, positions, place_cell_activations)

        # split into train and val
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
