import lightning as L
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split


class PathIntegrationDataModule(L.LightningDataModule):
    def __init__(
        self,
        trajectory_type: str, # "ornstein_uhlenbeck"
        velocity_representation: str,  # "cartesian" | "polar" | "sincos_polar"
        dt: float,
        num_time_steps: int,
        arena_size: float,
        # Place cell parameters (needed early for behavioral timescale computation)
        num_place_cells: int,
        place_cell_rf: float,
        DoG: bool,
        surround_scale: float,
        place_cell_layout: str, # "random" | "uniform"
        # Speed parameterization: either direct or via behavioral timescale
        # Option A: Direct speed parameters
        linear_speed_mean: float = None,      # m/s
        linear_speed_std: float = None,       # m/s
        # Option B: Behavioral timescale parameters (τ_behavior = place_cell_rf / linear_speed)
        behavioral_timescale_mean: float = None,  # seconds
        behavioral_timescale_std: float = None,   # seconds
        # OU dynamics (shared by both parameterizations)
        linear_speed_tau: float = 1.0,        # OU autocorrelation time (s)
        # Angular speed parameters
        angular_speed_mean: float = 0.0,      # rad/s
        angular_speed_std: float = 1.0,       # rad/s
        angular_speed_tau: float = 0.4,       # s
        # DataLoader parameters
        num_trajectories: int = 10000,
        batch_size: int = 200,
        num_workers: int = 4,
        train_val_split: float = 0.8,
    ) -> None:
        """
        Initialize the PathIntegrationDataModule.

        Speed can be parameterized in two ways:
        
        Option A - Direct speed parameters:
            linear_speed_mean, linear_speed_std
            
        Option B - Behavioral timescale parameters:
            behavioral_timescale_mean, behavioral_timescale_std
            where τ_behavior = place_cell_rf / linear_speed
            
        If behavioral timescale params are provided, they take precedence and
        linear_speed params are computed from them.

        :param trajectory_type: Trajectory generation type (only "ornstein_uhlenbeck" supported)
        :param velocity_representation: Input encoding - "cartesian", "polar", or "sincos_polar"
        :param dt: Simulation time step size (s)
        :param num_time_steps: Number of time steps to simulate
        :param arena_size: Size of the arena (m)
        :param num_place_cells: Number of place cells
        :param place_cell_rf: Place cell receptive field radius (m)
        :param DoG: Whether to use DoG place cell activation
        :param surround_scale: Surround scale for DoG place cell activation
        :param place_cell_layout: Place cell layout ("random" | "uniform")
        :param linear_speed_mean: Mean linear speed (m/s) - Option A
        :param linear_speed_std: Std of linear speed (m/s) for OU process - Option A
        :param behavioral_timescale_mean: Mean behavioral timescale (s) - Option B
        :param behavioral_timescale_std: Std of behavioral timescale (s) - Option B
        :param linear_speed_tau: Autocorrelation time for speed OU process (s)
        :param angular_speed_mean: Mean angular velocity (rad/s), typically 0.0
        :param angular_speed_std: Std of angular velocity (rad/s) for OU process
        :param angular_speed_tau: Autocorrelation time for angular velocity (s)
        :param num_trajectories: Number of trajectories to simulate
        :param batch_size: Batch size
        :param num_workers: Number of workers for data loading
        :param train_val_split: Train/val split ratio
        """
        super().__init__()
        
        self.trajectory_type = trajectory_type
        self.velocity_representation = velocity_representation
        self.dt = dt
        self.num_time_steps = num_time_steps
        self.arena_size = arena_size
        
        # Place cell parameters (store first, needed for behavioral timescale conversion)
        self.num_place_cells = num_place_cells
        self.place_cell_rf = place_cell_rf
        self.DoG = DoG
        self.surround_scale = surround_scale
        self.place_cell_layout = place_cell_layout
        
        # Determine speed parameterization
        use_behavioral_timescale = (
            behavioral_timescale_mean is not None and 
            behavioral_timescale_std is not None
        )
        
        if use_behavioral_timescale:
            # Option B: Convert behavioral timescale to speed
            # τ_behavior = rf / v  =>  v = rf / τ_behavior
            # 
            # For the std, using Taylor expansion around the mean:
            # σ_v ≈ (rf / τ_mean²) × σ_τ
            self.behavioral_timescale_mean = behavioral_timescale_mean
            self.behavioral_timescale_std = behavioral_timescale_std
            
            self.linear_speed_mean = place_cell_rf / behavioral_timescale_mean
            self.linear_speed_std = (place_cell_rf / behavioral_timescale_mean**2) * behavioral_timescale_std
            
            print(f"Behavioral timescale parameterization:")
            print(f"  τ_behavior: mean={behavioral_timescale_mean:.3f}s, std={behavioral_timescale_std:.3f}s")
            print(f"  → linear_speed: mean={self.linear_speed_mean:.3f}m/s, std={self.linear_speed_std:.3f}m/s")
        else:
            # Option A: Direct speed parameters
            if linear_speed_mean is None or linear_speed_std is None:
                raise ValueError(
                    "Must provide either (linear_speed_mean, linear_speed_std) or "
                    "(behavioral_timescale_mean, behavioral_timescale_std)"
                )
            self.linear_speed_mean = linear_speed_mean
            self.linear_speed_std = linear_speed_std
            self.behavioral_timescale_mean = None
            self.behavioral_timescale_std = None
        
        self.linear_speed_tau = linear_speed_tau
        self.angular_speed_mean = angular_speed_mean
        self.angular_speed_std = angular_speed_std
        self.angular_speed_tau = angular_speed_tau        

        self.num_trajectories = num_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

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

    def _get_place_cell_activations(self, pos: torch.Tensor) -> torch.Tensor:
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

    def _simulate_unicycle_ou(self, device: str = "cpu"):
        """
        Generate (B,T) trajectories using a unicycle with OU on linear speed v and angular velocity ω.
        Outputs encoded per self.velocity_representation.
        Returns: inputs[B,T,C], positions[B,T,2], place_cell_activations[B,T,Np]
        """
        B, T, dt = self.num_trajectories, self.num_time_steps, self.dt
        R = self.arena_size / 2.0

        # --- Parameters ---
        mu_v   = float(self.linear_speed_mean)
        s_v    = float(self.linear_speed_std)
        tau_v  = float(self.linear_speed_tau)

        mu_om  = float(self.angular_speed_mean)
        s_om   = float(self.angular_speed_std)
        tau_om = float(self.angular_speed_tau)

        # OU diffusion scales from target std + tau: Var = (sigma^2 * tau)/2
        sig_v  = np.float32(np.sqrt(2.0 * (s_v**2)  / max(tau_v,  1e-6)))
        sig_om = np.float32(np.sqrt(2.0 * (s_om**2) / max(tau_om, 1e-6)))
        sqrt_dt = np.float32(np.sqrt(dt))

        # --- Init state ---
        pos = np.random.uniform(-R, R, size=(B, 2)).astype(np.float32)
        th  = np.random.uniform(-np.pi, np.pi, size=(B,)).astype(np.float32)
        v   = np.clip(np.random.normal(mu_v,  s_v,  size=(B,)), 0.0, None).astype(np.float32)
        om  = np.random.normal(mu_om, s_om, size=(B,)).astype(np.float32)

        pos_list = [pos.copy()]
        th_list  = [th.copy()]
        v_list   = [v.copy()]

        for _ in range(T-1):
            # --- OU updates ---
            v  += (-(v  - mu_v )/tau_v )*dt + sig_v * sqrt_dt * np.random.randn(B).astype(np.float32)
            v   = np.maximum(v, 0.0)
            om += (-(om - mu_om)/tau_om)*dt + sig_om* sqrt_dt * np.random.randn(B).astype(np.float32)

            # --- Kinematics ---
            th = (th + om*dt + np.pi) % (2*np.pi) - np.pi
            pos = pos + np.stack([v*np.cos(th)*dt, v*np.sin(th)*dt], axis=-1)

            # --- Robust square reflection (position + heading) ---
            # X axis (up to two reflections to handle large steps)
            for _ in range(2):
                over  = pos[:, 0] >  R
                under = pos[:, 0] < -R
                if np.any(over):
                    pos[over, 0] =  2*R - pos[over, 0]
                    th[over]     =  np.pi - th[over]
                if np.any(under):
                    pos[under, 0] = -2*R - pos[under, 0]
                    th[under]     =  np.pi - th[under]
            # Y axis (up to two reflections)
            for _ in range(2):
                over  = pos[:, 1] >  R
                under = pos[:, 1] < -R
                if np.any(over):
                    pos[over, 1] =  2*R - pos[over, 1]
                    th[over]     = -th[over]
                if np.any(under):
                    pos[under, 1] = -2*R - pos[under, 1]
                    th[under]     = -th[under]

            pos_list.append(pos.copy())
            th_list.append(th.copy())
            v_list.append(v.copy())

        pos_all = np.stack(pos_list, axis=1).astype(np.float32)  # (B,T,2)
        th_all  = np.stack(th_list,  axis=1).astype(np.float32)  # (B,T)
        v_all   = np.stack(v_list,   axis=1).astype(np.float32)  # (B,T)

        # --- Input encodings ---
        rep = self.velocity_representation
        if rep == "cartesian":
            vx = v_all * np.cos(th_all)
            vy = v_all * np.sin(th_all)
            inputs = np.stack([vx, vy], axis=-1).astype(np.float32)           # (B,T,2)
        elif rep == "polar":
            inputs = np.stack([th_all, v_all], axis=-1).astype(np.float32)     # (B,T,2)
        elif rep == "sincos_polar":
            inputs = np.stack([np.cos(th_all), np.sin(th_all), v_all], axis=-1).astype(np.float32)  # (B,T,3)
        else:
            raise ValueError("velocity_representation must be one of "
                             "['cartesian','polar','sincos_polar']")

        # --- Place cells unchanged ---
        pos_tensor = torch.tensor(pos_all, dtype=torch.float32, device=device)
        pc = self._get_place_cell_activations(pos_tensor).cpu().numpy()

        return inputs, pos_all, pc

    def simulate_trajectories(
        self, device: str = "cpu"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates trajectories based on the specified trajectory type.

        Returns:
            inputs: numpy array of shape (batch, T, C) where C depends on velocity_representation
                   - cartesian: C=2 (vx, vy)
                   - polar: C=2 (heading, speed)
                   - sincos_polar: C=3 (cos(heading), sin(heading), speed)
            positions: numpy array of shape (batch, T, 2)
            place_cell_activations: numpy array of shape (batch, T, num_place_cells)
        """
        if self.trajectory_type == "ornstein_uhlenbeck":
            # Now uses unicycle-OU core
            return self._simulate_unicycle_ou(device)
        else:
            raise NotImplementedError(
                f"Trajectory type '{self.trajectory_type}' is not implemented. "
                f"Currently supported: ['ornstein_uhlenbeck']"
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
