"""
Spatial analysis tools for RNN hidden units.

Tools for computing and visualizing spatial rate maps, identifying place cells,
grid cells, and other spatially-tuned units.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Optional, List
from timescales.rnns.rnn import RNN
from timescales.rnns.multitimescale_rnn import MultiTimescaleRNN
from timescales.scores import GridScorer



class SpatialAnalyzer:
    """
    Analyzer for spatial properties of RNN hidden units.

    Computes spatial rate maps and provides tools for identifying
    spatially-selective units like place cells and grid cells.
    """

    def __init__(
        self,
        model: RNN | MultiTimescaleRNN,
        device: str,
        model_type: str = "vanilla",
    ):
        """
        Initialize spatial analyzer.

        Parameters:
        -----------
        model : RNN or MultiTimescaleRNN
            Trained model to analyze
        device : str
            Device to run model on ("cuda" or "cpu")
        model_type : str
            Type of model ("vanilla" or "multitimescale")
        """
        self.model = model
        self.device = device
        self.model_type = model_type

        # Will be populated after computing rate maps
        self.rate_maps = None
        self.occupancy_map = None
        self.bin_centers_x = None
        self.bin_centers_y = None
        self.bin_size = None
        self.arena_size = None
        self.time_lag = None

    def compute_rate_maps(
        self,
        eval_loader,
        arena_size: float = 2.2,
        bin_size: float = 0.1,
        num_trajectories: int = 1000,
        min_occupancy: int = 5,
        time_lag: int = 0,
    ) -> "SpatialAnalyzer":
        """
        Compute spatial rate maps for all hidden units with optional time lag.

        Parameters:
        -----------
        eval_loader : DataLoader
            DataLoader providing trajectories
        arena_size : float
            Size of the arena (meters)
        bin_size : float
            Size of spatial bins (meters)
        num_trajectories : int
            Number of trajectories to use
        min_occupancy : int
            Minimum visits to a bin for reliable estimate
        time_lag : int
            Time lag between neural activity and position (in time steps):
            - time_lag = 0: Concurrent encoding (activity at t vs position at t)
            - time_lag > 0: Prospective encoding (activity at t vs position at t+lag)
            - time_lag < 0: Retrospective encoding (activity at t vs position at t+lag, past)
            Example: time_lag=5 means "does neuron at time t encode position 5 steps ahead?"

        Returns:
        --------
        self : SpatialAnalyzer
            Returns self for method chaining
        """
        lag_description = (
            "concurrent" if time_lag == 0
            else f"prospective (lag={time_lag})" if time_lag > 0
            else f"retrospective (lag={time_lag})"
        )
        print(f"Computing {lag_description} spatial rate maps using {num_trajectories} trajectories...")

        # Store parameters
        self.arena_size = arena_size
        self.bin_size = bin_size
        self.time_lag = time_lag

        # Define spatial bins
        x_min, x_max = -arena_size / 2, arena_size / 2
        y_min, y_max = -arena_size / 2, arena_size / 2

        x_bins = np.arange(x_min, x_max + bin_size, bin_size)
        y_bins = np.arange(y_min, y_max + bin_size, bin_size)

        n_bins_x = len(x_bins) - 1
        n_bins_y = len(y_bins) - 1

        print(f"Spatial grid: {n_bins_x} x {n_bins_y} bins of size {bin_size}")

        # Get bin centers for plotting
        self.bin_centers_x: np.ndarray = (x_bins[:-1] + x_bins[1:]) / 2
        self.bin_centers_y: np.ndarray = (y_bins[:-1] + y_bins[1:]) / 2

        # Determine hidden size
        hidden_size = self._get_hidden_size(eval_loader)
        print(f"Hidden units to analyze: {hidden_size}")

        # Initialize spatial maps
        activation_sums = np.zeros((hidden_size, n_bins_x, n_bins_y))
        occupancy_counts = np.zeros((n_bins_x, n_bins_y))

        # Collect data from trajectories
        trajectory_count = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, positions, place_cells) in enumerate(eval_loader):
                if trajectory_count >= num_trajectories:
                    break

                batch_size = inputs.shape[0]
                if trajectory_count + batch_size > num_trajectories:
                    # Take only what we need
                    n_take = num_trajectories - trajectory_count
                    inputs = inputs[:n_take]
                    positions = positions[:n_take]
                    place_cells = place_cells[:n_take]
                    batch_size = n_take

                inputs = inputs.to(self.device)
                place_cells = place_cells.to(self.device)

                # Get hidden states
                hidden_states = self._get_hidden_states(inputs, place_cells)

                # Convert to numpy
                hidden_np = hidden_states.cpu().numpy()  # (batch, time, hidden)
                positions_np = positions.cpu().numpy()  # (batch, time, 2)

                # Process each trajectory in the batch
                for traj_idx in range(batch_size):
                    traj_hidden = hidden_np[traj_idx]  # (time, hidden)
                    traj_positions = positions_np[traj_idx]  # (time, 2)

                    # For each time step
                    for t in range(traj_hidden.shape[0]):
                        # Apply time lag: compare activity at t with position at t+lag
                        t_pos = t + time_lag
                        
                        # Check temporal bounds
                        if not (0 <= t_pos < traj_positions.shape[0]):
                            continue  # Skip if lagged position is out of bounds
                        
                        pos_x, pos_y = traj_positions[t_pos]  # Position at t+lag
                        activations = traj_hidden[t]  # Activity at t

                        # Find spatial bin
                        x_bin = np.digitize(pos_x, x_bins) - 1
                        y_bin = np.digitize(pos_y, y_bins) - 1

                        # Check spatial bounds
                        if 0 <= x_bin < n_bins_x and 0 <= y_bin < n_bins_y:
                            # Add activations to this spatial bin
                            activation_sums[:, x_bin, y_bin] += activations
                            occupancy_counts[x_bin, y_bin] += 1

                trajectory_count += batch_size

                if batch_idx % 50 == 0:
                    print(
                        f"Processed {trajectory_count}/{num_trajectories} trajectories..."
                    )

        print(f"Data collection complete. Total trajectories: {trajectory_count}")

        # Create rate maps by normalizing by occupancy
        self.rate_maps: np.ndarray = np.zeros_like(activation_sums)

        for i in range(hidden_size):
            for x in range(n_bins_x):
                for y in range(n_bins_y):
                    if occupancy_counts[x, y] >= min_occupancy:
                        self.rate_maps[i, x, y] = (
                            activation_sums[i, x, y] / occupancy_counts[x, y]
                        )
                    else:
                        self.rate_maps[i, x, y] = np.nan  # Not enough data

        self.occupancy_map: np.ndarray = occupancy_counts

        print(
            f"Rate maps created. Bins with sufficient data: {np.sum(occupancy_counts >= min_occupancy)}/{n_bins_x * n_bins_y}"
        )

        return self

    def plot_rate_maps(
        self,
        units_to_plot: Optional[List[int]] = None,
        num_units: int = 20,
        selection_method: str = "spatial_info",
        figsize_per_unit: Tuple[int, int] = (3, 3),
        cols: int = 5,
        cmap: str = "viridis",
        random_seed: int = 42,
    ) -> "SpatialAnalyzer":
        """
        Plot spatial rate maps for selected hidden units.

        Parameters:
        -----------
        units_to_plot : list, optional
            Specific unit indices to plot. If None, will auto-select
        num_units : int
            How many units to plot (when units_to_plot is None)
        selection_method : str
            "spatial_info" (highest variance) or "random"
        figsize_per_unit : tuple
            Size of each subplot
        cols : int
            Number of columns in plot grid
        cmap : str
            Colormap for rate maps
        random_seed : int
            Seed for random selection

        Returns:
        --------
        self : SpatialAnalyzer
            Returns self for method chaining
        """
        assert self.rate_maps is not None, "Must compute rate maps first"

        if units_to_plot is None:
            units_to_plot = self._select_units(num_units, selection_method, random_seed)

        n_plot = len(units_to_plot)
        rows = (n_plot + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * figsize_per_unit[0], rows * figsize_per_unit[1])
        )
        if n_plot == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        title_prefix = f"{self.model_type.capitalize()}"

        for idx, unit_idx in enumerate(units_to_plot):
            row = idx // cols
            col = idx % cols

            if rows == 1:
                ax = axes[col] if cols > 1 else axes[0]
            else:
                ax = axes[row, col]

            rate_map = self.rate_maps[unit_idx].T  # Transpose for proper orientation

            # Create masked array to handle NaN values
            rate_map_masked = np.ma.masked_where(np.isnan(rate_map), rate_map)

            im = ax.imshow(
                rate_map_masked,
                extent=[
                    self.bin_centers_x[0],
                    self.bin_centers_x[-1],
                    self.bin_centers_y[0],
                    self.bin_centers_y[-1],
                ],
                origin="lower",
                cmap=cmap,
                aspect="equal",
            )

            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")

            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Add spatial information in title
            valid_rates = rate_map[~np.isnan(rate_map)]
            if len(valid_rates) > 0:
                spatial_var = np.var(valid_rates)
                ax.set_title(f"{title_prefix} {unit_idx}\n(var={spatial_var:.3f})")
            else:
                ax.set_title(f"{title_prefix} {unit_idx}")

        # Hide empty subplots
        for idx in range(n_plot, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows == 1:
                ax = axes[col] if cols > 1 else axes[0]
            else:
                ax = axes[row, col]
            ax.set_visible(False)

        # Create title with time lag information
        if self.time_lag is not None and self.time_lag != 0:
            lag_info = f"lag={self.time_lag:+d}" if self.time_lag != 0 else ""
            lag_type = "prospective" if self.time_lag > 0 else "retrospective"
            title = f"Spatial Rate Maps - {lag_type} ({lag_info}, {selection_method}, n={n_plot})"
        else:
            title = f"Spatial Rate Maps ({selection_method}, n={n_plot})"
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

        return self

    def plot_occupancy_map(
        self, figsize: Tuple[int, int] = (8, 6)
    ) -> "SpatialAnalyzer":
        """Plot the spatial occupancy map."""
        assert self.occupancy_map is not None, "Must compute occupancy map first"

        plt.figure(figsize=figsize)
        plt.imshow(
            self.occupancy_map.T,
            extent=[
                self.bin_centers_x[0],
                self.bin_centers_x[-1],
                self.bin_centers_y[0],
                self.bin_centers_y[-1],
            ],
            origin="lower",
            cmap="Blues",
            aspect="equal",
        )
        plt.colorbar(label="Occupancy (time steps)")
        plt.title("Spatial Occupancy Map")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.show()

        return self

    def get_spatial_info_scores(self) -> np.ndarray:
        """
        Get spatial information scores for all units.

        Returns:
        --------
        spatial_info : np.ndarray
            Array of spatial information scores (variance of firing rates)
        """
        if self.rate_maps is None:
            raise ValueError("Must compute rate maps first using compute_rate_maps()")

        spatial_info = []
        for i in range(self.rate_maps.shape[0]):
            rate_map = self.rate_maps[i]
            valid_rates = rate_map[~np.isnan(rate_map)]
            if len(valid_rates) > 0:
                spatial_info.append(np.var(valid_rates))
            else:
                spatial_info.append(0)

        return np.array(spatial_info)

    def compute_grid_scores(
        self,
        mask_parameters=None,
        min_max=False,
    ) -> dict:
        """
        Compute grid scores for all units using spatial autocorrelation.
        
        Parameters:
        -----------
        mask_parameters : list of tuples, optional
            List of (mask_min, mask_max) tuples defining ring masks for scoring.
            Default: [(0.4, 1.0)] - reasonable for grid cells
        min_max : bool
            If True, use min/max scoring; otherwise use mean
            
        Returns:
        --------
        dict with keys:
            - 'scores_60': Array of 60-degree grid scores
            - 'scores_90': Array of 90-degree grid scores  
            - 'mask_params_60': Best mask parameters for each unit (60-degree)
            - 'mask_params_90': Best mask parameters for each unit (90-degree)
            - 'sacs': Spatial autocorrelograms for each unit
        """
        assert self.rate_maps is not None, "Must compute rate maps first"
        
        if mask_parameters is None:
            # Default mask parameters - reasonable for grid cells
            mask_parameters = [(0.4, 1.0), (0.6, 1.0), (0.5, 1.0)]
        
        # Get number of bins from rate maps
        n_bins = self.rate_maps.shape[1]  # Assuming square rate maps
        
        # Define coordinate range based on arena size
        coords_range = [
            [-self.arena_size/2, self.arena_size/2],
            [-self.arena_size/2, self.arena_size/2]
        ]
        
        # Initialize grid scorer
        scorer = GridScorer(
            nbins=n_bins,
            coords_range=coords_range,
            mask_parameters=mask_parameters,
            min_max=min_max
        )
        
        n_units = self.rate_maps.shape[0]
        scores_60 = np.zeros(n_units)
        scores_90 = np.zeros(n_units)
        mask_params_60 = []
        mask_params_90 = []
        sacs = []
        
        print(f"Computing grid scores for {n_units} units...")
        
        for i in range(n_units):
            rate_map = self.rate_maps[i]  # Shape: (n_bins, n_bins)
            
            # Get grid scores
            score_60, score_90, mask_60, mask_90, sac = scorer.get_scores(rate_map)
            
            scores_60[i] = score_60
            scores_90[i] = score_90
            mask_params_60.append(mask_60)
            mask_params_90.append(mask_90)
            sacs.append(sac)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{n_units} units...")
        
        print(f"Grid score computation complete!")
        print(f"  Mean 60° score: {np.mean(scores_60):.3f} ± {np.std(scores_60):.3f}")
        print(f"  Mean 90° score: {np.mean(scores_90):.3f} ± {np.std(scores_90):.3f}")
        print(f"  Max 60° score: {np.max(scores_60):.3f}")
        print(f"  Max 90° score: {np.max(scores_90):.3f}")
        
        # Store results
        self.grid_scores_60 = scores_60
        self.grid_scores_90 = scores_90
        self.grid_mask_params_60 = mask_params_60
        self.grid_mask_params_90 = mask_params_90
        self.sacs = sacs
        
        return {
            'scores_60': scores_60,
            'scores_90': scores_90,
            'mask_params_60': mask_params_60,
            'mask_params_90': mask_params_90,
            'sacs': sacs
        }

    def plot_top_grid_cells(
        self,
        num_cells=9,
        score_type='60',
        figsize=(12, 12),
        cmap='viridis',
    ):
        """
        Plot rate maps and autocorrelograms for top grid cells.
        
        Parameters:
        -----------
        num_cells : int
            Number of top grid cells to plot
        score_type : str
            '60' for hexagonal grids, '90' for square grids
        figsize : tuple
            Figure size
        cmap : str
            Colormap for rate maps and SACs
        """
        assert hasattr(self, 'grid_scores_60'), "Must compute grid scores first"
        
        # Select top units by grid score
        if score_type == '60':
            scores = self.grid_scores_60
            mask_params = self.grid_mask_params_60
            title_suffix = "60° (hexagonal)"
        else:
            scores = self.grid_scores_90
            mask_params = self.grid_mask_params_90
            title_suffix = "90° (square)"
        
        top_indices = np.argsort(scores)[-num_cells:][::-1]
        
        # Create grid scorer for plotting
        n_bins = self.rate_maps.shape[1]
        coords_range = [
            [-self.arena_size/2, self.arena_size/2],
            [-self.arena_size/2, self.arena_size/2]
        ]
        scorer = GridScorer(
            nbins=n_bins,
            coords_range=coords_range,
            mask_parameters=[(0.4, 1.0)],  # Dummy, won't be used
            min_max=False
        )
        
        # Plot
        cols = 3
        rows = (num_cells * 2 + cols - 1) // cols  # 2 rows per cell (ratemap + SAC)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        
        for plot_idx, unit_idx in enumerate(top_indices):
            # Plot rate map
            ax_ratemap = axes[plot_idx * 2]
            rate_map = self.rate_maps[unit_idx].T
            rate_map_masked = np.ma.masked_where(np.isnan(rate_map), rate_map)
            
            im1 = ax_ratemap.imshow(
                rate_map_masked,
                extent=[
                    self.bin_centers_x[0], self.bin_centers_x[-1],
                    self.bin_centers_y[0], self.bin_centers_y[-1]
                ],
                origin='lower',
                cmap=cmap,
                aspect='equal'
            )
            ax_ratemap.set_title(
                f"Unit {unit_idx}: Score={scores[unit_idx]:.3f}",
                fontsize=10
            )
            ax_ratemap.set_xlabel("X")
            ax_ratemap.set_ylabel("Y")
            plt.colorbar(im1, ax=ax_ratemap, fraction=0.046)
            
            # Plot SAC
            ax_sac = axes[plot_idx * 2 + 1]
            sac = self.sacs[unit_idx]
            scorer.plot_sac(
                sac,
                mask_params=mask_params[unit_idx],
                ax=ax_sac,
                title="Autocorrelogram",
                cmap=cmap
            )
        
        # Hide empty subplots
        for idx in range(num_cells * 2, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f"Top {num_cells} Grid Cells ({title_suffix})", fontsize=14, y=0.995)
        plt.tight_layout()
        plt.show()

    def _get_hidden_size(self, eval_loader) -> int:
        """Determine the size of the hidden layer."""
        sample_batch = next(iter(eval_loader))
        inputs, positions, place_cells = sample_batch
        inputs = inputs.to(self.device)
        place_cells = place_cells.to(self.device)

        with torch.no_grad():
            if self.model_type in ["vanilla", "multitimescale"]:
                hidden_states, _ = self.model(
                    inputs=inputs, place_cells_0=place_cells[:, 0, :]
                )
                return hidden_states.shape[-1]
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_hidden_states(self, inputs, place_cells):
        """Get hidden states from the model."""
        if self.model_type in ["vanilla", "multitimescale"]:
            hidden_states, _ = self.model(
                inputs=inputs, place_cells_0=place_cells[:, 0, :]
            )
            return hidden_states
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _select_units(
        self, num_units: int, selection_method: str, random_seed: int
    ) -> np.ndarray:
        """Select units for plotting based on method."""
        n_units = self.rate_maps.shape[0]
        num_units = min(num_units, n_units)

        if selection_method == "spatial_info":
            spatial_info = self.get_spatial_info_scores()
            units_to_plot = np.argsort(spatial_info)[-num_units:][::-1]
            print(f"Selected top {len(units_to_plot)} units by spatial information")
        elif selection_method == "random":
            np.random.seed(random_seed)
            units_to_plot = np.random.choice(n_units, size=num_units, replace=False)
            units_to_plot = np.sort(units_to_plot)
            print(f"Selected {len(units_to_plot)} random units: {units_to_plot}")
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}")

        return units_to_plot
