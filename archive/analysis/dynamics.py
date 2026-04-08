"""
Neural dynamics analysis tools.

Tools for analyzing the temporal dynamics of RNN hidden states including
PCA analysis, trajectory visualization, and phase space analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from typing import Optional, Tuple


def plot_pca_spectrum(
    pca_trained: PCA,
    pca_untrained: PCA,
    population_name: str = "Hidden States",
    n_components_to_show: int = 50,
    figsize: Tuple[int, int] = (15, 6),
) -> None:
    """
    Plot PCA spectrum comparison between trained and untrained models.

    Parameters:
    -----------
    pca_trained : sklearn.PCA
        Fitted PCA object for trained model
    pca_untrained : sklearn.PCA
        Fitted PCA object for untrained model
    population_name : str
        Name of the population being analyzed
    n_components_to_show : int
        Number of components to show in plots
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Individual component variance
    ax1 = axes[0]
    n_components_to_show = min(
        n_components_to_show, len(pca_trained.explained_variance_ratio_)
    )
    component_indices = np.arange(1, n_components_to_show + 1)

    ax1.semilogy(
        component_indices,
        pca_trained.explained_variance_ratio_[:n_components_to_show],
        "b-",
        linewidth=2,
        label="Trained",
        marker="o",
        markersize=4,
    )
    ax1.semilogy(
        component_indices,
        pca_untrained.explained_variance_ratio_[:n_components_to_show],
        "gray",
        linewidth=2,
        label="Untrained",
        marker="s",
        markersize=4,
    )

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio (log scale)")
    ax1.set_title("PCA Spectrum: Individual Components")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add key statistics as text
    ax1.text(
        0.02,
        0.98,
        f"PC1 (Trained): {pca_trained.explained_variance_ratio_[0]:.3f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )
    ax1.text(
        0.02,
        0.85,
        f"PC1 (Untrained): {pca_untrained.explained_variance_ratio_[0]:.3f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.7),
    )

    # Right plot: Cumulative variance
    ax2 = axes[1]
    trained_cumvar = np.cumsum(
        pca_trained.explained_variance_ratio_[:n_components_to_show]
    )
    untrained_cumvar = np.cumsum(
        pca_untrained.explained_variance_ratio_[:n_components_to_show]
    )

    ax2.plot(
        component_indices,
        trained_cumvar,
        "b-",
        linewidth=2,
        label="Trained",
        marker="o",
        markersize=4,
    )
    ax2.plot(
        component_indices,
        untrained_cumvar,
        "gray",
        linewidth=2,
        label="Untrained",
        marker="s",
        markersize=4,
    )

    # Add horizontal lines for common variance thresholds
    for threshold in [0.5, 0.8, 0.95]:
        ax2.axhline(y=threshold, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax2.text(
            n_components_to_show * 0.7,
            threshold + 0.02,
            f"{threshold*100}%",
            fontsize=9,
            color="red",
            alpha=0.8,
        )

    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("PCA Spectrum: Cumulative Variance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)

    # Add text showing how many components needed for different thresholds
    def components_for_variance(explained_var_ratio, threshold):
        cumsum = np.cumsum(explained_var_ratio)
        idx = np.where(cumsum >= threshold)[0]
        return idx[0] + 1 if len(idx) > 0 else len(explained_var_ratio)

    for i, threshold in enumerate([0.5, 0.8, 0.95]):
        n_trained = components_for_variance(
            pca_trained.explained_variance_ratio_, threshold
        )
        n_untrained = components_for_variance(
            pca_untrained.explained_variance_ratio_, threshold
        )

        ax2.text(
            0.02,
            0.98 - i * 0.08,
            f"{threshold*100}% variance: Trained={n_trained}, Untrained={n_untrained}",
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
        )

    plt.suptitle(f"PCA Analysis: {population_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print(f"\n=== PCA Spectrum Analysis for {population_name} ===")
    print(
        f"First 10 components (Trained):   {pca_trained.explained_variance_ratio_[:10]}"
    )
    print(
        f"First 10 components (Untrained): {pca_untrained.explained_variance_ratio_[:10]}"
    )

    print("\nVariance explained by first N components:")
    for n in [1, 2, 5, 10, 20]:
        if n <= len(pca_trained.explained_variance_ratio_):
            trained_var = pca_trained.explained_variance_ratio_[:n].sum()
            untrained_var = pca_untrained.explained_variance_ratio_[:n].sum()
            print(
                f"  {n:2d} components: Trained={trained_var:.3f}, Untrained={untrained_var:.3f}"
            )

    print("\nEffective dimensionality (95% variance):")
    print(
        f"  Trained:   {components_for_variance(pca_trained.explained_variance_ratio_, 0.95)} components"
    )
    print(
        f"  Untrained: {components_for_variance(pca_untrained.explained_variance_ratio_, 0.95)} components"
    )

    # Check for signs of low-dimensional structure
    pc1_dominance_trained = pca_trained.explained_variance_ratio_[0]
    pc1_dominance_untrained = pca_untrained.explained_variance_ratio_[0]

    print("\nFirst PC dominance:")
    print(f"  Trained PC1 explains {pc1_dominance_trained:.1%} of variance")
    print(f"  Untrained PC1 explains {pc1_dominance_untrained:.1%} of variance")

    if pc1_dominance_trained > 0.1:
        print("  → Trained model shows structured low-dimensional dynamics")
    else:
        print("  → Trained model shows high-dimensional/noisy dynamics")

    if pc1_dominance_untrained > 0.1:
        print("  → Untrained model shows some structure (unexpected)")
    else:
        print("  → Untrained model shows expected high-dimensional noise")


class DynamicsAnalyzer:
    """
    Analyzer for neural dynamics of RNN hidden states.

    Provides PCA analysis, trajectory visualization, and comparison
    between trained and untrained models.
    """

    def __init__(
        self,
        trained_model,
        untrained_model,
        device: str,
        population_to_visualize: str = "pop1",
    ):
        """
        Initialize dynamics analyzer.

        Parameters:
        -----------
        trained_model : MultiTimescaleRNN
            Trained model to analyze
        untrained_model : MultiTimescaleRNN
            Untrained model for comparison
        device : str
            Device to run models on
        population_to_visualize : str
            For multitimescale models: "pop1", "pop2", or "both"
        """
        self.trained_model = trained_model
        self.untrained_model = untrained_model
        self.device = device
        self.population_to_visualize = population_to_visualize

        self.trained_hidden_states: Optional[torch.Tensor] = None
        self.untrained_hidden_states: Optional[torch.Tensor] = None
        self.inputs: Optional[torch.Tensor] = None
        self.positions: Optional[torch.Tensor] = None
        self.place_cells: Optional[torch.Tensor] = None
        self.pca_trained: Optional[PCA] = None
        self.pca_untrained: Optional[PCA] = None
        self.trained_pca_data: Optional[np.ndarray] = None
        self.untrained_pca_data: Optional[np.ndarray] = None

    def compute_hidden_states(
        self,
        eval_loader,
        num_batches: int = 1000,
    ) -> "DynamicsAnalyzer":
        """
        Compute hidden states for both trained and untrained models.

        Parameters:
        -----------
        eval_loader : DataLoader
            DataLoader providing trajectories
        num_batches : int
            Number of batches to process

        Returns:
        --------
        self : DynamicsAnalyzer
            Returns self for method chaining
        """
        print(f"Computing hidden states from {num_batches} batches...")

        # Collect trained model states
        trained_states = []
        untrained_states = []
        all_inputs = []
        all_positions = []
        all_place_cells = []

        self.trained_model.eval()
        self.untrained_model.eval()

        with torch.no_grad():
            for i, (inputs, positions, place_cells) in enumerate(eval_loader):
                if i >= num_batches:
                    break

                inputs = inputs.to(self.device)
                positions = positions.to(self.device)
                place_cells = place_cells.to(self.device)

                # Get trained model hidden states
                trained_hidden = self._get_hidden_states(
                    self.trained_model, inputs, place_cells
                )
                trained_states.append(trained_hidden)

                # Get untrained model hidden states
                # Process one trajectory at a time to match your existing approach
                batch_size = inputs.shape[0]
                untrained_batch_states = []

                for j in range(batch_size):
                    single_input = inputs[j : j + 1]
                    single_place_cells = place_cells[j : j + 1]

                    untrained_hidden = self._get_hidden_states(
                        self.untrained_model, single_input, single_place_cells
                    )
                    untrained_batch_states.append(untrained_hidden)

                untrained_states.append(torch.cat(untrained_batch_states, dim=0))

                all_inputs.append(inputs)
                all_positions.append(positions)
                all_place_cells.append(place_cells)

                if i % 100 == 0:
                    print(f"Processed batch {i+1}/{num_batches}")

        # Concatenate and store
        self.trained_hidden_states = torch.cat(trained_states, dim=0)
        self.untrained_hidden_states = torch.cat(untrained_states, dim=0)
        self.inputs = torch.cat(all_inputs, dim=0)
        self.positions = torch.cat(all_positions, dim=0)
        self.place_cells = torch.cat(all_place_cells, dim=0)

        print("Hidden states computed:")
        print(f"  Trained shape: {self.trained_hidden_states.shape}")
        print(f"  Untrained shape: {self.untrained_hidden_states.shape}")

        return self

    def compute_pca(self, n_components: int = 6) -> "DynamicsAnalyzer":
        """
        Compute PCA on the hidden states.

        Parameters:
        -----------
        n_components : int
            Number of PCA components to compute

        Returns:
        --------
        self : DynamicsAnalyzer
            Returns self for method chaining
        """
        assert (
            self.trained_hidden_states is not None
        ), "Must compute hidden states first"
        assert (
            self.untrained_hidden_states is not None
        ), "Must compute hidden states first"

        print(f"Computing PCA with {n_components} components...")

        # Reshape for PCA
        trained_data = self.trained_hidden_states.reshape(
            -1, self.trained_hidden_states.shape[-1]
        ).cpu()
        untrained_data = self.untrained_hidden_states.reshape(
            -1, self.untrained_hidden_states.shape[-1]
        ).cpu()

        print("Data shapes for PCA:")
        print(f"  Trained: {trained_data.shape}")
        print(f"  Untrained: {untrained_data.shape}")
        print(
            f"  Ratio (samples/features): {trained_data.shape[0] / trained_data.shape[1]:.2f}"
        )

        # Fit PCAs
        self.pca_trained = PCA(n_components=n_components)
        self.pca_untrained = PCA(n_components=n_components)

        self.trained_pca_data = self.pca_trained.fit_transform(trained_data)
        self.untrained_pca_data = self.pca_untrained.fit_transform(untrained_data)

        print("PCA completed:")
        print(
            f"  Trained variance explained: {self.pca_trained.explained_variance_ratio_.sum() * 100:.2f}%"
        )
        print(
            f"  Untrained variance explained: {self.pca_untrained.explained_variance_ratio_.sum() * 100:.2f}%"
        )

        return self

    def plot_pca_spectrum(
        self, n_components_to_show: int = 50, figsize: Tuple[int, int] = (15, 6)
    ) -> "DynamicsAnalyzer":
        """
        Plot PCA spectrum comparison.

        Parameters:
        -----------
        n_components_to_show : int
            Number of components to show in plots
        figsize : tuple
            Figure size

        Returns:
        --------
        self : DynamicsAnalyzer
            Returns self for method chaining
        """
        if self.pca_trained is None:
            raise ValueError("Must compute PCA first using compute_pca()")

        population_name = f"MultiTimescaleRNN {self.population_to_visualize}"

        plot_pca_spectrum(
            self.pca_trained,
            self.pca_untrained,
            population_name,
            n_components_to_show,
            figsize,
        )

        return self

    def plot_2d_pca_trajectories(
        self, num_trajectories: int = 5, figsize: Tuple[int, int] = (12, 10)
    ) -> "DynamicsAnalyzer":
        """
        Plot 2D PCA trajectories colored by spatial position.

        Parameters:
        -----------
        num_trajectories : int
            Number of trajectories to plot
        figsize : tuple
            Figure size

        Returns:
        --------
        self : DynamicsAnalyzer
            Returns self for method chaining
        """
        assert self.trained_pca_data is not None, "Must compute PCA first"
        assert self.untrained_pca_data is not None, "Must compute PCA first"
        assert self.positions is not None, "Must compute hidden states first"
        assert self.pca_trained is not None, "Must compute PCA first"
        assert self.pca_untrained is not None, "Must compute PCA first"

        # Extract spatial coordinates
        positions_flat = self.positions.reshape(-1, 2).cpu().numpy()
        x_coords = positions_flat[:, 0]
        y_coords = positions_flat[:, 1]

        # Extract PC1 and PC2 for 2D plots
        trained_2d = self.trained_pca_data[:, :2]
        untrained_2d = self.untrained_pca_data[:, :2]

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Row 1: Color by X coordinate
        scatter1 = axes[0, 0].scatter(
            untrained_2d[:, 0],
            untrained_2d[:, 1],
            c=x_coords,
            cmap="viridis",
            s=20,
            alpha=0.7,
        )
        axes[0, 0].set_title("Untrained (colored by X)")
        axes[0, 0].set_xlabel(
            f"PC1 ({self.pca_untrained.explained_variance_ratio_[0]*100:.1f}%)"
        )
        axes[0, 0].set_ylabel(
            f"PC2 ({self.pca_untrained.explained_variance_ratio_[1]*100:.1f}%)"
        )
        axes[0, 0].grid(True, alpha=0.3)

        scatter2 = axes[0, 1].scatter(
            trained_2d[:, 0],
            trained_2d[:, 1],
            c=x_coords,
            cmap="viridis",
            s=20,
            alpha=0.7,
        )
        axes[0, 1].set_title("Trained (colored by X)")
        axes[0, 1].set_xlabel(
            f"PC1 ({self.pca_trained.explained_variance_ratio_[0]*100:.1f}%)"
        )
        axes[0, 1].set_ylabel(
            f"PC2 ({self.pca_trained.explained_variance_ratio_[1]*100:.1f}%)"
        )
        axes[0, 1].grid(True, alpha=0.3)

        # Row 2: Color by Y coordinate
        scatter3 = axes[1, 0].scatter(
            untrained_2d[:, 0],
            untrained_2d[:, 1],
            c=y_coords,
            cmap="plasma",
            s=20,
            alpha=0.7,
        )
        axes[1, 0].set_title("Untrained (colored by Y)")
        axes[1, 0].set_xlabel(
            f"PC1 ({self.pca_untrained.explained_variance_ratio_[0]*100:.1f}%)"
        )
        axes[1, 0].set_ylabel(
            f"PC2 ({self.pca_untrained.explained_variance_ratio_[1]*100:.1f}%)"
        )
        axes[1, 0].grid(True, alpha=0.3)

        scatter4 = axes[1, 1].scatter(
            trained_2d[:, 0],
            trained_2d[:, 1],
            c=y_coords,
            cmap="plasma",
            s=20,
            alpha=0.7,
        )
        axes[1, 1].set_title("Trained (colored by Y)")
        axes[1, 1].set_xlabel(
            f"PC1 ({self.pca_trained.explained_variance_ratio_[0]*100:.1f}%)"
        )
        axes[1, 1].set_ylabel(
            f"PC2 ({self.pca_trained.explained_variance_ratio_[1]*100:.1f}%)"
        )
        axes[1, 1].grid(True, alpha=0.3)

        # Add colorbars
        plt.colorbar(scatter1, ax=axes[0, 0], shrink=0.8, label="X position")
        plt.colorbar(scatter2, ax=axes[0, 1], shrink=0.8, label="X position")
        plt.colorbar(scatter3, ax=axes[1, 0], shrink=0.8, label="Y position")
        plt.colorbar(scatter4, ax=axes[1, 1], shrink=0.8, label="Y position")

        # Overall title
        population_name = f"MultiTimescaleRNN {self.population_to_visualize}"

        fig.suptitle(
            f"Hidden States Colored by Spatial Position (2D PCA) - {population_name}",
            fontsize=16,
            y=0.95,
        )
        plt.tight_layout()
        plt.show()

        return self

    def _get_hidden_states(self, model, inputs, place_cells):
        """Get hidden states from a model."""
        hidden_states, _ = model(inputs=inputs, init_context=place_cells[:, 0, :])
        return hidden_states

    def plot_spatial_vs_latent_distance(
        self,
        num_pairs: int = 10000,
        max_spatial_distance: Optional[float] = None,
        distance_metric: str = "euclidean",  # or "cosine"
        figsize: Tuple[int, int] = (10, 8),
        alpha: float = 0.1,
        sample_seed: int = 42,
    ) -> "DynamicsAnalyzer":
        """
        Analyze relationship between spatial distances and latent distances.

        For trained and untrained models, this plots the relationship between:
        - X-axis: Real spatial distance ||x(t_i) - x(t_j)||₂
        - Y-axis: Hidden representation distance d(h(t_i), h(t_j))

        Parameters:
        -----------
        num_pairs : int
            Number of random timepoint pairs to sample
        max_spatial_distance : float, optional
            Maximum spatial distance to include (for focusing on local structure)
        distance_metric : str
            "euclidean" for L2 norm or "cosine" for cosine distance
        figsize : tuple
            Figure size
        alpha : float
            Point transparency for scatter plot
        sample_seed : int
            Random seed for reproducible sampling

        Returns:
        --------
        self for method chaining
        """
        assert (
            self.trained_hidden_states is not None
        ), "Must compute hidden states first"
        assert (
            self.untrained_hidden_states is not None
        ), "Must compute hidden states first"
        assert self.positions is not None, "Must compute hidden states first"

        np.random.seed(sample_seed)

        # Flatten data for easy pair sampling
        # Shape: (total_timepoints, feature_dim)
        positions_flat = self.positions.reshape(-1, 2).cpu().numpy()  # (B*T, 2)
        trained_hidden_flat = (
            self.trained_hidden_states.reshape(-1, self.trained_hidden_states.shape[-1])
            .cpu()
            .numpy()
        )  # (B*T, H)
        untrained_hidden_flat = (
            self.untrained_hidden_states.reshape(
                -1, self.untrained_hidden_states.shape[-1]
            )
            .cpu()
            .numpy()
        )  # (B*T, H)

        total_timepoints = positions_flat.shape[0]

        # Sample random pairs of timepoints
        idx_pairs = np.random.choice(
            total_timepoints, size=(num_pairs, 2), replace=True
        )

        # Function to compute distance based on metric
        def compute_latent_distance(vec1, vec2, metric):
            if metric == "euclidean":
                return np.linalg.norm(vec1 - vec2)
            elif metric == "cosine":
                # Cosine distance = 1 - cosine_similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 == 0 or norm2 == 0:
                    return 1.0  # Maximum cosine distance
                cosine_sim = dot_product / (norm1 * norm2)
                return 1 - cosine_sim
            else:
                raise ValueError(f"Unknown distance metric: {metric}")

        # Compute distances
        spatial_distances = []
        trained_latent_distances = []
        untrained_latent_distances = []

        for i, j in idx_pairs:
            # Spatial distance in 2D
            spatial_dist = np.linalg.norm(positions_flat[i] - positions_flat[j])

            # Latent distances in high-D
            trained_latent_dist = compute_latent_distance(
                trained_hidden_flat[i], trained_hidden_flat[j], distance_metric
            )
            untrained_latent_dist = compute_latent_distance(
                untrained_hidden_flat[i], untrained_hidden_flat[j], distance_metric
            )

            spatial_distances.append(spatial_dist)
            trained_latent_distances.append(trained_latent_dist)
            untrained_latent_distances.append(untrained_latent_dist)

        spatial_distances = np.array(spatial_distances)
        trained_latent_distances = np.array(trained_latent_distances)
        untrained_latent_distances = np.array(untrained_latent_distances)

        # Filter by max spatial distance if specified
        if max_spatial_distance is not None:
            mask = spatial_distances <= max_spatial_distance
            spatial_distances = spatial_distances[mask]
            trained_latent_distances = trained_latent_distances[mask]
            untrained_latent_distances = untrained_latent_distances[mask]

        # Create scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Metric-specific labels
        distance_label = {
            "euclidean": "||h(t_i) - h(t_j)||₂",
            "cosine": "1 - cos(h(t_i), h(t_j))",
        }[distance_metric]

        # Trained model
        ax1.scatter(
            spatial_distances, trained_latent_distances, alpha=alpha, s=1, c="blue"
        )
        ax1.set_xlabel("Spatial Distance ||x(t_i) - x(t_j)||₂")
        ax1.set_ylabel(f"Latent Distance {distance_label}")
        ax1.set_title("Trained Model")
        ax1.grid(True, alpha=0.3)

        # Untrained model
        ax2.scatter(
            spatial_distances, untrained_latent_distances, alpha=alpha, s=1, c="red"
        )
        ax2.set_xlabel("Spatial Distance ||x(t_i) - x(t_j)||₂")
        ax2.set_ylabel(f"Latent Distance {distance_label}")
        ax2.set_title("Untrained Model")
        ax2.grid(True, alpha=0.3)

        # Compute correlations
        trained_corr = np.corrcoef(spatial_distances, trained_latent_distances)[0, 1]
        untrained_corr = np.corrcoef(spatial_distances, untrained_latent_distances)[
            0, 1
        ]

        # Add correlation text
        ax1.text(
            0.05,
            0.95,
            f"Correlation: {trained_corr:.3f}",
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        )
        ax2.text(
            0.05,
            0.95,
            f"Correlation: {untrained_corr:.3f}",
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
        )

        population_name = f"MultiTimescaleRNN {self.population_to_visualize}"
        metric_name = distance_metric.capitalize()
        plt.suptitle(
            f"Spatial vs Latent Distance Analysis ({metric_name}) - {population_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\n=== Spatial vs Latent Distance Analysis ({metric_name}) ===")
        print(f"Number of timepoint pairs analyzed: {len(spatial_distances)}")
        print(
            f"Spatial distance range: [{spatial_distances.min():.3f}, {spatial_distances.max():.3f}]"
        )
        print(
            f"Trained latent distance range: [{trained_latent_distances.min():.3f}, {trained_latent_distances.max():.3f}]"
        )
        print(
            f"Untrained latent distance range: [{untrained_latent_distances.min():.3f}, {untrained_latent_distances.max():.3f}]"
        )
        print(f"Trained correlation (spatial vs latent): {trained_corr:.3f}")
        print(f"Untrained correlation (spatial vs latent): {untrained_corr:.3f}")

        if distance_metric == "cosine":
            print(
                "\nNote: Cosine distance measures angular similarity, independent of vector magnitude"
            )

        return self
