"""
Analysis classes for evaluating models under varying test conditions.

Analyses define how to vary test conditions: they take a model, config, and measurement,
then apply the measurement across different conditions (e.g., varying trajectory lengths).

Note: In a sweep, each seed has its own config and place_cell_centers, so Analysis
instances should be created fresh for each seed.
"""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
import lightning as L

from timescales.datamodules import PathIntegrationDataModule
from .measurements import Measurement


@dataclass
class AnalysisResult:
    """Container for single model analysis results."""

    test_conditions: list  # e.g., [25, 50, 100] for trajectory lengths
    measurements: list[float]  # Measurement value for each condition
    condition_name: str  # e.g., "trajectory_length"
    metadata: dict | None = None  # Optional metadata about the analysis


class Analysis(ABC):
    """
    Abstract base class for all analyses.

    An analysis varies test conditions and applies a measurement to each.

    Note: Analysis instances are tied to a specific config and place_cell_centers,
    so in a sweep context, create fresh instances for each seed.
    """

    def __init__(self, config: dict, place_cell_centers: torch.Tensor):
        """
        Initialize the analysis.

        Args:
            config: Model configuration dictionary (from training)
            place_cell_centers: Place cell centers tensor [num_place_cells, 2]
        """
        self.config = config
        self.place_cell_centers = place_cell_centers

    @abstractmethod
    def run(
        self,
        model: torch.nn.Module,
        measurement: Measurement,
    ) -> AnalysisResult:
        """
        Run analysis on a single model using the given measurement.

        Args:
            model: Model to analyze
            measurement: Measurement to apply under different conditions

        Returns:
            AnalysisResult containing measurements across all test conditions
        """
        pass


class OODAnalysis(Analysis):
    """
    Out-of-distribution generalization analysis.

    Tests model performance on trajectory lengths longer than those seen during training.
    Varies sequence length and measures how well the model generalizes.
    """

    def __init__(
        self,
        config: dict,
        place_cell_centers: torch.Tensor,
        test_lengths: list[int],
        num_test_trajectories: int = 100,
    ):
        """
        Initialize OOD analysis.

        Args:
            config: Base configuration (from training)
            place_cell_centers: Place cell centers from training [num_place_cells, 2]
            test_lengths: List of trajectory lengths to test (in time steps)
            num_test_trajectories: Number of test trajectories per length
        """
        super().__init__(config, place_cell_centers)
        self.test_lengths = test_lengths
        self.num_test_trajectories = num_test_trajectories
        self.training_length = config["num_time_steps"]

    def _create_test_datamodule(
        self,
        test_length: int,
    ) -> L.LightningDataModule:
        """Create a datamodule with specified trajectory length."""
        test_config = self.config.copy()
        test_config.update(
            {
                "num_time_steps": test_length,
                "num_trajectories": self.num_test_trajectories,
                "batch_size": min(
                    self.config["batch_size"], self.num_test_trajectories
                ),
            }
        )

        datamodule = PathIntegrationDataModule(
            num_trajectories=test_config["num_trajectories"],
            batch_size=test_config["batch_size"],
            num_workers=1,  # Reduce workers for test
            train_val_split=0.0,  # All data goes to validation
            velocity_representation=test_config["velocity_representation"],
            dt=test_config["dt"],
            num_time_steps=test_config["num_time_steps"],
            arena_size=test_config["arena_size"],
            speed_scale=test_config["speed_scale"],
            sigma_speed=test_config["sigma_speed"],
            tau_vel=test_config["tau_vel"],
            sigma_rotation=test_config["sigma_rotation"],
            border_region=test_config["border_region"],
            num_place_cells=test_config["num_place_cells"],
            place_cell_rf=test_config["place_cell_rf"],
            surround_scale=test_config["surround_scale"],
            DoG=test_config["DoG"],
            trajectory_type=test_config["trajectory_type"],
            place_cell_layout=test_config["place_cell_layout"],
        )

        datamodule.place_cell_centers = self.place_cell_centers

        # IMPORTANT: set the place cell centers before calling setup
        datamodule.setup()

        return datamodule

    def run(
        self,
        model: torch.nn.Module,
        measurement: Measurement,
    ) -> AnalysisResult:
        """
        Run OOD analysis by testing on different trajectory lengths.

        Args:
            model: Model to analyze
            measurement: Measurement to compute (e.g., position decoding error)

        Returns:
            AnalysisResult with measurements for each trajectory length
        """
        measurements = []

        for test_length in self.test_lengths:
            # Create datamodule with this trajectory length
            test_datamodule = self._create_test_datamodule(test_length)

            # Compute measurement on this test condition
            error = measurement.compute(model, test_datamodule)
            measurements.append(error)

        return AnalysisResult(
            test_conditions=self.test_lengths,
            measurements=measurements,
            condition_name="trajectory_length",
            metadata={
                "training_length": self.training_length,
                "num_test_trajectories": self.num_test_trajectories,
            },
        )


class NoiseRobustnessAnalysis(Analysis):
    """
    Noise robustness analysis (placeholder for future implementation).

    Tests model performance under varying levels of input noise.
    """

    def __init__(
        self,
        config: dict,
        place_cell_centers: torch.Tensor,
        noise_levels: list[float],
    ):
        """
        Initialize noise robustness analysis.

        Args:
            config: Base configuration
            place_cell_centers: Place cell centers from training
            noise_levels: List of noise standard deviations to test
        """
        super().__init__(config, place_cell_centers)
        self.noise_levels = noise_levels

    def run(
        self,
        model: torch.nn.Module,
        measurement: Measurement,
    ) -> AnalysisResult:
        """Run noise robustness analysis."""
        # TODO: Implement noise injection and measurement
        raise NotImplementedError("NoiseRobustnessAnalysis not yet implemented")
