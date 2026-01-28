"""
Measurement classes for computing various model performance metrics.

Measurements define what to compute: they take a model and datamodule,
and return a single value.
"""

import torch
from abc import ABC, abstractmethod
import lightning as L

from timescales.rnns.multitimescale_rnn import MultiTimescaleRNN


class Measurement(ABC):
    """
    Abstract base class for all measurements.

    A measurement computes a single value from a given model and dataset.
    Measurement instances should only store measurement-specific parameters,
    not model-specific config (which varies per seed in a sweep).
    """

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """
        Initialize the measurement.

        Args:
            **kwargs: Measurement-specific parameters
        """
        pass

    @abstractmethod
    def compute(
        self,
        model: torch.nn.Module,
        datamodule: L.LightningDataModule,
    ) -> float:
        """
        Compute the measurement for a given model and dataset (datamodule).

        Args:
            model: model to evaluate (PyTorch module)
            datamodule: DataModule with test data

        Returns:
            Single float value representing the computed metric
        """
        pass


class PositionDecodingMeasurement(Measurement):
    """
    Measures position decoding error using weighted sum of place cell activations.

    This measurement computes how well positions can be decoded from the model's
    place cell outputs using activations as weights for place cell centers.
    """

    def __init__(self) -> None:
        """Initialize the position decoding measurement."""
        super().__init__()

    def decode_position_from_place_cells(
        self,
        activation: torch.Tensor,
        place_cell_centers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode position from place cell activations using weighted sum.
        
        Uses softmax activations as weights to compute weighted average of
        place cell center positions.

        Args:
            activation: Place cell activations [batch, time, num_place_cells]
                       (should be softmaxed, i.e., sum to 1 along last dim)
            place_cell_centers: Place cell center positions [num_place_cells, 2]

        Returns:
            Decoded positions [batch, time, 2]
        """
        centers = place_cell_centers.to(activation.device)
        # Weighted sum: activation @ centers
        # activation: [B, T, N], centers: [N, 2] -> [B, T, 2]
        pred_pos = torch.einsum('btn,nd->btd', activation, centers)
        return pred_pos

    def compute(
        self,
        model: torch.nn.Module,
        datamodule: L.LightningDataModule,
    ) -> float:
        """
        Compute position decoding error.

        Args:
            model: Trained model to evaluate
            datamodule: DataModule containing test trajectories

        Returns:
            Mean L2 position decoding error in meters
        """
        model.eval()
        total_error = 0.0
        total_samples = 0

        # Get device from model
        model_device = next(model.parameters()).device
        dataloader = datamodule.val_dataloader()
        place_cell_centers = datamodule.place_cell_centers.to(model_device)

        with torch.no_grad():
            for batch in dataloader:
                inputs, target_positions, target_place_cells = batch

                inputs = inputs.to(model_device)  # [B, T, 2]
                target_positions = target_positions.to(model_device)  # [B, T, 2]
                target_place_cells = target_place_cells.to(
                    model_device
                )  # [B, T, num_place_cells]

                _, outputs = model(
                    inputs=inputs,
                    init_context=target_place_cells[:, 0, :],
                )

                place_cell_probs = torch.softmax(outputs, dim=-1)
                predicted_positions = self.decode_position_from_place_cells(
                    place_cell_probs,
                    place_cell_centers,
                )

                position_error = torch.sqrt(
                    ((target_positions - predicted_positions) ** 2).sum(-1),
                )

                total_error += position_error.sum().item()
                total_samples += position_error.numel()

        return total_error / total_samples
