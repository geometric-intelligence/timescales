"""
Measurement classes for computing various model performance metrics.

Measurements define what to compute: they take a model and datamodule,
and return a single value.
"""

import torch
from abc import ABC, abstractmethod
import lightning as L

from timescales.rnns.rnn import RNN
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
    Measures position decoding error using top-k place cell activations.

    This measurement computes how well positions can be decoded from the model's
    place cell outputs using a top-k averaging method.
    """

    def __init__(
        self,
        decode_k: int = 3,
    ) -> None:
        """
        Initialize the position decoding measurement.

        Args:
            decode_k: Number of top place cells to use for position decoding
        """
        super().__init__()
        self.decode_k = decode_k

    def decode_position_from_place_cells(
        self,
        activation: torch.Tensor,
        place_cell_centers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode position from place cell activations using top-k method.
        (Standard; see https://github.com/ganguli-lab/grid-pattern-formation/blob/401dd6b5e20a754267b16eeb5bd88239b9af33e9/place_cells.py#L66)

        Args:
            activation: Place cell activations [batch, time, num_place_cells]
            place_cell_centers: Place cell center positions [num_place_cells, 2]

        Returns:
            Decoded positions [batch, time, 2]
        """
        centers = place_cell_centers.to(activation.device)
        _, idxs = torch.topk(activation, k=self.decode_k, dim=-1)  # [B, T, k]
        pred_pos = centers[idxs].mean(-2)  # [B, T, 2]
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

                assert isinstance(model, (RNN, MultiTimescaleRNN)), "Unknown model type"

                _, outputs = model(
                    inputs=inputs,
                    place_cells_0=target_place_cells[:, 0, :],
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
