"""
Unit tests for measurement classes.
"""

import pytest
import torch
import lightning as L
from unittest.mock import Mock
from torch.utils.data import DataLoader, TensorDataset

from timescales.analysis.measurements import Measurement, PositionDecodingMeasurement
from timescales.rnns.multitimescale_rnn import MultiTimescaleRNN


class TestMeasurementBase:
    """Test the abstract Measurement base class."""

    def test_measurement_is_abstract(self):
        """Test that Measurement cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Measurement()

    def test_measurement_requires_compute_implementation(self):
        """Test that subclasses must implement compute method."""

        class IncompleteMeasurement(Measurement):
            pass

        with pytest.raises(TypeError):
            IncompleteMeasurement()


class TestPositionDecodingMeasurement:
    """Test the PositionDecodingMeasurement class."""

    @pytest.fixture
    def measurement(self):
        """Create a PositionDecodingMeasurement instance."""
        return PositionDecodingMeasurement()

    @pytest.fixture
    def place_cell_centers(self):
        """Create dummy place cell centers."""
        return torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
            dtype=torch.float32,
        )

    @pytest.fixture
    def mock_datamodule(self, place_cell_centers):
        """Create a mock Lightning DataModule."""
        # Create dummy data
        batch_size = 2
        seq_len = 10
        num_place_cells = 5

        inputs = torch.randn(batch_size, seq_len, 2)
        target_positions = torch.randn(batch_size, seq_len, 2)
        target_place_cells = torch.softmax(
            torch.randn(batch_size, seq_len, num_place_cells), dim=-1
        )

        dataset = TensorDataset(inputs, target_positions, target_place_cells)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        mock_dm = Mock(spec=L.LightningDataModule)
        mock_dm.val_dataloader.return_value = dataloader
        mock_dm.place_cell_centers = place_cell_centers

        return mock_dm

    def test_initialization(self):
        """Test that PositionDecodingMeasurement initializes correctly."""
        measurement = PositionDecodingMeasurement()
        assert measurement is not None

    def test_decode_position_weighted_sum(self, measurement, place_cell_centers):
        """Test the position decoding method using weighted sum."""
        batch_size, seq_len, num_place_cells = 2, 5, 5

        # Create activations that are already softmaxed
        # If we put all weight on one place cell, we should get that cell's center
        activations = torch.zeros(batch_size, seq_len, num_place_cells)
        
        # First batch: all weight on place cell 0 at [0.0, 0.0]
        activations[0, :, 0] = 1.0
        
        # Second batch: all weight on place cell 4 at [0.5, 0.5]
        activations[1, :, 4] = 1.0

        decoded_positions = measurement.decode_position_from_place_cells(
            activations, place_cell_centers
        )

        assert decoded_positions.shape == (batch_size, seq_len, 2)

        # First batch should be at [0.0, 0.0]
        torch.testing.assert_close(
            decoded_positions[0, 0],
            place_cell_centers[0],
            atol=1e-6,
            rtol=1e-6,
        )
        
        # Second batch should be at [0.5, 0.5]
        torch.testing.assert_close(
            decoded_positions[1, 0],
            place_cell_centers[4],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_decode_position_uniform_weights(self, measurement, place_cell_centers):
        """Test decoding with uniform weights gives center of all cells."""
        batch_size, seq_len, num_place_cells = 1, 1, 5
        
        # Uniform weights (sum to 1)
        activations = torch.ones(batch_size, seq_len, num_place_cells) / num_place_cells
        
        decoded_positions = measurement.decode_position_from_place_cells(
            activations, place_cell_centers
        )
        
        # Should be the mean of all place cell centers
        expected_center = place_cell_centers.mean(dim=0)
        torch.testing.assert_close(
            decoded_positions[0, 0],
            expected_center,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_decode_position_device_handling(self, measurement):
        """Test that position decoding handles device placement correctly."""
        # Create tensors on CPU
        activations = torch.randn(1, 1, 5)
        activations = torch.softmax(activations, dim=-1)  # Normalize
        place_cell_centers = torch.randn(5, 2)

        # Test CPU computation
        result_cpu = measurement.decode_position_from_place_cells(
            activations, place_cell_centers
        )
        assert result_cpu.device == torch.device("cpu")

        # Test GPU computation if available
        if torch.cuda.is_available():
            activations_gpu = activations.cuda()
            result_gpu = measurement.decode_position_from_place_cells(
                activations_gpu, place_cell_centers
            )
            assert result_gpu.device.type == "cuda"

    def test_compute_with_multitimescale_model(self, measurement, mock_datamodule):
        """Test compute method with a MultiTimescaleRNN model."""
        mock_model = Mock(spec=MultiTimescaleRNN)
        mock_model.eval.return_value = None

        # Mock the forward pass
        def mock_forward(inputs, init_context):
            batch_size, seq_len, _ = inputs.shape
            num_place_cells = init_context.shape[-1]
            hidden = torch.randn(batch_size, seq_len, 64)
            outputs = torch.randn(batch_size, seq_len, num_place_cells)
            return hidden, outputs

        # Use side_effect to make the mock callable and return tuple
        mock_model.side_effect = mock_forward
        # return iterator, not list
        mock_param = torch.tensor([1.0])
        mock_model.parameters.return_value = iter([mock_param])

        error = measurement.compute(mock_model, mock_datamodule)

        assert isinstance(error, float)
        assert error >= 0.0
        mock_model.eval.assert_called_once()


class TestIntegration:
    """Integration tests that test the interaction between components."""

    def test_measurement_with_real_tensors(self):
        """Test with more realistic tensor shapes and values."""
        measurement = PositionDecodingMeasurement()

        # Create realistic place cell centers (grid layout)
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        place_cell_centers = torch.stack(
            [xx.flatten(), yy.flatten()], dim=-1
        )  # [25, 2]

        # Create mock data that resembles actual trajectories
        batch_size, seq_len = 3, 20
        num_place_cells = 25

        inputs = torch.randn(batch_size, seq_len, 2) * 0.5 + 0.5  # Positions in [0, 1]
        target_positions = inputs + torch.randn_like(inputs) * 0.01  # Small noise

        # Create place cell activations based on distance to centers
        target_place_cells = torch.zeros(batch_size, seq_len, num_place_cells)
        for b in range(batch_size):
            for t in range(seq_len):
                pos = target_positions[b, t]
                distances = torch.norm(place_cell_centers - pos, dim=1)
                # Activate cells based on inverse distance (with some noise)
                activations = torch.exp(-distances * 5)
                target_place_cells[b, t] = torch.softmax(activations, dim=0)

        # Test the decoding
        decoded_positions = measurement.decode_position_from_place_cells(
            target_place_cells, place_cell_centers
        )

        # Decoded positions should be reasonably close to target positions
        errors = torch.norm(decoded_positions - target_positions, dim=-1)
        mean_error = errors.mean().item()

        # With good place cell activations, error should be small
        assert mean_error < 0.5, f"Mean decoding error {mean_error} is too high"
        assert decoded_positions.shape == target_positions.shape
