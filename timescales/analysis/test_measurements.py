"""
Unit tests for measurement classes.
"""

import pytest
import torch
import lightning as L
from unittest.mock import Mock
from torch.utils.data import DataLoader, TensorDataset

from timescales.analysis.measurements import Measurement, PositionDecodingMeasurement
from timescales.rnns.rnn import RNN
from timescales.rnns.multitimescale_rnn import MultiTimescaleRNN


class TestMeasurementBase:
    """Test the abstract Measurement base class."""

    def test_measurement_is_abstract(self):
        """Test that Measurement cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Measurement(config={})

    def test_measurement_requires_compute_implementation(self):
        """Test that subclasses must implement compute method."""

        class IncompleteMeasurement(Measurement):
            pass

        with pytest.raises(TypeError):
            IncompleteMeasurement(config={})


class TestPositionDecodingMeasurement:
    """Test the PositionDecodingMeasurement class."""

    @pytest.fixture
    def config(self):
        """Basic config for position decoding measurement."""
        return {"decode_k": 3}

    @pytest.fixture
    def measurement(self, config):
        """Create a PositionDecodingMeasurement instance."""
        return PositionDecodingMeasurement(config)

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

    @pytest.fixture
    def mock_rnn_model(self):
        """Create a mock RNN model."""
        mock_model = Mock(spec=RNN)
        mock_model.eval.return_value = None

        # Mock the forward pass to return realistic outputs
        def mock_forward(inputs, place_cells_0):
            batch_size, seq_len, _ = inputs.shape
            num_place_cells = place_cells_0.shape[-1]
            hidden = torch.randn(batch_size, seq_len, 64)  # Mock hidden states
            outputs = torch.randn(
                batch_size, seq_len, num_place_cells
            )  # Mock place cell outputs
            return hidden, outputs

        mock_model.side_effect = mock_forward

        mock_param = torch.tensor([1.0])
        mock_model.parameters.return_value = iter([mock_param])

        return mock_model

    def test_initialization(self, config):
        """Test that PositionDecodingMeasurement initializes correctly."""
        measurement = PositionDecodingMeasurement(config)
        assert measurement.config == config
        assert measurement.decode_k == 3

    def test_initialization_missing_decode_k(self):
        """Test that initialization fails without decode_k in config."""
        with pytest.raises(KeyError):
            PositionDecodingMeasurement({})

    def test_decode_position_from_place_cells(self, measurement, place_cell_centers):
        """Test the position decoding method."""
        batch_size, seq_len, num_place_cells = 2, 5, 5

        # Create activations that work with decode_k=3
        # Make sure the top 3 activations average to the expected position
        activations = torch.zeros(batch_size, seq_len, num_place_cells)

        # For first batch: strongly activate place cells 0, 1, 2 (all near [0,0])
        activations[0, :, 0] = 10.0  # [0.0, 0.0]
        activations[0, :, 1] = 9.0  # [1.0, 0.0]
        activations[0, :, 2] = 8.0  # [0.0, 1.0]
        # Average of top 3: ([0,0] + [1,0] + [0,1])/3 = [0.33, 0.33]

        # For second batch: strongly activate place cell 4 and make others much smaller
        activations[1, :, 4] = 100.0  # [0.5, 0.5] - Much stronger
        activations[1, :, 3] = 1.0  # [1.0, 1.0] (much smaller)
        activations[1, :, 2] = 0.5  # [0.0, 1.0] (much smaller)
        # With 100.0 vs 1.0 vs 0.5, the average will be heavily dominated by place cell 4

        decoded_positions = measurement.decode_position_from_place_cells(
            activations, place_cell_centers
        )

        assert decoded_positions.shape == (batch_size, seq_len, 2)

        # More reasonable expectations for decode_k=3
        # First batch should be around the average of the top 3 place cells
        expected_pos_0 = (
            place_cell_centers[0] + place_cell_centers[1] + place_cell_centers[2]
        ) / 3

        # Second batch: calculate actual expected position with the activations we set
        # Top 3 are: place_cell_4 (100.0), place_cell_3 (1.0), place_cell_2 (0.5)
        expected_pos_1 = (
            place_cell_centers[4] + place_cell_centers[3] + place_cell_centers[2]
        ) / 3

        # Use looser tolerances since we're averaging multiple cells
        torch.testing.assert_close(
            decoded_positions[0].mean(0),  # Average over time
            expected_pos_0,
            atol=0.2,
            rtol=0.2,
        )
        torch.testing.assert_close(
            decoded_positions[1].mean(0),  # Average over time
            expected_pos_1,  # Use the correct expected position
            atol=0.2,
            rtol=0.2,
        )

    def test_decode_position_device_handling(self, measurement):
        """Test that position decoding handles device placement correctly."""
        # Create tensors on CPU
        activations = torch.randn(1, 1, 5)
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

    def test_compute_with_rnn_model(self, measurement, mock_rnn_model, mock_datamodule):
        """Test compute method with an RNN model."""
        error = measurement.compute(mock_rnn_model, mock_datamodule)

        # Should return a float
        assert isinstance(error, float)
        assert error >= 0.0  # Position error should be non-negative

        # Verify that model.eval() was called
        mock_rnn_model.eval.assert_called_once()

        # Verify that datamodule.val_dataloader() was called
        mock_datamodule.val_dataloader.assert_called_once()

    def test_compute_with_multitimescale_model(self, measurement, mock_datamodule):
        """Test compute method with a MultiTimescaleRNN model."""
        mock_model = Mock(spec=MultiTimescaleRNN)
        mock_model.eval.return_value = None

        # Mock the forward pass
        def mock_forward(inputs, place_cells_0):
            batch_size, seq_len, _ = inputs.shape
            num_place_cells = place_cells_0.shape[-1]
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

    def test_compute_with_unknown_model_type(self, measurement, mock_datamodule):
        """Test that compute fails with unknown model types."""
        mock_model = Mock()  # Not an RNN or MultiTimescaleRNN
        # return iterator, not list
        mock_param = torch.tensor([1.0])
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.eval.return_value = None

        with pytest.raises(AssertionError, match="Unknown model type"):
            measurement.compute(mock_model, mock_datamodule)

    def test_decode_k_from_config(self):
        """Test that decode_k is correctly read from config."""
        config_k5 = {"decode_k": 5}
        measurement = PositionDecodingMeasurement(config_k5)
        assert measurement.decode_k == 5

        config_k1 = {"decode_k": 1}
        measurement = PositionDecodingMeasurement(config_k1)
        assert measurement.decode_k == 1

    def test_position_decoding_top_k_logic(self, place_cell_centers):
        """Test that top-k selection works correctly."""
        config = {"decode_k": 2}
        measurement = PositionDecodingMeasurement(config)

        # Create activations where we know which cells should be selected
        activations = torch.tensor(
            [[[1.0, 5.0, 2.0, 8.0, 3.0]]]
        )  # Top 2 should be indices 3, 1

        decoded_pos = measurement.decode_position_from_place_cells(
            activations, place_cell_centers
        )

        # Expected position should be average of place cells 3 and 1
        expected_pos = (place_cell_centers[3] + place_cell_centers[1]) / 2

        torch.testing.assert_close(
            decoded_pos[0, 0], expected_pos, atol=1e-6, rtol=1e-6
        )


class TestIntegration:
    """Integration tests that test the interaction between components."""

    def test_measurement_with_real_tensors(self):
        """Test with more realistic tensor shapes and values."""
        config = {"decode_k": 3}
        measurement = PositionDecodingMeasurement(config)

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
                activations = (
                    torch.exp(-distances * 5) + torch.randn(num_place_cells) * 0.1
                )
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
