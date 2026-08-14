"""Tests for preserving loss-history prefixes in checkpoint continuations."""

import json

from timescales.callbacks import LossLoggerCallback


def test_loss_logger_loads_an_initial_curve_without_aliasing(tmp_path):
    initial = {
        "steps": [10, 20],
        "train_losses": [0.9],
        "val_losses": [0.8, 0.7],
        "train_objectives": [2.0],
        "val_objectives": [1.8, 1.6],
        "train_accuracies": [0.6],
        "val_accuracies": [0.65, 0.7],
        "val_losses_per_bit": {"channel_0": [0.8, 0.7]},
        "val_accuracies_per_bit": {"channel_0": [0.65, 0.7]},
    }
    curve_path = tmp_path / "initial.json"
    curve_path.write_text(json.dumps(initial))

    callback = LossLoggerCallback(
        save_dir=str(tmp_path / "continued"),
        initial_curve_path=str(curve_path),
    )

    assert callback.steps == [10, 20]
    assert callback.val_losses == [0.8, 0.7]
    assert callback.val_losses_per_bit == {"channel_0": [0.8, 0.7]}

    callback.val_losses_per_bit["channel_0"].append(0.6)
    assert initial["val_losses_per_bit"]["channel_0"] == [0.8, 0.7]
