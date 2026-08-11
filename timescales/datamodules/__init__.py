"""
DataModules for timescales experiments.

Available datamodules:
- FlipFlopDataModule: N-bit flip-flop memory task
- SignedFlipFlopDataModule: N-bit flip-flop with signed -1/+1 targets
- SineWaveDataModule: Autonomous sine-wave generation task
- CumulativeVectorAdditionDataModule: Running sum of vector inputs
- NullDataModule: Zero-input sequences for random/untrained network analysis
"""

from .flip_flop import FlipFlopDataModule
from .signed_flip_flop import SignedFlipFlopDataModule
from .sine_wave import SineWaveDataModule
from .cumulative_vector_addition import CumulativeVectorAdditionDataModule
from .null_task import NullDataModule

__all__ = [
    "FlipFlopDataModule",
    "SignedFlipFlopDataModule",
    "SineWaveDataModule",
    "CumulativeVectorAdditionDataModule",
    "NullDataModule",
]
