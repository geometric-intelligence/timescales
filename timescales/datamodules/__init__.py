"""
DataModules for timescales experiments.

Available datamodules:
- PathIntegrationDataModule: Path integration task with place cell outputs (2D)
- PathIntegration1DDataModule: Simplified 1D path integration for theory validation
- HierarchicalCounterDataModule: Hierarchical binary counter task
- FlipFlopDataModule: N-bit flip-flop memory task
- NullDataModule: Zero-input sequences for random network analysis
- TeacherStudentDataModule: Teacher-student regression task
- SineWaveDataModule: Autonomous sine-wave generation task
"""

from .path_integration import PathIntegrationDataModule
from .path_integration_1d import PathIntegration1DDataModule
from .binary_counter import HierarchicalCounterDataModule
from .flip_flop import FlipFlopDataModule
from .null_task import NullDataModule
from .teacher_student import TeacherStudentDataModule
from .sine_wave import SineWaveDataModule

__all__ = [
    "PathIntegrationDataModule",
    "PathIntegration1DDataModule",
    "HierarchicalCounterDataModule",
    "FlipFlopDataModule",
    "NullDataModule",
    "TeacherStudentDataModule",
    "SineWaveDataModule",
]

