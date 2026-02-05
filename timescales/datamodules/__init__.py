"""
DataModules for timescales experiments.

Available datamodules:
- PathIntegrationDataModule: Path integration task with place cell outputs (2D)
- PathIntegration1DDataModule: Simplified 1D path integration for theory validation
- HierarchicalCounterDataModule: Hierarchical binary counter task
"""

from .path_integration import PathIntegrationDataModule
from .path_integration_1d import PathIntegration1DDataModule
from .binary_counter import HierarchicalCounterDataModule

__all__ = [
    "PathIntegrationDataModule",
    "PathIntegration1DDataModule",
    "HierarchicalCounterDataModule",
]

