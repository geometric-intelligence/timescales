"""
DataModules for timescales experiments.

Available datamodules:
- PathIntegrationDataModule: Path integration task with place cell outputs
- HierarchicalCounterDataModule: Hierarchical binary counter task
"""

from .path_integration import PathIntegrationDataModule
from .binary_counter import HierarchicalCounterDataModule

__all__ = [
    "PathIntegrationDataModule",
    "HierarchicalCounterDataModule",
]

