"""
Null (zero-input) DataModule for random network analysis.

Provides batches of zero inputs so that recurrent dynamics are driven
purely by the initial condition and injected noise. Targets are also
zeros (unused). This allows running random networks through the same
training/inference pipeline used for task-driven networks.
"""

import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch


class NullDataModule(L.LightningDataModule):
    def __init__(
        self,
        input_size: int = 1,
        num_time_steps: int = 5000,
        num_trajectories: int = 100,
        batch_size: int = 10,
        num_workers: int = 0,
        train_val_split: float = 0.8,
    ) -> None:
        """
        :param input_size: Dimension of the (zero) input vector.
        :param num_time_steps: Sequence length per trajectory.
        :param num_trajectories: Total number of trajectories.
        :param batch_size: Batch size for dataloaders.
        :param num_workers: Number of workers for data loading.
        :param train_val_split: Fraction of data used for training.
        """
        super().__init__()
        self._input_size = input_size
        self.num_time_steps = num_time_steps
        self.num_trajectories = num_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

    def setup(self, stage=None) -> None:
        B, T, C = self.num_trajectories, self.num_time_steps, self._input_size

        inputs = torch.zeros(B, T, C)
        aux_info = torch.zeros(B, T, 1)
        targets = torch.zeros(B, T, 1)

        full_dataset = TensorDataset(inputs, aux_info, targets)
        train_size = int(self.train_val_split * B)
        val_size = B - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return 1
