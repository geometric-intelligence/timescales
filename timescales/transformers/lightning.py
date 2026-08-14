"""Lightning wrapper for transformer-timescale sequence models."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn


class TransformerSequenceLightning(L.LightningModule):
    """Train a causal sequence model with component-wise MSE diagnostics."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float = 0.0,
        optimizer_name: str = "adam",
        use_lr_scheduler: bool = False,
        lr_step_size: int = 1000,
        lr_scheduler_gamma: float = 1.0,
        sgld_beta: float = 2000.0,
        sgld_add_noise: bool = True,
        clark_loss_scaling: bool = True,
        prediction_mode: str = "last",
    ) -> None:
        super().__init__()
        optimizer_name = optimizer_name.lower()
        if optimizer_name not in {"adam", "sgld"}:
            raise ValueError("optimizer_name must be 'adam' or 'sgld'")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = optimizer_name
        self.use_lr_scheduler = bool(use_lr_scheduler)
        self.lr_step_size = int(lr_step_size)
        self.lr_scheduler_gamma = float(lr_scheduler_gamma)
        self.sgld_beta = float(sgld_beta)
        self.sgld_add_noise = bool(sgld_add_noise)
        self.clark_loss_scaling = bool(clark_loss_scaling)
        if prediction_mode not in {"all", "last"}:
            raise ValueError("prediction_mode must be 'all' or 'last'")
        self.prediction_mode = prediction_mode

    @property
    def task_loss_scale(self) -> float:
        gamma = getattr(self.model, "output_coupling_gamma", None)
        if gamma is None or not self.clark_loss_scaling:
            return 1.0
        return 0.5 * self.model.d_model * gamma**2

    def _losses(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        per_channel = ((outputs - targets) ** 2).mean(dim=(0, 1))
        return per_channel.mean(), per_channel

    def _select_predictions(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prediction_mode == "last":
            return outputs[:, -1:, :], targets[:, -1:, :]
        return outputs, targets

    def _objective(self, task_loss: torch.Tensor) -> torch.Tensor:
        objective = self.task_loss_scale * task_loss
        if self.weight_decay:
            squared_norm = sum(
                (parameter**2).sum() for parameter in self.model.parameters()
            )
            objective = objective + self.weight_decay * squared_norm
        return objective

    def _log_batch(self, prefix: str, batch) -> torch.Tensor:
        inputs, _aux_info, targets = batch
        outputs = self.model(inputs)
        outputs, targets = self._select_predictions(outputs, targets)
        task_loss, per_channel = self._losses(outputs, targets)
        objective = self._objective(task_loss)
        on_step = prefix == "train"
        self.log(
            f"{prefix}_loss",
            task_loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_objective",
            objective,
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
        )
        for index, loss in enumerate(per_channel):
            self.log(
                f"{prefix}_loss_channel_{index}",
                loss,
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )

        sign_accuracy = (
            (torch.sign(outputs) == torch.sign(targets)).float().mean(dim=(0, 1))
        )
        self.log(
            f"{prefix}_accuracy",
            sign_accuracy.mean(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for index, accuracy in enumerate(sign_accuracy):
            self.log(
                f"{prefix}_accuracy_channel_{index}",
                accuracy,
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )
        return objective if prefix == "train" else task_loss

    def training_step(self, batch, batch_idx=0) -> torch.Tensor:
        return self._log_batch("train", batch)

    def validation_step(self, batch, batch_idx=0) -> torch.Tensor:
        return self._log_batch("val", batch)

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.0,
            )
        else:
            from timescales.optimizers import SGLD

            optimizer = SGLD(
                self.model.parameters(),
                lr=self.learning_rate,
                beta=self.sgld_beta,
                add_noise=self.sgld_add_noise,
            )
        if not self.use_lr_scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_step_size,
            gamma=self.lr_scheduler_gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
