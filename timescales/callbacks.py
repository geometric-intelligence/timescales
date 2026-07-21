import lightning as L
import json
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np
from timescales.analysis.measurements import PositionDecodingMeasurement


class LossLoggerCallback(L.Callback):
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_accuracies: list[float] = []
        self.steps: list[int] = []

        self.val_losses_per_bit: dict[str, list[float]] = {}
        self.val_accuracies_per_bit: dict[str, list[float]] = {}

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.logged_metrics.get("train_loss_epoch", None)
        if train_loss is not None:
            self.train_losses.append(float(train_loss))

        train_acc = trainer.logged_metrics.get("train_accuracy_epoch", None)
        if train_acc is not None:
            self.train_accuracies.append(float(train_acc))

    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.sanity_checking:
            return

        val_loss = trainer.logged_metrics.get("val_loss", None)
        if val_loss is not None:
            self.val_losses.append(float(val_loss))
            self.steps.append(trainer.global_step)

        val_acc = trainer.logged_metrics.get("val_accuracy", None)
        if val_acc is not None:
            self.val_accuracies.append(float(val_acc))

        for key, value in trainer.logged_metrics.items():
            if key.startswith("val_loss_channel_"):
                ch = key.removeprefix("val_loss_")
                self.val_losses_per_bit.setdefault(ch, []).append(float(value))
            elif key.startswith("val_accuracy_channel_"):
                ch = key.removeprefix("val_accuracy_")
                self.val_accuracies_per_bit.setdefault(ch, []).append(float(value))

        self._save_losses()

    @rank_zero_only
    def _save_losses(self):
        os.makedirs(self.save_dir, exist_ok=True)

        loss_data = {
            "steps": self.steps,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "val_losses_per_bit": self.val_losses_per_bit,
            "val_accuracies_per_bit": self.val_accuracies_per_bit,
        }

        with open(os.path.join(self.save_dir, "training_losses.json"), "w") as f:
            json.dump(loss_data, f, indent=2)


class PositionDecodingCallback(L.Callback):
    """Callback to compute and log position decoding error during validation."""

    def __init__(
        self,
        measurement: PositionDecodingMeasurement,
        datamodule: L.LightningDataModule,
        log_every_n_epochs: int = 1,
        save_dir: str = None,  # Add save_dir parameter
    ):
        super().__init__()
        self.measurement = measurement
        self.datamodule = datamodule
        self.log_every_n_epochs = log_every_n_epochs
        self.save_dir = save_dir

        self.position_errors_epoch: list[float] = []
        self.epochs: list[int] = []

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Compute position decoding error at the end of each validation epoch."""

        if trainer.sanity_checking:
            return

        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        position_error = self.measurement.compute(pl_module.model, self.datamodule)

        pl_module.log(
            "val_position_error",
            position_error,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.position_errors_epoch.append(position_error)
        self.epochs.append(trainer.current_epoch)

        if self.save_dir is not None:
            self._save_position_errors()

    @rank_zero_only
    def _save_position_errors(self):
        """Save position decoding errors to JSON file."""

        os.makedirs(self.save_dir, exist_ok=True)

        error_data = {
            "epochs": self.epochs,
            "position_errors_epoch": self.position_errors_epoch,
        }

        with open(
            os.path.join(self.save_dir, "position_decoding_errors.json"), "w"
        ) as f:
            json.dump(error_data, f, indent=2)


class TrajectoryVisualizationCallback(L.Callback):
    """Callback to visualize and log trajectory predictions to wandb."""

    def __init__(
        self,
        place_cell_centers: torch.Tensor,
        arena_size: float,
        log_every_n_epochs: int = 5,
        num_trajectories_to_plot: int = 3,
    ):
        super().__init__()
        self.place_cell_centers = place_cell_centers
        self.arena_size = arena_size
        self.log_every_n_epochs = log_every_n_epochs
        self.num_trajectories_to_plot = num_trajectories_to_plot

    def decode_position_from_place_cells(
        self, activation: torch.Tensor
    ) -> torch.Tensor:
        """Decode position from place cell activations using weighted sum."""
        centers = self.place_cell_centers.to(activation.device)
        # Weighted sum: activation @ centers
        pred_pos = torch.einsum('btn,nd->btd', activation, centers)
        return pred_pos

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # Only run every N epochs
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Get a validation batch
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        # Get first batch from validation set
        batch = next(iter(val_dataloader))
        inputs, target_positions, target_place_cells = batch

        # Move to correct device
        inputs = inputs.to(pl_module.device)
        target_positions = target_positions.to(pl_module.device)
        target_place_cells = target_place_cells.to(pl_module.device)

        # Get model predictions
        with torch.no_grad():
            _, outputs = pl_module.model(
                inputs=inputs, init_context=target_place_cells[:, 0, :]
            )

            # Convert to probabilities and decode positions
            place_cell_probs = torch.softmax(outputs, dim=-1)
            predicted_positions = self.decode_position_from_place_cells(
                place_cell_probs
            )

        # Set consistent axis limits based on arena size (centered at origin)
        lim = self.arena_size / 2
        xlim = [-lim, lim]
        ylim = [-lim, lim]

        # Create plots for first few trajectories
        figs = []
        for i in range(min(self.num_trajectories_to_plot, inputs.shape[0])):
            # Create figure with 5 subplots: trajectory + 4 place cell plots
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))

            # === TRAJECTORY PLOT ===
            ax_traj = axes[0]

            # Ground truth trajectory
            gt_traj = target_positions[i].cpu().numpy()
            pred_traj = predicted_positions[i].cpu().numpy()

            # Plot trajectories
            ax_traj.plot(
                gt_traj[:, 0],
                gt_traj[:, 1],
                "b-",
                linewidth=3,
                label="Ground Truth",
                alpha=0.8,
            )
            ax_traj.plot(
                pred_traj[:, 0],
                pred_traj[:, 1],
                "r--",
                linewidth=3,
                label="Predicted",
                alpha=0.8,
            )

            # Mark start and end points
            ax_traj.scatter(
                gt_traj[0, 0],
                gt_traj[0, 1],
                c="green",
                s=150,
                marker="o",
                label="Start",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )
            ax_traj.scatter(
                gt_traj[-1, 0],
                gt_traj[-1, 1],
                c="red",
                s=150,
                marker="s",
                label="End",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )

            # Mark all points to see trajectory detail
            ax_traj.scatter(
                gt_traj[:, 0],
                gt_traj[:, 1],
                c=range(len(gt_traj)),
                cmap="plasma",
                s=30,
                alpha=0.7,
                zorder=3,
            )

            # Mark predicted trajectory points with different colormap
            ax_traj.scatter(
                pred_traj[:, 0],
                pred_traj[:, 1],
                c=range(len(pred_traj)),
                cmap="spring",
                s=30,
                alpha=0.7,
                zorder=3,
                marker="^",
            )

            ax_traj.set_xlabel("X Position")
            ax_traj.set_ylabel("Y Position")
            ax_traj.set_title("Trajectory Comparison")
            ax_traj.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=4)
            ax_traj.grid(True, alpha=0.3)
            ax_traj.set_aspect("equal")
            ax_traj.set_xlim(xlim)
            ax_traj.set_ylim(ylim)

            # Calculate and display error
            error = torch.sqrt(
                ((target_positions[i] - predicted_positions[i]) ** 2).sum(-1)
            ).mean()
            ax_traj.text(
                0.02,
                0.98,
                f"Mean Error: {error:.4f}",
                transform=ax_traj.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # === PLACE CELL ACTIVATION PLOTS ===
            place_centers = self.place_cell_centers.cpu().numpy()

            # START TIME - Ground Truth
            ax_start_gt = axes[1]
            start_gt_activations = target_place_cells[i, 0, :].cpu().numpy()  # t=0
            start_pos = target_positions[i, 0].cpu().numpy()

            scatter_start_gt = ax_start_gt.scatter(
                place_centers[:, 0],
                place_centers[:, 1],
                c=start_gt_activations,
                cmap="viridis",
                s=30,
                alpha=0.8,
            )
            # Mark agent position
            ax_start_gt.scatter(
                start_pos[0],
                start_pos[1],
                c="green",
                s=200,
                marker="*",
                label="Agent",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )

            ax_start_gt.set_xlabel("X Position")
            ax_start_gt.set_ylabel("Y Position")
            ax_start_gt.set_title(
                f"START: GT Place Cells\n({start_pos[0]:.3f}, {start_pos[1]:.3f})"
            )
            ax_start_gt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center")
            ax_start_gt.grid(True, alpha=0.3)
            ax_start_gt.set_aspect("equal")
            ax_start_gt.set_xlim(xlim)
            ax_start_gt.set_ylim(ylim)
            cbar_start_gt = plt.colorbar(scatter_start_gt, ax=ax_start_gt)
            cbar_start_gt.set_label("Activation")

            # START TIME - Predicted
            ax_start_pred = axes[2]
            start_pred_activations = place_cell_probs[i, 0, :].cpu().numpy()  # t=0

            # Decode position from predicted activations
            start_decoded_pos = (
                self.decode_position_from_place_cells(
                    place_cell_probs[i : i + 1, 0:1, :]  # Keep batch and time dims
                )[0, 0]
                .cpu()
                .numpy()
            )  # Extract single position

            scatter_start_pred = ax_start_pred.scatter(
                place_centers[:, 0],
                place_centers[:, 1],
                c=start_pred_activations,
                cmap="viridis",
                s=30,
                alpha=0.8,
            )
            # Mark DECODED agent position
            ax_start_pred.scatter(
                start_decoded_pos[0],
                start_decoded_pos[1],
                c="green",
                s=200,
                marker="*",
                label="Decoded Pos",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )

            ax_start_pred.set_xlabel("X Position")
            ax_start_pred.set_ylabel("Y Position")
            ax_start_pred.set_title(
                f"START: Predicted Place Cells\n({start_decoded_pos[0]:.3f}, {start_decoded_pos[1]:.3f})"
            )
            ax_start_pred.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center")
            ax_start_pred.grid(True, alpha=0.3)
            ax_start_pred.set_aspect("equal")
            ax_start_pred.set_xlim(xlim)
            ax_start_pred.set_ylim(ylim)
            cbar_start_pred = plt.colorbar(scatter_start_pred, ax=ax_start_pred)
            cbar_start_pred.set_label("Activation")

            # END TIME - Ground Truth
            ax_end_gt = axes[3]
            end_gt_activations = target_place_cells[i, -1, :].cpu().numpy()  # t=final
            end_pos = target_positions[i, -1].cpu().numpy()

            scatter_end_gt = ax_end_gt.scatter(
                place_centers[:, 0],
                place_centers[:, 1],
                c=end_gt_activations,
                cmap="viridis",
                s=30,
                alpha=0.8,
            )
            # Mark agent position
            ax_end_gt.scatter(
                end_pos[0],
                end_pos[1],
                c="red",
                s=200,
                marker="*",
                label="Agent",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )

            ax_end_gt.set_xlabel("X Position")
            ax_end_gt.set_ylabel("Y Position")
            ax_end_gt.set_title(
                f"END: GT Place Cells\n({end_pos[0]:.3f}, {end_pos[1]:.3f})"
            )
            ax_end_gt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center")
            ax_end_gt.grid(True, alpha=0.3)
            ax_end_gt.set_aspect("equal")
            ax_end_gt.set_xlim(xlim)
            ax_end_gt.set_ylim(ylim)
            cbar_end_gt = plt.colorbar(scatter_end_gt, ax=ax_end_gt)
            cbar_end_gt.set_label("Activation")

            # END TIME - Predicted
            ax_end_pred = axes[4]
            end_pred_activations = place_cell_probs[i, -1, :].cpu().numpy()  # t=final

            # Decode position from predicted activations
            end_decoded_pos = (
                self.decode_position_from_place_cells(
                    place_cell_probs[i : i + 1, -1:, :]  # Keep batch and time dims
                )[0, 0]
                .cpu()
                .numpy()
            )  # Extract single position

            scatter_end_pred = ax_end_pred.scatter(
                place_centers[:, 0],
                place_centers[:, 1],
                c=end_pred_activations,
                cmap="viridis",
                s=30,
                alpha=0.8,
            )
            # Mark DECODED agent position
            ax_end_pred.scatter(
                end_decoded_pos[0],
                end_decoded_pos[1],
                c="red",
                s=200,
                marker="*",
                label="Decoded Pos",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )

            ax_end_pred.set_xlabel("X Position")
            ax_end_pred.set_ylabel("Y Position")
            ax_end_pred.set_title(
                f"END: Predicted Place Cells\n({end_decoded_pos[0]:.3f}, {end_decoded_pos[1]:.3f})"
            )
            ax_end_pred.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center")
            ax_end_pred.grid(True, alpha=0.3)
            ax_end_pred.set_aspect("equal")
            ax_end_pred.set_xlim(xlim)
            ax_end_pred.set_ylim(ylim)
            cbar_end_pred = plt.colorbar(scatter_end_pred, ax=ax_end_pred)
            cbar_end_pred.set_label("Activation")

            # Add overall title
            fig.suptitle(
                f"Epoch {trainer.current_epoch} - Trajectory {i+1}", fontsize=16
            )

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Make room for legends below

            figs.append(fig)

        # Log to wandb
        if trainer.logger is not None and hasattr(
            trainer.logger, "experiment"
        ):  # Check if wandb logger
            for i, fig in enumerate(figs):
                trainer.logger.experiment.log(
                    {
                        f"trajectory_analysis_{i+1}": wandb.Image(fig),
                    }
                )

        # Close figures to free memory
        for fig in figs:
            plt.close(fig)


class GradientStatisticsCallback(L.Callback):
    """
    Callback to track gradient statistics during training.
    
    Tracks gradient statistics across all parameters and optionally per weight matrix.
    
    Gradient variance: Var([∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ]) - variance of gradient 
    elements treated as a distribution. Low variance + low norm indicates vanishing 
    gradients; high variance can indicate instability.
    
    For RNNs, tracks separate statistics for:
    - W_in (input weights)
    - W_rec (recurrent weights)  
    - W_out (output/readout weights)
    - Other parameters (biases, init weights)
    """

    def __init__(
        self,
        save_dir: str,
        log_every_n_steps: int = 100,
        track_per_weight_matrix: bool = True,
    ):
        """
        Args:
            save_dir: Directory to save gradient statistics
            log_every_n_steps: Log gradients every N training steps
            track_per_weight_matrix: If True, track W_in, W_rec, W_out separately
        """
        super().__init__()
        self.save_dir = save_dir
        self.log_every_n_steps = log_every_n_steps
        self.track_per_weight_matrix = track_per_weight_matrix

        # Storage for global gradient statistics
        # "Global" = variance/norm/etc. computed across ALL gradient elements
        self.global_stats = {
            "step": [],
            "epoch": [],
            "grad_variance": [],  # Var of all gradient elements
            "grad_mean": [],       # Mean of all gradient elements
            "grad_norm": [],       # L2 norm of gradient vector
            "grad_max": [],        # Max gradient element
            "grad_min": [],        # Min gradient element
        }

        # Storage for per-weight-matrix statistics
        self.weight_matrix_stats = {}

    def _categorize_parameter(self, param_name: str) -> str:
        """
        Categorize parameter by type for grouped tracking.
        
        Maps parameter names to weight matrix categories:
        - W_in: Input weights (e.g., "rnn_step.W_in.weight")
        - W_rec: Recurrent weights (e.g., "rnn_step.W_rec.weight")
        - W_out: Output/readout weights (e.g., "W_out.weight")
        - W_h_init: Initial state encoder
        - biases: All bias terms
        - other: Everything else
        """
        # Remove "model." prefix if present
        name = param_name.replace("model.", "")
        
        if "W_in" in name or "input" in name.lower():
            if "bias" in name:
                return "biases"
            return "W_in"
        elif "W_rec" in name or "recurrent" in name.lower():
            if "bias" in name:
                return "biases"
            return "W_rec"
        elif "W_out" in name or "readout" in name.lower():
            return "W_out"
        elif "W_h_init" in name or "h_init" in name:
            return "W_h_init"
        elif "bias" in name:
            return "biases"
        else:
            return "other"

    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Called after loss.backward() and before optimizers step.
        
        Computes gradient statistics:
        1. Global: variance, norm, etc. across ALL gradient elements
        2. Per-matrix: separate stats for W_in, W_rec, W_out, etc.
        """
        # Only log every N steps
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Collect all gradients and group by weight matrix type
        all_grads = []
        weight_matrix_grads = {}

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.detach().cpu().flatten()
                all_grads.append(grad_data)

                if self.track_per_weight_matrix:
                    # Categorize this parameter
                    category = self._categorize_parameter(name)
                    if category not in weight_matrix_grads:
                        weight_matrix_grads[category] = []
                    weight_matrix_grads[category].append(grad_data)

        if len(all_grads) == 0:
            return

        # Concatenate all gradients into single vector
        all_grads_tensor = torch.cat(all_grads)

        # Compute global statistics (variance of all gradient elements)
        grad_variance = float(torch.var(all_grads_tensor))
        grad_mean = float(torch.mean(all_grads_tensor))
        grad_norm = float(torch.norm(all_grads_tensor))
        grad_max = float(torch.max(all_grads_tensor))
        grad_min = float(torch.min(all_grads_tensor))

        # Store global statistics
        self.global_stats["step"].append(trainer.global_step)
        self.global_stats["epoch"].append(trainer.current_epoch)
        self.global_stats["grad_variance"].append(grad_variance)
        self.global_stats["grad_mean"].append(grad_mean)
        self.global_stats["grad_norm"].append(grad_norm)
        self.global_stats["grad_max"].append(grad_max)
        self.global_stats["grad_min"].append(grad_min)

        # Log global statistics to wandb
        pl_module.log("train/grad_variance", grad_variance, on_step=True, on_epoch=False)
        pl_module.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)
        pl_module.log("train/grad_mean", grad_mean, on_step=True, on_epoch=False)
        pl_module.log("train/grad_max", grad_max, on_step=True, on_epoch=False)
        pl_module.log("train/grad_min", grad_min, on_step=True, on_epoch=False)

        # Compute and log per-weight-matrix statistics
        if self.track_per_weight_matrix:
            for matrix_type, grads in weight_matrix_grads.items():
                matrix_grad_tensor = torch.cat(grads)
                matrix_var = float(torch.var(matrix_grad_tensor))
                matrix_norm = float(torch.norm(matrix_grad_tensor))
                matrix_mean = float(torch.mean(matrix_grad_tensor))

                # Initialize storage for this matrix type if needed
                if matrix_type not in self.weight_matrix_stats:
                    self.weight_matrix_stats[matrix_type] = {
                        "step": [],
                        "epoch": [],
                        "variance": [],
                        "norm": [],
                        "mean": [],
                    }

                # Store statistics
                self.weight_matrix_stats[matrix_type]["step"].append(trainer.global_step)
                self.weight_matrix_stats[matrix_type]["epoch"].append(trainer.current_epoch)
                self.weight_matrix_stats[matrix_type]["variance"].append(matrix_var)
                self.weight_matrix_stats[matrix_type]["norm"].append(matrix_norm)
                self.weight_matrix_stats[matrix_type]["mean"].append(matrix_mean)

                # Log to wandb
                pl_module.log(
                    f"train/grad_variance/{matrix_type}",
                    matrix_var,
                    on_step=True,
                    on_epoch=False,
                )
                pl_module.log(
                    f"train/grad_norm/{matrix_type}",
                    matrix_norm,
                    on_step=True,
                    on_epoch=False,
                )
                pl_module.log(
                    f"train/grad_mean/{matrix_type}",
                    matrix_mean,
                    on_step=True,
                    on_epoch=False,
                )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Save gradient statistics at the end of each epoch."""
        self._save_statistics()

    @rank_zero_only
    def _save_statistics(self):
        """Save gradient statistics to JSON files."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Save global statistics (variance across all gradient elements)
        global_stats_path = os.path.join(self.save_dir, "gradient_statistics.json")
        with open(global_stats_path, "w") as f:
            json.dump(self.global_stats, f, indent=2)

        # Save per-weight-matrix statistics if tracked
        if self.track_per_weight_matrix and self.weight_matrix_stats:
            matrix_stats_path = os.path.join(
                self.save_dir, "gradient_statistics_weight_matrices.json"
            )
            with open(matrix_stats_path, "w") as f:
                json.dump(self.weight_matrix_stats, f, indent=2)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Create summary plots at the end of training."""
        self._save_statistics()
        self._create_summary_plots()

    @rank_zero_only
    def _create_summary_plots(self):
        """Create summary plots of gradient statistics."""
        if len(self.global_stats["step"]) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Gradient Statistics Throughout Training", fontsize=16)

        steps = self.global_stats["step"]

        # Plot 1: Gradient Variance
        axes[0, 0].plot(steps, self.global_stats["grad_variance"], "b-", linewidth=1)
        axes[0, 0].set_xlabel("Training Step")
        axes[0, 0].set_ylabel("Gradient Variance")
        axes[0, 0].set_title("Gradient Variance")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale("log")

        # Plot 2: Gradient Norm
        axes[0, 1].plot(steps, self.global_stats["grad_norm"], "g-", linewidth=1)
        axes[0, 1].set_xlabel("Training Step")
        axes[0, 1].set_ylabel("Gradient Norm")
        axes[0, 1].set_title("Gradient Norm (L2)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")

        # Plot 3: Gradient Mean (absolute value)
        axes[1, 0].plot(
            steps,
            np.abs(self.global_stats["grad_mean"]),
            "r-",
            linewidth=1,
            label="abs(mean)",
        )
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("|Gradient Mean|")
        axes[1, 0].set_title("Absolute Gradient Mean")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale("log")

        # Plot 4: Gradient Range (max/min)
        axes[1, 1].plot(steps, self.global_stats["grad_max"], "orange", linewidth=1, label="Max")
        axes[1, 1].plot(steps, np.abs(self.global_stats["grad_min"]), "purple", linewidth=1, label="abs(Min)")
        axes[1, 1].set_xlabel("Training Step")
        axes[1, 1].set_ylabel("Gradient Value")
        axes[1, 1].set_title("Gradient Range")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale("log")

        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, "gradient_statistics.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Gradient statistics plot saved to: {plot_path}")

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({"gradient_statistics_plot": wandb.Image(plot_path)})


class TauTrajectoryCallback(L.Callback):
    """Snapshot per-unit time constants at every validation step.

    Dumps `tau_trajectory.pt` at end of training with shape (T_snapshots, N).
    For frozen-tau (learn_time_constants=False) runs, the trajectory has a
    single row + a `frozen=True` flag (kept for provenance).
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        self.steps: list[int] = []
        self.taus: list[torch.Tensor] = []
        self.alphas: list[torch.Tensor] = []
        self._frozen_recorded = False

    def _model(self, pl_module):
        return getattr(pl_module, "model", pl_module)

    def _record(self, trainer, pl_module):
        model = self._model(pl_module)
        rnn_step = getattr(model, "rnn_step", None)
        if rnn_step is None:
            return
        self.steps.append(int(trainer.global_step))
        self.taus.append(rnn_step.current_time_constants.detach().cpu().clone())
        self.alphas.append(rnn_step.current_alphas.detach().cpu().clone())

    def on_fit_start(self, trainer, pl_module):
        self._record(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        model = self._model(pl_module)
        if not getattr(model, "learn_time_constants", True):
            if self._frozen_recorded:
                return
            self._frozen_recorded = True
        self._record(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        if not self.taus:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        model = self._model(pl_module)
        out = {
            "steps": torch.tensor(self.steps, dtype=torch.long),
            "taus": torch.stack(self.taus, dim=0),
            "alphas": torch.stack(self.alphas, dim=0),
            "frozen": not getattr(model, "learn_time_constants", True),
        }
        path = os.path.join(self.save_dir, "tau_trajectory.pt")
        torch.save(out, path)
        print(f"Tau trajectory saved to: {path}  (shape={tuple(out['taus'].shape)})")


class SpectralTrajectoryCallback(L.Callback):
    """Track top-k Jacobian eigenvalues throughout training.

    The primary trajectory uses the same discrete-time linearized Jacobian as
    SpectralSnapshotCallback:

        J = diag(1 - alpha) + diag(alpha) @ (g * W_rec)

    Top-k eigenvalues are rank trajectories sorted independently at each logged
    step by |lambda_J|. Full eigenspectra are saved too, so mode identity can be
    revisited post hoc with matching heuristics if needed.
    """

    def __init__(
        self,
        save_dir: str,
        top_k: int = 2,
        log_every_n_validation_epochs: int = 1,
        include_wrec: bool = True,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.top_k = top_k
        self.log_every_n_validation_epochs = log_every_n_validation_epochs
        self.include_wrec = include_wrec
        self.records: list[dict] = []

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        self._record(trainer, pl_module, tag="init")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.log_every_n_validation_epochs != 0:
            return
        self._record(trainer, pl_module, tag="validation")

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        self._record(trainer, pl_module, tag="final")

    def _record(self, trainer, pl_module, tag: str):
        model = getattr(pl_module, "model", pl_module)
        rnn_step = getattr(model, "rnn_step", None)
        if rnn_step is None:
            print(f"[SpectralTrajectoryCallback] no rnn_step on model; skipping {tag}.")
            return

        with torch.no_grad():
            W_rec = rnn_step.W_rec.weight.detach().cpu().numpy().astype(np.float64)
            alphas = rnn_step.current_alphas.detach().cpu().numpy().astype(np.float64)

        g = float(rnn_step.recurrent_gain)
        dt = float(model.dt)
        J = np.diag(1.0 - alphas) + (alphas[:, None] * W_rec) * g

        eigvals_jacobian = np.linalg.eigvals(J)
        jacobian_order = np.argsort(-np.abs(eigvals_jacobian))
        top_jacobian = eigvals_jacobian[jacobian_order[: self.top_k]]
        top_abs = np.abs(top_jacobian)
        top_log_abs = np.log(np.clip(top_abs, 1e-12, None))
        top_tau_eff = np.full_like(top_abs, np.nan, dtype=np.float64)
        stable = top_log_abs < -1e-12
        top_tau_eff[stable] = -dt / top_log_abs[stable]

        record = {
            "tag": tag,
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "alphas": alphas,
            "recurrent_gain": g,
            "eigvals_jacobian": eigvals_jacobian,
            "top_jacobian": top_jacobian,
            "top_jacobian_abs": top_abs,
            "top_jacobian_tau_eff": top_tau_eff,
            "top_jacobian_indices": jacobian_order[: self.top_k],
        }

        if self.include_wrec:
            eigvals_wrec = np.linalg.eigvals(W_rec)
            wrec_order = np.argsort(-np.abs(eigvals_wrec))
            top_wrec = eigvals_wrec[wrec_order[: self.top_k]]
            record.update(
                {
                    "eigvals_wrec": eigvals_wrec,
                    "top_wrec": top_wrec,
                    "top_wrec_demo_transform": 1.0 / (1.0 - top_wrec),
                    "top_wrec_indices": wrec_order[: self.top_k],
                }
            )

        self.records.append(record)
        self._save()

    @rank_zero_only
    def _save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, "spectral_trajectory.pt")
        torch.save(
            {
                "top_k": self.top_k,
                "ranked_by": "abs(lambda_jacobian)",
                "records": self.records,
            },
            path,
        )


class SpectralSnapshotCallback(L.Callback):
    """Snapshot the linearised Jacobian + Schur/eigen factors at fit start and end.

    Saves two .pt blobs in `save_dir`:
      - spectral_init.pt   (before any optimizer step)
      - spectral_final.pt  (after training)

    Each blob contains:
      W_rec, W_in, W_out, alphas, recurrent_gain
      J = (I - A) + g A W_rec
      Real Schur: Q (orthogonal), T_mat (quasi-triangular), eigvals_schur
      Eigendecomposition: eigvals_eig
      Connectivity-style couplings:
          coup_schur:   |W_out @ Q|        (output_size, N)
          coup_eig_re:  |Re(W_out @ V)|   (output_size, N)
          coup_neuron:  |W_out|             (output_size, N)
          drive_schur:  |Q.T @ (alpha * W_in)|   (N, input_size)
          drive_neuron: |alpha * W_in|            (N, input_size)

    Notes:
      - The Jacobian is the linearisation of the *Identity* dynamics at the
        origin. For nonlinear activations this is the linearisation point;
        document this externally if you extend it.
      - Correlation-based couplings (Pearson r between mode activity and
        targets) are deferred to a post-hoc notebook pass that runs a single
        forward sweep on val data.
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        self._dump(pl_module, tag="init")

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        self._dump(pl_module, tag="final")

    def _dump(self, pl_module, tag: str):
        from scipy.linalg import schur as _schur

        model = getattr(pl_module, "model", pl_module)
        rnn_step = getattr(model, "rnn_step", None)
        if rnn_step is None:
            print(f"[SpectralSnapshotCallback] no rnn_step on model; skipping {tag}.")
            return

        with torch.no_grad():
            W_rec = rnn_step.W_rec.weight.detach().cpu().numpy().astype(np.float64)
            W_in = rnn_step.W_in.weight.detach().cpu().numpy().astype(np.float64)
            W_in_bias = rnn_step.W_in.bias.detach().cpu().numpy().astype(np.float64) \
                if rnn_step.W_in.bias is not None else None
            W_out = model.W_out.weight.detach().cpu().numpy().astype(np.float64)
            alphas = rnn_step.current_alphas.detach().cpu().numpy().astype(np.float64)
            taus = rnn_step.current_time_constants.detach().cpu().numpy().astype(np.float64)

        g = float(getattr(model, "rnn_step").recurrent_gain)
        N = W_rec.shape[0]

        J = np.diag(1.0 - alphas) + (alphas[:, None] * W_rec) * g

        T_mat, Q = _schur(J, output="real")

        eigvals_schur = []
        i = 0
        while i < N:
            if i < N - 1 and abs(T_mat[i + 1, i]) > 1e-12:
                a = T_mat[i, i]
                b = T_mat[i, i + 1]
                c = T_mat[i + 1, i]
                d = T_mat[i + 1, i + 1]
                disc = (a - d) ** 2 / 4.0 + b * c
                if disc < 0:
                    im = np.sqrt(-disc)
                    eigvals_schur.append(complex((a + d) / 2.0, im))
                    eigvals_schur.append(complex((a + d) / 2.0, -im))
                else:
                    re = np.sqrt(disc)
                    eigvals_schur.append(complex((a + d) / 2.0 + re, 0.0))
                    eigvals_schur.append(complex((a + d) / 2.0 - re, 0.0))
                i += 2
            else:
                eigvals_schur.append(complex(T_mat[i, i], 0.0))
                i += 1
        eigvals_schur = np.array(eigvals_schur)

        eigvals_eig, V = np.linalg.eig(J)

        coup_schur = np.abs(W_out @ Q)
        coup_eig_re = np.abs((W_out @ V).real)
        coup_neuron = np.abs(W_out)
        drive_schur = np.abs(Q.T @ (alphas[:, None] * W_in))
        drive_neuron = np.abs(alphas[:, None] * W_in)

        blob = {
            "W_rec": W_rec,
            "W_in": W_in,
            "W_in_bias": W_in_bias,
            "W_out": W_out,
            "alphas": alphas,
            "taus": taus,
            "recurrent_gain": g,
            "J": J,
            "Q": Q,
            "T_mat": T_mat,
            "eigvals_schur": eigvals_schur,
            "V": V,
            "eigvals_eig": eigvals_eig,
            "coup_schur": coup_schur,
            "coup_eig_re": coup_eig_re,
            "coup_neuron": coup_neuron,
            "drive_schur": drive_schur,
            "drive_neuron": drive_neuron,
        }
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"spectral_{tag}.pt")
        torch.save(blob, path)
        print(f"Spectral snapshot ({tag}) saved to: {path}")
