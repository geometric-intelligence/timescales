import lightning as L
import json
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np
from timescales.rnns.rnn import RNN
from timescales.rnns.multitimescale_rnn import MultiTimescaleRNN
from timescales.analysis.measurements import PositionDecodingMeasurement


class LossLoggerCallback(L.Callback):
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

        self.train_losses_epoch: list[float] = []
        self.val_losses_epoch: list[float] = []
        self.epochs: list[int] = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.logged_metrics.get("train_loss_epoch", None)
        if train_loss is not None:
            self.train_losses_epoch.append(float(train_loss))

    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.sanity_checking:
            print("Sanity checking, skipping validation loss logging")
            return

        val_loss = trainer.logged_metrics.get("val_loss", None)
        if val_loss is not None:
            self.val_losses_epoch.append(float(val_loss))
            self.epochs.append(trainer.current_epoch)

        self._save_losses()

    @rank_zero_only
    def _save_losses(self):
        os.makedirs(self.save_dir, exist_ok=True)

        loss_data = {
            "epochs": self.epochs,
            "train_losses_epoch": self.train_losses_epoch,
            "val_losses_epoch": self.val_losses_epoch,
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
        decode_k: int = 3,
        log_every_n_epochs: int = 5,
        num_trajectories_to_plot: int = 3,
    ):
        super().__init__()
        self.place_cell_centers = place_cell_centers
        self.arena_size = arena_size
        self.decode_k = decode_k
        self.log_every_n_epochs = log_every_n_epochs
        self.num_trajectories_to_plot = num_trajectories_to_plot

    def decode_position_from_place_cells(
        self, activation: torch.Tensor
    ) -> torch.Tensor:
        """Decode position from place cell activations using top-k method."""
        centers = self.place_cell_centers.to(activation.device)
        _, idxs = torch.topk(activation, k=self.decode_k, dim=-1)  # [B, T, k]
        pred_pos = centers[idxs].mean(-2)  # [B, T, 2]
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
            if isinstance(pl_module.model, RNN):
                _, outputs = pl_module.model(
                    inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
                )
            elif isinstance(pl_module.model, MultiTimescaleRNN):
                _, outputs = pl_module.model(
                    inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
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


class TimescaleVisualizationCallback(L.Callback):
    """Callback to visualize and track timescale distributions for MultiTimescaleRNN.
    
    For fixed timescales: logs once at the start of training.
    For learnable timescales: logs periodically and saves timescale history to file.
    """

    def __init__(
        self,
        save_dir: str = None,
        log_every_n_epochs: int = 10,
        log_at_epoch: int = 0,  # Kept for backward compatibility
    ):
        """
        :param save_dir: Directory to save timescale history (required for learnable timescales)
        :param log_every_n_epochs: How often to log visualizations for learnable timescales
        :param log_at_epoch: Epoch at which to log for fixed timescales (backward compat)
        """
        super().__init__()
        self.save_dir = save_dir
        self.log_every_n_epochs = log_every_n_epochs
        self.log_at_epoch = log_at_epoch
        self.logged_fixed = False  # For fixed timescales (log once)
        
        # History tracking for learnable timescales
        self.timescale_history = {
            "epochs": [],
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
            "percentile_10": [],
            "percentile_25": [],
            "percentile_50": [],
            "percentile_75": [],
            "percentile_90": [],
            # Store full timescale vectors at key epochs
            "full_timescales": {},
        }

    def _should_log(self, trainer: L.Trainer, is_learnable: bool) -> bool:
        """Determine if we should log at this epoch."""
        epoch = trainer.current_epoch
        
        if is_learnable:
            # For learnable: log at epoch 0 and every N epochs
            return epoch == 0 or epoch % self.log_every_n_epochs == 0
        else:
            # For fixed: log once at specified epoch
            return epoch == self.log_at_epoch and not self.logged_fixed

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Track and log timescale evolution at end of each epoch."""

        # Only works with MultiTimescaleRNN
        if not isinstance(pl_module.model, MultiTimescaleRNN):
            return

        model = pl_module.model
        is_learnable = model.learn_timescales
        timescales = model.rnn_step.current_timescales.detach().cpu().numpy()
        alphas = model.rnn_step.current_alphas.detach().cpu().numpy()
        dt = model.dt
        epoch = trainer.current_epoch

        # Always record statistics for learnable timescales
        if is_learnable:
            self.timescale_history["epochs"].append(epoch)
            self.timescale_history["mean"].append(float(timescales.mean()))
            self.timescale_history["std"].append(float(timescales.std()))
            self.timescale_history["min"].append(float(timescales.min()))
            self.timescale_history["max"].append(float(timescales.max()))
            self.timescale_history["percentile_10"].append(float(np.percentile(timescales, 10)))
            self.timescale_history["percentile_25"].append(float(np.percentile(timescales, 25)))
            self.timescale_history["percentile_50"].append(float(np.percentile(timescales, 50)))
            self.timescale_history["percentile_75"].append(float(np.percentile(timescales, 75)))
            self.timescale_history["percentile_90"].append(float(np.percentile(timescales, 90)))
            
            # Save full timescales at key epochs (start, every N epochs, and will add final at end)
            if epoch == 0 or epoch % self.log_every_n_epochs == 0:
                self.timescale_history["full_timescales"][str(epoch)] = timescales.tolist()
            
            # Save history to file
            if self.save_dir is not None:
                history_path = os.path.join(self.save_dir, "timescale_history.json")
                with open(history_path, "w") as f:
                    json.dump(self.timescale_history, f, indent=2)

        # Visualize if appropriate
        if not self._should_log(trainer, is_learnable):
            return
        
        if not is_learnable:
            self.logged_fixed = True

        # Get the original configuration if available
        timescale_config = getattr(model, "_timescale_config", None)

        # Determine if discrete or continuous
        unique_timescales = np.unique(timescales)
        is_discrete = len(unique_timescales) <= 20

        if is_discrete:
            # Create a 2x1 layout for discrete case
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # === DISCRETE TIMESCALE VALUES ===
            ax1 = axes[0]
            unique_vals, counts = np.unique(timescales, return_counts=True)

            # Use stem plot instead of bars for discrete values
            markerline, stemlines, baseline = ax1.stem(unique_vals, counts, basefmt=" ")
            markerline.set_markersize(12)
            markerline.set_markerfacecolor("skyblue")
            markerline.set_markeredgecolor("navy")
            stemlines.set_linewidth(3)
            stemlines.set_color("navy")

            # Add value labels on top of each stem
            for val, count in zip(unique_vals, counts, strict=False):
                ax1.annotate(
                    f"{count}",
                    (val, count),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )
                ax1.annotate(
                    f"τ={val:.3f}",
                    (val, 0),
                    textcoords="offset points",
                    xytext=(0, -25),
                    ha="center",
                    fontsize=10,
                    rotation=45,
                )

            ax1.set_xlabel("Timescale Values")
            ax1.set_ylabel("Number of Units")
            ax1.set_title(
                f"Discrete Timescale Distribution\n{len(unique_vals)} unique values"
            )
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(counts) * 1.2)

            # === CORRESPONDING ALPHA VALUES ===
            ax2 = axes[1]
            unique_alphas = 1 - np.exp(-dt / unique_vals)

            markerline2, stemlines2, baseline2 = ax2.stem(
                unique_alphas, counts, basefmt=" "
            )
            markerline2.set_markersize(12)
            markerline2.set_markerfacecolor("lightcoral")
            markerline2.set_markeredgecolor("darkred")
            stemlines2.set_linewidth(3)
            stemlines2.set_color("darkred")

            # Add value labels
            for alpha, count in zip(unique_alphas, counts, strict=False):
                ax2.annotate(
                    f"{count}",
                    (alpha, count),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )
                ax2.annotate(
                    f"α={alpha:.3f}",
                    (alpha, 0),
                    textcoords="offset points",
                    xytext=(0, -25),
                    ha="center",
                    fontsize=10,
                    rotation=45,
                )

            ax2.set_xlabel("Alpha Values")
            ax2.set_ylabel("Number of Units")
            ax2.set_title(f"Corresponding Alpha Distribution\n(dt={dt})")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, max(counts) * 1.2)

        else:
            # Create a 2x2 layout for continuous case
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # === TIMESCALE HISTOGRAM ===
            ax1 = axes[0, 0]
            n_bins = min(50, len(unique_timescales))
            counts, bins, patches = ax1.hist(
                timescales, bins=n_bins, alpha=0.7, color="skyblue", edgecolor="black"
            )
            ax1.set_xlabel("Timescale (time units)")
            ax1.set_ylabel("Number of Units")
            ax1.set_title("Timescale Distribution (Actual Values)")
            ax1.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f"Min: {timescales.min():.3f}\nMax: {timescales.max():.3f}\nMean: {timescales.mean():.3f}\nStd: {timescales.std():.3f}"
            ax1.text(
                0.02,
                0.98,
                stats_text,
                transform=ax1.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # === ALPHA HISTOGRAM ===
            ax2 = axes[0, 1]
            n_bins_alpha = min(50, len(np.unique(alphas)))
            ax2.hist(
                alphas,
                bins=n_bins_alpha,
                alpha=0.7,
                color="lightcoral",
                edgecolor="black",
            )
            ax2.set_xlabel("Alpha (update rate)")
            ax2.set_ylabel("Number of Units")
            ax2.set_title("Alpha Distribution (Derived Values)")
            ax2.grid(True, alpha=0.3)

            # Add statistics text
            alpha_stats_text = f"Min: {alphas.min():.3f}\nMax: {alphas.max():.3f}\nMean: {alphas.mean():.3f}\nStd: {alphas.std():.3f}"
            ax2.text(
                0.02,
                0.98,
                alpha_stats_text,
                transform=ax2.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # === THEORETICAL DISTRIBUTION ===
            ax3 = axes[1, 0]
            if timescale_config is not None:
                self._plot_theoretical_distribution(ax3, timescale_config, timescales)
            else:
                # Show KDE if available
                try:
                    from scipy.stats import gaussian_kde

                    kde = gaussian_kde(timescales)
                    x_range = np.linspace(timescales.min(), timescales.max(), 200)
                    ax3.plot(x_range, kde(x_range), "g-", linewidth=2, label="KDE")
                    ax3.fill_between(
                        x_range, kde(x_range), alpha=0.3, color="lightgreen"
                    )
                    ax3.set_xlabel("Timescale")
                    ax3.set_ylabel("Density")
                    ax3.set_title("Empirical Density (KDE)")
                    ax3.legend()
                except ImportError:
                    ax3.text(
                        0.5,
                        0.5,
                        "scipy not available\nfor KDE",
                        transform=ax3.transAxes,
                        ha="center",
                        va="center",
                    )
                    ax3.set_title("Density Plot Unavailable")
            ax3.grid(True, alpha=0.3)

            # === TIMESCALE RANK PLOT (much more useful!) ===
            ax4 = axes[1, 1]

            # Sort timescales and show rank order
            sorted_idx = np.argsort(timescales)
            sorted_timescales = timescales[sorted_idx]
            ranks = np.arange(len(sorted_timescales))

            # Plot timescales by rank
            ax4.plot(
                ranks, sorted_timescales, "o-", markersize=2, linewidth=1, alpha=0.7
            )
            ax4.set_xlabel("Unit Rank (sorted by timescale)")
            ax4.set_ylabel("Timescale")
            ax4.set_title("Timescale Spectrum\n(Units sorted by timescale)")
            ax4.grid(True, alpha=0.3)

            # Add percentile lines
            percentiles = [10, 25, 50, 75, 90]
            colors = ["red", "orange", "green", "orange", "red"]
            for p, color in zip(percentiles, colors, strict=False):
                val = np.percentile(timescales, p)
                ax4.axhline(
                    val,
                    color=color,
                    linestyle="--",
                    alpha=0.7,
                    label=f"{p}th percentile: {val:.3f}",
                )
            ax4.legend(fontsize=8)

        # Overall title - include epoch for learnable timescales
        is_learnable = model.learn_timescales
        epoch = trainer.current_epoch
        
        if is_learnable:
            title_prefix = f"Epoch {epoch} - Learned"
        else:
            config_str = (
                f"Config: {timescale_config}" if timescale_config else "No config available"
            )
            title_prefix = f"Fixed - {config_str}"
            
        distribution_type = "Discrete" if is_discrete else "Continuous"
        fig.suptitle(
            f"Timescale Analysis ({distribution_type}) - {len(timescales)} units\n{title_prefix}",
            fontsize=14,
        )

        plt.tight_layout()

        # Log to wandb
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            log_key = f"timescale_analysis_epoch_{epoch}" if is_learnable else "timescale_analysis"
            trainer.logger.experiment.log(
                {
                    log_key: wandb.Image(fig),
                }
            )

        plt.close(fig)

    @rank_zero_only
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save final timescales at the end of training."""
        if not isinstance(pl_module.model, MultiTimescaleRNN):
            return
        
        model = pl_module.model
        if not model.learn_timescales:
            return
        
        # Save final timescales
        timescales = model.rnn_step.current_timescales.detach().cpu().numpy()
        final_epoch = trainer.current_epoch
        
        # Add final epoch to full timescales if not already there
        if str(final_epoch) not in self.timescale_history["full_timescales"]:
            self.timescale_history["full_timescales"][str(final_epoch)] = timescales.tolist()
        
        # Save final history
        if self.save_dir is not None:
            history_path = os.path.join(self.save_dir, "timescale_history.json")
            with open(history_path, "w") as f:
                json.dump(self.timescale_history, f, indent=2)
            print(f"Timescale history saved to: {history_path}")

    def _plot_theoretical_distribution(
        self, ax, config: dict, actual_timescales: np.ndarray
    ):
        """Plot the theoretical distribution based on configuration."""

        timescale_type = config.get("type", "unknown")

        if timescale_type == "uniform":
            # Single value - show as a vertical line
            value = config.get("value", 1.0)
            ax.axvline(value, color="red", linewidth=3, label=f"Uniform τ = {value}")
            ax.set_xlim(value * 0.8, value * 1.2)
            ax.set_xlabel("Timescale")
            ax.set_ylabel("Density")
            ax.set_title("Theoretical: Uniform (Single Value)")
            ax.legend()

        elif timescale_type == "discrete":
            # Bar plot of discrete values
            values = config.get("values", [])
            counts = [np.sum(actual_timescales == v) for v in values]
            ax.bar(values, counts, alpha=0.7, color="orange", edgecolor="black")
            ax.set_xlabel("Timescale Values")
            ax.set_ylabel("Count")
            ax.set_title("Theoretical: Discrete Values")

        elif timescale_type == "continuous":
            distribution = config.get("distribution", "unknown")

            # Create x range for plotting
            min_tau = actual_timescales.min()
            max_tau = actual_timescales.max()
            x = np.linspace(min_tau, max_tau, 200)

            if distribution == "lognormal":
                mu = config.get("mu", 0)
                sigma = config.get("sigma", 1)

                # PDF of lognormal: (1/(x*σ*√(2π))) * exp(-(ln(x)-μ)²/(2σ²))
                pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * ((np.log(x) - mu) / sigma) ** 2
                )

                # Handle any potential issues with very small values
                pdf = np.where(x > 0, pdf, 0)

                ax.plot(
                    x, pdf, "g-", linewidth=2, label=f"LogNormal(μ={mu}, σ={sigma})"
                )
                ax.fill_between(x, pdf, alpha=0.3, color="green")
                ax.set_title("Theoretical: Log-Normal Distribution")

            elif distribution == "uniform":
                min_val = config.get("min_timescale", min_tau)
                max_val = config.get("max_timescale", max_tau)

                # Uniform PDF
                pdf = np.where(
                    (x >= min_val) & (x <= max_val), 1 / (max_val - min_val), 0
                )

                ax.plot(
                    x,
                    pdf,
                    "b-",
                    linewidth=2,
                    label=f"Uniform({min_val:.2f}, {max_val:.2f})",
                )
                ax.fill_between(x, pdf, alpha=0.3, color="blue")
                ax.set_title("Theoretical: Uniform Distribution")

            elif distribution == "powerlaw":
                alpha_param = config.get("alpha", 2.0)
                min_val = config.get("min_timescale", min_tau)
                max_val = config.get("max_timescale", max_tau)

                if abs(alpha_param - 1.0) < 1e-6:
                    # Log-uniform case
                    pdf = np.where(
                        (x >= min_val) & (x <= max_val),
                        1 / (x * np.log(max_val / min_val)),
                        0,
                    )
                    label_str = "PowerLaw(α≈1, log-uniform)"
                else:
                    # General power law
                    C = (alpha_param - 1) / (
                        max_val ** (1 - alpha_param) - min_val ** (1 - alpha_param)
                    )
                    pdf = np.where(
                        (x >= min_val) & (x <= max_val), C * x ** (-alpha_param), 0
                    )
                    label_str = f"PowerLaw(α={alpha_param:.1f})"

                ax.plot(x, pdf, "purple", linewidth=2, label=label_str)
                ax.fill_between(x, pdf, alpha=0.3, color="purple")
                ax.set_title("Theoretical: Power-Law Distribution")

            elif distribution == "beta":
                # Beta distribution scaled to range
                alpha_param = config.get("alpha_param", 2.0)
                beta_param = config.get("beta_param", 5.0)
                min_val = config.get("min_timescale", min_tau)
                max_val = config.get("max_timescale", max_tau)

                # Transform x to [0,1] for beta distribution
                x_norm = (x - min_val) / (max_val - min_val)
                x_norm = np.clip(x_norm, 0, 1)

                # Beta PDF (simplified)
                from scipy.special import beta as beta_func

                pdf_norm = (
                    x_norm ** (alpha_param - 1) * (1 - x_norm) ** (beta_param - 1)
                ) / beta_func(alpha_param, beta_param)
                pdf = pdf_norm / (max_val - min_val)  # Scale for the transformed range

                ax.plot(
                    x,
                    pdf,
                    "brown",
                    linewidth=2,
                    label=f"Beta(α={alpha_param}, β={beta_param})",
                )
                ax.fill_between(x, pdf, alpha=0.3, color="brown")
                ax.set_title("Theoretical: Beta Distribution")

            ax.set_xlabel("Timescale")
            ax.set_ylabel("Density")
            ax.legend()


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
