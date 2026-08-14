import lightning as L
import json
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np


class LossLoggerCallback(L.Callback):
    def __init__(
        self,
        save_dir: str,
        initial_curve_path: str | None = None,
    ):
        self.save_dir = save_dir

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_objectives: list[float] = []
        self.val_objectives: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_accuracies: list[float] = []
        self.steps: list[int] = []

        self.val_losses_per_bit: dict[str, list[float]] = {}
        self.val_accuracies_per_bit: dict[str, list[float]] = {}

        if initial_curve_path is not None:
            self._load_initial_curve(initial_curve_path)

    def _load_initial_curve(self, path: str) -> None:
        """Seed a continuation run with an earlier run's logged curve."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"initial training curve not found: {path}")
        with open(path) as f:
            data = json.load(f)

        list_fields = (
            "steps",
            "train_losses",
            "val_losses",
            "train_objectives",
            "val_objectives",
            "train_accuracies",
            "val_accuracies",
        )
        for field in list_fields:
            values = data.get(field, [])
            if not isinstance(values, list):
                raise ValueError(f"{field} must be a list in {path}")
            setattr(self, field, list(values))

        dict_fields = (
            "val_losses_per_bit",
            "val_accuracies_per_bit",
        )
        for field in dict_fields:
            values = data.get(field, {})
            if not isinstance(values, dict) or any(
                not isinstance(series, list) for series in values.values()
            ):
                raise ValueError(f"{field} must map names to lists in {path}")
            setattr(
                self,
                field,
                {name: list(series) for name, series in values.items()},
            )

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.logged_metrics.get("train_loss_epoch", None)
        if train_loss is not None:
            self.train_losses.append(float(train_loss))

        train_objective = trainer.logged_metrics.get("train_objective_epoch", None)
        if train_objective is not None:
            self.train_objectives.append(float(train_objective))

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

        val_objective = trainer.logged_metrics.get("val_objective", None)
        if val_objective is not None:
            self.val_objectives.append(float(val_objective))

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
            "train_objectives": self.train_objectives,
            "val_objectives": self.val_objectives,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "val_losses_per_bit": self.val_losses_per_bit,
            "val_accuracies_per_bit": self.val_accuracies_per_bit,
        }

        with open(os.path.join(self.save_dir, "training_losses.json"), "w") as f:
            json.dump(loss_data, f, indent=2)


class RecurrentRestructuringCallback(L.Callback):
    """Track how the recurrent matrix moves away from its initialization.

    Frobenius displacement is recorded alongside scale-sensitive and directional
    diagnostics so no single scalar is treated as a complete measure of feature
    learning. The effective metrics include the forward multiplier (g or
    g/sqrt(N), depending on parameterization).
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        self.initial_weight: torch.Tensor | None = None
        self.records: list[dict[str, float | int | str]] = []

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        model = getattr(pl_module, "model", pl_module)
        self.initial_weight = (
            model.rnn_step.W_rec.weight.detach().cpu().double().clone()
        )
        self._record(trainer, pl_module, tag="init", log_metrics=False)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._record(trainer, pl_module, tag="validation", log_metrics=True)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        self._record(trainer, pl_module, tag="final", log_metrics=False)

    def _record(self, trainer, pl_module, tag: str, log_metrics: bool) -> None:
        if self.initial_weight is None:
            return

        model = getattr(pl_module, "model", pl_module)
        current = model.rnn_step.W_rec.weight.detach().cpu().double()
        initial = self.initial_weight
        delta = current - initial

        eps = torch.finfo(torch.float64).eps
        n_elements = initial.numel()
        init_fro = torch.linalg.vector_norm(initial)
        current_fro = torch.linalg.vector_norm(current)
        delta_fro = torch.linalg.vector_norm(delta)
        relative_delta = delta_fro / torch.clamp(init_fro, min=eps)

        initial_flat = initial.flatten()
        current_flat = current.flatten()
        delta_flat = delta.flatten()
        cosine = torch.dot(initial_flat, current_flat) / torch.clamp(
            init_fro * current_fro, min=eps
        )
        parallel_coefficient = torch.dot(delta_flat, initial_flat) / torch.clamp(
            init_fro**2, min=eps
        )
        orthogonal_delta = delta - parallel_coefficient * initial
        orthogonal_delta_fro = torch.linalg.vector_norm(orthogonal_delta)

        recurrent_scale = float(model.rnn_step.recurrent_weight_scale)
        record: dict[str, float | int | str] = {
            "tag": tag,
            "step": int(trainer.global_step),
            "epoch": int(trainer.current_epoch),
            "wrec_init_fro": float(init_fro),
            "wrec_current_fro": float(current_fro),
            "wrec_delta_fro": float(delta_fro),
            "wrec_relative_delta_fro": float(relative_delta),
            "wrec_delta_rms": float(delta_fro / n_elements**0.5),
            "wrec_cosine_from_init": float(cosine),
            "wrec_delta_parallel_coefficient": float(parallel_coefficient),
            "wrec_delta_orthogonal_fro": float(orthogonal_delta_fro),
            "recurrent_weight_scale": recurrent_scale,
            "effective_wrec_delta_fro": float(abs(recurrent_scale) * delta_fro),
            "effective_wrec_delta_rms": float(
                abs(recurrent_scale) * delta_fro / n_elements**0.5
            ),
        }
        self.records.append(record)

        if log_metrics:
            for key in (
                "wrec_delta_fro",
                "wrec_relative_delta_fro",
                "wrec_delta_rms",
                "wrec_cosine_from_init",
                "wrec_delta_orthogonal_fro",
                "effective_wrec_delta_fro",
                "effective_wrec_delta_rms",
            ):
                pl_module.log(
                    f"restructuring/{key}",
                    record[key],
                    on_step=False,
                    on_epoch=True,
                )

        self._save()

    @rank_zero_only
    def _save(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, "recurrent_restructuring.json")
        with open(path, "w") as f:
            json.dump({"records": self.records}, f, indent=2)


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
        recurrent_weight_scale = float(rnn_step.recurrent_weight_scale)
        dt = float(model.dt)
        J = (
            np.diag(1.0 - alphas)
            + (alphas[:, None] * W_rec) * recurrent_weight_scale
        )

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
            "recurrent_weight_scale": recurrent_weight_scale,
            "recurrent_parameterization": rnn_step.recurrent_parameterization,
            "eigvals_jacobian": eigvals_jacobian,
            "top_jacobian": top_jacobian,
            "top_jacobian_abs": top_abs,
            "top_jacobian_tau_eff": top_tau_eff,
            "top_jacobian_indices": jacobian_order[: self.top_k],
        }

        if self.include_wrec:
            effective_W_rec = recurrent_weight_scale * W_rec
            eigvals_wrec = np.linalg.eigvals(effective_W_rec)
            wrec_order = np.argsort(-np.abs(eigvals_wrec))
            top_wrec = eigvals_wrec[wrec_order[: self.top_k]]
            record.update(
                {
                    "eigvals_wrec": eigvals_wrec,
                    "effective_W_rec": effective_W_rec,
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

    def __init__(self, save_dir: str, eps_real_axis: float = 0.1):
        super().__init__()
        self.save_dir = save_dir
        self.eps_real_axis = eps_real_axis

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
            W_out_parameter = (
                model.W_out.weight.detach().cpu().numpy().astype(np.float64)
            )
            W_out = (
                model.effective_readout_weight.detach().cpu().numpy().astype(np.float64)
            )
            alphas = rnn_step.current_alphas.detach().cpu().numpy().astype(np.float64)
            taus = rnn_step.current_time_constants.detach().cpu().numpy().astype(np.float64)

        g = float(model.rnn_step.recurrent_gain)
        recurrent_weight_scale = float(rnn_step.recurrent_weight_scale)
        N = W_rec.shape[0]

        J = (
            np.diag(1.0 - alphas)
            + (alphas[:, None] * W_rec) * recurrent_weight_scale
        )
        effective_W_rec = recurrent_weight_scale * W_rec

        singular_values_wrec = np.linalg.svd(W_rec, compute_uv=False)
        singular_floor = (
            max(W_rec.shape)
            * np.finfo(np.float32).eps
            * singular_values_wrec[0]
        )
        wrec_numerical_rank = int(np.sum(singular_values_wrec > singular_floor))
        wrec_stable_rank = float(
            np.sum(singular_values_wrec**2) / singular_values_wrec[0] ** 2
        )
        eigvals_effective_wrec = np.linalg.eigvals(effective_W_rec)
        wrec_spectral_radius = float(np.max(np.abs(eigvals_effective_wrec)))

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
            "W_out_parameter": W_out_parameter,
            "readout_scale": float(model.readout_scale),
            "alphas": alphas,
            "taus": taus,
            "recurrent_gain": g,
            "recurrent_weight_scale": recurrent_weight_scale,
            "recurrent_parameterization": rnn_step.recurrent_parameterization,
            "wrec_init": getattr(model, "wrec_init", None),
            "wrec_init_config": getattr(model, "wrec_init_config", None),
            "singular_values_wrec": singular_values_wrec,
            "wrec_rank_tolerance": float(singular_floor),
            "wrec_numerical_rank": wrec_numerical_rank,
            "wrec_stable_rank": wrec_stable_rank,
            "wrec_frobenius_norm": float(np.linalg.norm(W_rec)),
            "wrec_spectral_norm": float(singular_values_wrec[0]),
            "effective_W_rec": effective_W_rec,
            "eigvals_effective_wrec": eigvals_effective_wrec,
            "wrec_spectral_radius": wrec_spectral_radius,
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

        # Lightweight JSON sidecar of named pinching statistics, so aggregation
        # can read them without opening the .pt blob (spec Workstream C2).
        from timescales.spectral_stats import spectral_pinching_stats

        stats = spectral_pinching_stats(eigvals_eig, eps_real_axis=self.eps_real_axis)
        stats.update(
            {
                "wrec_numerical_rank": wrec_numerical_rank,
                "wrec_stable_rank": wrec_stable_rank,
                "wrec_frobenius_norm": float(np.linalg.norm(W_rec)),
                "wrec_spectral_norm": float(singular_values_wrec[0]),
                "wrec_spectral_radius": wrec_spectral_radius,
            }
        )
        stats_path = os.path.join(self.save_dir, f"spectral_stats_{tag}.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Spectral stats ({tag}) saved to: {stats_path}")
