import numpy as np
import torch
import torch.nn as nn
import lightning as L
from scipy.stats import levy_stable


class RNNStep(nn.Module):
    """
    RNN step where each hidden unit has its own update rate (time constant).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float,
        time_constants: torch.Tensor | None = None,
        activation: type[nn.Module] = nn.Tanh,
        learn_time_constants: bool = False,
        init_time_constant: float | None = None,
        shared_time_constant: bool = False,
        normalize_hidden: bool = False,
        zero_diag_wrec: bool = True,
        recurrent_gain: float = 1.0,
        noise_std: float = 0.0,
        alpha_parameterization: str = "exponential",
        dynamics_type: str = "rate",
    ) -> None:
        """
        :param input_size: The size of the input.
        :param hidden_size: The size of the hidden state (number of neurons).
        :param dt: The time step.
        :param time_constants: Tensor of shape (hidden_size,) containing time constants for each unit.
                          Only used when learn_time_constants=False.
        :param activation: The activation function.
        :param learn_time_constants: If True, time constants become trainable parameters.
        :param init_time_constant: If provided and learn_time_constants=True, initialize all time constants
                              to this value (uniform initialization). If None, use random init.
        :param shared_time_constant: If True and learn_time_constants=True, use a single shared time constant
                                for all neurons instead of per-neuron time constants.
        :param normalize_hidden: If True, apply LayerNorm to hidden state after each step.
        :param zero_diag_wrec: If True, enforce diag(W_rec) = 0 (no self-connections).
        :param recurrent_gain: Multiplicative gain applied to W_rec output in the forward pass.
        :param noise_std: Standard deviation of Gaussian noise injected at each step.
                         Scaled by sqrt(dt/tau) per unit for consistent OU-process noise.
        :param alpha_parameterization: How alpha is computed from dt and tau.
                         "exponential": alpha = 1 - exp(-dt/tau)  (exact discretization)
                         "linear":      alpha = dt/tau             (first-order Euler)
        :param dynamics_type: Which ODE formulation to discretize.
                         "rate":    tau*dr/dt = -r + phi(g*W*r + W_in*u)   (rate-based)
                         "voltage": tau*dx/dt = -x + g*W*phi(x) + W_in*u  (voltage-based)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.activation = activation()
        self.learn_time_constants = learn_time_constants
        self.shared_time_constant = shared_time_constant
        self.normalize_hidden = normalize_hidden
        self.zero_diag_wrec = zero_diag_wrec
        self.recurrent_gain = recurrent_gain
        self.noise_std = noise_std
        self.alpha_parameterization = alpha_parameterization
        self.dynamics_type = dynamics_type

        if normalize_hidden:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None

        self.register_buffer("_dt_val", torch.tensor(dt))

        if learn_time_constants:
            if init_time_constant is not None:
                init_log_tau = float(torch.log(torch.tensor(init_time_constant)))
            else:
                init_log_tau = -0.5  # ~0.6s default
            
            if shared_time_constant:
                log_time_constants = torch.tensor([init_log_tau])
            else:
                if init_time_constant is not None:
                    log_time_constants = torch.full((hidden_size,), init_log_tau)
                else:
                    log_time_constants = torch.randn(hidden_size) * 0.5 - 0.5
            
            self.log_time_constants = nn.Parameter(log_time_constants)
        else:
            if time_constants is None:
                raise ValueError("time_constants must be provided when learn_time_constants=False")
            self.register_buffer("time_constants", time_constants)
            alphas = self._compute_alpha(dt, time_constants)
            self.register_buffer("alphas", alphas)

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        
        if zero_diag_wrec:
            self.W_rec.weight.data.fill_diagonal_(0)
            self.W_rec.weight.register_hook(lambda g: g.clone().fill_diagonal_(0))

    def _compute_alpha(self, dt, time_constants):
        """Compute alpha from dt and time constants using the configured parameterization."""
        if self.alpha_parameterization == "exponential":
            return 1 - torch.exp(-dt / time_constants)
        elif self.alpha_parameterization == "linear":
            return dt / time_constants
        else:
            raise ValueError(f"Unknown alpha_parameterization: {self.alpha_parameterization}")

    @property
    def current_time_constants(self) -> torch.Tensor:
        """Get current time constant values (computed from log_time_constants if learnable).
        
        For shared_time_constant=True, returns a scalar tensor.
        For shared_time_constant=False (default), returns tensor of shape (hidden_size,).
        """
        if self.learn_time_constants:
            return torch.exp(self.log_time_constants)
        else:
            return self.time_constants

    @property
    def current_alphas(self) -> torch.Tensor:
        """Get current alpha values (computed from log_time_constants if learnable).
        
        Always returns tensor of shape (hidden_size,) for broadcasting in forward pass.
        For shared_time_constant=True, expands the single alpha to all neurons.
        """
        if self.learn_time_constants:
            time_constants = torch.exp(self.log_time_constants)
            alphas = self._compute_alpha(self._dt_val, time_constants)
            if self.shared_time_constant:
                alphas = alphas.expand(self.hidden_size)
            return alphas
        else:
            return self.alphas

    def _sample_noise(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Sample tau-normalized OU noise consistent with the alpha discretization.

        Convention: stationary variance = sigma^2/2, independent of tau.
          linear (Euler-Maruyama): scale = sigma * sqrt(dt / tau)
          exponential (exact OU):  scale = sigma * sqrt(tau/2 * (1 - exp(-2*dt/tau)))
        """
        time_constants = self.current_time_constants
        dt = self._dt_val
        if self.alpha_parameterization == "linear":
            noise_scale = self.noise_std * torch.sqrt(dt / time_constants)
        else:
            noise_scale = self.noise_std * torch.sqrt(
                time_constants / 2 * (1 - torch.exp(-2 * dt / time_constants))
            )
        return torch.randn_like(hidden) * noise_scale

    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with per-unit update rates.

        :param input: (batch, input_size)
        :param hidden: (batch, hidden_size)
        :return: new_hidden: (batch, hidden_size)
        """
        alphas = self.current_alphas

        if self.dynamics_type == "rate":
            pre_activation = self.recurrent_gain * self.W_rec(hidden) + self.W_in(input)
            drive = self.activation(pre_activation)
        elif self.dynamics_type == "voltage":
            drive = self.recurrent_gain * self.W_rec(self.activation(hidden)) + self.W_in(input)
        else:
            raise ValueError(f"Unknown dynamics_type: {self.dynamics_type}")

        new_hidden = (1 - alphas) * hidden + alphas * drive

        if self.noise_std > 0.0 and self.training:
            new_hidden = new_hidden + self._sample_noise(new_hidden)

        if self.layer_norm is not None:
            new_hidden = self.layer_norm(new_hidden)

        return new_hidden


class RNN(nn.Module):
    """
    RNN where each hidden unit has its own update rate (time constant).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dt: float,
        time_constants_config: dict | None = None,
        activation: type[nn.Module] = nn.Tanh,
        learn_time_constants: bool = False,
        init_time_constant: float | None = None,
        shared_time_constant: bool = False,
        normalize_hidden: bool = False,
        zero_diag_wrec: bool = False,
        recurrent_gain: float = 1.0,
        noise_std: float = 0.0,
        wrec_init: str = "orthogonal",
        alpha_parameterization: str = "exponential",
        stability_param: float = 2.0,
        dynamics_type: str = "rate",
    ) -> None:
        """
        :param input_size: The size of the input.
        :param hidden_size: The size of the hidden state (number of neurons).
        :param output_size: The size of the output vector.
        :param dt: The time step size.
        :param time_constants_config: Dictionary specifying how to set the time constants.
                                  Only used when learn_time_constants=False.
        :param activation: The activation function.
        :param learn_time_constants: If True, time constants become trainable parameters
                                 (randomly initialized, time_constants_config is ignored).
        :param init_time_constant: If provided and learn_time_constants=True, initialize all 
                              time constants to this value (uniform). If None, use random init.
        :param shared_time_constant: If True and learn_time_constants=True, use a single shared
                                time constant for all neurons instead of per-neuron time constants.
        :param normalize_hidden: If True, apply LayerNorm to hidden state after each step.
        :param zero_diag_wrec: If True, enforce diag(W_rec) = 0 (no self-connections).
        :param recurrent_gain: Multiplicative gain on recurrent weights in forward pass.
        :param noise_std: Std of injected Gaussian noise per step (0 = no noise).
        :param wrec_init: Initialization scheme for W_rec.
                         "orthogonal": orthogonal initialization.
                         "normal_scaled": N(0, 1/N) entries (Sompolinsky et al. 1988).
                         "levy_stable": alpha-stable (Levy) distribution.
                         "lognormal": |W_ij| ~ LogNormal(0, 1) with random signs, scaled by 1/sqrt(N).
        :param alpha_parameterization: "exponential" for 1-exp(-dt/tau), "linear" for dt/tau.
        :param stability_param: Stability exponent for Levy-stable distribution (only for wrec_init="levy_stable").
        :param dynamics_type: ODE formulation to discretize ("rate" or "voltage").
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt
        self.learn_time_constants = learn_time_constants
        self.init_time_constant = init_time_constant
        self.shared_time_constant = shared_time_constant
        self.normalize_hidden = normalize_hidden
        self.zero_diag_wrec = zero_diag_wrec
        self.stability_param = stability_param
        self.wrec_init = wrec_init

        if learn_time_constants:
            time_constants = None
            if shared_time_constant:
                print(f"Time constants are LEARNABLE (SHARED, init at τ={init_time_constant}s)")
            elif init_time_constant is not None:
                print(f"Time constants are LEARNABLE (per-neuron, uniform init at τ={init_time_constant}s)")
            else:
                print(f"Time constants are LEARNABLE (per-neuron, randomly initialized)")
        else:
            if time_constants_config is None:
                raise ValueError("time_constants_config must be provided when learn_time_constants=False")
            time_constants = self._generate_time_constants(hidden_size, time_constants_config)

        self.rnn_step = RNNStep(
            input_size=input_size,
            hidden_size=hidden_size,
            dt=dt,
            time_constants=time_constants,
            activation=activation,
            learn_time_constants=learn_time_constants,
            init_time_constant=init_time_constant,
            shared_time_constant=shared_time_constant,
            normalize_hidden=normalize_hidden,
            zero_diag_wrec=zero_diag_wrec,
            recurrent_gain=recurrent_gain,
            noise_std=noise_std,
            alpha_parameterization=alpha_parameterization,
            dynamics_type=dynamics_type,
        )
        self.W_out = nn.Linear(hidden_size, output_size, bias=False)

        self.W_h_init = nn.Linear(output_size, hidden_size, bias=False)

        self._initialize_weights()

    def _generate_time_constants(
        self, hidden_size: int, time_constants_config: dict
    ) -> torch.Tensor:
        """
        Generate time constants based on configuration.

        :param hidden_size: Number of hidden units
        :param time_constants_config: Configuration dictionary
        :return: Tensor of time constants of shape (hidden_size,)
        """
        tc_type = time_constants_config["type"]

        if tc_type == "discrete":
            discrete_values = time_constants_config["values"]
            discrete_values = torch.tensor(discrete_values, dtype=torch.float32)

            if len(discrete_values) == 1:
                time_constants = discrete_values.repeat(hidden_size)
            else:
                indices = torch.randint(0, len(discrete_values), (hidden_size,))
                time_constants = discrete_values[indices]

        elif tc_type == "continuous":
            distribution = time_constants_config["distribution"]

            if distribution == "uniform":
                min_tc = float(time_constants_config["min_time_constant"])
                max_tc = float(time_constants_config["max_time_constant"])
                time_constants = torch.uniform(
                    min_tc, max_tc, size=(hidden_size,)
                )

            elif distribution == "powerlaw":
                exponent = float(time_constants_config["exponent"])
                min_tc = float(time_constants_config["min_time_constant"])
                max_tc = float(time_constants_config["max_time_constant"])

                u = torch.rand(hidden_size)

                if abs(exponent - 1.0) < 1e-6:
                    time_constants = min_tc * torch.pow(
                        torch.tensor(max_tc / min_tc), u
                    )
                else:
                    ratio = max_tc / min_tc
                    exponent_term = 1.0 - exponent
                    power_term = (ratio**exponent_term) * u + (1 - u)
                    time_constants = min_tc * torch.pow(
                        power_term, 1.0 / exponent_term
                    )
            elif distribution == "gaussian":
                mean = float(time_constants_config["mean"])
                std = float(time_constants_config["std"])
                
                time_constants = torch.normal(mean, std, size=(hidden_size,))
                
                min_tc = time_constants_config.get("min_time_constant", 0.01)
                max_tc = time_constants_config.get("max_time_constant", 1.0)
                time_constants = torch.clamp(time_constants, min=min_tc, max=max_tc)

            else:
                raise ValueError(f"Unknown continuous distribution: {distribution}")

        return time_constants

    def forward(
        self,
        inputs: torch.Tensor,
        init_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.

        :param inputs: (batch, time, input_size)
        :param init_context: (batch, output_size) - optional context to initialize hidden state.
        :return: hidden_states: (batch, time, hidden_size)
        :return: outputs: (batch, time, output_size)
        """
        batch_size, seq_len, _ = inputs.shape

        hidden_states = []
        outputs = []
        if init_context is not None:
            hidden = self.W_h_init(init_context)
        else:
            hidden = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        for t in range(seq_len):
            input_t = inputs[:, t, :]
            hidden = self.rnn_step(input_t, hidden)
            hidden_states.append(hidden)
            outputs.append(self.W_out(hidden))

        return torch.stack(hidden_states, dim=1), torch.stack(outputs, dim=1)

    def _initialize_weights(self) -> None:
        """Initialize weights for RNN training."""
        nn.init.xavier_uniform_(self.rnn_step.W_in.weight)
        nn.init.zeros_(self.rnn_step.W_in.bias)

        n = self.hidden_size
        if self.wrec_init == "orthogonal":
            nn.init.orthogonal_(self.rnn_step.W_rec.weight)
        elif self.wrec_init == "normal_scaled":
            nn.init.normal_(self.rnn_step.W_rec.weight, mean=0.0, std=1.0 / n**0.5)
        elif self.wrec_init == "levy_stable":
            samples = levy_stable.rvs(
                alpha=self.stability_param,
                beta=0,
                loc=0,
                scale=1.0 / n**0.5,
                size=(n, n),
            )
            with torch.no_grad():
                self.rnn_step.W_rec.weight.copy_(torch.tensor(samples, dtype=torch.float32))
        elif self.wrec_init == "lognormal":
            magnitudes = np.random.lognormal(mean=0.0, sigma=1.0, size=(n, n))
            signs = np.random.choice([-1.0, 1.0], size=(n, n))
            samples = signs * magnitudes / n**0.5
            with torch.no_grad():
                self.rnn_step.W_rec.weight.copy_(torch.tensor(samples, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown wrec_init: {self.wrec_init}")
        nn.init.zeros_(self.rnn_step.W_rec.bias)

        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.xavier_uniform_(self.W_h_init.weight)

    def get_time_constant_stats(self) -> dict:
        """Return statistics about the time constant values.
        
        Works for both fixed and learnable time constants.
        """
        time_constants = self.rnn_step.current_time_constants
        alphas = self.rnn_step.current_alphas
        
        return {
            "time_constant_min": time_constants.min().item(),
            "time_constant_max": time_constants.max().item(),
            "time_constant_mean": time_constants.mean().item(),
            "time_constant_std": time_constants.std().item(),
            "alpha_min": alphas.min().item(),
            "alpha_max": alphas.max().item(),
            "alpha_mean": alphas.mean().item(),
            "unique_time_constants": len(torch.unique(time_constants)),
            "learn_time_constants": self.learn_time_constants,
        }


class RNNLightning(L.LightningModule):
    def __init__(
        self,
        model: "RNN",
        learning_rate: float,
        weight_decay: float,
        step_size: int,
        gamma: float,
        task: str = "path_integration",
        precondition_gradients: bool = False,
        eps_alpha: float = 1e-2,
        lr_interval: str = "epoch",
    ) -> None:
        """
        :param model: The RNN model.
        :param learning_rate: The learning rate.
        :param weight_decay: The weight decay for the recurrent weights.
        :param step_size: The step size for the learning rate scheduler.
        :param gamma: The gamma for the learning rate scheduler.
        :param task: The task type.
        :param precondition_gradients: If True, apply alpha-based gradient preconditioning.
        :param eps_alpha: Damping constant for numerical stability in preconditioner.
        :param lr_interval: LR scheduler interval — "epoch" or "step".
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.task = task
        self.precondition_gradients = precondition_gradients
        self.eps_alpha = eps_alpha
        self.lr_interval = lr_interval
        
        if task in ("binary_counter", "flip_flop"):
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif task == "teacher_student":
            self.loss_fn = nn.MSELoss(reduction='none')

        if precondition_gradients:
            print(f"Gradient preconditioning ENABLED (eps_alpha={eps_alpha})")
        else:
            print(f"Gradient preconditioning DISABLED")

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if self.task in ("path_integration", "path_integration_1d"):
            y = targets.reshape(-1, self.model.output_size)
            yhat = torch.softmax(outputs.reshape(-1, self.model.output_size), dim=-1)
            loss = -(y * torch.log(yhat + 1e-8)).sum(-1).mean()
            return loss, None
        elif self.task in ("binary_counter", "flip_flop"):
            batch_size, seq_len, n_channels = outputs.shape
            outputs_flat = outputs.reshape(-1, n_channels)
            targets_flat = targets.reshape(-1, n_channels)
            per_sample_loss = self.loss_fn(outputs_flat, targets_flat)
            per_channel_loss = per_sample_loss.mean(dim=0)
            total_loss = per_channel_loss.mean()
            per_channel_dict = {
                f"channel_{i}": per_channel_loss[i].item()
                for i in range(n_channels)
            }
            return total_loss, per_channel_dict
        elif self.task == "teacher_student":
            batch_size, seq_len, n_channels = outputs.shape
            outputs_flat = outputs.reshape(-1, n_channels)
            targets_flat = targets.reshape(-1, n_channels)
            per_sample_loss = self.loss_fn(outputs_flat, targets_flat)
            per_channel_loss = per_sample_loss.mean(dim=0)
            total_loss = per_channel_loss.mean()
            per_channel_dict = {
                f"channel_{i}": per_channel_loss[i].item()
                for i in range(n_channels)
            }
            return total_loss, per_channel_dict
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _compute_accuracy(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor | None, dict[str, float] | None]:
        if self.task in ("binary_counter", "flip_flop"):
            preds = (torch.sigmoid(outputs) > 0.5).float()
            per_channel_acc = (preds == targets).float().mean(dim=(0, 1))
            overall_acc = per_channel_acc.mean()
            per_channel_dict = {
                f"channel_{i}": per_channel_acc[i].item()
                for i in range(per_channel_acc.shape[0])
            }
            return overall_acc, per_channel_dict
        elif self.task in ("path_integration", "path_integration_1d"):
            pred_idx = outputs.reshape(-1, self.model.output_size).argmax(dim=-1)
            tgt_idx = targets.reshape(-1, self.model.output_size).argmax(dim=-1)
            acc = (pred_idx == tgt_idx).float().mean()
            return acc, None
        elif self.task == "teacher_student":
            mse = ((outputs - targets) ** 2).mean()
            var = targets.var()
            r_squared = 1.0 - mse / (var + 1e-8)
            per_channel_r2 = {}
            for i in range(outputs.shape[-1]):
                ch_mse = ((outputs[..., i] - targets[..., i]) ** 2).mean()
                ch_var = targets[..., i].var()
                per_channel_r2[f"channel_{i}"] = (1.0 - ch_mse / (ch_var + 1e-8)).item()
            return r_squared, per_channel_r2
        else:
            return None, None

    def training_step(self, batch) -> torch.Tensor:
        inputs, aux_info, targets = batch
        
        if self.task in ("path_integration", "path_integration_1d"):
            init_context = targets[:, 0, :]
        else:
            init_context = None
            
        hidden_states, outputs = self.model(
            inputs=inputs, init_context=init_context
        )

        loss, per_channel_losses = self._compute_loss(outputs, targets)
        loss += self.weight_decay * (self.model.rnn_step.W_rec.weight**2).sum()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if per_channel_losses is not None:
            for channel_name, channel_loss in per_channel_losses.items():
                self.log(f"train_loss_{channel_name}", channel_loss, on_step=True, on_epoch=True, sync_dist=True)

        accuracy, per_channel_acc = self._compute_accuracy(outputs, targets)
        if accuracy is not None:
            self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            if per_channel_acc is not None:
                for channel_name, acc in per_channel_acc.items():
                    self.log(f"train_accuracy_{channel_name}", acc, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch) -> torch.Tensor:
        inputs, aux_info, targets = batch
        
        if self.task in ("path_integration", "path_integration_1d"):
            init_context = targets[:, 0, :]
        else:
            init_context = None
            
        hidden_states, outputs = self.model(
            inputs=inputs, init_context=init_context
        )

        loss, per_channel_losses = self._compute_loss(outputs, targets)
        loss += self.weight_decay * (self.model.rnn_step.W_rec.weight**2).sum()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if per_channel_losses is not None:
            for channel_name, channel_loss in per_channel_losses.items():
                self.log(f"val_loss_{channel_name}", channel_loss, on_step=False, on_epoch=True, sync_dist=True)

        accuracy, per_channel_acc = self._compute_accuracy(outputs, targets)
        if accuracy is not None:
            self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if per_channel_acc is not None:
                for channel_name, acc in per_channel_acc.items():
                    self.log(f"val_accuracy_{channel_name}", acc, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.lr_interval,
                "monitor": "val_loss",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        """Apply alpha-based gradient preconditioning before optimizer step."""
        if not self.precondition_gradients:
            return

        alphas = self.model.rnn_step.current_alphas.detach()
        preconditioner = 1.0 / (alphas + self.eps_alpha)
        
        if self.model.rnn_step.W_in.weight.grad is not None:
            self.model.rnn_step.W_in.weight.grad *= preconditioner.unsqueeze(1)
        if self.model.rnn_step.W_in.bias.grad is not None:
            self.model.rnn_step.W_in.bias.grad *= preconditioner
        if self.model.rnn_step.W_rec.weight.grad is not None:
            self.model.rnn_step.W_rec.weight.grad *= preconditioner.unsqueeze(1)
        if self.model.rnn_step.W_rec.bias.grad is not None:
            self.model.rnn_step.W_rec.bias.grad *= preconditioner
        if self.model.W_h_init.weight.grad is not None:
            self.model.W_h_init.weight.grad *= preconditioner.unsqueeze(1)

    def on_train_start(self):
        """Log time constant statistics at the start of training."""
        tc_stats = self.model.get_time_constant_stats()
        print(f"Time constant statistics: {tc_stats}")
        if self.precondition_gradients:
            print(f"Gradient preconditioning ENABLED (eps_alpha={self.eps_alpha})")
