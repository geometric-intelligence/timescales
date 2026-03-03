import torch
import torch.nn as nn
import lightning as L
from scipy.stats import levy_stable


class MultiTimescaleRNNStep(nn.Module):
    """
    Multi-timescale RNN step where each hidden unit has its own update rate (timescale).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float,
        timescales: torch.Tensor | None = None,
        activation: type[nn.Module] = nn.Tanh,
        learn_timescales: bool = False,
        init_timescale: float | None = None,
        shared_timescale: bool = False,
        normalize_hidden: bool = False,
        zero_diag_wrec: bool = True,
        recurrent_gain: float = 1.0,
        noise_std: float = 0.0,
        alpha_parameterization: str = "exponential",
        dynamics_type: str = "rate",
    ) -> None:
        """
        Initialize the Multi-timescale RNN step.

        :param input_size: The size of the velocity input (= dimension of space).
        :param hidden_size: The size of the hidden state (number of neurons).
        :param dt: The time step.
        :param timescales: Tensor of shape (hidden_size,) containing timescales for each unit.
                          Only used when learn_timescales=False. If None and learn_timescales=True,
                          timescales are randomly initialized.
        :param activation: The activation function.
        :param learn_timescales: If True, timescales become trainable parameters.
        :param init_timescale: If provided and learn_timescales=True, initialize all timescales
                              to this value (uniform initialization). If None, use random init.
        :param shared_timescale: If True and learn_timescales=True, use a single shared timescale
                                for all neurons instead of per-neuron timescales.
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
        self.learn_timescales = learn_timescales
        self.shared_timescale = shared_timescale
        self.normalize_hidden = normalize_hidden
        self.zero_diag_wrec = zero_diag_wrec
        self.recurrent_gain = recurrent_gain
        self.noise_std = noise_std
        self.alpha_parameterization = alpha_parameterization
        self.dynamics_type = dynamics_type

        # Optional LayerNorm for hidden state normalization
        if normalize_hidden:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None

        self.register_buffer("_dt_val", torch.tensor(dt))

        if learn_timescales:
            # Learnable timescales via log-parameterization
            # log_timescales is unconstrained; timescales = exp(log_timescales) > 0
            if init_timescale is not None:
                init_log_tau = float(torch.log(torch.tensor(init_timescale)))
            else:
                init_log_tau = -0.5  # ~0.6s default
            
            if shared_timescale:
                # Single shared timescale for all neurons (scalar)
                log_timescales = torch.tensor([init_log_tau])
            else:
                # Per-neuron timescales (vector)
                if init_timescale is not None:
                    log_timescales = torch.full((hidden_size,), init_log_tau)
                else:
                    # Random initialization
                    log_timescales = torch.randn(hidden_size) * 0.5 - 0.5
            
            self.log_timescales = nn.Parameter(log_timescales)
        else:
            # Fixed timescales (original behavior)
            if timescales is None:
                raise ValueError("timescales must be provided when learn_timescales=False")
            self.register_buffer("timescales", timescales)
            alphas = self._compute_alpha(dt, timescales)
            self.register_buffer("alphas", alphas)

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        
        # Enforce zero diagonal on W_rec if requested (init to 0 + freeze via gradient hook)
        if zero_diag_wrec:
            self.W_rec.weight.data.fill_diagonal_(0)
            self.W_rec.weight.register_hook(lambda g: g.clone().fill_diagonal_(0))

    def _compute_alpha(self, dt, timescales):
        """Compute alpha from dt and timescales using the configured parameterization."""
        if self.alpha_parameterization == "exponential":
            return 1 - torch.exp(-dt / timescales)
        elif self.alpha_parameterization == "linear":
            return dt / timescales
        else:
            raise ValueError(f"Unknown alpha_parameterization: {self.alpha_parameterization}")

    @property
    def current_timescales(self) -> torch.Tensor:
        """Get current timescale values (computed from log_timescales if learnable).
        
        For shared_timescale=True, returns a scalar tensor.
        For shared_timescale=False (default), returns tensor of shape (hidden_size,).
        """
        if self.learn_timescales:
            return torch.exp(self.log_timescales)
        else:
            return self.timescales

    @property
    def current_alphas(self) -> torch.Tensor:
        """Get current alpha values (computed from log_timescales if learnable).
        
        Always returns tensor of shape (hidden_size,) for broadcasting in forward pass.
        For shared_timescale=True, expands the single alpha to all neurons.
        """
        if self.learn_timescales:
            timescales = torch.exp(self.log_timescales)
            alphas = self._compute_alpha(self._dt_val, timescales)
            if self.shared_timescale:
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
        timescales = self.current_timescales
        dt = self._dt_val
        if self.alpha_parameterization == "linear":
            noise_scale = self.noise_std * torch.sqrt(dt / timescales)
        else:
            noise_scale = self.noise_std * torch.sqrt(
                timescales / 2 * (1 - torch.exp(-2 * dt / timescales))
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
            # tau*dr/dt = -r + phi(g*W*r + W_in*u)
            pre_activation = self.recurrent_gain * self.W_rec(hidden) + self.W_in(input)
            drive = self.activation(pre_activation)
        elif self.dynamics_type == "voltage":
            # tau*dx/dt = -x + g*W*phi(x) + W_in*u
            drive = self.recurrent_gain * self.W_rec(self.activation(hidden)) + self.W_in(input)
        else:
            raise ValueError(f"Unknown dynamics_type: {self.dynamics_type}")

        new_hidden = (1 - alphas) * hidden + alphas * drive

        if self.noise_std > 0.0 and self.training:
            new_hidden = new_hidden + self._sample_noise(new_hidden)

        if self.layer_norm is not None:
            new_hidden = self.layer_norm(new_hidden)

        return new_hidden


class MultiTimescaleRNN(nn.Module):
    """
    Multi-timescale RNN where each hidden unit has its own update rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dt: float,
        timescales_config: dict | None = None,
        activation: type[nn.Module] = nn.Tanh,
        learn_timescales: bool = False,
        init_timescale: float | None = None,
        shared_timescale: bool = False,
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
        Initialize the Multi-timescale RNN.

        :param input_size: The size of the velocity input (= dimension of space).
        :param hidden_size: The size of the hidden state (number of neurons).
        :param output_size: The size of the output vector (number of place cells).
        :param dt: The time step size.
        :param timescales_config: Dictionary specifying how to set the timescales.
                                  Only used when learn_timescales=False.
        :param activation: The activation function.
        :param learn_timescales: If True, timescales become trainable parameters
                                 (randomly initialized, timescales_config is ignored).
        :param init_timescale: If provided and learn_timescales=True, initialize all 
                              timescales to this value (uniform). If None, use random init.
        :param shared_timescale: If True and learn_timescales=True, use a single shared
                                timescale for all neurons instead of per-neuron timescales.
        :param normalize_hidden: If True, apply LayerNorm to hidden state after each step.
        :param zero_diag_wrec: If True, enforce diag(W_rec) = 0 (no self-connections).
        :param recurrent_gain: Multiplicative gain on recurrent weights in forward pass.
        :param noise_std: Std of injected Gaussian noise per step (0 = no noise).
        :param wrec_init: Initialization scheme for W_rec.
                         "orthogonal": orthogonal initialization.
                         "normal_scaled": N(0, 1/N) entries (Sompolinsky et al. 1988).
                         "levy_stable": alpha-stable (Levy) distribution with
                             beta=0, loc=0, scale=1/sqrt(N). The stability exponent
                             is set via stability_param.
        :param alpha_parameterization: "exponential" for 1-exp(-dt/tau), "linear" for dt/tau.
        :param stability_param: Stability exponent for the Levy-stable distribution
                               (only used when wrec_init="levy_stable"). Must be in (0, 2].
                               2.0 = Gaussian, 1.0 = Cauchy, <2 = heavy-tailed.
        :param dynamics_type: ODE formulation to discretize.
                         "rate":    tau*dr/dt = -r + phi(g*W*r + W_in*u)   (rate-based)
                         "voltage": tau*dx/dt = -x + g*W*phi(x) + W_in*u  (voltage-based)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt
        self.learn_timescales = learn_timescales
        self.init_timescale = init_timescale
        self.shared_timescale = shared_timescale
        self.normalize_hidden = normalize_hidden
        self.zero_diag_wrec = zero_diag_wrec
        self.stability_param = stability_param
        self.wrec_init = wrec_init

        if learn_timescales:
            # Timescales are learned
            timescales = None
            if shared_timescale:
                print(f"Timescales are LEARNABLE (SHARED, init at τ={init_timescale}s)")
            elif init_timescale is not None:
                print(f"Timescales are LEARNABLE (per-neuron, uniform init at τ={init_timescale}s)")
            else:
                print(f"Timescales are LEARNABLE (per-neuron, randomly initialized)")
        else:
            # Generate fixed timescales based on configuration
            if timescales_config is None:
                raise ValueError("timescales_config must be provided when learn_timescales=False")
            timescales = self._generate_timescales(hidden_size, timescales_config)

        self.rnn_step = MultiTimescaleRNNStep(
            input_size=input_size,
            hidden_size=hidden_size,
            dt=dt,
            timescales=timescales,
            activation=activation,
            learn_timescales=learn_timescales,
            init_timescale=init_timescale,
            shared_timescale=shared_timescale,
            normalize_hidden=normalize_hidden,
            zero_diag_wrec=zero_diag_wrec,
            recurrent_gain=recurrent_gain,
            noise_std=noise_std,
            alpha_parameterization=alpha_parameterization,
            dynamics_type=dynamics_type,
        )
        self.W_out = nn.Linear(hidden_size, output_size, bias=False)

        # Layer to initialize hidden state
        self.W_h_init = nn.Linear(output_size, hidden_size, bias=False)

        self._initialize_weights()

    def _generate_timescales(
        self, hidden_size: int, timescales_config: dict
    ) -> torch.Tensor:
        """
        Generate timescales based on configuration.

        :param hidden_size: Number of hidden units
        :param timescales_config: Configuration dictionary
        :return: Tensor of timescales of shape (hidden_size,)
        """
        timescale_type = timescales_config["type"]

        if timescale_type == "discrete":
            discrete_values = timescales_config["values"]
            discrete_values = torch.tensor(discrete_values, dtype=torch.float32)

            if len(discrete_values) == 1:
                timescales = discrete_values.repeat(hidden_size)
            else:
                # Each timescale picked uniformly from K discrete values
                indices = torch.randint(0, len(discrete_values), (hidden_size,))
                timescales = discrete_values[indices]

        elif timescale_type == "continuous":
            distribution = timescales_config["distribution"]

            if distribution == "uniform":
                min_timescale = float(timescales_config["min_timescale"])
                max_timescale = float(timescales_config["max_timescale"])
                timescales = torch.uniform(
                    min_timescale, max_timescale, size=(hidden_size,)
                )

            elif distribution == "powerlaw":
                # Power-law distribution: P(x) ∝ x^(-α)
                # We use inverse transform sampling since PyTorch doesn't have a built-in power-law
                exponent = float(timescales_config["exponent"])
                min_timescale = float(timescales_config["min_timescale"])
                max_timescale = float(timescales_config["max_timescale"])

                # Generate uniform random samples
                u = torch.rand(hidden_size)

                # Inverse transform sampling for bounded power-law
                # F^(-1)(u) = x_min * [(x_max/x_min)^(1-α) * u + (1-u)]^(1/(1-α))
                if abs(exponent - 1.0) < 1e-6:
                    # Special case when α ≈ 1 (log-uniform distribution)
                    timescales = min_timescale * torch.pow(
                        torch.tensor(max_timescale / min_timescale), u
                    )
                else:
                    # General case
                    ratio = max_timescale / min_timescale
                    exponent_term = 1.0 - exponent
                    power_term = (ratio**exponent_term) * u + (1 - u)
                    timescales = min_timescale * torch.pow(
                        power_term, 1.0 / exponent_term
                    )
            elif distribution == "gaussian":
                # Gaussian (normal) distribution
                mean = float(timescales_config["mean"])
                std = float(timescales_config["std"])
                
                # Generate samples from normal distribution
                timescales = torch.normal(mean, std, size=(hidden_size,))
                
                # Ensure all timescales are positive by clipping to a small minimum value
                # This prevents numerical issues with negative or zero timescales
                min_timescale = timescales_config.get("min_timescale", 0.01)
                max_timescale = timescales_config.get("max_timescale", 1.0)
                timescales = torch.clamp(timescales, min=min_timescale, max=max_timescale)

            else:
                raise ValueError(f"Unknown continuous distribution: {distribution}")

        return timescales

    def forward(
        self,
        inputs: torch.Tensor,
        init_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the multi-timescale RNN.

        :param inputs: (batch, time, input_size)
        :param init_context: (batch, output_size) - optional context to initialize hidden state.
                            If None, hidden state is initialized to zeros.

        :return: hidden_states: (batch, time, hidden_size)
        :return: outputs: (batch, time, output_size)
        """
        batch_size, seq_len, _ = inputs.shape

        # Initialize hidden state
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
            # W_ij ~ N(0, 1/N) — Sompolinsky et al. 1988
            nn.init.normal_(self.rnn_step.W_rec.weight, mean=0.0, std=1.0 / n**0.5)
        elif self.wrec_init == "levy_stable":
            # W_ij ~ S(stability_param, beta=0, loc=0, scale=1/sqrt(N))
            samples = levy_stable.rvs(
                alpha=self.stability_param,
                beta=0,
                loc=0,
                scale=1.0 / n**0.5,
                size=(n, n),
            )
            with torch.no_grad():
                self.rnn_step.W_rec.weight.copy_(torch.tensor(samples, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown wrec_init: {self.wrec_init}")
        nn.init.zeros_(self.rnn_step.W_rec.bias)

        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.xavier_uniform_(self.W_h_init.weight)

    def get_timescale_stats(self) -> dict:
        """Return statistics about the timescale values.
        
        Works for both fixed and learnable timescales.
        """
        # Use properties to get current values (works for both fixed and learned)
        timescales = self.rnn_step.current_timescales
        alphas = self.rnn_step.current_alphas
        
        return {
            "timescale_min": timescales.min().item(),
            "timescale_max": timescales.max().item(),
            "timescale_mean": timescales.mean().item(),
            "timescale_std": timescales.std().item(),
            "alpha_min": alphas.min().item(),
            "alpha_max": alphas.max().item(),
            "alpha_mean": alphas.mean().item(),
            "unique_timescales": len(torch.unique(timescales)),
            "learn_timescales": self.learn_timescales,
        }


class MultiTimescaleRNNLightning(L.LightningModule):
    def __init__(
        self,
        model: MultiTimescaleRNN,
        learning_rate: float,
        weight_decay: float,
        step_size: int,
        gamma: float,
        task: str = "path_integration",
        precondition_gradients: bool = False,
        eps_alpha: float = 1e-2,
    ) -> None:
        """
        Initialize the Multi-timescale RNN Lightning module.

        :param model: The MultiTimescaleRNN model.
        :param learning_rate: The learning rate.
        :param weight_decay: The weight decay for the recurrent weights.
        :param step_size: The step size for the learning rate scheduler.
        :param gamma: The gamma for the learning rate scheduler.
        :param task: The task type ("path_integration" or "binary_counter").
        :param precondition_gradients: If True, apply alpha-based gradient preconditioning.
        :param eps_alpha: Damping constant for numerical stability in preconditioner (1/(alpha + eps)).
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
        
        # Task-specific loss function
        if task in ("binary_counter", "flip_flop"):
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        if precondition_gradients:
            print(f"Gradient preconditioning ENABLED (eps_alpha={eps_alpha})")
        else:
            print(f"Gradient preconditioning DISABLED")

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """
        Compute task-specific loss.
        
        Returns:
            total_loss: Scalar loss value
            per_channel_losses: Dict mapping channel names to per-channel losses (None for path_integration)
        """
        if self.task in ("path_integration", "path_integration_1d"):
            # Cross-entropy loss for place cell prediction (same for 1D and 2D)
            y = targets.reshape(-1, self.model.output_size)
            yhat = torch.softmax(outputs.reshape(-1, self.model.output_size), dim=-1)
            loss = -(y * torch.log(yhat + 1e-8)).sum(-1).mean()
            return loss, None
        elif self.task in ("binary_counter", "flip_flop"):
            # BCE loss for binary state prediction
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

    def training_step(self, batch) -> torch.Tensor:
        inputs, aux_info, targets = batch
        
        if self.task in ("path_integration", "path_integration_1d"):
            # Use first place cell activation for initialization
            init_context = targets[:, 0, :]
        else:
            # Use zeros for initialization in other tasks
            init_context = None
            
        hidden_states, outputs = self.model(
            inputs=inputs, init_context=init_context
        )

        loss, per_channel_losses = self._compute_loss(outputs, targets)

        # Weight regularization on recurrent weights
        loss += self.weight_decay * (self.model.rnn_step.W_rec.weight**2).sum()

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        
        # Log per-channel losses for binary_counter task
        if per_channel_losses is not None:
            for channel_name, channel_loss in per_channel_losses.items():
                self.log(
                    f"train_loss_{channel_name}",
                    channel_loss,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )

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

        # Weight regularization on recurrent weights
        loss += self.weight_decay * (self.model.rnn_step.W_rec.weight**2).sum()

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        
        # Log per-channel losses for binary_counter task
        if per_channel_losses is not None:
            for channel_name, channel_loss in per_channel_losses.items():
                self.log(
                    f"val_loss_{channel_name}",
                    channel_loss,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return loss

    def configure_optimizers(self):
        """Configure the optimizer and scheduler for the Multi-timescale RNN model."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0,  # we do manual weight decay on recurrent weights in the loss
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        """Apply alpha-based gradient preconditioning before optimizer step.
        
        Scales gradients by P = diag(1/(alpha_i + eps)) to compensate for
        different update rates across neurons with different timescales.
        """
        if not self.precondition_gradients:
            return

        
        # Get alphas (detach to avoid gradient flow through preconditioner)
        alphas = self.model.rnn_step.current_alphas.detach()
        
        # Compute preconditioner: p_i = 1 / (alpha_i + eps)
        preconditioner = 1.0 / (alphas + self.eps_alpha)  # shape: (hidden_size,)
        
        # Apply to W_in gradient (shape: hidden_size × input_size)
        if self.model.rnn_step.W_in.weight.grad is not None:
            self.model.rnn_step.W_in.weight.grad *= preconditioner.unsqueeze(1)
        
        # Apply to W_in bias gradient (shape: hidden_size)
        if self.model.rnn_step.W_in.bias.grad is not None:
            self.model.rnn_step.W_in.bias.grad *= preconditioner
        
        # Apply to W_rec gradient (shape: hidden_size × hidden_size)
        if self.model.rnn_step.W_rec.weight.grad is not None:
            self.model.rnn_step.W_rec.weight.grad *= preconditioner.unsqueeze(1)
        
        # Apply to W_rec bias gradient (shape: hidden_size)
        if self.model.rnn_step.W_rec.bias.grad is not None:
            self.model.rnn_step.W_rec.bias.grad *= preconditioner
        
        # Apply to W_h_init gradient (shape: hidden_size × output_size)
        if self.model.W_h_init.weight.grad is not None:
            self.model.W_h_init.weight.grad *= preconditioner.unsqueeze(1)

    def on_train_start(self):
        """Log timescale statistics at the start of training."""
        timescale_stats = self.model.get_timescale_stats()
        print(f"Timescale statistics: {timescale_stats}")
        if self.precondition_gradients:
            print(f"Gradient preconditioning ENABLED (eps_alpha={self.eps_alpha})")
