import torch
import torch.nn as nn
import lightning as L


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
        normalize_hidden: bool = False,
        zero_diag_wrec: bool = True,
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
        :param normalize_hidden: If True, apply LayerNorm to hidden state after each step.
        :param zero_diag_wrec: If True, enforce diag(W_rec) = 0 (no self-connections).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.activation = activation()
        self.learn_timescales = learn_timescales
        self.normalize_hidden = normalize_hidden
        self.zero_diag_wrec = zero_diag_wrec

        # Optional LayerNorm for hidden state normalization
        if normalize_hidden:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None

        if learn_timescales:
            # Learnable timescales via log-parameterization
            # log_timescales is unconstrained; timescales = exp(log_timescales) > 0
            if init_timescale is not None:
                # Uniform initialization: all timescales start at init_timescale
                log_timescales = torch.full((hidden_size,), float(torch.log(torch.tensor(init_timescale))))
            else:
                # Random initialization
                # Initialize so that exp(log_timescales) gives τ roughly in [0.1, 1.0]
                # log(0.3) ≈ -1.2, so init ~ N(−0.5, 0.5) gives reasonable starting τ
                log_timescales = torch.randn(hidden_size) * 0.5 - 0.5
            self.log_timescales = nn.Parameter(log_timescales)
            # Register dt as buffer for alpha computation
            self.register_buffer("_dt", torch.tensor(dt))
        else:
            # Fixed timescales (original behavior)
            if timescales is None:
                raise ValueError("timescales must be provided when learn_timescales=False")
            self.register_buffer("timescales", timescales)
            alphas = 1 - torch.exp(-dt / timescales)
            self.register_buffer("alphas", alphas)

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        
        # Enforce zero diagonal on W_rec if requested (init to 0 + freeze via gradient hook)
        if zero_diag_wrec:
            self.W_rec.weight.data.fill_diagonal_(0)
            self.W_rec.weight.register_hook(lambda g: g.clone().fill_diagonal_(0))

    @property
    def current_timescales(self) -> torch.Tensor:
        """Get current timescale values (computed from log_timescales if learnable)."""
        if self.learn_timescales:
            return torch.exp(self.log_timescales)
        else:
            return self.timescales

    @property
    def current_alphas(self) -> torch.Tensor:
        """Get current alpha values (computed from log_timescales if learnable)."""
        if self.learn_timescales:
            timescales = torch.exp(self.log_timescales)
            return 1 - torch.exp(-self._dt / timescales)
        else:
            return self.alphas

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
        pre_activation = self.W_in(input) + self.W_rec(hidden)
        activated = self.activation(pre_activation)

        # Get alphas (either fixed or computed from learned log_timescales)
        alphas = self.current_alphas

        # Per-unit leaky integration: h_new = (1-α)*h_old + α*activated
        new_hidden = (1 - alphas) * hidden + alphas * activated

        # Optional LayerNorm
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
        normalize_hidden: bool = False,
        zero_diag_wrec: bool = False,
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
        :param normalize_hidden: If True, apply LayerNorm to hidden state after each step.
        :param zero_diag_wrec: If True, enforce diag(W_rec) = 0 (no self-connections).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt
        self.learn_timescales = learn_timescales
        self.init_timescale = init_timescale
        self.normalize_hidden = normalize_hidden
        self.zero_diag_wrec = zero_diag_wrec

        if learn_timescales:
            # Timescales are learned
            timescales = None
            if init_timescale is not None:
                print(f"Timescales are LEARNABLE (uniform init at τ={init_timescale}s)")
            else:
                print(f"Timescales are LEARNABLE (randomly initialized)")
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
            normalize_hidden=normalize_hidden,
            zero_diag_wrec=zero_diag_wrec,
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
        """Initialize weights for stable RNN training"""
        # 1. Input weights (W_in) - Xavier initialization
        nn.init.xavier_uniform_(self.rnn_step.W_in.weight)
        nn.init.zeros_(self.rnn_step.W_in.bias)

        # 2. Recurrent weights (W_rec) - Orthogonal initialization
        nn.init.orthogonal_(self.rnn_step.W_rec.weight)
        nn.init.zeros_(self.rnn_step.W_rec.bias)

        # 3. Output weights (W_out) - Xavier initialization
        nn.init.xavier_uniform_(self.W_out.weight)

        # 4. Initial hidden state encoder (W_h_init) - Xavier initialization
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
    ) -> None:
        """
        Initialize the Multi-timescale RNN Lightning module.

        :param model: The MultiTimescaleRNN model.
        :param learning_rate: The learning rate.
        :param weight_decay: The weight decay for the recurrent weights.
        :param step_size: The step size for the learning rate scheduler.
        :param gamma: The gamma for the learning rate scheduler.
        :param task: The task type ("path_integration" or "binary_counter").
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.task = task
        
        # Task-specific loss function
        if task == "binary_counter":
            self.loss_fn = nn.BCEWithLogitsLoss()

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss."""
        if self.task == "path_integration":
            # Cross-entropy loss for place cell prediction
            y = targets.reshape(-1, self.model.output_size)
            yhat = torch.softmax(outputs.reshape(-1, self.model.output_size), dim=-1)
            loss = -(y * torch.log(yhat + 1e-8)).sum(-1).mean()
        elif self.task == "binary_counter":
            # BCE loss for binary state prediction
            loss = self.loss_fn(
                outputs.reshape(-1, self.model.output_size),
                targets.reshape(-1, self.model.output_size)
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
        return loss

    def training_step(self, batch) -> torch.Tensor:
        inputs, aux_info, targets = batch
        
        if self.task == "path_integration":
            # Use first place cell activation for initialization
            init_context = targets[:, 0, :]
        else:
            # Use zeros for initialization in other tasks
            init_context = None
            
        hidden_states, outputs = self.model(
            inputs=inputs, init_context=init_context
        )

        loss = self._compute_loss(outputs, targets)

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

        return loss

    def validation_step(self, batch) -> torch.Tensor:
        inputs, aux_info, targets = batch
        
        if self.task == "path_integration":
            init_context = targets[:, 0, :]
        else:
            init_context = None
            
        hidden_states, outputs = self.model(
            inputs=inputs, init_context=init_context
        )

        loss = self._compute_loss(outputs, targets)

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

    def on_train_start(self):
        """Log timescale statistics at the start of training."""
        timescale_stats = self.model.get_timescale_stats()
        print(f"Timescale statistics: {timescale_stats}")
