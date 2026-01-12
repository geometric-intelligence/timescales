import torch
import torch.nn as nn
import lightning as L


class RNNStep(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        alpha: float,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        """
        A single time step of the RNN.
        """
        super(RNNStep, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.activation = activation()

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        h = (1 - self.alpha) * hidden + self.alpha * self.activation(
            self.W_in(input) + self.W_rec(hidden)
        )
        return h


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        alpha: float,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        """
        Initialize the Path Integrating RNN.
        :param input_size: The size of the velocity input (= dimension of space).
        :param hidden_size: The size of the hidden state (number of neurons/"grid cells").
        :param output_size: The size of the output vector (number of place cells).
        :param alpha: RNN update rate.
        :param activation: The activation function.
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn_step = RNNStep(input_size, hidden_size, alpha, activation)
        self.W_out = nn.Linear(hidden_size, output_size, bias=False)

        # Layer to initialize hidden state
        self.W_h_init = nn.Linear(output_size, hidden_size, bias=False)

        self.initialize_weights()

    def forward(
        self, inputs: torch.Tensor, init_context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.
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

    def initialize_weights(self) -> None:
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


class RNNLightning(L.LightningModule):
    def __init__(
        self,
        model: RNN,
        learning_rate: float,
        weight_decay: float,
        step_size: int,
        gamma: float,
        task: str = "path_integration",
    ) -> None:
        """
        Initialize the RNN Lightning module.
        :param model: The RNN model.
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
        """Configure the optimizer and scheduler for the Hypergraph RNN model."""
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
