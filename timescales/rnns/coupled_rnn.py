import numpy as np
import torch
import torch.nn as nn
import lightning as L


class CoupledRNNStep(nn.Module):
    """
    Single timestep of two coupled networks:
      Network 1 (nonlinear): tau_r * dr/dt = -r + phi(W_rec r + W_in u + V s)
      Network 2 (linear):    tau_s * ds/dt = -s + W_s s + U r

    Discretized via alpha = 1 - exp(-dt/tau):
      r_new = (1 - alpha_r) r + alpha_r phi(W_rec r + W_in u + V s)
      s_new = (1 - alpha_s) s + alpha_s (W_s s + U r)

    Both updates use the "old" r, s (symmetric / parallel evaluation).

    Trainable: W_rec, W_in, V, U.   W_s fixed by default (trainable_w_s=True to learn it).
    """

    def __init__(
        self,
        input_size: int,
        r_hidden_size: int,
        s_hidden_size: int,
        dt: float,
        tau_r: float,
        tau_s: float,
        activation: type[nn.Module] = nn.Tanh,
        zero_diag_wrec: bool = True,
        recurrent_gain: float = 1.0,
        noise_std: float = 0.0,
        w_s_gain: float = 1.0,
        trainable_w_s: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.r_hidden_size = r_hidden_size
        self.s_hidden_size = s_hidden_size
        self.dt = dt
        self.tau_r = tau_r
        self.tau_s = tau_s
        self.activation = activation()
        self.zero_diag_wrec = zero_diag_wrec
        self.recurrent_gain = recurrent_gain
        self.w_s_gain = w_s_gain
        self.noise_std = noise_std

        alpha_r = 1.0 - np.exp(-dt / tau_r)
        alpha_s = 1.0 - np.exp(-dt / tau_s)
        self.register_buffer("alpha_r", torch.tensor(alpha_r))
        self.register_buffer("alpha_s", torch.tensor(alpha_s))

        # --- r-network (nonlinear) trainable weights ---
        self.W_in = nn.Linear(input_size, r_hidden_size)
        self.W_rec = nn.Linear(r_hidden_size, r_hidden_size)
        self.V = nn.Linear(s_hidden_size, r_hidden_size, bias=False)

        if zero_diag_wrec:
            self.W_rec.weight.data.fill_diagonal_(0)
            self.W_rec.weight.register_hook(lambda g: g.clone().fill_diagonal_(0))

        # --- s-network (linear) ---
        # U is trainable
        self.U = nn.Linear(r_hidden_size, s_hidden_size, bias=False)

        W_s_data = torch.randn(s_hidden_size, s_hidden_size) / s_hidden_size**0.5
        if trainable_w_s:
            self.W_s = nn.Parameter(W_s_data)
        else:
            self.register_buffer("W_s", W_s_data)

    def forward(
        self,
        input: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One timestep.

        :param input: (batch, input_size)
        :param r: (batch, r_hidden_size)
        :param s: (batch, s_hidden_size)
        :return: (r_new, s_new)
        """
        # r-network drive
        pre_act = (
            self.recurrent_gain * self.W_rec(r) + self.W_in(input) + self.V(s)
        )
        drive_r = self.activation(pre_act)
        r_new = (1 - self.alpha_r) * r + self.alpha_r * drive_r

        # s-network drive (linear)
        drive_s = self.w_s_gain * torch.nn.functional.linear(s, self.W_s) + self.U(r)
        s_new = (1 - self.alpha_s) * s + self.alpha_s * drive_s

        if self.noise_std > 0.0 and self.training:
            noise_scale_r = self.noise_std * (self.dt / self.tau_r) ** 0.5
            noise_scale_s = self.noise_std * (self.dt / self.tau_s) ** 0.5
            r_new = r_new + torch.randn_like(r_new) * noise_scale_r
            s_new = s_new + torch.randn_like(s_new) * noise_scale_s

        return r_new, s_new


class CoupledRNN(nn.Module):
    """
    Coupled RNN with a nonlinear r-network and a linear s-network.

    Forward signature matches the existing MultiTimescaleRNN convention,
    but returns three tensors: (r_states, s_states, outputs).
    """

    def __init__(
        self,
        input_size: int,
        r_hidden_size: int,
        s_hidden_size: int,
        output_size: int,
        dt: float,
        tau_r: float,
        tau_s: float,
        activation: type[nn.Module] = nn.Tanh,
        zero_diag_wrec: bool = True,
        recurrent_gain: float = 1.0,
        noise_std: float = 0.0,
        wrec_init: str = "orthogonal",
        w_s_gain: float = 1.0,
        trainable_w_s: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.r_hidden_size = r_hidden_size
        self.s_hidden_size = s_hidden_size
        self.output_size = output_size
        self.dt = dt
        self.tau_r = tau_r
        self.tau_s = tau_s
        self.wrec_init = wrec_init

        self.rnn_step = CoupledRNNStep(
            input_size=input_size,
            r_hidden_size=r_hidden_size,
            s_hidden_size=s_hidden_size,
            dt=dt,
            tau_r=tau_r,
            tau_s=tau_s,
            activation=activation,
            zero_diag_wrec=zero_diag_wrec,
            recurrent_gain=recurrent_gain,
            noise_std=noise_std,
            w_s_gain=w_s_gain,
            trainable_w_s=trainable_w_s,
        )

        self.W_out = nn.Linear(r_hidden_size, output_size, bias=False)

        self._initialize_weights()

    def forward(
        self,
        inputs: torch.Tensor,
        init_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param inputs: (batch, time, input_size)
        :param init_context: unused, kept for API compatibility.
        :return: r_states  (batch, time, r_hidden_size)
                 s_states  (batch, time, s_hidden_size)
                 outputs   (batch, time, output_size)
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        r = torch.zeros(batch_size, self.r_hidden_size, device=device)
        s = torch.zeros(batch_size, self.s_hidden_size, device=device)

        r_states, s_states, outputs = [], [], []

        for t in range(seq_len):
            r, s = self.rnn_step(inputs[:, t, :], r, s)
            r_states.append(r)
            s_states.append(s)
            outputs.append(self.W_out(r))

        return (
            torch.stack(r_states, dim=1),
            torch.stack(s_states, dim=1),
            torch.stack(outputs, dim=1),
        )

    def _initialize_weights(self) -> None:
        n = self.r_hidden_size

        nn.init.xavier_uniform_(self.rnn_step.W_in.weight)
        nn.init.zeros_(self.rnn_step.W_in.bias)

        if self.wrec_init == "orthogonal":
            nn.init.orthogonal_(self.rnn_step.W_rec.weight)
        elif self.wrec_init == "normal_scaled":
            nn.init.normal_(self.rnn_step.W_rec.weight, mean=0.0, std=1.0 / n**0.5)
        else:
            raise ValueError(f"Unknown wrec_init: {self.wrec_init}")
        if self.rnn_step.zero_diag_wrec:
            self.rnn_step.W_rec.weight.data.fill_diagonal_(0)
        nn.init.zeros_(self.rnn_step.W_rec.bias)

        nn.init.xavier_uniform_(self.rnn_step.V.weight)
        nn.init.xavier_uniform_(self.rnn_step.U.weight)
        nn.init.xavier_uniform_(self.W_out.weight)


class CoupledRNNLightning(L.LightningModule):
    def __init__(
        self,
        model: CoupledRNN,
        learning_rate: float,
        weight_decay: float,
        step_size: int,
        gamma: float,
        task: str = "flip_flop",
        lr_interval: str = "epoch",
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.task = task
        self.lr_interval = lr_interval

        if task in ("binary_counter", "flip_flop"):
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        elif task == "teacher_student":
            self.loss_fn = nn.MSELoss(reduction="none")

    # ------------------------------------------------------------------
    # Loss & accuracy (same logic as MultiTimescaleRNNLightning)
    # ------------------------------------------------------------------

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if self.task in ("path_integration", "path_integration_1d"):
            y = targets.reshape(-1, self.model.output_size)
            yhat = torch.softmax(outputs.reshape(-1, self.model.output_size), dim=-1)
            loss = -(y * torch.log(yhat + 1e-8)).sum(-1).mean()
            return loss, None

        if self.task in ("binary_counter", "flip_flop"):
            n_channels = outputs.shape[-1]
            outputs_flat = outputs.reshape(-1, n_channels)
            targets_flat = targets.reshape(-1, n_channels)
            per_sample = self.loss_fn(outputs_flat, targets_flat)
            per_channel = per_sample.mean(dim=0)
            total = per_channel.mean()
            per_ch = {f"channel_{i}": per_channel[i].item() for i in range(n_channels)}
            return total, per_ch

        if self.task == "teacher_student":
            n_channels = outputs.shape[-1]
            outputs_flat = outputs.reshape(-1, n_channels)
            targets_flat = targets.reshape(-1, n_channels)
            per_sample = self.loss_fn(outputs_flat, targets_flat)
            per_channel = per_sample.mean(dim=0)
            total = per_channel.mean()
            per_ch = {f"channel_{i}": per_channel[i].item() for i in range(n_channels)}
            return total, per_ch

        raise ValueError(f"Unknown task: {self.task}")

    def _compute_accuracy(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor | None, dict[str, float] | None]:
        if self.task in ("binary_counter", "flip_flop"):
            preds = (torch.sigmoid(outputs) > 0.5).float()
            per_ch_acc = (preds == targets).float().mean(dim=(0, 1))
            overall = per_ch_acc.mean()
            per_ch = {f"channel_{i}": per_ch_acc[i].item() for i in range(per_ch_acc.shape[0])}
            return overall, per_ch
        if self.task in ("path_integration", "path_integration_1d"):
            pred_idx = outputs.reshape(-1, self.model.output_size).argmax(dim=-1)
            tgt_idx = targets.reshape(-1, self.model.output_size).argmax(dim=-1)
            return (pred_idx == tgt_idx).float().mean(), None
        if self.task == "teacher_student":
            mse = ((outputs - targets) ** 2).mean()
            var = targets.var()
            r2 = 1.0 - mse / (var + 1e-8)
            per_ch_r2 = {}
            for i in range(outputs.shape[-1]):
                ch_mse = ((outputs[..., i] - targets[..., i]) ** 2).mean()
                ch_var = targets[..., i].var()
                per_ch_r2[f"channel_{i}"] = (1.0 - ch_mse / (ch_var + 1e-8)).item()
            return r2, per_ch_r2
        return None, None

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def _shared_step(self, batch, prefix: str) -> torch.Tensor:
        inputs, _aux, targets = batch

        init_context = None
        if self.task in ("path_integration", "path_integration_1d"):
            init_context = targets[:, 0, :]

        _r_states, _s_states, outputs = self.model(inputs=inputs, init_context=init_context)

        loss, per_ch_losses = self._compute_loss(outputs, targets)
        loss += self.weight_decay * (self.model.rnn_step.W_rec.weight ** 2).sum()

        on_step = prefix == "train"
        self.log(f"{prefix}_loss", loss, on_step=on_step, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        if per_ch_losses is not None:
            for ch, v in per_ch_losses.items():
                self.log(f"{prefix}_loss_{ch}", v, on_step=on_step,
                         on_epoch=True, sync_dist=True)

        accuracy, per_ch_acc = self._compute_accuracy(outputs, targets)
        if accuracy is not None:
            self.log(f"{prefix}_accuracy", accuracy, on_step=on_step,
                     on_epoch=True, prog_bar=True, sync_dist=True)
            if per_ch_acc is not None:
                for ch, v in per_ch_acc.items():
                    self.log(f"{prefix}_accuracy_{ch}", v, on_step=on_step,
                             on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.0
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
