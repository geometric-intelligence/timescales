"""
Schur-parameterized RNN.

The recurrent Jacobian is factored as:
    J = Q T Q^T       (real Schur decomposition)
where Q is orthogonal and T is quasi-upper-triangular (real).

The effective recurrent weight is then:
    W_rec_eff = (J - (1 - alpha) * I) / (alpha * g)

Training modes (controlled by train_q / train_t):
  - train_t=True,  train_q=False  →  learn eigenvalue structure, fix basis
  - train_t=False, train_q=True   →  learn basis, fix eigenvalue structure
  - train_t=True,  train_q=True   →  learn both (~ unconstrained W_rec)

Q parameterization (q_parameterization):
  - "cayley"  →  Q = (I - A)(I + A)^{-1}  for a skew-symmetric A = U - U^T.
                 Q stays exactly on SO(N) for any A, at the cost of a linear
                 solve per forward pass.
  - "free"    →  Q is an unconstrained nn.Parameter (faster but drifts off
                 the orthogonal group).  J is reconstructed as Q T Q^{-1}.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from scipy.linalg import schur


# ---------------------------------------------------------------------------
# Cayley map helpers
# ---------------------------------------------------------------------------

def _skew_from_upper(upper_tri: torch.Tensor, n: int) -> torch.Tensor:
    """Build a skew-symmetric matrix from a flat upper-triangle parameter vector."""
    A = torch.zeros(n, n, device=upper_tri.device, dtype=upper_tri.dtype)
    idx = torch.triu_indices(n, n, offset=1)
    A[idx[0], idx[1]] = upper_tri
    A = A - A.T
    return A


def _cayley(A: torch.Tensor) -> torch.Tensor:
    """
    Cayley map: Q = (I - A)(I + A)^{-1} for skew-symmetric A.
    Returns an orthogonal matrix Q.
    """
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I + A, I - A)


# ---------------------------------------------------------------------------
# SchurRNNStep
# ---------------------------------------------------------------------------

class SchurRNNStep(nn.Module):
    """
    Single-timescale rate RNN step with Schur-parameterized recurrent weights.

    Dynamics (Identity activation, rate form):
        h_{t+1} = (1 - alpha) h_t  +  alpha * (g * W_rec_eff @ h_t  +  W_in @ u_t)

    where W_rec_eff is reconstructed from the Schur factors on every forward call.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float,
        tau: float,
        recurrent_gain: float = 1.0,
        noise_std: float = 0.0,
        wrec_init: str = "normal_scaled",
        train_t: bool = True,
        train_q: bool = False,
        q_parameterization: str = "cayley",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.recurrent_gain = recurrent_gain
        self.noise_std = noise_std
        self.train_t = train_t
        self.train_q = train_q
        self.q_parameterization = q_parameterization

        alpha = 1.0 - np.exp(-dt / tau)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        # Input projection (always trainable)
        self.W_in = nn.Linear(input_size, hidden_size)

        # --- Build initial W_rec and decompose ---
        n = hidden_size
        if wrec_init == "normal_scaled":
            W_init = np.random.randn(n, n) / np.sqrt(n)
        elif wrec_init == "orthogonal":
            W_init = np.linalg.svd(np.random.randn(n, n))[0]
        else:
            raise ValueError(f"Unknown wrec_init: {wrec_init!r}")

        # Jacobian at init: J = (1-alpha)*I + alpha*g*W
        J_init = (1.0 - alpha) * np.eye(n) + alpha * recurrent_gain * W_init
        T_np, Q_np = schur(J_init)   # J = Q T Q^T, T real quasi-upper-tri, Q orthogonal

        T_t = torch.tensor(T_np, dtype=torch.float32)
        Q_t = torch.tensor(Q_np, dtype=torch.float32)

        if train_t:
            self.T = nn.Parameter(T_t)
        else:
            self.register_buffer("T", T_t)

        if train_q:
            if q_parameterization == "cayley":
                # Store upper-triangle of skew-symmetric A (Q_init is orthogonal,
                # so we initialise A = 0 → Q = I, which is a valid choice since
                # we baked the initial Q into J via the Schur decomp; the actual
                # starting Q is encoded in T already when we reconstruct W_rec).
                n_upper = n * (n - 1) // 2
                self.A_upper = nn.Parameter(torch.zeros(n_upper))
                # Keep Q_init as a fixed reference for the "zero-A" meaning
                self.register_buffer("Q_init", Q_t)
            else:  # "free"
                self.Q_param = nn.Parameter(Q_t.clone())
        else:
            self.register_buffer("Q", Q_t)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_Q(self) -> torch.Tensor:
        if not self.train_q:
            return self.Q
        if self.q_parameterization == "cayley":
            n = self.hidden_size
            A = _skew_from_upper(self.A_upper, n)
            # Apply Q_init on the right so that A=0 recovers the initial Q
            return _cayley(A) @ self.Q_init
        else:
            return self.Q_param

    def _effective_W_rec(self) -> torch.Tensor:
        """Reconstruct W_rec from current Q and T."""
        Q = self._effective_Q()
        n = self.hidden_size
        I = torch.eye(n, device=Q.device, dtype=Q.dtype)

        if self.train_q and self.q_parameterization == "free":
            # Q may not be orthogonal; use solve instead of Q.T
            J = Q @ self.T @ torch.linalg.solve(Q, I)
        else:
            J = Q @ self.T @ Q.T

        W_rec_eff = (J - (1.0 - self.alpha) * I) / (self.alpha * self.recurrent_gain)
        return W_rec_eff

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        :param input:  (batch, input_size)
        :param hidden: (batch, hidden_size)
        :return:       (batch, hidden_size)
        """
        W_rec = self._effective_W_rec()
        pre_act = self.recurrent_gain * F.linear(hidden, W_rec) + self.W_in(input)
        new_hidden = (1.0 - self.alpha) * hidden + self.alpha * pre_act

        if self.noise_std > 0.0 and self.training:
            noise_scale = self.noise_std * (self.dt / self.tau) ** 0.5
            new_hidden = new_hidden + torch.randn_like(new_hidden) * noise_scale

        return new_hidden


# ---------------------------------------------------------------------------
# SchurRNN
# ---------------------------------------------------------------------------

class SchurRNN(nn.Module):
    """
    Rate RNN with Schur-parameterized recurrent weight.

    Only supports Identity activation (linear dynamics), since the scientific
    question is about the eigenstructure of the discrete-time Jacobian.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dt: float,
        tau: float,
        recurrent_gain: float = 1.0,
        noise_std: float = 0.0,
        wrec_init: str = "normal_scaled",
        train_t: bool = True,
        train_q: bool = False,
        q_parameterization: str = "cayley",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn_step = SchurRNNStep(
            input_size=input_size,
            hidden_size=hidden_size,
            dt=dt,
            tau=tau,
            recurrent_gain=recurrent_gain,
            noise_std=noise_std,
            wrec_init=wrec_init,
            train_t=train_t,
            train_q=train_q,
            q_parameterization=q_parameterization,
        )

        self.W_out = nn.Linear(hidden_size, output_size, bias=False)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.xavier_uniform_(self.rnn_step.W_in.weight)
        nn.init.zeros_(self.rnn_step.W_in.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        init_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: (batch, time, input_size)
        :return:  hidden_states (batch, time, hidden_size),
                  outputs       (batch, time, output_size)
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        h = torch.zeros(batch_size, self.hidden_size, device=device)

        hidden_states, outputs = [], []
        for t in range(seq_len):
            h = self.rnn_step(inputs[:, t, :], h)
            hidden_states.append(h)
            outputs.append(self.W_out(h))

        return torch.stack(hidden_states, dim=1), torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# SchurRNNLightning
# ---------------------------------------------------------------------------

class SchurRNNLightning(L.LightningModule):
    def __init__(
        self,
        model: SchurRNN,
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
        else:
            raise ValueError(f"SchurRNN only supports flip_flop / binary_counter; got {task!r}")

    def _compute_loss(self, outputs, targets):
        n_ch = outputs.shape[-1]
        per_sample = self.loss_fn(
            outputs.reshape(-1, n_ch), targets.reshape(-1, n_ch)
        )
        per_ch = per_sample.mean(dim=0)
        total = per_ch.mean()
        per_ch_dict = {f"channel_{i}": per_ch[i].item() for i in range(n_ch)}
        return total, per_ch_dict

    def _compute_accuracy(self, outputs, targets):
        preds = (torch.sigmoid(outputs) > 0.5).float()
        per_ch = (preds == targets).float().mean(dim=(0, 1))
        overall = per_ch.mean()
        per_ch_dict = {f"channel_{i}": per_ch[i].item() for i in range(per_ch.shape[0])}
        return overall, per_ch_dict

    def _shared_step(self, batch, prefix: str) -> torch.Tensor:
        inputs, _aux, targets = batch
        _hidden, outputs = self.model(inputs)

        loss, per_ch_losses = self._compute_loss(outputs, targets)

        on_step = prefix == "train"
        self.log(f"{prefix}_loss", loss, on_step=on_step, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        for ch, v in per_ch_losses.items():
            self.log(f"{prefix}_loss_{ch}", v, on_step=on_step,
                     on_epoch=True, sync_dist=True)

        accuracy, per_ch_acc = self._compute_accuracy(outputs, targets)
        self.log(f"{prefix}_accuracy", accuracy, on_step=on_step,
                 on_epoch=True, prog_bar=True, sync_dist=True)
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
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
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
