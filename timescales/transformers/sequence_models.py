"""A hierarchy of linear and nonlinear residual-stream sequence models.

The architecture names deliberately distinguish a globally linear sequence
map from "linear attention", whose attention pattern is bilinear in the input.
All models consume ``[batch, time, input]`` tensors and predict one output
vector at every causal sequence position.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normal_fan_in_(weight: torch.Tensor, fan_in: int, scale: float = 1.0) -> None:
    nn.init.normal_(weight, mean=0.0, std=float(scale) / math.sqrt(fan_in))


class _OutputScaledModel(nn.Module):
    """Shared Clark-style output scaling for transformer experiments."""

    def __init__(self, d_model: int, output_coupling_gamma: float | None) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if output_coupling_gamma is not None and output_coupling_gamma <= 0.0:
            raise ValueError("output_coupling_gamma must be positive")
        self.d_model = int(d_model)
        self.output_coupling_gamma = output_coupling_gamma

    @property
    def readout_scale(self) -> float:
        if self.output_coupling_gamma is None:
            return 1.0
        return 1.0 / (self.d_model * self.output_coupling_gamma)


class LinearFIR(_OutputScaledModel):
    """Unfactorized causal finite-impulse-response linear predictor."""

    architecture = "linear_fir"
    is_input_linear = True

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int,
        max_context_length: int,
        residual_init_scale: float = 0.08,
        output_coupling_gamma: float | None = None,
    ) -> None:
        super().__init__(d_model, output_coupling_gamma)
        if input_size <= 0 or output_size <= 0 or max_context_length <= 0:
            raise ValueError("input, output, and context sizes must be positive")
        if residual_init_scale < 0.0:
            raise ValueError("residual_init_scale must be nonnegative")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.max_context_length = int(max_context_length)
        self.residual_init_scale = float(residual_init_scale)
        self.kernel = nn.Parameter(
            torch.empty(output_size, input_size, max_context_length)
        )
        nn.init.normal_(self.kernel, mean=0.0, std=self.residual_init_scale)

    @property
    def effective_readout_weight(self) -> torch.Tensor:
        return self.readout_scale * self.kernel

    def forward(
        self, inputs: torch.Tensor, *, return_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"expected inputs [B, T, {self.input_size}], got {tuple(inputs.shape)}"
            )
        padded = F.pad(
            inputs.transpose(1, 2),
            (self.max_context_length - 1, 0),
        )
        outputs = self.readout_scale * F.conv1d(
            padded,
            self.kernel.flip(-1),
        )
        outputs = outputs.transpose(1, 2)
        if not return_cache:
            return outputs
        return outputs, {
            "residual_streams": [],
            "attention_patterns": [],
            "head_results": [],
            "effective_lag_kernel": self.readout_scale * self.kernel,
        }


class StaticLinearAttention(_OutputScaledModel):
    """Factorized transformer-shaped sequence mixer linear in its input."""

    architecture = "static_linear_attention"
    is_input_linear = True

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int,
        n_heads: int,
        max_context_length: int,
        residual_init_scale: float = 0.08,
        residual_gain: float = 1.0,
        output_coupling_gamma: float | None = None,
    ) -> None:
        super().__init__(d_model, output_coupling_gamma)
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(input_size, output_size, n_heads, max_context_length) <= 0:
            raise ValueError("all model dimensions must be positive")
        if residual_init_scale < 0.0:
            raise ValueError("residual_init_scale must be nonnegative")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.max_context_length = int(max_context_length)
        self.residual_init_scale = float(residual_init_scale)
        self.residual_gain = float(residual_gain)

        self.W_embed = nn.Linear(input_size, d_model, bias=False)
        self.W_v = nn.Parameter(torch.empty(n_heads, d_model, self.d_head))
        self.W_o = nn.Parameter(torch.empty(n_heads, self.d_head, d_model))
        self.lag_kernel = nn.Parameter(
            torch.empty(n_heads, max_context_length)
        )
        self.W_unembed = nn.Linear(d_model, output_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _normal_fan_in_(self.W_embed.weight, self.input_size)
        _normal_fan_in_(self.W_v, self.d_model)
        _normal_fan_in_(self.W_o, self.d_head, self.residual_init_scale)
        nn.init.normal_(
            self.lag_kernel,
            mean=0.0,
            std=1.0 / math.sqrt(self.max_context_length),
        )
        nn.init.normal_(self.W_unembed.weight, mean=0.0, std=1.0)

    @property
    def effective_readout_weight(self) -> torch.Tensor:
        return self.readout_scale * self.W_unembed.weight

    def forward(
        self, inputs: torch.Tensor, *, return_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"expected inputs [B, T, {self.input_size}], got {tuple(inputs.shape)}"
            )
        embedded = self.W_embed(inputs)
        values = torch.einsum("btd,hdf->bthf", embedded, self.W_v)
        values_by_head = values.permute(0, 2, 3, 1)
        values_flat = values_by_head.reshape(
            inputs.shape[0], self.n_heads * self.d_head, inputs.shape[1]
        )
        padded = F.pad(
            values_flat,
            (self.max_context_length - 1, 0),
        )
        filters = self.lag_kernel.repeat_interleave(
            self.d_head, dim=0
        ).unsqueeze(1)
        mixed_flat = F.conv1d(
            padded,
            filters.flip(-1),
            groups=self.n_heads * self.d_head,
        )
        mixed_values = mixed_flat.reshape(
            inputs.shape[0], self.n_heads, self.d_head, inputs.shape[1]
        ).permute(0, 3, 1, 2)
        head_results = torch.einsum(
            "bthd,hdf->bthf", mixed_values, self.W_o
        )
        residual = embedded + self.residual_gain * head_results.sum(dim=2)
        outputs = self.readout_scale * self.W_unembed(residual)
        if not return_cache:
            return outputs
        return outputs, {
            "residual_streams": [embedded, residual],
            "attention_patterns": [self.lag_kernel],
            "head_results": [head_results],
            "mlp_outputs": [],
        }


class AttentionSequenceModel(_OutputScaledModel):
    """One-layer causal residual transformer with optional softmax and MLP."""

    is_input_linear = False

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int,
        n_heads: int,
        max_context_length: int,
        attention_type: str,
        use_mlp: bool = False,
        d_mlp: int | None = None,
        residual_init_scale: float = 0.08,
        residual_gain: float = 1.0,
        attention_logit_scale: float = 1.0,
        output_coupling_gamma: float | None = None,
    ) -> None:
        super().__init__(d_model, output_coupling_gamma)
        if attention_type not in {"linear", "softmax"}:
            raise ValueError("attention_type must be 'linear' or 'softmax'")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(input_size, output_size, n_heads, max_context_length) <= 0:
            raise ValueError("all model dimensions must be positive")
        if residual_init_scale < 0.0:
            raise ValueError("residual_init_scale must be nonnegative")
        if attention_logit_scale <= 0.0:
            raise ValueError("attention_logit_scale must be positive")

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.max_context_length = int(max_context_length)
        self.attention_type = attention_type
        self.use_mlp = bool(use_mlp)
        self.d_mlp = int(d_mlp or 2 * d_model)
        self.residual_init_scale = float(residual_init_scale)
        self.residual_gain = float(residual_gain)
        self.attention_logit_scale = float(attention_logit_scale)
        if attention_type == "linear":
            self.architecture = "linear_attention"
        elif use_mlp:
            self.architecture = "softmax_tanh"
        else:
            self.architecture = "softmax_attention"

        self.W_embed = nn.Linear(input_size, d_model, bias=False)
        self.W_q = nn.Parameter(torch.empty(n_heads, d_model, self.d_head))
        self.W_k = nn.Parameter(torch.empty(n_heads, d_model, self.d_head))
        self.W_v = nn.Parameter(torch.empty(n_heads, d_model, self.d_head))
        self.W_o = nn.Parameter(torch.empty(n_heads, self.d_head, d_model))
        self.relative_scores = nn.Parameter(
            torch.zeros(n_heads, max_context_length)
        )
        if self.use_mlp:
            self.W_mlp_in = nn.Linear(d_model, self.d_mlp, bias=False)
            self.W_mlp_out = nn.Linear(self.d_mlp, d_model, bias=False)
        self.W_unembed = nn.Linear(d_model, output_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _normal_fan_in_(self.W_embed.weight, self.input_size)
        _normal_fan_in_(self.W_q, self.d_model)
        _normal_fan_in_(self.W_k, self.d_model)
        _normal_fan_in_(self.W_v, self.d_model)
        _normal_fan_in_(self.W_o, self.d_head, self.residual_init_scale)
        nn.init.zeros_(self.relative_scores)
        if self.use_mlp:
            _normal_fan_in_(self.W_mlp_in.weight, self.d_model)
            _normal_fan_in_(
                self.W_mlp_out.weight,
                self.d_mlp,
                self.residual_init_scale,
            )
        nn.init.normal_(self.W_unembed.weight, mean=0.0, std=1.0)

    @property
    def effective_readout_weight(self) -> torch.Tensor:
        return self.readout_scale * self.W_unembed.weight

    def _relative_score_matrix(
        self, sequence_length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_index = torch.arange(sequence_length, device=device)[:, None]
        key_index = torch.arange(sequence_length, device=device)[None, :]
        lags = query_index - key_index
        causal = lags >= 0
        within_context = lags < self.max_context_length
        valid = causal & within_context
        safe_lags = lags.clamp(min=0, max=self.max_context_length - 1)
        relative = self.relative_scores[:, safe_lags]
        return relative, valid

    def forward(
        self, inputs: torch.Tensor, *, return_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"expected inputs [B, T, {self.input_size}], got {tuple(inputs.shape)}"
            )
        embedded = self.W_embed(inputs)
        queries = torch.einsum("btd,hdf->bthf", embedded, self.W_q)
        keys = torch.einsum("btd,hdf->bthf", embedded, self.W_k)
        values = torch.einsum("btd,hdf->bthf", embedded, self.W_v)
        scores = torch.einsum("bthd,bshd->bhts", queries, keys)
        scores = (
            self.attention_logit_scale * scores / math.sqrt(self.d_head)
        )
        relative, valid = self._relative_score_matrix(inputs.shape[1], inputs.device)
        scores = scores + relative.unsqueeze(0)

        if self.attention_type == "softmax":
            scores = scores.masked_fill(~valid[None, None, :, :], -torch.inf)
            attention = torch.softmax(scores, dim=-1)
        else:
            attention = scores.masked_fill(~valid[None, None, :, :], 0.0)
            attention = attention / self.max_context_length

        mixed_values = torch.einsum("bhts,bshd->bthd", attention, values)
        head_results = torch.einsum(
            "bthd,hdf->bthf", mixed_values, self.W_o
        )
        post_attention = embedded + self.residual_gain * head_results.sum(dim=2)
        residual = post_attention
        mlp_output = None
        if self.use_mlp:
            mlp_output = self.W_mlp_out(torch.tanh(self.W_mlp_in(residual)))
            residual = residual + self.residual_gain * mlp_output
        outputs = self.readout_scale * self.W_unembed(residual)
        if not return_cache:
            return outputs
        residual_streams = [embedded, post_attention]
        if self.use_mlp:
            residual_streams.append(residual)
        return outputs, {
            "residual_streams": residual_streams,
            "attention_patterns": [attention],
            "head_results": [head_results],
            "mlp_outputs": [] if mlp_output is None else [mlp_output],
        }


def create_sequence_model(architecture: str, **kwargs) -> nn.Module:
    """Construct a sequence model from its explicit architecture name."""
    architecture = architecture.lower()
    if architecture == "linear_fir":
        allowed = {
            "input_size",
            "output_size",
            "d_model",
            "max_context_length",
            "residual_init_scale",
            "output_coupling_gamma",
        }
        return LinearFIR(
            **{key: value for key, value in kwargs.items() if key in allowed}
        )
    if architecture == "static_linear_attention":
        allowed = {
            "input_size",
            "output_size",
            "d_model",
            "n_heads",
            "max_context_length",
            "residual_init_scale",
            "residual_gain",
            "output_coupling_gamma",
        }
        return StaticLinearAttention(
            **{key: value for key, value in kwargs.items() if key in allowed}
        )
    if architecture in {"linear_attention", "softmax_attention", "softmax_tanh"}:
        return AttentionSequenceModel(
            **kwargs,
            attention_type=(
                "linear" if architecture == "linear_attention" else "softmax"
            ),
            use_mlp=architecture == "softmax_tanh",
        )
    raise ValueError(f"unknown sequence architecture: {architecture!r}")
