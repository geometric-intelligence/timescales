"""Optimizers used by the RNN training code."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer


class SGLD(Optimizer):
    """Unpreconditioned stochastic-gradient Langevin dynamics.

    For step size ``lr`` and inverse temperature ``beta``, each update is

    ``theta <- theta - lr * grad + sqrt(2 * lr / beta) * Normal(0, I)``.

    This is the update used by Clark et al.'s released RNN code.  Their
    experiments use full-batch gradients; this optimizer deliberately does not
    make assumptions about the data loader, so it can also be used with this
    project's online task batches.  Quadratic parameter priors belong in the
    objective and are therefore not implemented as optimizer weight decay.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        beta: float,
        add_noise: bool = True,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"SGLD lr must be positive, got {lr}")
        if beta <= 0.0:
            raise ValueError(f"SGLD beta must be positive, got {beta}")
        defaults = {"lr": float(lr), "beta": float(beta), "add_noise": add_noise}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            add_noise = group["add_noise"]
            noise_std = math.sqrt(2.0 * lr / beta)

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("SGLD does not support sparse gradients")

                param.add_(param.grad, alpha=-lr)
                if add_noise:
                    param.add_(torch.randn_like(param), alpha=noise_std)

        return loss
