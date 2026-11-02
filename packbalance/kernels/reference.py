"""PyTorch reference implementations used by tests and CPU fallbacks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_hat = xf * rstd
    return (x_hat * weight.float()).to(x.dtype)


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def chunked_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    z_loss_coeff: float = 0.0,
) -> torch.Tensor:
    """Memory-bounded linear + CE reference.

    The full ``[tokens, vocab]`` logits tensor is never materialized.  Each
    chunk is a normal autograd graph, so this is also the CPU test oracle.
    """
    x2d = x.reshape(-1, x.shape[-1])
    t1d = targets.reshape(-1)
    if x2d.shape[0] != t1d.shape[0]:
        raise ValueError(f"x has {x2d.shape[0]} rows but targets has {t1d.shape[0]}")
    total = x2d.new_zeros(())
    for start in range(0, x2d.shape[0], chunk_size):
        end = min(start + chunk_size, x2d.shape[0])
        logits = x2d[start:end] @ weight.t()
        y = t1d[start:end]
        total = total + F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="sum")
        if z_loss_coeff:
            lse = torch.logsumexp(logits.float(), dim=-1)
            valid = y != ignore_index
            total = total + z_loss_coeff * (lse[valid] ** 2).sum()
    return total
