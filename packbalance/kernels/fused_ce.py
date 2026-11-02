"""Fused linear cross-entropy.

The LM head is the largest activation in a small-vocabulary LLM: ``[tokens,
vocab]`` logits are expensive to materialize, and even more expensive to keep
for the backward pass.  This module never materializes more than one chunk of
logits and recomputes that chunk during backward.  The CE math is a Triton
row-wise online-softmax kernel on CUDA; the reference chunked implementation is
the fallback and the test oracle.
"""

from __future__ import annotations

import torch

from packbalance.kernels.reference import chunked_linear_cross_entropy

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - macOS/CPU
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _ce_kernel(
        LOGITS,
        TARGETS,
        LOSSES,
        DLOGITS,
        M,
        V,
        stride_m,
        stride_v,
        ignore_index,
        z_loss_coeff,
        BLOCK_V: tl.constexpr,
        WRITE_DLOGITS: tl.constexpr,
    ):
        row = tl.program_id(0)
        target = tl.load(TARGETS + row)
        valid = target != ignore_index
        safe_target = tl.where(valid, target, 0)
        cols = tl.arange(0, BLOCK_V)

        # Pass 1: online max and sum(exp).
        max_val = -float("inf")
        sum_exp = 0.0
        for off in range(0, V, BLOCK_V):
            c = off + cols
            x = tl.load(LOGITS + row * stride_m + c * stride_v, mask=c < V, other=-float("inf")).to(
                tl.float32
            )
            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(max_val, block_max)
            sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
            max_val = new_max

        logsumexp = max_val + tl.log(sum_exp)
        target_logit = tl.load(LOGITS + row * stride_m + safe_target * stride_v).to(tl.float32)
        loss = logsumexp - target_logit
        loss += z_loss_coeff * logsumexp * logsumexp
        loss = tl.where(valid, loss, 0.0)
        tl.store(LOSSES + row, loss)

        if WRITE_DLOGITS:
            inv_sum = 1.0 / sum_exp
            for off in range(0, V, BLOCK_V):
                c = off + cols
                x = tl.load(
                    LOGITS + row * stride_m + c * stride_v,
                    mask=c < V,
                    other=-float("inf"),
                ).to(tl.float32)
                p = tl.exp(x - max_val) * inv_sum
                grad = p * (1.0 + 2.0 * z_loss_coeff * logsumexp) - tl.where(
                    c == safe_target, 1.0, 0.0
                )
                grad = tl.where(valid, grad, 0.0)
                tl.store(
                    DLOGITS + row * stride_m + c * stride_v,
                    grad,
                    mask=c < V,
                )

    def _ce_launch(
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        write_dlogits: bool,
        ignore_index: int,
        z_loss_coeff: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        m, v = logits.shape
        losses = torch.empty(m, device=logits.device, dtype=torch.float32)
        dlogits = torch.empty_like(logits) if write_dlogits else None
        block_v = min(4096, triton.next_power_of_2(v))
        _ce_kernel[(m,)](
            logits,
            targets,
            losses,
            dlogits if dlogits is not None else logits,
            m,
            v,
            logits.stride(0),
            logits.stride(1),
            ignore_index,
            z_loss_coeff,
            BLOCK_V=block_v,
            WRITE_DLOGITS=write_dlogits,
            num_warps=8,
        )
        return losses, dlogits


def _reference_linear_ce(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    ignore_index: int,
    z_loss_coeff: float,
) -> torch.Tensor:
    return chunked_linear_cross_entropy(
        x,
        weight,
        targets,
        chunk_size=chunk_size,
        ignore_index=ignore_index,
        z_loss_coeff=z_loss_coeff,
    )


def use_triton(x: torch.Tensor) -> bool:
    return HAS_TRITON and x.is_cuda


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int,
        ignore_index: int,
        z_loss_coeff: float,
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])
        t1d = targets.reshape(-1)
        total = x2d.new_zeros((), dtype=torch.float32)
        for start in range(0, x2d.shape[0], chunk_size):
            end = min(start + chunk_size, x2d.shape[0])
            logits = x2d[start:end] @ weight.t()
            losses, _ = _ce_launch(
                logits,
                t1d[start:end],
                write_dlogits=False,
                ignore_index=ignore_index,
                z_loss_coeff=z_loss_coeff,
            )
            total = total + losses.sum()
        ctx.save_for_backward(x2d, weight, t1d)
        ctx.x_shape = x.shape
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.z_loss_coeff = z_loss_coeff
        return total

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x2d, weight, t1d = ctx.saved_tensors
        dx = torch.empty_like(x2d)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        for start in range(0, x2d.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, x2d.shape[0])
            logits = x2d[start:end] @ weight.t()
            _, dlogits = _ce_launch(
                logits,
                t1d[start:end],
                write_dlogits=True,
                ignore_index=ctx.ignore_index,
                z_loss_coeff=ctx.z_loss_coeff,
            )
            assert dlogits is not None
            dx[start:end] = dlogits @ weight
            dw += dlogits.t().float() @ x2d[start:end].float()
        scale = grad_output.to(torch.float32)
        dx = dx.reshape(ctx.x_shape)
        return (
            (dx * scale).to(x2d.dtype),
            (dw * scale).to(weight.dtype),
            None,
            None,
            None,
            None,
        )


def linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    z_loss_coeff: float = 0.0,
) -> torch.Tensor:
    """Return the summed CE loss for ``x @ weight.T`` without full logits."""
    if use_triton(x):
        return _FusedLinearCE.apply(x, weight, targets, chunk_size, ignore_index, z_loss_coeff)
    return _reference_linear_ce(x, weight, targets, chunk_size, ignore_index, z_loss_coeff)
