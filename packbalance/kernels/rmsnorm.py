"""Fused RMSNorm forward/backward.

RMSNorm is memory-bound: the eager expression reads ``x`` several times and
materializes intermediate tensors.  The Triton kernel reads each row once,
keeps it in registers/SRAM for the reduction, and writes the normalized output
plus the cached reciprocal RMS.  Backward reuses the cached value.
"""

from __future__ import annotations

import torch

from packbalance.kernels.reference import rmsnorm_reference

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
    def _rmsnorm_fwd(
        X,
        W,
        Y,
        RSTD,
        stride,
        N,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(mean_sq + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y + row * stride + cols, y, mask=mask)
        tl.store(RSTD + row, rstd)

    @triton.jit
    def _rmsnorm_bwd(
        DY,
        X,
        W,
        RSTD,
        DX,
        stride,
        N,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row).to(tl.float32)
        x_hat = x * rstd
        g = dy * w
        c = tl.sum(g * x_hat, axis=0) / N
        dx = rstd * (g - x_hat * c)
        tl.store(DX + row * stride + cols, dx, mask=mask)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])
        n = x2d.shape[1]
        block = triton.next_power_of_2(n)
        if block > 65536:
            raise RuntimeError("RMSNorm feature dimension is too large for this kernel")
        y = torch.empty_like(x2d)
        rstd = torch.empty(x2d.shape[0], device=x.device, dtype=torch.float32)
        _rmsnorm_fwd[(x2d.shape[0],)](
            x2d,
            weight,
            y,
            rstd,
            x2d.stride(0),
            n,
            eps,
            BLOCK=block,
            num_warps=min(max(block // 256, 1), 8),
        )
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.block = block
        ctx.n = n
        return y.reshape_as(x)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x2d, weight, rstd = ctx.saved_tensors
        dy2d = dy.reshape(-1, dy.shape[-1])
        dx = torch.empty_like(dy2d)
        _rmsnorm_bwd[(x2d.shape[0],)](
            dy2d,
            x2d,
            weight,
            rstd,
            dx,
            x2d.stride(0),
            ctx.n,
            BLOCK=ctx.block,
            num_warps=min(max(ctx.block // 256, 1), 8),
        )
        x_hat = x2d.float() * rstd.unsqueeze(-1)
        dw = (dy2d.float() * x_hat).sum(dim=0).to(weight.dtype)
        return dx.reshape_as(dy), dw, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if HAS_TRITON and x.is_cuda:
        return _RMSNorm.apply(x, weight, eps)
    return rmsnorm_reference(x, weight, eps)
