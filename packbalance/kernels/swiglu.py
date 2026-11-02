"""Fused SwiGLU elementwise kernel."""

from __future__ import annotations

import torch

from packbalance.kernels.reference import swiglu_reference

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _swiglu_fwd(GATE, UP, OUT, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        y = (g * sig) * u
        tl.store(OUT + offs, y, mask=mask)

    @triton.jit
    def _swiglu_bwd(GATE, UP, DY, DGATE, DUP, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        silu = g * sig
        dgate = dy * u * (sig * (1.0 + g * (1.0 - sig)))
        dup = dy * silu
        tl.store(DGATE + offs, dgate, mask=mask)
        tl.store(DUP + offs, dup, mask=mask)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(gate)
        n = gate.numel()
        block = 1024
        _swiglu_fwd[(triton.cdiv(n, block),)](gate, up, out, n, BLOCK=block, num_warps=4)
        ctx.save_for_backward(gate, up)
        ctx.numel = n
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        gate, up = ctx.saved_tensors
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        block = 1024
        _swiglu_bwd[(triton.cdiv(ctx.numel, block),)](
            gate, up, dy, dgate, dup, ctx.numel, BLOCK=block, num_warps=4
        )
        return dgate, dup


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if HAS_TRITON and gate.is_cuda:
        return _SwiGLU.apply(gate, up)
    return swiglu_reference(gate, up)
