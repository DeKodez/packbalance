import pytest
import torch
import torch.nn.functional as F

from packbalance.kernels.fused_ce import linear_cross_entropy
from packbalance.kernels.reference import (
    chunked_linear_cross_entropy,
    rmsnorm_reference,
    swiglu_reference,
)
from packbalance.kernels.rmsnorm import rmsnorm
from packbalance.kernels.swiglu import swiglu


def _full_ce(x, w, targets, z_loss_coeff=0.0):
    logits = x @ w.t()
    loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
    if z_loss_coeff:
        lse = torch.logsumexp(logits.float(), dim=-1)
        valid = targets != -100
        loss = loss + z_loss_coeff * (lse[valid] ** 2).sum()
    return loss


def test_rmsnorm_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 32)
    w = torch.randn(32)
    eps = 1e-5
    ref = rmsnorm_reference(x, w, eps)
    x2 = x.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    out = rmsnorm(x2, w2, eps)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    grad = torch.randn_like(out)
    out.backward(grad)
    x_ref = x.clone().requires_grad_(True)
    w_ref = w.clone().requires_grad_(True)
    ref2 = rmsnorm_reference(x_ref, w_ref, eps)
    ref2.backward(grad)
    torch.testing.assert_close(x2.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w2.grad, w_ref.grad, atol=1e-5, rtol=1e-5)


def test_swiglu_matches_reference() -> None:
    torch.manual_seed(0)
    gate = torch.randn(8, 16)
    up = torch.randn(8, 16)
    ref = swiglu_reference(gate, up)
    g2 = gate.clone().requires_grad_(True)
    u2 = up.clone().requires_grad_(True)
    out = swiglu(g2, u2)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
    grad = torch.randn_like(out)
    out.backward(grad)
    g_ref = gate.clone().requires_grad_(True)
    u_ref = up.clone().requires_grad_(True)
    ref2 = swiglu_reference(g_ref, u_ref)
    ref2.backward(grad)
    torch.testing.assert_close(g2.grad, g_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(u2.grad, u_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("z_loss", [0.0, 0.01])
def test_chunked_ce_matches_full(z_loss: float) -> None:
    torch.manual_seed(0)
    x = torch.randn(37, 32)
    w = torch.randn(64, 32)
    targets = torch.randint(0, 64, (37,))
    targets[::7] = -100
    x1 = x.clone().requires_grad_(True)
    w1 = w.clone().requires_grad_(True)
    loss = _full_ce(x1, w1, targets, z_loss)
    loss.backward()
    x2 = x.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    loss2 = chunked_linear_cross_entropy(x2, w2, targets, chunk_size=8, z_loss_coeff=z_loss)
    loss2.backward()
    torch.testing.assert_close(loss2, loss, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(x2.grad, x1.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w2.grad, w1.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_triton_flce_matches_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(0)
    x = torch.randn(2, 32, 32, device="cuda")
    w = torch.randn(128, 32, device="cuda")
    targets = torch.randint(0, 128, (2, 32), device="cuda")
    targets[:, ::5] = -100
    x1 = x.clone().requires_grad_(True)
    w1 = w.clone().requires_grad_(True)
    loss = chunked_linear_cross_entropy(x1, w1, targets, chunk_size=16, z_loss_coeff=0.01)
    loss.backward()
    x2 = x.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    loss2 = linear_cross_entropy(x2, w2, targets, chunk_size=16, z_loss_coeff=0.01)
    loss2.backward()
    torch.testing.assert_close(loss2, loss, atol=2e-2, rtol=2e-2)
    assert x2.grad is not None and x2.grad.shape == x.shape
    torch.testing.assert_close(x2.grad, x1.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w2.grad, w1.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["rmsnorm", "swiglu"])
def test_triton_elementwise_kernels(kernel: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(0)
    if kernel == "rmsnorm":
        x = torch.randn(32, 64, device="cuda")
        w = torch.randn(64, device="cuda")
        grad = torch.randn(32, 64, device="cuda")
        x1 = x.clone().requires_grad_(True)
        w1 = w.clone().requires_grad_(True)
        ref = rmsnorm_reference(x1, w1, 1e-5)
        ref.backward(grad)
        x2 = x.clone().requires_grad_(True)
        w2 = w.clone().requires_grad_(True)
        out = rmsnorm(x2, w2, 1e-5)
        out.backward(grad)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(x2.grad, x1.grad, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(w2.grad, w1.grad, atol=2e-2, rtol=2e-2)
    else:
        gate = torch.randn(16, 32, device="cuda")
        up = torch.randn(16, 32, device="cuda")
        grad = torch.randn(16, 32, device="cuda")
        g1 = gate.clone().requires_grad_(True)
        u1 = up.clone().requires_grad_(True)
        ref = swiglu_reference(g1, u1)
        ref.backward(grad)
        g2 = gate.clone().requires_grad_(True)
        u2 = up.clone().requires_grad_(True)
        out = swiglu(g2, u2)
        out.backward(grad)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(g2.grad, g1.grad, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(u2.grad, u1.grad, atol=2e-2, rtol=2e-2)
