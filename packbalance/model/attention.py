"""Document-masked attention over packed sequences.

FlexAttention turns the document-mask closure into a block-sparse mask, so a
packed sequence pays only for the document-local causal blocks.  The eager path
is the correctness oracle and the CPU fallback.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    HAS_FLEX = True
except ImportError:  # pragma: no cover - old PyTorch
    create_block_mask = None
    flex_attention = None
    HAS_FLEX = False

_COMPILED_FLEX = None


def build_doc_block_mask(
    doc_ids: torch.Tensor,
    n_heads: int,
    block_size: int = 128,
):
    """Build a BlockMask allowing causal attention inside each document."""
    if not HAS_FLEX:
        raise RuntimeError("FlexAttention is not available in this PyTorch build")
    if doc_ids.dim() == 2:
        if doc_ids.shape[0] != 1:
            raise ValueError("build_doc_block_mask expects a single packed sequence")
        doc_ids = doc_ids[0]
    seq_len = doc_ids.numel()
    block_size = min(block_size, max(16, seq_len))

    def mask_mod(b, h, q_idx, kv_idx):
        return (doc_ids[q_idx] == doc_ids[kv_idx]) & (q_idx >= kv_idx)

    return create_block_mask(
        mask_mod,
        B=1,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=doc_ids.device,
        BLOCK_SIZE=block_size,
    )


def _flex_fn(compile_attention: bool):
    global _COMPILED_FLEX
    if not compile_attention or not torch.cuda.is_available():
        return flex_attention
    if _COMPILED_FLEX is None:
        _COMPILED_FLEX = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    doc_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-document SDPA; exact but intentionally simple."""
    batch, heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]
    if kv_heads != heads:
        repeat = heads // kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    out = torch.zeros_like(q)
    for b in range(batch):
        ids = doc_ids[b]
        for doc in torch.unique(ids):
            doc = int(doc)
            if doc < 0:
                continue
            idx = (ids == doc).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            q_doc = q[b : b + 1, :, idx, :]
            k_doc = k[b : b + 1, :, idx, :]
            v_doc = v[b : b + 1, :, idx, :]
            out[b : b + 1, :, idx, :] = F.scaled_dot_product_attention(
                q_doc, k_doc, v_doc, is_causal=True
            )
    return out


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    doc_ids: torch.Tensor,
    *,
    block_mask=None,
    backend: str = "auto",
    compile_attention: bool = True,
) -> torch.Tensor:
    if backend == "auto":
        backend = "flex" if q.is_cuda and HAS_FLEX else "eager"
    if backend == "flex":
        if not q.is_cuda or not HAS_FLEX:
            return eager_attention(q, k, v, doc_ids)
        if block_mask is None:
            block_mask = build_doc_block_mask(doc_ids, q.shape[1])
        fn = _flex_fn(compile_attention)
        return fn(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=(q.shape[1] != k.shape[1]),
        )
    if backend == "eager":
        return eager_attention(q, k, v, doc_ids)
    raise ValueError(f"unknown attention backend {backend}")
