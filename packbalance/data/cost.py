"""Analytical FLOP model used for packing and scheduling.

Two facts drive PackBalance:

1. A token pays a roughly constant price in the linear layers.
2. Attention pays ``O(L^2)`` per *document*, so a pack containing one 4K-token
   document is much more expensive than a pack containing eight 512-token
   documents, even though both contain 4K tokens.

The model deliberately ignores softmax/norm/activation details.  It is a
scheduler, not a roofline replacement: relative cost is what matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from packbalance.config import ModelConfig


@dataclass(frozen=True)
class FlopModel:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> FlopModel:
        return cls(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            d_ff=cfg.d_ff,
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def attn_params_per_layer(self) -> int:
        qkv = self.d_model * (self.d_model + 2 * self.d_kv)
        out = self.d_model * self.d_model
        return qkv + out

    @property
    def mlp_params_per_layer(self) -> int:
        # SwiGLU: gate, up, down.
        return 3 * self.d_model * self.d_ff

    @property
    def linear_params(self) -> int:
        """Parameters used in matmuls per token, including the LM head."""
        per_layer = self.attn_params_per_layer + self.mlp_params_per_layer
        return self.n_layers * per_layer + self.vocab_size * self.d_model

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def total_params(self) -> int:
        norms = 2 * self.d_model * self.n_layers + self.d_model
        return (
            self.embedding_params
            + self.n_layers * (self.attn_params_per_layer + self.mlp_params_per_layer)
            + norms
        )

    @property
    def linear_flops_per_token(self) -> float:
        return 2.0 * self.linear_params

    @property
    def elementwise_flops_per_token(self) -> float:
        # Norms, residual adds, SiLU, softmax-ish work.  Small, but keeping it
        # avoids treating a 32-token document as free.
        return 8.0 * self.n_layers * self.d_model

    @property
    def attention_flops_coeff(self) -> float:
        # Causal QK^T + AV: 2 * L^2 * d_model per layer.
        return 2.0 * self.n_layers * self.d_model

    def doc_cost(self, length: int | np.ndarray) -> float | np.ndarray:
        """Forward-pass-equivalent FLOPs for one document of ``length``."""
        length = np.asarray(length, dtype=np.float64)
        return (
            self.linear_flops_per_token + self.elementwise_flops_per_token
        ) * length + self.attention_flops_coeff * length * length

    def pack_cost(self, lengths: list[int] | np.ndarray) -> float:
        lengths = np.asarray(lengths, dtype=np.float64)
        if lengths.size == 0:
            return 0.0
        return float(
            (self.linear_flops_per_token + self.elementwise_flops_per_token) * lengths.sum()
            + self.attention_flops_coeff * np.square(lengths).sum()
        )

    def padded_batch_cost(self, lengths: list[int] | np.ndarray, pad_len: int) -> float:
        """Cost of ``len(lengths)`` rows padded to ``pad_len``.

        Linear layers process every padded token.  Attention is charged as a
        dense causal ``pad_len x pad_len`` computation per row, which matches
        the common padded baseline before a mask-aware kernel can skip blocks.
        """
        lengths = np.asarray(lengths, dtype=np.float64)
        rows = float(len(lengths))
        linear = (self.linear_flops_per_token + self.elementwise_flops_per_token) * rows * pad_len
        attention = self.attention_flops_coeff * rows * pad_len * pad_len
        return float(linear + attention)

    def train_flops(self, lengths: list[int] | np.ndarray) -> float:
        """Approximate fwd+bwd FLOPs for a pack (3x forward)."""
        return 3.0 * self.pack_cost(lengths)

    def useful_tokens_per_step(self, lengths: list[int] | np.ndarray) -> int:
        return int(np.asarray(lengths, dtype=np.int64).sum())


def cost_stats(costs: list[float] | np.ndarray) -> dict[str, float]:
    costs = np.asarray(costs, dtype=np.float64)
    if costs.size == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0}
    mean = float(costs.mean())
    std = float(costs.std())
    return {
        "mean": mean,
        "std": std,
        "cv": std / mean if mean else 0.0,
        "p50": float(np.percentile(costs, 50)),
        "p99": float(np.percentile(costs, 99)),
        "max": float(costs.max()),
    }
