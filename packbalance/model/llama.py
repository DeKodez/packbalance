"""A compact Llama-style decoder built for packed-sequence training.

The architecture is intentionally standard (pre-norm, RMSNorm, SwiGLU, GQA,
RoPE, tied embeddings) so the systems work is easy to audit.  The two
non-standard details are:

* packed documents are separated by a FlexAttention BlockMask, and
* the LM head + cross-entropy can run as a fused, chunked kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from packbalance.config import ModelConfig
from packbalance.kernels.fused_ce import linear_cross_entropy
from packbalance.kernels.reference import chunked_linear_cross_entropy, rmsnorm_reference
from packbalance.kernels.rmsnorm import rmsnorm
from packbalance.kernels.swiglu import swiglu
from packbalance.model.attention import attention_forward, build_doc_block_mask
from packbalance.model.rope import RotaryEmbedding, apply_rotary_pos_emb


@dataclass
class CausalLMOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, use_fused: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.use_fused = use_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fused:
            return rmsnorm(x, self.weight, self.eps)
        return rmsnorm_reference(x, self.weight, self.eps)


class LlamaMLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.use_fused = cfg.use_fused_swiglu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = swiglu(gate, up) if self.use_fused else F.silu(gate) * up
        return self.down_proj(hidden)


class LlamaAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.attention_backend = cfg.attention_backend
        self.compile_attention = cfg.compile_attention

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        doc_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden.shape
        q = self.q_proj(hidden).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(hidden).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = attention_forward(
            q,
            k,
            v,
            doc_ids,
            block_mask=block_mask,
            backend=self.attention_backend,
            compile_attention=self.compile_attention,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.use_fused_rmsnorm)
        self.self_attn = LlamaAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.use_fused_rmsnorm)
        self.mlp = LlamaMLP(cfg)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        doc_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden), cos, sin, doc_ids, block_mask
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([LlamaDecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.use_fused_rmsnorm)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        doc_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary(position_ids)
        if (
            block_mask is None
            and input_ids.is_cuda
            and self.cfg.attention_backend in ("auto", "flex")
        ):
            block_mask = build_doc_block_mask(doc_ids, self.cfg.n_heads)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden,
                    cos,
                    sin,
                    doc_ids,
                    block_mask,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, cos, sin, doc_ids, block_mask)
        return self.norm(hidden)


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = LlamaModel(cfg)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            # No duplicate parameter is created; the property below resolves it.
            pass
        self._scale_residual_projections()

    @property
    def lm_head_weight(self) -> torch.Tensor:
        if self.cfg.tie_embeddings:
            return self.model.embed_tokens.weight
        assert self.lm_head is not None
        return self.lm_head.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        for layer in self.model.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=self.cfg.init_std * scale)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=self.cfg.init_std * scale)

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        block_mask=None,
    ) -> CausalLMOutput:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        if doc_ids is None:
            doc_ids = torch.zeros_like(input_ids)
        hidden = self.model(input_ids, position_ids, doc_ids, block_mask)
        if labels is not None:
            weight = self.lm_head_weight
            if self.cfg.use_fused_ce or not hidden.is_cuda:
                loss = linear_cross_entropy(
                    hidden,
                    weight,
                    labels,
                    chunk_size=self.cfg.ce_chunk_size,
                    ignore_index=-100,
                    z_loss_coeff=self.cfg.z_loss_coeff,
                )
            else:
                loss = chunked_linear_cross_entropy(
                    hidden,
                    weight,
                    labels,
                    chunk_size=self.cfg.ce_chunk_size,
                    ignore_index=-100,
                    z_loss_coeff=self.cfg.z_loss_coeff,
                )
            return CausalLMOutput(loss=loss, hidden_states=hidden)
        logits = hidden @ self.lm_head_weight.t()
        return CausalLMOutput(logits=logits, hidden_states=hidden)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        # Generation recomputes the prefix and uses tiny prompt lengths.  Eager
        # attention avoids compiling a fresh FlexAttention kernel per length.
        attention_modules = [layer.self_attn for layer in self.model.layers]
        old_backends = [module.attention_backend for module in attention_modules]
        for module in attention_modules:
            module.attention_backend = "eager"
        try:
            ids = input_ids
            for _ in range(max_new_tokens):
                if ids.shape[1] >= self.cfg.max_seq_len:
                    break
                position_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
                doc_ids = torch.zeros_like(ids)
                out = self.forward(ids, position_ids=position_ids, doc_ids=doc_ids)
                assert out.logits is not None
                logits = out.logits[:, -1, :].float()
                if temperature <= 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        k = min(top_k, logits.shape[-1])
                        values, _ = torch.topk(logits, k)
                        cutoff = values[:, -1:]
                        logits = logits.masked_fill(logits < cutoff, -float("inf"))
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id], dim=1)
            return ids
        finally:
            for module, backend in zip(attention_modules, old_backends, strict=True):
                module.attention_backend = backend


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
