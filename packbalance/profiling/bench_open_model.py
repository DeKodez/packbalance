"""Optional check: run the fused LM-head CE on a real open-source model.

This loads the public SmolLM-135M checkpoint, extracts the hidden states and
tied LM-head weight, and compares the memory/time of full logits against the
chunked fused objective.  It is deliberately not part of the training path: the
project trains a compact Llama-style model from scratch, and this script shows
that the kernel works on a real open checkpoint too.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from packbalance.kernels.fused_ce import HAS_TRITON, linear_cross_entropy, use_triton
from packbalance.kernels.reference import chunked_linear_cross_entropy


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device: torch.device, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / iters * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="artifacts/open_model_bench.json")
    args = parser.parse_args()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("pip install -e '.[data]' to use this benchmark") from exc
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()
    max_len = getattr(model.config, "max_position_embeddings", args.tokens)
    tokens = min(args.tokens, max_len)
    vocab = int(model.config.vocab_size)
    input_ids = torch.randint(0, vocab, (1, tokens), device=device)
    with torch.no_grad():
        hidden = model(input_ids, output_hidden_states=True).hidden_states[-1]
    hidden = hidden.detach().reshape(-1, hidden.shape[-1])
    weight = model.get_output_embeddings().weight.detach()
    targets = torch.randint(0, vocab, (tokens,), device=device)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def full() -> None:
        x = hidden.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        logits = x @ w.t()
        F.cross_entropy(logits, targets, reduction="sum").backward()

    def chunked() -> None:
        x = hidden.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        chunked_linear_cross_entropy(x, w, targets, chunk_size=args.chunk).backward()

    def fused() -> None:
        x = hidden.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        linear_cross_entropy(x, w, targets, chunk_size=args.chunk).backward()

    variants = [("full_logits", full), ("chunked_reference", chunked)]
    if use_triton(hidden) and HAS_TRITON:
        variants.append(("fused_triton", fused))
    results = []
    for name, fn in variants:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ms = _time(fn, device)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        results.append(
            {
                "model": args.model,
                "variant": name,
                "tokens": tokens,
                "vocab": vocab,
                "dim": int(hidden.shape[-1]),
                "ms": ms,
                "peak_gb": peak,
            }
        )
        print(f"{name:<20} {ms:8.3f} ms  peak {peak:6.2f} GB", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
