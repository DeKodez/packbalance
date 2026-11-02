"""Microbenchmarks for the fused kernels.

Reports latency, peak memory, and achieved effective bandwidth / TFLOPs.  On
CPU the Triton paths are skipped and only the reference implementations run.
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
from packbalance.kernels.reference import (
    chunked_linear_cross_entropy,
    rmsnorm_reference,
    swiglu_reference,
)
from packbalance.kernels.rmsnorm import rmsnorm
from packbalance.kernels.swiglu import swiglu


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device: torch.device, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / iters * 1e3


def _peak_mb(fn, device: torch.device) -> float:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    fn()
    _sync(device)
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def _rmsnorm_bench(device: torch.device, rows: int, dim: int) -> list[dict]:
    x = torch.randn(rows, dim, device=device)
    w = torch.randn(dim, device=device)
    results = []
    variants = [("reference", lambda a, b: rmsnorm_reference(a, b, 1e-5))]
    if use_triton(x):
        variants.append(("triton", lambda a, b: rmsnorm(a, b, 1e-5)))
    for name, fn in variants:
        ms = _time(lambda fn=fn: fn(x, w), device)
        bytes_moved = rows * dim * 4 * 2 + dim * 4  # one read + one write
        results.append(
            {
                "kernel": "rmsnorm",
                "variant": name,
                "shape": f"{rows}x{dim}",
                "ms": ms,
                "gbps": bytes_moved / (ms / 1e3) / 1e9,
            }
        )
    return results


def _swiglu_bench(device: torch.device, rows: int, dim: int) -> list[dict]:
    gate = torch.randn(rows, dim, device=device)
    up = torch.randn(rows, dim, device=device)
    results = []
    variants = [("reference", lambda a, b: swiglu_reference(a, b))]
    if use_triton(gate):
        variants.append(("triton", lambda a, b: swiglu(a, b)))
    for name, fn in variants:
        ms = _time(lambda fn=fn: fn(gate, up), device)
        bytes_moved = rows * dim * 4 * 3
        results.append(
            {
                "kernel": "swiglu",
                "variant": name,
                "shape": f"{rows}x{dim}",
                "ms": ms,
                "gbps": bytes_moved / (ms / 1e3) / 1e9,
            }
        )
    return results


def _ce_bench(device: torch.device, tokens: int, dim: int, vocab: int, chunk: int) -> list[dict]:
    torch.manual_seed(0)
    x = torch.randn(tokens, dim, device=device)
    w = torch.randn(vocab, dim, device=device)
    targets = torch.randint(0, vocab, (tokens,), device=device)
    results = []

    def full() -> None:
        x1 = x.detach().clone().requires_grad_(True)
        w1 = w.detach().clone().requires_grad_(True)
        logits = x1 @ w1.t()
        loss = F.cross_entropy(logits, targets, reduction="sum")
        loss.backward()

    def chunked() -> None:
        x1 = x.detach().clone().requires_grad_(True)
        w1 = w.detach().clone().requires_grad_(True)
        loss = chunked_linear_cross_entropy(x1, w1, targets, chunk_size=chunk)
        loss.backward()

    def fused() -> None:
        x1 = x.detach().clone().requires_grad_(True)
        w1 = w.detach().clone().requires_grad_(True)
        loss = linear_cross_entropy(x1, w1, targets, chunk_size=chunk)
        loss.backward()

    variants = [("full_logits", full), ("chunked_reference", chunked)]
    if use_triton(x) and HAS_TRITON:
        variants.append(("fused_triton", fused))
    flops = 6.0 * tokens * dim * vocab
    for name, fn in variants:
        ms = _time(fn, device, warmup=2, iters=5)
        peak = _peak_mb(fn, device)
        results.append(
            {
                "kernel": "linear_cross_entropy",
                "variant": name,
                "shape": f"{tokens}x{dim}x{vocab}",
                "ms": ms,
                "tflops": flops / (ms / 1e3) / 1e12,
                "peak_mb": peak,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=49152)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--output", default="artifacts/kernel_bench.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    results: list[dict] = []
    results += _rmsnorm_bench(device, args.rows, args.dim)
    results += _swiglu_bench(device, args.rows, args.dim)
    results += _ce_bench(device, args.tokens, args.dim, args.vocab, args.chunk)
    print(
        f"{'kernel':<22} {'variant':<18} {'shape':<20} {'ms':>8} "
        f"{'GB/s':>8} {'TFLOP/s':>9} {'peak MB':>9}"
    )
    for row in results:
        print(
            f"{row['kernel']:<22} {row['variant']:<18} {row['shape']:<20} "
            f"{row['ms']:>8.3f} {row.get('gbps', 0):>8.1f} "
            f"{row.get('tflops', 0):>9.2f} {row.get('peak_mb', 0):>9.0f}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
