"""End-to-end step-time benchmark across packing and scheduling modes.

Each mode uses the same model, token budget and optimizer, but a different way
of turning documents into sequences.  The reported useful tokens/s is the
metric that matters: padded tokens are compute the model pays for but the loss
does not use.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from packbalance.config import load_config
from packbalance.data.cost import FlopModel
from packbalance.data.dataset import build_dataloader, to_device
from packbalance.data.packer import build_packs, pack_stats
from packbalance.data.prepare import prepare_synthetic
from packbalance.data.sampler import build_schedule
from packbalance.data.store import TokenStore
from packbalance.model.attention import build_doc_block_mask
from packbalance.model.llama import LlamaForCausalLM
from packbalance.utils import get_autocast_dtype, seed_everything


def _forward(model, batch, device, n_heads):
    input_ids = batch["input_ids"].unsqueeze(0)
    labels = batch["labels"].unsqueeze(0)
    position_ids = batch["position_ids"].unsqueeze(0)
    doc_ids = batch["doc_ids"].unsqueeze(0)
    block_mask = build_doc_block_mask(doc_ids, n_heads) if device.type == "cuda" else None
    return model(
        input_ids,
        labels=labels,
        position_ids=position_ids,
        doc_ids=doc_ids,
        block_mask=block_mask,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_mode(
    cfg, mode: str, store: TokenStore, flop_model: FlopModel, device: torch.device, args
):
    data_cfg = replace(cfg.data, mode=mode)
    packs = build_packs(store, data_cfg, flop_model, split="train")
    stats = pack_stats(packs, flop_model)
    schedule = build_schedule(
        packs,
        world_size=1,
        grad_accum=cfg.train.grad_accum,
        seed=cfg.data.seed,
        mode=cfg.data.schedule,
    )
    steps = min(args.steps, len(schedule.steps))
    if steps <= 0:
        raise ValueError(f"no schedule steps for mode {mode}")
    warmup = min(args.warmup, max(0, len(schedule.steps) - steps))
    total_needed = (warmup + steps) * cfg.train.grad_accum
    indices = schedule.rank_indices(0)[:total_needed]
    loader = build_dataloader(store, packs, data_cfg, indices)
    iterator = iter(loader)

    seed_everything(cfg.train.seed)
    model = LlamaForCausalLM(cfg.model).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay,
        fused=device.type == "cuda",
    )
    amp_dtype = get_autocast_dtype(cfg.train.dtype) if device.type == "cuda" else None
    autocast_ctx = (
        autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else __import__("contextlib").nullcontext()
    )
    step_times: list[float] = []
    data_times: list[float] = []
    useful = 0
    compute = 0
    flops = 0.0
    useful_per_step = 0.0
    compute_per_step = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(warmup + steps):
        plan = schedule.steps[step]
        _sync(device)
        t0 = time.perf_counter()
        d0 = time.perf_counter()
        local_packs = [next(iterator) for _ in range(cfg.train.grad_accum)]
        data_ms = (time.perf_counter() - d0) * 1e3
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            for batch in local_packs:
                batch = to_device(batch, device)
                out = _forward(model, batch, device, cfg.model.n_heads)
                assert out.loss is not None
                (out.loss / plan.step_tokens).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        _sync(device)
        elapsed = (time.perf_counter() - t0) * 1e3
        if step >= warmup:
            step_times.append(elapsed)
            data_times.append(data_ms)
            useful += plan.step_tokens
            compute += plan.step_compute_tokens
            useful_per_step += plan.step_tokens
            compute_per_step += plan.step_compute_tokens
            flops += 3.0 * plan.step_cost
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    total_s = sum(step_times) / 1e3
    useful_per_s = useful / total_s
    compute_per_s = compute / total_s
    mfu = (
        flops / total_s / (cfg.train.peak_tflops * 1e12)
        if cfg.train.peak_tflops > 0
        else float("nan")
    )
    result = {
        "mode": mode,
        "num_packs": len(packs),
        "steps": steps,
        "padding_waste": stats["padding_waste"],
        "pack_cost_cv": stats["cv"],
        "step_cost_cv": schedule.stats()["step_cost_cv"],
        "rank_cost_imbalance": schedule.stats()["rank_cost_imbalance"],
        "useful_tokens_per_s": useful_per_s,
        "compute_tokens_per_s": compute_per_s,
        "useful_tokens_per_step": useful_per_step / max(1, steps),
        "compute_tokens_per_step": compute_per_step / max(1, steps),
        "step_ms_p50": float(np.percentile(step_times, 50)),
        "step_ms_p99": float(np.percentile(step_times, 99)),
        "data_ms_mean": float(np.mean(data_times)),
        "peak_gpu_gb": peak_gb,
        "mfu": mfu,
        "usd_per_b_tokens": (
            (1e9 / useful_per_s) * (cfg.train.gpu_hourly_cost / 3600.0)
            if cfg.train.gpu_hourly_cost > 0
            else 0.0
        ),
    }
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--modes", default="pad,pack_random,pack_ffd,isoflop")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", default="artifacts/train_bench.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.train.max_steps = args.steps
    args.steps = cfg.train.max_steps
    data_dir = Path(cfg.data.data_dir)
    if args.prepare or not (data_dir / "meta.json").exists():
        prepare_synthetic(cfg.data, data_dir)
    store = TokenStore.open(data_dir)
    flop_model = FlopModel.from_config(cfg.model)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = []
    for mode in modes:
        print(f"\n=== {mode} ===", flush=True)
        result = bench_mode(cfg, mode, store, flop_model, device, args)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        f"\n{'mode':<12} {'waste':>7} {'step cv':>8} {'useful tok/s':>14} "
        f"{'p50 ms':>8} {'p99 ms':>8} {'peak GB':>8} {'MFU':>7} {'$/B tok':>8}"
    )
    for row in results:
        print(
            f"{row['mode']:<12} {row['padding_waste']:>6.1%} {row['step_cost_cv']:>8.3f} "
            f"{row['useful_tokens_per_s']:>14,.0f} {row['step_ms_p50']:>8.1f} "
            f"{row['step_ms_p99']:>8.1f} {row['peak_gpu_gb']:>8.2f} "
            f"{row['mfu']:>7.3f} {row['usd_per_b_tokens']:>8.2f}"
        )
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
