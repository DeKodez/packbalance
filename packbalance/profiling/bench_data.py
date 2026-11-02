"""Data-pipeline throughput benchmark.

Measures the loader in isolation: useful tokens/s, compute tokens/s, packing
waste, and the cost-balance of the schedule.  No GPU is required.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from packbalance.config import load_config
from packbalance.data.cost import FlopModel
from packbalance.data.dataset import build_dataloader
from packbalance.data.packer import build_packs, pack_stats
from packbalance.data.prepare import prepare_synthetic
from packbalance.data.sampler import build_schedule
from packbalance.data.store import TokenStore


def _bench_loader(store, packs, cfg, indices, num_batches: int, warmup: int) -> dict:
    loader = build_dataloader(store, packs, cfg, indices)
    iterator = iter(loader)
    for _ in range(warmup):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)
    useful = 0
    compute = 0
    start = time.perf_counter()
    for _ in range(num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        useful += int(batch["useful_tokens"])
        compute += int(batch["compute_tokens"])
    elapsed = time.perf_counter() - start
    return {
        "useful_tokens": useful,
        "compute_tokens": compute,
        "useful_tokens_per_s": useful / elapsed,
        "compute_tokens_per_s": compute / elapsed,
        "packs_per_s": num_batches / elapsed,
        "mean_batch_ms": elapsed / num_batches * 1e3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--modes", default="pad,pack_random,pack_ffd,isoflop")
    parser.add_argument("--workers", default="0,2,4,8")
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", default="artifacts/data_bench.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg.data.data_dir)
    if args.prepare or not (data_dir / "meta.json").exists():
        prepare_synthetic(cfg.data, data_dir)
    store = TokenStore.open(data_dir)
    flop_model = FlopModel.from_config(cfg.model)
    workers = [int(w) for w in args.workers.split(",") if w.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = []
    print(
        f"{'mode':<12} {'workers':>7} {'useful tok/s':>14} {'compute tok/s':>14} "
        f"{'packs/s':>9} {'batch ms':>9} {'waste':>7} {'cost cv':>8}"
    )
    for mode in modes:
        data_cfg = replace(cfg.data, mode=mode)
        packs = build_packs(store, data_cfg, flop_model, split="train")
        stats = pack_stats(packs, flop_model)
        schedule = build_schedule(
            packs, world_size=1, grad_accum=1, seed=cfg.data.seed, mode=cfg.data.schedule
        )
        indices = schedule.rank_indices(0)
        for worker_count in workers:
            run_cfg = replace(
                data_cfg,
                num_workers=worker_count,
                pin_memory=False,
                persistent_workers=worker_count > 0,
            )
            metrics = _bench_loader(store, packs, run_cfg, indices, args.num_batches, args.warmup)
            row = {
                "mode": mode,
                "workers": worker_count,
                **metrics,
                "padding_waste": stats["padding_waste"],
                "pack_cost_cv": stats["cv"],
                "step_cost_cv": schedule.stats()["step_cost_cv"],
            }
            results.append(row)
            print(
                f"{mode:<12} {worker_count:>7} {metrics['useful_tokens_per_s']:>14,.0f} "
                f"{metrics['compute_tokens_per_s']:>14,.0f} {metrics['packs_per_s']:>9.1f} "
                f"{metrics['mean_batch_ms']:>9.2f} {stats['padding_waste']:>6.1%} "
                f"{schedule.stats()['step_cost_cv']:>8.3f}"
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
