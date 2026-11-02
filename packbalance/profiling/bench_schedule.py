"""Compare naive round-robin scheduling with PackBalance' two-level scheduler.

No model, no GPU: this isolates how well a schedule balances work across steps
and ranks for a real pack list.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from packbalance.config import load_config
from packbalance.data.cost import FlopModel
from packbalance.data.packer import build_packs
from packbalance.data.prepare import prepare_synthetic
from packbalance.data.sampler import build_schedule
from packbalance.data.store import TokenStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--modes", default="pad,pack_random,pack_ffd,isoflop")
    parser.add_argument("--schedules", default="random,balanced")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", default="artifacts/schedule_bench.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir = Path(cfg.data.data_dir)
    if args.prepare or not (data_dir / "meta.json").exists():
        prepare_synthetic(cfg.data, data_dir)
    store = TokenStore.open(data_dir)
    flop_model = FlopModel.from_config(cfg.model)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    schedules = [s.strip() for s in args.schedules.split(",") if s.strip()]
    results = []
    print(
        f"{'mode':<12} {'schedule':<9} {'steps':>6} {'step cv':>8} "
        f"{'p99/mean':>9} {'rank imb':>9} {'tokens':>12}"
    )
    for mode in modes:
        data_cfg = replace(cfg.data, mode=mode)
        packs = build_packs(store, data_cfg, flop_model, split="train")
        for schedule_mode in schedules:
            schedule = build_schedule(
                packs,
                world_size=args.world_size,
                grad_accum=cfg.train.grad_accum,
                seed=cfg.data.seed,
                mode=schedule_mode,
            )
            stats = schedule.stats()
            row = {"mode": mode, "schedule": schedule_mode, **stats}
            results.append(row)
            print(
                f"{mode:<12} {schedule_mode:<9} {stats['steps']:>6} "
                f"{stats['step_cost_cv']:>8.4f} {stats['step_cost_p99_over_mean']:>9.4f} "
                f"{stats['rank_cost_imbalance']:>9.4f} {stats['tokens']:>12,}"
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
