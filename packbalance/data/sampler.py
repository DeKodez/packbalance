"""Global, deterministic step scheduling.

A pack list is not a training schedule.  Even with perfect packing, a random
round-robin assignment can put three long-document packs on one rank and three
short-document packs on another.  This module balances two levels:

* packs per **step** (so step latency is stable), and
* packs per **rank within a step** (so DDP ranks arrive at the all-reduce
  together).

Every rank builds the same schedule from the same pack list, so no extra
coordination traffic is needed.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from packbalance.data.packer import Pack


@dataclass
class StepPlan:
    step: int
    rank_packs: list[list[int]]
    step_tokens: int
    step_compute_tokens: int
    step_cost: float
    rank_costs: list[float]


@dataclass
class Schedule:
    steps: list[StepPlan]
    world_size: int
    grad_accum: int

    def rank_indices(self, rank: int) -> list[int]:
        out: list[int] = []
        for plan in self.steps:
            out.extend(plan.rank_packs[rank])
        return out

    @property
    def steps_per_epoch(self) -> int:
        return len(self.steps)

    @property
    def tokens_per_epoch(self) -> int:
        return sum(p.step_tokens for p in self.steps)

    def stats(self) -> dict[str, float]:
        step_costs = np.asarray([p.step_cost for p in self.steps], dtype=np.float64)
        rank_costs = np.asarray([[c for c in p.rank_costs] for p in self.steps], dtype=np.float64)
        if step_costs.size == 0:
            return {}
        mean = float(step_costs.mean())
        rank_mean = float(rank_costs.mean())
        rank_max = float(rank_costs.max())
        return {
            "steps": len(self.steps),
            "step_cost_mean": mean,
            "step_cost_cv": float(step_costs.std() / mean) if mean else 0.0,
            "step_cost_p99_over_mean": (
                float(np.percentile(step_costs, 99) / mean) if mean else 0.0
            ),
            "rank_cost_imbalance": (rank_max / rank_mean - 1.0) if rank_mean else 0.0,
            "tokens": int(sum(p.step_tokens for p in self.steps)),
        }


def _assign_ranks(
    pack_ids: list[int],
    packs: list[Pack],
    world_size: int,
    grad_accum: int,
    global_costs: list[float],
) -> tuple[list[list[int]], list[float]]:
    """Greedy list scheduling with a per-rank cardinality constraint.

    ``global_costs`` carries the epoch-to-date load of each rank.  Using it in
    the tie-break rotates expensive packs across ranks instead of always giving
    them to rank 0.
    """
    rank_packs: list[list[int]] = [[] for _ in range(world_size)]
    rank_costs = [0.0] * world_size
    ordered = sorted(pack_ids, key=lambda i: packs[i].cost, reverse=True)
    for pack_id in ordered:
        best_rank = -1
        best_key = None
        for r in range(world_size):
            if len(rank_packs[r]) >= grad_accum:
                continue
            key = (global_costs[r] + rank_costs[r], rank_costs[r], r)
            if best_key is None or key < best_key:
                best_key = key
                best_rank = r
        if best_rank == -1:
            raise RuntimeError("rank assignment failed; step does not have W*A slots")
        rank_packs[best_rank].append(pack_id)
        rank_costs[best_rank] += packs[pack_id].cost
    return rank_packs, rank_costs


def build_schedule(
    packs: list[Pack],
    world_size: int,
    grad_accum: int,
    *,
    seed: int = 0,
    mode: str = "balanced",
) -> Schedule:
    if world_size < 1 or grad_accum < 1:
        raise ValueError("world_size and grad_accum must be >= 1")
    group = world_size * grad_accum
    num_steps = len(packs) // group
    if num_steps == 0:
        raise ValueError(
            f"not enough packs ({len(packs)}) for W*A={group}; reduce grad_accum or add data"
        )
    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(packs))[: num_steps * group]
    selected = [int(i) for i in selected]

    if mode == "random":
        # Naive round-robin: preserve a random data order, ignore cost.
        rng.shuffle(selected)
        step_groups = [selected[i * group : (i + 1) * group] for i in range(num_steps)]
        step_plans = []
        for step, group_ids in enumerate(step_groups):
            rank_packs: list[list[int]] = [[] for _ in range(world_size)]
            for slot, pack_id in enumerate(group_ids):
                rank_packs[slot % world_size].append(pack_id)
            rank_costs = [sum(packs[i].cost for i in ids) for ids in rank_packs]
            step_plans.append(
                StepPlan(
                    step=step,
                    rank_packs=rank_packs,
                    step_tokens=sum(packs[i].tokens for i in group_ids),
                    step_compute_tokens=sum(packs[i].compute_tokens for i in group_ids),
                    step_cost=sum(packs[i].cost for i in group_ids),
                    rank_costs=rank_costs,
                )
            )
        return Schedule(step_plans, world_size, grad_accum)

    if mode != "balanced":
        raise ValueError(f"unknown schedule mode {mode}")

    # LPT at the step level: largest packs go to the currently cheapest step.
    # Every step has exactly `group` pack slots, so the heap only contains
    # steps that still have room.
    ordered = sorted(selected, key=lambda i: (-packs[i].cost, rng.random()))
    step_ids: list[list[int]] = [[] for _ in range(num_steps)]
    step_costs = [0.0] * num_steps
    heap: list[tuple[float, int]] = [(0.0, i) for i in range(num_steps)]
    heapq.heapify(heap)
    for pack_id in ordered:
        while heap:
            cost, step = heapq.heappop(heap)
            if len(step_ids[step]) < group:
                break
        else:
            raise RuntimeError("step assignment failed")
        step_ids[step].append(pack_id)
        step_costs[step] += packs[pack_id].cost
        if len(step_ids[step]) < group:
            heapq.heappush(heap, (step_costs[step], step))

    plans: list[StepPlan] = []
    global_costs = [0.0] * world_size
    for step, ids in enumerate(step_ids):
        rank_packs, rank_costs = _assign_ranks(ids, packs, world_size, grad_accum, global_costs)
        for r in range(world_size):
            global_costs[r] += rank_costs[r]
        plans.append(
            StepPlan(
                step=step,
                rank_packs=rank_packs,
                step_tokens=sum(packs[i].tokens for i in ids),
                step_compute_tokens=sum(packs[i].compute_tokens for i in ids),
                step_cost=step_costs[step],
                rank_costs=rank_costs,
            )
        )

    # Step costs are balanced; shuffle the step order to mix length regimes.
    perm = rng.permutation(len(plans))
    plans = [plans[int(i)] for i in perm]
    for i, plan in enumerate(plans):
        plan.step = i
    return Schedule(plans, world_size, grad_accum)


def schedule_cost_curve(schedule: Schedule) -> list[float]:
    return [p.step_cost for p in schedule.steps]
