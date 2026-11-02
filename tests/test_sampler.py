from packbalance.data.packer import Pack
from packbalance.data.sampler import build_schedule


def _make_packs(costs: list[float], tokens: list[int] | None = None) -> list[Pack]:
    tokens = tokens or [10] * len(costs)
    return [
        Pack(
            pack_id=i,
            segments=[],
            tokens=tokens[i],
            compute_tokens=100,
            cost=costs[i],
        )
        for i in range(len(costs))
    ]


def test_balanced_schedule_is_complete_and_capacity_respecting() -> None:
    packs = _make_packs([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    schedule = build_schedule(packs, world_size=2, grad_accum=2, seed=0, mode="balanced")
    seen = []
    for plan in schedule.steps:
        assert len(plan.rank_packs) == 2
        assert all(len(ids) == 2 for ids in plan.rank_packs)
        for ids in plan.rank_packs:
            seen.extend(ids)
    assert sorted(seen) == list(range(len(packs)))
    assert schedule.stats()["rank_cost_imbalance"] < 1e-9


def test_balanced_schedule_beats_random_on_skewed_costs() -> None:
    # Heavy-tail costs: a random round-robin will put several expensive packs
    # on one rank, while the balanced scheduler should not.
    costs = [1.0] * 24 + [50.0] * 8
    packs = _make_packs(costs)
    balanced = build_schedule(packs, world_size=4, grad_accum=1, seed=3, mode="balanced")
    random = build_schedule(packs, world_size=4, grad_accum=1, seed=3, mode="random")
    # A pack is atomic, so one expensive pack still makes one rank the straggler
    # in its step.  The scheduler's job is to make step totals uniform and to
    # rotate expensive packs across ranks over the epoch.
    assert balanced.stats()["step_cost_cv"] < random.stats()["step_cost_cv"]
    assert balanced.stats()["step_cost_p99_over_mean"] < random.stats()["step_cost_p99_over_mean"]
    assert balanced.stats()["rank_cost_imbalance"] <= random.stats()["rank_cost_imbalance"] + 1e-9


def test_schedule_rank_indices_match_plans() -> None:
    packs = _make_packs(list(range(1, 17)))
    schedule = build_schedule(packs, world_size=2, grad_accum=2, seed=1)
    for rank in range(2):
        indices = schedule.rank_indices(rank)
        expected = []
        for plan in schedule.steps:
            expected.extend(plan.rank_packs[rank])
        assert indices == expected
