"""Document splitting and sequence packing.

The packer is deliberately separated from the scheduler.  The packer decides
*which documents share a sequence*; the scheduler decides *which sequences share
a step and a rank*.  Both matter, and Both are measured independently.
"""

from __future__ import annotations

import bisect
import heapq
from dataclasses import dataclass

import numpy as np

from packbalance.config import DataConfig
from packbalance.data.cost import FlopModel, cost_stats
from packbalance.data.store import TokenStore


@dataclass(frozen=True)
class Segment:
    doc_id: int
    start: int
    length: int
    cost: float


@dataclass
class Pack:
    pack_id: int
    segments: list[Segment]
    tokens: int
    compute_tokens: int
    cost: float
    kind: str = "packed"
    rows: int = 1

    @property
    def useful_tokens(self) -> int:
        return self.tokens

    @property
    def utilization(self) -> float:
        return self.tokens / self.compute_tokens if self.compute_tokens else 0.0


def split_documents(
    store: TokenStore,
    cfg: DataConfig,
    flop_model: FlopModel,
    doc_ids: np.ndarray,
) -> list[Segment]:
    """Flatten documents into segments that fit the packing context.

    Long documents are cut into ``max_doc_len`` windows.  Windows are separate
    attention documents, so the model never sees a fake cross-window context.
    """
    max_len = min(cfg.max_doc_len, cfg.seq_len)
    segments: list[Segment] = []
    for doc_id in doc_ids:
        doc_id = int(doc_id)
        start = int(store.doc_offsets[doc_id])
        length = int(store.doc_lengths[doc_id])
        if length <= max_len:
            if length >= cfg.min_doc_len:
                segments.append(Segment(doc_id, start, length, flop_model.doc_cost(length)))
            continue
        offset = 0
        while offset < length:
            chunk = min(max_len, length - offset)
            if chunk >= cfg.min_doc_len:
                segments.append(
                    Segment(
                        doc_id,
                        start + offset,
                        chunk,
                        flop_model.doc_cost(chunk),
                    )
                )
            offset += chunk
    if not segments:
        raise ValueError("no usable documents after filtering; lower min_doc_len")
    return segments


def _pack_best_fit(
    segments: list[Segment],
    seq_len: int,
    flop_model: FlopModel,
    *,
    order: str = "length",
    seed: int = 0,
) -> list[Pack]:
    """Best-fit decreasing by token length (or cost) with an O(n log n) index."""
    if order == "length":
        ordered = sorted(segments, key=lambda s: (-s.length, s.doc_id))
    elif order == "cost":
        ordered = sorted(segments, key=lambda s: (-s.cost, -s.length, s.doc_id))
    elif order == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(segments))
        ordered = [segments[i] for i in perm]
    else:
        raise ValueError(f"unknown pack order {order}")

    bins: list[list[Segment]] = []
    bin_tokens: list[int] = []
    bin_costs: list[float] = []
    keys: list[int] = []  # sorted remaining capacities
    ids: list[int] = []  # parallel bin ids

    for seg in ordered:
        idx = bisect.bisect_left(keys, seg.length)
        if idx == len(keys):
            bin_id = len(bins)
            bins.append([seg])
            bin_tokens.append(seg.length)
            bin_costs.append(seg.cost)
            remaining = seq_len - seg.length
            key = remaining
        else:
            bin_id = ids.pop(idx)
            remaining = keys.pop(idx) - seg.length
            bins[bin_id].append(seg)
            bin_tokens[bin_id] += seg.length
            bin_costs[bin_id] += seg.cost
            key = remaining
        pos = bisect.bisect_left(keys, key)
        keys.insert(pos, key)
        ids.insert(pos, bin_id)

    packs = []
    for i, (segs, tokens, cost) in enumerate(zip(bins, bin_tokens, bin_costs, strict=False)):
        packs.append(
            Pack(
                pack_id=i,
                segments=segs,
                tokens=tokens,
                compute_tokens=seq_len,
                cost=cost,
            )
        )
    return packs


def _pack_cost_aware(
    segments: list[Segment],
    seq_len: int,
    flop_model: FlopModel,
) -> list[Pack]:
    """Place high-cost documents first, preferring the cheapest feasible bin.

    The cost of a bin is the sum of its documents' costs.  Bins that cannot fit
    a high-cost document are re-inserted; later (usually shorter) documents can
    still use them.
    """
    ordered = sorted(segments, key=lambda s: (-s.cost, -s.length, s.doc_id))
    bins: list[list[Segment]] = []
    bin_tokens: list[int] = []
    bin_costs: list[float] = []
    heap: list[tuple[float, int]] = []

    for seg in ordered:
        if seg.length > seq_len:
            raise ValueError(f"segment length {seg.length} exceeds seq_len {seq_len}")
        deferred: list[tuple[float, int]] = []
        chosen = -1
        while heap:
            cost, bin_id = heapq.heappop(heap)
            if bin_tokens[bin_id] + seg.length <= seq_len:
                chosen = bin_id
                break
            deferred.append((cost, bin_id))
        for item in deferred:
            heapq.heappush(heap, item)
        if chosen == -1:
            chosen = len(bins)
            bins.append([])
            bin_tokens.append(0)
            bin_costs.append(0.0)
        bins[chosen].append(seg)
        bin_tokens[chosen] += seg.length
        bin_costs[chosen] += seg.cost
        heapq.heappush(heap, (bin_costs[chosen], chosen))

    packs = []
    for i, (segs, tokens, cost) in enumerate(zip(bins, bin_tokens, bin_costs, strict=False)):
        packs.append(
            Pack(
                pack_id=i,
                segments=segs,
                tokens=tokens,
                compute_tokens=seq_len,
                cost=cost,
            )
        )
    return packs


def _rebalance_packs(
    packs: list[Pack],
    seq_len: int,
    iterations: int,
) -> list[Pack]:
    """Move small documents from expensive packs to cheap packs.

    The scheduler can hide some pack-level variance, so this is intentionally a
    cheap local search rather than an exact bin-packing solver.
    """
    if len(packs) < 2:
        return packs
    for _ in range(max(0, iterations)):
        order = sorted(range(len(packs)), key=lambda i: packs[i].cost, reverse=True)
        improved = False
        for _ in range(len(order) // 2):
            hi_i = order[0]
            lo_i = order[-1]
            hi, lo = packs[hi_i], packs[lo_i]
            if hi.cost - lo.cost <= 1e-9:
                break
            best = None
            for seg in hi.segments:
                if lo.tokens + seg.length > seq_len:
                    continue
                new_hi = hi.cost - seg.cost
                new_lo = lo.cost + seg.cost
                if max(new_hi, new_lo) < max(hi.cost, lo.cost) or (
                    new_hi - new_lo < hi.cost - lo.cost
                ):
                    score = max(new_hi, new_lo) - min(new_hi, new_lo)
                    if best is None or score < best[0]:
                        best = (score, seg)
            if best is None:
                break
            seg = best[1]
            hi.segments.remove(seg)
            lo.segments.append(seg)
            hi.tokens -= seg.length
            lo.tokens += seg.length
            hi.cost -= seg.cost
            lo.cost += seg.cost
            improved = True
            break
        if not improved:
            break
    return packs


def _renumber(packs: list[Pack]) -> list[Pack]:
    for i, pack in enumerate(packs):
        pack.pack_id = i
    return packs


def build_packs(
    store: TokenStore,
    cfg: DataConfig,
    flop_model: FlopModel,
    *,
    split: str = "train",
) -> list[Pack]:
    """Build the pack list for a split."""
    n = store.num_docs
    if not 0.0 <= cfg.val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    val_n = int(round(n * cfg.val_fraction))
    if split == "val":
        doc_ids = np.arange(max(0, n - val_n), n, dtype=np.int64)
    else:
        doc_ids = np.arange(0, max(0, n - val_n), dtype=np.int64)
    if len(doc_ids) == 0:
        raise ValueError(f"empty {split} split (docs={n}, val_fraction={cfg.val_fraction})")
    if cfg.max_docs is not None:
        doc_ids = doc_ids[: cfg.max_docs]

    segments = split_documents(store, cfg, flop_model, doc_ids)
    rng = np.random.default_rng(cfg.seed + (0 if split == "train" else 1))

    if cfg.mode == "pad":
        order = rng.permutation(len(segments))
        batches: list[Pack] = []
        for batch_id, start in enumerate(range(0, len(segments), cfg.pad_batch_size)):
            segs = [segments[int(i)] for i in order[start : start + cfg.pad_batch_size]]
            if not segs:
                continue
            lengths = [s.length for s in segs]
            cost = flop_model.padded_batch_cost(lengths, cfg.seq_len)
            batches.append(
                Pack(
                    pack_id=batch_id,
                    segments=segs,
                    tokens=int(sum(lengths)),
                    compute_tokens=cfg.seq_len * len(segs),
                    cost=cost,
                    kind="padded",
                    rows=len(segs),
                )
            )
        packs = batches
    elif cfg.mode == "pack_random":
        packs = _pack_best_fit(segments, cfg.seq_len, flop_model, order="random", seed=cfg.seed)
    elif cfg.mode == "pack_ffd":
        packs = _pack_best_fit(segments, cfg.seq_len, flop_model, order="length")
    elif cfg.mode == "isoflop":
        packs = _pack_cost_aware(segments, cfg.seq_len, flop_model)
        packs = _rebalance_packs(packs, cfg.seq_len, cfg.isoflop_iterations)
    else:
        raise ValueError(f"unknown packing mode {cfg.mode}")

    packs = _renumber(packs)
    # Pack construction is deterministic; shuffle pack order so the schedule
    # does not always start with the same length regime.
    if split == "train":
        perm = rng.permutation(len(packs))
        packs = [packs[int(i)] for i in perm]
        packs = _renumber(packs)
    return packs


def pack_stats(packs: list[Pack], flop_model: FlopModel) -> dict[str, float]:
    costs = [p.cost for p in packs]
    useful = sum(p.useful_tokens for p in packs)
    compute = sum(p.compute_tokens for p in packs)
    docs = sum(len(p.segments) for p in packs)
    stats = cost_stats(costs)
    stats.update(
        {
            "num_packs": len(packs),
            "num_docs": docs,
            "useful_tokens": useful,
            "compute_tokens": compute,
            "padding_waste": 1.0 - useful / compute if compute else 0.0,
            "mean_docs_per_pack": docs / len(packs) if packs else 0.0,
            "mean_utilization": float(np.mean([p.utilization for p in packs])) if packs else 0.0,
        }
    )
    return stats
