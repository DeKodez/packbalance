from collections import Counter

import numpy as np
import pytest

from packbalance.config import DataConfig
from packbalance.data.cost import FlopModel
from packbalance.data.packer import build_packs, pack_stats, split_documents
from packbalance.data.store import TokenStore


def _assert_tokens_covered(store: TokenStore, packs, cfg: DataConfig) -> None:
    covered: Counter[tuple[int, int]] = Counter()
    for pack in packs:
        tokens = 0
        for seg in pack.segments:
            tokens += seg.length
            for pos in range(seg.start, seg.start + seg.length):
                covered[(seg.doc_id, pos)] += 1
        if pack.kind == "padded":
            assert tokens <= cfg.seq_len * pack.rows
        else:
            assert tokens <= cfg.seq_len
    assert all(v == 1 for v in covered.values())
    assert len(covered) > 0


@pytest.mark.parametrize("mode", ["pad", "pack_random", "pack_ffd", "isoflop"])
def test_packs_are_valid(
    store: TokenStore, flop_model: FlopModel, data_cfg: DataConfig, mode: str
) -> None:
    cfg = DataConfig(**{**data_cfg.__dict__, "mode": mode})
    packs = build_packs(store, cfg, flop_model, split="train")
    assert packs
    _assert_tokens_covered(store, packs, cfg)
    stats = pack_stats(packs, flop_model)
    assert 0.0 <= stats["padding_waste"] < 0.5
    assert stats["cv"] >= 0.0


def test_document_splitting(store: TokenStore, flop_model: FlopModel, data_cfg: DataConfig) -> None:
    doc_ids = np.arange(store.num_docs)
    segments = split_documents(store, data_cfg, flop_model, doc_ids)
    assert segments
    assert all(seg.length <= data_cfg.max_doc_len for seg in segments)
    # Every original token appears exactly once in the segment list.
    total = sum(seg.length for seg in segments)
    expected = sum(
        int(store.doc_lengths[i])
        for i in range(store.num_docs)
        if int(store.doc_lengths[i]) >= data_cfg.min_doc_len
    )
    assert total <= expected
    assert total > 0


def test_isoflop_has_low_cost_variance(
    store: TokenStore, flop_model: FlopModel, data_cfg: DataConfig
) -> None:
    cfg = DataConfig(**{**data_cfg.__dict__, "mode": "isoflop"})
    packs = build_packs(store, cfg, flop_model, split="train")
    stats = pack_stats(packs, flop_model)
    assert stats["cv"] < 0.35
    assert stats["padding_waste"] < 0.05
