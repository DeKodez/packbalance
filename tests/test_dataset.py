import numpy as np
import torch

from packbalance.config import DataConfig
from packbalance.data.cost import FlopModel
from packbalance.data.dataset import PackedDataset
from packbalance.data.packer import build_packs
from packbalance.data.store import TokenStore


def _find_pack_with_docs(packs, min_docs: int = 2):
    for pack in packs:
        if pack.kind != "padded" and len(pack.segments) >= min_docs:
            return pack
    raise AssertionError("no multi-document pack found")


def test_packed_example_is_well_formed(
    store: TokenStore, data_cfg: DataConfig, flop_model: FlopModel
) -> None:
    packs = build_packs(store, data_cfg, flop_model, split="train")
    pack = _find_pack_with_docs(packs)
    ds = PackedDataset(store, [pack], data_cfg)
    item = ds[0]
    input_ids = item["input_ids"].numpy()
    labels = item["labels"].numpy()
    position_ids = item["position_ids"].numpy()
    doc_ids = item["doc_ids"].numpy()
    assert input_ids.shape == (data_cfg.seq_len,)
    offset = 0
    for seg_id, seg in enumerate(pack.segments):
        tokens = store.read(seg.start, seg.length)
        length = seg.length
        np.testing.assert_array_equal(input_ids[offset : offset + length], tokens)
        if length > 1:
            np.testing.assert_array_equal(labels[offset : offset + length - 1], tokens[1:])
        assert labels[offset + length - 1] == -100
        np.testing.assert_array_equal(position_ids[offset : offset + length], np.arange(length))
        np.testing.assert_array_equal(doc_ids[offset : offset + length], seg_id)
        offset += length
    assert np.all(labels[offset:] == -100)
    assert np.all(doc_ids[offset:] == -1)


def test_padded_example_is_well_formed(
    store: TokenStore, data_cfg: DataConfig, flop_model: FlopModel
) -> None:
    cfg = DataConfig(**{**data_cfg.__dict__, "mode": "pad"})
    packs = build_packs(store, cfg, flop_model, split="train")
    pack = packs[0]
    ds = PackedDataset(store, [pack], cfg)
    item = ds[0]
    assert item["input_ids"].shape == (cfg.seq_len * pack.rows,)
    for row, seg in enumerate(pack.segments):
        base = row * cfg.seq_len
        tokens = store.read(seg.start, seg.length)
        assert np.array_equal(item["input_ids"][base : base + seg.length].numpy(), tokens)
        assert item["labels"][base + seg.length - 1].item() == -100
        assert torch.all(item["doc_ids"][base : base + cfg.seq_len] == row)
