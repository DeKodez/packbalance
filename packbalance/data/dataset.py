"""Turn pack metadata into model-ready tensors.

Every example is one *sequence*: either a packed sequence of ``seq_len`` tokens
or a padded batch represented as ``rows * seq_len`` tokens.  The model receives
only token ids, labels, position ids, and a document id per token; the
document id is what makes packed attention exact.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from packbalance.config import DataConfig
from packbalance.data.packer import Pack
from packbalance.data.store import TokenStore


class PackedDataset(Dataset):
    def __init__(self, store: TokenStore, packs: list[Pack], cfg: DataConfig) -> None:
        self.store = store
        self.packs = packs
        self.seq_len = cfg.seq_len
        self.pad_token_id = cfg.pad_token_id

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pack = self.packs[index]
        total = pack.rows * self.seq_len if pack.kind == "padded" else self.seq_len
        input_ids = np.full(total, self.pad_token_id, dtype=np.int64)
        labels = np.full(total, -100, dtype=np.int64)
        position_ids = np.zeros(total, dtype=np.int64)
        doc_ids = np.full(total, -1, dtype=np.int64)

        if pack.kind == "padded":
            for row, seg in enumerate(pack.segments):
                base = row * self.seq_len
                tokens = self.store.read(seg.start, seg.length)
                length = seg.length
                input_ids[base : base + length] = tokens
                if length > 1:
                    labels[base : base + length - 1] = tokens[1:]
                position_ids[base : base + length] = np.arange(length, dtype=np.int64)
                # The padded row is one attention document, including its pad
                # suffix; causal masking keeps real tokens independent of it.
                doc_ids[base : base + self.seq_len] = row
        else:
            offset = 0
            for seg_id, seg in enumerate(pack.segments):
                tokens = self.store.read(seg.start, seg.length)
                length = seg.length
                input_ids[offset : offset + length] = tokens
                if length > 1:
                    labels[offset : offset + length - 1] = tokens[1:]
                position_ids[offset : offset + length] = np.arange(length, dtype=np.int64)
                doc_ids[offset : offset + length] = seg_id
                offset += length
            if offset > self.seq_len:
                raise ValueError(f"pack {pack.pack_id} overflows seq_len: {offset}")

        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "position_ids": torch.from_numpy(position_ids),
            "doc_ids": torch.from_numpy(doc_ids),
            "pack_id": pack.pack_id,
            "useful_tokens": pack.tokens,
            "compute_tokens": pack.compute_tokens,
            "cost": pack.cost,
        }


class ListSampler(Sampler[int]):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _worker_init(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % (2**32 - 1) + worker_id)


def build_dataloader(
    store: TokenStore,
    packs: list[Pack],
    cfg: DataConfig,
    indices: list[int],
) -> DataLoader:
    dataset = PackedDataset(store, packs, cfg)
    sampler = ListSampler(indices)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "sampler": sampler,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
        "worker_init_fn": _worker_init,
    }
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(**kwargs)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out
