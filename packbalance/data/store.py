"""Memory-mapped token store shared by all data-pipeline components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TokenStore:
    data_dir: Path
    tokens: np.memmap
    doc_offsets: np.ndarray
    doc_lengths: np.ndarray
    meta: dict

    @classmethod
    def open(cls, data_dir: str | Path) -> TokenStore:
        data_dir = Path(data_dir)
        meta_path = data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. Run `python -m packbalance.data.prepare --config ...` first."
            )
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        tokens = np.memmap(
            data_dir / "tokens.bin",
            dtype=np.dtype(meta.get("dtype", "uint32")),
            mode="r",
        )
        offsets = np.load(data_dir / "doc_offsets.npy")
        lengths = np.load(data_dir / "doc_lengths.npy")
        if len(offsets) != len(lengths):
            raise ValueError("doc_offsets.npy and doc_lengths.npy disagree")
        if len(tokens) != int(meta["num_tokens"]):
            raise ValueError(f"tokens.bin has {len(tokens)} tokens, meta says {meta['num_tokens']}")
        return cls(
            data_dir=data_dir, tokens=tokens, doc_offsets=offsets, doc_lengths=lengths, meta=meta
        )

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def read(self, start: int, length: int) -> np.ndarray:
        return np.asarray(self.tokens[start : start + length], dtype=np.int64)

    def read_many(self, starts: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
        return [self.read(int(s), int(n)) for s, n in zip(starts, lengths, strict=False)]
