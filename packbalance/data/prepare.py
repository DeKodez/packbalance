"""Build a compact, memory-mappable token corpus.

Real data uses the open TinyStories corpus and an open tokenizer.  Synthetic
data is useful for systems benchmarks because it lets us control the document
length distribution without changing anything else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from packbalance.config import DataConfig, load_config


def _write_corpus(
    data_dir: Path,
    docs: list[np.ndarray],
    meta: dict,
    token_dtype: str = "uint32",
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([len(d) for d in docs], dtype=np.int64)
    offsets = np.zeros_like(lengths)
    if len(lengths) > 1:
        offsets[1:] = np.cumsum(lengths)[:-1]
    total = int(lengths.sum())
    tokens = np.empty(total, dtype=np.dtype(token_dtype))
    pos = 0
    for d in docs:
        tokens[pos : pos + len(d)] = d
        pos += len(d)
    tokens.tofile(data_dir / "tokens.bin")
    np.save(data_dir / "doc_offsets.npy", offsets)
    np.save(data_dir / "doc_lengths.npy", lengths)
    meta = dict(meta)
    meta.update(
        {
            "num_docs": len(docs),
            "num_tokens": total,
            "dtype": token_dtype,
            "min_doc_len": int(lengths.min()) if len(lengths) else 0,
            "max_doc_len": int(lengths.max()) if len(lengths) else 0,
            "mean_doc_len": float(lengths.mean()) if len(lengths) else 0.0,
        }
    )
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"wrote {len(docs):,} docs / {total:,} tokens to {data_dir} "
        f"(mean length {meta['mean_doc_len']:.1f})"
    )


def prepare_synthetic(cfg: DataConfig, data_dir: Path) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.synthetic_num_docs
    lengths = rng.lognormal(mean=cfg.synthetic_mu, sigma=cfg.synthetic_sigma, size=n).astype(
        np.int64
    )
    lengths = np.clip(lengths, cfg.synthetic_min_len, cfg.synthetic_max_len)
    vocab = int(getattr(cfg, "synthetic_vocab_size", 49152))
    docs = []
    for length in tqdm(lengths, desc="synthetic docs"):
        ids = rng.integers(0, vocab, size=length, dtype=np.int64)
        ids[-1] = cfg.eos_token_id
        docs.append(ids)
    meta = {
        "source": "synthetic",
        "seed": cfg.seed,
        "tokenizer": None,
        "vocab_size": vocab,
        "eos_token_id": cfg.eos_token_id,
        "pad_token_id": cfg.pad_token_id,
        "distribution": {
            "name": "lognormal",
            "mu": cfg.synthetic_mu,
            "sigma": cfg.synthetic_sigma,
            "min": cfg.synthetic_min_len,
            "max": cfg.synthetic_max_len,
        },
    }
    _write_corpus(data_dir, docs, meta)


def prepare_tinystories(cfg: DataConfig, data_dir: Path, max_tokens: int | None = None) -> None:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "TinyStories preparation requires `datasets` and `transformers`. "
            "Install with: pip install -e '.[data]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"tokenizer {cfg.tokenizer} has no eos token")
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos)
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    limit = cfg.max_docs
    docs: list[np.ndarray] = []
    total_tokens = 0
    iterator = iter(dataset)
    for _ in tqdm(range(limit) if limit is not None else None, desc="tokenizing TinyStories"):
        try:
            example = next(iterator)
        except StopIteration:
            break
        text = example["text"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < cfg.min_doc_len:
            continue
        arr = np.asarray(ids + [eos], dtype=np.int64)
        docs.append(arr)
        total_tokens += len(arr)
        if max_tokens is not None and total_tokens >= max_tokens:
            break
    meta = {
        "source": "roneneldan/TinyStories",
        "seed": cfg.seed,
        "tokenizer": cfg.tokenizer,
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token_id": eos,
        "pad_token_id": pad,
    }
    _write_corpus(data_dir, docs, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config with a data section")
    parser.add_argument("--dataset", choices=["synthetic", "tinystories"], default=None)
    parser.add_argument("--out", default=None, help="override data.data_dir")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config.data
    if args.dataset is not None:
        cfg.dataset = args.dataset
    if args.max_docs is not None:
        cfg.max_docs = args.max_docs
    out = Path(args.out or cfg.data_dir)
    if cfg.dataset == "synthetic":
        prepare_synthetic(cfg, out)
    elif cfg.dataset == "tinystories":
        prepare_tinystories(cfg, out, max_tokens=args.max_tokens)
    else:
        raise ValueError(f"unknown dataset {cfg.dataset}")


if __name__ == "__main__":
    main()
