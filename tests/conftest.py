from __future__ import annotations

from pathlib import Path

import pytest

from packbalance.config import DataConfig, ModelConfig
from packbalance.data.cost import FlopModel
from packbalance.data.prepare import prepare_synthetic
from packbalance.data.store import TokenStore


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        max_seq_len=64,
        tie_embeddings=True,
        attention_backend="eager",
        compile_attention=False,
        use_fused_rmsnorm=True,
        use_fused_swiglu=True,
        use_fused_ce=False,
        ce_chunk_size=16,
    )


@pytest.fixture
def data_cfg(tmp_path: Path) -> DataConfig:
    return DataConfig(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        seq_len=64,
        mode="isoflop",
        pad_batch_size=4,
        max_doc_len=64,
        min_doc_len=4,
        synthetic_num_docs=240,
        synthetic_mu=3.5,
        synthetic_sigma=0.8,
        synthetic_min_len=8,
        synthetic_max_len=120,
        synthetic_vocab_size=128,
        num_workers=0,
        prefetch_factor=1,
        val_fraction=0.2,
        seed=7,
    )


@pytest.fixture
def store(data_cfg: DataConfig) -> TokenStore:
    prepare_synthetic(data_cfg, Path(data_cfg.data_dir))
    return TokenStore.open(data_cfg.data_dir)


@pytest.fixture
def flop_model(model_cfg: ModelConfig) -> FlopModel:
    return FlopModel.from_config(model_cfg)
