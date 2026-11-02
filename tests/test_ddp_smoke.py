from __future__ import annotations

import os
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from packbalance.config import DataConfig, ModelConfig, PackBalanceConfig, TrainConfig


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.slow
def test_two_rank_ddp_smoke(tmp_path: Path, store, data_cfg) -> None:
    if sys.platform != "linux":
        pytest.skip("gloo process-group init hangs on this macOS/PyTorch build; run on Linux")
    cfg = PackBalanceConfig(
        model=ModelConfig(
            vocab_size=128,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            max_seq_len=64,
            attention_backend="eager",
            use_fused_ce=False,
            ce_chunk_size=16,
        ),
        data=DataConfig(
            dataset="synthetic",
            data_dir=data_cfg.data_dir,
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
        ),
        train=TrainConfig(
            output_dir=str(tmp_path / "ddp"),
            max_steps=1,
            grad_accum=1,
            lr=1e-3,
            min_lr=1e-4,
            warmup_steps=1,
            dtype="fp32",
            log_interval=1,
            save_interval=0,
            eval_interval=0,
            gpu_hourly_cost=0.0,
            peak_tflops=0.0,
        ),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    port = _free_port()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "-m",
            "packbalance.train.train",
            "--config",
            str(config_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=240,
        env={
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
