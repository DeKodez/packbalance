from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from packbalance.config import DataConfig, ModelConfig, PackBalanceConfig, TrainConfig


def _config(tmp_path: Path, data_dir: str) -> PackBalanceConfig:
    return PackBalanceConfig(
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
            data_dir=data_dir,
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
            schedule="balanced",
        ),
        train=TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=2,
            grad_accum=1,
            lr=1e-3,
            min_lr=1e-4,
            warmup_steps=1,
            dtype="fp32",
            log_interval=1,
            save_interval=1,
            eval_interval=0,
            gpu_hourly_cost=0.0,
            peak_tflops=0.0,
        ),
    )


def _run(config_path: Path, *args: str) -> subprocess.CompletedProcess:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-m", "packbalance.train.train", "--config", str(config_path), *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_training_smoke_and_resume(tmp_path: Path, store, data_cfg) -> None:
    cfg = _config(tmp_path, data_cfg.data_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
    result = _run(config_path, "--max-steps", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    run_dir = tmp_path / "run"
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "checkpoints" / "step_0000001" / "state.pt").exists()

    result = _run(config_path, "--max-steps", "2", "--resume")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run_dir / "checkpoints" / "step_0000002" / "state.pt").exists()
