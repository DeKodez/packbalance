"""Typed configuration for PackBalance.

The config is intentionally boring: YAML in, dataclasses out, unknown keys are
rejected so a typo cannot silently change a benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ATTENTION_BACKENDS = ("auto", "flex", "eager")
PACKING_MODES = ("pad", "pack_random", "pack_ffd", "isoflop")
SCHEDULES = ("balanced", "random")


@dataclass
class ModelConfig:
    vocab_size: int = 49152
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 1536
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0
    attention_backend: str = "auto"
    compile_attention: bool = True
    use_fused_rmsnorm: bool = True
    use_fused_swiglu: bool = True
    use_fused_ce: bool = True
    ce_chunk_size: int = 1024
    z_loss_coeff: float = 0.0
    gradient_checkpointing: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {ATTENTION_BACKENDS}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.d_ff <= 0 or self.n_layers <= 0:
            raise ValueError("d_ff and n_layers must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.head_dim


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    data_dir: str = "data/synthetic"
    tokenizer: str = "HuggingFaceTB/SmolLM-135M"
    seq_len: int = 2048
    mode: str = "isoflop"
    schedule: str = "balanced"
    pad_batch_size: int = 8
    max_docs: int | None = None
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    val_fraction: float = 0.01
    seed: int = 42
    pad_token_id: int = 0
    eos_token_id: int = 0
    min_doc_len: int = 8
    max_doc_len: int = 2048
    synthetic_num_docs: int = 20000
    synthetic_mu: float = 5.5
    synthetic_sigma: float = 1.0
    synthetic_min_len: int = 16
    synthetic_max_len: int = 4096
    synthetic_vocab_size: int = 49152
    isoflop_iterations: int = 8
    drop_last: bool = True

    def __post_init__(self) -> None:
        if self.mode not in PACKING_MODES:
            raise ValueError(f"mode must be one of {PACKING_MODES}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")


@dataclass
class TrainConfig:
    output_dir: str = "artifacts/run"
    seed: int = 42
    max_steps: int = 2000
    grad_accum: int = 4
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bf16"
    log_interval: int = 10
    eval_interval: int = 0
    eval_steps: int = 20
    save_interval: int = 500
    keep_last: int = 2
    resume: bool = False
    fsdp: bool = False
    compile: bool = False
    profile: bool = False
    profile_steps: int = 5
    profile_warmup: int = 2
    gpu_hourly_cost: float = 2.0
    peak_tflops: float = 0.0
    deterministic: bool = False
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.grad_accum <= 0:
            raise ValueError("grad_accum must be positive")


@dataclass
class PackBalanceConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(dc_type: type, values: dict[str, Any], path: str) -> Any:
    if not isinstance(values, dict):
        raise TypeError(f"{path} must be a mapping, got {type(values).__name__}")
    known = {f.name for f in fields(dc_type)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {path} keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(dc_type):
        if f.name not in values:
            continue
        value = values[f.name]
        if is_dataclass(f.type):
            kwargs[f.name] = _build(f.type, value, f"{path}.{f.name}")
        else:
            kwargs[f.name] = value
    return dc_type(**kwargs)


def load_config(path: str | Path) -> PackBalanceConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError("config root must be a mapping")
    known = {"model", "data", "train"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    return PackBalanceConfig(
        model=_build(ModelConfig, raw.get("model", {}), "model"),
        data=_build(DataConfig, raw.get("data", {}), "data"),
        train=_build(TrainConfig, raw.get("train", {}), "train"),
    )


def save_config(config: PackBalanceConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)


def config_to_dict(config: PackBalanceConfig) -> dict[str, Any]:
    return asdict(config)
