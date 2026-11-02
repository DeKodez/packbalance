"""Small, dependency-free helpers shared by training, data, and benchmarks."""

from __future__ import annotations

import contextlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def human_int(n: float) -> str:
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return f"{n:.0f}"


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class CudaTimer:
    """Wall-clock timer that synchronizes CUDA on both ends."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self) -> CudaTimer:
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.elapsed_ms = (time.perf_counter() - self._start) * 1e3


@contextlib.contextmanager
def nullcontext():
    yield None


def elapsed_ms(fn) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


class JsonlLogger:
    """Rank-0 JSONL metrics plus readable console output."""

    def __init__(self, path: str | Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled and is_main_process()
        self.path = Path(path) if path is not None else None
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        line = json.dumps(record, sort_keys=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if "step" in record:
            parts = [f"step {record['step']}"]
            for key in (
                "loss",
                "lr",
                "grad_norm",
                "tokens_per_sec",
                "step_ms",
                "data_ms",
                "gpu_mem_gb",
                "mfu",
                "usd_per_b_tokens",
            ):
                if key in record and record[key] is not None:
                    value = record[key]
                    if isinstance(value, float):
                        parts.append(f"{key}={value:.4g}")
                    else:
                        parts.append(f"{key}={value}")
            print(" | ".join(parts), flush=True)
        else:
            print(line, flush=True)


class AsyncCheckpointWriter:
    """Copy a state dict to CPU and write it off the training thread.

    This is deliberately small: it hides checkpoint I/O latency for DDP-style
    state dicts without pretending to be a full distributed checkpointing
    service.
    """

    def __init__(self) -> None:
        self._thread = None
        self._path: Path | None = None

    def submit(self, state: dict[str, Any], path: str | Path) -> bool:
        import threading

        if self._thread is not None and self._thread.is_alive():
            return False

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu_state = _to_cpu(state)

        def _write() -> None:
            tmp = path.with_suffix(path.suffix + ".tmp")
            torch.save(cpu_state, tmp)
            os.replace(tmp, path)

        self._path = path
        self._thread = threading.Thread(target=_write, name="packbalance-ckpt", daemon=True)
        self._thread.start()
        return True

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    return value


def get_autocast_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return None


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
