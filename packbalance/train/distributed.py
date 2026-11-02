"""Distributed setup and model wrapping (DDP and FSDP2)."""

from __future__ import annotations

import contextlib
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from packbalance.config import TrainConfig
from packbalance.utils import get_local_rank, get_rank, get_world_size


def init_distributed(device_override: str | None = None) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            local_rank = get_local_rank()
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cpu")
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: torch.nn.Module, cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    world_size = get_world_size()
    if world_size <= 1:
        return model
    if cfg.fsdp:
        from torch.distributed.fsdp import fully_shard

        # FSDP2: shard each transformer block bottom-up, then the root.
        layers = getattr(getattr(model, "model", model), "layers", None)
        if layers is not None:
            for layer in layers:
                fully_shard(layer)
        fully_shard(model)
        return model
    return DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        bucket_cap_mb=cfg.bucket_cap_mb,
        find_unused_parameters=cfg.find_unused_parameters,
    )


@contextlib.contextmanager
def no_sync(model: torch.nn.Module, enabled: bool):
    if not enabled:
        yield
        return
    no_sync_fn = getattr(model, "no_sync", None)
    if no_sync_fn is None:
        yield
        return
    with no_sync_fn():
        yield


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def is_main() -> bool:
    return get_rank() == 0
