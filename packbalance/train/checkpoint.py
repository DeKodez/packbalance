"""Checkpointing with atomic writes for DDP and DCP for FSDP2.

Reliability is a training-systems feature: a run that cannot resume is a run
that has to start over.  DDP writes are atomic (tmp + rename); FSDP uses
torch.distributed.checkpoint so sharded state never has to be gathered on one
rank.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from packbalance.utils import AsyncCheckpointWriter, is_main_process


def _latest_step_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.split("_")[-1]))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str | Path,
    *,
    fsdp: bool,
    async_writer: AsyncCheckpointWriter | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if fsdp and dist.is_available() and dist.is_initialized():
        import torch.distributed.checkpoint as dcp

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        dcp.save(state, checkpoint_id=str(ckpt_dir / "dcp"))
        if is_main_process():
            (ckpt_dir / "meta.json").write_text(
                json.dumps({"step": step, **(extra or {})}, indent=2), encoding="utf-8"
            )
        return ckpt_dir

    if not is_main_process():
        return ckpt_dir
    raw = model.module if hasattr(model, "module") else model
    state = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    path = ckpt_dir / "state.pt"
    if async_writer is not None:
        accepted = async_writer.submit(state, path)
        if not accepted:
            async_writer.wait()
            async_writer.submit(state, path)
    else:
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
    (ckpt_dir / "meta.json").write_text(
        json.dumps({"step": step, **(extra or {})}, indent=2), encoding="utf-8"
    )
    return ckpt_dir


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    fsdp: bool,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(path)
    if fsdp and dist.is_available() and dist.is_initialized():
        import torch.distributed.checkpoint as dcp

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else {},
        }
        dcp.load(state, checkpoint_id=str(path / "dcp"))
        meta_path = path / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return meta
    raw = model.module if hasattr(model, "module") else model
    state = torch.load(path / "state.pt", map_location=device, weights_only=False)
    raw.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return {"step": int(state.get("step", 0)), **state.get("extra", {})}


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    return _latest_step_dir(Path(output_dir) / "checkpoints")
