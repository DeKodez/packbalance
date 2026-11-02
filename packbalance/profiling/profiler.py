"""torch.profiler setup with Chrome trace export and a readable summary."""

from __future__ import annotations

import time
from pathlib import Path

import torch


def make_profiler(
    output_dir: str | Path,
    *,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def on_trace_ready(prof) -> None:
        path = output_dir / f"trace_{int(time.time())}.json"
        prof.export_chrome_trace(str(path))
        print(f"[profiler] wrote {path}", flush=True)
        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=25), flush=True)

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    return torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
