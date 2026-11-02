"""Distributed packed-sequence pretraining entrypoint.

Usage:
    torchrun --standalone --nproc_per_node=8 -m packbalance.train.train --config configs/small.yaml

The loop is deliberately explicit: a systems project should make it easy to see
where a step's time and memory go.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.profiler import record_function

from packbalance.config import PackBalanceConfig, load_config, save_config
from packbalance.data.cost import FlopModel
from packbalance.data.dataset import build_dataloader, to_device
from packbalance.data.packer import build_packs, pack_stats
from packbalance.data.sampler import build_schedule
from packbalance.data.store import TokenStore
from packbalance.model.attention import build_doc_block_mask
from packbalance.model.llama import LlamaForCausalLM, count_parameters
from packbalance.profiling.profiler import make_profiler
from packbalance.train.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from packbalance.train.distributed import (
    cleanup_distributed,
    init_distributed,
    no_sync,
    unwrap_model,
    wrap_model,
)
from packbalance.utils import (
    AsyncCheckpointWriter,
    JsonlLogger,
    get_autocast_dtype,
    get_rank,
    get_world_size,
    is_main_process,
    seed_everything,
)


def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _make_optimizer(model: torch.nn.Module, cfg: PackBalanceConfig) -> torch.optim.Optimizer:
    kwargs = dict(
        lr=cfg.train.lr,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay,
    )
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _make_scheduler(optimizer, cfg: PackBalanceConfig, last_epoch: int = -1):
    warmup = max(1, cfg.train.warmup_steps)
    total = max(1, cfg.train.max_steps)
    min_ratio = cfg.train.min_lr / cfg.train.lr if cfg.train.lr else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _clip_grads(model: torch.nn.Module, max_norm: float) -> torch.Tensor:
    fn = getattr(model, "clip_grad_norm_", None)
    if callable(fn):
        try:
            return fn(max_norm)
        except (TypeError, RuntimeError):
            pass
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _build_loader(
    store: TokenStore,
    packs,
    cfg: PackBalanceConfig,
    indices: list[int],
):
    return build_dataloader(store, packs, cfg.data, indices)


def _forward_microbatch(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    cfg: PackBalanceConfig,
    *,
    need_block_mask: bool,
):
    input_ids = batch["input_ids"].unsqueeze(0)
    labels = batch["labels"].unsqueeze(0)
    position_ids = batch["position_ids"].unsqueeze(0)
    doc_ids = batch["doc_ids"].unsqueeze(0)
    block_mask = None
    if need_block_mask:
        with record_function("packbalance/block_mask"):
            block_mask = build_doc_block_mask(doc_ids, cfg.model.n_heads)
    with record_function("packbalance/forward"):
        return model(
            input_ids,
            labels=labels,
            position_ids=position_ids,
            doc_ids=doc_ids,
            block_mask=block_mask,
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    store: TokenStore,
    val_packs,
    cfg: PackBalanceConfig,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    indices = list(range(rank, len(val_packs), world_size))
    limit = min(len(indices), cfg.train.eval_steps * cfg.train.grad_accum)
    indices = indices[:limit]
    if not indices:
        model.train()
        return {"val_loss": float("nan"), "val_tokens": 0}
    loader = _build_loader(store, val_packs, cfg, indices)
    total = torch.zeros((), device=device, dtype=torch.float32)
    tokens = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = _forward_microbatch(
            model,
            batch,
            device,
            cfg,
            need_block_mask=device.type == "cuda",
        )
        assert out.loss is not None
        total += out.loss.detach().float()
        tokens += int(batch["useful_tokens"])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    tokens_t = torch.tensor([tokens], device=device, dtype=torch.float64)
    _all_reduce_sum(tokens_t)
    global_tokens = int(tokens_t.item())
    model.train()
    return {
        "val_loss": float(total.item() / global_tokens) if global_tokens else float("nan"),
        "val_tokens": global_tokens,
    }


def train(cfg: PackBalanceConfig, device: torch.device, resume: bool = False) -> None:
    rank = get_rank()
    world_size = get_world_size()
    output_dir = Path(cfg.train.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, output_dir / "config.yaml")
    seed_everything(cfg.train.seed, cfg.train.deterministic)

    flop_model = FlopModel.from_config(cfg.model)
    store = TokenStore.open(cfg.data.data_dir)
    train_packs = build_packs(store, cfg.data, flop_model, split="train")
    val_packs = (
        build_packs(store, cfg.data, flop_model, split="val") if cfg.data.val_fraction > 0 else []
    )
    stats = pack_stats(train_packs, flop_model)
    if is_main_process():
        print(
            f"[data] {cfg.data.mode}: {stats['num_packs']:,} packs, "
            f"{stats['useful_tokens']:,} useful tokens, "
            f"padding waste {stats['padding_waste']:.2%}, "
            f"cost CV {stats['cv']:.3f}",
            flush=True,
        )

    epoch = 0
    schedule = build_schedule(
        train_packs,
        world_size,
        cfg.train.grad_accum,
        seed=cfg.data.seed,
        mode=cfg.data.schedule,
    )
    steps_per_epoch = schedule.steps_per_epoch
    if is_main_process():
        s = schedule.stats()
        print(
            f"[schedule] steps/epoch={steps_per_epoch:,} "
            f"step_cost_cv={s['step_cost_cv']:.3f} "
            f"rank_imbalance={s['rank_cost_imbalance']:.3f}",
            flush=True,
        )

    start_step = 0
    latest = find_latest_checkpoint(output_dir) if resume else None

    model = LlamaForCausalLM(cfg.model).to(device)
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if cfg.train.compile:
        if world_size > 1:
            print("[warn] torch.compile is disabled for distributed runs", flush=True)
        else:
            model = torch.compile(model)

    model = wrap_model(model, cfg.train, device)
    optimizer = _make_optimizer(model, cfg)
    if latest is not None:
        meta = load_checkpoint(
            latest, unwrap_model(model), optimizer, fsdp=cfg.train.fsdp, device=device
        )
        start_step = int(meta.get("step", 0))
        if is_main_process():
            print(f"[resume] loaded {latest} at step {start_step}", flush=True)
    scheduler = _make_scheduler(optimizer, cfg, last_epoch=start_step - 1)

    amp_dtype = get_autocast_dtype(cfg.train.dtype) if device.type == "cuda" else None
    use_scaler = device.type == "cuda" and cfg.train.dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    logger = JsonlLogger(output_dir / "metrics.jsonl")
    writer = AsyncCheckpointWriter() if not cfg.train.fsdp else None
    if is_main_process():
        logger.log(
            {
                "event": "start",
                "world_size": world_size,
                "device": str(device),
                "params": count_parameters(unwrap_model(model)),
                "pack_stats": stats,
                "schedule_stats": schedule.stats(),
                "config": asdict(cfg),
            }
        )

    prof = None
    if cfg.train.profile:
        prof = make_profiler(
            output_dir / "profile",
            wait=1,
            warmup=cfg.train.profile_warmup,
            active=cfg.train.profile_steps,
        )
        prof.start()

    if start_step >= steps_per_epoch and steps_per_epoch > 0:
        epoch = start_step // steps_per_epoch
        epoch_step = start_step % steps_per_epoch
        schedule = build_schedule(
            train_packs,
            world_size,
            cfg.train.grad_accum,
            seed=cfg.data.seed + epoch,
            mode=cfg.data.schedule,
        )
    else:
        epoch_step = start_step

    indices = schedule.rank_indices(rank)
    if epoch_step > 0:
        indices = indices[epoch_step * cfg.train.grad_accum :]
    loader = _build_loader(store, train_packs, cfg, indices)
    loader_iter = iter(loader)
    step = start_step
    running_loss = 0.0
    running_tokens = 0
    running_data_ms = 0.0
    run_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        while step < cfg.train.max_steps:
            if epoch_step >= len(schedule.steps):
                epoch += 1
                epoch_step = 0
                schedule = build_schedule(
                    train_packs,
                    world_size,
                    cfg.train.grad_accum,
                    seed=cfg.data.seed + epoch,
                    mode=cfg.data.schedule,
                )
                indices = schedule.rank_indices(rank)
                loader = _build_loader(store, train_packs, cfg, indices)
                loader_iter = iter(loader)
            plan = schedule.steps[epoch_step]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.perf_counter()
            data_start = time.perf_counter()
            with record_function("packbalance/data"):
                local_packs = [next(loader_iter) for _ in range(cfg.train.grad_accum)]
            data_ms = (time.perf_counter() - data_start) * 1e3
            step_tokens = plan.step_tokens
            if step_tokens <= 0:
                raise RuntimeError("step has no useful tokens")
            loss_scale = world_size / step_tokens
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device=device, dtype=torch.float32)

            for micro, batch in enumerate(local_packs):
                batch = to_device(batch, device)
                last = micro == cfg.train.grad_accum - 1
                autocast_ctx = (
                    autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_dtype is not None
                    else contextlib.nullcontext()
                )
                with no_sync(model, enabled=world_size > 1 and not last):
                    with autocast_ctx:
                        out = _forward_microbatch(
                            model,
                            batch,
                            device,
                            cfg,
                            need_block_mask=device.type == "cuda",
                        )
                    assert out.loss is not None
                    loss = out.loss * loss_scale
                    with record_function("packbalance/backward"):
                        if use_scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                local_loss += out.loss.detach().float()

            with record_function("packbalance/optimizer"):
                if use_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = _clip_grads(model, cfg.train.grad_clip)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_ms = (time.perf_counter() - step_start) * 1e3
            with record_function("packbalance/comm"):
                global_loss = _all_reduce_sum(local_loss.detach()) / step_tokens
            tokens_per_sec = step_tokens / max(step_ms / 1e3, 1e-9)
            step_flops = 3.0 * plan.step_cost
            mfu = None
            if cfg.train.peak_tflops > 0 and step_ms > 0:
                mfu = step_flops / (step_ms / 1e3) / (world_size * cfg.train.peak_tflops * 1e12)
            usd_per_b = (
                (1e9 / tokens_per_sec) * (world_size * cfg.train.gpu_hourly_cost / 3600.0)
                if tokens_per_sec > 0
                else None
            )
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
            running_loss += float(global_loss.item()) * step_tokens
            running_tokens += step_tokens
            running_data_ms += data_ms

            if step % cfg.train.log_interval == 0:
                logger.log(
                    {
                        "step": step,
                        "loss": float(global_loss.item()),
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm)
                        if not isinstance(grad_norm, torch.Tensor)
                        else float(grad_norm.item()),
                        "tokens_per_sec": tokens_per_sec,
                        "step_ms": step_ms,
                        "data_ms": data_ms,
                        "gpu_mem_gb": gpu_mem,
                        "mfu": mfu,
                        "usd_per_b_tokens": usd_per_b,
                        "tokens": step_tokens,
                        "cost_cv": schedule.stats()["step_cost_cv"],
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

            if cfg.train.eval_interval and val_packs and (step + 1) % cfg.train.eval_interval == 0:
                metrics = evaluate(model, store, val_packs, cfg, device, rank, world_size)
                if is_main_process():
                    logger.log({"step": step, **metrics})

            if cfg.train.save_interval and (step + 1) % cfg.train.save_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step + 1,
                    output_dir / "checkpoints",
                    fsdp=cfg.train.fsdp,
                    async_writer=writer,
                    extra={"loss": float(global_loss.item()), "tokens": step_tokens},
                )

            if prof is not None:
                prof.step()
            step += 1
            epoch_step += 1

    except KeyboardInterrupt:
        if is_main_process():
            print("[train] interrupted; saving checkpoint", flush=True)
        save_checkpoint(
            model,
            optimizer,
            step,
            output_dir / "checkpoints",
            fsdp=cfg.train.fsdp,
            async_writer=writer,
            extra={"interrupted": True},
        )
    finally:
        if prof is not None:
            prof.stop()
        if writer is not None:
            writer.wait()
        if step > start_step:
            elapsed = time.perf_counter() - run_start
            if is_main_process():
                avg_loss = running_loss / max(1, running_tokens)
                logger.log(
                    {
                        "event": "finish",
                        "step": step,
                        "avg_loss": avg_loss,
                        "tokens": running_tokens,
                        "elapsed_s": elapsed,
                        "avg_data_ms": running_data_ms / max(1, step - start_step),
                    }
                )
        final_dir = save_checkpoint(
            model,
            optimizer,
            step,
            output_dir / "checkpoints",
            fsdp=cfg.train.fsdp,
            async_writer=writer,
            extra={"final": True},
        )
        if writer is not None:
            writer.wait()
        if is_main_process():
            print(f"[train] final checkpoint: {final_dir}", flush=True)
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-fused-ce", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    if args.fsdp:
        cfg.train.fsdp = True
    if args.compile:
        cfg.train.compile = True
    if args.profile:
        cfg.train.profile = True
    if args.no_fused_ce:
        cfg.model.use_fused_ce = False
    device = init_distributed(args.device)
    train(cfg, device, resume=args.resume)


if __name__ == "__main__":
    main()
