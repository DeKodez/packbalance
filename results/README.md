# Results

Raw artifacts behind the README tables and plots. All GPU runs used the validation pod:

- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition MIG `1g.24gb` (25.4 GB usable), 8 vCPU
- **Software:** PyTorch 2.9.1+cu130, CUDA 13.0, Triton 3.5.1
- **Rate:** $0.69/hr; the pod was stopped after collection.

These are engineering measurements, not a leaderboard claim. The systems benchmark uses a synthetic log-normal document-length distribution to isolate padding and scheduling effects; the TinyStories run is a short real-data sanity run, not a full pretraining run.

## `gpu/`

| file | contents |
|---|---|
| `train_bench_gpu.json` | 100M model, 2048-token sequences, 20 timed steps, pad vs packed modes |
| `kernel_bench_gpu.json` | RMSNorm, SwiGLU, and linear+CE microbenchmarks |
| `schedule_bench_gpu.json` | random vs balanced scheduler on the same pack list |
| `open_model_bench.json` | SmolLM-135M LM-head check at 1024 tokens |
| `profile_report_gpu.md` / `.json` | 8-step profile: phase breakdown and top kernels |

Notes:
- The padded benchmark uses 4 rows × 2048 tokens. A 16,384-token padded batch hit a PyTorch/MIG `create_block_mask` allocator bug; the packed 2,048-token path is unaffected.
- The `train_bench` MFU uses the config's assumed 312 TFLOPS. It is indicative only; the MIG slice's calibrated peak was not measured.
- `train_bench` runs with `use_fused_ce=true`. The kernel table shows the fused CE tradeoff: lower memory than full logits, but slower on this shape.

## `thesis/`

- `report.md` and `results.json`: the CPU pre-GPU validation of the padding tax, quadratic attention, cost model, scheduler simulation, CE memory, and document-mask control.
- Reproduce: `python -m packbalance.profiling.prove_intuition --config configs/bench.yaml`.

## `tinystories/`

- `config.yaml`: exact run config (49.3M parameters, 1024-token context, 5,000 docs, 1,000 steps, `pack_ffd`, balanced scheduler, bf16, chunked CE).
- `metrics.jsonl`: full train/eval log.
- `loss_curve.svg` / `loss_curve.png`: train loss (raw + EMA 50), validation loss, and tokens/s.
- `sample.txt`: three generations from the final checkpoint.

Summary:

| metric | value |
|---|---:|
| documents / tokens | 5,000 / 1.03M |
| model parameters | 49.3M |
| optimizer steps | 1,000 |
| tokens seen | 4.02M |
| wall time | 151.5s |
| train loss | 10.76 → 2.16 |
| validation loss | 6.79 → 2.896 |
| tokens/s after warmup | ~36.5k |
| peak GPU memory | 1.60 GB |

Reproduce:

```bash
python -m packbalance.data.prepare --config configs/small.yaml \
  --dataset tinystories --max-docs 5000 --max-tokens 1500000 --out data/tinystories
torchrun --standalone --nproc_per_node=1 -m packbalance.train.train \
  --config configs/tinystories.yaml
python scripts/plot_metrics.py \
  --metrics artifacts/tinystories/metrics.jsonl \
  --output results/tinystories/loss_curve.svg
```

The final checkpoint is not included; it lived on the stopped pod. This is a short systems/learning sanity run, not a full pretraining result.
