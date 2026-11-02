#!/usr/bin/env bash
# Run the GPU validation + benchmark suite on a Linux GPU box.
# Set SKIP_CPU_TESTS=1 on a slow pod; the CPU suite is validated separately.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install -e ".[data,dev,gpu]" >/dev/null
python -m packbalance.data.prepare --config configs/tiny.yaml >/dev/null
python -m packbalance.data.prepare --config configs/bench.yaml >/dev/null

if [[ "${SKIP_CPU_TESTS:-0}" != "1" ]]; then
  echo "== CPU tests =="
  pytest -m "not slow" -q
fi

echo "== GPU kernel tests =="
pytest -m gpu -q
echo "== FlexAttention parity =="
pytest tests/test_attention.py::test_flex_attention_matches_eager_on_cuda -q

echo "== kernel microbenchmarks =="
python -m packbalance.profiling.bench_kernels --device cuda \
  --rows 8192 --dim 768 --tokens 2048 --vocab 49152 --chunk 1024 \
  --output artifacts/kernel_bench_gpu.json

echo "== scheduler balance =="
python -m packbalance.profiling.bench_schedule --config configs/bench.yaml \
  --world-size 8 --output artifacts/schedule_bench_gpu.json

echo "== end-to-end packing benchmark =="
python -m packbalance.profiling.bench_train --config configs/bench.yaml \
  --steps 20 --warmup 5 --device cuda --output artifacts/train_bench_gpu.json

echo "== profile =="
rm -rf artifacts/profile_gpu
python -m packbalance.train.train --config configs/bench.yaml --max-steps 8 \
  --profile --output-dir artifacts/profile_gpu
TRACE="$(ls -t artifacts/profile_gpu/profile/trace_*.json | head -1)"
python -m packbalance.profiling.analyze_trace --trace "$TRACE" \
  --output artifacts/profile_report_gpu.md

echo "== open-source model LM-head check =="
python -m packbalance.profiling.bench_open_model --model HuggingFaceTB/SmolLM-135M \
  --tokens 1024 --output artifacts/open_model_bench.json

if [[ "${WORLD_SIZE_GPUS:-1}" -ge 2 ]]; then
  echo "== 2-GPU DDP smoke =="
  torchrun --standalone --nproc_per_node=2 -m packbalance.train.train \
    --config configs/tiny.yaml --max-steps 2 --output-dir artifacts/ddp_smoke
  echo "== 2-GPU FSDP2 smoke =="
  torchrun --standalone --nproc_per_node=2 -m packbalance.train.train \
    --config configs/tiny.yaml --max-steps 2 --fsdp --output-dir artifacts/fsdp_smoke
else
  echo "== 2-rank CPU/gloo DDP/FSDP smoke =="
  CUDA_VISIBLE_DEVICES="" timeout 300 torchrun --standalone --nproc_per_node=2 \
    -m packbalance.train.train --config configs/tiny.yaml --max-steps 1 \
    --output-dir artifacts/ddp_smoke
  CUDA_VISIBLE_DEVICES="" timeout 300 torchrun --standalone --nproc_per_node=2 \
    -m packbalance.train.train --config configs/tiny.yaml --max-steps 1 --fsdp \
    --output-dir artifacts/fsdp_smoke
fi

echo "done"
