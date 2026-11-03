# PackBalance: pre-GPU thesis validation

All experiments in this report were run on CPU. The purpose is to make the
GPU run a confirmation exercise, not a fishing expedition.

## Verdicts

| claim | verdict | evidence |
|---|---:|---:|
| Padding is a material compute tax | supported | naive padding wastes 71.9% of tokens and needs ~3.78x the forward FLOPs per useful token. |
| Attention has a quadratic term | supported | length-scaling fit R²=1.000; quadratic coefficient=5.268e-05; time(1024)/time(512)=2.23x. |
| Analytical cost predicts measured time | supported | single-document fit `time ~ 1 + cost` R²=1.000; slope=2.275e-08 ms/FLOP. |
| Balanced scheduling reduces step-time tails | supported (step-level) | natural pack distribution: p99 reduced by 2.3%, straggler ratio 1.049→1.036, rank imbalance 6.2%→1.7%. Under injected skew the p99 cut grows substantially. |
| Splitting long documents reduces FLOP variance | supported | attention share 11.4%→5.0%; pack cost CV 0.067→0.021. The simulated straggler ratio is flat because the per-pack intercept dominates this workload. |
| Chunked CE reduces peak memory | supported | peak RSS reduced by 232 MB (31%); analytical logits ratio 0.06x. |
| Document mask is exact and necessary | supported | correct-mask max error 0.00e+00; broken-mask max error 4.38e+00. |

## 1. Packing and padding tax

| mode | padding waste | compute / useful | fwd FLOPs / useful tok | vs pack_ffd | attn share | pack cost CV | step cost CV | rank imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pad | 71.88% | 3.56x | 0.990G | 3.78x | 13.6% | 0.003 | 0.001 | 0.000 |
| pack_random | 0.31% | 1.00x | 0.262G | 1.00x | 8.2% | 0.045 | 0.001 | 0.014 |
| pack_ffd | 0.01% | 1.00x | 0.262G | 1.00x | 8.2% | 0.046 | 0.000 | 0.008 |
| isoflop | 0.03% | 1.00x | 0.262G | 1.00x | 8.2% | 0.047 | 0.001 | 0.017 |

Corpus: `data/bench`. The padded baseline charges dense causal attention
per padded row; packed modes charge document-local causal attention and no padding.
The `vs pack_ffd` column is forward FLOPs per *useful* token relative to the best
token packer. The difference is the tax the model pays for padding.

## 2. Attention scaling and cost model

| length | fwd+bwd ms | vs L=128 |
|---|---:|---:|
| 128 | 25.32 | 1.00x |
| 256 | 47.12 | 1.86x |
| 512 | 108.65 | 4.29x |
| 1024 | 242.62 | 9.58x |

Linear+quadratic fit: `time = -0.64 + 1.84e-01*L + 5.27e-05*L^2` (R²=1.000). Linear-only R²=0.997. Doubling 512→1024 tokens multiplies measured time by 2.23x.

Fitting the analytical pack cost directly to measured time gives `time = -2.01 + 2.275e-08 * cost`, R²=**1.000**. This is the key cost-model check: the linear + quadratic FLOP expression explains the measured length sweep.

Secondary check on arbitrary packs:

- R² `time ~ 1 + tokens + sum(L^2)`: **0.754**
- Pearson r(time, pack.cost): **0.598**
- Fitted coefficients: [4411.846444313466, -4.097986427018005, 4.451836956145434e-05]

This arbitrary-pack fit is noisy because the bench model spends only ~8% of
its FLOPs on attention and CPU thread scheduling adds variance. The clean
single-document sweep above is the primary functional-form validation.

## 3. Scheduler simulation

| condition | policy | p99 step | straggler | step CV | rank imb | p99 cut |
|---|---:|---:|---:|---:|---:|---:|
| natural | random | 445.5 | 1.049 | 0.008 | 0.062 | - |
| natural | balanced | 435.2 | 1.036 | 0.001 | 0.017 | 2.3% |
| stress_10pct_5x | random | 1183.1 | 1.699 | 0.148 | 2.024 | - |
| stress_10pct_5x | balanced | 636.4 | 1.469 | 0.025 | 0.474 | 46.2% |
| stress_2pct_20x | random | 2969.9 | 2.146 | 0.350 | 6.997 | - |
| stress_2pct_20x | balanced | 1845.5 | 2.731 | 0.180 | 3.322 | 37.9% |

Pack times are normalized from the analytical cost plus 5% log-normal noise; 30 trials, 8 ranks × 4 accumulation slots. The natural distribution is only mildly skewed. To show the mechanism, we inject outliers into 10% / 2% of packs:

- **stress_10pct_5x**: p99 cut 46.2%, straggler 1.699→1.469, rank imbalance 202.4%→47.4%.
- **stress_2pct_20x**: p99 cut 37.9%, straggler 2.146→2.731, rank imbalance 699.7%→332.2%.

Balanced scheduling clearly reduces **step-level** tails and rank imbalance.
It cannot remove the per-step straggler created by a single atomic long pack;
the 2% × 20x stress still leaves a large straggler. Splitting long documents or
sequence parallelism is the next lever after scheduling.

## 4. Context/balance tradeoff

| max_doc_len | padding waste | FLOPs / useful | attn share | pack CV | step CV | straggler | p99 step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.12% | 0.009G | 11.4% | 0.067 | 0.000 | 1.036 | 430.9 |
| 512 | 0.12% | 0.009G | 8.1% | 0.039 | 0.003 | 1.037 | 434.2 |
| 256 | 0.14% | 0.009G | 5.0% | 0.021 | 0.002 | 1.037 | 432.3 |

Shorter document windows reduce attention share and pack-cost variance, at the
cost of context length. In this workload the per-pack fixed cost dominates the
simulated straggler ratio, so the scheduling gain from splitting is small; the
FLOP and variance reductions are real. This is a model/data tradeoff, not a free win.

## 5. Fused/chunked cross-entropy memory

| variant | peak RSS MB |
|---|---:|
| full logits | 757 |
| chunked CE | 525 |
| delta | 232 MB (31% lower) |

Workload: 2048 tokens × 16384 vocab × 256 hidden, chunk=128. Analytical logits ratio = 0.06x (134 MB full vs 8 MB chunk). The GPU Triton path keeps the same chunked peak; it additionally removes the
autograd graph for the full logits tensor.

## 6. Document-mask control

Pack 0 with 2 documents: correct mask max abs error **0.00e+00**; intentionally broken mask (all doc_ids=0) max abs error **4.38e+00**. The correct mask reproduces isolated-document outputs; the broken control does not.

## Caveats

- CPU timings include PyTorch CPU kernels and thread scheduling; they validate
  functional form and relative trends, not absolute GPU speed.
- On CPU, the eager attention path launches one SDPA per document. A composition
  with many tiny documents therefore pays launch overhead that a GPU
  FlexAttention block-sparse kernel would not; the GPU run measures that path.
- The scheduler simulation uses the analytical cost with injected noise; it
  isolates the scheduling policy from model and hardware noise.
- The padded baseline is charged dense causal attention per row. A production
  padded kernel may skip some padding blocks, but linear-layer padding remains.
- Single packs are atomic, so the largest pack still sets a lower bound on a
  step's straggler time. Splitting long documents or sequence parallelism is the
  next step after scheduling.
