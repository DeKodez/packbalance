# Profile report: `trace_1793616749.json`

- Profiled steps: **5**
- Wall time: **2340.41 ms**

## Step phases

| phase | ms | % of step |
|---|---:|---:|
| packbalance/data | 7.03 | 0.3% |
| packbalance/block_mask | 177.48 | 7.6% |
| packbalance/forward | 981.68 | 41.9% |
| packbalance/backward | 877.77 | 37.5% |
| packbalance/optimizer | 37.53 | 1.6% |
| packbalance/comm | 1.00 | 0.0% |
| other | 257.92 | 11.0% |

## Top GPU kernels

| kernel | ms | % of kernel time |
|---|---:|---:|
| `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long)` | 87.62 | 19.3% |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8::Params)` | 75.71 | 16.7% |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>(cutlass_80_simt_sgemm_128x128_8x4_nn_align1::Params)` | 36.74 | 8.1% |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nt_align1>(cutlass_80_simt_sgemm_128x128_8x4_nt_align1::Params)` | 31.73 | 7.0% |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nt_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nt_align8::Params)` | 26.36 | 5.8% |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8::Params)` | 24.46 | 5.4% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 23.97 | 5.3% |
| `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)` | 22.35 | 4.9% |
| `_ce_kernel` | 16.82 | 3.7% |
| `triton_tem_fused_zeros_1` | 15.37 | 3.4% |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::FusedOptimizerTensorListMetadata<4>, at::native::(anonymous namespace)::FusedAdamMathFunctor<float, 4, (at::native::ADAM_MODE)1, false>, float const*, double, double, double, double, double, bool, float const*, float const*>(at::native::(anonymous namespace)::FusedOptimizerTensorListMetadata<4>, at::native::(anonymous namespace)::FusedAdamMathFunctor<float, 4, (at::native::ADAM_MODE)1, false>, float const*, double, double, double, double, double, bool, float const*, float const*)` | 12.02 | 2.6% |
| `triton_tem_fused_0` | 10.53 | 2.3% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)` | 10.11 | 2.2% |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::native::sum_functor<long, long, long>::operator()(at::TensorIterator&)::{lambda(long, long)#1}>, unsigned int, long, 4, 4> >(at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::native::sum_functor<long, long, long>::operator()(at::TensorIterator&)::{lambda(long, long)#1}>, unsigned int, long, 4, 4>)` | 8.90 | 2.0% |
| `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<2>, at::native::memory::StoreWithCast<1> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<2>, at::native::memory::StoreWithCast<1>)` | 5.42 | 1.2% |

Total kernel time: **453.71 ms**

## Top CPU ops

| op | ms |
|---|---:|
| `aten::item` | 482.70 |
| `aten::_local_scalar_dense` | 482.53 |
| `aten::index` | 368.89 |
| `aten::linear` | 259.61 |
| `aten::to` | 172.96 |
| `autograd::engine::evaluate_function: MmBackward0` | 164.09 |
| `aten::_to_copy` | 158.92 |
| `aten::mm` | 156.66 |
| `MmBackward0` | 144.56 |
| `autograd::engine::evaluate_function: _RMSNormBackward` | 112.37 |
| `aten::copy_` | 111.61 |
| `_RMSNormBackward` | 97.14 |
| `autograd::engine::evaluate_function: ToCopyBackward0` | 96.65 |
| `aten::matmul` | 84.62 |
| `autograd::engine::evaluate_function: CompiledFunctionBackward` | 82.39 |

Total CPU op time: **4813.41 ms**

> Phase ranges are `record_function` labels. DataLoader work inside a worker appears as `packbalance/data` in the main process; CPU preprocessing is visible in the worker trace only when worker tracing is enabled.