---
id: 20260803_024259_wave-i-slolibgpu-coverage-85-88-cuda-causal-gelu-bug-fixes
title: Wave I: slolib/gpu coverage (85% -> 88%), CUDA causal + gelu bug fixes
status: done
tags: coverage,gpu,slolib
created: 2026-08-03T02:42:59.470755+00:00
---

Wave I: slolib/gpu coverage (85% -> 88%), CUDA causal + gelu bug fixes

Wave I complete: domains/slolib/gpu/__init__.py at 88% coverage (779 stmts, 91 missed), up from 85%.

## Production bug fixes (gpu/__init__.py)
1. Fused-attention tiled causal mask: cm diag offset k=1+max(0,t_start-N) -> k=1-t_start (line 513). Numeric verify: old maxdiff 2.14 vs fixed 2.2e-16 vs full-attention reference.
2. Cross-entropy validity: flat_t < shape[1] -> (flat_t>=0) & (flat_t<shape[1]) (line 555).
3. conv2d np.pad (padding,) -> (padding,padding) (line 437, numpy 2.5.1 ValueError).
4. conv2d reshape: matmul(w_col,cols.T).reshape(n,oc,oh,ow) -> .T.reshape(n,oh,ow,oc).transpose(0,3,1,2) - old code mixed batch/oc order. Loop diff 9.59 -> 6.2e-7.
5. _CUDABackend.vram_gb(): mem_info bound method subscripted mem[1] -> always TypeError -> silent 4.0 fallback; now calls mem_info().
6. _CUDABackend.scaled_dot_attention causal: triu(ones((n,s))) -> triu(ones((n,s)),k=1) - was masking diagonal+below instead of future.
7. _CUDABackend.gelu: sqrt(pi/pi)=1.0 -> sqrt(2/pi) - CUDA arm diverged from _Accelerator.gelu.

## Tests (tests/test_slolib_gpu.py, 157 tests)
- TestAcceleratorBase: base-class fallback softmax/log_softmax/vram_gb 0.0/memory_hint tier lite/defaults.
- Fallback tests via sys.modules None-ing: resource_manager, psutil.
- TestCudaWithFakeCupy: numpy-backed cupy proxy (_CupArr + _FakeCupy classmethods) exercising CUDA dispatch arms - available/sync/vram tiers/dtype/to-from device/matmul fp32+fp16/scaled_dot causal+mask/layer_norm+gelu. Exposed bugs 5-7 above.
- Precision tests: monkeypatched cpu.matmul returns input with time.sleep(0.02) on deliberately-slower precision (deterministic, no timing flakiness).

## Coverage path
85% -> 88%. Remaining 91 misses are the dispatch floor: Metal torch arms (748-891) and OpenCL pyopencl arms (1045-1101) - libraries not installed; pyopencl path executes real CL kernels, torch path calls real torch ops. Regression: test_gpu_accelerator + test_slonet_kernels + test_fused_kernels + test_slonet_generate all pass.