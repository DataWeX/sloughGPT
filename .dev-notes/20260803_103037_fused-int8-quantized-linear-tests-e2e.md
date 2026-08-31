---
id: 20260803_103037_fused-int8-quantized-linear-tests-e2e
title: Fused int8 quantized linear: tests + E2E
status: done
tags: quantization,performance,c
created: 2026-08-03T10:30:37.324644+00:00
---

Fused int8 quantized linear: tests + E2E

Fused per-token quantize + AVX2 int8 GEMM + dequantize + bias C kernel (matmul_int8_f32) verified.

- Added TestFusedInt8 (14 tests) to test_quant_core.py: per-tensor/per-row scale, M>1, odd K, large N, zero-act rows, no-bias, 3D input, asymmetric/x_scale skip-fused guards, fallback when C unavailable.
- Added fused-vs-unfused speed benchmark to test_quantization_benchmark.py.
- Full quant suites: 46 passed (quant_core + benchmark), 124 passed 3 skipped (quantization + provider + coverage).
- E2E quantized Qwen2.5-0.5B: fused 56-58 ms/token vs unfused 63-66 ms/token (~7-8 ms/token saved), generated tokens bit-identical.
- Output verified: 'The capital of France is Paris...'