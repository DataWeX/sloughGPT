---
id: 20260803_082831_avx2-int8-quantization-quality-fix
title: AVX2 int8 quantization quality fix
status: done
tags: quant,perf,slonet
created: 2026-08-03T08:28:31.052635+00:00
---

AVX2 int8 quantization quality fix

Root cause: per-channel quantized lm_head fed to _nb_lm_head_argmax_int8 with scale[0] in greedy path. Fix round 1: per-channel falls to forward_numpy (AVX2). Fix round 2: removed the fused-argmax branch entirely — without numba it was the 206ms numpy-dequant path (18x slower than AVX2 GEMM at 11.6ms); both greedy sites (slonet.py ~4134, ~4477) now use forward_numpy+argmax for any quantized lm_head. Fixed load_weights() to populate _error_report so summary() reports real quality (avg_cosine=0.9999, avg_mse=4.8e-08) instead of the misleading 0.0000 in the startup log. Verified: 2+2 correct, 13.4 tok/s on server11 (Qwen2.5-0.5B, W8A8 AVX2). Tests: 182 passed/3 skipped (slonet+quantization core); 110 passed/1 flaky timing benchmark (int4_vs_int8, CPU contention, not a code regression — ratio 1.16x idle).