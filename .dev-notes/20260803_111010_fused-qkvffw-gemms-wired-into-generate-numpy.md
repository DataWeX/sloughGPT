---
id: 20260803_111010_fused-qkvffw-gemms-wired-into-generate-numpy
title: Fused QKV/FFW GEMMs wired into generate_numpy
status: done
tags: sloughgpt,quantization,inference,optimization
created: 2026-08-03T11:10:10.383119+00:00
---

Fused QKV/FFW GEMMs wired into generate_numpy

Fused same-input quantized GEMMs in the inlined generate_numpy/generate_numpy_stream loops: [W_q;W_k;W_v] (1152x896) and [w1;w3] (9728x896) built once via _fuse_quant_weights and emitted as single quantized_linear calls, sliced back to q/k/v and gate/up. Blocks also fuse in forward_numpy via SloMultiHeadAttention._fused_qkv and SloFeedForward._fused_gate_up. Result: 52.69 vs 55.92 ms/token (3.23 ms/token saved, ~6%), tokens bit-identical. GEMM profile: (1,128,896)x432 and (1,4864,896)x432 replaced by (1,1152,896)x216 and (1,9728,896)x216. Added test_fused_gemm_generation_bit_identical to test_slonet_provider_real.py (monkeypatches _fuse_quant_weights to None). All suites green: provider_real 26, quantization 124, quant_core 14.