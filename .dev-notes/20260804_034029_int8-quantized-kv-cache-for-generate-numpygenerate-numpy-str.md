---
id: 20260804_034029_int8-quantized-kv-cache-for-generate-numpygenerate-numpy-str
title: int8 quantized KV cache for generate_numpy/generate_numpy_stream (~4x KV memory)
status: done
tags: quantization,inference
created: 2026-08-04T03:40:29.824207+00:00
---

int8 quantized KV cache for generate_numpy/generate_numpy_stream (~4x KV memory)

Added quantize_kv (None=auto/True=int8/False=fp32) to generate_numpy and generate_numpy_stream. New quantize_kv_tensor/dequantize_kv_tensor per-token-per-head int8 helpers in quantization.py with zero-vector guard; new _alloc_kv_cache in slonet.py returning int8+scale buffers (3.76x total memory ratio at head_dim=64, data exactly 4x smaller). Auto-enables on quantized models, bit-exact vs explicit True; fp32 models unchanged by default. Greedy token agreement 100% on trained model, >=96% on untrained fixture. 8 new tests in TestInt8QuantizedKvCache; 23 tests in test_quantization_integration.py pass; full default suite 11166 passed, 41 skipped, 0 failures. QUANTIZATION_BENCHMARK_PLAN.md updated. (test_shell_repl_more.py failures observed on some runs are flaky untracked shell-TUI-wave tests referencing missing _cmd_jobs/_cmd_more, unrelated to this change.)