---
id: 20260804_130036_quantization-benchmark-runnable-tiny-mode-e2e-tests
title: Quantization benchmark: runnable tiny mode + e2e tests
status: done
tags: quantization,benchmark
created: 2026-08-04T13:00:36.239768+00:00
---

Quantization benchmark: runnable tiny mode + e2e tests

Stabilized scripts/benchmark_quantization.py as a runnable offline benchmark: deterministic tiny in-process SloTransformer (seeded), interleaved NQ/Q timing, contiguous throughput-test window (fixes [1]-vs-[7] contradiction from OpenBLAS CPU-boost drift), informational tiny speed gates, clean --json (human output suppressed, +bits/tiny keys).

Full int8 sweep: gen geomean 1.18x, prompt 1.30x, temp 1.24x, regression 1.54x, 4.0x weight compression, logit cosine 0.9996, 7/7 PASS. Full int4 sweep: gen 0.46x, prompt 0.70x, 8.0x compression, cosine 0.8524, 7/7 PASS. int4 wins memory (8x), int8 wins speed; >1.3x is a GPT-2-scale claim gated only on --model gpt2.

Added packages/core-py/tests/test_quantization_benchmark_e2e.py (10 tests): full quick int8 run gates (7/7 names/order, ~4x compression, cosine>=0.95, JSON shape, throughput metrics, markdown report shape), int4 compression+quality (8x, cosine>=0.85), seeded-model determinism, tiny speed-gate band.

Added --report [PATH] flag (to_markdown()/write_report): env/config header, per-test metric tables (tps speedups, memory, cosine, cold/warm), Notes block; works alongside --json. Combined quantization suite 154 passed / 3 skipped / 0 failed (incl. test_quantization.py, integration, kv_reuse 44).

Updated docs/features/QUANTIZATION_BENCHMARK_PLAN.md: status Implemented, runnable usage block (incl. --report), stability explanation, measured int8/int4 tables, Files-to-Create and Success Criteria/Execution Order reconciled to implemented state.