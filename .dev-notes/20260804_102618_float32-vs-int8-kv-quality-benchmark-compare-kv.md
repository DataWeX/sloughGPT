---
id: 20260804_102618_float32-vs-int8-kv-quality-benchmark-compare-kv
title: float32-vs-int8 KV quality benchmark (--compare-kv)
status: done
tags: kv-cache,quantization,benchmark
created: 2026-08-04T10:26:18.335442+00:00
---

float32-vs-int8 KV quality benchmark (--compare-kv)

Added compare_kv_quality() + print_quality_report() to scripts/benchmark_kv_reuse.py. Builds one shared conversation, then generates cold float32 vs int8 KV outputs on identical prompts per turn. Measured: 96.7% overall identical (turn 0/2/3 100%, turn 1 87% - one near-tie argmax flip then re-convergence). Exits 1 with WARNING below 50% agreement. tests/test_benchmark_kv_reuse.py 32->39 (+7: structure, length bounds, turn-0 perfect, >=90% overall, prefix bounds, both exit paths). Doc updated.