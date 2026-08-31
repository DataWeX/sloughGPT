---
id: 20260804_102237_kv-memory-column-in-kv-reuse-benchmark-int8-vs-fp32
title: KV memory column in KV-reuse benchmark (int8 vs fp32)
status: done
tags: kv-cache,quantization,benchmark
created: 2026-08-04T10:22:37.111780+00:00
---

KV memory column in KV-reuse benchmark (int8 vs fp32)

Added KV KiB column to scripts/benchmark_kv_reuse.py via kv_state_memory_kb() (sums nbytes over per-block K/V buffers + int8 scale buffers). Reported per turn in all three modes (direct, --stack, --stack --stream). float32 grows 48->92->136->180 KiB, int8 15->28.8->42.5->56.2 KiB across 4 turns (3.2x at head_dim=16, matches 8E vs 2E+8 theory). tests/test_benchmark_kv_reuse.py 27->32 (+5: empty state->0, exact 3.2x int8 ratio, memory grows with turns, int8 < float32 at matched turns, stack rows carry session KV memory). Doc updated.