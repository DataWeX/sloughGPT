---
id: 20260804_082626_cross-turn-kv-reuse-benchmark-script
title: Cross-turn KV reuse benchmark script
status: done
tags: inference,kv-cache,benchmark
created: 2026-08-04T08:26:26.974600+00:00
---

Cross-turn KV reuse benchmark script

Completed scripts/benchmark_kv_reuse.py: runnable SloTransformer benchmark measuring per-turn cached-token reuse, warm vs cold latency, speedup, and output consistency. Conversation-shaped prompts built from real previous outputs. Verified: reused 0/12/23/34 across 4 turns, 100% warm/cold consistency, overall 1.30x. 11 tests in test_benchmark_kv_reuse.py all pass. Plan doc updated with measured results.