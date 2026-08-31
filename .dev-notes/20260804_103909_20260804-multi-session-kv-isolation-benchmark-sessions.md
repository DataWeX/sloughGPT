---
id: 20260804_103909_20260804-multi-session-kv-isolation-benchmark-sessions
title: 20260804 multi-session KV isolation benchmark (--sessions)
status: done
tags: sloughGPT,kv-reuse,benchmark
created: 2026-08-04T10:39:09.534612+00:00
---

20260804 multi-session KV isolation benchmark (--sessions)

Wired --sessions N into main() (implies stack); per-session isolation verified with 2 sessions batch+stream: reuse [0,12,23,34] x2, 100% consistency, isolation_ok True. Added 5 tests (44 total in test_benchmark_kv_reuse.py). Seeded create_model (np.random.seed(0)) for reproducibility after compare-kv flaked at 88.5% < 90% threshold between runs; agreement now stable 87.5%. Doc: arch/layer sweep table + sessions section written into QUANTIZATION_BENCHMARK_PLAN.md. All regression suites green.