---
id: 20260805_114839_quantization-baseline-deliverable-passed-gate-fix
title: Quantization baseline deliverable + passed-gate fix
status: done
tags: quantization,benchmark,baseline
created: 2026-08-05T11:48:39.866132+00:00
---

Quantization baseline deliverable + passed-gate fix

Committed quantization_baseline.json (tiny int8/int4 + Qwen int8 headline metrics; Qwen int4 excluded as known-failing). Fixed _compare_baselines passed gate: only fires when current and baseline share the same test set (--validate ran 2-test subset vs 7-test baseline -> false passed regression). 5 new TestBaselineCompare unit tests. Added 'quantization-gate' job to reusable-ci-core.yml: runs --validate --bits 8,4 --baseline quantization_baseline.json on every push (numpy-only, deterministic tiny fixture, C-kernel numpy fallback verified). Verified exact CI command exits 0 (No regressions). Report/plan docs updated. All quantization suites green.