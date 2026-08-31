---
id: 20260805_060238_quantization-benchmark-report-regenerated
title: Quantization benchmark report regenerated
status: done
tags: benchmark,quantization,docs
created: 2026-08-05T06:02:38.708845+00:00
---

Quantization benchmark report regenerated

Regenerated docs/features/QUANTIZATION_BENCHMARK_REPORT.md with the current tool (scripts/benchmark_quantization.py --bits 8,4 --quick --report). The prior report was from 2026-07-13 (real GPT-2 on macOS) and predated token agreement, perplexity, recommendations, and baseline features. New report: tiny-fixture int8+int4 results (7/7 both), per-test tables, precision comparison, recommendations (tiny -> int4, score 0.910), baseline regression check section, updated usage/config/files, and the legacy GPT-2 run preserved as a labeled appendix.