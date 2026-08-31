---
id: 20260804_235859_quantization-benchmark-models-multi-model-comparison
title: Quantization benchmark --models multi-model comparison
status: done
tags: benchmark,quantization,cli
created: 2026-08-04T23:58:59.746233+00:00
---

Quantization benchmark --models multi-model comparison

Added teacher-forced perplexity to the quality test: _perplexity (single forward pass, log-sum-exp scoring of token t against logits t-1, returns 1.0 for seq < 2), per-prompt NQ/Q perplexity, top-level nq_perplexity/q_perplexity/perplexity_ratio (rounded 3), gate passed on perplexity_ratio < 1.5 alongside the cosine floor. Surfaced on every comparison surface: PPL ratio (Q/NQ) row in _comparison_table, perplexity_ratio in _comparison_json/_model_comparison/_csv_output and baseline _headline_metrics. Added _BASELINE_LOWER_IS_BETTER (perplexity_ratio: regression = value rises beyond 15%) since the generic compare treats all non-cosine metrics as higher-is-better. Verified live: int8 ratio 1.011, int4 0.96 on tiny, both PASS. Added 8 e2e tests (TestPerplexity). Full quantization suite: 134 prior + 8 new = 142 passed, 3 skipped (74 e2e).