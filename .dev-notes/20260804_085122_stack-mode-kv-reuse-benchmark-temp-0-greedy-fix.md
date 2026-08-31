---
id: 20260804_085122_stack-mode-kv-reuse-benchmark-temp-0-greedy-fix
title: Stack-mode KV reuse benchmark + temp-0 greedy fix
status: done
tags: inference,kv-cache,benchmark
created: 2026-08-04T08:51:22.729594+00:00
---

Stack-mode KV reuse benchmark + temp-0 greedy fix

Extended scripts/benchmark_kv_reuse.py with --stack mode driving SloNetServer.generate(session_id=) through the real provider session-KV path (vocab=256 char tokenizer). Found core bug: serving stack defaults top_k=50/top_p=0.9, so temperature=0.0 was not greedy (both _is_greedy and the _sample_from_logits fast path required top_p is None) and fell through to unseeded np.random.choice -> identical cold prompts produced different tokens (stack consistency 33.3%). Fixed _sample_from_logits to return argmax whenever temperature<1e-6 (top-k/nucleus filtering cannot change the argmax). Stack run: reuse 0->11->22->33, overall 1.49x, 100% consistency. Added 4 tests (stack structure/growth/consistency + greedy regression); 15 total in test_benchmark_kv_reuse.py. KV/chat suites green.