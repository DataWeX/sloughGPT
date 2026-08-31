---
id: 20260803_055037_wave-o-native-inference-100-coverage
title: Wave O: native inference 100% coverage
status: done
tags: inference,native-engine,coverage
created: 2026-08-03T05:50:37.972257+00:00
---

Wave O: native inference 100% coverage

Wave O complete. Target modules at 100% line coverage via tests/test_native_engine_real.py: native/bindings.py (44 stmts), native/engine.py (227), native/weight_mapper.py (85), ct_provider.py (44), tokenizer.py (4).

Fixes: weight_mapper._get returns np.zeros(n) for missing weights (was shape-0 broadcast error); tokenizer.py shim fixes dead get_tokenizer import in engine; ct_provider import fixed to .tokenizer.

Tests added: env-var lib load, empty tensors->zeros, explicit mlp biases, llama/gpt2/default chat format dispatch, load_weights failure raises, EOS stop (generate+stream), real-tokenizer tokenize/detokenize + fallbacks, provider chat_stream error path, ct_provider tokenizer init + missing-tokenizer tolerance + metadata.

All 52 tests in test_native_engine_real pass; 362 related slnc/slonet/npu/ops tests pass; tokenizer suites pass. Note: stale __pycache__ after editing tokenizer.py caused transient order-dependent failures — cleared per AGENTS.md.