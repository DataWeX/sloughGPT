---
id: 20260731_111700_test-blitz-round-7-shared-ops-inference-be
title: Test blitz round 7: shared ops + inference backend
status: done
tags: tests,coverage,blitz
created: 2026-07-31T11:17:00.000000+00:00
---

Test blitz round 7: shared ops + inference backend

Round 7: test_ml_types (69), test_arch_config (17), test_numpy_ops_forward (34), test_conversion_tracker (31), test_dataset_manifest (20), test_bpe_tokenizer (26). Two real bugs fixed: (1) ArchConfig.resolve substituted {i} BEFORE weight_map lookup so mapped templates never matched; (2) forward_fast SwiGLU branch checked substituted key instead of literal layers.{i}.ffn.gate.weight:0, so branch never ran. FOUND+fixed third real bug: RoPE applied on (heads, seq, hd) transposed layout but rope() expects (seq, heads, hd) — rotated per-head instead of per-position, breaking LLaMA incremental KV parity; fixed all three forward paths. Round 7 batch (8 files): 256 tests collected, exit 0.