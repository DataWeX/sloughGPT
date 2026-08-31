---
id: 20260803_010322_wave-f-slonet-provider-test-coverage-43-91
title: Wave F: slonet_provider test coverage 43% -> 91%
status: done
tags: slonet,slc,provider,tests
created: 2026-08-03T01:03:22.506293+00:00
---

Wave F: slonet_provider test coverage 43% -> 91%

Wave F complete. tests/test_slonet_wave_f.py: 36 tests covering SloNetChatProvider: build_slnc helper (header, metadata, JSON config, entry table, aligned data offset), from_slnc (metadata/generate/resource-manager apply+error, rope config), module helpers, constructor+safetensors BF16, _build_prompt branches, tokenizer load, chat/generate/stream paths, quantize=True paths, converter edge cases, generate_with_logprobs seed, fused QKV bias, LLaMA SwiGLU. Combined provider coverage: 43% -> 91% (461 stmts, 42 miss). Non-measurable: 236/243-259 (provably-unreachable dead paths: transpose global keys, fused-QKV fallback), 801-805/817-820/828-844/851-853 (chat_stream 30s/120s timeout + err_q race-dependent branches). Family run: 510 passed, 3 skipped; adjacent suites 309 passed.