---
id: 20260803_011803_wave-g-morph-tokenizer-test-coverage-68-99
title: Wave G: morph_tokenizer test coverage 68% -> 99%
status: done
tags: slonet,coverage,tokenizer
created: 2026-08-03T01:18:03.326856+00:00
---

Wave G: morph_tokenizer test coverage 68% -> 99%

Wave G: morph_tokenizer.py coverage 68% -> 99% (401 stmts, 1 miss = line 636, provably unreachable: the fallback's len(word) > len(suffix)+2 gate is identical to decompose's, so a non-empty-form suffix can never leave a root short enough to re-enter the fallback).

- tests/test_morph_tokenizer_wave_g.py: 48 tests added
  - TestFromPretrainedLayouts: snapshot/flat cache layouts, vocab-as-list, byte-level + byte-fallback detection, eos-as-list, post-processor end_token_id, added_tokens, chat_template, invalid JSON tolerance, project-local cache candidate
  - TestSyntheticBpeModes: added-tokens fast path/split/decode, byte-level in-vocab + fallback, char-level, byte_fallback normalize/hex
  - TestChatTemplate: fallback + _render_chat_template im_start/generic/chatml branches
  - TestMorphologyEdges: decompose/stem/form/root/related/batch/index paths
- Combined family (wave_g + existing tokenizer suites): 136 tests, 99% coverage
- slonet-ecosystem regression (provider + tokenizer + legacy): 180 tests pass