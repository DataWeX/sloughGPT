---
id: 20260924_053_fix_stream_replay_whitespace
status: done
title: Fix remaining whitespace-losing decodes: coalescer replay, morph ▁, multimodal BPE
tags: [bug, streaming, tokenizer, follow-up]
---
# Follow-up to 20260924_049 — remaining whitespace-losing decodes

## Scope (from card)
1. `apps/api/server/routers/inference.py` coalescer replay (~1386, ~2419):
   `cached.split()` + `" ".join` flattens newlines / multi-space on reconnect replay.
2. `domain/infrastructure/_internal/morph_tokenizer.py` byte_fallback decode:
   drops ▁ (U+2581) space for every token when called single-token (`first=True` always)
   on the chat stream path via slonet_provider.
3. `domain/multimodal/_internal/bpe_tokenizer.py` `BPETokenizer.decode` `.strip()` —
   same class of bug.

## Constraints
- Keep full-sequence round-trip tests green.
- TDD: failing test first per site.
- Gates: ruff on touched surface, targeted pytest suites, then relevant benchmark.

## Result
Fixed all 3 sites, TDD RED→GREEN, merged to main `408878f7c` (commit `2bc1bfd17`).

1. `inference.py` — new `_chunk_replay(text, size=40)` char-slicing helper; both
   coalescer replay sites (generate ~1398, chat ~2433) yield 40-char chunks via
   it instead of `split()`+`" ".join`; reconnect replay now byte-for-byte exact.
2. `morph_tokenizer.py:808` — `first = len(tokens) > 1` in byte_fallback branch:
   single-token (streaming) keeps ▁ as space, full sequence still suppresses
   first-token ▁ (SentencePiece convention preserved).
3. `bpe_tokenizer.py:143` — `if len(tokens) <= 1: return text` before `.strip()`;
   card-049 SloBPE pattern applied to multimodal BPETokenizer.

Gates:
- 10 new tests (morph 4, bpe 2, replay 4 incl. endpoint wiring) RED→GREEN.
- Merged-tree run: 95 passed, 40 skipped (morph HF-model skips), ruff clean.
- Streaming consumers: 404 passed (provider/chat_stream/slonet suites).
- Adjacent 7-suite batch: 33 failed = identical baseline with fix reverted
  (pre-existing, none in touched files — card-051 territory).
- Benchmark `scripts/benchmark_stability.py` skipped: needs live :8000 server
  with loaded model (same skip as card 049).

Board card flipped done; conflict on `.kanban/board.jsonl` vs main resolved
with branch side (strict superset: 0 main-only card ids).
