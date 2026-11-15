---
id: 20260929_2e3f6ddc
title: "Canonical local embedder — one formalized implementation + delegating shims"
status: done
date: 2026-09-29
tags: [inference, embeddings, refactor]
---

# Canonical local embedder — one formalized implementation + delegating shims

## Background

Standardize ALL text embedding into one canonical local embedder (user ask:
no API money/bandwidth, no torch/sentence-transformers). Chain: SloNet
checkpoint (quality-gated) → word n-gram TF-IDF (numpy), dim=384 + L2
discipline owned by the canonical class. Kills the second algorithms
(InMemoryEmbedder md5-sin, meta_weights builtin `hash()` — nondeterministic
across processes).

## Done (2026-09-29)

- Canonical: `domain/inference/_internal/text_embedder.py` (`TextEmbedder`)
- All entry points delegate: `simple_embed`, `EmbeddingService`,
  context_core/auto_ingest wrappers, `Embedder`/`InMemoryEmbedder`,
  meta-weights, memory consolidation
- TDD `tests/test_canonical_embedder.py` (10/10), benchmark
  `scripts/benchmark_embeddings.py` (5613 docs/s, 100% cross-entry-point
  agreement, dim=384, norm=1.0)
- Commit `70f3cda65`, pushed to `feat/shared-governance`
