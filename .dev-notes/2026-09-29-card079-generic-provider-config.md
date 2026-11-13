---
id: 20260929_079
board: 90839119-6c74-4938-9353-f266a7c58f9a
title: "Card 079 — Generic provider config: api_key + base_url (drop openai-specific)"
status: done
date: 2026-09-29
tags: [config, api]
---

# Generic provider config: api_key + base_url (drop openai-specific)

## Background

Replace openai_api_key (domain/infrastructure/_internal/config.py:93) with generic api_key + base_url so embeddings/any endpoint work with any provider (asked twice 2026-08-28: 'no make it generic, no specific to openai so we can have any api key and url'). Keep back-compat alias; update consumers, docs, tests.

## Source
Mined from stale opencode sessions (Aug 26-31 2026) at user's request: ~/.cache/slog-journeys/session-exports/.

## Landed (2026-09-29, branch `feat/generic-provider-config`, worktree `sloughgpt-provider`)
- `EmbeddingConfig`: `api_key` + `base_url` replace `openai_api_key`; input alias via
  `AliasChoices("api_key", "openai_api_key")` (constructor/YAML/`model_validate`),
  read alias via `openai_api_key` property, `populate_by_name=True`.
- `_LEGACY_ENV_KEYS` in `_apply_env_overrides`: `SLO_EMBEDDING__OPENAI_API_KEY` →
  `embedding.api_key`; explicitly-set `SLO_EMBEDDING__API_KEY` wins over legacy.
- `OpenAIEmbedder`/`Embedder`/`create_embedder`: `base_url` param, config fallback to
  `embedding.api_key`/`embedding.base_url`, `OpenAI(api_key=..., base_url=...)` only when set,
  error message now names `SLO_EMBEDDING__API_KEY`.
- Docs: `.env.example` embedding block + module/class docstrings (any OpenAI-compatible
  endpoint: vLLM/Ollama/LM Studio). No docs/*.md referenced the old key.
- Tests: +13 (config alias/env/dump round-trip; base_url forwarding/omission/config fallback).
  Gates: ruff clean; `test_config.py + test_embeddings.py` 79 passed;
  `test_embedding_service/test_embeddings2/test_trainer_embeddings` 142 passed.
- Not done (deliberate): renaming `OpenAIEmbedder`/`provider="openai"` (protocol family,
  not vendor; would churn 7 tests for no behavior change).
