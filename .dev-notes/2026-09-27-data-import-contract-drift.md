---
title: Fix data_import test contract drift (30 failures)
status: done
date: 2026-09-27
---

# data_import contract drift (card 066)

## Scope (actual: 30 failures, not 6)
- `test_data_import.py`: 25 — `_is_isbn` returns bare digits (7 asserts
  expected stale `isbn:` prefix), `_fetch_gutenberg_text` gained `isbn`
  param (6 direct calls + 5 monkeypatch lambdas were 2-arg),
  `import_data` dispatches to RepoImporter/HuggingFaceImporter/URLImporter
  (tests patched DataImporter methods that no longer route → real git
  clone / HF network calls + IndexError on empty call lists), exact search
  URL fields grew 4 more params, `get_by_isbn` real-network 45s timeouts,
  not-found returns before mkdir.
- `test_import_lora_config.py`: 5 — same `_is_isbn` prefix drift.

## Changes (ed23d43e1)
Tests aligned to current impl contracts + made hermetic (get_by_isbn
patches); `_patch_method` retargets to actual dispatch classes; impl: skip
empty-isbn gutendex lookup.

## Result
Merged to main as `254b59096`. Gates: data_import 214/214, lora+datasets
263/263, post-merge combined 420/420, ruff clean.
