---
id: 20260801_071938_downcraft-refactor-move-hf-tests-to-core-py-hf-hub
title: downcraft refactor: move HF tests to core-py hf_hub
status: done
tags: downcraft,core-py,refactor
created: 2026-08-01T07:19:38.189361+00:00
---

downcraft refactor: move HF tests to core-py hf_hub

Refactor complete and verified. downcraft is now a pure HF-agnostic HTTP downloader; all HF logic lives in packages/core-py/domains/infrastructure/hf_hub.py. Final suite state:
- downcraft: 57 passed (50 existing + 7 new CLI tests in tests/test_main.py covering the refactored 'url|status|list' surface, removed 'hf'/'verify' subcommands)
- core-py hf/download/size/loader: 147 passed
Combined: 204 passed. No pending fixups for this work item.