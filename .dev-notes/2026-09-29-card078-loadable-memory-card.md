---
id: 20260929_078
board: f74b9e75-7673-4b6b-9c00-4e6082c346bb
title: "Card 078 — Loadable memory card: persistent model memory"
status: done
date: 2026-09-29
tags: [memory]
---

# Loadable memory card: persistent model memory

## Background

Implement the loadable 'memory card' (asked 2026-08-27: 'build the memory card like a real memory card that can be loaded so the model remembers'). Spec already exists in docs/FORMAL_SPECS.md; zero implementation (no MemoryCard/load_memory code). TDD: save/load/restore lifecycle + integration with model load path.

## Source
Mined from stale opencode sessions (Aug 26-31 2026) at user's request: ~/.cache/slog-journeys/session-exports/.

## Done (2026-09-30)

Implemented on branch `feat/loadable-memory-card`: `domain/memory/_internal/memory_card.py`
(slo-memory-card/v1, checksummed, atomic writes), MemoryService save/load/list/delete card
methods (replace = autosave-first, abort on backup failure; merge = idempotent),
`SLO_MEMORY_ACTIVE_CARD` auto-load on `POST /models/load`, `/memory/cards` API.
24 TDD tests; full suite 4410 passed.
