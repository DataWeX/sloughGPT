---
title: 2026-10-24 — episodic tie-sort TypeError port (card 5bfa16e5)
status: open
date: 2026-10-24
tags: [kanban, domain, context-core, chat-stream, port, bugfix]
card: 5bfa16e5-10d5-45b1-a20d-71d782ed6109
branch: feat/npu-caps
commit: 6f3238f00
---

# Tie-sort TypeError on /chat/stream — ported from unlanded branch `70b021b6a`

**Card**: `Chat stream: tie-sort 500 + reload watcher + SIGTERM swallow (all fixed)` —
moved wip→review; closes when `feat/npu-caps` reaches origin/main.

## Screening finding (why this card stayed wip)

Card claimed "all fixed" with shas `70b021b6a` + `3b605d205` — **neither is an
ancestor of origin/main** (they live on unlanded branches incl.
`chore/wt-hygiene-20261019`). Content check on origin/main:

- **Fix A (tie-sort)**: STILL BROKEN on main —
  `origin/main:domain/infrastructure/_internal/context_core.py:274`
  `scored.sort(reverse=True)` on `(score, episode_dict)` tuples → dict
  comparison on score tie → TypeError → 500 + zero streamed tokens.
- **Fix B (repo-root bootstrap in main.py)**: landed on main by other means
  (main.py has its own path-bootstrapping section).
- **Fix C (uvicorn reload globs)**: landed on main by other means (targeted
  `reload_includes` globs + excludes at main.py:696, not the branch's
  `reload_dirs` approach — same effect, no node_modules blowup).

So the card is 2/3 done on main; only A was missing → port (never re-implement:
fix hunk + TDD test copied verbatim from `70b021b6a`).

## What landed (commit 6f3238f00)

- `domain/infrastructure/_internal/context_core.py`: `scored.sort(key=lambda
  item: item[0], reverse=True)` + the branch's why-comment (4-line change).
- `packages/core-py/tests/test_context_core.py`:
  `test_get_episodic_context_tie_scores_do_not_raise` (ported): identical
  messages → identical scores → ≥2 episodic entries → `get_episodic_context`
  must return str, not TypeError.

## Gate / evidence

- RED: test fails at `context_core.py:274` with the EXACT production error
  `TypeError: '<' not supported between instances of 'dict' and 'dict'`.
- GREEN: one-liner applied → context_core suite 113/113 (exit 0), ruff check +
  format clean.
- `git show --stat HEAD`: 2 files, +21/−1, owned paths only.

## Status

- Board: wip→review (2026-10-24). The fix rides the stack — **the bug remains on
  origin/main until the stack lands** (same lifecycle as the other review cards).
