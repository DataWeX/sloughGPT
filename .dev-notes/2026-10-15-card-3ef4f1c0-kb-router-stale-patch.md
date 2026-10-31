---
id: 3ef4f1c0
title: "test_kb_router 7F both trunks: @patch('domain.learner.knowledge.get_knowledge_memory') target obsolete — router now binds domain.knowledge.engine.get_knowledge_engine"
status: done
board_id: Main
date: 2026-10-15
tags: [tests, kb, drift, stale-baseline]
---

# Card 3ef4f1c0 — closed: already fixed on main, card mined from a stale baseline

**No code change.** The card's described defect is real as written, but it was
fixed **eleven days before the card was mined**. Verified by measurement, not
inference.

## What the card claims

7 failures in `tests/server/test_kb_router.py` on local main *and* origin/main
(`TestListKnowledge` x2, `TestCreateKnowledge`, `TestSearchKnowledge`,
`TestBatchDelete`, `TestDeleteKnowledge` x2), because every test decorates with
`@patch("domain.learner.knowledge.get_knowledge_memory")` — a module path that
no longer resolves as an attribute chain (`knowledge` moved to
`domain/learner/_internal/` and is imported lazily), so
`pkgutil.resolve_name` dies with
`AttributeError: module 'domain.learner' has no attribute 'knowledge'` at
decoration time. Root cause and prescribed fix are both correct on their face.

## What is true now

- `tests/server/test_kb_router.py` on **current `main` (`cff97ef5a`)**:
  **46 passed, 1 skipped, 0 failed** — run in a detached worktree at `main`,
  not in this session's branch, so the result cannot be attributed to local
  changes. Same result on this session's tree.
- The card's exact root cause was fixed by **`bc77e199e` — "tests: fix baseline
  7 — stale patch targets + optional-dep skip" (2026-10-04)**, which touches
  `tests/server/test_kb_router.py` (15 lines changed). Note "baseline 7" in
  that message — the same seven failures.
- The obsolete target is gone: `grep -c "domain.learner.knowledge.
  get_knowledge_memory" tests/server/test_kb_router.py` → **0**.

## The lesson: this card was mined from a stale delta

The card says *"Mined 2026-10-15 from the ca647de2 local delta."* That delta
reported 7 failures that had been gone from `main` since 2026-10-04, so the
baseline it was computed against predated the fix by eleven days.

This is worth carrying forward, because two sibling cards were mined from the
same delta and had different fates:

| card | mined from ca647de2 | actual state |
|---|---|---|
| `3ef4f1c0` kb_router 7F | "7F both trunks" | **already fixed** (`bc77e199e`, 2026-10-04) |
| `b0d8e72f` quantization 13F | "13F both trunks" | **already on main** |
| `97aebbe4` kernel_devices 24F | "24F both trunks" | real, then fixed today by another session (`57cbe039c`, still stranded) |

**Recommendation for future mining runs:** re-run the candidate file against
current `main` in a throwaway detached worktree before filing the card. Filing
from a delta computed on a stale checkout generates phantom work — a card that
looks open, costs an investigation to disprove, and dilutes the queue.

## Evidence

- Detached worktree at `main` (`cff97ef5a`): `tests/server/test_kb_router.py`
  → `46 passed, 1 skipped, 1 warning in 7.50s`.
- `git merge-base --is-ancestor bc77e199e main` → true; commit dated
  2026-10-04; `git show --stat` lists `tests/server/test_kb_router.py`.
- Obsolete patch-target occurrence count in the test file → 0.
- No collision: no commit anywhere names `3ef4f1c0`, and the knowledge
  branches (`feat/knowledge-consolidate`, `feat/migrate-knowledge-plurals`,
  `fix/knowledge-preexisting`) hold no worktrees.
