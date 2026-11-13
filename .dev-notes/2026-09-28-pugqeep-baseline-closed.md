---
id: 20260928_072_pugqeep_baseline_closed
status: done
title: "Card 072 — pugqeep: close 21-test red baseline in canonical tree"
tags: [pugqeep, tests, engine, debt]
sprint: current
gh: null
---
# Card 072 — pugqeep baseline closure

## Scope (user-directed)

"Youre not building pugeeq well" → close the documented 21-test red baseline
against `docs/PUGQEEP.md` contracts. User decisions during audit:

- Untracked broken `test_compressed_transfer_wire.py`: **leave it, use
  `--ignore`** — never touch another session's file.
- Impl-vs-test truth: **check git, check docs, check their session first** —
  verdict: tests encode intent; debt documented since Sep 24
  (card `20260924_047`: "21f IDENTICAL to baseline").

## Audit findings

- Baseline `21f` = **13 ImportError** (`decompress_tree`/`load_from_points`
  missing from `model_tree` re-exports) + **8 contract drift** (health shape,
  wait_for timeout return, elapsed, spawn message, stream snapshot,
  CancelManager registration, library exports).
- Same 21 fail on `main` — proven via `git archive main` run in throwaway dir.
- Dual-tree: this branch's plural tree mid-deletion (35 staged D, another
  session); canonical = `domain/infrastructure/_internal/pugqeep/` on `main`.

## Fix (branch `fix/pugqeep-suite-red`, isolated worktree)

- `model_tree.py`: re-export `decompress_tree`, `load_from_points` + `__all__`
- `engine.py`: `Engine.health["trees"]`; `GuardTree.health[
  "active_subprocesses"]`; `ProcessGroup._created_at` + age-fallback
  `elapsed`; detached error matches `"not attached"`; `wait_for` returns live
  process on timeout; `stream_results` snapshot copy;
  `spawn(register_cancel=True)` → `CancelManager.register(op_id=proc.id,
  cancel_fn=proc.cancel)` finished via `on_complete`/`on_fail`
- `__init__.py`: export `save_library`, `load_library`
- `docs/PUGQEEP.md`: stale `engine.submit` example → `spawn`/`wait_for`;
  test paths → `packages/core-py/tests/`

## Verification

- 4 previously-red files green; full pugqeep suite **1804 passed** (exit 0);
  cancel_manager consumer suites **95 passed**; ruff check/format clean.
- Benchmarks: n/a (test-conformance only).
- Landed: fast-forward to `main` @ `114d76b19` (tree hash identical to
  verified branch `e78cfd11f`). Temp worktree + branch cleaned up.

## Open

- Wire-test file (`test_compressed_transfer_wire.py`) left untouched per
  user — still breaks root pytest collection on the gateway branch unless
  `--ignore`d.
