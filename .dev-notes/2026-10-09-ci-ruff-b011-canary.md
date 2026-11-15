---
title: CI ruff gate RED on B011 — the "never run" canary also dies under python -O
date: 2026-10-09
status: done
branch: feat/aml-stream-loader
tags: [ci, lint, ruff, tests, guard, bug]
---

# Problem

The exact command CI runs (`.github/workflows/ci_cd.yml:35`,
`.github/workflows/reusable-ci-core.yml:27`):

```
ruff check packages/core-py/ apps/api/server/ --exclude '__pycache__'
```

exits **1** — CI is red.

```
B011 Do not `assert False` (`python -O` removes these calls), raise `AssertionError()`
 --> packages/core-py/tests/zslough_check_never_run.py:3:8
```

`per-file-ignores` covers `N801/F811/B017` for `packages/core-py/tests/*.py`
and `F811/B017` for `**/tests/*.py` — **B011 is not among them**, so the
guard trips the gate.

# The guard was also latently broken

The file is a *canary*: its docstring says it proves the file is never
collected (name does not start with `test_`), and it must blow up if anyone
runs it. But `assert False` is compiled out under `python -O`, so:

- normal: `AssertionError: intentional collection guard…` ✓
- `-O`: **guard silently passed** ✗ (verified empirically)

So under optimized mode the canary could be imported without complaint —
the exact failure it exists to catch.

# Fix

`assert False, "…"` → `raise AssertionError("…")`.

Same behavior in normal mode, correct behavior under `-O`, satisfies B011,
and **no lint scope widened** — the alternative (adding `B011` to
`per-file-ignores`) would disable that rule for every test file in the repo.

# Verification

- Exact CI command → exit 0.
- Guard fires in both normal and `-O`.
- Still never collected (filename does not match `test_*.py`; zero references
  anywhere in the repo).

# Also noted (not this card)

Repo-wide `ruff check .` reports only these 2: this one plus `F822` in
`zzz_test.py` — which is an **untracked** scratch file at the repo root, outside
the CI paths, so it cannot turn CI red. The kanban card "Fix 3417 ruff lint
errors across codebase" is therefore **stale** (2, not 3417).
