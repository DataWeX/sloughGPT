---
id: 20261109_134510_flake-test-ai-fails-fast-without-model-cap-dependent-count-d
status: done
board: 20261109_134510_flake-test-ai-fails-fast-without-model-cap-dependent-count-d
title: "flake: test_ai_fails_fast_without_model — cap-dependent count delta on the shared LogBuffer ring"
tags: ["ci-flake", "test-core-py", "shell"]
sprint: "2026-11"
gh: null
---

# flake: test_ai_fails_fast_without_model — cap-dependent count delta on the shared LogBuffer ring

# flake: test_ai_fails_fast_without_model — cap-dependent count delta on the shared LogBuffer ring

**Status:** todo
**Sprint:** 2026-11
**Tags:** ci-flake, test-core-py, shell

## What fails

CI/CD `test-core-py` (run `42603882090`, commit `eb6f94fdb`):
`tests/core-py/test_shell_repl_more.py::TestCmdAiLLMExecution::test_ai_fails_fast_without_model`
— `AssertionError: no ERROR entry added to shell log buffer` (`assert 1209 > 1209`).
Suite: **1 failed / 43335 passed** (2894s) — everything else green.

Second occurrence ever. First: single sighting recorded as "left open" in today's
AGENT_SYNC entry (quant A-B flake card `20261105_170745`).

## Why it is not a lane's diff (three-strike, proven 2026-11-09)

1. **Pristine by construction** — the card `20261101_154058` commits
   (`4cf21a117`, `a5eaa6933`, `eb6f94fdb`, 21 files) have **zero overlap** with
   `domain/shell/*` or `test_shell_repl_more.py`. The failing path is 100% untouched code.
2. **Deterministic local repro, no lane code involved** — isolated run passes
   (fresh buffer). A ring-saturation mirror of CI's state fails **byte-identical**.
3. **Documented pre-existing** — AGENT_SYNC: "`test_shell_repl_more::test_ai_fails_fast_without_model`
   (single occurrence, left open)".

## Root mechanism

`domain/shell/_internal/log_buffer.py`: `LogBuffer` is a `deque(maxlen=2000)` **ring**,
process-wide singleton (`get_log_buffer()`; `repl._log_buffer` points at it — `repl.py:395`).
The test does a **count delta**:

```python
before = len(repl._log_buffer.get(level="ERROR"))
out = capture_cmd(repl, repl._cmd_ai, "show models")
...
errs = repl._log_buffer.get(level="ERROR")
assert len(errs) > before, "no ERROR entry added to shell log buffer"
```

Once a full-suite session (43k tests) fills the ring to 2000, appending the command's
one ERROR **evicts one ERROR** (deque head = oldest): the ERROR-filtered count stays
flat → false negative. Pass/fail depends only on whether the oldest entry at eviction
time is an ERROR — a coin flip that **can only fire in full-suite CI**, never in
isolation. Line 882's `errs[-3:]` tail-slice has the same exposure to interleaved
background-logger ERRORs (`slo.pugqeep.engine` etc. — visible in the CI log).

## Repro (evidence: `~/.cache/slo-logs/shell-repl-flake-triage.log`)

- Isolated: `pytest "tests/core-py/test_shell_repl_more.py::TestCmdAiLLMExecution::test_ai_fails_fast_without_model" -p no:randomly -q` → **1 passed**.
- Ring mirror: fill singleton to 2000 (1209 ERROR oldest-first + 791 INFO = CI's exact
  shape) via plugin `/tmp/opencode/training-consolidation/ringfill.py`, rerun →
  **`assert 1209 > 1209`**, byte-identical to CI.

## Fix direction (user ruling 2026-11-09: card only, no code now)

Strengthen (never weaken) — replace the count-delta with entry-identity comparison:
keep the `before` list (object refs prevent id-reuse), then
`new = [e for e in errs if not any(e is b for b in before)]` — assert `new` non-empty
AND apply the `model`-message check to `new` instead of `errs[-3:]`.
Watched RED = the ringfill repro; GREEN = isolated + ring-saturation both pass.

## Notes

- CI invocation: `pytest tests/core-py/ --ignore=… --cov=domain --cov-fail-under=50 -q --tb=short`
  (`ci_cd.yml` "Run core-py tests", job timeout 60 min; this run's step: 48m14s, no timeout).
- User ruling (2026-11-09): **card it only** — no test edit, no AGENT_SYNC append. **Superseded 2026-11-12** by queue pickup; see RESULTS below.

## RESULTS (done 2026-11-12, branch feat/shell-flake-log-identity, pushed: 321ea750a impl + fd9a0f417 sync)

**Ruling supersession:** the 2026-11-09 "card only — no test edit" ruling was temporal
(triage day, no repro in hand). The user's queue pickup ("yes") supersedes it; flagged
in the close report for veto.

**Fix shipped (strengthen, never weaken):**
```python
before = list(repl._log_buffer.get(level="ERROR"))   # refs keep old entries alive
...
errs = repl._log_buffer.get(level="ERROR")
new = [e for e in errs if not any(e is b for b in before)]
assert new, "no ERROR entry added to shell log buffer"
assert any("model" in e.message.lower() for e in new)   # on new, not errs[-3:]
```
Docstring documents the ring-flake rationale. Only count-delta site in the file
(grep-proven) — no other test touched.

**TDD gates (all watched):**
- RED: `-p ringfill` plugin (1209 ERROR + 791 INFO = CI's exact shape) → `assert 1209 > 1209`,
  byte-identical to CI run 42603882090 (head evicted fill-err-0 → fill-err-1).
- GREEN saturation: same plugin → **1 passed** (7.2s).
- GREEN isolated (no plugin): **1 passed** (1.2s).
- GREEN full file: `test_shell_repl_more.py` → **4066 passed / 56 skipped** (42.9s).
- Post-rebase saturation re-verify: **1 passed**.
- ruff check + format --check: clean.

**MUST NOT held:** assertions strengthened not weakened (identity vs count is strictly
stronger); no pytest.ini marker changes; no test_results/*.json touched; production
code (domain/shell/*) untouched — the ring is correct by design, the test's view of it
was wrong.

**Repro plugin:** `/tmp/opencode/ringfill.py` (fill-only in pytest_runtest_call
tryfirst — the runner's default impl runs the test after the hook).
