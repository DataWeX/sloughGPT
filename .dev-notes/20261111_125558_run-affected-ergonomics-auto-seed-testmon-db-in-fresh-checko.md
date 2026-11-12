---
id: 20261111_125558_run-affected-ergonomics-auto-seed-testmon-db-in-fresh-checko
status: wip
board: 20261111_125558_run-affected-ergonomics-auto-seed-testmon-db-in-fresh-checko
title: "run_affected ergonomics: auto-seed testmon DB in fresh checkouts + fast-phase before slow-phase (tests must feel fast)"
tags: [testing, ci, tooling]
sprint: ""
gh: null
---

# run_affected ergonomics: auto-seed testmon DB + fast-phase before slow-phase

Board card `20261111_125558` (description = the full problem statement + plan;
user ruling 2026-11-11: slow tests RUN AFTER fast subset first; tests faster
overall).

## Design (user-approved via "continue building" 2026-11-12)

**Part A — auto-seed:** selective mode with missing `$ROOT/.testmondata` →
snapshot the primary checkout's DB (located via `git worktree list --porcelain`
first entry, no hardcoded paths) with the sqlite3 backup API (consistent while
a suite records there; manual proof in card: 48k → 355 selected, ~2s). Never
overwrite an existing DB; any seed failure falls back to today's full record
run (unknown state stays a full run — Test Rule clause 3). Helper:
`scripts/seed_testmon.py` (unit-testable; scope addition agreed).

**Part B — fast/slow split:** tracked list `scripts/slow_paths.txt` (exact
file paths, comments; initial = `tests/core-py/test_vm_life_program.py` +
unmarked UI/journey files — grows empirically via `--durations`, never by
weakening). Phase 1 = normal run + `--ignore` each slow path → summary +
**its exit code is the script's exit code**. Phase 2 = same testmon mode over
exactly those paths (partition by construction), `nohup` background after
phase 1, appends `=== slow phase ===` + summary + `EXIT=` marker to
`.pytest_cache/slow-phase.log` (in-checkout, not /tmp). `RUN_ALL=1` gets the
same split (noselect both phases; union = full set). Next invocation warns on
a stale `EXIT!=0` slow-phase status (no silent reds). Nothing deselected, no
marker edits, no `tests/core-py/test_results/*.json`, CI unchanged.

**Known-slow facts from research (2026-11-12):**
- `mark.slow` (60 tests) already excluded by `pytest.ini` addopts
  `-m "not slow"` in BOTH modes — not this card's problem.
- Real tail = unmarked stragglers: `tests/core-py/test_vm_life_program.py`
  (23 tests, `pytestmark = [no_cover, timeout(600)]`, 3M-step emulation),
  `tests/test_ui_journey_*`, `tests/test_journeys_vertical.py`,
  `tests/test_comprehensive_journeys.py` (all unmarked).
- CI calls pytest directly — run_affected.sh is local-only (card 20261104 =
  PR-CI affected closure, related, do not duplicate).
- No existing tests for run_affected.sh.

## Steps

1. TDD `scripts/seed_testmon.py` (RED→GREEN: missing source → fallback,
   existing dest untouched, corrupt source → fallback, backup parity).
2. Wire Part A into run_affected.sh + fresh-worktree proof (no DB → 9-file
   diff selects <1k, fast phase ≤5min).
3. `scripts/slow_paths.txt` + phase split (RED with stub → GREEN).
4. Docs: Test Rule §6 amendment.
5. Gates: seed tests, script twice back-to-back, ruff/shellcheck,
   change-scoped closure. Pathspec commit + AGENT_SYNC + push.

## Status log

- 2026-11-12: card picked as next after landings (user answer); design
  presented, "continue building" = go with empirical slow-list seed +
  agreed scope (`scripts/seed_testmon.py`, `scripts/slow_paths.txt`,
  `tests/test_run_affected_seed.py`). Code NOT started — landings
  (Phase 2 → card 1 → card 2) run first per user's sequence answer.
