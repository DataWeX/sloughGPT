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
  `tests/test_run_affected_seed.py`, phases test). Code started in
  worktree `fe08f8/run-affected` (branch `test/run-affected` @ `2a818d30f`)
  while run 4 (Phase 2's gate) executes — landings still run first for
  main.
- TDD: seed helper RED (`ModuleNotFoundError`) → GREEN **11/11**; phases
  RED against the old script (5 failed / 1 passed — the passthrough pin)
  → GREEN **17/17**. ruff check+format clean on all 3 files; `sh -n`
  clean; AGENTS.md Test Rule clauses 3 + 6 amended (seed exception +
  fast/slow bullet).
- Design correction found during research: primary worktree's `.testmondata`
  is 9 days stale (16MB Nov 3) while session worktrees hold the live 52MB
  one → seed source = **freshest across worktrees**, not primary
  (TESTMON_SEED_DB overrides). Primary path contains a space
  (`Default Project/sloughGPT`) → `worktree ` prefix-strip, never
  whitespace-split (pinned by test).
- Collision note: another session checked out `feat/run-affected-phases` in
  `Default Project/sloughGPT/.wt-test-runs` at 08:17 (from its
  `feat/work-history` lane) — tree clean, no own commits, run_affected.sh
  untouched. Flagged to user; my work is on a separate branch/worktree.
- Pending gates: fresh-worktree proof (real run: seed message + selection
  <1k + fast phase ≤5min), partition collect-only evidence (union of the
  two phases == plain run), change-scoped closure.
