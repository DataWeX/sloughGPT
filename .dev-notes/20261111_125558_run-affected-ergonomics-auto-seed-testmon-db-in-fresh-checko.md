---
id: 20261111_125558_run-affected-ergonomics-auto-seed-testmon-db-in-fresh-checko
status: done
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
- 2026-11-12: ALL GATES PASS. TDD RED→GREEN 20/20 (9 phase + 11 seed);
  ruff check+format clean, `sh -n` clean (shellcheck not installed on box);
  fresh-worktree proof (no DB → seeded 52MB, 7 selected <1k, fast phase 21s
  ≤5min, background EXIT=0); back-to-back ×2 (seed skip when DB exists,
  honest RC passthrough); union proof tests/core-py collect-only: plain
  42705 = phase1 42484 ∪ phase2 221, 0 strays / 0 dropped / p1∩p2=0;
  slow list repaired (4 dead tests/ paths → tests/core-py/) + ratchet test.
  Slow-tail defects found en route (part of the fix): scope positionals
  re-collected in phase 2 (stripped now), explicit-slow-arg double-run
  (excluded from both sides). Pre-existing, NOT this card: phoneme-router
  collection error on default scope (kill 63e682360; CI --ignores
  tests/server/) + 4 documented env/lane reds — testmon file-reads prove
  none of the 7 read any changed card file. Code: `ca31b32c8` (pre-rebase).
- 2026-11-12: LANDED ON MAIN (card complete). User ruling "all done once" =
  also fix the phoneme orphan + merge. Rebased onto origin/main twice (it
  moved mid-flight); final main history: feat `567335989`, sync entry
  `a3c015654`, phoneme-orphan delete `a03957e3f`, slow-path +user_journeys
  `945101e0e` (external 11:25 edit kept verbatim), citation fix-forward
  `96621b339` (rebase edit-fold stopped one commit late — entry cited its
  own docs commit; pushed main never rewritten). Pushed via
  `git push origin HEAD:main` (local main checked out by quiet-pixel lane,
  Core-Rules-sanctioned lane push); lane `test/run-affected` force-synced
  to match. Gates post-rebase: 28 passed (capability_surface +
  docs_api_parity + the 20), sh -n, ruff clean; tests/server collects
  clean (2134/2204, 0 errors). CI on main: `CI` workflow success 5m32s;
  `CI/CD` full-suite backstop in progress at close. AGENT_SYNC entry cites
  `567335989`.
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
- 2026-11-12 (landed): concurrent session made on-card scope improvements in
  this worktree at 10:31-10:36 (scope-aware deferral, explicit slow args run
  once, phase-2 scope stripping + 3 contract tests — all adopted, verified
  green); user ruling "both keep building, whoever lands first wins" →
  their push at 11:31:34 won the race. Landed on main as `567335989`
  (+ sync `a3c015654`, phoneme-orphan deletion `a03957e3f`, this session's
  `test_user_journeys` slow-path entry `945101e0e`, sync sha-fix
  `96621b339`). Pre-push gates (all green): 20/20 tests, ruff check+format,
  `sh -n`, domain baseline 8/8, board chain 0, feature tags OK; partition
  proof exact at 48,685 nodeids (A = B∪C, B∩C = ∅, both directions 0 diff,
  8-entry list); fresh-worktree proof (seed message, 420 selected vs 48k,
  fast phase 296s, slow phase EXIT=0, no dead entries). CI quick
  `43201134448` = success; CI/CD `43201134343` running at time of writing.
- Run-4 triage (Phase 2's gate): 13 unique failures, all classified —
  5 ledger (feature_flags ×2 order-dependent, infra_thread card
  `2026106_115359`, ml_types cuda + slonet numba env), 3 shared-stack
  stale-dock (comprehensive_training_journeys ×2 + e2e_training_trigger —
  byte-identical `training interrupted` NotificationDock overlay from zombie
  jobs in the shared :8000; documented + user-ruled in journey-env-gate
  card `20261104_085623`; e2e also red on pristine main), 5 contention
  flakes passing isolated (config env-override, learner background loop,
  export presets ×2, user_journeys training load).
