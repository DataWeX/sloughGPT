---
id: 20261112_122610_testmon-seed-env-guard-environment-fingerprint-match-rows0-w
status: done
board: 20261112_122610_testmon-seed-env-guard-environment-fingerprint-match-rows0-w
title: "testmon seed env-guard: environment-fingerprint match + rows>0 + wiped-DB reseed — never claim a penalty skip the copy cannot deliver (follow-up to 20261111_125558)"
tags: ["testing", "tooling"]
sprint: ""
gh: null
---

# testmon seed env-guard: environment-fingerprint match + rows>0 + wiped-DB reseed — never claim a penalty skip the copy cannot deliver (follow-up to 20261111_125558)

FOLLOW-UP to 20261111_125558 (closed: run_affected auto-seed + two-phase split, 567335989 — freshness-only seed source).

DEFECT CLASS (proven on the parent card, live): a seed copy whose environment differs from the current interpreter is a SILENT NO-OP with a FALSE success message. Root cause chain: testmon fetch_or_create_environment REPLACES the environment row when system_packages or python_version drift + test_execution.environment_id is ON DELETE CASCADE (PRAGMA foreign_keys=TRUE at db.py:33) -> the copied rows are purged on the destination's FIRST run -> packages_change forces a full run anyway. The runner then prints 'fresh-worktree penalty skipped' — a lie. Measured incident 2026-11-12 - RETRACTED as a measurement artifact: the '0 of 114 match' reading came from comparing RAW package versions against testmon's patch-dropped storage (drop_patch_version, testmon_core.py:212 <- common.py:90 - 'PyJWT 2.14', never 'PyJWT 2.14.0') AND from scanning with PYTHONNOUSERSITE=1 (hides ~75 user-site dists); 0 METADATA files changed that day, and under testmon's exact predicate the recorded fingerprints match the live env (the lone downcraft-0.3 delta = gitignored egg-info presence, per-worktree). The defect class stands regardless: any real drift (package change, env upgrade) makes a freshness-seed copy doomed rows that the cascade purges on the first run.

PLAN: (1) seed_testmon.find_seed_db discovery filter = valid testmon schema AND rows(test_execution)>0 AND environment fingerprint (system_packages + python_version, latest row) == current interpreter (testmon.common.get_system_packages + platform.python_version); freshest MATCH wins; when nothing matches -> None (honest full record, Test Rule clause 3). Unknown fingerprint on either side (no column / no rows / testmon not importable) -> freshness fallback, never a hard fail. TESTMON_SEED_DB explicit override keeps its current semantics (explicit user intent wins; warn on mismatch). (2) Wiped-DB recovery: dest that exists, is a valid testmon DB, but has 0 test_execution rows is RE-SEEDABLE (today exists() short-circuits; .wt-test-runs hit exactly this after the cascade wipe) — health check lives in the run_affected.sh heredoc (dest_state), seed_testmon_db's never-overwrite contract untouched for non-empty/invalid dests. (3) AGENTS.md Test Rule line: 'freshest' -> 'freshest environment-matching recording', plus one clause documenting the cascade invalidation (after a package change every recorded history is invalid — one RUN_ALL rebuilds).

OWNED: scripts/seed_testmon.py, scripts/run_affected.sh (heredoc only), tests/test_seed_env_guard.py (new), tests/test_run_affected_seed.py (fixture realism: _make_db gains a test_execution row — assertions unchanged), AGENTS.md Test Rule line.
GATES: RED-first (freshness-picks-mismatch and empty-DB-picked must fail first); parent suites tests/test_run_affected_seed.py + tests/test_run_affected_phases.py stay green; fresh-worktree proof — scratch worktree with NO .testmondata seeds from an environment-MATCH source and select-only count <1k for a small diff; wiped-DB proof — .wt-test-runs 0-row DB re-seeds on next run_affected; ruff + sh -n.
MUST NOT: weaken selection (rewriting the copied environment row to the current fingerprint would mark stale rows as fresh — forbidden), overwrite any non-empty or invalid dest, touch pytest.ini markers, edit tests/core-py/test_results/*.json.

RESULTS (done 2026-11-12, branch feat/seed-env-guard, pushed: 823d323b3 impl + 6d84108bf sync):
- Two refinements beyond the plan: (a) the fingerprint is compared against the env rows the recorded tests POINT AT (JOIN test_execution.environment_id - the cascade deletes exactly those, newest-row-alone could be another generation); (b) dest_state health runs on EVERY selective invocation, and an empty dest is unlinked including stale -wal/-shm before re-seeding.
- TDD: RED 12 watched first (env-match beats fresher mismatch, freshest-match, no-match->None, unknown fallback, empty-DB skipped x2, override still copies, fingerprint-format pin, dest_state x4) -> GREEN 32/32 (12 new + 11 seed + 9 phase unchanged); _make_db fixture gained a test_execution row (assertions unchanged); ruff check+format clean, sh -n clean; post-rebase re-run 32/32.
- PROOFS (both live): wiped-DB - .wt-test-runs 0-row DB -> both messages fired -> 49,513 rows, 18 changed files, 250 selected; fresh worktree (/tmp/opencode/scratch-wt2 at the tip, no DB) -> env-match seed -> 49,513 rows, 250 selected (<1k gate).
- AGENTS.md Test Rule 158/162 amended: 'freshest' -> 'freshest environment-matching recording' + cascade-invalidation clause (package change invalidates every recording - one RUN_ALL=1 rebuilds).
- MUST NOT held: no environment-row rewrite, no non-empty/invalid dest overwritten, pytest.ini markers untouched, tests/core-py/test_results/*.json untouched. Supersedes unpushed feat/run-affected-phases (7d2039555/807456e4a) - branch deleted after push; 567335989 remains the base implementation.
