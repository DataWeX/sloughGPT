---
id: 20261005_avion_converge_main
title: "avion: converge packages/avion to main (3547 lines stale, 8 test files + 4 src modules missing)"
status: done
date: 2026-10-05
tags: [avion, sync, drift, testing]
---

USER DIRECTIVE 2026-10-05: "continue building" → chose **Full converge,
preserve real fixes**.

ROOT CAUSE (measured): this branch's `packages/avion` is a stale fork of
main's. `git diff --shortstat main HEAD -- packages/avion` = **35 files,
+203 / -3547**. Missing entirely (12 files, ~2446 lines):
- src: `events/journal.py` (180), `events/logger.py` (152),
  `events/sinks.py` (107), `sync.py` (90)
- tests: `test_agent_loop` (464), `test_cdp` (414), `test_event_logger`
  (379), `test_journal` (138), `test_naming` (166), `test_page_controls`
  (174), `test_stdlib_only` (105), `test_sync` (77)
- `avion/__init__.py` lacks main's `PageControls`, `SyncRunner`,
  `get_default_runner`, `from __future__ import annotations`
- `pyproject.toml` lacks `cdp = ["websockets>=12"]`

This is why the same bug class surfaced three times in one session
(cdp.py/cli.py/__main__.py missing → test_tools.py without impl → rename
guard missing): symptoms of one stale copy, found one failing test at a
time. Card 122109a9 is already **closed on main** (2026-10-01); only this
branch's 339-behind board copy still reads `todo`.

PLAN: (1) snapshot the 203 HEAD-only lines to
/tmp/avion-head-only-reference.diff; (2) `git checkout main -- packages/avion`;
(3) re-apply only genuinely branch-owned work; (4) gates.

PRESERVE LIST (from the reference diff, classified):
- `src/avion/ai/slo.py` — `domains.` → `domain.` import fix; a prior note
  called this an intentional local change. MUST survive.
- `arken_*` string literals — main keeps these too, so checkout is
  no-op on them (session-name default, logger name, ctx key, output dir).
- everything else in the 203 is a re-implementation of older main code
  (hand-ported TOOL_CALL, my session.py structured-logger fix) — main
  already has better versions; dropping them is the point (user requires
  matching main to avoid future merge conflicts).

GATES: ruff, py_compile, full avion suite (baseline 233 passed / 1
skipped), tests/test_shared_indexes, arken shim import, live CDP CLI
(open/snapshot/click/wait/console on :9222), PlaywrightBackend E2E.

SCOPE: stage ONLY `packages/avion`. Other session's dirty files
(domain/soul, domain/inference, domain/training, domain/models, .kanban,
scripts/benchmark_soul_paths.py, tests/test_soul_path.py) untouched.

RESULT 2026-10-05 — committed 074475d26.

Converged `packages/avion` + `scripts/run_journey_tests.py` to main; both
are byte-identical to main at commit time (verified with
`git diff --stat main -- <path>` = empty).

**Preserve list came out EMPTY** — all 203 branch-only lines classified
against /tmp/avion-head-only-reference.diff, none re-applied:
- `ai/slo.py` `domains.` was NOT a branch fix; main's `domain.` came from
  `5318fff2c refactor(dual-tree): retarget domains.* → domain.* where
  singular is canonical`. Branch was pre-refactor. (Both resolve at
  runtime; no avion test exercises `from_soul`.)
- `agent.py` (+53), `skills.py` (+13), `cli.py` (+9) were older shapes —
  main verified to have save_trajectories, transcript_path, deadline,
  on_step, find() ranking, all 4 `--by` + XPATH.
- `session/recorder/structured` — main has newer `logger=self._event_logger`
  wiring.
- `SOTA_COMPARISON.md` (+11) — stale Voyager prose ("We have none of these
  three", "201 passing, 14 skipped").
- `arken_*` literals — survived untouched; main keeps them.

Second-order find: the converge's new repo-wide guard (`test_naming.py`)
failed on `scripts/run_journey_tests.py` — still had `from arken import
Arken` (L37), `from arken.backends.api import ApiBackend` (L76),
`sys.path.insert(... "arken"/"src")` (L32). Pure 3-line rename, converged
to main. Confirmed by repo-wide scan that these were the ONLY strays
outside `packages/voyager`.

GATES (all green): ruff clean; py_compile clean; **avion suite 368
passed / 1 skipped / 0 failed** (baseline before converge: 233 passed /
1 skipped → +135 tests); tests/test_shared_indexes 6 passed;
voyager→arken→avion resolves to one class; `python -m avion` OK; live CDP
CLI open/snapshot/click/wait/console/network on :9222; Playwright E2E
click → 'clicked'; session start/stop through the new EventLogger
wiring. Re-ran suite + live gates AFTER lint-staged mutated the files.

FOLLOW-UPS (new, not in this card):
1. **ruff format width mismatch = churn generator.** Repo config is
   `line-length = 100`, but main's avion files were formatted at ruff's
   default 88. lint-staged reformats them on every commit (12 files,
   +128/-146 this commit — pure line-wrapping, proven by suite). So
   "byte-identical to main" is unachievable for long lines until main is
   normalized once with `ruff format`. Content/logic DOES match.
2. **Stale board cards.** `122109a9` reads `todo` on this branch's board
   but is closed on main (2026-10-01) — this branch is 339 behind, so its
   board copy is stale. 3 avion cards sit in `review` already merged to
   main (`24676ff2` event logger, `553da7a7` agent loop, `e26dc68c`
   unify drivers) — those are exactly the 12 files this branch was
   missing.

