---
id: 20261112_064224_work-history-recordhistorycompare-score-card-close-outcomes
status: done
board: 20261112_064224_work-history-recordhistorycompare-score-card-close-outcomes
title: "work history: record/history/compare — score card-close outcomes against previous windows"
tags: ["tooling", "metrics", "agent-workflow"]
sprint: ""
gh: null
---

# work history: record/history/compare — score card-close outcomes against previous windows

User-approved 2026-11-12 ("build it minimal and focused"). Derived from EXISTING artifacts, no new telemetry: scripts/work_history.py mirroring benchmark_results.py record/history/compare but minimal — source = ~/.config/kanban/board.jsonl done cards (CLOSED evidence text, created_at/updated_at, tags); metrics = closed count, evidence coverage, zero-attributable triage mentions, time-in-wip, queue size (todo/wip), reopened count. Snapshot JSONL; compare = simple deltas vs previous window. NO git source, NO sigma-band, NO notification surfaces (never-touch), scoring OUTCOMES not raw reds. TDD parity test: fixture board determinism + parser + compare deltas. Scope ruling from discussion: Gen dispatch/brief quality = substrate episodes (future card if wanted), not this script.

CLOSED 2026-11-12 (OPENCODE work-history lane) — pushed branch `feat/work-history` (db1ba93f4 impl 2 files, 8daf78f2f AGENT_SYNC; remote tip 8daf78f2f, 0 ahead; lane-only, NOT merged). User ruling: minimal + focused. scripts/work_history.py record/history/compare (156 lines) + tests/test_work_history.py (82 lines) — source = board.jsonl ONLY (done cards CLOSED evidence + created/updated stamps + queue columns), zero new telemetry, storage mirrors benchmark_results (stamped JSON in ~/.local/share/sloughgpt/work_history, SLO_WORK_HISTORY_DIR override, microsecond stamps, sort_keys). Metrics score OUTCOMES: closed{total, with_evidence, zero_attributable} (three-strike verdict — raw reds deliberately unscored), queue{todo,wip,review}, avg_close_latency_s, reopened. TDD: RED ModuleNotFoundError watched -> GREEN 5/5 (fixture counts, record determinism, history listing, compare deltas, compare-needs-2); two own bugs caught en route (30h-vs-36h spec arithmetic, multi-line fixture that was not jsonl). First real snapshot: closed=131 evidence=14 todo=49 wip=5 (evidence coverage 14/131 = most historical closes predate evidence notes). GATES: ruff check+format clean; change-scoped closure (run_affected + 3 established --ignores): 7 failed / 28 passed / 15 skipped / 1324 deselected in 10:31 — the identical 7 reds triaged earlier today with isolated/pristine proofs (4 ledger env+pollution, vm_life load-flake green isolated 7.09s, datasets ImportUrl x2 pristine-proven carded 20261112_062928), zero attributable; the 5 new tests green in-selection. Benchmark not owed (tooling).
