---
id: 20261004_test_suite_resilience_timeout_watchdog
title: "Test suite resilience: default per-test timeout + reboot/chunked runner + fork-deadlock babysitter (4 power-key reboots killed 4 runs tonight)"
status: review
date: 2026-10-04
tags: [tests, infra]
board_id: ca167dbd-913f-48ff-8ad0-c85e24ccb3c1
board: ca167dbd-913f-48ff-8ad0-c85e24ccb3c1
hash: 54f52750de4b1961f74decfecd3fec713fe02e9fbd02534f7402e5ebcce23ebe
---

Evidence: verification sweep 2026-10-04 on worktree /home/mana/Documents/sloughgpt-api-std @ 5bfbb8e08 (full suite 619F/45211P/180E in 1h56m; logs in ~/.cache/sweep-logs/: backend5.log full run, detail.log+detail2b.log exception detail, full-summary.txt, failed-by-file.txt).

Evidence: full suite needs ~2h but machine was power-keyed off 4× tonight (22:52, 23:12, 00:42, 01:42) killing runs at 82%/25%/60%/49%; only the 5th attempt (uninterrupted window) finished. `packages/core-py/tests/test_vector_backend.py` fork-deadlocks (parent 18 threads + multiprocessing fork → children idle forever, parent in do_wait; hung 14+ min until manually killed) — pytest.ini sets NO timeout; other sessions' runs all pass `--timeout=120` for exactly this reason. Working prototypes built during the sweep: ~/.cache/sweep-logs/watchdog.sh (relaunches remaining files if run dies without summary) + babysitter.sh (kills idle deadlocked children ≥90s with cpu<10s).

FIX: (a) set `timeout = 120` + `timeout_method = signal` in pytest.ini (or mark fork tests explicitly); (b) promote watchdog/babysitter into scripts/test_resilient.sh (chunked runner: split file list, resume from last completed file, skip-if-summary-present); (c) fix test_vector_backend's fork pattern (use spawn or join with timeout).

ACCEPTANCE: killing pytest mid-run then rerunning completes the remainder automatically; vector_backend completes <2min without babysitter; CI keeps current totals.

COORDINATION: other sessions already run with --timeout=120 + --ignore flags (seen in ps) — agree the ini default so flags stop diverging.
