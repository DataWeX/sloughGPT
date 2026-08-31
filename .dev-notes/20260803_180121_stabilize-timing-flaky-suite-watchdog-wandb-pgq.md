---
id: 20260803_180121_stabilize-timing-flaky-suite-watchdog-wandb-pgq
title: stabilize-timing-flaky-suite-watchdog-wandb-pgq
status: done
tags: tests,infrastructure
created: 2026-08-03T18:01:21.996544+00:00
---

stabilize-timing-flaky-suite-watchdog-wandb-pgq

Default pytest suite now green: 102 failed -> 8 (all thread-timing) -> 3 (watchdog only) -> 0 failed (9946 passed, 29 skipped, 448 deselected, 476s). Root cause of remaining flakes: tests/test_watchdog.py fast_loop fixture patched time.sleep to a no-op lambda, turning the watchdog inner loop (domains/infrastructure/watchdog.py:110) into a GIL-hogging busy loop that starved watchdog threads under full-suite CPU load. Fix: fixture now uses real_sleep(0) to yield the GIL each iteration (keeps fast spin); _wait_until uses a module-level threading.Event().wait(0.002) unaffected by the sleep patch; test_single_failure_does_not_recover made deterministic (waits for calls>=3 so healthy reset of _consecutive_failures is guaranteed). test_wandb_server.py: replaced fixed 0.08s sleeps with _wait_until_logged() polling (interval 0.01s). test_pugqeep_cache_tasks.py::test_submit_training: polls for pgq.library.has('sys.w') with timeout (was an immediate assert on background-thread completion). Verified: 174/174 in isolation, 10/10 runs clean under 4x CPU load, full suite green.