---
id: 20261105_160000_test-thread-safety-flake-3-thread-comprehensivetrainerrun-fu
status: wip
board: 20261105_160000_test-thread-safety-flake-3-thread-comprehensivetrainerrun-fu
title: "test_thread_safety flake — 3-thread ComprehensiveTrainer.run_full_cycle returns non-success (test-core-py CI flake)"
tags: [training, test-flake, ci-flake]
sprint: ""
gh: null
---

# test_thread_safety flake — 3-thread ComprehensiveTrainer.run_full_cycle returns non-success (test-core-py CI flake)

CI evidence (PR #267 rerun, run 42036453877 job 127309721013): tests/core-py/test_training_robustness.py::TestConcurrentSafety::test_thread_safety fails at assert all(r.success for r in results) — 3 threads each run ComprehensiveTrainer().run_full_cycle(data, FAST_CONFIG), no thread exceptions (len(errors)==0 and len(results)==3 pass), but one result.success is False. The captured stdout only shows normal phase logs (validating/configuring/preprocessing) — no error surfaced, so the diagnosis needs the result object's error field printed on failure. Different failure from run 1 of the same PR (test_learner_continual shutdown join) — two runs, two different single failures among ~42,790 tests = flaky concurrency suite under CI load, both in code byte-identical to origin/main (PR #267 touches only apps/web). Precedent: the same suite PASSED fully on PR #268's run (42032079341). Suggested fix direction: log/print r.error (or full result repr) in the assertion message first — diagnose which shared resource the concurrent trainers contend on (checkpoint/temp/state path), then fix the contention or the test's isolation.

## Work log

(claimed by foreman session 2026-11-07, dispatched to Gen)
