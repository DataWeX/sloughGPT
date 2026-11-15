---
id: 20261106_115359_fixci-test-core-py-red-on-every-cicd-run-test-infra-thread-t
status: done
board: 20261106_115359_fixci-test-core-py-red-on-every-cicd-run-test-infra-thread-t
title: "fix(ci): test-core-py red on every CI/CD run — test_infra_thread_teardown idle-loop 2 failing since before 2026-11-06 route-map lane"
tags: ["ci", "pre-existing", "core-py"]
sprint: ""
gh: null
---

# fix(ci): test-core-py red on every CI/CD run — test_infra_thread_teardown idle-loop 2 failing since before 2026-11-06 route-map lane

CI/CD test-core-py job fails on all recent main runs: 42196160504 (3e328feaa, route-map card 1), 42188981939 (c92e66fc2, started BEFORE that push), 42208307653 (23372e147 pugqeep merge), 42207996859, 42203826033 — identical 2 failures every time: tests/core-py/test_infra_thread_teardown.py::TestIdleReset::test_reset_stops_idle_loop_promptly (assert False) and ::TestOrphanInstances::test_stop_all_idle_managers_kills_orphan_instances (assert []). Pre-existing, unrelated to any lane's diff (reproduced on SHAs with zero overlap). Blocks every CI/CD verdict on main (deploy skipped). Fix: diagnose idle-manager teardown timing in CI (likely slower runner vs local); do not weaken assertions.

IN FLIGHT (noticing 2026-11-07): PR #294 `fix/infra-thread-teardown-detect` is green (16/16) + MERGEABLE and fixes exactly this — root cause is NOT timing: f2f91016a migrated the idle loop to spawn_loop() (engine Process), the test still scans threading.enumerate() for name 'idle-manager' which returns [] always. Do NOT pick this card; PR #294's session owns it.
