---
id: 20261105_155940_pugqeep-engine-run-loop-iterates-processes-bare-dict-changed
status: done
board: 20261105_155940_pugqeep-engine-run-loop-iterates-processes-bare-dict-changed
title: "pugqeep Engine run-loop iterates _processes bare — dict-changed-size race kills engine thread (test-core-py CI flake)"
tags: ["pugqeep", "race", "ci-flake"]
sprint: ""
gh: null
---

# pugqeep Engine run-loop iterates _processes bare — dict-changed-size race kills engine thread (test-core-py CI flake)

CI evidence (PR #267 run 42036453877, job 127275760867): tests/core-py/test_learner_continual.py::test_shutdown_joins_live_thread fails with the engine-shared thread dying at domain/infrastructure/_internal/pugqeep/engine.py:1669 RuntimeError: dictionary changed size during iteration (all(p.is_done for p in self._processes.values()) inside Engine.run, OUTSIDE the try/except). Engine.run's loop then exits, wait_for never satisfies, ThreadHandle.is_alive() stays True, join(timeout=5) times out, assertion fails. Reproduced locally on pristine origin/main code 1/5 runs (byte-identical domain/+tests/, my PR touches only apps/web). Root race: spawn writes self._processes BARE (lines 1465, 1470, 1935) while the run loop iterates bare at 1656/1661/1669 - the same loop already snapshots correctly at 1632 (list(self._processes.values())) so the idiom exists, the three sites just don't use it. Fix shape: snapshot iterations (GIL-atomic list()) at bare iteration sites, or take self._lock around writes+iterations consistently; also guard on_progress sums (1656/1661, only reached when on_progress is not None). Never re-decide: Engine._lock exists (1395) but covers trees, not _processes.

LANDED: PR #269 -> merge 23372e147 (2026-11-06T11:10:52Z). Fix: 15 bare _processes iteration sites wrapped in list() (13198a8b3); rehash caf5b6d15 (domain-baseline 3fc9e407 -> 68b535cb); docs f71b15eb9. CI evidence on f71b15eb9: test-core-py 2 failed/42820 passed — both inherited test_infra_thread_teardown (byte-identical to main CI/CD run 42177637251), 0 dictionary-changed-size crashes; test-root PASS; E2E PASS; all other checks green.
