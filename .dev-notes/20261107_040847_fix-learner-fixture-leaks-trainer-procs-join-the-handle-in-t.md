---
id: 20261107_040847_fix-learner-fixture-leaks-trainer-procs-join-the-handle-in-t
status: done
board: 20261107_040847_fix-learner-fixture-leaks-trainer-procs-join-the-handle-in-t
title: "fix: learner fixture leaks trainer procs — join the handle in teardown (test_shutdown_joins_live_thread flake)"
tags: ["testing", "infra", "ci"]
sprint: ""
gh: null
---

# fix: learner fixture leaks trainer procs — join the handle in teardown (test_shutdown_joins_live_thread flake)

learner fixture sets _running=False but never joins inst._thread -> every one of ~40 tests in test_learner_continual.py leaks a pending continual-trainer engine proc. Later test_shutdown_joins_live_thread queues behind them, join(timeout=5) x2 times out, ThreadHandle.is_alive() (=not is_done) stays True -> assert not True. Pre-existing flake: fails on #263 (no learner changes), ~30% CI / ~5% local. Fix: fixture teardown calls inst.shutdown(); de-duplicate the finally shutdown.
