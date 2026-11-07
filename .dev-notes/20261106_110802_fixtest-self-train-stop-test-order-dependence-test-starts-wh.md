---
id: 20261106_110802_fixtest-self-train-stop-test-order-dependence-test-starts-wh
status: wip
board: 20261106_110802_fixtest-self-train-stop-test-order-dependence-test-starts-wh
title: "fix(test): self_train stop test order-dependence — test_starts_when_not_running leaks a live process"
tags: [test-isolation, pre-existing]
sprint: ""
gh: null
---

# fix(test): self_train stop test order-dependence — test_starts_when_not_running leaks a live process

tests/server/test_self_train_router.py::TestSelfTrainDeterministic::test_stop_not_running asserts status=='not_running' but gets 'stopped' when it runs after TestSelfTrainStart::test_starts_when_not_running (which starts a real self-train pid and never stops it). Fails in any combined run; passes alone. Reproduced on pristine base 3e328feaa (not caused by card 20261101_154058). Fix: stop/cleanup the started process in the first test's teardown (never weaken the assertion).

## Work log

(claimed by foreman session 2026-11-07, dispatched to Gen)
