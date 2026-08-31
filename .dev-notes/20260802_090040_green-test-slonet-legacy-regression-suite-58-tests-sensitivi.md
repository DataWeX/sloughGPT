---
id: 20260802_090040_green-test-slonet-legacy-regression-suite-58-tests-sensitivi
title: green test_slonet_legacy regression suite (58 tests) + sensitivity seed
status: done
tags: slonet,legacy,tests
created: 2026-08-02T09:00:40.268595+00:00
---

green test_slonet_legacy regression suite (58 tests) + sensitivity seed

Concurrent editor added test_slonet_legacy.py in 5 waves (348->701 lines, 70 tests, all green). Each wave verified + committed:

1. Core legacy surfaces (36 tests): SOU round-trip (v1 format/bad magic/missing/sanitize), flat-param rebuild guard rails, demo trainers, SloDataLoader, silu numpy path, Tensor.to, GQA n_rep>1 backward. 7706006.
2. +TestLossUtils/TestSoftmaxNumpyPath/TestSloNetFit/TestUserAdapters/TestMHAForwardNumpy/TestLSTMNumbaFallback/TestAccelOpDispatch (58 tests). 30a4c12.
3. +TestNormalize/TestMultinomial/TestTensorScatter (64 tests). 550bc4d.
4. +TestSiluAccelerator/TestSoftmaxAccelerator accel dispatch (70 tests).
Plus seed fix in test_slonet_compute_sensitivity (9/9, file in pytest.ini ignore list).

Key findings:
- All 70 tests pass with NO source fixes -- the legacy suite validates previously-fixed slonet surfaces (import_from_sou, _rebuild_net_from_params, EMA, _accel_op, conv/batchnorm grads).
- Transient 3-test failure (TestMHAForwardNumpy, ValueError slonet.py:2128) during combined sweep was concurrent-editor interference: its coverage-instrumented pytest (left slonet.py,cover artifact, deleted) + mid-write file caused order-dependent state. After quiescence: 3x stable, 10-file sweep 427 passed.
- Full core-py suite is ~6000 tests (~15+ min); not run to completion -- targeted sweeps cover the changed surface.

Cleanup: removed slonet.py,cover artifact; pycache cleaned each run. Tree clean, journal + kanban synced.