---
id: 20260803_010840_wave-f-slonet-kernels-numpy-fallback-coverage
title: wave F: slonet_kernels numpy-fallback coverage
status: done
tags: slonet,kernels,coverage,tests
created: 2026-08-03T01:08:40.706516+00:00
---

wave F: slonet_kernels numpy-fallback coverage

Wave F complete. test_slonet_kernels.py: 50 tests (30 fallback + 20 kernel-body/ensure-branch), all green. Fake-numba injection (identity njit) executes the real @njit algorithms as pure Python and verifies them against numpy refs; module globals snapshotted/restored, no cross-module leakage. Coverage (combined sweep 11 files + meta_path-blocker probe): slonet_kernels.py 99% (0 missed stmts, 1 partial = warmup softmax fixed-ascending-data branch); slonet.py stays 97% (line 107 provably dead). Sweep 222 tests green. Committed as ef1d6bb.