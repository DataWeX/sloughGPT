---
id: 20260731_060224_regression-test-fixes-parallel-pool-shell-health-init-boot-n
title: Regression test fixes: parallel pool, shell health, init boot, neural kernel
status: done
tags: tests,regression,bugfix
created: 2026-07-31T06:02:24.826643+00:00
---

Regression test fixes: parallel pool, shell health, init boot, neural kernel

Fixed 4 stale tests after regression rerun. (1) test_parallel_execution.py::test_default_size_from_cpu_count now asserts InferencePool._max_workers == get_resource_manager().inference_pool_size instead of min(cpu_count,8) — pool sizing routes through the resource manager (env SLO_INFERENCE_POOL_SIZE). (2) test_shell_repl.py::test_safe_command_not_blocked mocks domains.shell.commands._api_get to return healthy so the health command exits 0 without a live API server. (3) test_init.py::test_boot_runlevel_1 asserts 'boot-critical'/'Boot complete' markers instead of the literal 'Booting'. (4) test_neural_e2e.py::test_boot_creates_neural_kernel asserts plain Kernel + neural addon (NeuralKernel is deprecated). Verified 494 tests pass across all previously-failing modules.