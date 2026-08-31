---
id: 20260803_100644_wave-v-kernel-family-init-100-coverage
title: test(shell): wave V — kernel_process/interrupts/memory/devices/scheduler/syscall/init to 100%
status: done
tags: shell,coverage,wave-v
created: 2026-08-03T10:07:00.108333+00:00
---

test(shell): wave V — kernel_process/interrupts/memory/devices/scheduler/syscall/init to 100%

Committed ca46030 (6 new test files + test_kernel_coverage extension, ~130 tests, 7 modules / 1158 stmts to 100%):
- kernel_process 99%->100%: release_tensor unknown-id, acquire+release tensor roundtrip
- kernel_interrupts 85%->100%: unregister, mask/unmask, handler-raise False, _max_history truncation, priority deque, process_pending counts, stats, convenience handlers
- kernel_memory 88%->100%: num_elements, util, allocate OOM/dtypes/next-id, free incl pid bookkeeping, free_pid, defragment, zero-cap stats
- kernel_devices 85%->100%: base NotImplementedError, ERROR open False, unregister closes, bad-fd/disconnect, DeviceManager, NullDevice
- kernel_scheduler 48%->100%: queue mapping (NORMAL==2->high, IDLE==4->low), sleeping requeue/expiry, wake WAITING-only, deps, tick semantics
- kernel_syscall 79%->100%: both dispatch arg orders, register forms, stopped-process guard, all builtin handlers
- init.py 96%->100%: 32-test suite; FIXED source bugs: _load_definitions %e->%s format ValueError, missing InitSystem._lock (boot crashed w/o decorator), dead guard in _resolve_deps
Regression: 7-module sweep 1158 stmts 0 missed. Foreign WIP untouched.