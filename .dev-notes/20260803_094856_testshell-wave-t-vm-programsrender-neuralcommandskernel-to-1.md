---
id: 20260803_094856_testshell-wave-t-vm-programsrender-neuralcommandskernel-to-1
title: test(shell): wave T — vm_programs/render_neural/commands/kernel to 100%
status: done
tags: shell,coverage,wave-t
created: 2026-08-03T09:48:56.752528+00:00
---

test(shell): wave T — vm_programs/render_neural/commands/kernel to 100%

Committed 0c09953 (4 new test files, ~180 tests, 4 modules +727 stmts to 100%):
- vm_programs.py 98%->100% (128 stmts): X86Assembler programs, self_test, disk/boot/bios image builders; covered padding fallbacks (2729/2760) and oversized-bootloader truncation (3293) via stub assembler
- render_neural.py 93%->100% (139 stmts): channel-stacking edge cases (missing/2D tensors), forward_raw last-inputs/no-input paths
- commands.py ~60%->100% (194 stmts): all ShellCommands wrappers + _api_get/post/delete helpers via fake requests module in sys.modules
- kernel.py 77%->100% (266 stmts): install_addon spec fallback + idempotency, boot addon failure paths, kill/get/close-by-fd, syscall caller inference, interrupt handlers, tick/run termination, hooks, get_kernel/reset_kernel singletons
Regression: 551 passed (11 suites). Foreign WIP untouched.