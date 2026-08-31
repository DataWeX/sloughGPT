---
id: 20260806_143527_session-core-py-test-sweep-datasetspage-flake-fix
title: Session: core-py test sweep + DatasetsPage flake fix
status: done
tags: testing,maintenance
created: 2026-08-06T14:35:27.580361+00:00
---

Session: core-py test sweep + DatasetsPage flake fix

FULL SWEEP GREEN. Complete: 284/284 files PASS, 0 failures (rc 0/5 = PASS), per-file timeout 900s.

Latest fixes (Aug 7, post-277-file update):
7. repl.py fd-leak (production fix): per-instance RotatingFileHandler (~/.config/sloughgpt/shell_infra.log) + LogBufferHandler on shared 'slo' logger leaked 1 fd + duplicated writes per ShellREPL. Replaced with module-level singletons _get_file_handler()/_get_log_buffer_handler() with closed-attr detection (test fixture closes them each test). fd delta stable +2 across 10 instantiations. test_shell_repl.py 275 passed, test_shell_repl_more.py 2722 passed/4 skipped.
8. test_model_size.py FAIL(124): write_weight_file used write_bytes(b'\0'*size) up to 1.5GB in RAM -> OOM/hang. Now open('wb')+f.truncate(size) (sparse, O(1), same st_size). 28 passed 0.3s 3x stable.
9. test_point_library.py FAIL(124): genuinely slow (217s NumPyEngine shared-library test + real Qwen weights). Fixed via sweep timeout 600->900s. 72/72 pass.
10. test_multimodal_v2.py FAIL(1): transient (mid-edit race with concurrent editor), 50/50 isolation.
11. VM wait syscalls (production behavior change Aug 7 03:56): _sys_wait now POSIX ECHILD - returns -1 when current is None OR no children (never blocks a childless process), blocks only when live children exist w/ none terminated. Two stale tests updated: test_wait_no_children asserts 0xFFFFFFFF; test_wait_blocks_when_no_terminated_child registers a real live child (blocked -> 0, current None). test_vm.py + test_vm_syscalls_shell.py 707 passed.

INFRA: sweep at ~/.cache/sloughGPT-sweep/ (resume.sh, core-sweep.log, resume.out). Detached sweeps get reaped by env - run foreground chunks. Concurrent editor rewrites web tests (page.test.tsx -> *Page.test.tsx) and runs its own pytest - re-read files before/after edits.