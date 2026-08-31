---
id: 20260803_164727_tui-live-tests-ctrlc-interrupt-path-coverage-to-95
title: TUI live tests: Ctrl+C interrupt path + coverage to 95%
status: done
tags: shell,tui,coverage,testing
created: 2026-08-03T16:47:27.393768+00:00
---

TUI live tests: Ctrl+C interrupt path + coverage to 95%

## Coverage-gap close for tui_repl.py (curses event loop)

Measured combined coverage via opt-in child-side coverage in the live pty tests, then closed genuinely-untested branches. Final: **tui_repl.py 95% (34 missed of 727 stmts)**.

### New unit tests (tests/test_shell_tui_repl.py, all pass)
- test_complete_path_bare_name_no_slash — bare-name branch (line 89); monkeypatch.chdir(tmp_path), asserts './alpha.txt'.
- test_kill_ring_trims_to_max — _push_kill cap (515-516); 12 x _kill_to_start, ring capped at _KILL_RING_MAX.
- test_tui_io_flush_and_read — TuiIo.flush() no-op + read() raises NotImplementedError (165, 168).

### New live test (tests/test_shell_tui_live.py, all pass)
- test_ctrl_c_interrupts_running_command — Ctrl+C while a command thread is alive (1086-1089). Uses 'py sum(i*i for i in range(10**9))' because the async-exc KeyboardInterrupt only lands at a bytecode boundary; a C-level time.sleep is NOT reliably interruptible (per _interrupt_active docstring). Asserts 'Aborted' appears quickly and prompt stays usable.

### Key findings
- The async-exc injection is best-effort: threads blocked in C syscalls (time.sleep) only return to Python later, so a sleep command is NOT a reliable interrupt target. Pure-Python bytecode (py eval) is.
- A child whose command thread stays alive is SIGKILLed by _TuiSession.close(), which loses its coverage file entirely — the interrupt test must end with the thread dead so the child exits cleanly and saves coverage.
- Root-cause of earlier flaky child coverage (resolved in prior session): coverage.Coverage parent_section default makes new-child files incompatible with a pre-existing combined .coverage in the same dir; always wipe /tmp/opencode/tui_cov before re-measuring.

### Remaining 34 missed lines (all defensive/fallback, intentionally not forced)
- 33-34 ctypes _SET_ASYNC_EXC fallback; 148 alt_map non-printable; 240-241 cli_logger import fallback; curses.error guards 255-256/260/267-268/271-272/310-311/313-314/323-324/326/362-363; 634/636-638 async-exc res==0/res!=1/except race paths; 762-763 set_escdelay; 839-840/843/848-849 terminal-size/resizeterm guards; 859-860 KeyboardInterrupt on getch.

### Verification
- Combined measure: 42 child files combined (1 skipped), tui_repl.py 727 stmts, 34 missed, 95%.
- Regression: 198 passed (test_shell_tui_live 42 + test_shell_tui_repl 105 + test_shell_pane + test_shell_surface).