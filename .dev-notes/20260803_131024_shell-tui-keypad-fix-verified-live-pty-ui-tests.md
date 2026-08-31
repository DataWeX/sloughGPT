---
id: 20260803_131024_shell-tui-keypad-fix-verified-live-pty-ui-tests
title: Shell TUI: keypad fix verified + live pty UI tests
status: done
tags: shell,tui,testing,pty
created: 2026-08-03T13:10:24.185721+00:00
---

Shell TUI: keypad fix verified + live pty UI tests

Shell TUI resize gap fixed + full live pty suite green.

Root cause (resize): the shell imports 'readline' (line-mode history) before curses boots; CPython's readline module claims SIGWINCH (rl_catch_sigwinch), so ncurses never installs its own handler and getch() never returns KEY_RESIZE — the TUI kept rendering at the stale 80x24 geometry forever. Isolated via pty bisect: KEY_RESIZE handled=True for plain/stderr/ioctl/numpy/rich/logging-cli, False for 'import readline' (and runtime/repl).

Fix (tui_repl.py): keep the KEY_RESIZE branch, but also poll the kernel window size each 100ms poll tick (os.get_terminal_size). On change: curses.resizeterm + shared _resize() closure (nonlocal regions/windows) that recomputes pane regions, resizes surfaces, recreates windows and redraws. Without readline it is a no-op because KEY_RESIZE still arrives normally.

Result: 444 shell tests pass (32.5s) incl. previously-FAILING test_resize_remaps_layout (status '100x30', input row 29, responsive) and test_ctrl_c_interrupts_running_command. Also refactored the old inline KEY_RESIZE body into _resize/_detect_resize.