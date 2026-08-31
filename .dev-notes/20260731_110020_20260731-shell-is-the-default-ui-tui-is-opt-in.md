---
id: 20260731_110020_20260731-shell-is-the-default-ui-tui-is-opt-in
title: 20260731 Shell is the default UI; TUI is opt-in
status: done
tags: shell,cli,tui
created: 2026-07-31T11:00:20.620759+00:00
---

20260731 Shell is the default UI; TUI is opt-in

Flipped ShellREPL so line mode is the default interactive shell; the curses TUI is now opt-in. repl.py __init__: use_tui=None resolves to MAN_TUI==1 (was auto-detect via isatty + MAN_NO_TUI). CLI: sloughgpt tui forces the TUI (invokes shell with tui=True); sloughgpt shell gained --tui flag. Docs updated: SHELL.md TUI section (launch via sloughgpt tui / shell --tui / MAN_TUI=1), __init__.py, ROADMAP CLI bullet. 4 new unit tests for default/opt-in selection; shell/logging gate 569 passed, 12 skipped. py_compile + CLI --help + shell -c echo verified. PTY probe shell_pty_line.py: boots line mode (no [OUTPUT] LIVE banner, no alt-screen escape), readline Alt+D kill-word + Ctrl+U kill-line verified, echo output correct — all steps OK; stray API server killed.