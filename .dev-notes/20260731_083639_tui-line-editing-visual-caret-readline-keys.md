---
id: 20260731_083639_tui-line-editing-visual-caret-readline-keys
title: TUI line editing + visual caret (readline keys)
status: done
tags: shell,tui,feature
created: 2026-07-31T08:36:39.471715+00:00
---

TUI line editing + visual caret (readline keys)

Built readline-style line editing into the split-pane TUI input row (packages/core-py/domains/shell/tui_repl.py):

- _input_view(): pure caret-column computation with horizontal scroll so the caret stays visible on overlong lines; _render_input now parks the curses cursor on the caret via win.move (previously the blinking cursor never tracked the caret).
- Editing helpers: _move_home/_move_end/_kill_to_start/_kill_to_end/_delete_at_cursor/_delete_word_back (unix-word-rubout semantics).
- New bindings in _main: Home/End, Ctrl+A, Ctrl+E, Ctrl+U, Ctrl+K, Ctrl+W, Ctrl+D; KEY_DC routed through _delete_at_cursor.

Verification: 18 new curses-free tests in test_shell_tui_repl.py (31 total in file); 208 shell/logging tests + 298 with test_shell_repl.py all pass; py_compile clean; pycache cleared. PTY probe (tui_pty_edit.py) verified Ctrl+W/U/K/D, Ctrl+D no-op, and Home caret movement end-to-end on a real PTY; no terminal leak.