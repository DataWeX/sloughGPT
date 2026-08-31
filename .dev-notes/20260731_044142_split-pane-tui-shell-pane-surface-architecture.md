---
id: 20260731_044142_split-pane-tui-shell-pane-surface-architecture
title: Split-pane TUI shell (pane + surface architecture)
status: done
tags: shell,tui,architecture
created: 2026-07-31T04:41:42.463360+00:00
---

Split-pane TUI shell (pane + surface architecture)

Built the 3-pane TUI shell per the split-window (NixOS) model: pane.py = pure layout engine (arranger, no curses), surface.py = content surfaces (TextSurface, LogSurface), tui_repl.py rewritten as thin display layer composing them. TUI is now the default interactive shell in run() (TTY detect, MAN_NO_TUI=1 or non-TTY falls back to line mode). 36 new tests (pane+surface). Then added scrollback + live output: render() now takes offset (scroll into ring buffer), tui_repl polls keys at 100ms (stdscr.timeout) so background ai/gen output streams live, PgUp/PgDn scroll the focused pane, Ctrl+O toggles output/log focus. 4 offset tests added. Shell suite: 432 passed, 1 pre-existing API-dependent failure (test_safe_command_not_blocked).