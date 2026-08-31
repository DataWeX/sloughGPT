---
id: 20260731_080938_split-pane-tui-fixes-pty-verified-search-escape-terminal-lea
title: Split-pane TUI fixes PTY-verified (search, escape, terminal leak)
status: done
tags: shell,tui,bugfix
created: 2026-07-31T08:09:38.916807+00:00
---

Split-pane TUI fixes PTY-verified (search, escape, terminal leak)

Fixed and PTY-verified three split-pane curses TUI issues in packages/core-py/domains/shell/tui_repl.py and cli_logger.py.

1. Bug #3 CLI-bridge terminal leak: cli_logger.py gained _TERMINAL_ENABLED/set_cli_terminal/_cli_print (14 call sites); TuiRepl.run() toggles set_cli_terminal(False) around curses.wrapper, restored in finally. PTY frames show no raw [info] lines during the session; only legitimate post-TUI shutdown line remains.
2. Reverse-i-search bug: _search_back rewrote — clamps start into range when start==len(history), returns -1 on no match, proper fwd/back bounds; _apply_search only applies found>=0. Previously buffer became 'help' instead of matching entry.
3. Escape delay: curses.set_escdelay(25) after curses.raw() in _main so Esc cancels search immediately.

Verification: /tmp/opencode/tui_pty2.py PTY probe (24x100, TERM=xterm-256color, SLO_AUTOLOAD_MODEL='') boots API server (~1.5s with wait-condition fix), runs date/help/he+Ctrl-R+d/a+Esc/lp+Enter/exit. Frames show: search 'd' matches 'date' (was broken), Esc restores 'he', typing resumes to 'help', no terminal leak frames 0-9. Decoder upgraded to buffer partial escapes and decode UTF-8 incrementally (split-escape and UTF-8 artifacts eliminated; wait-condition fixed to [OUTPUT]+lambda). 115 shell/logging tests pass.