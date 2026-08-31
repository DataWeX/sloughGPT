---
id: 20260731_102722_20260731-tui-output-pane-content-search
title: 20260731 TUI output-pane content search (/)
status: done
tags: shell,tui,readline
created: 2026-07-31T10:27:22.935638+00:00
---

20260731 TUI output-pane content search (/)

Output-pane content search in the split-pane curses TUI (tui_repl.py), the final readline checklist item.

- '/' at an empty prompt opens output search; typing refines the query and jumps the scroll so the matched line lands at the top of the output pane.
- Case-insensitive substring search over the whole buffer with wrap (forward/backward); Enter accepts (scroll kept at the match), Esc/Ctrl+G/Ctrl+C cancel (scroll restored).
- No-match shows '(failed output-search)' indicator with the query intact — 'n'/'N' inside a query are literal characters (fixed a bug where they were intercepted as match cycling).
- After accept, 'n'/'N' at an empty prompt repeat the last search from the current match (wrapping), gated on scrolled-back state (scroll > 0) so command text keeps its n/N.
- State: _out_searching, _out_search_q, _out_search_last, _out_search_sel, _out_search_save, _out_search_failed; helpers _out_find, _apply_out_search, _repeat_out_search.
- Tests: test_shell_tui_repl.py 92 passing (13 prior + 5 new _repeat_out_search tests); shell/logging 10-file subset 560 passed, 12 skipped.
- PTY probe /tmp/opencode/tui_pty_outsearch.py verified all steps: /target_042 jump to pane top, Enter accept, /zzz_no_such literal-n + failed indicator + Esc restore, /search_target_04 n x10 cycle-with-wrap, N x10 reverse wrap, Ctrl+L + literal '/' in echo.
- docs/SHELL.md keybinding table gains the '/' row; docs/ROADMAP.md CLI bullet mentions output search + n/N repeat.