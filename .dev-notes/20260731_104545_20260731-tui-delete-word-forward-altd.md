---
id: 20260731_104545_20260731-tui-delete-word-forward-altd
title: 20260731 TUI delete word forward (Alt+D)
status: done
tags: shell,tui,readline
created: 2026-07-31T10:45:45.168260+00:00
---

20260731 TUI delete word forward (Alt+D)

Added readline kill-word (Alt+D) to split-pane TUI. _delete_word_forward() computes whitespace-skip + word range from _input_cursor, deletes it, caret unchanged, pushes killed text via _push_kill(). Escape-prefix branch passes {'f':'fwd','b':'bwd','d':'delword'} and dispatches alt:delword. 5 unit tests (at-start, mid-word, at-whitespace, end no-op, kill-ring push): suite at 97 passed. PTY probe tui_pty_delword.py drives Alt+D x3 (echo foo bar -> lone prompt), no-op on empty/end-of-line, Ctrl+Y yanks killed word — all steps OK. Regression gate 565 passed, 12 skipped. SHELL.md + ROADMAP.md updated.