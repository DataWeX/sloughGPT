---
id: 20260803_143113_tui-live-tests-for-editing-parity
title: TUI live tests for editing parity
status: done
tags: shell,tui,tests
created: 2026-08-03T14:31:13.508430+00:00
---

TUI live tests for editing parity

Added 9 live pty tests to test_shell_tui_live.py closing the documented-but-untested TUI editing paths: Alt+D delete-word-forward + kill ring, Ctrl+T transpose (end/mid), Ctrl+D delete-at-caret, Ctrl+Y kill-ring cycling, n forward + N backward repeat of the last output-pane search, Ctrl+K kill-to-end + yank, Ctrl+W kill-word ring push + yank, Ctrl+A/Ctrl+E caret motion. Fixed the _Screen harness: added CSI P (DCH) and @ (ICH) handlers — curses differential refresh emits DCH for mid-line deletions which the model previously ignored. No production code changes. Shell regression: 453 passed (444 + 9 new) in 57s.