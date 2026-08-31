---
id: 20260731_094501_20260731-tui-word-movement-altfb-ctrlarrows
title: 20260731 TUI word movement (Alt+F/B, Ctrl+arrows)
status: done
tags: shell,tui,readline
created: 2026-07-31T09:45:01.335659+00:00
---

20260731 TUI word movement (Alt+F/B, Ctrl+arrows)

Added readline word movement to the split-pane curses TUI. _move_word_forward() (Alt+F / Ctrl+Right) skips whitespace then advances to the end of the word; _move_word_backward() (Alt+B / Ctrl+Left) skips back to the start of the current/previous word. Word = maximal run of non-whitespace. ESC-prefix decoding via new module-level _read_escape_remainder(): polls the next bytes with a 0ms timeout, decodes Alt+<char> (alt:f/b) and raw ESC [ 1;5C/D sequences (seq:ctrl-right/left), restores the 100ms poll timeout; a lone Esc resolves instantly to no-op (search-cancel path unaffected). ncurses folds Ctrl+arrows into opaque codes (measured 554/569 via a keycodes PTY probe) not exposed by Python curses, so added _KEY_CTRL_LEFT/_KEY_CTRL_RIGHT constants via getattr(curses,...) fallback. Unit tests: 69 TUI tests pass (15 new: 8 word-move edge cases incl. mid-word, whitespace, at-start/end no-ops; 7 ESC-decoder tests incl. lone Esc, double Esc, alt known/unknown, ctrl-left/right, timeout restore). PTY probe /tmp/opencode/tui_pty_wordmove.py verified all four bindings plus lone-Esc no-op by inserting markers at the new caret positions. Docs: docs/SHELL.md keybinding table (word-move rows), docs/ROADMAP.md CLI bullet.