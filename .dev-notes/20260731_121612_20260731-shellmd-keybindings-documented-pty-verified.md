---
id: 20260731_121612_20260731-shellmd-keybindings-documented-pty-verified
title: 20260731 SHELL.md keybindings documented + PTY-verified
status: done
tags: shell,docs,readline
created: 2026-07-31T12:16:12.960704+00:00
---

20260731 SHELL.md keybindings documented + PTY-verified

Rewrote SHELL.md Keyboard Shortcuts section (plain-shell, now default UI): readline Emacs binding groups — Completion & History (Tab, Up/Down/Ctrl+P/Ctrl+N, Ctrl+R/S), Cursor movement (Ctrl+A/Home, Ctrl+E/End, Alt+B/F), Editing (Backspace, Ctrl+D, Alt+D, Ctrl+W, Ctrl+K, Ctrl+U, Ctrl+T), Kill ring & process control (Ctrl+Y, Ctrl+L, Ctrl+C), plain-line fallback note. PTY probe /tmp/opencode/shell_pty_keys.py verified all 8 steps under a REAL controlling terminal (pty.fork): line mode boots without TUI banner/alt-screen; Tab pw->pwd; Alt+D kill-word (echo alpha beta -> echo  beta); Ctrl+K kill-to-end; Ctrl+Y yank; Ctrl+U kill-line; Ctrl+C aborts 'sleep 30' -> Aborted + post-abort echo ok; Up-arrow history recall. Root causes found: (1) probe step-2 false negative — readline redraws with backspace/CSI-P sequences, never reprints the edited line; assert on command output. (2) Ctrl+C 'Aborted missing' was a probe artifact — subprocess.Popen pty has no controlling terminal so ^C never signals; pty.fork (setsid+TIOCSCTTY) delivers real SIGINT. (3) /dev/tty open with mode 'r+' always fails on a non-seekable tty (io.UnsupportedOperation) so ConsoleIO._tty is effectively always None -> read() uses input() -> stdlib readline editing is always active; confirmed Tab+Alt+D work under pty.fork. Regression gate: 10-file shell/logging subset 569 passed, 12 skipped. Stray API server killed; pycache cleaned.

FINALIZED: full 20-check probe /tmp/opencode/shell_pty_keys_all.py -> FULL KEYBINDINGS PROBE OK. Covers line-mode boot/no-TUI, Tab, Alt+D, Ctrl+K/Y/U, Backspace, Ctrl+D delete-char, Ctrl+W, Ctrl+T, Ctrl+P/N, Up-arrow, Ctrl+R isearch, Alt+B/F, Home/End, Ctrl+L, Ctrl+S, Ctrl+D EOF-exit.

Ctrl+S FINDING: forward-search-history is UNREACHABLE in this stack. 0x13 is consumed by the terminal as XOFF (output pause) under IXON, same as bash; verified freeze on Ctrl+S and resume on Ctrl+Q. Also verified against a MINIMAL readline child (no slough shell) with IXON disabled (slave-side termios, setsid+TIOCSCTTY) — forward-search still never engages and Enter then fails to exit isearch. Conclusion: readline/terminal-level limitation, no product defect. SHELL.md Ctrl+S row updated: documented as subject to flow control, with stty -ixon suggestion. Probe check converted to a positive XOFF pause/resume verification (moved last, self-cleaning via Ctrl+U; the Ctrl+U backspace-redraw bytes were mistaken for buffer corruption during debugging).

PROBE BUG FOUND: shell lowercases the command token in the unknown-command message (repl.py cmd = parts[0].lower()), so buffer 'Xecho b1' prints as 'xecho'. Recreated probe asserted uppercase markers and falsely failed Alt+B/Alt+F/Home; markers corrected to lowercase. No product bug.

Final gate: 569 passed, 12 skipped (exit 0). Strays killed; pycache/__pycache__ cleaned.