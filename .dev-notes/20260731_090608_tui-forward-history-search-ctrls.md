---
id: 20260731_090608_tui-forward-history-search-ctrls
title: TUI forward history search (Ctrl+S)
status: done
tags: shell,tui,cli
created: 2026-07-31T09:06:08.816013+00:00
---

TUI forward history search (Ctrl+S)

Ctrl+S enters forward-i-search (Ctrl+R/Ctrl+S switch direction mid-search, Ctrl+F stays as forward alias). _apply_search direction-aware with fresh-query sentinels (-1 fwd / history_pos rev); status bar shows forward-i-search vs reverse-i-search. 6 new tests; 41 TUI + 267 repl = 308, full shell/logging 498 pass. PTY probe: entered forward search, typed 'echo' -> oldest match, Ctrl+S -> newer, Ctrl+R -> reverse, Esc cancel restored buffer.