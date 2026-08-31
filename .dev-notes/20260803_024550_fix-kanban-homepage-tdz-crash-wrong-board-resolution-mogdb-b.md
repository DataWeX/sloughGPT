---
id: 20260803_024550_fix-kanban-homepage-tdz-crash-wrong-board-resolution-mogdb-b
title: Fix kanban homepage TDZ crash + wrong board resolution + mogdb backend
status: done
tags: planner,gui,kanban,config
created: 2026-08-03T02:45:50.675034+00:00
---

Fix kanban homepage TDZ crash + wrong board resolution + mogdb backend

Fixed three issues with the kanban homepage (planner gui).

1. Homepage crash (TDZ): cardHtml in gui.py rendered the notes chip as `const n = c.notes.length ? "${n} note..." : ""` — a temporal dead zone error. Any card with notes threw 'Cannot access n before initialization', aborting renderBoard so the board stayed blank. The 13-card fallback board had no notes, masking the bug. Fixed to reference c.notes.length. Regression test test_card_html_note_chip_has_no_tdz_self_reference.

2. Wrong board when launched outside repo: find_project_root() walked up from CWD only, so launching planner gui from /home/mana/Documents/Default Project fell back to stale ~/.config/kanban/board.json (13 cards) instead of repo .kanban/board.json (121). Added package-location fallback (_walk_for_board from Path(config.__file__)) used when the CWD search misses. Two new config tests.

3. mogdb backend: with dirs now resolving to repo, default_backend() infers mogdb from .dev-notes/store/notes.journal.jsonl; GUI now serves the repo board (121 cards) with backend mogdb.

Verification: jsdom render of the real SPA shows the full board (104KB HTML, 0 toast errors); 130 planner tests pass (129 + 1 new GUI regression); py_compile clean; pycache cleared.