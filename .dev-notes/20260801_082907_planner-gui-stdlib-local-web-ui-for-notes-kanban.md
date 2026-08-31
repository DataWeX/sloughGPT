---
id: 20260801_082907_planner-gui-stdlib-local-web-ui-for-notes-kanban
title: planner gui — stdlib local web UI for notes + kanban
status: done
tags: planner,gui,tools
created: 2026-08-01T08:29:07.355413+00:00
---

planner gui — stdlib local web UI for notes + kanban

Built zero-dependency web GUI: packages/planner/src/planner/gui.py (ThreadingHTTPServer + embedded HTML/CSS/JS SPA, no external assets). REST API: GET /, /api/notes[/{id}], /api/board, /api/tags, /api/stats; POST /api/notes, /api/board/move, /api/sync; PUT/DELETE /api/notes/{id}. Board drag-drop syncs note status (column->status map); Sync button mirrors sync-notes-to-board. Wired 'planner gui' / 'python -m planner gui' subcommand in core.py cli_main + __main__.py; installed editable. Args: --notes-dir, --board-dir, --backend file|mogdb, --host, --port (8787), --no-open, --sync. 36 tests (18 x file+mogdb) in packages/planner/tests/test_gui.py, all pass. Now part of the unified planner CLI under packages/planner.