---
id: 20260801_080900_unify-planner-tools-into-one-cli-shared-config
title: Unify planner tools into one CLI + shared config
status: done
tags: planner,cli,tools
created: 2026-08-01T08:09:00.260464+00:00
---

Unify planner tools into one CLI + shared config

Unified notes/kanban/gui/sync behind one planner CLI with shared config (tools/planner/src/planner/config.py). Resolution: CLI flag > env (PLANNER_NOTES_DIR/PLANNER_BOARD_DIR/PLANNER_BACKEND) > project root (.kanban/board.json ancestor) > ~/.config fallback. Backend inferred from notes dir (MogDB journal -> mogdb). Single STATUS_TO_COLUMN/COLUMN_TO_STATUS maps, STATUS_ICONS in core, shared sync_notes_to_board for 'planner sync', GUI Sync button, and sync-notes-to-board. kanban store honors config defaults. 58 planner tests pass.