---
id: 20260801_082334_bidirectional-notes-board-sync
title: Bidirectional notes->board sync
status: done
tags: planner,sync,kanban
created: 2026-08-01T08:23:34.357520+00:00
---

Bidirectional notes->board sync

Extended planner sync (planner.sync.sync_notes_to_board) to be bidirectional for status: existing cards are moved to the column matching the note's current status, not just created when missing. Returns (added, updated, total); updated call sites in planner sync CLI, GUI /api/sync + Sync button toast, and gui --sync startup print. Repo: 3 pre-existing drift cards moved in_progress->done (Infrastructure test coverage round 2/3, SloEngine tests); rerun is idempotent 0/0. Tests: 8 sync + 16 config + 36 gui = 60 passed.