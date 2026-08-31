---
id: 20260801_064640_mogdb-journal-replay-bug-fix-updatedelete-ops-clobber-docs-o
title: MogDB journal replay bug fix — update/delete ops clobber docs on load
status: done
tags: mogdb,notes,storage
created: 2026-08-01T06:46:40.537829+00:00
---

MogDB journal replay bug fix — update/delete ops clobber docs on load

Collection._load() replayed every journal line as a standalone document keyed by _id. An update op became a stripped {_id, update} doc that overwrote the real inserted doc, so notes edited after insert vanished from reads (title -> (untitled), note_by_id -> not found). This hid 11 notes from sync-notes-to-board (TUI session note included).

Fix: _load() now distinguishes journal ops (insert/update/update_many/delete/delete_many) from compacted plain-doc snapshots (.mogdb) and replays ops in order via _apply_update. Added 5 regression tests (127 total, all pass). Re-running sync-notes-to-board added the 11 missing cards (76 -> 87). Files: packages/mogdb/src/mogdb/collection.py, packages/mogdb/tests/test_mogdb.py. Not committed (awaiting user).