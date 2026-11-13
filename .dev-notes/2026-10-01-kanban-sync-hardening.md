---
id: 20261001_kanban_sync_hardening
title: "Kanban sync hardening: no silent column reversion (card 3ccc38b7)"
status: done
sprint: 2026-10-01
tags: [kanban, sync, planner, hardening]
gh: ""
created: 2026-10-01T18:50:00Z
board: 3ccc38b7
hash: 6b5ebf0527d9bf286c42c341198db686cd05be5824effca429fe777bdd5866d6
---

# Kanban sync hardening: no silent column reversion (card 3ccc38b7)

## Decision record (user-approved design)

- Default sync: unmappable note status LEAVES card column unchanged; pessimistic reset only via
  explicit `planner sync --repair` (counted report, dry-run) — DR path for the 348-null legacy journal.
- Status vocabulary: canonical `wip` -> `doing` (emission); read-side legacy aliases `wip` / `in_progress`
  stay valid -> `in_progress` column (migration precedent migrate_boards.py:46).
- Anti-drift layers: alias tolerance + verify.py finding + guard test (all three selected).
- Junk cleanup approved: rows My Note x3 + Synced Task AND their source notes (git-tracked, revertible);
  af19fb40 char-tag repair.

## Root cause evidence

- `config.STATUS_TO_COLUMN.get(status, "todo")` in store.sync():512, sync.py:34, verify.py:168 —
  unknown/absent status forces column `todo` (demo: done card + status in_progress/doing/''/None/partial
  all reverts to todo).
- Live journal statuses are canonical (open/wip/done/todo/review); deviating values live in md notes
  (in_progress x9, doing x1, partial x1 — copy-paste donor: 2026-08-31-hash-tree-implementation-plan.md).
- Journal tags: 446 comma-string records in legacy store/ journal; sync does list(note.tags or []) -> chars.
- auto_sync fires on every notes CLI mutation (someone moved card 075 -> review at 18:38).

## Status

**done** — delivered on governance `24ff79996`:
- `d386dcec1` card filed
- `0357cb865` implementation + tests (11 files, +460/-55)
- `24ff79996` data hygiene (junk rows+notes dropped, af19fb40 tags repaired)
- close row committed after.

Gates: 191 passed (24 new) x2 runs, ruff clean, mypy 20->20 identical
(baseline verified against HEAD). Incident log: edit-tool paths resolved
to main tree (session cwd) while shell cd'd to .wt-shot — 7 source edits
landed in main tree, were copied into the worktree, main tree restored
clean (`git status` empty for those files).
