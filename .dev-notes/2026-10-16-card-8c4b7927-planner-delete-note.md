---
id: 8c4b7927
title: "BUG: PlannerStore.delete_note returns True while the note survives (union-on-write re-appends deleted ids)"
status: doing
board_id: Main
date: 2026-10-16
branch: test/planner-delete-finality
tags: [planner, store, delete, occ, regression-test]
---

# Card 8c4b7927 — delete_note resurrection: measured state on the main lane

## Pickup verification (2026-10-16, session .wt-gpuops)

Re-measured everything before acting; the card's diagnosis was accurate for
2026-10-12 but the tree has moved:

- **Blocker cleared**: store.py is clean in every registered worktree
  (including the main tree and .wt-planner-del) — no lane owns it right now.
- **The union-on-write never landed on main**: `git log main -S "union" |
  -S "never saw"` on store.py = empty. The card was filed against another
  session's *uncommitted* WIP; main's committed history never carried it.
- **The bug does NOT reproduce on current main**: the card's own probe
  (`test_core.py::TestNoteStore::test_delete_note`: create → delete →
  get_note is None) PASSES; full `packages/app-planner/tests` =
  **261 passed / 0 failed**. The resurrection class is structurally
  impossible in main's mechanism: `89d464266` (OCC write path) replaced
  snapshot-serializing writes with generation-token guarded commits
  (`_current_token` sha256 → `_commit` flock+revalidate+atomic-replace →
  `_occ` bounded retry). All three note writers pass their read token, so a
  stale snapshot cannot silently re-append anything — it conflicts and
  re-reads.
- **Sibling on the integration lane (do not duplicate)**: Oluwatobi's
  `ed9fc70f2` on `fix/planner-delete-note` (worktree .wt-planner-del,
  committed 2026-10-16 07:38, clean/idle) ports the union+tombstone
  contract onto the *integration* lane's store.py — which still has the
  union. That lane needs tombstones; main does not.

## Residual taken here

Main's OCC solves the bug but **pins none of the delete-finality contract**:
`test_concurrency.py` covers forced-conflict retries, exhaustion, and the
notes *create* path — nothing asserts that a delete survives a stale
publish. The pin: a writer holding a pre-delete snapshot publishing with its
old token must raise `BoardWriteConflict` (loud) and the deleted id must
stay dead — one test in `packages/app-planner/tests/test_concurrency.py`,
test-only change, branch `test/planner-delete-finality`.

Downstream: `815c3287` (residue note undeletable) is unblocked by main's
mechanism — its actual deletion is user-data removal and stays with that
card's owner/user approval.

## Delivered — `3a57ab421` on `test/planner-delete-finality` (land-ready)

`test_delete_finality_stale_snapshot_cannot_resurrect` added to
`packages/app-planner/tests/test_concurrency.py` (+29/−1, test-only):
pre-delete snapshot + stale token → `_write_notes` → refused with
`BoardWriteConflict`; deleted id stays dead; survivor untouched.

Gates (all on branch cut from main `225c32ec0`):
- ruff on touched file: **All checks passed**
- targeted `test_concurrency.py` + `TestNoteStore` + `test_store.py`: **48 passed**
- full `packages/app-planner/tests`: **262 passed / 0 failed** (261 baseline + this pin)
- baseline probe `test_core.py::TestNoteStore::test_delete_note`: passes even
  *pre-change* — direct proof the bug is absent from committed main and this
  is a contract pin, not a bug fix here.

Card stays `in_progress` until the branch lands (main checked out in the
api-std tree; landing queue: fix/seam-holes-main, feat/credential-authority,
feat/web-terminal-phase1, this branch — `feat/uxjourneys-rehome` REMOVED:
superseded by the harness deletion on card 92af1aa0, must not land).

## WHERE THE BUG STILL LIVES (measured 2026-10-16 09:45) — the card's original blocker is STILL there

The union-on-write WIP the card diagnosed at pickup ("another session's
uncommitted change") **is still uncommitted in the main-tree working copy**:

- `git -C <main-tree> status -- packages/app-planner/src/app_planner/store.py`
  → ` M` (dirty, branch feat/tokenizer-checkpoint-binding).
- Its working `_write_notes(self, notes)` at :452 carries the union at
  :458/:470/:491 ("append it never saw... re-read at write time and union by
  id... records this snapshot never saw").
- Live repro through the real CLI/controller in that tree:
  `NotesController.delete` → **`NoteStoreError: store reported deleting note
  '...' but it is still readable — PlannerStore._write_notes re-appends ids
  found on disk`**. The honest-failure defense at
  `notes_controller.py:391-414` is COMMITTED on main (both trees identical),
  so the CLI now fails loudly instead of printing a false success — but the
  delete still cannot COMPLETE in that tree.
- Against committed main's store (this worktree): full real-engine round-trip
  **PASSES** — create → `ctl.delete` → gone → 0 notes left.

Consequences:
- `815c3287` (residue note 20e88d42, lives in the main tree's journal)
  remains blocked **while that tree's store.py stays dirty**; deletion is
  also user-data removal and needs approval either way.
- The WIP file is another lane's uncommitted work — NOT touched here.
  Resolution = its owner commits or discards it (main's committed OCC store
  already provides the wipe-protection the WIP was reaching for).

## Board mechanics lesson (cost a card-hour to diagnose — record it here)

Cards were filed through the `note` CLI, so **journal notes
(`.dev-notes/notes.journal.jsonl`) with title-matched statuses drive sync**;
a bare `board move` gets reverted on the next sync run if the journal note
disagrees. My two board moves at 09:29 were reverted within seconds by this.
The three layers and their fix points:

1. **journal note status** (`note update <uuid> --status doing`) — sync's
   authority; auto-sync moves the card itself. THIS is the correct lever.
2. **board column** (`board move`) — only sticks when no title-matched
   journal note says otherwise (e.g. a9e2c61e: no journal note → stuck ✓).
3. **md backing note** (`~/.config/dev-notes/*.md`) — the `verify` layer
   only; verify never writes.

Also: the note status vocabulary is
`["open","doing","wip","done","blocked","review","todo"]` — **`in_progress`
is NOT a status**: `STATUS_TO_COLUMN.get("in_progress", "todo")` silently
maps it to `todo`. Four older md notes still carry invalid
`in_progress`/`in-progress` (browser-shell-ansi, mock-target-shim,
device-contract, gene-seam) — left alone (fixing note without moving card
would create verify mismatches); hygiene candidate for their owner.
