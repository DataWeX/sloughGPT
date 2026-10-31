---
id: 20260929_081
board: 278fdeaf-bd55-4b00-86c9-f56148f037a3
title: "Card 081 — Kanban content hashes: note-provenance + root-hash audit chain"
status: doing
date: 2026-09-29
tags: [kanban, infra]
---

# Kanban content hashes: note-provenance + root-hash audit chain

## Background

Bakery design (2026-08-31): root hash = hash(card+slot+notes), note hash never changes unless edited, hash chain tracks state changes (audit trail), note-provenance hash resolves bidirectional sync conflicts ('add a note hash... which one is written first notes or kanban'). Phase 1 = provenance hash on notes<->board sync; phase 2 = root-hash chain; phase 3 = vectorized task-graph retrieval. board.jsonl currently has 0 hashes.

## Coordination
Parallel session already owns the core of this: `.opencode/plans/hash-tree-system.md` (on feat/shared-governance). Verify delivery; phase 3 (vectorized retrieval) remains ours.

## Source
Mined from stale opencode sessions (Aug 26-31 2026) at user's request: ~/.cache/slog-journeys/session-exports/.

## Delivery verification — 2026-09-30 (read-out from card 093 session, read-only, no code written)

**Verdict: the core was never delivered — it was built, then deleted, and the
plan was written afterwards without noticing.** Evidence below is all from
`main` / all-refs git inspection.

### What exists today

| Layer | State |
|---|---|
| TS hash-tree core | `apps/web/app/api/planner/helpers.ts:387‑575` — `sha256`, `HashTreeData`, `readHashTrees`/`writeHashTrees`/`getHashTree`/`createHashTree`/`addNoteToTree`/`removeNoteFromTree`/`rehashTree`. **Zero external callers**; `helpers.test.ts` has 0 hash-tree tests → dead code |
| API route | deleted in `1709e4c9f` ("no API route in new system"); `api/planner/` has only `board/notes/sync/tags/stats` |
| CardEditor display | deleted in `1709e4c9f` |
| Python core | `packages/app-planner/src/app_planner/hashtree.py` (370 lines) + `test_hashtree.py` (335) deleted in `9244a8eaf` as "dead code from old package" (150 tests green after removal). **No ref in the repo still contains the file** — not even `feat/shared-governance` |
| Data-model fields | `root_hash` on `BoardCard` (`helpers.ts:72`, `apps/web/components/planner/types.ts:20`) always written `''` (`helpers.ts:256`); no `root_hash`/`note_hash` anywhere in `packages/app-planner` |
| Sync path | `sync.py`, `kanban.py`, `sync/route.ts` contain **no hashing at all**; no `simpleHash` anywhere either → plan's "Current State" is stale in both directions |
| Stored data | `.planner/hashtrees/trees.jsonl` — tracked, clean, **345 roots / 79 notes / 79 history entries, every note `version: 1`** |

### The data is the only trail back — freeze it, don't regenerate

- All 345 roots have `placed_at` inside a **98-minute window on 2026-09-01
  (10:19–11:57)**, note IDs suffixed `_tagged` — one batch tagging pass.
- Sampled orphan card IDs **and** their note IDs appear in git history in
  exactly one commit: `558203e2f` (the commit that added `trees.jsonl`).
  They appear in **no** version of `.kanban/board.json`/`board.jsonl` across
  all 92 commits touching the board, on any branch. → the old cards were
  never committed anywhere else; **this file is the sole surviving record.**
- Caveat: it stores **hashes only, no note text**. It proves *which* version
  of a note existed (fingerprint/verification), it cannot reconstruct it.
- Therefore: **adopt the existing 345 roots as-is — never recompute them.**
  Recomputing with a new algorithm or a fresh timestamp severs the only link
  back to that generation. New/live cards (zero overlap with the 345) get new
  roots; this file becomes a read-only archive.

### Phase-1 hook exists but was never populated

Every one of the 345 records carries a top-level **`commits: []`** — the
schema for "bind this note hash when a new commit is made" is present and
**empty in 100% of records**. `rehashTree()` (chains `new_hash` off the last
history entry) has no caller, so no rehash has ever run.

### Hazard if re-wired

`writeHashTrees()` (`helpers.ts:450`) rewrites the **entire file** from a
Map. A caller passing a partial map would clobber the other records — i.e.
destroy the only trail. Any rebuild must make writes merge/append-only and
back up first (git tracking is the safety net: a destructive write shows as a
reviewable diff).

### Plan staleness (`.opencode/plans/hash-tree-system.md`)

- Layout says `packages/planner/src/planner/` → real package is
  `packages/app-planner/src/app_planner/`; `apps/web/app/api/oon/` no longer
  exists (now `api/planner/`).
- Premise says "no hash tree; `simpleHash()` (djb2) on note bodies" → reality
  is the inverse (Python tree deleted, TS sha256 tree present but unplugged).
- Plan file is tracked on `main` (last touched `558203e2f`) and also lives on
  `feat/shared-governance` (145 commits unmerged vs `main`) — that branch
  holds the plan and the data, **but no `hashtree.py` either.**

### Asks for the owning session

1. Rebuild Python-first (plan Phase 1) or unplug-and-wire the existing TS
   core? Either way `trees.jsonl` is inherited, never regenerated.
2. `trees.jsonl` → formalise as read-only archive (it is currently tracked
   but unguarded).
3. Decide whether the commit-binding (`commits[]`) lands as part of Phase 1
   provenance or as a separate git-side step.

*Coordination note: written by the card 093 (systems audit) session at the
user's request, read-only — no code, no board edits. Card statuses are the
standard three values: `todo`, `in_progress`, `completed`.*

## Delivery — 2026-10-01 (board hash chain, from the hash-chain session)

Phase-2-style root-hash chain landed for the board: `PlannerStore` computes
`chain_index` / `chain_prev` / `chain_hash` over all cards (topological over
`blocked_by`, stable by `(created_at, id)`), rewrites `board.jsonl` in chain
order at `sync()`, and `verify_chain()` reports tampered/unsequenced indices.
Live board: 869 cards, 0 cycles, verify ok. Code:
`packages/app-planner/src/app_planner/store.py` + `tests/test_store_chain.py`
(12 new tests; app-planner suite 167 green), commit `1d257ac5d` on
`feat/card-hash-chain`. Tracker card: `a8e408dc-19ce-4af3-b34c-9aa0e4964a37`.
`trees.jsonl` untouched (frozen per above). Caveat observed: syncs from
old-store code strip chain fields from cards they rewrite — self-heals on the
next new-code sync.

## Decision — user, 2026-10-01 (hash simplification, executed)

User rule: **no card-level hashes; one root hash per whole tray-slot** (slot
only, never card content); **cards carry notes and the notes are what get
hashed** (head + file, body off), stored with the rest of the metadata.

Executed the same day, no commits:

1. **Card chain removed** — `chain_hash`/`chain_prev`/`chain_index` stripped
   from all 881 card records in `.kanban/board.jsonl` (891 lines, 0 JSON
   errors; backup `~/.local/state/sloughgpt-gene/board.jsonl.bak-pre-hashstrip`).
   The unmerged `feat/card-hash-chain` tooling (commit `1d257ac5d`) re-applies
   these on demand — that session must be told the chain is retired, or it
   will re-add the fields on its next run.
2. **Root = slot-only** is the rule going forward; the 345 frozen
   `trees.jsonl` roots stay untouched as the archive (never-recompute stands
   for history; file is git-tracked either way).
3. **Note hashes inline** — the 15 card-linked dev-notes gained
   `hash: sha256(frontmatter-head + filename)` as the last frontmatter key;
   `app-planner verify` still reports all consistent.
4. **Dead hash code removed** — `helpers.ts` hash block (sha256,
   HashTreeData, read/write/get/create/add/remove/rehash Tree, `root_hash`
   field + write): 572 → 384 lines, had zero callers; vitest 27 + 69 green,
   `tsc --noEmit` clean. UI-side `root_hash` in `components/planner/types.ts`
   + the CardItem dot remain (never populated) — strip only on request.

## Built — slot chain with branch-on-removal, 2026-10-02 (user-directed)

Mechanism (user's design): each slot (`column:position`) has its own
append-only lineage. `slot_hash = sha256(slot_prev || column:position ||
notes_hash)` — the removed occupant's hash is the **parent** of the next
node, so a reused slot can never hash twice to the same value, and the
removed node stays in the journal as the root of its own branch. Card body
(title/description) never enters any hash; only notes do.

- Tool: `scripts/kanban_slot_chain.py` (`apply` / `verify`).
- Per card: `slot_position`, `notes_hash` (linked dev-note frontmatter
  hashes + inline `notes[]` texts; 100 cards non-empty), `slot_prev`,
  `slot_hash`.
- Journal (append-only, frozen): `.kanban/slot_history.jsonl` — 891 nodes,
  first-occupant parents `''`.
- board-meta: `tray_roots` = sha256 over each tray's slot tips (5 trays);
  `slot_history` filename recorded.
- Retired `chain_*` fields asserted absent by `verify`.

Gates: apply idempotent (2nd run = 0 nodes), verify ok (891 cards / 891
nodes / 5 trays), app-planner suite 167 passed, `app-planner verify` on
card-102 note "all consistent". Backup: `board.jsonl.bak-slotchain`
(pre-state also at `board.jsonl.bak-pre-hashstrip` from the chain strip).
No commits made.

## Wired — slot chain as canonical-writer hook, 2026-10-02

Core moved from the script into the shared package and hooked into the store:

- `app_planner/slot_chain.py` — canonical `apply_board()` / `verify_board()`.
  Raw-preserving writes (only dirty lines reformat; no-op runs touch nothing).
  New per-card `note_hashes` field = the individual digests (linked dev-note
  `hash:` values + inline `notes[]` sha256s); aggregate `notes_hash` keeps the
  legacy `dev:`/`inline:{i}:` prefixed formula → byte-parity with all existing
  journal nodes (re-apply = 0 new nodes).
- `PlannerStore` hooks: `_chain_apply()` runs after `_append_card`,
  `_surgical_rewrite`, `_write_board` — add/move/update/delete/sync all end up
  chained automatically; boards without the schema header skip (SlotChainError).
  `Card` dataclass gained the 5 slot fields so load→rewrite round-trips no
  longer strip them (`to_dict` emits them only once chained — byte-parity for
  unchained cards).
- `scripts/kanban_slot_chain.py` = thin CLI wrapper (sys.path bootstrap).
- Tests: `tests/test_slot_chain.py` — 9 cases (add-chained, update-preserves,
  move→branch parent, delete→branch root, inline/dev note inputs, tamper
  detection, retired-field strip, headerless skip). One existing byte-for-byte
  test updated: header may refresh chain aggregates, card lines stay identical.

Gates: app-planner 176 passed (167+9), ruff clean, shared-index test 6 passed,
real-board verify ok (893 cards / 1236 journal nodes / 895 slots / 5 trays),
app-planner verify "all consistent". No commits.

### Scope correction — user, 2026-10-02: NOT programmatic, one-time migrate

The store hooks were reverted in full: `store.py` and `test_store.py` are
byte-identical to HEAD again (verified via `git diff` — empty). The chain is
a **manual migration**: `scripts/kanban_slot_chain.py apply` whenever you want
a consistent snapshot; `verify` tells you when board edits drifted (any
store/UI rewrite of a card line drops its slot fields until the next apply —
that is accepted and by design). Tests rewritten to drive `apply_board`
directly (10 cases: apply/idempotency/body-edit-no-rehash/move+delete branch
parents/note inputs/tamper/retired-strip/headerless). Module docstring states
the no-write-path rule.

Gates after revert: app-planner **177 passed**, ruff clean, shared-index 6
passed, real-board verify ok (894 cards / 1495 journal nodes / 897 slots).
No commits.

### Final scope — user, 2026-10-02: one line in the kanban logic

Clarified after the full revert: the chain needs exactly **one line** in the
kanban code or every store write (add/update/**move**/delete/sync) loses the
slot fields again. Final wiring:
- `PlannerStore._atomic_write` ends with `apply_board(self._board_file)` —
  the single funnel all three write paths share; placed after the try/except
  so a successful write is never reported as failed. Import line only besides.
- `Card` dataclass stays slot-unaware (no fields, no to_dict logic) — the
  transiently stripped line is healed by apply within the same write.
- `apply_board` is now a quiet no-op on headerless boards (legacy surgical
  updates keep working); `verify` still flags them.
- Header refresh of `tray_roots` accepted in the byte-for-byte store test
  (card lines still asserted identical); new test
  `test_store_writes_keep_the_chain` covers add→chained, update→hash kept,
  move→new slot node.

Gates: app-planner **178 passed**, ruff clean, shared-index 6 passed,
real-board verify ok (897 cards / 1760 journal nodes / 899 slots — live
sessions' store writes are chaining automatically since the line landed).
No commits.
