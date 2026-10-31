---
title: Hash Tree Implementation Plan — Buffet System
date: 2026-08-31
tags: [planner, kanban, hashtree, implementation, buffet, recovered, crypto]
status: doing
---

# Hash Tree Implementation Plan — Buffet System

## Recovered Conversation

Lost conversation from `ses_faa3fe4b9ffeR4MxXIHB2nXth8` (Oon Kanban board demo) was recovered from opencode SQLite DB. Full transcript: `~/.config/dev-notes/2026-08-31-full-recovered-conversation.md`

## The Metaphor (from lost conversation)

| Food Term | System Term | Description |
|-----------|-------------|-------------|
| **Buffet** | GUI / App | Many people can serve food |
| **Table (Buffet)** | Account / User board | Your personal section |
| **Tray** | State column | picks, choices, mades, themes |
| **Slot** | Position in tray | Shufflable — moves between trays |
| **Card** | Task/note item | The food — has a content hash |
| **Boomerang algorithm** | Hash cycle | Content → hash → vector → similarity → back |

## Key Rules

1. Trays can be shuffled (reorder within tray)
2. Slots move between trays (state changes: picks → choices → mades → themes)
3. When a state has been moved, we keep track
4. Cards rot/shed if stuck too long (staleness detection)
5. Cards can be grouped (rice with soup = related tasks)
6. Cards belong to a "mother" hash, not a tray — move between trays without breaking chain

## The Hash Chain

```
Card content → Hash (content fingerprint)
                  ↓
         Root hash (card + slot) — NEVER CHANGES (mother hash)
                  ↓
         Note hashes derived from root (like BIP32)
                  ↓
         Color swatch → hash → pixel → vector space
                  ↓
         Boomerang: similarity search → verify hash → show data
```

## Boomerang Algorithm

The full cycle (from recovered conversation):
1. Content → hash (fingerprint — encrypted, can't read raw data)
2. Hash → color swatch (generate hash from colors)
3. Color → pixel (swatches become pixels)
4. Pixel → vector space (just a line, no data — structure without content)
5. Vector → similarity search (find related cards)
6. Similarity → back to hash (verify hash validity → decrypt → show data)
7. Integrity: root hash never changed

## Key Concepts (from recovered conversation)

| Concept | Meaning |
|---------|---------|
| **Boomerang** | Hash → Vector → Similarity → Back to hash (encryption/verification) |
| **Mother hash** | Root hash that holds cards and slots (never changes) |
| **Design** | Semantic fingerprint of the card |
| **Swatch** | Color that generates a hash → pixel |
| **Shuffling** | Moving cards between trays without breaking hash chain |
| **Sentence decomposition** | Break down, transform, reconstruct without losing meaning |
| **Typing errors** | System preserves semantic accuracy through typos |

## Key Files (from original plan)
- `.opencode/plans/hash-tree-system.md` — original plan (189 lines)
- `packages/planner/src/planner/hashtree.py` — implemented (568 lines)
- `packages/planner/src/planner/kanban.py` — Card dataclass (needs root_hash field)
- `packages/planner/src/planner/core.py` — Note dataclass (needs note_hash field)

## Implementation Phases

### Phase 1: Core data structures (Python) ✓ DONE
1. ✅ Create `packages/planner/src/planner/hashtree.py` (568 lines)
   - ✅ `CardSlotHash` — root, card_id, slot_id, tray, position, color, design, placed_at, created_at
   - ✅ `NoteHash` — note_id, hash_value, root_ref, version, created_at, updated_at
   - ✅ `HashHistoryEntry` — root_ref, old_hash, new_hash, change_type, note_id, timestamp
   - ✅ `HashCommit` — immutable commit with parent_commit chaining, color, pixel
   - ✅ `HashTree` — compute, derive_note_hash, on_note_add/edit/delete, get_all_hashes, verify, vectorize
   - ✅ `HashTreeStore` — save/load/delete/list/get_commits/verify_all persistence

2. ✅ Modify `kanban.py` — Card gets `root_hash` field, add_card auto-computes hash tree, add_note derives note hash
3. ✅ Modify `core.py` — Note gets `note_hash` field, serialized in markdown frontmatter

### Phase 2: TS hash tree (web layer) ✓ DONE
4. ✅ Add note/hash helpers to `helpers.ts` (listNotes, addNote, deleteNote, BoardCard.notes.created_at)
5. ✅ Create API routes:
   - `apps/web/app/api/oon/board/route.ts` — GET board
   - `apps/web/app/api/oon/board/move/route.ts` — POST move card
   - `apps/web/app/api/oon/board/cards/route.ts` — POST create card
   - `apps/web/app/api/oon/board/cards/[id]/route.ts` — PUT/DELETE card
   - `apps/web/app/api/oon/board/cards/[id]/notes/route.ts` — GET/POST notes
   - `apps/web/app/api/oon/board/cards/[id]/notes/[noteId]/route.ts` — DELETE note
   - `apps/web/app/api/oon/hashtree/[cardId]/route.ts` — GET/POST/DELETE hash tree
   - `apps/web/app/api/oon/tags/route.ts` — GET tags

### Phase 3: Integration ✓ DONE
6. ✅ Wire sync.py to use hash tree (add_card creates tree, add_note updates tree)
7. ✅ Notes UI in CardEditor — load/add/delete notes, Enter to add, × to delete
8. ✅ Wire planner-client.ts to `/api/oon/*` routes (was pointing at non-existent `/api/planner/*`)
9. ✅ Export hashtree from planner package __init__.py
10. ✅ Type fixes: KanbanCardData.notes, BoardCard.notes, PlannerCard.notes all have created_at

## Verification ✓ DONE
- 35 hashtree unit tests (all pass)
- 181 total planner tests (all pass)
- TypeScript clean (excluding pre-existing SystemHealthPanel + test file issues)

## Open Questions (from conversation)
- What's the final name for "slot"? (conversation cut off at "the slot is called")
- Is "picks/choices/mades/themes" the tray naming convention?
- Cards on multiple trays — user confirmed: cards have a "mother" hash, belong to mother not tray
- Vectorize: discuss later, get structure first (user instruction)
- Sentence decomposition: break down sentence by sentence, recreate without breaking meaning

## What was built in this session
- Full `hashtree.py` module (568 lines) with boomerang algorithm structure
- Color swatch → hash → pixel pipeline
- Commit chain (like git commits, immutable, parent_commit field)
- BIP32-like derivation (root → all note hashes)
- Vectorize method (structure hash, note hashes, pixel data)
- JSON persistence via HashTreeStore
- All 3 phases complete: core data structures, TS web layer, integration
- 8 API routes (board, move, cards, notes, hashtree, tags)
- Notes UI in CardEditor (load/add/delete)
- 35 hashtree tests + 181 total planner tests
- Full stack: Python backend → Next.js API → React frontend
