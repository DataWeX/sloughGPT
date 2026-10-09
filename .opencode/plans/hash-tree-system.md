# Plan: Hash Tree System for Kanban Cards

## Context

The user wants a cryptographic hash tree for their kanban/note system. The root hash (card+slot fused) is a stable seed — like a BIP32 HD wallet — that can derive all child note hashes. The system must be a reusable data structure (like their OPG system), vectorized for graph operations, with boomerang encryption so hashes don't expose raw data.

## Current State

- **Python** (`kanban.py`): Card dataclass with inline `notes[]`. Board stored as `board.json`.
- **TS** (`helpers.ts`): BoardCard with `note_ref` linking to `.dev-notes/*.md` files. `simpleHash()` (djb2-like) on note bodies. Bidirectional sync between notes and board.
- **No hash tree**: Current hashes are simple content checksums, not a hierarchical tree.

## Architecture

### The Hash Tree

```
Root Hash (card + slot) — NEVER CHANGES, the seed
    │
    ├── Note Hash 1 — changes only when note edited
    ├── Note Hash 2 — changes only when note edited
    └── Note Hash 3 — changes only when note edited

Root = hash(card_content + slot_position + slot_tray + placed_at)
Note_hash = hash(root_seed + note_content)  — derived from root, not independent
```

### Key Rules

| Rule | Implementation |
|------|---------------|
| Root never changes | Computed once at card+slot creation, stored permanently |
| Note hash derived from root | `note_hash = hash(root + note_content)` — root is the key |
| Note edit → new note hash | Root stays, note hash recomputed via root |
| History stored | Every root hash + note hash change logged |
| One root retrieves all | From root hash, derive all current note hashes (like BIP32) |
| Boomerang encryption | Root hash encrypted — shows structure, not data |

### Data Structures

#### 1. `CardSlotHash` (root, never changes)

```python
@dataclass
class CardSlotHash:
    root: str           # hash(card + slot) — permanent
    card_id: str        # reference to card
    slot_id: str        # reference to slot (tray + position)
    tray: str           # which tray (column/state)
    position: int       # position within tray
    placed_at: str      # when card entered this slot
    created_at: str     # when root hash was computed
```

#### 2. `NoteHash` (derived from root)

```python
@dataclass
class NoteHash:
    note_id: str        # reference to note
    hash_value: str     # hash(root + note_content)
    root_ref: str       # which root this belongs to
    version: int        # incremented on each edit
    created_at: str
    updated_at: str
```

#### 3. `HashHistory` (audit trail)

```python
@dataclass
class HashHistoryEntry:
    root_ref: str       # which root changed
    old_hash: str       # previous hash
    new_hash: str       # new hash
    change_type: str    # "note_edit" | "note_add" | "note_delete"
    note_id: str        # which note changed
    timestamp: str
```

#### 4. `HashTree` (the full structure)

```python
@dataclass
class HashTree:
    root: CardSlotHash
    notes: list[NoteHash]
    history: list[HashHistoryEntry]

    def derive_note_hash(self, note_content: str) -> str:
        """From root, derive a note hash."""
        return hash(self.root.root + note_content)

    def on_note_edit(self, note_id: str, new_content: str) -> NoteHash:
        """Root stays, recompute note hash, log history."""
        ...

    def get_all_hashes(self) -> dict:
        """From root, return all current hashes (like BIP32 derive)."""
        ...

    def verify(self) -> bool:
        """Verify all note hashes match their root derivation."""
        ...
```

### File Layout

```
packages/planner/src/planner/
├── hashtree.py          # NEW: CardSlotHash, NoteHash, HashHistory, HashTree
├── kanban.py            # MODIFY: Card gets root_hash field
├── core.py              # MODIFY: Note gets note_hash field

apps/web/app/api/planner/
├── helpers.ts           # MODIFY: add hash tree functions
├── hashtree/
│   └── route.ts         # NEW: GET/POST hash tree endpoints
```

## Implementation Steps

### Phase 1: Core data structures (Python)

1. Create `packages/planner/src/planner/hashtree.py`:
   - `CardSlotHash` dataclass (root, card_id, slot_id, tray, position, placed_at)
   - `NoteHash` dataclass (note_id, hash_value, root_ref, version)
   - `HashHistoryEntry` dataclass (root_ref, old_hash, new_hash, change_type, note_id, timestamp)
   - `HashTree` dataclass with methods:
     - `compute_root(card_content, tray, position, placed_at)` — compute and store root (never changes)
     - `derive_note_hash(note_content)` — from root, derive note hash
     - `on_note_edit(note_id, new_content)` — recompute note hash, log history
     - `on_note_add(note_id, content)` — add note hash, log history
     - `on_note_delete(note_id)` — remove note hash, log history
     - `get_all_hashes()` — from root, return all current hashes
     - `verify()` — verify all note hashes match root derivation
   - `HashTreeStore` — persistence (JSONL file per tree)

2. Modify `kanban.py`:
   - Add `root_hash: str` field to Card dataclass
   - When card is created in a slot, compute root hash
   - Store root hash on card

3. Modify `core.py`:
   - Add `note_hash: str` field to Note dataclass
   - When note is created/edited, derive hash from parent card's root

### Phase 2: TS hash tree (web layer)

4. Add to `helpers.ts`:
   - `computeRootHash(card, slot)` — compute root from card+slot
   - `deriveNoteHash(root, noteContent)` — derive note hash from root
   - `onNoteEdit(tree, noteId, content)` — update note hash, log history
   - `getAllHashes(root)` — derive all hashes from root
   - `verifyTree(tree)` — verify integrity

5. Create `apps/web/app/api/planner/hashtree/route.ts`:
   - GET: retrieve hash tree for a card
   - POST: create/update hash tree

### Phase 3: Integration

6. Wire `syncNotesToBoard()` to use hash tree:
   - When syncing, compute/update hashes
   - Log history entries on changes

7. Add hash tree display to UI:
   - Show root hash on card
   - Show note hashes
   - Show hash history (audit trail)

## Verification

- Unit tests for hash derivation (root never changes, notes derive from root)
- Unit tests for history logging
- Unit tests for `get_all_hashes()` (one root → all children)
- Unit tests for `verify()` (integrity check)
- Integration test: sync notes → verify hashes → edit note → verify new hash, old root unchanged

## Files to Create/Modify

| File | Action |
|------|--------|
| `packages/planner/src/planner/hashtree.py` | CREATE |
| `packages/planner/src/planner/kanban.py` | MODIFY (add root_hash to Card) |
| `packages/planner/src/planner/core.py` | MODIFY (add note_hash to Note) |
| `apps/web/app/api/planner/helpers.ts` | MODIFY (add hash functions) |
| `apps/web/app/api/planner/hashtree/route.ts` | CREATE |
| `packages/planner/tests/test_hashtree.py` | CREATE |
