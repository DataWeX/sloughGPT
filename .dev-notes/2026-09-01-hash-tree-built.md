---
title: Hash Tree System Built
date: 2026-09-01
tags: [planner, hashtree, crypto, boomerang, kanban]
status: done
---

# Hash Tree System Built

## What was done

### 1. Python Core (`packages/planner/src/planner/hashtree.py`)
- `CardSlotHash` — root hash (card + slot, never changes)
- `NoteHash` — derived from root, recompute on edit
- `HashHistoryEntry` — audit trail for changes
- `HashCommit` — immutable commit with parent chaining (like git)
- `HashTree` — full tree with add_note, delete_note, commit, verify, vectorize
- `HashTreeStore` — JSONL persistence
- `create_hash_tree()` — factory function

### 2. Data Model Updates
- `Card` (`store.py`) — added `root_hash` and `notes` fields
- `Note` (`core.py`) — added `note_hash` field, serialized in frontmatter
- Exports in `__init__.py`

### 3. Sync Integration (`sync.py`)
- When syncing notes to board, creates hash trees for new cards
- Updates hash trees when notes are synced
- Cards get `root_hash` set on creation

### 4. Web API Routes
- `GET /api/oon/hashtree?cardId=xxx` — get hash tree for a card
- `GET /api/oon/hashtree` — list all hash trees
- `POST /api/oon/hashtree` — create hash tree
- `DELETE /api/oon/hashtree?cardId=xxx` — delete hash tree

### 5. TypeScript Helpers (`helpers.ts`)
- `sha256()` — SHA-256 hash
- `HashTreeData` interface
- `readHashTrees()`, `writeHashTrees()`, `getHashTree()`, `createHashTree()`, `addNoteToTree()`

### 6. UI Updates
- `CardEditor` — shows hash tree info (root, tray, notes count, history count)
- `CardItem` — purple dot indicator when hash tree is active
- `types.ts` — added `HashTree` interface

### 7. Tests
- 37 hashtree unit tests (all pass)
- 135 total planner tests (all pass)

## Files Changed
- `packages/planner/src/planner/hashtree.py` — NEW
- `packages/planner/src/planner/store.py` — Card gets root_hash, notes fields
- `packages/planner/src/planner/core.py` — Note gets note_hash field
- `packages/planner/src/planner/sync.py` — Wire hash tree to sync
- `packages/planner/src/planner/__init__.py` — Export hashtree classes
- `apps/web/app/api/oon/hashtree/route.ts` — NEW API routes
- `apps/web/app/api/planner/helpers.ts` — Add hash tree helpers
- `apps/web/components/planner/CardEditor.tsx` — Hash tree display
- `apps/web/components/planner/CardItem.tsx` — Purple dot indicator
- `apps/web/components/planner/types.ts` — HashTree interface
- `packages/planner/tests/test_hashtree.py` — 37 tests

## Verification
- All 37 hashtree tests pass
- All 135 planner tests pass
- TypeScript clean
- Next.js build succeeds
