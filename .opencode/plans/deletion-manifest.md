# Deletion Manifest — Planner Rebuild

Everything below will be deleted. This document is the reference.

---

## Python Package: `packages/planner/` (29 files, 7,827 lines total)

### Source (2,704 lines)

| File | Lines | What it contains |
|------|-------|-----------------|
| `src/planner/__init__.py` | 11 | Re-exports: Note, NoteStore, Board, Card, KanbanStore, sync_notes_to_board |
| `src/planner/__main__.py` | 13 | CLI dispatcher: kanban/gui/sync/notes subcommands |
| `src/planner/config.py` | 120 | STATUS_TO_COLUMN, COLUMN_TO_STATUS, find_project_root, default_notes_dir/board_dir, backend inference |
| `src/planner/core.py` | 816 | NoteStore: create/get/update/delete/list/search/today/export/sprints/timeline. File + MogDB backends. YAML frontmatter markdown. CLI with 12 subcommands |
| `src/planner/kanban.py` | 675 | KanbanStore: init/load/add/get/update/delete/move/list/search/archive/stats/columns/notes. JSON backend. ASCII board renderer. CLI with 15 subcommands |
| `src/planner/sync.py` | 106 | sync_notes_to_board: title-matching, status->column, idempotent. CLI with --quiet |
| `src/planner/gui.py` | 973 | GuiServer (stdlib HTTP), GuiHandler (REST API), GUI_HTML (~500 lines inline SPA). Routes: /api/notes, /api/board, /api/sync, /api/tags, /api/stats |

### Tests (1,222 lines)

| File | Lines | Tests |
|------|-------|-------|
| `tests/test_kanban.py` | 321 | 32 tests: init, CRUD, move, columns, notes, archive, search, stats |
| `tests/test_core.py` | 335 | 27 tests: CRUD, search, today, export, tags, timeline, sprints. Parameterized file/mogdb |
| `tests/test_sync.py` | 103 | 8 tests: create, idempotent, status derivation, bidirectional |
| `tests/test_config.py` | 157 | 15 tests: root detection, path resolution, backend inference |
| `tests/test_gui.py` | 306 | 20 tests: HTTP API integration, port stepping |

### Config & Docs

| File | Lines | Content |
|------|-------|---------|
| `pyproject.toml` | 3 | setuptools>=64 build system |
| `README.md` | 134 | Full system docs: philosophy, commands, config, sync semantics |

---

## Web App: `apps/web/` (16 files)

### Pages (2,582 lines)

| File | Lines | Content |
|------|-------|---------|
| `app/(app)/kanban/page.tsx` | 617 | Board with drag-drop, search, tag filter, CreateCardDialog, CardEditor |
| `app/(app)/kanban/page.test.tsx` | 504 | 30 tests |
| `app/(app)/kanban/preview/page.tsx` | 27 | KanbanTray preview |
| `app/(app)/oon/page.tsx` | 762 | Duplicate board with sprint/gh/notes display |
| `app/(app)/oon/page.test.tsx` | 640 | 33 tests |

### API Routes (321 lines)

| File | Lines | Content |
|------|-------|---------|
| `app/api/kanban/route.ts` | 14 | Legacy GET, reads board.json (unused) |
| `app/api/oon/helpers.ts` | 207 | readBoard/writeBoard JSONL, card CRUD, tags, stats |
| `app/api/oon/board/route.ts` | 13 | GET board |
| `app/api/oon/board/move/route.ts` | 22 | POST move card |
| `app/api/oon/board/cards/route.ts` | 29 | POST create card |
| `app/api/oon/board/cards/[id]/route.ts` | 27 | PUT/DELETE card |
| `app/api/oon/tags/route.ts` | 13 | GET tags |

### Components (753 lines)

| File | Lines | Content |
|------|-------|---------|
| `components/kanban/CardEditor.tsx` | 702 | Slide-in editor, baby pastel theme, notes UI |
| `components/kanban/KanbanTray.tsx` | 19 | Tray wrapper |
| `components/kanban/types.ts` | 32 | KanbanColumn, KanbanCardData, KanbanBoard, TagCount |

### Client Library (117 lines)

| File | Lines | Content |
|------|-------|---------|
| `lib/planner-client.ts` | 117 | fetchBoard, moveCard, createCard, updateCard, deleteCard, fetchTags |

---

## Plans & Dev Notes

| File | Action |
|------|--------|
| `.opencode/plans/hash-tree-system.md` | Delete |
| `.opencode/plans/mogdb-js-wrapper.md` | Delete (if exists) |

---

## Orphaned Bytecode

| File | Action |
|------|--------|
| `packages/planner/src/planner/__pycache__/hashtree.cpython-312.pyc` | Delete |
| `packages/planner/tests/__pycache__/test_hashtree.cpython-312-pytest-9.1.1.pyc` | Delete |

---

## Files to EDIT (not delete)

| File | Change |
|------|--------|
| `apps/web/lib/navigation.ts` | Remove `/kanban` entry from tools section (line ~89), remove `'/kanban': IconGrid` from SIDEBAR_ICONS (line ~129) |
| `apps/web/cypress/e2e/visual-headers.cy.ts` | Remove `{ path: '/kanban', name: 'kanban' }` from PAGES array (line ~54) |

---

## Data Files (backup, don't delete)

| File | Action |
|------|--------|
| `.kanban/board.json` | Rename to `board.json.pre-rebuild` |
| `.kanban/board.jsonl` | Rename to `board.jsonl.pre-rebuild` |
| `.kanban/board.json.bak` | Keep |
| `.kanban/notes/` | Keep (empty) |
| `.dev-notes/store/` | Keep (source of truth) |
| `~/.config/dev-notes/*.md` | Keep (historical reference) |

---

## Key Format Notes for Rebuild

1. **Python**: `board.json` (single JSON). **Web**: `board.jsonl` (newline-delimited). **Incompatible.**
2. **Card IDs**: Python `YYYYMMDD_HHMMSS_slug` vs Web `card-{ts}-{rand}`
3. **Card fields**: Web has `sprint`, `gh` fields that Python KanbanStore doesn't natively support
4. **CardEditor.tsx** is shared between both pages
5. **planner-client.ts** routes everything through `/api/oon/*`
6. **The `/oon` route is hidden** — not in sidebar nav, not in Cypress tests
