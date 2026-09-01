# Specs: Current Planner / Kanban / Notes System

## Purpose

A self-contained project management system with two subsystems:
1. **Notes** — a developer journal (markdown files with YAML frontmatter)
2. **Kanban** — a task board (cards in columns with drag-and-drop)

Notes are the source of truth. The board is a derived view. Sync bridges them.

---

## Data Model

### Note
```
id:         string    (YYYYMMDD_HHMMSS_slug)
title:      string
created_at: datetime
updated_at: datetime
tags:       string[]  (comma-separated)
status:     string    (open | wip | done | blocked | review | todo)
sprint:     string
gh:         string    (GitHub issue ref)
body:       string    (markdown content)
```

Storage: markdown files with YAML frontmatter in `.dev-notes/` (file backend) or MogDB journal (mogdb backend).

### Card
```
id:          string    (YYYYMMDD_HHMMSS_slug or card-{ts}-{rand})
title:       string
description: string
column:      string    (todo | in_progress | review | done)
priority:    string    (low | medium | high | critical)
tags:        string[]
due_date:    string
assignee:    string
sprint:      string
gh:          string
notes:       { id, text, author, created_at }[]
created_at:  datetime
updated_at:  datetime
```

Storage: `board.json` (Python single JSON) or `board.jsonl` (web newline-delimited JSON).

### Column
```
name:      string
wip_limit: number
order:     number
```

Default columns: todo (WIP 5), in_progress (WIP 3), review (WIP 2), done (WIP 0).

### Status-Column Mapping
```
open   -> todo
wip    -> in_progress
review -> review
done   -> done
blocked -> todo
todo   -> todo
```

---

## Python Package: `packages/planner/`

### Modules

| Module | Lines | Role |
|--------|-------|------|
| `config.py` | 120 | Directory resolution, status/column mappings |
| `core.py` | 816 | NoteStore (create/get/update/delete/list/search/export/timeline/sprints) |
| `kanban.py` | 675 | KanbanStore (init/load/add/get/update/delete/move/list/search/archive/stats/columns/notes) |
| `sync.py` | 106 | sync_notes_to_board (create missing cards, move cards to match note status) |
| `gui.py` | 973 | Local web GUI (stdlib HTTP server, embedded SPA, REST API) |
| `__init__.py` | 11 | Re-exports |
| `__main__.py` | 13 | CLI dispatcher |

### CLI Commands
- `planner kanban <subcommand>` — board operations
- `planner notes <subcommand>` — note operations
- `planner sync` — notes -> board sync
- `planner gui` — start local web server (port 8787)

### Tests
- `test_kanban.py` — 32 tests
- `test_core.py` — 27 tests (parameterized over file/mogdb)
- `test_sync.py` — 8 tests
- `test_config.py` — 15 tests
- `test_gui.py` — 20 tests
- **Total: 102 tests**

---

## Web App: `apps/web/`

### Routes

| Path | Component | Lines | Notes |
|------|-----------|-------|-------|
| `/kanban` | `kanban/page.tsx` | 617 | Main board (in sidebar nav) |
| `/oon` | `oon/page.tsx` | 762 | Duplicate with extra fields (NOT in nav) |

### API Routes (all under `/api/oon/`)

| Endpoint | Method | Handler | Lines |
|----------|--------|---------|-------|
| `/api/oon/board` | GET | `board/route.ts` | 13 |
| `/api/oon/board/move` | POST | `board/move/route.ts` | 22 |
| `/api/oon/board/cards` | POST | `board/cards/route.ts` | 29 |
| `/api/oon/board/cards/[id]` | PUT/DELETE | `board/cards/[id]/route.ts` | 27 |
| `/api/oon/tags` | GET | `tags/route.ts` | 13 |

### Legacy Route
| Endpoint | Method | Handler | Notes |
|----------|--------|---------|-------|
| `/api/kanban` | GET | `api/kanban/route.ts` | Reads board.json, unused by UI |

### Client Library
- `lib/planner-client.ts` — 117 lines, fetch wrappers for `/api/oon/*`

### Components
- `components/kanban/CardEditor.tsx` — 702 lines, slide-in editor with baby pastel theme
- `components/kanban/KanbanTray.tsx` — 19 lines, tray wrapper
- `components/kanban/types.ts` — 32 lines, shared types

### Tests
- `kanban/page.test.tsx` — 30 tests
- `oon/page.test.tsx` — 33 tests
- **Total: 63 frontend tests**

---

## Known Problems

1. **Two duplicated pages** — `/kanban` (617 lines) and `/oon` (762 lines) are ~90% identical. Neither is canonical.
2. **Dual storage formats** — Python uses `board.json`, web uses `board.jsonl`. Not interoperable.
3. **Card ID formats differ** — Python: `YYYYMMDD_HHMMSS_slug`. Web: `card-{ts}-{rand}`.
4. **Sync is one-way** — Notes -> board only. The AGENTS.md claims bidirectional but code only pushes note status to cards.
5. **No notes UI in web** — CardEditor has a notes section but no API routes to persist card notes.
6. **The `/oon` route is hidden** — Not in sidebar navigation, not in visual tests.
7. **Legacy `/api/kanban` route** — Reads `board.json` directly, not used by any frontend.
8. **Hash tree was never implemented** — Plan exists, source files were deleted, orphaned .pyc remains.

---

## File Inventory (what exists today)

### Python package (`packages/planner/`)
```
src/planner/__init__.py
src/planner/__main__.py
src/planner/config.py
src/planner/core.py
src/planner/kanban.py
src/planner/sync.py
src/planner/gui.py
tests/test_kanban.py
tests/test_core.py
tests/test_sync.py
tests/test_config.py
tests/test_gui.py
pyproject.toml
README.md
```

### Web app (`apps/web/`)
```
app/(app)/kanban/page.tsx
app/(app)/kanban/page.test.tsx
app/(app)/kanban/preview/page.tsx
app/(app)/oon/page.tsx
app/(app)/oon/page.test.tsx
app/api/kanban/route.ts
app/api/oon/helpers.ts
app/api/oon/board/route.ts
app/api/oon/board/move/route.ts
app/api/oon/board/cards/route.ts
app/api/oon/board/cards/[id]/route.ts
app/api/oon/tags/route.ts
components/kanban/CardEditor.tsx
components/kanban/KanbanTray.tsx
components/kanban/types.ts
lib/planner-client.ts
```

### Data files
```
.kanban/board.json
.kanban/board.jsonl
.kanban/board.json.bak
.kanban/notes/ (empty)
.dev-notes/store/notes.journal.jsonl
```

### Plans & dev notes
```
.opencode/plans/hash-tree-system.md
~/.config/dev-notes/2026-08-31-full-recovered-conversation.md
~/.config/dev-notes/2026-08-31-hash-tree-implementation-plan.md
~/.config/dev-notes/2026-08-31-planner-session.md
~/.config/dev-notes/2026-08-31-workflow-notes-to-board.md
~/.config/dev-notes/2026-08-31-cli-suggestions.md
~/.config/dev-notes/2026-08-31-serve-inplace-updates.md
~/.config/dev-notes/2026-08-31-tui-linux-fixes.md
```

### Orphaned bytecode
```
packages/planner/src/planner/__pycache__/hashtree.cpython-312.pyc
packages/planner/tests/__pycache__/test_hashtree.cpython-312-pytest-9.1.1.pyc
```
