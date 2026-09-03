# Plan: Rebuild Kanban as Planner + Notes App

## Context

The current planner/kanban/notes system has 2,800+ lines of duplicated code across two near-identical pages, two incompatible storage formats, and dead code (hash tree, legacy routes). We're starting fresh — deleting the current implementation and rebuilding it as a single, clean planner + notes app.

**Approach: Build web UI first** to see the architecture, then build the Python backend to match.

## Design Principles

1. **One page, not two** — merge `/kanban` and `/oon` into a single canonical `/planner` page
2. **One storage format** — JSONL everywhere (Python and web share the same format)
3. **Notes as source of truth** — the dev journal drives the board, not the other way around
4. **No dead code** — no hash tree, no legacy routes, no orphaned files
5. **Use the design system** — Noir Violet tokens, not hardcoded baby pastel colors
6. **Many buffets** — the building has multiple pages (buffets), connected by the hallway (sidebar)
7. **Engine pattern** — each buffet is a self-contained renderer, owns its own state and lifecycle

---

## Phase 1: Clean Slate

### Files to DELETE

**Python package** (`packages/planner/`):
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

**Web app** (`apps/web/`):
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

**Plans & dev notes**:
```
.opencode/plans/hash-tree-system.md
.opencode/plans/mogdb-js-wrapper.md (if exists)
```

**Orphaned bytecode**:
```
packages/planner/src/planner/__pycache__/hashtree.cpython-312.pyc
packages/planner/tests/__pycache__/test_hashtree.cpython-312-pytest-9.1.1.pyc
```

**Data files** (backup, don't delete):
```
.kanban/board.json      — rename to .kanban/board.json.pre-rebuild
.kanban/board.jsonl     — rename to .kanban/board.jsonl.pre-rebuild
.kanban/board.json.bak  — keep
.kanban/notes/          — keep
.dev-notes/store/       — keep (source of truth)
```

**Navigation** (`apps/web/lib/navigation.ts`):
- Remove `/kanban` entry from `nav.section.tools`
- Will be re-added as `/planner` in Phase 4

---

## Phase 2: Web App Rebuild (build first to see architecture)

### New structure

```
apps/web/
├── app/(app)/planner/
│   ├── page.tsx              — renders <BuffetEngine /> directly
│   └── page.test.tsx
├── app/api/planner/
│   ├── helpers.ts            — shared JSONL read/write
│   ├── notes/route.ts        — GET (list) + POST (create)
│   ├── notes/[id]/route.ts   — GET + PUT + DELETE
│   ├── board/route.ts        — GET (full board)
│   ├── board/cards/route.ts  — POST (create card)
│   ├── board/cards/[id]/route.ts — PUT + DELETE
│   ├── board/move/route.ts   — POST (move card)
│   ├── sync/route.ts         — POST (run sync)
│   ├── tags/route.ts         — GET
│   └── stats/route.ts        — GET
├── lib/
│   └── planner-client.ts     — browser-side fetch wrappers
└── components/
    └── planner/
        ├── BuffetEngine.tsx   — the engine (owns all state)
        ├── Scene.tsx          — renders the scene
        ├── Table.tsx          — board surface (grid of trays)
        ├── Tray.tsx           — column with slots
        ├── Slot.tsx           — position in tray (drop target)
        ├── CardItem.tsx       — draggable task card
        ├── CardEditor.tsx     — slide-in editor panel
        ├── NotesView.tsx      — notes panel (toggleable)
        ├── NoteEditor.tsx     — note create/edit form
        └── types.ts           — shared types
```

### Component scaffolding (from component-generation skill)

Each component follows this pattern:

```tsx
'use client'

import { cn, Button } from '@sloughgpt/strui'
import { IconSomething } from '@/components/icons/NavIcons'

interface MyComponentProps {
  // explicit props, no inline types
}

export function MyComponent({ ... }: MyComponentProps) {
  // hooks first
  // handlers with useCallback
  // render
}
```

### Page layout — many buffets, connected by hallway

```tsx
// app/(app)/planner/page.tsx
// The buffet engine renders directly — self-contained, but accessible via sidebar
export default function PlannerPage() {
  return <BuffetEngine />
}
```

The `BuffetEngine` component owns everything:
- Its own state (board, notes, UI state)
- Its own header (title, sync, new card)
- Its own toolbar (search, filters)
- Its own content (board grid or notes list)
- No `PageContainer` wrapper — the engine IS the page

The sidebar (hallway) connects this buffet to other buffets in the building.

### Design system compliance

- **Colors**: All via `rgb(var(--primary))` tokens, never hardcoded hex
- **Priority chips**: `bg-success/15 text-success` pattern (low=success, medium=warning, high=accent, critical=destructive)
- **Column headers**: `text-xs font-semibold uppercase tracking-wider text-muted-foreground`
- **Cards**: `rounded-lg border border-border bg-card shadow-sm hover:-translate-y-0.5 hover:shadow-md`
- **Focus rings**: `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring`
- **Drag feedback**: `opacity-40 scale-[0.98]` on dragged card

### Component details — BuffetEngine architecture

#### `BuffetEngine.tsx` (the engine — owns all state)
- Internal state: `scene` (board + notes), `input` (drag, hover, selected), `sync` (status)
- Engine methods: `init()`, `tick()`, `moveCard()`, `shuffleSlots()`, `sync()`
- Renders: header, toolbar, scene (table → trays → slots → cards)
- No props — fully self-contained

#### `Scene.tsx` (the rendered scene)
- Renders the table (grid of trays)
- Handles keyboard shortcuts, global events

#### `Table.tsx` (the board surface)
- 4-column grid: `grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4`
- Contains trays (columns)

#### `Tray.tsx` (a column)
- Props: `name, cards, wipLimit, onDrop`
- Drop zone: `onDragOver` / `onDrop` handlers
- Empty state: dashed border, "Drop cards here"

#### `Slot.tsx` (position in tray)
- Index in the column
- Drop target for cards
- Shufflable within tray

#### `CardItem.tsx` (a task)
- Draggable card with priority badge, tags (max 3 + overflow), assignee, due date
- Click to open CardEditor
- Keyboard: Enter/Space to open

#### `CardEditor.tsx` (slide-in editor)
- Slide-in panel from right
- Fields: title, description, priority, column, tags, assignee, due date, sprint, GitHub, notes
- Uses Noir Violet tokens (not baby pastel)
- Animated enter/exit

#### `NotesView.tsx` (notes panel)
- Toggleable panel within the engine
- List of notes with status badges, tags, search
- Click to open NoteEditor

#### `NoteEditor.tsx` (note form)
- Form: title, status, tags, sprint, GitHub, body (textarea)
- Create/edit modes

### API helpers (`helpers.ts`)

```typescript
// Reads/writes .kanban/board.jsonl (JSONL format)
export function readBoard(): Board
export function writeBoard(board: Board): void
export function moveCard(cardId: string, column: string): boolean
export function createCard(data: CreateCardData): BoardCard
export function updateCard(id: string, data: Partial<BoardCard>): BoardCard | null
export function deleteCard(id: string): boolean
export function getAllTags(): TagCount[]
export function getStats(): Stats

// Reads/writes .dev-notes/store/notes.journal.jsonl
export function listNotes(filters?: NoteFilters): Note[]
export function createNote(data: CreateNoteData): Note
export function updateNote(id: string, data: Partial<Note>): Note | null
export function deleteNote(id: string): boolean
```

### Client library (`planner-client.ts`)

```typescript
// All calls prefixed with /api/planner
export function fetchBoard(): Promise<{ board: Board }>
export function moveCard(payload: { card_id: string; column: string }): Promise<void>
export function createCard(payload: CreateCardData): Promise<{ card: BoardCard }>
export function updateCard(id: string, payload: Partial<BoardCard>): Promise<{ card: BoardCard }>
export function deleteCard(id: string): Promise<void>
export function fetchTags(): Promise<{ tags: TagCount[] }>
export function fetchNotes(filters?: NoteFilters): Promise<{ notes: Note[] }>
export function createNote(payload: CreateNoteData): Promise<{ note: Note }>
export function updateNote(id: string, payload: Partial<Note>): Promise<{ note: Note }>
export function deleteNote(id: string): Promise<void>
export function syncNotes(): Promise<{ added: number; updated: number; total: number }>
```

### Tests (~30 target)

Following the test pattern from component-generation skill:
```tsx
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('next/navigation', () => ({ usePathname: () => '/planner' }))
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))
```

Test cases:
- Rendering columns and cards
- Create/edit/delete card flows
- Drag-and-drop (data transfer mock)
- Search and tag filtering
- Notes tab: list, create, edit, delete
- Error handling and loading states

---

## Phase 3: Python Package Rebuild

### New structure: `packages/planner/`

```
packages/planner/
├── pyproject.toml
├── README.md
├── src/planner/
│   ├── __init__.py
│   ├── config.py        — directory resolution, status mappings
│   ├── store.py         — unified NoteStore + BoardStore (JSONL backend)
│   ├── sync.py          — notes ↔ board reconciliation
│   └── cli.py           — single CLI entry point
└── tests/
    ├── test_store.py
    ├── test_sync.py
    └── test_cli.py
```

### Key changes from current

| Current | New |
|---------|-----|
| `core.py` (816 lines) + `kanban.py` (675 lines) | `store.py` (~400 lines) — unified store |
| `gui.py` (973 lines) — embedded SPA | **Removed** — web UI handles this |
| `config.py` (120 lines) | `config.py` (~80 lines) — simplified |
| `sync.py` (106 lines) — one-way | `sync.py` (~80 lines) — bidirectional |
| CLI dispatches to 4 subcommands | Single `cli.py` with subcommands |
| `board.json` (single JSON) | `board.jsonl` (JSONL, same as web) |
| File backend + MogDB backend | JSONL only (no MogDB dependency) |

### Store API

```python
class PlannerStore:
    def __init__(self, data_dir: Path):
        self.notes_path = data_dir / "notes.journal.jsonl"
        self.board_path = data_dir / "board.jsonl"

    # Notes
    def create_note(self, title, tags, status, sprint, gh, body) -> Note
    def get_note(self, note_id) -> Note | None
    def update_note(self, note_id, **kwargs) -> Note | None
    def delete_note(self, note_id) -> bool
    def list_notes(self, tag, status, sprint, limit) -> list[Note]
    def search_notes(self, query, limit) -> list[Note]

    # Board
    def load_board(self) -> Board
    def add_card(self, title, column, priority, ...) -> Card
    def get_card(self, card_id) -> Card | None
    def update_card(self, card_id, **kwargs) -> Card | None
    def delete_card(self, card_id) -> bool
    def move_card(self, card_id, column) -> Card | None
    def list_cards(self, column, priority, tag) -> list[Card]
    def search_cards(self, query) -> list[Card]

    # Sync
    def sync(self) -> tuple[int, int, int]

    # Stats
    def stats(self) -> dict
    def tags(self) -> list[dict]
```

### Tests (~60 target)

- `test_store.py` — Note CRUD, Card CRUD, Board operations, search, tags
- `test_sync.py` — Bidirectional sync, idempotency, conflict resolution
- `test_cli.py` — CLI command dispatch

---

## Phase 4: Navigation & Cleanup

1. Add `/planner` to sidebar navigation under "Tools" with `IconGrid`
2. Remove `/kanban` and `/oon` from sidebar navigation
3. The planner buffet is a self-contained engine — sidebar is how you navigate TO it
4. Update Cypress visual tests to include `/planner`
5. Update AGENTS.md with new architecture
6. Clean up dev notes (keep as historical reference, don't sync)

---

## Verification

1. `cd apps/web && npx tsc --noEmit` — no type errors
2. `cd apps/web && npx vitest run` — all frontend tests pass
3. Manual: start dev server, open `/planner`, test board + notes views
4. `cd packages/planner && python -m pytest tests/ -v` — all Python tests pass
5. Verify sync: create a note with status=wip, run sync, card appears in in_progress

---

## Implementation Order

1. Delete all files listed in Phase 1
2. Build BuffetEngine + scene components (Phase 2)
3. Build API routes (Phase 2)
4. Build Python package (Phase 3)
5. Navigation + cleanup (Phase 4)
