# app-planner

Unified notes + kanban CLI. One entry point (`app-planner`) with subcommands for
both notes and board operations. Note mutations auto-sync to the kanban board by
default. Pure Python stdlib: no external dependencies, no servers.

## Philosophy

- **One CLI.** Notes and kanban share one command tree under `app_planner.cli`.
  `planner`, `notes`, `kanban`, and `sync-notes-to-board` all resolve to the
  same entry point.
- **One source of truth.** Data locations, storage backend, and the
  status <-> column mapping live in `app_planner/config.py`, not scattered across
  commands. Every tool reads the same answer.
- **Repo-aware by default.** Run any command from inside a repo and it finds
  the repo's own `.dev-notes/` and `.kanban/` (version-controlled alongside
  the code). Outside a repo it falls back to `~/.config/`.
- **Backend is inferred.** A notes directory that already contains a MogDB
  journal (`store/notes.journal.jsonl`) is opened as `mogdb`; anything else is
  plain `file`. `APP_PLANNER_BACKEND` overrides when set.
- **Status and column are the same fact.** Notes store a status; cards live in
  a column. `sync` reconciles both directions — missing cards are created and
  out-of-step cards are moved. Use `--no-sync` to skip auto-sync on mutations.

## Install

```bash
pip install -e packages/app-planner        # editable, console scripts auto-installed
```

## Quick start

```bash
# Notes
app-planner new "Fix boot order" --tags kernel,os --status wip
app-planner list --today
app-planner show <short-id>
app-planner edit <short-id> --status done --body "Completed: ..."
app-planner search "keyword"

# Kanban
app-planner board                          # ASCII board, grouped by column
app-planner add "Ship v2.0" --column in_progress --tags release
app-planner move <card-id> done

# Sync notes to board explicitly
app-planner sync

# Local web UI (stdlib HTTP server, embedded SPA)
app-planner gui
```

## Commands

| Command | Purpose |
|---------|---------|
| `new` | Create a note |
| `list` | List notes |
| `show <id>` | Show a note |
| `edit <id>` | Edit a note |
| `delete` / `rm` | Delete a note |
| `search <query>` | Search notes |
| `today` | Show today's notes |
| `export` | Export all notes |
| `tags` | List all tags |
| `status` | Status summary |
| `timeline` | Show notes grouped by day |
| `sprint <name>` | Sprint operations |
| `init` | Initialize a new board |
| `add` | Add a card |
| `cards` | List cards |
| `card <id>` | Show card details |
| `edit-card <id>` | Edit a card |
| `move <id> <column>` | Move card to another column |
| `block <card> <blocker>` | Block a card by another card |
| `unblock <card> <blocker>` | Unblock a card |
| `blocked` | List blocked cards |
| `delete-card` / `rm-card` | Delete a card |
| `board` | Show ASCII kanban board |
| `note add <card> <text>` | Add note to a card |
| `note list <card>` | List notes on a card |
| `note delete <card> <id>` | Delete note from a card |
| `columns` | List columns |
| `column-add <name>` | Add a column |
| `column-rename <old> <new>` | Rename a column |
| `column-rm <name>` | Remove a column |
| `archive` | Archive (delete) all done cards |
| `search-cards <query>` | Search cards |
| `stats` | Board statistics |
| `export-board` | Export board to JSON |
| `import-board <file>` | Import board from JSON |
| `sync` | Sync notes to board |
| `gui` | Launch web GUI |

Global options: `--backend {file,mogdb}`, `--notes-dir`, `--board-dir`, `--no-sync`

## Configuration

Resolution order, first match wins:

| Priority | Source |
|----------|--------|
| 1 | Explicit CLI flag (`--notes-dir`, `--board-dir`, `--backend`) |
| 2 | Environment: `APP_PLANNER_NOTES_DIR`, `APP_PLANNER_BOARD_DIR`, `APP_PLANNER_BACKEND` |
| 3 | Repo root: nearest ancestor containing `.kanban/board.json` -> `<root>/.dev-notes`, `<root>/.kanban` |
| 4 | User config fallback: `~/.config/dev-notes`, `~/.config/kanban` |

## Status <-> column mapping

| Note status | Board column |
|-------------|--------------|
| `open`      | `todo`       |
| `wip`       | `in_progress`|
| `review`    | `review`     |
| `done`      | `done`       |
| `blocked`   | `todo`       |

Mapping lives in `app_planner.config.STATUS_TO_COLUMN` / `COLUMN_TO_STATUS`.
Board columns define the inverse: `todo -> open`, `in_progress -> wip`,
`review -> review`, `done -> done`. Moving a card in the GUI updates the
matching note's status and vice versa.

## Sync semantics

`app_planner.sync.sync_notes_to_board(note_store, kanban_store)`:

1. A note matches a card by title (case-sensitive).
2. Notes without a card get one, placed in the column for their status.
3. Existing cards whose column differs from the note's status are **moved** —
   this is what keeps the board correct when a note is marked `done`.
4. Returns `(added, updated, total)`; repeated runs are idempotent.

```python
from app_planner.core import NoteStore
from app_planner.kanban import KanbanStore
from app_planner.sync import sync_notes_to_board

notes = NoteStore(notes_dir="<root>/.dev-notes", backend="mogdb")
board = KanbanStore(board_dir="<root>/.kanban")
added, updated, total = sync_notes_to_board(notes, board)
```

## Storage

- **`file`** — one markdown file per note under `<notes-dir>/`.
- **`mogdb`** — append-only journal `store/notes.journal.jsonl` in the notes
  directory, replayed on load (the project's own document DB, in `packages/mogdb`).

## Web GUI

`app-planner gui` serves a zero-dependency single-page app on `127.0.0.1:8787`
(`ThreadingHTTPServer` + embedded HTML/CSS/JS, no external assets). If the
requested port is taken it steps to the next free port and reports the change;
`--port 0` asks the kernel for an ephemeral port.

- `GET  /api/notes`, `/api/notes/{id}`, `/api/board`, `/api/tags`, `/api/stats`
- `POST /api/notes`, `/api/board/move`, `/api/sync`
- `PUT`/`DELETE /api/notes/{id}`

## Development

```bash
cd packages/app-planner
PYTHONPATH=packages/app-planner/src:packages/mogdb/src python3 -m pytest tests/ -p no:anyio
```

The package is installed editable (`__editable__.app-planner-*.pth` points at
`packages/app-planner/src`), so source edits take effect immediately; reinstall
only regenerates the console scripts.
