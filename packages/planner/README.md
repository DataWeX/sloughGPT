# planner

Local-first notes + kanban toolchain, delivered as one CLI. Four commands —
`planner`, `notes`, `kanban`, `sync-notes-to-board` — share a single config
module (`planner.config`) so they always resolve the same stores and stay in
step with each other. Pure Python stdlib: no external dependencies, no servers.

## Philosophy

- **One source of truth.** Data locations, storage backend, and the
  status <-> column mapping live in `planner/config.py`, not scattered across
  commands. Every tool reads the same answer.
- **Repo-aware by default.** Run any command from inside a repo and it finds
  the repo's own `.dev-notes/` and `.kanban/` (version-controlled alongside
  the code). Outside a repo it falls back to `~/.config/`.
- **Backend is inferred.** A notes directory that already contains a MogDB
  journal (`store/notes.journal.jsonl`) is opened as `mogdb`; anything else is
  plain `file`. `PLANNER_BACKEND` overrides when set.
- **Status and column are the same fact.** Notes store a status; cards live in
  a column. `sync` reconciles both directions — missing cards are created and
  out-of-step cards are moved.

## Install

```bash
pip install -e packages/planner        # editable, console scripts auto-installed
```

## Quick start

```bash
# Notes
notes new "Fix boot order" --tags kernel,os --status wip
notes list --today
notes show <short-id>
notes edit <short-id> --status done --body "Completed: ..."
notes search "keyword"

# Kanban
kanban board                          # ASCII board, grouped by column
kanban add "Ship v2.0" --column in_progress --tags release
kanban move <card-id> done

# Keep notes and board in sync (adds missing cards, moves stale columns)
sync-notes-to-board

# Local web UI (stdlib HTTP server, embedded SPA)
planner gui
```

## Commands

| Command | Entry point | Purpose |
|---------|-------------|---------|
| `planner` | `planner.core:cli_main` | Notes CLI. Subcommands: `new`, `list`, `show`, `edit`, `delete`/`rm`, `search`, `today`, `export`, `tags`, `status`, `timeline`, `sprint` |
| `planner gui` | `planner.gui:main` | Local web interface for notes + board (`--host --port --no-open --sync`) |
| `planner sync` | `planner.sync:cli_main` | Sync notes -> board (`--quiet` for one-line summary) |
| `notes` | `planner.core:cli_main` | Alias for `planner` |
| `kanban` | `planner.kanban:cli_main` | Board CLI. Subcommands: `init`, `add`, `list`, `show`, `edit`, `move`, `delete`, `board`, `note`, `columns`, `column-add/rename/rm`, `archive`, `search`, `stats` |
| `sync-notes-to-board` | `planner.sync:cli_main` | Alias for `planner sync` |

## Configuration

Resolution order, first match wins:

| Priority | Source |
|----------|--------|
| 1 | Explicit CLI flag (`--notes-dir`, `--board-dir`, `--backend`) |
| 2 | Environment: `PLANNER_NOTES_DIR`, `PLANNER_BOARD_DIR`, `PLANNER_BACKEND` |
| 3 | Repo root: nearest ancestor containing `.kanban/board.json` -> `<root>/.dev-notes`, `<root>/.kanban` |
| 4 | User config: `~/.config/dev-notes`, `~/.config/kanban` |

## Status <-> column mapping

| Note status | Board column |
|-------------|--------------|
| `open`      | `todo`       |
| `wip`       | `in_progress`|
| `review`    | `review`     |
| `done`      | `done`       |
| `blocked`   | `todo`       |

Mapping lives in `planner.config.STATUS_TO_COLUMN` / `COLUMN_TO_STATUS`.
Board columns define the inverse: `todo -> open`, `in_progress -> wip`,
`review -> review`, `done -> done`. Moving a card in the GUI updates the
matching note's status and vice versa.

## Sync semantics

`planner.sync.sync_notes_to_board(note_store, kanban_store)`:

1. A note matches a card by title (case-sensitive).
2. Notes without a card get one, placed in the column for their status.
3. Existing cards whose column differs from the note's status are **moved** —
   this is what keeps the board correct when a note is marked `done`.
4. Returns `(added, updated, total)`; repeated runs are idempotent.

```python
from planner.core import NoteStore
from planner.kanban import KanbanStore
from planner.sync import sync_notes_to_board

notes = NoteStore(notes_dir="<root>/.dev-notes", backend="mogdb")
board = KanbanStore(board_dir="<root>/.kanban")
added, updated, total = sync_notes_to_board(notes, board)
```

## Storage

- **`file`** — one markdown file per note under `<notes-dir>/`.
- **`mogdb`** — append-only journal `store/notes.journal.jsonl` in the notes
  directory, replayed on load (the project's own document DB, in `packages/mogdb`).

## Web GUI

`planner gui` serves a zero-dependency single-page app on `127.0.0.1:8787`
(`ThreadingHTTPServer` + embedded HTML/CSS/JS, no external assets). If the
requested port is taken it steps to the next free port and reports the change;
`--port 0` asks the kernel for an ephemeral port.

- `GET  /api/notes`, `/api/notes/{id}`, `/api/board`, `/api/tags`, `/api/stats`
- `POST /api/notes`, `/api/board/move`, `/api/sync`
- `PUT`/`DELETE /api/notes/{id}`

## Development

```bash
cd packages/planner
python3 -m pytest tests/        # 60 tests: config (16), sync (8), gui (36)
```

The package is installed editable (`__editable__.planner-*.pth` points at
`packages/planner/src`), so source edits take effect immediately; reinstall
only regenerates the console scripts.
