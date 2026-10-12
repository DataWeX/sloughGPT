# planner

Unified notes + kanban CLI. **One app, one command**: `planner` is the single
entry point, with `notes` and `board` subcommand groups under it. Note mutations
auto-sync to the kanban board by default. Pure Python stdlib: no external
dependencies, no servers.

`app-planner` and `kanban` are installed as aliases of the same program, so
either spelling resolves here.

## Philosophy

- **One CLI.** Notes and kanban share one command tree under `app_planner.cli`.
  The spec (`.opencode/plans/planner-notes-specs.md`) fixes the shape as
  `planner notes | kanban | sync | gui`, and that is what ships.
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
# Notes  (group: planner note  — alias `notes`)
planner note new "Fix boot order" --tags kernel,os --status wip
planner note list --status wip
planner note show <short-id>
planner note update <short-id> --status done
planner note search "keyword"

# Kanban (group: planner board — alias `kanban`)
planner board                       # ASCII board, grouped by column
planner board add "Ship v2.0" --column in_progress --tags release
planner board move <card-id> done
planner board stats

# Sync notes to board explicitly
planner sync

# Three-way diff: note frontmatter (status/board/landed) <-> board card <-> git
planner verify card092 --repo /path/to/repo   # exit 1 on any drift

# Local web UI (stdlib HTTP server, embedded SPA)
planner gui

# Report (or, with --apply, remove) card lines sharing one card id
planner board dedupe
```

## Commands

Top level: `planner` — plus the `app-planner` and `kanban` aliases.

Global options: `--notes-dir`, `--board-dir`

### `planner note` (alias `notes`)

| Command | Purpose |
|---------|---------|
| `new <title>` | Create a note (`--tags`, `--status`, `--body`, `--sprint`, `--gh`, `--author`) |
| `list` / `ls` | List notes (`--tag`, `--status`, `--limit`) |
| `show <id>` | Show a note |
| `update <id>` | Edit a note (`--title`, `--status`, `--tags`, `--body`) |
| `delete <id>` / `rm <id>` | Delete a note |
| `search <query>` | Search notes |

### `planner board` (alias `kanban`)

| Command | Purpose |
|---------|---------|
| `show [id ...]` / `ls` | Show the board or specific card(s) |
| `add <title>` | Add a card (`--column`, `--priority`, `--description`, `--tags`) |
| `move <id> <column>` | Move card to another column |
| `delete <id>` / `rm <id>` | Delete a card |
| `tags` | List tags |
| `stats` | Board statistics |
| `dedupe [--apply]` | Report duplicate card-id lines; `--apply` removes them |

### Top-level commands

| Command | Purpose |
|---------|---------|
| `sync [--quiet]` | Sync notes to board |
| `verify <note>` | Three-way diff: note frontmatter ↔ board card ↔ landed git shas (exit 1 on drift). `--repo`, `--board-file` |
| `gui` | Launch web GUI. `--host`, `--port`, `--backend` |

Environment: `APP_PLANNER_NOTES_DIR`, `APP_PLANNER_BOARD_DIR`, `APP_PLANNER_BACKEND`, `APP_PLANNER_NO_SYNC` (set to any value to disable auto-sync)

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

`planner gui` serves a zero-dependency single-page app on `127.0.0.1:8787`
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
