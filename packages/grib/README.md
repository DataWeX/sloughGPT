# grib

Unified notes + kanban CLI. One entry point (`grib`) with subcommands for
both notes and board operations. Note mutations auto-sync to the kanban board by
default. Pure Python stdlib: no external dependencies, no servers.

## Philosophy

- **One CLI.** Notes and kanban share one command tree under `grib.cli`;
  `grib` is the single entry point, with `notes`/`kanban` as aliases of the
  `note`/`board` subcommands.
- **One source of truth.** Data locations, storage backend, and the
  status <-> column mapping live in `grib/config.py`, not scattered across
  commands. Every tool reads the same answer.
- **Repo-aware by default.** Run any command from inside a repo and it finds
  the repo's own `.dev-notes/` and `.kanban/` (version-controlled alongside
  the code). Outside a repo it falls back to `~/.config/`.
- **Backend is inferred.** A notes directory that already contains a MogDB
  journal (`store/notes.journal.jsonl`) is opened as `mogdb`; anything else is
  plain `file`. `GRIB_BACKEND` overrides when set.
- **Status and column are the same fact.** Notes store a status; cards live in
  a column. `sync` reconciles both directions — missing cards are created and
  out-of-step cards are moved.

## Install

```bash
# grib + the planner engine it imports (grib board/gui validation runs
# through planner.engine). Install planner explicitly — never add it as a
# grib pyproject dependency: the PyPI name `planner` is an unrelated
# package.
pip install -e packages/grib -e packages/planner
```

## Quick start

```bash
# Notes
grib note new "Fix boot order" --tags kernel,os --status wip
grib note list
grib note show <short-id>
grib note update <short-id> --status done --body "Completed: ..."
grib note search "keyword"

# Kanban
grib board show                      # ASCII board, grouped by column
grib board add "Ship v2.0" --column wip --tags release
grib board move <card-id> done

# Sync notes to board explicitly
grib sync

# Three-way diff: note frontmatter (status/board/landed) <-> board card <-> git
grib verify <note> --repo /path/to/repo   # exit 1 on any drift

# Local web UI (stdlib HTTP server, embedded SPA)
grib gui
```

## Commands

| Command                               | Purpose                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| `note new <title>`                    | Create a note (`--tags`, `--status`, `--body`, `--sprint`, `--gh`)              |
| `note list`                           | List notes (`--tag`, `--status`, `--limit`)                                     |
| `note show <id>`                      | Show a note                                                                    |
| `note update <id>`                    | Update a note (`--title`, `--status`, `--tags`, `--body`)                       |
| `note delete <id>`                    | Delete a note (alias `rm`)                                                     |
| `note search <query>`                 | Search notes                                                                   |
| `board show`                          | Show the ASCII board; `board show <id>` shows one card                          |
| `board add <title>`                   | Add a card (`--column`, `--priority`, `--description`, `--tags`)                |
| `board move <id> <column>`            | Move card to another column                                                    |
| `board delete <id>`                   | Delete a card (alias `rm`)                                                     |
| `board tags`                          | List tags                                                                      |
| `board stats`                         | Board statistics                                                               |
| `board index`                         | Rebuild the read-only MogDB query index                                        |
| `board query`                         | Filter cards via the index (file fallback)                                     |
| `sync`                                | Sync notes to board                                                            |
| `verify <note>`                       | Three-way diff: note frontmatter ↔ board card ↔ landed git shas (exit 1 on drift) |
| `migrate`                             | Import repo-local `.kanban`/`.dev-notes` into the canonical store              |
| `snapshot`                            | Copy the canonical board into `<repo>/.kanban/snapshots/`                       |
| `export`                              | Mirror canonical board + notes into the repo checkout                          |
| `gui`                                 | Launch the web GUI (`--backend {file,mogdb}`, `--port`)                         |

Aliases: `notes` → `note`, `kanban` → `board`.

Global options: `--notes-dir`, `--board-dir` (`--backend {file,mogdb}` belongs to `gui`)

Environment: `GRIB_NOTES_DIR`, `GRIB_BOARD_DIR`, `GRIB_BACKEND`

## Configuration

Resolution order, first match wins:

| Priority | Source                                                                                               |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 1        | Explicit CLI flag (`--notes-dir`, `--board-dir`, `--backend`)                                        |
| 2        | Environment: `GRIB_NOTES_DIR`, `GRIB_BOARD_DIR`, `GRIB_BACKEND`                                      |
| 3        | Repo root: nearest ancestor containing `.kanban/board.json` -> `<root>/.dev-notes`, `<root>/.kanban` |
| 4        | User config fallback: `~/.config/dev-notes`, `~/.config/kanban`                                      |

## Status <-> column mapping

| Note status | Board column |
| ----------- | ------------ |
| `open`      | `todo`       |
| `wip`       | `wip`        |
| `review`    | `review`     |
| `done`      | `done`       |
| `blocked`   | `todo`       |

Mapping lives in `grib.config.STATUS_TO_COLUMN` / `COLUMN_TO_STATUS`.
Board columns define the inverse: `todo -> open`, `wip -> wip`,
`review -> review`, `done -> done`. Moving a card in the GUI updates the
matching note's status and vice versa.

## Sync semantics

`grib.sync.sync_notes_to_board(note_store, kanban_store)`:

1. A note matches a card by title (case-sensitive).
2. Notes without a card get one, placed in the column for their status.
3. Existing cards whose column differs from the note's status are **moved** —
   this is what keeps the board correct when a note is marked `done`.
4. Returns `(added, updated, total)`; repeated runs are idempotent.

```python
from grib.core import NoteStore
from grib.kanban import KanbanStore
from grib.sync import sync_notes_to_board

notes = NoteStore(notes_dir="<root>/.dev-notes", backend="mogdb")
board = KanbanStore(board_dir="<root>/.kanban")
added, updated, total = sync_notes_to_board(notes, board)
```

## Storage

- **`file`** — one markdown file per note under `<notes-dir>/`.
- **`mogdb`** — append-only journal `store/notes.journal.jsonl` in the notes
  directory, replayed on load (the project's own document DB, in `packages/mogdb`).

## Web GUI

`grib gui` serves a zero-dependency single-page app on `127.0.0.1:8787`
(`ThreadingHTTPServer` + embedded HTML/CSS/JS, no external assets). If the
requested port is taken it steps to the next free port and reports the change;
`--port 0` asks the kernel for an ephemeral port.

- `GET  /api/notes`, `/api/notes/{id}`, `/api/board`, `/api/tags`, `/api/stats`
- `POST /api/notes`, `/api/board/move`, `/api/sync`
- `PUT`/`DELETE /api/notes/{id}`

## Development

```bash
cd packages/grib
PYTHONPATH=packages/grib/src:packages/mogdb/src python3 -m pytest tests/ -p no:anyio
```

The package is installed editable (`__editable__.grib-*.pth` points at
`packages/grib/src`), so source edits take effect immediately; reinstall
only regenerates the console scripts.
