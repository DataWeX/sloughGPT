"""
Shared planner configuration — single source of truth for data locations,
storage backend, and status <-> column mappings used by every tool
(``planner``, ``notes``, ``kanban``, ``planner gui``, ``planner sync``).

Resolution order (first match wins):
    1. Explicit CLI flag (per-command).
    2. Environment variable (``PLANNER_NOTES_DIR``, ``PLANNER_BOARD_DIR``,
       ``PLANNER_BACKEND``).
    3. Repository detection: walking up from the current directory, the
       first ancestor containing ``.kanban/board.json`` is treated as the
       project root, giving ``<root>/.dev-notes`` and ``<root>/.kanban``.
    4. User config fallback (``~/.config/dev-notes`` and
       ``~/.config/kanban``).
"""

from __future__ import annotations

import os
from pathlib import Path

NOTES_FALLBACK = Path.home() / ".config" / "dev-notes"
BOARD_FALLBACK = Path.home() / ".config" / "kanban"
BACKENDS = ("file", "mogdb")

STATUS_TO_COLUMN = {
    "done": "done",
    "wip": "in_progress",
    "review": "review",
    "todo": "todo",
    "open": "todo",
    "blocked": "todo",
    "": "todo",
    None: "todo",
}

COLUMN_TO_STATUS = {
    "todo": "open",
    "in_progress": "wip",
    "review": "review",
    "done": "done",
}

STATUSES = ["open", "wip", "done", "blocked", "review", "todo"]

STATUS_ICONS = {
    "open": "\u25cb",
    "wip": "\u25d0",
    "done": "\u25cf",
    "blocked": "\u2715",
    "review": "\u25c8",
    "todo": "\u25cb",
}


def _walk_for_board(start: Path) -> Path | None:
    """Return the nearest ancestor of *start* containing ``.kanban/board.json``."""
    cur = start.resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / ".kanban" / "board.json").is_file():
            return candidate
    return None


def find_project_root(start: Path | None = None) -> Path | None:
    """Return the nearest project root that contains a kanban board.

    With an explicit *start*, only *start* and its ancestors are searched.
    Without one, the current directory is searched first, then the directory
    tree of the installed ``planner`` package itself. The package fallback
    keeps every planner tool (``notes``, ``kanban``, ``planner gui``) pointed
    at the repository's board even when launched from a directory outside the
    repo (the package is typically an editable install living inside it).
    Returns ``None`` when no ancestor qualifies.
    """
    if start is not None:
        return _walk_for_board(start)
    root = _walk_for_board(Path(os.getcwd()))
    if root is not None:
        return root
    return _walk_for_board(Path(__file__))


def project_notes_dir(root: Path) -> Path:
    return root / ".dev-notes"


def project_board_dir(root: Path) -> Path:
    return root / ".kanban"


def default_notes_dir() -> Path:
    """Notes directory: env override > project root > user config fallback."""
    env = os.environ.get("PLANNER_NOTES_DIR")
    if env:
        return Path(env)
    root = find_project_root()
    if root is not None:
        return project_notes_dir(root)
    return NOTES_FALLBACK


def default_board_dir() -> Path:
    """Board directory: env override > project root > user config fallback."""
    env = os.environ.get("PLANNER_BOARD_DIR")
    if env:
        return Path(env)
    root = find_project_root()
    if root is not None:
        return project_board_dir(root)
    return BOARD_FALLBACK


def default_backend(notes_dir: Path | None = None) -> str:
    """Storage backend: env override > inferred from the notes directory.

    ``PLANNER_BACKEND`` wins when set to a known backend. Otherwise the
    backend is inferred from the resolved notes directory: a MogDB journal
    (``store/notes.journal.jsonl``) selects ``mogdb``; everything else
    defaults to ``file``. Pass *notes_dir* explicitly to infer from a
    directory other than the default.
    """
    backend = os.environ.get("PLANNER_BACKEND")
    if backend in BACKENDS:
        return backend
    dir_path = Path(notes_dir or default_notes_dir())
    if (dir_path / "store" / "notes.journal.jsonl").is_file():
        return "mogdb"
    return "file"
