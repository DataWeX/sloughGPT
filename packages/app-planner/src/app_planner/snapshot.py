"""
app_planner.snapshot — export the canonical board into the repo for history.

The canonical-state flip moved the live board to the user-config location
(``config.BOARD_FALLBACK``) so worktrees stop forking it. Versioned history
is opt-in: ``snapshot()`` copies the canonical ``board.jsonl`` into
``<repo>/.kanban/snapshots/board-<utc-stamp>.jsonl`` — an append-only
directory of dated, byte-exact copies that can be committed when wanted.

No git operations happen here (the CLI prints the suggested commit); this
module only writes the snapshot file and reports its path.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config

SNAPSHOT_DIR = "snapshots"


class SnapshotError(RuntimeError):
    """Raised when no canonical board exists to snapshot."""


def snapshot(
    repo_root: Path | None = None,
    *,
    board_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy the canonical board into ``<repo>/.kanban/snapshots/``.

    Args:
        repo_root: Destination repo root (default: cwd). Snapshot files
            land under ``<repo_root>/.kanban/snapshots/``.
        board_dir: Canonical board dir (default ``config.BOARD_FALLBACK``).

    Returns:
        Dict with the snapshot path, source path, and byte size.

    Raises:
        SnapshotError: the canonical board file does not exist.
    """
    src = (Path(board_dir) if board_dir else config.BOARD_FALLBACK) / "board.jsonl"
    if not src.is_file():
        raise SnapshotError(f"no canonical board at {src} — nothing to snapshot")
    dest_dir = (repo_root if repo_root is not None else Path.cwd()) / ".kanban" / SNAPSHOT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"board-{stamp}.jsonl"
    # Never clobber an existing snapshot (two runs within one second):
    # step a numeric suffix until the path is free.
    counter = 1
    while dest.exists():
        dest = dest_dir / f"board-{stamp}-{counter}.jsonl"
        counter += 1
    shutil.copy2(src, dest)
    return {
        "snapshot": str(dest),
        "source": str(src),
        "bytes": dest.stat().st_size,
    }
