"""Shared MogDB connection pool — singleton instances per database path.

Every router that calls ``_get_db()`` was creating a new MogDB instance per
request, paying initialization + file descriptor + locking costs each time.
This module provides a process-wide singleton cache keyed by ``(db_path, sync_path)``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mogdb import MogDB

_REPO_ROOT: Path | None = None


def _get_repo_root() -> Path:
    """Resolve repo root once per process."""
    global _REPO_ROOT
    if _REPO_ROOT is None:
        _REPO_ROOT = Path(__file__).resolve().parents[5]
    return _REPO_ROOT


# Process-wide singleton cache: (db_path, sync_path) -> MogDB instance
_POOL: dict[tuple[str, str], MogDB] = {}


def get_db(db_name: str, *, sync_dir_name: str | None = None) -> MogDB:
    """Return a shared MogDB instance for the given database name.

    Args:
        db_name: Directory name under ``data/`` (e.g. ``"uploads_mogdb"``).
        sync_dir_name: Optional sync directory name. Defaults to
            ``db_name.replace("_mogdb", "_json")``.

    Returns:
        A reused ``MogDB`` instance for this process.
    """
    from mogdb import MogDB  # deferred to avoid import-time cost

    if sync_dir_name is None:
        sync_dir_name = db_name.replace("_mogdb", "_json")

    repo_root = _get_repo_root()
    db_path = os.path.join(repo_root, "data", db_name)
    sync_path = os.path.join(repo_root, "data", sync_dir_name)

    key = (db_path, sync_path)
    if key not in _POOL:
        _POOL[key] = MogDB(db_path, sync_dir=sync_path)
    return _POOL[key]
