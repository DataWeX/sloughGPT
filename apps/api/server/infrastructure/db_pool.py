"""Shared MogDB connection pool — singleton instances per database path.

Every router that calls ``_get_db()`` was creating a new MogDB instance per
request, paying initialization + file descriptor + locking costs each time.
This module provides a process-wide singleton cache keyed by ``(db_path, sync_path)``.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mogdb import MogDB

logger = logging.getLogger(__name__)

_REPO_ROOT: Path | None = None
_LEGACY_ROOT: Path | None = None


def _get_repo_root() -> Path:
    """Resolve repo root once per process (marker-based, same as main.py)."""
    global _REPO_ROOT
    if _REPO_ROOT is None:
        from domain.shared import find_repo_root

        _REPO_ROOT = find_repo_root(Path(__file__).resolve())
    return _REPO_ROOT


def _get_legacy_root() -> Path:
    """Pre-fix data root: ``parents[5]`` placed ``data/`` above the repo.

    Retained only so existing user data is adopted (copied, never moved)
    into the corrected location on first use.
    """
    global _LEGACY_ROOT
    if _LEGACY_ROOT is None:
        _LEGACY_ROOT = Path(__file__).resolve().parents[5]
    return _LEGACY_ROOT


def _adopt_legacy_dir(new_dir: str | Path, legacy_dir: str | Path) -> str:
    """Copy pre-fix data from *legacy_dir* into *new_dir* once.

    Never overwrites: adoption happens only when the destination is missing
    or empty. Returns the path to use — the legacy directory if adoption was
    required but failed, so the app keeps working on the old location.
    """
    new, legacy = Path(new_dir), Path(legacy_dir)
    try:
        if legacy.resolve() == new.resolve():
            return str(new)
        if not legacy.is_dir():
            return str(new)
        target_empty = not new.exists() or not any(new.iterdir())
        if target_empty:
            shutil.copytree(legacy, new, dirs_exist_ok=True)
            logger.info("Adopted legacy data %s -> %s", legacy, new)
        elif any(legacy.iterdir()):
            logger.warning(
                "Both %s and %s contain data; using %s (legacy untouched)",
                new,
                legacy,
                new,
            )
        return str(new)
    except OSError as exc:
        logger.warning(
            "Could not adopt legacy data %s -> %s: %s; using legacy path",
            legacy,
            new,
            exc,
        )
        return str(legacy)


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

    data_root = _get_repo_root() / "data"
    db_path = os.path.join(data_root, db_name)
    sync_path = os.path.join(data_root, sync_dir_name)

    legacy_root = _get_legacy_root() / "data"
    if legacy_root.is_dir():
        db_path = _adopt_legacy_dir(db_path, legacy_root / db_name)
        sync_path = _adopt_legacy_dir(sync_path, legacy_root / sync_dir_name)

    key = (db_path, sync_path)
    if key not in _POOL:
        _POOL[key] = MogDB(db_path, sync_dir=sync_path)
    return _POOL[key]
