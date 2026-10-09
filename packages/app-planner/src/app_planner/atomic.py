"""Atomic, fsync'd file writes — the planner's store-of-record primitive.

A crash during a truncate-in-place write (``open(path, "w")`` /
``Path.write_text``) can leave a store empty or torn: the old content is
already gone while the new content may never reach disk. Every store write
therefore goes **tmp → fsync → ``os.replace`` → directory fsync**, so a
crash at any point leaves either the complete old file or the complete new
file, and an acknowledged write survives power loss (not just SIGKILL).

Sibling of ``mogdb.durability.atomic_write`` (the bytes variant used for
journals); kept local so app-planner does not depend on the mogdb package.
Enforced by ``tests/core-py/test_store_write_guard.py``.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path

__all__ = ["append_lines", "atomic_write_text", "fsync_dir"]


def fsync_dir(dir_path: Path | str) -> None:
    """Fsync a directory so a rename within it survives power loss."""
    dfd = os.open(dir_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write *text* to *path* atomically and durably.

    Uses a same-directory ``mkstemp`` (atomic replace requires same
    filesystem), fsyncs the data before publishing it, then fsyncs the
    directory so the rename itself is durable. On any failure the tmp file
    is removed and the original file is left untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    fsync_dir(path.parent)


def append_lines(path: Path | str, lines: list[str]) -> None:
    """Append complete lines to an append-only journal, durably.

    Single ``O_APPEND`` write per line under an exclusive ``flock``, then
    ``fsync`` — concurrent appenders can never interleave bytes and an
    acknowledged line survives power loss. Safe to flock the data file
    itself here: unlike MogDB journals this file is never atomically
    rewritten, so its inode (the lock target) is stable.
    """
    if not lines:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(lines).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
