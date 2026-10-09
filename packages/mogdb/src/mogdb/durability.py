"""Crash-resistant journal primitives for MogDB.

Three guarantees, enforced in the write primitive rather than per-store:

1. **Cross-process serialization** — every journal/snapshot mutation holds an
   exclusive ``fcntl`` lock on a *stable* ``<name>.lock`` file. The lock file
   is never replaced or unlinked (unlike the data files, which are rewritten
   atomically), so the locked inode stays the same across rewrites and two
   processes can never interleave writes or race a load-time rewrite.

2. **Torn-write containment** — records are appended with a single
   ``os.write`` call on an ``O_APPEND`` fd (full line + ``\\n``), so a record
   lands all-or-nothing; a crash can only damage the *tail* of the file.

3. **Durability + recovery** — each append is ``fsync``-ed by default, and
   load-time corruption (torn tail, NUL poisoning, non-JSON lines) is
   quarantined to a ``<name>.corrupt-<ts>`` sidecar with counts reported in
   ``Collection.open_report`` — never silently dropped.

Compact writes the snapshot to ``<name>.mogdb.tmp``, fsyncs, then
``os.replace``s it into place — a crash mid-compact leaves the journal
authoritative (see ``Collection.compact`` for the seq-watermark that makes
snapshot+residual-journal exactly-once).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("slo.mogdb.durability")


@contextmanager
def locked(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive cross-process lock for the duration of the block.

    ``lock_path`` must be a stable file: never replaced via ``os.replace``
    and never unlinked while it is in use. Locking the data file itself is
    not sufficient — atomic rewrites change its inode, which would let a
    blocked writer acquire the lock on the *old* inode while a rewriter
    publishes the new one.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte (``os.write`` may write short)."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def append_record(path: Path, record: bytes, *, lock_path: Path, fsync: bool = True) -> None:
    """Append one complete record under the cross-process lock.

    ``record`` must be the full line including its trailing newline; it is
    written with a single ``os.write`` on an ``O_APPEND`` fd so concurrent
    appenders (serialized by the lock) can never interleave bytes, and a
    crash can never leave a *partial* record followed by a valid one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked(lock_path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            _write_all(fd, record)
            if fsync:
                os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(path: Path, data: bytes, *, fsync: bool = True) -> None:
    """Publish ``data`` at ``path`` atomically: tmp → fsync → ``os.replace``.

    A crash before the rename leaves the previous file (or none) fully
    intact — the tmp file (``<name>.tmp``) is ignored by loaders and
    overwritten by the next attempt.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            _write_all(fd, data)
            if fsync:
                os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    if fsync:
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _parse_record(line: bytes) -> dict | None:
    """Parse one record; ``None`` marks it corrupt.

    NUL-free, decodable, and a JSON object — anything else (torn tail,
    NUL poisoning, stray scalar) goes to quarantine. The parsed object is
    returned rather than a bool: classification and application each
    needing their own ``json.loads`` doubled every replay (reopen
    benchmark: 0.42s → 0.94s on 20k records before this collapsed both
    into one parse).
    """
    if b"\x00" in line:
        return None
    try:
        obj = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def read_records(path: Path, *, lock_path: Path, corrupt_stem: str) -> tuple[list[dict], int]:
    """Read JSONL records, quarantining corrupt lines.

    Caller must NOT hold ``lock_path`` (this acquires it). Returns
    ``(records, quarantined_count)`` — each record parsed exactly once, so
    callers apply them without a second ``json.loads``. Corrupt lines —
    torn tails, NUL poisoning, non-JSON garbage — are appended verbatim to
    ``<corrupt_stem>.corrupt-<ts>`` and then removed from the file via an
    atomic rewrite (under the same lock, so no writer can slip between read
    and rewrite). Healthy files are returned untouched (no rewrite, no
    inode change). Blank lines are dropped from the returned list but are
    not counted as corruption.
    """
    with locked(lock_path):
        return read_records_locked(
            path, corrupt_stem=corrupt_stem, sidecar_dir=path.parent
        )


def read_records_locked(
    path: Path, *, corrupt_stem: str, sidecar_dir: Path
) -> tuple[list[dict], int]:
    """``read_records`` variant for callers already holding the lock.

    Reading and rewriting must happen in one critical section — a writer
    blocked between them would append to the old inode and lose its record.
    """
    if not path.exists():
        return [], 0
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()

    valid: list[tuple[bytes, dict]] = []
    corrupt: list[bytes] = []
    for line in lines:
        if not line.strip():
            continue
        obj = _parse_record(line)
        if obj is None:
            corrupt.append(line)
        else:
            valid.append((line, obj))
    records = [obj for _, obj in valid]

    if not corrupt:
        return records, 0

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    sidecar = sidecar_dir / f"{corrupt_stem}.corrupt-{ts}"
    with open(sidecar, "ab") as f:
        for line in corrupt:
            f.write(line + b"\n")
        f.flush()
        os.fsync(f.fileno())

    # Remove the corruption from circulation while keeping every valid
    # record: atomic rewrite under the lock appenders use.
    payload = b"".join(line + b"\n" for line, _ in valid)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)

    logger.warning(
        "mogdb: quarantined %d corrupt line(s) in %s -> %s (%d valid kept)",
        len(corrupt),
        path.name,
        sidecar.name,
        len(records),
    )
    return records, len(corrupt)
