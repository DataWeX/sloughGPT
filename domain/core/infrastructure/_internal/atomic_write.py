"""Atomic file replacement — one owner of the write-temp-then-rename dance.

Every durable writer in the tree used to hand-roll this: ``mkstemp`` in the
target directory, write, rename, unlink the temp on failure. Copying it around
produced five different temp-name conventions (``x.tmp``, ``x.card.json.tmp``,
``.x.tmp``, ``x.tmp.npz``, ad-hoc ``suffix=".tmp"``) and — worse — five
different cleanup behaviours, because the ``except: unlink`` tail is the part
people forget. A writer that forgets it leaves a partial temp behind, and the
checkpoint scanner then has to know how to skip that temp.

Two rules the copies disagreed on, now stated once:

1. **The temp lives in the target's directory**, never in ``/tmp``. ``rename``
   is atomic only within one filesystem; a cross-device rename raises
   ``EXDEV``, and ``shutil.move`` degrades silently to copy-then-delete — which
   is exactly the non-atomic window this exists to close.
2. **``os.replace``, not ``os.rename``.** POSIX makes both atomic, but
   ``rename`` fails when the destination exists on some platforms; ``replace``
   is the overwrite-always form and is what "swap in the new version" means.

Durability here is rename atomicity (readers see the old file or the new one,
never a half-written one). It is not a crash-durability guarantee: no
``fsync`` of the file or its directory, so a power cut immediately after the
rename can still lose it. Add one only where that matters.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

# The single temp spelling. ``<stem>.tmp.npz`` (slonet's compressed checkpoint
# writer) is the one historical variant, kept in the same predicate so a
# scanner never has to guess which convention a writer used.
TEMP_SUFFIX = ".tmp"
TEMP_SUFFIXES = (TEMP_SUFFIX, TEMP_SUFFIX + ".npz")


def is_temp_name(name: str) -> bool:
    """Whether ``name`` is an in-progress temp artifact, not a finished file.

    Args:
        name: A file *name* (not a path) — only the final component carries
            the grammar.

    Returns:
        True for both temp spellings: ``x.tmp`` and ``x.tmp.npz``.
    """
    return name.endswith(TEMP_SUFFIXES)


def _create_temp(path: str | os.PathLike[str]) -> tuple[int, str]:
    """Create a temp file next to ``path`` and return ``(fd, temp_path)``.

    The parent directory is created if missing so a writer never has to
    ``mkdir`` before calling.
    """
    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    return tempfile.mkstemp(dir=directory, suffix=TEMP_SUFFIX)


@contextmanager
def atomic_writer(
    path: str | os.PathLike[str],
    *,
    binary: bool = True,
    encoding: str = "utf-8",
) -> Iterator[IO]:
    """Yield a handle whose contents replace ``path`` only on clean exit.

    The caller writes through the handle; the rename happens after the body
    returns. If the body raises — or the rename itself fails — the temp is
    removed and the original ``path`` is left exactly as it was.

    Args:
        path: Destination file.
        binary: ``True`` yields a ``wb`` handle, ``False`` a text handle.
        encoding: Text encoding when ``binary`` is False.

    Yields:
        An open file handle positioned at 0.

    Side effects:
        - creates and (on failure) removes a ``*.tmp`` beside ``path``
        - on success atomically replaces ``path``
    """
    target = os.fspath(path)
    fd, tmp_path = _create_temp(target)
    try:
        handle = os.fdopen(fd, "wb") if binary else os.fdopen(fd, "w", encoding=encoding)
    except BaseException:
        # fdopen failed, so nobody owns the fd yet — close it explicitly or
        # it leaks for the life of the process.
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    try:
        with handle:
            yield handle
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write(
    path: str | os.PathLike[str],
    payload: bytes | str,
    *,
    encoding: str = "utf-8",
) -> str:
    """Write ``payload`` to ``path`` atomically.

    Args:
        path: Destination file.
        payload: Bytes to write, or a str encoded with ``encoding``.
        encoding: Encoding used when ``payload`` is a str.

    Returns:
        ``path`` as a string (the destination, never the temp).

    Side effects:
        - replaces ``path`` atomically; creates its parent directory
    """
    data = payload.encode(encoding) if isinstance(payload, str) else payload
    with atomic_writer(path, binary=True) as handle:
        handle.write(data)
    return os.fspath(path)
