"""Atomic file replacement — one owner of the write-temp-then-rename dance.

Every durable writer in the tree used to hand-roll this: ``mkstemp`` in the
target directory, write, rename, unlink the temp on failure. Copying it around
produced five different temp-name conventions (``x.tmp``, ``x.card.json.tmp``,
``.x.tmp``, ``x.tmp.npz``, ad-hoc ``suffix=".tmp"``) and — worse — five
different cleanup behaviours, because the bare-except ``unlink`` tail is the part
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
never a half-written one). That is *not* a crash-durability guarantee, so
``durable=True`` adds the full flush sequence for callers that need one:

    fsync(temp) -> os.replace -> fsync(parent directory)

The two axes are priced separately (measured on this repo's filesystem with
this module, a few hundred bytes per write):

    raw write    ~0.3ms    torn reads possible
    atomic       ~0.3ms    no torn reads, not power-safe   <- default
    durable      ~28ms     power-safe (measured range 14-45ms)

Atomic is the default because it costs no more than a plain write — measured
identical, within noise — while durable is opt-in because it costs two orders
of magnitude more. The split is not theoretical: flux's ``advance()``
heartbeat rides on rename atomicity while the checkpoint claim about weights
earns the flush, and that distinction is what keeps a durable record cheap.

Where the durable cost actually lives (interleaved medians, 7 reps x 100
iters, so every tier saw the same system load):

    flush file             ~12ms   fsync of the temp before the rename
    flush dir when dirty   ~15ms   fsync of the parent after the rename
    flush dir when clean   ~0.01ms nothing to commit
    makedirs               ~0.03ms

The directory flush is free until a rename dirties it — a ~1000x ratio — and
that is why *batching* is the lever, not a faster fsync. Eight writes behind
one directory flush measured 21ms total (2.6ms/write) against 191ms
(23.9ms/write) when flushing after every rename: 9x. Flush once per batch,
never once per write.

Absolute figures swing 2-3x with machine load (durable ranged 14-45ms across
reps on a busy, nearly-full filesystem); the ratios above are the durable
facts, the milliseconds are the least of them.

The directory flush is not decoration. ``os.replace`` changes a *directory
entry*; without ``fsync`` of the parent directory the rename itself can be
lost on a power cut and the old file reappears with the new one gone. A
``durable`` call that skipped it would be a false guarantee, which is worse
than none — callers stop being careful.
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


def _fsync_dir(directory: str) -> None:
    """Flush a directory's entries — the rename — to stable storage.

    Args:
        directory: An existing directory path.

    Raises:
        OSError: If the filesystem cannot flush a directory (some FUSE and
            network mounts). Propagated deliberately: ``durable=True`` promises
            durability, and silently swallowing this would leave a caller
            believing in a guarantee the filesystem never gave.
    """
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def atomic_writer(
    path: str | os.PathLike[str],
    *,
    binary: bool = True,
    encoding: str = "utf-8",
    durable: bool = False,
) -> Iterator[IO]:
    """Yield a handle whose contents replace ``path`` only on clean exit.

    The caller writes through the handle; the rename happens after the body
    returns. If the body raises — or the rename itself fails — the temp is
    removed and the original ``path`` is left exactly as it was.

    Args:
        path: Destination file.
        binary: ``True`` yields a ``wb`` handle, ``False`` a text handle.
        encoding: Text encoding when ``binary`` is False.
        durable: Also flush the temp to stable media *before* the rename and
            the parent directory *after* it, so the write survives a power
            cut. Costs ~50-100x the atomic path (see the module docstring for
            the breakdown and how batching pays for it) — spend it on
            artifacts and the claims about them, not on heartbeats.

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
            if durable:
                # Still inside the `with`, while the fd is open: push the
                # handle's buffer into the OS, then the OS's page cache onto
                # the disk. Once the `with` closes it there is nothing left to
                # fsync, and an fsync after the rename covers only half the
                # transaction (the file, not the entry that names it).
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        if durable:
            # The rename above is a directory-entry change; flush that too, or
            # a power cut can resurrect the OLD file. See the module docstring.
            _fsync_dir(os.path.dirname(os.path.abspath(target)) or ".")
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
    durable: bool = False,
) -> str:
    """Write ``payload`` to ``path`` atomically.

    Args:
        path: Destination file.
        payload: Bytes to write, or a str encoded with ``encoding``.
        encoding: Encoding used when ``payload`` is a str.
        durable: Flush to stable media before and after the rename so the
            write survives a power cut. See ``atomic_writer``.

    Returns:
        ``path`` as a string (the destination, never the temp).

    Side effects:
        - replaces ``path`` atomically; creates its parent directory
    """
    data = payload.encode(encoding) if isinstance(payload, str) else payload
    with atomic_writer(path, binary=True, durable=durable) as handle:
        handle.write(data)
    return os.fspath(path)
