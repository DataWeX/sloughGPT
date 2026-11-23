"""Seed a fresh worktree's pytest-testmon DB from elsewhere in the same repo.

Card 20261111_125558 (Part A): a worktree with no ``.testmondata`` degrades to
a full record run — 48k tests selected even when the diff touches 9 files —
which is the dominant "fresh checkout penalty" of the local loop.

The seed copies an existing recording with the sqlite3 **backup API**, which
takes a consistent snapshot even while another suite is writing the source
(this is the "safe against a live writer" property the card's manual proof
relied on). Paths are per-repo (pytest-testmon stores paths relative to the
repo root), so a DB recorded in one worktree selects correctly in another:
files changed since that recording get selected — exactly the honest answer.

Source choice, measured 2026-11-12 — NOT "the primary checkout": the primary
worktree held a 9-day-stale 16MB DB while a session worktree held the live
52MB one. So the source is the FRESHEST ``.testmondata`` across the repo's
worktrees, overridable with ``TESTMON_SEED_DB``.

Fallback contract (Test Rule clause 3): no seed source, a corrupt/non-testmon
source, or any error means NO database is written and the caller degrades to
today's behavior — a full record run. Unknown state is a full run, never a
guess; a failed seed must never leave a partial ``.testmondata`` behind
(pytest-testmon would then trust it).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# pytest-testmon 2.x schema — a DB missing these is not a testmon recording.
_REQUIRED_TABLES = frozenset(
    {"metadata", "environment", "test_execution", "file_fp", "test_execution_file_fp"}
)

ENV_OVERRIDE = "TESTMON_SEED_DB"


def find_seed_db(repo: Path) -> Path | None:
    """Locate the best seed source for ``repo``'s worktrees.

    Order: ``TESTMON_SEED_DB`` (if it names a readable testmon DB) → the
    freshest ``.testmondata`` among the repo's registered worktrees → None.
    """
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        candidate = Path(override)
        if _is_testmon_db(candidate):
            return candidate
        logger.warning("%s=%s is not a readable testmon DB — ignoring", ENV_OVERRIDE, override)

    try:
        res = subprocess.run(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    newest: tuple[float, Path] | None = None
    for line in res.stdout.splitlines():
        # Path may contain spaces ('/home/.../Default Project/sloughGPT') —
        # strip the prefix, never split on whitespace.
        if not line.startswith("worktree "):
            continue
        db = Path(line[len("worktree ") :]) / ".testmondata"
        try:
            mtime = db.stat().st_mtime
        except OSError:
            continue  # worktree gone, or no recording yet
        if newest is None or mtime > newest[0]:
            newest = (mtime, db)
    return newest[1] if newest else None


def seed_testmon_db(
    dest: Path,
    *,
    source: Path | None = None,
    repo: Path | None = None,
) -> bool:
    """Copy the seed recording to ``dest``. True = seeded, False = caller falls back.

    Never overwrites an existing ``dest`` (an existing recording is trusted
    as-is, same as before this helper existed). On any failure the destination
    is removed so the caller's full-record fallback starts clean.
    """
    if dest.exists():
        return False
    if source is None:
        if repo is None:
            return False
        source = find_seed_db(repo)
        if source is None or source.resolve() == dest.resolve():
            return False
    if not _is_testmon_db(source):
        return False

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Read-only URI: the seed must never mutate the source recording.
        src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            dst = sqlite3.connect(str(dest))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        if not _is_testmon_db(dest):
            raise sqlite3.DatabaseError("seeded DB lost the testmon schema")
    except (sqlite3.Error, OSError):
        logger.warning("testmon seed from %s failed — full record run", source, exc_info=True)
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    logger.info("seeded %s from %s", dest, source)
    return True


def _is_testmon_db(path: Path) -> bool:
    """True when ``path`` is a sqlite file carrying the testmon schema."""
    if not path.is_file():
        return False
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            con.close()
    except sqlite3.Error:
        return False
    return _REQUIRED_TABLES <= names
