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

Env guard (card 20261112_122610): freshness alone can hand over a recording
whose environment differs from the current interpreter — testmon then REPLACES
the environment row on first run and ``test_execution.environment_id`` is
ON DELETE CASCADE, so every copied row is purged: a silent no-op behind a
"fresh-worktree penalty skipped" lie. Discovery therefore takes only
recordings whose fingerprint (``system_packages`` + ``python_version``) equals
what testmon will record here — computed with testmon's own
``drop_patch_version`` (storage drops the patch level: ``PyJWT 2.14``, never
``PyJWT 2.14.0``) — with rows > 0 (a 0-row DB is the post-wipe state, nothing
to seed). No match → None → the honest full record. An unknown fingerprint on
either side (no testmon, no env row, unreadable) falls back to freshness,
never a hard fail; ``TESTMON_SEED_DB`` stays explicit user intent (warn on
mismatch, still copy).

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


def _current_fingerprint() -> tuple[str, str] | None:
    """``(system_packages, python_version)`` exactly as testmon stores them here.

    Uses testmon's OWN helpers so the comparison matches storage byte for byte
    (``drop_patch_version`` is applied on write — raw versions would never
    match). ``None`` = can't compute (testmon absent) → callers fall back to
    freshness, never a hard fail.
    """
    try:
        import platform

        from testmon.common import drop_patch_version, get_system_packages
    except Exception:  # pragma: no cover - only in envs without testmon
        return None
    try:
        return drop_patch_version(get_system_packages()), platform.python_version()
    except Exception:  # pragma: no cover - defensive: unknown, not fatal
        return None


def _env_status(path: Path, current: tuple[str, str] | None) -> bool | None:
    """True/False = does the recording's environment match ``current``; None = unknown.

    Compares against the environment rows the RECORDED TESTS point at (the
    cascade deletes exactly those on drift) — the newest row alone could belong
    to a different generation. ``None`` covers every unknown: no current
    fingerprint, no env row yet, or a schema without the columns.
    """
    if current is None:
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = list(
                con.execute(
                    "SELECT DISTINCT e.system_packages, e.python_version"
                    " FROM environment e"
                    " JOIN test_execution t ON t.environment_id = e.id"
                )
            )
        finally:
            con.close()
    except sqlite3.Error:
        return None
    if not rows:
        return None
    return all((r[0], r[1]) == current for r in rows)


def _has_recorded_tests(path: Path) -> bool:
    """True when the recording holds at least one test_execution row."""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            n = con.execute("SELECT COUNT(*) FROM test_execution").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return False
    return bool(n)


def dest_state(path: Path) -> str:
    """Seed health of an existing (or absent) destination DB.

    ``missing``   — no recording: seedable.
    ``empty``     — valid testmon schema, 0 recorded tests: the cascade wipe
                    already purged it; nothing to select from → re-seedable.
    ``healthy``   — recorded tests on disk: trusted as-is (never overwritten).
    ``invalid``   — exists but isn't a usable testmon DB: trusted as-is too —
                    an unknown file is never destroyed (Test Rule: unknown
                    state is a full run, never a guess).
    """
    if not path.exists():
        return "missing"
    if not _is_testmon_db(path):
        return "invalid"
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            n = con.execute("SELECT COUNT(*) FROM test_execution").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return "invalid"
    return "empty" if n == 0 else "healthy"


def find_seed_db(repo: Path) -> Path | None:
    """Locate the best seed source for ``repo``'s worktrees.

    Order: ``TESTMON_SEED_DB`` (explicit override — copied even on env
    mismatch, warned) → the freshest ENVIRONMENT-MATCHING ``.testmondata``
    with recorded tests among the repo's registered worktrees → (fingerprint
    unknown everywhere) the freshest such recording → None.
    """
    current = _current_fingerprint()
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        candidate = Path(override)
        if _is_testmon_db(candidate):
            if _env_status(candidate, current) is False:
                logger.warning(
                    "%s=%s environment differs from this interpreter — "
                    "copying anyway (explicit override); its rows may be "
                    "purged on the first run",
                    ENV_OVERRIDE,
                    override,
                )
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

    matched: list[tuple[float, Path]] = []
    unknown: list[tuple[float, Path]] = []
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
        if not _is_testmon_db(db) or not _has_recorded_tests(db):
            continue  # unreadable schema, or the cascade-wiped 0-row state
        status = _env_status(db, current)
        if status is True:
            matched.append((mtime, db))
        elif status is None:
            # Fingerprint unknown on one side — freshness fallback, never a
            # hard fail (the conservative full run only follows a None here).
            unknown.append((mtime, db))
        # status is False: copying it would be purged on the destination's
        # first run — a seed that cannot deliver its claim is no seed at all.
    pool = matched or unknown
    if not pool:
        return None
    return max(pool)[1]


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
