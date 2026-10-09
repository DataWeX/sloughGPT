"""Durability pin for the repo's SQLite stores (post-faa1cfa7).

Census 2026-10-09: every live SQLite store runs SQLite's defaults —
``journal_mode=delete`` (persisted in the file header) with the default
``synchronous=FULL`` — so an acknowledged commit survives process crash and
power loss, and no source file assigns these PRAGMAs. That "no gap" finding
is only worth having if it cannot silently regress.

Two pins:

* **On-disk journal_mode** — the one setting SQLite persists. Any file
  flipped out of ``delete`` means a deliberate tooling change; the test must
  be updated consciously, exactly like a schema change.
* **Source guard** — ``synchronous`` is connection-scoped and *not*
  persisted, so querying a file proves nothing about the connections that
  write it. The guarantee therefore lives in the code, and this test fails
  the moment any non-test source assigns ``PRAGMA synchronous=`` or
  ``PRAGMA journal_mode=``. Flipping sync to OFF/NORMAL would void the same
  acknowledged-write durability invariant the MogDB/planner crash batteries
  enforce for JSON stores.

Busy databases (live writers during a test run) are skipped, never failed —
flakiness would bury the real signal.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Production source roots. tests/ deliberately excluded (fixtures may set
# pragmas to exercise their own scenarios).
SCAN_ROOTS = [
    REPO / "domain",
    REPO / "apps",
    REPO / "services",
    REPO / "packages",
    REPO / "scripts",
    REPO / "testing",
]

# Where live (non-test) SQLite stores are allowed to live.
DB_BASES = [
    REPO / "data",
    REPO / "packages" / "core-py",
    REPO / "apps" / "api" / "server",
]
DB_PATTERNS = ("*.db", "*.sqlite", "*.sqlite3")

PRAGMA_SET_RE = re.compile(r"pragma\s+(synchronous|journal_mode)\s*=", re.IGNORECASE)

_SKIP_PARTS = ("/tests/", "/test_", "__pycache__", "/.wt-", "node_modules", "/.venv/")


def _live_dbs() -> list[Path]:
    """Every SQLite file under the live data bases (MogDB dirs are not files)."""
    found: set[Path] = set()
    for base in DB_BASES:
        if not base.is_dir():
            continue
        for pattern in DB_PATTERNS:
            for path in base.rglob(pattern):
                if path.is_file() and not any(s in str(path) for s in _SKIP_PARTS):
                    found.add(path)
    return sorted(found)


def test_live_sqlite_files_keep_durable_journal_mode() -> None:
    """On-disk journal_mode stays at the durable default (delete/rollback)."""
    dbs = _live_dbs()
    if not dbs:
        pytest.skip("no live SQLite stores present on this machine")

    modes: dict[str, str] = {}
    busy = 0
    for db in dbs:
        rel = db.relative_to(REPO).as_posix()
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
            try:
                modes[rel] = con.execute("PRAGMA journal_mode").fetchone()[0]
            finally:
                con.close()
        except sqlite3.OperationalError:
            busy += 1  # writer holds the lock right now — not a durability failure

    if not modes:
        pytest.skip(f"all {len(dbs)} live SQLite stores busy (active writers)")

    wrong = {path: mode for path, mode in modes.items() if mode != "delete"}
    assert not wrong, (
        "SQLite file(s) left the durable delete/rollback journal mode — if this "
        "was a deliberate migration, pair it with synchronous=FULL in code and "
        "update this pin with the reason:\n"
        + "\n".join(f"  {p}: {m}" for p, m in sorted(wrong.items()))
    )


def test_no_source_assigns_sqlite_pragmas() -> None:
    """Durability rides SQLite's defaults — no non-test code may retune them."""
    offenders: list[str] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(s in str(path) for s in _SKIP_PARTS):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if PRAGMA_SET_RE.search(line):
                    rel = path.relative_to(REPO).as_posix()
                    offenders.append(f"{rel}:{lineno}  {line.strip()[:120]}")

    assert not offenders, (
        "source assigns PRAGMA synchronous/journal_mode — acknowledged-write "
        "durability depends on the SQLite default (FULL); an OFF/NORMAL here "
        "silently voids it. Route a deliberate migration through this test "
        "with the reason:\n" + "\n".join(offenders)
    )
