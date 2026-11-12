"""Tests for scripts/seed_testmon.py — auto-seed of the testmon DB (card 20261111_125558).

A fresh worktree has no `.testmondata`, so run_affected.sh degrades to a full
record run (48k selected) even for a 9-file diff. The seed copies an existing
DB from elsewhere in the same repo via the sqlite3 backup API (consistent even
while a suite records into it). Selection honesty is untouched: unknown state
still means a full run when NO seed source exists (Test Rule clause 3).

Source choice is empirical, not "primary checkout": measured 2026-11-12, the
primary worktree's DB was 9 days stale (16MB, Nov 3) while a session worktree
held the live 52MB one — so the seed takes the FRESHEST `.testmondata` across
the repo's worktrees, with TESTMON_SEED_DB as an explicit override.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import seed_testmon  # noqa: E402

# The real pytest-testmon 2.x schema (measured on the primary checkout DB).
TESTMON_TABLES = {
    "metadata",
    "environment",
    "test_execution",
    "file_fp",
    "test_execution_file_fp",
    "suite_execution_file_fsha",
}


def _make_db(path: Path, tables: set[str] | None = None) -> None:
    """Create a sqlite file with the given (testmon) tables + one marker row."""
    con = sqlite3.connect(path)
    try:
        for t in sorted(tables if tables is not None else TESTMON_TABLES):
            con.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY, v TEXT)")
        con.execute(
            f"INSERT INTO {tables and sorted(tables)[0] or 'metadata'} (v) VALUES ('marker')"
        )
        # At least one recorded test: discovery only considers DBs with rows
        # (a 0-row DB is the cascade-wiped state — never a seed source).
        con.execute("INSERT INTO test_execution (v) VALUES ('recorded-test')")
        con.commit()
    finally:
        con.close()


def _rows(path: Path, table: str) -> list:
    con = sqlite3.connect(path)
    try:
        return list(con.execute(f"SELECT v FROM {table}"))
    finally:
        con.close()


def test_missing_source_returns_false_and_creates_no_dest(tmp_path: Path) -> None:
    dest = tmp_path / "wt" / ".testmondata"
    dest.parent.mkdir()
    assert seed_testmon.seed_testmon_db(dest, source=tmp_path / "nope") is False
    assert not dest.exists()


def test_existing_dest_is_never_overwritten(tmp_path: Path) -> None:
    dest = tmp_path / ".testmondata"
    dest.write_bytes(b"precious existing recording")
    assert seed_testmon.seed_testmon_db(dest, source=tmp_path / "nope") is False
    assert dest.read_bytes() == b"precious existing recording"


def test_backup_roundtrip_preserves_testmon_schema_and_rows(tmp_path: Path) -> None:
    src = tmp_path / "primary" / ".testmondata"
    src.parent.mkdir()
    _make_db(src)
    dest = tmp_path / "fresh-wt" / ".testmondata"
    dest.parent.mkdir()

    assert seed_testmon.seed_testmon_db(dest, source=src) is True
    con = sqlite3.connect(dest)
    try:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()
    assert names == TESTMON_TABLES
    assert _rows(dest, "metadata") == [("marker",)]


def test_corrupt_source_falls_back_cleanly(tmp_path: Path) -> None:
    src = tmp_path / "garbage"
    src.write_text("this is not sqlite at all")
    dest = tmp_path / "fresh-wt" / ".testmondata"
    dest.parent.mkdir()

    assert seed_testmon.seed_testmon_db(dest, source=src) is False
    assert not dest.exists(), "a failed seed must not leave a partial DB behind"


def test_non_testmon_sqlite_is_rejected(tmp_path: Path) -> None:
    """Any sqlite file (e.g. an unrelated DB) must not be seeded as testmon state."""
    src = tmp_path / "other.db"
    con = sqlite3.connect(src)
    con.execute("CREATE TABLE unrelated (x TEXT)")
    con.commit()
    con.close()
    dest = tmp_path / ".testmondata"

    assert seed_testmon.seed_testmon_db(dest, source=src) is False
    assert not dest.exists()


def test_find_seed_db_returns_none_outside_git(tmp_path: Path) -> None:
    assert seed_testmon.find_seed_db(tmp_path) is None


def test_find_seed_db_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "explicit" / ".testmondata"
    override.parent.mkdir()
    _make_db(override)
    monkeypatch.setenv("TESTMON_SEED_DB", str(override))
    assert seed_testmon.find_seed_db(tmp_path) == override


def test_find_seed_db_env_override_unreadable_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TESTMON_SEED_DB", str(tmp_path / "gone.db"))
    # No git repo here either → the broken override must not hard-fail.
    assert seed_testmon.find_seed_db(tmp_path) is None


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"},
    )


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", str(path)],
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"},
    )
    (path / "file.txt").write_text("x\n")
    _git(path, "-c", "user.email=t@t", "-c", "user.name=t", "add", "file.txt")
    _git(path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    return path


def test_find_seed_db_returns_none_when_no_worktree_has_a_db(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    assert seed_testmon.find_seed_db(repo) is None


def test_find_seed_db_picks_freshest_worktree_db(tmp_path: Path) -> None:
    """Freshest wins: the primary's stale DB loses to a session worktree's live one."""
    repo = _init_repo(tmp_path / "repo")
    primary_db = repo / ".testmondata"
    _make_db(primary_db)
    old = 1_600_000_000  # 2020 — clearly the stale one
    os.utime(primary_db, (old, old))

    wt2 = tmp_path / "session-wt"
    _git(repo, "worktree", "add", "-q", str(wt2))
    live_db = wt2 / ".testmondata"
    _make_db(live_db)  # mtime = now → fresher

    assert seed_testmon.find_seed_db(repo) == live_db


def test_find_seed_db_worktree_paths_with_spaces(tmp_path: Path) -> None:
    """The real primary path is '.../Default Project/sloughGPT' — spaces must survive."""
    repo = _init_repo(tmp_path / "Default Project" / "sloughGPT")
    db = repo / ".testmondata"
    _make_db(db)

    assert seed_testmon.find_seed_db(repo) == db
