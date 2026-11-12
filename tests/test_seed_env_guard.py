"""Env-guard for the testmon seed (card 20261112_122610, follow-up to 20261111_125558).

A seed copy whose environment differs from the current interpreter is a
SILENT NO-OP with a FALSE success message: testmon's ``fetch_or_create_environment``
replaces the environment row on drift and ``test_execution.environment_id`` is
ON DELETE CASCADE — the copied rows are purged on the destination's first run,
and ``packages_change`` forces a full run anyway.

So discovery takes only recordings whose recorded environment fingerprint
(``system_packages`` + ``python_version``) equals what testmon will record in
THIS interpreter — computed with testmon's own ``drop_patch_version`` (storing
drops the patch level: ``PyJWT 2.14``, never ``PyJWT 2.14.0``). No match →
None → the honest full record (Test Rule clause 3); an unknown fingerprint on
either side falls back to freshness, never a hard fail. ``TESTMON_SEED_DB``
stays an explicit user override (warn on mismatch, still copy).

The other half: a valid testmon DB with 0 ``test_execution`` rows is the
post-cascade-wipe state — nothing to select from, re-seedable (``dest_state``).
"""

from __future__ import annotations

import os
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import seed_testmon  # noqa: E402

# What "the current interpreter's fingerprint" means in these tests — patched
# over seed_testmon._current_fingerprint so assertions don't depend on the
# machine's real site-packages.
CURRENT = ("downcraft 0.2, pytest 8.4, requests 2.32", "3.12.14")

TESTMON_TABLES = {
    "metadata",
    "environment",
    "test_execution",
    "file_fp",
    "test_execution_file_fp",
}


@pytest.fixture()
def current_fp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seed_testmon, "_current_fingerprint", lambda: CURRENT)


def _make_recording(path: Path, *, pkgs: str | None, py: str | None, rows: int = 1) -> None:
    """A testmon-schema DB with a controllable environment row + row count.

    ``pkgs=None`` leaves the environment table empty (unknown fingerprint);
    ``rows=0`` is the cascade-wiped state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        for t in sorted(TESTMON_TABLES):
            con.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY, v TEXT)")
        con.execute("ALTER TABLE environment ADD COLUMN system_packages TEXT")
        con.execute("ALTER TABLE environment ADD COLUMN python_version TEXT")
        con.execute("ALTER TABLE test_execution ADD COLUMN environment_id INTEGER")
        if pkgs is not None and py is not None:
            con.execute(
                "INSERT INTO environment (system_packages, python_version) VALUES (?, ?)",
                (pkgs, py),
            )
        for i in range(rows):
            con.execute(
                "INSERT INTO test_execution (environment_id, v) VALUES (1, ?)",
                (f"t{i}",),
            )
        con.commit()
    finally:
        con.close()


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


# ── discovery filter ─────────────────────────────────────────────────────────


def test_env_match_beats_fresher_mismatch(tmp_path: Path, current_fp: None) -> None:
    """A stale-but-matching recording seeds; a fresher doomed one never wins."""
    repo = _init_repo(tmp_path / "repo")
    old_db = repo / ".testmondata"
    _make_recording(old_db, pkgs=CURRENT[0], py=CURRENT[1])
    old = 1_600_000_000  # 2020 — the STALE one
    os.utime(old_db, (old, old))

    wt2 = tmp_path / "wt2"
    _git(repo, "worktree", "add", "-q", str(wt2))
    doomed_db = wt2 / ".testmondata"
    _make_recording(doomed_db, pkgs="something-else 9.9", py="3.11.0")  # fresher mtime

    assert seed_testmon.find_seed_db(repo) == old_db


def test_freshest_env_match_wins(tmp_path: Path, current_fp: None) -> None:
    """Among MATCHING recordings freshness still decides (the original rule)."""
    repo = _init_repo(tmp_path / "repo")
    stale_db = repo / ".testmondata"
    _make_recording(stale_db, pkgs=CURRENT[0], py=CURRENT[1])
    old = 1_600_000_000
    os.utime(stale_db, (old, old))

    wt2 = tmp_path / "wt2"
    _git(repo, "worktree", "add", "-q", str(wt2))
    live_db = wt2 / ".testmondata"
    _make_recording(live_db, pkgs=CURRENT[0], py=CURRENT[1])

    assert seed_testmon.find_seed_db(repo) == live_db


def test_no_env_match_returns_none(tmp_path: Path, current_fp: None) -> None:
    """Every candidate is doomed → None → honest full record, never a lie."""
    repo = _init_repo(tmp_path / "repo")
    _make_recording(repo / ".testmondata", pkgs="other-env 1.0", py="3.11.0")
    assert seed_testmon.find_seed_db(repo) is None


def test_unknown_env_falls_back_to_freshness(tmp_path: Path, current_fp: None) -> None:
    """No fingerprint on record (empty environment table) → freshness, not a hard fail."""
    repo = _init_repo(tmp_path / "repo")
    _make_recording(repo / ".testmondata", pkgs=None, py=None)
    assert seed_testmon.find_seed_db(repo) == repo / ".testmondata"


def test_empty_db_skipped_even_if_freshest(tmp_path: Path, current_fp: None) -> None:
    """A 0-row DB (cascade-wiped) can never be a seed source, however fresh."""
    repo = _init_repo(tmp_path / "repo")
    good_db = repo / ".testmondata"
    _make_recording(good_db, pkgs=CURRENT[0], py=CURRENT[1])
    old = 1_600_000_000
    os.utime(good_db, (old, old))

    wt2 = tmp_path / "wt2"
    _git(repo, "worktree", "add", "-q", str(wt2))
    _make_recording(wt2 / ".testmondata", pkgs=CURRENT[0], py=CURRENT[1], rows=0)

    assert seed_testmon.find_seed_db(repo) == good_db


def test_only_empty_dbs_returns_none(tmp_path: Path, current_fp: None) -> None:
    repo = _init_repo(tmp_path / "repo")
    _make_recording(repo / ".testmondata", pkgs=CURRENT[0], py=CURRENT[1], rows=0)
    assert seed_testmon.find_seed_db(repo) is None


def test_override_env_mismatch_still_returned(
    tmp_path: Path, current_fp: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TESTMON_SEED_DB is explicit user intent — warn on mismatch, never block."""
    override = tmp_path / "explicit" / ".testmondata"
    _make_recording(override, pkgs="different-env 1.0", py="3.9.0")
    monkeypatch.setenv("TESTMON_SEED_DB", str(override))
    assert seed_testmon.find_seed_db(tmp_path) == override


# ── current-fingerprint computation ──────────────────────────────────────────


def test_current_fingerprint_matches_testmon_storage_format() -> None:
    """Same helpers testmon stores with: patch-dropped packages, full python."""
    fp = seed_testmon._current_fingerprint()
    assert fp is not None, "testmon is installed in the project env — must compute"
    pkgs, py = fp
    assert py == platform.python_version()
    assert pkgs, "non-empty package fingerprint"
    # Storage normalizes 'X 1.2.3' → 'X 1.2' (testmon drops the patch level);
    # comparing raw versions would never match any recorded environment.
    from testmon.common import drop_patch_version, get_system_packages

    assert pkgs == drop_patch_version(get_system_packages())


# ── dest health (the wiped-DB re-seed states) ────────────────────────────────


def test_dest_state_missing(tmp_path: Path) -> None:
    assert seed_testmon.dest_state(tmp_path / ".testmondata") == "missing"


def test_dest_state_invalid(tmp_path: Path) -> None:
    garbage = tmp_path / ".testmondata"
    garbage.write_text("not sqlite at all")
    assert seed_testmon.dest_state(garbage) == "invalid"


def test_dest_state_empty(tmp_path: Path) -> None:
    db = tmp_path / ".testmondata"
    _make_recording(db, pkgs=CURRENT[0], py=CURRENT[1], rows=0)
    assert seed_testmon.dest_state(db) == "empty"


def test_dest_state_healthy(tmp_path: Path) -> None:
    db = tmp_path / ".testmondata"
    _make_recording(db, pkgs=CURRENT[0], py=CURRENT[1], rows=3)
    assert seed_testmon.dest_state(db) == "healthy"
