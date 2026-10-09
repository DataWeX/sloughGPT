"""Regression: the shared DB pool must resolve the real repo root.

``db_pool._get_repo_root()`` hardcoded ``parents[5]`` from
``apps/api/server/infrastructure/db_pool.py``.  That path is 5 levels deep
below the repo root, so ``parents[5]`` lands one directory ABOVE the repo
(``.../Default Project``) — every pooled MogDB store (``uploads_mogdb``,
``companion_mogdb``, ``experiments_*``, ``auth_mogdb``) was created and
written outside the repository, invisible to git and to in-repo backups,
while other writers (``auth.py``, ``docstore.py``) kept using the real
``<repo>/data``.  The result was a split brain: the same logical store lived
in two places and readers saw whichever half their resolver picked.

The depth from this file to the repo root is 4 (``infrastructure`` ->
``server`` -> ``api`` -> ``apps`` -> root), so ``parents[5]`` was off by one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from infrastructure import db_pool

_TRUE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_cached_root(monkeypatch):
    """``_REPO_ROOT``/``_POOL`` are process-global — clear them around each test."""
    monkeypatch.setattr(db_pool, "_REPO_ROOT", None)
    monkeypatch.setattr(db_pool, "_POOL", {})
    yield
    monkeypatch.setattr(db_pool, "_REPO_ROOT", None)
    monkeypatch.setattr(db_pool, "_POOL", {})


def test_repo_root_is_the_real_repo_root() -> None:
    got = db_pool._get_repo_root()
    assert got == _TRUE_ROOT, f"_get_repo_root() resolved to {got}, expected {_TRUE_ROOT}"
    for marker in ("apps", "domain", "packages"):
        assert (got / marker).is_dir(), f"repo root {got} is missing {marker}/"


def test_repo_root_is_not_a_parent_of_the_repo() -> None:
    """The old off-by-one landed *above* the repo — pin that it never does."""
    got = db_pool._get_repo_root()
    assert got != _TRUE_ROOT.parent, (
        "resolved one directory above the repo root (the parents[5] off-by-one)"
    )
    assert got.is_relative_to(_TRUE_ROOT.parent) and got != _TRUE_ROOT.parent


def test_get_db_paths_land_inside_repo(monkeypatch, tmp_path) -> None:
    """Both the DB path and its sync dir must live under <repo>/data."""
    captured: dict[str, str] = {}

    class _FakeDB:
        def __init__(self, db_path: str, *, sync_dir: str | None = None) -> None:
            captured["db_path"] = db_path
            captured["sync_dir"] = sync_dir

    import mogdb

    monkeypatch.setattr(mogdb, "MogDB", _FakeDB)

    db_pool.get_db("uploads_mogdb")

    for key in ("db_path", "sync_dir"):
        value = Path(captured[key])
        assert value.is_relative_to(_TRUE_ROOT / "data"), f"{key}={value} escaped <repo>/data"


def test_get_db_reuses_the_singleton(monkeypatch) -> None:
    """The pool must return the same instance for the same (db, sync) pair."""
    instances: list[object] = []

    class _FakeDB:
        def __init__(self, db_path: str, *, sync_dir: str | None = None) -> None:
            instances.append(self)

    import mogdb

    monkeypatch.setattr(mogdb, "MogDB", _FakeDB)

    first = db_pool.get_db("uploads_mogdb")
    second = db_pool.get_db("uploads_mogdb")
    assert first is second
    assert len(instances) == 1, "second call constructed a new MogDB"


def test_module_is_importable_without_side_effects() -> None:
    """Importing db_pool must not create or touch any data directory."""
    assert "infrastructure.db_pool" in sys.modules
    assert db_pool._POOL == {}, "importing db_pool populated the pool"
