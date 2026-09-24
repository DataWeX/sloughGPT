"""Unit tests for the shared MogDB connection pool root resolution.

Covers the parents[5] off-by-one fix (card 055): repo-root discovery must be
marker-based, and pre-fix data outside the repo must be adopted by copy-once
semantics that never destroy the legacy location.
"""

from __future__ import annotations

import shutil

import pytest
from infrastructure import db_pool


@pytest.fixture(autouse=True)
def _reset_caches():
    db_pool._REPO_ROOT = None
    db_pool._LEGACY_ROOT = None
    yield
    db_pool._REPO_ROOT = None
    db_pool._LEGACY_ROOT = None


class TestRepoRoot:
    def test_root_has_apps_and_packages_markers(self):
        root = db_pool._get_repo_root()
        assert (root / "apps").is_dir()
        assert (root / "packages").is_dir()

    def test_root_is_cached(self):
        first = db_pool._get_repo_root()
        db_pool._REPO_ROOT = first / "apps"
        assert db_pool._get_repo_root() == first / "apps"

    def test_legacy_root_is_above_repo(self):
        # pre-fix behavior: parents[5] from db_pool.py = parent of repo root
        assert db_pool._get_legacy_root() == db_pool._get_repo_root().parent


class TestAdoptLegacyDir:
    def test_copies_when_target_missing(self, tmp_path):
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "presets.json").write_text('{"n": 1}')

        target = tmp_path / "new" / "presets"
        result = db_pool._adopt_legacy_dir(target, legacy)

        assert result == str(target)
        assert (target / "presets.json").read_text() == '{"n": 1}'
        # legacy untouched (copy, never move)
        assert (legacy / "presets.json").exists()

    def test_copies_when_target_empty(self, tmp_path):
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "a.json").write_text("{}")
        target = tmp_path / "new"
        target.mkdir()  # exists but empty

        db_pool._adopt_legacy_dir(target, legacy)

        assert (target / "a.json").exists()

    def test_never_overwrites_nonempty_target(self, tmp_path):
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "a.json").write_text('{"from": "legacy"}')
        target = tmp_path / "new"
        target.mkdir()
        (target / "a.json").write_text('{"from": "target"}')

        result = db_pool._adopt_legacy_dir(target, legacy)

        assert result == str(target)
        assert (target / "a.json").read_text() == '{"from": "target"}'
        assert (legacy / "a.json").read_text() == '{"from": "legacy"}'

    def test_noop_when_legacy_absent(self, tmp_path):
        target = tmp_path / "new"
        result = db_pool._adopt_legacy_dir(target, tmp_path / "missing")
        assert result == str(target)
        assert not target.exists()

    def test_noop_when_same_path(self, tmp_path):
        (tmp_path / "a.json").write_text("{}")
        result = db_pool._adopt_legacy_dir(tmp_path, tmp_path)
        assert result == str(tmp_path)
        assert (tmp_path / "a.json").exists()

    def test_falls_back_to_legacy_on_copy_failure(self, tmp_path, monkeypatch):
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "a.json").write_text("{}")
        target = tmp_path / "new"

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(shutil, "copytree", boom)

        result = db_pool._adopt_legacy_dir(target, legacy)

        assert result == str(legacy)
        assert not target.exists()
