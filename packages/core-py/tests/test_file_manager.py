"""Tests for domains.shell.file_manager — unified VFS + host FS resolution."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domains.shell.file_manager import (
    FileManager,
    get_file_manager,
    reset_file_manager,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global singleton before each test."""
    reset_file_manager()
    yield
    reset_file_manager()


class TestFileManagerReadText:
    """read_text: VFS priority → host FS fallback."""

    def test_reads_from_host_fs(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("world")
        fm = FileManager()
        assert fm.read_text(str(f)) == "world"

    def test_returns_none_for_missing_file(self, tmp_path):
        fm = FileManager()
        assert fm.read_text(str(tmp_path / "nope.txt")) is None

    def test_vfs_priority(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("host")
        vfs = MagicMock()
        vfs.read.return_value = "vfs_content"
        fm = FileManager()
        fm._vfs = vfs
        result = fm.read_text(str(f))
        assert result == "vfs_content"
        vfs.read.assert_called_once_with(str(f))

    def test_vfs_falls_through_when_none(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("from_disk")
        vfs = MagicMock()
        vfs.read.return_value = None
        fm = FileManager()
        fm._vfs = vfs
        result = fm.read_text(str(f))
        assert result == "from_disk"

    def test_expands_tilde(self, monkeypatch):
        monkeypatch.setenv("HOME", "/tmp")
        fm = FileManager()
        with patch.object(Path, "read_text", return_value="tilde"):
            result = fm.read_text("~/somefile")
        assert result == "tilde"


class TestFileManagerReadBytes:
    """read_bytes: host FS only (VFS is text-only)."""

    def test_reads_binary(self, tmp_path):
        f = tmp_path / "bin.dat"
        f.write_bytes(b"\x00\x01\x02")
        fm = FileManager()
        assert fm.read_bytes(str(f)) == b"\x00\x01\x02"

    def test_returns_none_for_missing(self, tmp_path):
        fm = FileManager()
        assert fm.read_bytes(str(tmp_path / "nope.bin")) is None


class TestFileManagerWriteText:
    """write_text: VFS priority → host FS fallback."""

    def test_writes_to_host_fs(self, tmp_path):
        target = tmp_path / "sub" / "out.txt"
        fm = FileManager()
        result = fm.write_text(str(target), "content")
        assert result is None
        assert target.read_text() == "content"

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.txt"
        fm = FileManager()
        fm.write_text(str(target), "deep")
        assert target.read_text() == "deep"

    def test_returns_error_string_on_failure(self, tmp_path):
        fm = FileManager()
        result = fm.write_text("/nonexistent_dir_abc123/file.txt", "data")
        assert result is not None
        assert isinstance(result, str)

    def test_vfs_write_success(self):
        vfs = MagicMock()
        vfs.write.return_value = None
        fm = FileManager()
        fm._vfs = vfs
        result = fm.write_text("/vfs/file", "data")
        assert result is None
        vfs.write.assert_called_once_with("/vfs/file", "data")

    def test_vfs_write_falls_through(self, tmp_path):
        f = tmp_path / "fallback.txt"
        vfs = MagicMock()
        vfs.write.return_value = "vfs_error"
        fm = FileManager()
        fm._vfs = vfs
        result = fm.write_text(str(f), "data")
        assert result is None
        assert f.read_text() == "data"


class TestFileManagerExists:
    def test_exists_true(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("x")
        fm = FileManager()
        assert fm.exists(str(f)) is True

    def test_exists_false(self, tmp_path):
        fm = FileManager()
        assert fm.exists(str(tmp_path / "nope.txt")) is False

    def test_vfs_checked_first(self):
        vfs = MagicMock()
        vfs.exists.return_value = True
        fm = FileManager()
        fm._vfs = vfs
        assert fm.exists("/anything") is True


class TestFileManagerIsFile:
    def test_isfile_true(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        fm = FileManager()
        assert fm.isfile(str(f)) is True

    def test_isfile_false_for_dir(self, tmp_path):
        fm = FileManager()
        assert fm.isfile(str(tmp_path)) is False


class TestFileManagerIsDir:
    def test_isdir_true(self, tmp_path):
        fm = FileManager()
        assert fm.isdir(str(tmp_path)) is True

    def test_isdir_false_for_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        fm = FileManager()
        assert fm.isdir(str(f)) is False


class TestFileManagerListdir:
    def test_listdir(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        fm = FileManager()
        result = fm.listdir(str(tmp_path))
        assert result == ["a.txt", "b.txt"]

    def test_listdir_none_for_missing(self, tmp_path):
        fm = FileManager()
        assert fm.listdir(str(tmp_path / "nope")) is None


class TestFileManagerResolve:
    def test_resolve_existing(self, tmp_path):
        f = tmp_path / "real.txt"
        f.write_text("x")
        fm = FileManager()
        result = fm.resolve(str(f))
        assert result == str(f)

    def test_resolve_none_for_missing(self, tmp_path):
        fm = FileManager()
        assert fm.resolve(str(tmp_path / "nope.txt")) is None


class TestGetVfsFallback:
    def test_import_error_sets_sentinel(self):
        fm = FileManager()
        with patch.dict("sys.modules", {"domains.shell.vfs": None}):
            result = fm._get_vfs()
        assert result is None
        assert fm._vfs is False  # sentinel

    def test_sentinel_not_retried(self):
        fm = FileManager()
        fm._vfs = False
        assert fm._get_vfs() is None


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = get_file_manager()
        b = get_file_manager()
        assert a is b

    def test_reset_creates_new(self):
        a = get_file_manager()
        reset_file_manager()
        b = get_file_manager()
        assert a is not b
