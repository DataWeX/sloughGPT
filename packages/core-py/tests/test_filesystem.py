"""Tests for domains.shell.addons.filesystem — VFS, VFSEntry, VFSDirectory."""

from __future__ import annotations

import os
import pytest
from pathlib import Path

from domains.shell.addons.filesystem import (
    VFSEntry,
    VFSGeneratedFile,
    VFSWriteOnlyFile,
    VFSDirectory,
    VFS,
    _dir_stat,
    _file_stat,
)


# ── VFSEntry ──────────────────────────────────────────────────────────────────

class TestVFSEntry:
    def test_defaults(self):
        e = VFSEntry("test.txt")
        assert e.name == "test.txt"
        assert e.is_dir is False
        assert e.size == 0
        assert e.read() == ""

    def test_write_noop(self):
        e = VFSEntry("test.txt")
        e.write("data")  # should not raise

    def test_directory(self):
        e = VFSEntry("dir", is_dir=True)
        assert e.is_dir is True


# ── VFSGeneratedFile ─────────────────────────────────────────────────────────

class TestVFSGeneratedFile:
    def test_read_calls_fn(self):
        f = VFSGeneratedFile("gen.txt", lambda: "generated content")
        assert f.read() == "generated content"

    def test_is_not_dir(self):
        f = VFSGeneratedFile("gen.txt", lambda: "")
        assert f.is_dir is False


# ── VFSWriteOnlyFile ─────────────────────────────────────────────────────────

class TestVFSWriteOnlyFile:
    def test_write_calls_fn(self):
        written = []
        f = VFSWriteOnlyFile("out.txt", lambda d: written.append(d) or None)
        f.write("hello")
        assert written == ["hello"]

    def test_read_returns_empty(self):
        f = VFSWriteOnlyFile("out.txt", lambda d: None)
        assert f.read() == ""


# ── VFSDirectory ──────────────────────────────────────────────────────────────

class TestVFSDirectory:
    def test_is_dir(self):
        d = VFSDirectory("root")
        assert d.is_dir is True

    def test_list_empty(self):
        d = VFSDirectory("root")
        assert d.list() == []

    def test_add_and_get(self):
        d = VFSDirectory("root")
        e = VFSEntry("file.txt")
        d.add(e)
        assert d.get("file.txt") is e
        assert d.get("missing") is None

    def test_list_sorted(self):
        d = VFSDirectory("root")
        d.add(VFSEntry("c.txt"))
        d.add(VFSEntry("a.txt"))
        d.add(VFSEntry("b.txt"))
        assert d.list() == ["a.txt", "b.txt", "c.txt"]

    def test_nested_directory(self):
        root = VFSDirectory("root")
        sub = VFSDirectory("sub")
        sub.add(VFSEntry("file.txt"))
        root.add(sub)
        assert root.get("sub").get("file.txt").name == "file.txt"


# ── VFS ───────────────────────────────────────────────────────────────────────

class TestVFS:
    def test_mount_unmount(self):
        vfs = VFS()
        d = VFSDirectory("test")
        vfs.mount("/test", d)
        assert "/test" in vfs.mounts()
        assert vfs.unmount("/test") is True
        assert "/test" not in vfs.mounts()

    def test_unmount_nonexistent(self):
        vfs = VFS()
        assert vfs.unmount("/nope") is False

    def test_exists_in_mount(self):
        vfs = VFS()
        d = VFSDirectory("dev")
        d.add(VFSEntry("cpu"))
        vfs.mount("/dev", d)
        assert vfs.exists("/dev") is True
        assert vfs.exists("/dev/cpu") is True
        assert vfs.exists("/dev/missing") is False

    def test_listdir_mount(self):
        vfs = VFS()
        d = VFSDirectory("dev")
        d.add(VFSEntry("a"))
        d.add(VFSEntry("b"))
        vfs.mount("/dev", d)
        assert vfs.listdir("/dev") == ["a", "b"]

    def test_read_from_mount(self):
        vfs = VFS()
        f = VFSGeneratedFile("uptime", lambda: "123.45")
        d = VFSDirectory("proc")
        d.add(f)
        vfs.mount("/proc", d)
        assert vfs.read("/proc/uptime") == "123.45"

    def test_read_nonexistent(self):
        vfs = VFS()
        assert vfs.read("/nonexistent") is None

    def test_write_to_mount(self):
        vfs = VFS()
        written = []
        f = VFSWriteOnlyFile("log", lambda d: written.append(d) or None)
        d = VFSDirectory("dev")
        d.add(f)
        vfs.mount("/dev", d)
        assert vfs.write("/dev/log", "hello") is None
        assert written == ["hello"]

    def test_read_real_file(self, tmp_path):
        vfs = VFS()
        f = tmp_path / "test.txt"
        f.write_text("real content")
        assert vfs.read(str(f)) == "real content"

    def test_write_real_file(self, tmp_path):
        vfs = VFS()
        f = tmp_path / "out.txt"
        assert vfs.write(str(f), "written") is None
        assert f.read_text() == "written"

    def test_stat_mount_file(self):
        vfs = VFS()
        f = VFSEntry("file.txt", size=100)
        d = VFSDirectory("dev")
        d.add(f)
        vfs.mount("/dev", d)
        s = vfs.stat("/dev/file.txt")
        assert s is not None
        assert s.st_size == 100

    def test_stat_dir(self):
        vfs = VFS()
        d = VFSDirectory("dev")
        vfs.mount("/dev", d)
        s = vfs.stat("/dev")
        assert s is not None
        assert os.path.isdir(str(s.st_mode)) is False  # S_ISDIR check

    def test_isfile(self):
        vfs = VFS()
        f = VFSEntry("file.txt")
        d = VFSDirectory("dev")
        d.add(f)
        vfs.mount("/dev", d)
        assert vfs.isfile("/dev/file.txt") is True
        assert vfs.isfile("/dev") is False

    def test_isdir(self):
        vfs = VFS()
        d = VFSDirectory("dev")
        vfs.mount("/dev", d)
        assert vfs.isdir("/dev") is True
        assert vfs.isdir("/dev/nonexistent") is False

    def test_write_to_dir_returns_error(self):
        vfs = VFS()
        d = VFSDirectory("dev")
        vfs.mount("/dev", d)
        assert vfs.write("/dev", "data") == "Is a directory"

    def test_nested_path(self):
        vfs = VFS()
        root = VFSDirectory("root")
        sub = VFSDirectory("sub")
        sub.add(VFSEntry("file.txt"))
        root.add(sub)
        vfs.mount("/mnt", root)
        assert vfs.read("/mnt/sub/file.txt") == ""

    def test_resolve_root_exact(self):
        vfs = VFS()
        d = VFSDirectory("root")
        vfs.mount("/mnt", d)
        assert vfs.exists("/mnt") is True


# ── _dir_stat / _file_stat ───────────────────────────────────────────────────

class TestStatHelpers:
    def test_dir_stat(self):
        s = _dir_stat()
        assert s.st_size == 4096

    def test_file_stat(self):
        s = _file_stat(1234)
        assert s.st_size == 1234
