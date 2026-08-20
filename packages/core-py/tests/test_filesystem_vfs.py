"""Tests for domains.shell.addons.filesystem — VFSEntry, VFSGeneratedFile, VFSDirectory, VFS."""

import os
import stat as stat_mod
import tempfile
from pathlib import Path

import pytest
from domains.shell.addons.filesystem import (
    VFSEntry, VFSGeneratedFile, VFSWriteOnlyFile, VFSDirectory, VFS,
)


class TestVFSEntry:
    def test_defaults(self):
        e = VFSEntry("test.txt")
        assert e.name == "test.txt"
        assert e.is_dir is False
        assert e.size == 0

    def test_read_returns_empty(self):
        e = VFSEntry("x")
        assert e.read() == ""

    def test_write_noop(self):
        e = VFSEntry("x")
        e.write("data")  # should not raise

    def test_custom_mode_and_size(self):
        e = VFSEntry("x", mode=0o666, size=42)
        assert e.mode == 0o666
        assert e.size == 42


class TestVFSGeneratedFile:
    def test_read_calls_fn(self):
        f = VFSGeneratedFile("uptime", lambda: "123.45\n")
        assert f.read() == "123.45\n"

    def test_is_not_dir(self):
        f = VFSGeneratedFile("x", lambda: "")
        assert f.is_dir is False


class TestVFSWriteOnlyFile:
    def test_write_calls_fn(self):
        received = []
        f = VFSWriteOnlyFile("input", lambda d: received.append(d) or None)
        f.write("hello")
        assert received == ["hello"]

    def test_read_returns_empty(self):
        f = VFSWriteOnlyFile("x", lambda d: None)
        assert f.read() == ""


class TestVFSDirectory:
    def test_is_dir(self):
        d = VFSDirectory("root")
        assert d.is_dir is True

    def test_list_empty(self):
        d = VFSDirectory("root")
        assert d.list() == []

    def test_add_and_list(self):
        d = VFSDirectory("root")
        d.add(VFSEntry("a.txt"))
        d.add(VFSEntry("b.txt"))
        assert d.list() == ["a.txt", "b.txt"]

    def test_get(self):
        d = VFSDirectory("root")
        d.add(VFSEntry("f"))
        assert d.get("f") is not None
        assert d.get("missing") is None

    def test_add_subdirectory(self):
        root = VFSDirectory("root")
        sub = VFSDirectory("sub")
        sub.add(VFSEntry("file.txt"))
        root.add(sub)
        assert root.get("sub").is_dir is True
        assert root.get("sub").list() == ["file.txt"]


class TestVFS:
    def test_mount_and_list(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("llm"))
        vfs.mount("/dev", dev)
        assert "/dev" in vfs.mounts()

    def test_unmount(self):
        vfs = VFS()
        vfs.mount("/tmp", VFSDirectory("tmp"))
        assert vfs.unmount("/tmp") is True
        assert "/tmp" not in vfs.mounts()

    def test_unmount_nonexistent(self):
        vfs = VFS()
        assert vfs.unmount("/nope") is False

    def test_resolve_mounted_path(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("llm"))
        vfs.mount("/dev", dev)
        result = vfs._resolve("/dev/llm")
        assert result is not None
        mount_obj, rel = result
        assert isinstance(mount_obj, VFSDirectory)
        assert rel == "llm"

    def test_resolve_exact_mount(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        vfs.mount("/dev", dev)
        result = vfs._resolve("/dev")
        assert result is not None
        mount_obj, rel = result
        assert mount_obj is dev
        assert rel == ""

    def test_read_from_vfs(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        f = VFSGeneratedFile("version", lambda: "1.0")
        dev.add(f)
        vfs.mount("/dev", dev)
        assert vfs.read("/dev/version") == "1.0"

    def test_read_file_not_found(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        vfs.mount("/dev", dev)
        assert vfs.read("/dev/missing") is None

    def test_listdir_vfs(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("a"))
        dev.add(VFSEntry("b"))
        vfs.mount("/dev", dev)
        assert vfs.listdir("/dev") == ["a", "b"]

    def test_listdir_subdir(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        sub = VFSDirectory("sub")
        sub.add(VFSEntry("x"))
        dev.add(sub)
        vfs.mount("/dev", dev)
        assert vfs.listdir("/dev/sub") == ["x"]

    def test_exists_vfs(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("f"))
        vfs.mount("/dev", dev)
        assert vfs.exists("/dev/f") is True
        assert vfs.exists("/dev/missing") is False

    def test_write_to_vfs(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        received = []
        f = VFSWriteOnlyFile("input", lambda d: received.append(d) or None)
        dev.add(f)
        vfs.mount("/dev", dev)
        result = vfs.write("/dev/input", "data")
        assert result is None
        assert received == ["data"]

    def test_write_to_dir_returns_error(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        vfs.mount("/dev", dev)
        result = vfs.write("/dev", "data")
        assert "directory" in result.lower()

    def test_stat_dir(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        vfs.mount("/dev", dev)
        s = vfs.stat("/dev")
        assert s is not None
        assert stat_mod.S_ISDIR(s.st_mode)

    def test_stat_file(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("f", size=100))
        vfs.mount("/dev", dev)
        s = vfs.stat("/dev/f")
        assert s is not None
        assert stat_mod.S_ISREG(s.st_mode)

    def test_stat_nonexistent(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        vfs.mount("/dev", dev)
        assert vfs.stat("/dev/missing") is None

    def test_isfile(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("f"))
        vfs.mount("/dev", dev)
        assert vfs.isfile("/dev/f") is True
        assert vfs.isfile("/dev") is False

    def test_isdir(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("f"))
        vfs.mount("/dev", dev)
        assert vfs.isdir("/dev") is True
        assert vfs.isdir("/dev/f") is False

    def test_fallback_to_real_fs(self):
        vfs = VFS()
        assert vfs.exists("/tmp") is True
        assert vfs.isdir("/tmp") is True

    def test_read_fallback(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            path = f.name
        try:
            vfs = VFS()
            content = vfs.read(path)
            assert content == "test content"
        finally:
            os.unlink(path)

    def test_write_fallback(self):
        path = tempfile.mktemp(suffix=".txt")
        try:
            vfs = VFS()
            vfs.write(path, "hello world")
            assert Path(path).read_text() == "hello world"
        finally:
            if os.path.exists(path):
                os.unlink(path)
