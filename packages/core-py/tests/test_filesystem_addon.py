"""
Tests for the filesystem addon — VFS mount points, path resolution, read/write.
"""

import os
import tempfile
import pytest

from domains.shell.kernel import Kernel
from domains.shell.addons import filesystem
from domains.shell.addons.filesystem import (
    VFS, VFSEntry, VFSDirectory, VFSGeneratedFile, VFSWriteOnlyFile,
)


# ---------------------------------------------------------------------------
# VFSEntry / VFSDirectory unit tests
# ---------------------------------------------------------------------------

class TestVFSEntry:
    def test_default_properties(self):
        e = VFSEntry("test.txt")
        assert e.name == "test.txt"
        assert e.is_dir is False
        assert e.size == 0
        assert e.read() == ""

    def test_write_noop(self):
        e = VFSEntry("test.txt")
        e.write("data")  # should not raise


class TestVFSGeneratedFile:
    def test_read_calls_fn(self):
        f = VFSGeneratedFile("uptime", lambda: "123.45\n")
        assert f.read() == "123.45\n"

    def test_generated_file_not_dir(self):
        f = VFSGeneratedFile("x", lambda: "")
        assert f.is_dir is False


class TestVFSWriteOnlyFile:
    def test_write_calls_fn(self):
        results = []
        f = VFSWriteOnlyFile("control", lambda d: results.append(d) or None)
        f.write("hello")
        assert results == ["hello"]

    def test_read_returns_empty(self):
        f = VFSWriteOnlyFile("control", lambda d: None)
        assert f.read() == ""


class TestVFSDirectory:
    def test_list_sorted(self):
        d = VFSDirectory("root")
        d.add(VFSEntry("b.txt"))
        d.add(VFSEntry("a.txt"))
        assert d.list() == ["a.txt", "b.txt"]

    def test_get_existing(self):
        d = VFSDirectory("root")
        e = VFSEntry("file.txt")
        d.add(e)
        assert d.get("file.txt") is e

    def test_get_missing(self):
        d = VFSDirectory("root")
        assert d.get("nope") is None

    def test_is_dir(self):
        d = VFSDirectory("root")
        assert d.is_dir is True


# ---------------------------------------------------------------------------
# VFS path resolution and I/O
# ---------------------------------------------------------------------------

class TestVFS:
    def test_mount_and_listdir(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("llm"))
        dev.add(VFSEntry("embed"))
        vfs.mount("/dev", dev)

        assert vfs.isdir("/dev") is True
        assert vfs.listdir("/dev") == ["embed", "llm"]

    def test_read_generated_file(self):
        vfs = VFS()
        proc = VFSDirectory("proc")
        proc.add(VFSGeneratedFile("uptime", lambda: "99.99\n"))
        vfs.mount("/proc", proc)

        assert vfs.read("/proc/uptime") == "99.99\n"
        assert vfs.exists("/proc/uptime") is True

    def test_read_nonexistent_file(self):
        vfs = VFS()
        proc = VFSDirectory("proc")
        vfs.mount("/proc", proc)
        assert vfs.read("/proc/nope") is None

    def test_read_real_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        vfs = VFS()
        assert vfs.read(str(f)) == "hello world"

    def test_write_real_file(self, tmp_path):
        f = tmp_path / "out.txt"
        vfs = VFS()
        err = vfs.write(str(f), "data")
        assert err is None
        assert f.read_text() == "data"

    def test_stat_virtual_directory(self):
        import stat as stat_mod
        vfs = VFS()
        dev = VFSDirectory("dev")
        vfs.mount("/dev", dev)
        s = vfs.stat("/dev")
        assert s is not None
        assert stat_mod.S_ISDIR(s.st_mode)

    def test_stat_virtual_file(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("llm", size=42))
        vfs.mount("/dev", dev)
        s = vfs.stat("/dev/llm")
        assert s is not None
        assert s.st_size == 42

    def test_isfile_virtual(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("llm"))
        vfs.mount("/dev", dev)
        assert vfs.isfile("/dev/llm") is True
        assert vfs.isfile("/dev") is False

    def test_nested_virtual_directory(self):
        vfs = VFS()
        root = VFSDirectory("root")
        sub = VFSDirectory("sub")
        sub.add(VFSEntry("deep.txt"))
        root.add(sub)
        vfs.mount("/mnt", root)

        assert vfs.exists("/mnt/sub/deep.txt") is True
        assert vfs.read("/mnt/sub/deep.txt") == ""
        assert vfs.listdir("/mnt/sub") == ["deep.txt"]

    def test_unmount(self):
        vfs = VFS()
        vfs.mount("/virtual_only", VFSDirectory("virtual_only"))
        assert vfs.exists("/virtual_only") is True
        assert vfs.unmount("/virtual_only") is True
        assert vfs.exists("/virtual_only") is False

    def test_unmount_nonexistent(self):
        vfs = VFS()
        assert vfs.unmount("/nope") is False

    def test_mounts_lists_all(self):
        vfs = VFS()
        vfs.mount("/a", VFSDirectory("a"))
        vfs.mount("/b", VFSDirectory("b"))
        assert vfs.mounts() == ["/a", "/b"]


# ---------------------------------------------------------------------------
# Kernel addon integration
# ---------------------------------------------------------------------------

class TestKernelFilesystemAddon:
    def _booted_kernel(self):
        from domains.shell.kernel import Kernel
        k = Kernel()
        k.boot()
        from domains.shell.addons import filesystem
        filesystem.setup(k)
        return k

    def test_get_kernel_has_filesystem_addon(self):
        k = self._booted_kernel()
        assert "filesystem" in k._addons
        assert hasattr(k, "_vfs")
        assert isinstance(k.vfs, VFS)

    def test_vfs_property_requires_addon(self):
        k = Kernel()
        k.boot()
        # Bare kernel should auto-install, but test the guard pattern
        if "filesystem" in k._addons:
            assert k.vfs is not None
        else:
            with pytest.raises(RuntimeError, match="filesystem"):
                _ = k.vfs

    def test_vfs_has_proc_mount(self):
        k = self._booted_kernel()
        mounts = k.vfs.mounts()
        assert "/proc" in mounts

    def test_vfs_read_proc_uptime(self):
        k = self._booted_kernel()
        content = k.vfs.read("/proc/uptime")
        assert content is not None
        assert float(content.strip().split()[0]) >= 0

    def test_vfs_read_proc_version(self):
        k = self._booted_kernel()
        content = k.vfs.read("/proc/version")
        assert "Dait" in content or "Linux" in content

    def test_vfs_is_singleton_on_kernel(self):
        k = self._booted_kernel()
        assert k.vfs is k.vfs  # property returns same VFS instance


# ---------------------------------------------------------------------------
# setup() function
# ---------------------------------------------------------------------------

class TestSetup:
    def test_setup_installs_vfs(self):
        k = Kernel()
        k.boot()
        filesystem.setup(k)
        assert "filesystem" in k._addons
        assert hasattr(k, "_vfs")
        assert isinstance(k._vfs, VFS)

    def test_setup_idempotent(self):
        k = Kernel()
        k.boot()
        filesystem.setup(k)
        old_vfs = k._vfs
        old_addons = dict(k._addons)
        filesystem.setup(k)  # second call should be no-op due to addon guard
        assert k._vfs is old_vfs
        assert k._addons == old_addons

    def test_setup_skips_if_already_installed(self):
        k = Kernel()
        k.boot()
        k._addons["filesystem"] = True
        k._vfs = VFS()
        old_vfs = k._vfs
        filesystem.setup(k)
        assert k._vfs is old_vfs
