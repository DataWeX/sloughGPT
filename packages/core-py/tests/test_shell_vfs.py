"""Tests for Dait Virtual File System."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

from domains.shell.vfs import (
    VFS,
    VFSDirectory,
    VFSEntry,
    VFSGeneratedFile,
    VFSWriteOnlyFile,
    get_vfs,
    reset_vfs,
    _dir_stat,
    _file_stat,
)


@pytest.fixture
def vfs():
    reset_vfs()
    v = get_vfs()
    yield v
    reset_vfs()


@pytest.fixture
def vfs_with_devices_and_kernel(vfs):
    from domains.shell.devices import DeviceManager, NullDevice, RandomDevice
    dm = DeviceManager()
    dm.register(NullDevice())
    dm.register(RandomDevice())
    vfs.set_devices(dm)

    class FakeKernel:
        uptime = 42.0
        def list_processes(self):
            return []
        @property
        def memory_usage_str(self):
            return "4K"

    vfs.set_kernel(FakeKernel())
    return vfs


class TestVFSEntry:
    def test_basic_entry(self):
        e = VFSEntry("test", mode=0o444, size=100)
        assert e.name == "test"
        assert e.mode == 0o444
        assert e.size == 100
        assert not e.is_dir
        assert e.read() == ""
        e.write("data")  # should not raise

    def test_dir_entry(self):
        e = VFSEntry("dir", is_dir=True)
        assert e.is_dir

    def test_generated_file(self):
        e = VFSGeneratedFile("gen", lambda: "hello\n")
        assert e.read() == "hello\n"

    def test_write_only_file(self):
        results = []
        def writer(data):
            results.append(data)
        e = VFSWriteOnlyFile("w", writer)
        e.write("test_data")
        assert results == ["test_data"]


class TestVFSDirectory:
    def test_empty(self):
        d = VFSDirectory("empty")
        assert d.list() == []

    def test_add_and_get(self):
        d = VFSDirectory("root")
        e = VFSEntry("foo")
        d.add(e)
        assert d.get("foo") is e
        assert d.get("bar") is None

    def test_list_sorted(self):
        d = VFSDirectory("root")
        d.add(VFSEntry("z"))
        d.add(VFSEntry("a"))
        d.add(VFSEntry("m"))
        assert d.list() == ["a", "m", "z"]


class TestVFS:
    def test_default_state(self, vfs):
        assert vfs.listdir("/nonexistent_virtual") is None
        assert vfs.read("/nonexistent_virtual") is None
        # writing to a nonexistent path falls through to real FS
        # real FS write attempt may fail depending on platform
        err = vfs.write("/nonexistent_virtual", "x")
        assert err is not None  # real FS says no

    def test_exists_falls_through_to_real_fs(self, vfs):
        assert vfs.exists("/tmp")
        assert not vfs.exists("/nonexistent_path_xyzzy_42")

    def test_listdir_real_fs(self, vfs):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "a.txt").write_text("a")
            Path(td, "b.txt").write_text("b")
            entries = vfs.listdir(td)
            assert entries is not None
            assert "a.txt" in entries
            assert "b.txt" in entries

    def test_read_real_fs(self, vfs):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("hello world")
            tmppath = f.name
        try:
            val = vfs.read(tmppath)
            assert val == "hello world"
        finally:
            os.unlink(tmppath)

    def test_write_real_fs(self, vfs):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            tmppath = f.name
        try:
            err = vfs.write(tmppath, "new data")
            assert err is None
            assert Path(tmppath).read_text() == "new data"
        finally:
            os.unlink(tmppath)

    def test_stat_real_fs(self, vfs):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            tmppath = f.name
        try:
            st = vfs.stat(tmppath)
            assert st is not None
        finally:
            os.unlink(tmppath)

    def test_real_isfile_isdir(self, vfs):
        assert vfs.isdir("/tmp")
        assert not vfs.isfile("/tmp")


class TestVFSDevMount:
    def test_dev_list(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        entries = v.listdir("/dev")
        assert entries is not None
        assert "null" in entries
        assert "random" in entries
        assert len(entries) == 2

    def test_dev_exists(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        assert v.exists("/dev/null")
        assert v.exists("/dev/random")
        assert not v.exists("/dev/nonexistent")

    def test_dev_read_null(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        val = v.read("/dev/null")
        assert val == ""

    def test_dev_write_null(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        err = v.write("/dev/null", "hello")
        assert err is None

    def test_dev_stat(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        st = v.stat("/dev/null")
        assert st is not None
        assert not st.st_mode & 0o40000  # not a dir

    def test_dev_isfile(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        assert v.isfile("/dev/null")


class TestVFSProcMount:
    def test_proc_list(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        entries = v.listdir("/proc")
        assert entries is not None
        for name in ["uptime", "version", "meminfo", "cpuinfo", "stat", "loadavg"]:
            assert name in entries, f"Missing /proc/{name}"

    def test_proc_entries_have_self(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        entries = v.listdir("/proc")
        assert "self" in entries

    def test_proc_uptime(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        val = v.read("/proc/uptime")
        assert val is not None
        assert "42" in val

    def test_proc_version(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        val = v.read("/proc/version")
        assert val == "Dait 0.1\n"

    def test_proc_loadavg(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        val = v.read("/proc/loadavg")
        assert val is not None

    def test_proc_cpuinfo(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        val = v.read("/proc/cpuinfo")
        assert val is not None
        assert "cpu cores" in val

    def test_proc_meminfo(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        val = v.read("/proc/meminfo")
        assert val is not None
        assert "MemTotal" in val

    def test_proc_stat(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        val = v.read("/proc/stat")
        assert val is not None
        assert "cpu" in val

    def test_proc_self_status(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        val = v.read("/proc/self/status")
        assert val is not None
        assert "Name:" in val
        assert "Dait" in val

    def test_proc_exists(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        assert v.exists("/proc/uptime")
        assert v.exists("/proc/self")
        assert not v.exists("/proc/nonexistent")

    def test_proc_stat(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        st = v.stat("/proc/uptime")
        assert st is not None
        assert not st.st_mode & 0o40000  # not a dir

    def test_proc_self_is_dir(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        st = v.stat("/proc/self")
        assert st is not None
        assert st.st_mode & 0o40000  # is a dir

    def test_proc_isdir_isfile(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        assert v.isdir("/proc")
        assert v.isdir("/proc/self")
        assert v.isfile("/proc/uptime")


class TestVFSIsolation:
    def test_get_vfs_singleton(self):
        reset_vfs()
        v1 = get_vfs()
        v2 = get_vfs()
        assert v1 is v2

    def test_reset_vfs(self):
        reset_vfs()
        v1 = get_vfs()
        reset_vfs()
        v2 = get_vfs()
        assert v1 is not v2

    def test_set_twice_doesnt_duplicate(self, vfs_with_devices_and_kernel):
        v = vfs_with_devices_and_kernel
        # Setting devices a second time rebuilds mount without duplicates
        from domains.shell.devices import DeviceManager, NullDevice
        dm = DeviceManager()
        dm.register(NullDevice())
        v.set_devices(dm)
        entries = v.listdir("/dev")
        assert entries is not None
        assert len(entries) == 1  # only null, not random

    def test_read_nonexistent_virtual_path(self, vfs):
        assert vfs.read("/dev/x") is None

    def test_write_virtual_root_dir(self, vfs):
        # Mount a directory at /mnt
        d = VFSDirectory("mnt")
        vfs.mount("/mnt", d)
        err = vfs.write("/mnt", "data")
        assert err == "Is a directory"

    def test_stat_nonexistent(self, vfs):
        assert vfs.stat("/dev/x") is None

    def test_listdir_nonexistent(self, vfs):
        assert vfs.listdir("/dev/x") is None


class TestStatHelpers:
    def test_dir_stat(self):
        st = _dir_stat()
        assert st.st_mode & 0o40000  # directory bit

    def test_file_stat(self):
        st = _file_stat(42)
        assert not st.st_mode & 0o40000  # not a dir
        assert st.st_size == 42
