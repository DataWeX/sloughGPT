"""
Additional tests for the filesystem addon — edge cases and branches not
covered by the primary suite: proc generator fallbacks, mount object types,
real-filesystem error paths, and write/stat dispatch branches.
"""

import builtins
import logging
import os
import stat as stat_mod
import sys
import types

import pytest

from domains.shell.addons.filesystem import (
    VFS, VFSEntry, VFSDirectory, VFSWriteOnlyFile,
)


def _psutil_stub(cpu_freq_val):
    """Programmatic psutil stand-in: real computed values, no hardcoded tables."""
    mod = types.ModuleType("psutil")
    mod.virtual_memory = lambda: types.SimpleNamespace(
        total=16 * 1024 ** 3,
        available=8 * 1024 ** 3,
    )
    mod.cpu_count = lambda: 4
    mod.cpu_freq = lambda: cpu_freq_val
    return mod


def _block_psutil(monkeypatch):
    """Force ``import psutil`` to raise ImportError regardless of environment."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _broken_resource_manager(monkeypatch):
    """Make resource_manager importable but raise at call time."""
    mod = types.ModuleType("domains.infrastructure.resource_manager")

    def _raise():
        raise RuntimeError("no resource manager in test")

    mod.get_resource_manager = _raise
    monkeypatch.setitem(sys.modules, "domains.infrastructure.resource_manager", mod)


# ---------------------------------------------------------------------------
# VFSWriteOnlyFile debug logging
# ---------------------------------------------------------------------------

class TestVFSWriteOnlyFileDebug:
    def test_write_logs_truthy_result(self, caplog):
        f = VFSWriteOnlyFile("control", lambda d: "handled:" + d)
        with caplog.at_level(logging.DEBUG, logger="slo.shell.addons.filesystem"):
            f.write("go")
        assert any("handled:go" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# set_devices / set_kernel mount rebuilds
# ---------------------------------------------------------------------------

class TestSetDevices:
    def test_set_devices_with_names_builds_mount(self):
        vfs = VFS()
        devices = types.SimpleNamespace(names=["llm", "embed", "knowledge"])
        vfs.set_devices(devices)
        assert "/dev" in vfs.mounts()
        assert vfs.listdir("/dev") == ["embed", "knowledge", "llm"]

    def test_set_devices_none_builds_empty_mount(self):
        vfs = VFS()
        vfs.set_devices(None)
        assert "/dev" in vfs.mounts()
        assert vfs.listdir("/dev") == []


class TestSetKernelProc:
    def test_set_kernel_builds_proc_mount(self):
        vfs = VFS()
        vfs.set_kernel(object())
        assert "/proc" in vfs.mounts()
        assert vfs.read("/proc/version") == "Dait 0.1\n"
        assert vfs.read("/proc/loadavg") is not None
        assert vfs.read("/proc/self/cmdline") == "dait\n"


# ---------------------------------------------------------------------------
# /proc generator branches
# ---------------------------------------------------------------------------

class TestProcGenerators:
    def test_meminfo_fallback_without_psutil(self, monkeypatch):
        _block_psutil(monkeypatch)
        vfs = VFS()
        vfs.set_kernel(object())
        out = vfs.read("/proc/meminfo")
        assert out is not None and "MemTotal" in out

    def test_meminfo_with_psutil(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", _psutil_stub(None))
        vfs = VFS()
        vfs.set_kernel(object())
        out = vfs.read("/proc/meminfo")
        assert "MemTotal:      16777216 kB" in out

    def test_cpuinfo_resource_manager_no_psutil(self, monkeypatch):
        _block_psutil(monkeypatch)
        vfs = VFS()
        vfs.set_kernel(object())
        out = vfs.read("/proc/cpuinfo")
        assert "cpu cores\t: 1" in out

    def test_cpuinfo_full_fallback_uses_os_cpu_count(self, monkeypatch):
        _broken_resource_manager(monkeypatch)
        _block_psutil(monkeypatch)
        vfs = VFS()
        vfs.set_kernel(object())
        out = vfs.read("/proc/cpuinfo")
        assert out is not None and "cpu cores" in out

    def test_cpuinfo_psutil_fallback_with_freq(self, monkeypatch):
        _broken_resource_manager(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "psutil",
            _psutil_stub(types.SimpleNamespace(current=2500.5)),
        )
        vfs = VFS()
        vfs.set_kernel(object())
        out = vfs.read("/proc/cpuinfo")
        assert "cpu cores\t: 4" in out
        assert "2500" in out

    def test_cpuinfo_psutil_fallback_no_freq(self, monkeypatch):
        _broken_resource_manager(monkeypatch)
        monkeypatch.setitem(sys.modules, "psutil", _psutil_stub(None))
        vfs = VFS()
        vfs.set_kernel(object())
        out = vfs.read("/proc/cpuinfo")
        assert "cpu cores\t: 4" in out
        assert "CPU MHz\t\t: 0" in out

    def test_stat_generator(self):
        vfs = VFS()
        vfs.set_kernel(object())
        out = vfs.read("/proc/stat")
        assert out.startswith("cpu  ")

    def test_self_status_generator(self):
        vfs = VFS()
        vfs.set_kernel(object())
        out = vfs.read("/proc/self/status")
        assert "Name:" in out
        assert str(os.getpid()) in out


# ---------------------------------------------------------------------------
# _resolve_in_dir edge cases
# ---------------------------------------------------------------------------

class TestResolveInDir:
    def test_empty_part_skipped(self):
        vfs = VFS()
        root = VFSDirectory("root")
        root.add(VFSEntry("a"))
        vfs.mount("/mnt", root)
        entry = vfs._resolve_in_dir(root, "/a")
        assert entry is not None and entry.name == "a"

    def test_non_directory_current_returns_none(self):
        vfs = VFS()
        assert vfs._resolve_in_dir("not-a-dir", "a") is None

    def test_intermediate_non_directory_returns_none(self):
        vfs = VFS()
        root = VFSDirectory("root")
        sub = VFSDirectory("sub")
        sub.add(VFSEntry("f"))
        root.add(sub)
        vfs.mount("/mnt", root)
        assert vfs.exists("/mnt/sub/f/x") is False

    def test_all_empty_parts_returns_none(self):
        vfs = VFS()
        root = VFSDirectory("root")
        vfs.mount("/mnt", root)
        assert vfs._resolve_in_dir(root, "/") is None


# ---------------------------------------------------------------------------
# callable (non-VFSDirectory) mount fallbacks
# ---------------------------------------------------------------------------

class TestCallableMounts:
    def test_exists_returns_true_for_callable_mount(self):
        vfs = VFS()
        vfs.mount("/magic", lambda p: None)
        assert vfs.exists("/magic/anything") is True

    def test_listdir_callable_mount_returns_none(self):
        vfs = VFS()
        vfs.mount("/magic", lambda p: None)
        assert vfs.listdir("/magic") is None

    def test_read_callable_mount_returns_none(self):
        vfs = VFS()
        vfs.mount("/magic", lambda p: None)
        assert vfs.read("/magic") is None

    def test_write_callable_mount_returns_none(self):
        vfs = VFS()
        vfs.mount("/magic", lambda p: None)
        assert vfs.write("/magic", "x") is None

    def test_stat_callable_mount_returns_none(self):
        vfs = VFS()
        vfs.mount("/magic", lambda p: None)
        assert vfs.stat("/magic") is None


# ---------------------------------------------------------------------------
# real-filesystem error paths
# ---------------------------------------------------------------------------

class TestRealFsErrorPaths:
    def test_listdir_real_directory(self, tmp_path):
        (tmp_path / "b").write_text("")
        (tmp_path / "a").write_text("")
        vfs = VFS()
        assert vfs.listdir(str(tmp_path)) == ["a", "b"]

    def test_listdir_missing_dir_returns_none(self, tmp_path):
        vfs = VFS()
        assert vfs.listdir(str(tmp_path / "missing")) is None

    def test_read_missing_file_returns_none(self, tmp_path):
        vfs = VFS()
        assert vfs.read(str(tmp_path / "missing.txt")) is None

    def test_stat_missing_returns_none(self, tmp_path):
        vfs = VFS()
        assert vfs.stat(str(tmp_path / "missing")) is None

    def test_write_to_directory_returns_error(self, tmp_path):
        vfs = VFS()
        err = vfs.write(str(tmp_path), "x")
        assert err is not None


# ---------------------------------------------------------------------------
# virtual directory / file dispatch branches
# ---------------------------------------------------------------------------

class TestVirtualDispatchBranches:
    def test_listdir_file_returns_none(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSEntry("llm"))
        vfs.mount("/dev", dev)
        assert vfs.listdir("/dev/llm") is None

    def test_read_virtual_dir_returns_none(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        vfs.mount("/dev", dev)
        assert vfs.read("/dev") is None

    def test_write_virtual_dir_returns_isdir(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        vfs.mount("/dev", dev)
        assert vfs.write("/dev", "x") == "Is a directory"

    def test_write_to_dir_entry_returns_isdir(self):
        vfs = VFS()
        root = VFSDirectory("root")
        sub = VFSDirectory("sub")
        root.add(sub)
        vfs.mount("/mnt", root)
        assert vfs.write("/mnt/sub", "x") == "Is a directory"

    def test_write_missing_entry_returns_notfound(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        vfs.mount("/dev", dev)
        assert vfs.write("/dev/nope", "x") == "Not found"

    def test_write_virtual_file_returns_none(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        dev.add(VFSWriteOnlyFile("ctl", lambda d: None))
        vfs.mount("/dev", dev)
        assert vfs.write("/dev/ctl", "x") is None

    def test_stat_missing_entry_returns_none(self):
        vfs = VFS()
        dev = VFSDirectory("dev")
        vfs.mount("/dev", dev)
        assert vfs.stat("/dev/nope") is None

    def test_stat_nested_dir_returns_dir_stat(self):
        vfs = VFS()
        root = VFSDirectory("root")
        sub = VFSDirectory("sub")
        root.add(sub)
        vfs.mount("/mnt", root)
        s = vfs.stat("/mnt/sub")
        assert s is not None and stat_mod.S_ISDIR(s.st_mode)
