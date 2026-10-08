"""
Filesystem addon — VFS (Virtual File System) for the kernel.

Provides mount-point based path resolution:
  /dev/* → device manager (LLM, embedding, knowledge, etc.)
  /proc/* → generated process/OS info
  everything else → host filesystem

Usage::

    domain.shell._internal.addons import filesystem
    kernel.install_addon(filesystem)

    kernel.vfs.read("/proc/uptime")
    kernel.vfs.listdir("/dev")
"""

from __future__ import annotations

import logging
import os
import stat as stat_mod
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("slo.shell.addons.filesystem")


class VFSEntry:
    """A virtual file or directory entry."""

    def __init__(self, name: str, mode: int = 0o444, size: int = 0, is_dir: bool = False):
        self.name = name
        self.mode = mode
        self._size = size
        self._is_dir = is_dir

    @property
    def is_dir(self) -> bool:
        return self._is_dir

    @property
    def size(self) -> int:
        return self._size

    def read(self) -> str:
        return ""

    def write(self, data: str) -> None:
        pass


class VFSGeneratedFile(VFSEntry):
    """File whose content is generated on read via a callback."""

    def __init__(self, name: str, read_fn: Callable[[], str], mode: int = 0o444):
        super().__init__(name, mode=mode)
        self._read_fn = read_fn

    def read(self) -> str:
        return self._read_fn()


class VFSWriteOnlyFile(VFSEntry):
    """File that accepts writes but returns nothing on read."""

    def __init__(self, name: str, write_fn: Callable[[str], str | None]):
        super().__init__(name, mode=0o222, size=0)
        self._write_fn = write_fn

    def write(self, data: str) -> None:
        result = self._write_fn(data)
        if result:
            logger.debug(result)


def _device_result(result: Any) -> str:
    """Unwrap a driver's ``SyscallResult`` (or a bare string) into text."""
    if hasattr(result, "success"):
        if result.success:
            return str(result.value) if result.value is not None else ""
        return str(result.error or "")
    return str(result or "")


class VFSDeviceEntry(VFSEntry):
    """A ``/dev`` node backed by a device driver.

    This is what makes ``cat /dev/ai`` and ``echo prompt > /dev/ai`` reach the
    driver instead of the inert placeholder every device node used to be.
    ``read_op`` / ``write_op`` come from the driver's ``vfs_ops(name)``;
    ``"auto"`` lets the written payload select an operation with an ``<op>:`` prefix.
    """

    def __init__(self, name: str, driver: Any, read_op: str, write_op: str):
        super().__init__(name, mode=0o666, size=0)
        self._driver = driver
        self._read_op = read_op
        self._write_op = write_op

    def read(self) -> str:
        # ``info`` is told which node was opened so an alias card names
        # itself; the other operations never take the node name as payload.
        payload = self.name if self._read_op == "info" else ""
        return _device_result(self._driver.ioctl(self._read_op, payload))

    def write(self, data: str) -> str:
        return _device_result(self._driver.ioctl(self._write_op, data))


class VFSDirectory:
    """Virtual directory listing."""

    def __init__(self, name: str, entries: dict[str, VFSEntry | VFSDirectory] | None = None):
        self.name = name
        self._entries = entries or {}

    @property
    def is_dir(self) -> bool:
        return True

    def list(self) -> list[str]:
        return sorted(self._entries.keys())

    def get(self, name: str) -> VFSEntry | VFSDirectory | None:
        return self._entries.get(name)

    def add(self, entry: VFSEntry | VFSDirectory) -> None:
        self._entries[entry.name] = entry


class VFS:
    """Virtual File System — resolves paths, dispatches reads/writes.

    Mount points intercept paths before falling through to the real FS.
    """

    def __init__(self):
        self._mounts: dict[str, VFSDirectory | Callable[[str], Any]] = {}
        self._devices: Any = None
        self._kernel: Any = None

    def set_devices(self, devices: Any) -> None:
        self._devices = devices
        self._rebuild_dev_mount()

    def set_kernel(self, kernel: Any) -> None:
        self._kernel = kernel
        self._rebuild_proc_mount()

    def _rebuild_dev_mount(self) -> None:
        """Rebuild /dev/ mount from device manager.

        Nodes whose driver exposes ``vfs_ops(name)`` become live entries that
        dispatch read/write to its ``ioctl``; everything else keeps the inert
        placeholder (the hardware drivers answer ioctls, not reads).
        Devices that are only ``.names`` (test doubles) get placeholders too.
        """
        dev = VFSDirectory("dev")
        if self._devices:
            getter = getattr(self._devices, "get", None)
            for name in self._devices.names:
                driver = getter(name) if callable(getter) else None
                ops = None
                if driver is not None and hasattr(driver, "vfs_ops"):
                    ops = driver.vfs_ops(name)
                if ops is None:
                    dev.add(VFSEntry(name, mode=0o666, size=0))
                else:
                    dev.add(VFSDeviceEntry(name, driver, *ops))
        self._mounts["/dev"] = dev

    def _rebuild_proc_mount(self) -> None:
        """Rebuild /proc/ mount from kernel state."""
        proc = VFSDirectory("proc")
        if self._kernel:
            proc.add(VFSGeneratedFile("uptime", lambda: f"{self._kernel.uptime:.2f}\n"))
            proc.add(VFSGeneratedFile("meminfo", self._gen_meminfo))
            proc.add(VFSGeneratedFile("version", lambda: "Dait 0.1\n"))
            proc.add(VFSGeneratedFile("loadavg", lambda: "0.00 0.00 0.00 1/1 1\n"))
            proc.add(VFSGeneratedFile("cpuinfo", self._gen_cpuinfo))
            proc.add(VFSGeneratedFile("stat", self._gen_stat))
            proc_pid = VFSDirectory("self")
            proc_pid.add(VFSGeneratedFile("status", self._gen_self_status))
            proc_pid.add(VFSGeneratedFile("cmdline", lambda: "dait\n"))
            proc.add(proc_pid)
        self._mounts["/proc"] = proc

    def _gen_meminfo(self) -> str:
        try:
            import psutil

            mem = psutil.virtual_memory()
            return (
                f"MemTotal:      {mem.total // 1024} kB\n"
                f"MemFree:       {mem.available // 1024} kB\n"
                f"MemAvailable:  {mem.available // 1024} kB\n"
            )
        except ImportError:
            return "MemTotal:  8388608 kB\nMemFree:   4194304 kB\n"

    def _gen_cpuinfo(self) -> str:
        try:
            from domain.infrastructure._internal.resource_manager import get_resource_manager

            rm = get_resource_manager()
            cores = rm.topology.logical_cores
        except Exception:
            try:
                import psutil

                cores = psutil.cpu_count()
            except Exception:
                cores = os.cpu_count() or 1
        try:
            import psutil

            freq = psutil.cpu_freq()
            freq_str = f"{freq.current:.0f}" if freq else "0"
            return f"processor\t: 0\ncpu cores\t: {cores}\nCPU MHz\t\t: {freq_str}\n"
        except ImportError:
            return "processor\t: 0\ncpu cores\t: 1\nCPU MHz\t\t: 2400\n"

    def _gen_stat(self) -> str:
        return f"cpu  0 0 0 0 0 0 0 0 0 0\nctxt 0\nbtime {int(time.time())}\n"

    def _gen_self_status(self) -> str:
        name = "Dait"
        state = "S (sleeping)"
        pid = os.getpid()
        return f"Name:\t{name}\nState:\t{state}\nPid:\t{pid}\n"

    def mount(self, path: str, mount: VFSDirectory | Callable) -> None:
        self._mounts[path] = mount

    def unmount(self, path: str) -> bool:
        if path in self._mounts:
            del self._mounts[path]
            return True
        return False

    def mounts(self) -> list[str]:
        return sorted(self._mounts.keys())

    def _resolve(self, path: str) -> tuple[Any, str] | None:
        """Resolve a path to (mount_obj, relative_path) or None for real FS."""
        abs_path = os.path.abspath(os.path.expanduser(path))

        for mount_point, mount_obj in self._mounts.items():
            if abs_path == mount_point:
                return mount_obj, ""
            if abs_path.startswith(mount_point + "/"):
                rel = abs_path[len(mount_point) + 1 :]
                return mount_obj, rel

        return None

    def _resolve_in_dir(self, mount_obj: VFSDirectory, rel: str) -> VFSEntry | VFSDirectory | None:
        """Walk a relative path through nested VFSDirectory entries."""
        parts = rel.split("/")
        current: Any = mount_obj
        for i, part in enumerate(parts):
            if not part:
                continue
            if isinstance(current, VFSDirectory):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
            if i == len(parts) - 1:
                return current
            if not isinstance(current, VFSDirectory):
                return None
        return None

    def exists(self, path: str) -> bool:
        result = self._resolve(path)
        if result is None:
            return os.path.exists(os.path.expanduser(path))

        mount_obj, rel = result
        if isinstance(mount_obj, VFSDirectory):
            if not rel:
                return True
            entry = self._resolve_in_dir(mount_obj, rel)
            return entry is not None
        return True

    def listdir(self, path: str) -> list[str] | None:
        result = self._resolve(path)
        if result is None:
            try:
                return sorted(os.listdir(os.path.expanduser(path)))
            except OSError:
                return None

        mount_obj, rel = result
        if isinstance(mount_obj, VFSDirectory):
            if not rel:
                return mount_obj.list()
            entry = self._resolve_in_dir(mount_obj, rel)
            if isinstance(entry, VFSDirectory):
                return entry.list()
            return None

        return None

    def read(self, path: str) -> str | None:
        result = self._resolve(path)
        if result is None:
            try:
                return Path(os.path.expanduser(path)).read_text()
            except OSError:
                return None

        mount_obj, rel = result
        if isinstance(mount_obj, VFSDirectory):
            if not rel:
                return None
            entry = self._resolve_in_dir(mount_obj, rel)
            if isinstance(entry, VFSEntry) and not entry.is_dir:
                return entry.read()
            return None
        return None

    def write(self, path: str, data: str) -> str | None:
        result = self._resolve(path)
        if result is None:
            try:
                Path(os.path.expanduser(path)).write_text(data)
                return None
            except OSError as e:
                return str(e)

        mount_obj, rel = result
        if isinstance(mount_obj, VFSDirectory):
            if not rel:
                return "Is a directory"
            entry = self._resolve_in_dir(mount_obj, rel)
            if isinstance(entry, VFSEntry):
                out = entry.write(data)
                # Ordinary files return None (success, nothing to show);
                # device nodes return the response they want printed.
                return out if isinstance(out, str) and out else None
            if isinstance(entry, VFSDirectory):
                return "Is a directory"
            return "Not found"
        return None

    def stat(self, path: str) -> os.stat_result | None:
        result = self._resolve(path)
        if result is None:
            try:
                return os.stat(os.path.expanduser(path))
            except OSError:
                return None

        mount_obj, rel = result
        if isinstance(mount_obj, VFSDirectory):
            if not rel:
                return _dir_stat()
            entry = self._resolve_in_dir(mount_obj, rel)
            if entry is None:
                return None
            if entry.is_dir:
                return _dir_stat()
            return _file_stat(entry.size)
        return None

    def isfile(self, path: str) -> bool:
        s = self.stat(path)
        return s is not None and not stat_mod.S_ISDIR(s.st_mode)

    def isdir(self, path: str) -> bool:
        s = self.stat(path)
        return s is not None and stat_mod.S_ISDIR(s.st_mode)


def _dir_stat() -> os.stat_result:
    now = int(time.time())
    return os.stat_result((stat_mod.S_IFDIR | 0o555, 0, 0, 0, 0, 0, 4096, now, now, now))


def _file_stat(size: int) -> os.stat_result:
    now = int(time.time())
    return os.stat_result((stat_mod.S_IFREG | 0o444, 0, 0, 0, 0, 0, size, now, now, now))


def setup(kernel: Any) -> None:
    """Install filesystem addon on the kernel. No-op if already installed."""
    if "filesystem" in getattr(kernel, "_addons", {}):
        return
    kernel._vfs = VFS()
    kernel._vfs.set_kernel(kernel)
    kernel._addons["filesystem"] = True
    logger.debug("Filesystem addon installed")
