"""
AI-Native Device Driver Framework — inference, training, storage, network.

Devices are not hardware registers. They are Python objects that expose
a standard interface: ioctl. The kernel discovers and manages them through
a device table with bit-based fd management.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .kernel_syscall import SyscallResult

# ── Device types as bit flags ─────────────────────────────────────────────


class DeviceType(IntEnum):
    """Device categories as bit positions."""

    INFERENCE = 1 << 0  # 0b000001
    TRAINING = 1 << 1  # 0b000010
    STORAGE = 1 << 2  # 0b000100
    NETWORK = 1 << 3  # 0b001000
    DISPLAY = 1 << 4  # 0b010000
    INPUT = 1 << 5  # 0b100000
    CUSTOM = 0  # no type


class DeviceState(IntEnum):
    CLOSED = 0
    OPEN = 1
    ERROR = 2


@dataclass
class DeviceHandle:
    """A file-descriptor-like handle to an open device."""

    fd: int
    device_name: str
    mode: str = "r"
    offset: int = 0


# ── Device driver (gates) ─────────────────────────────────────────────────


class DeviceDriver:
    """
    Base device driver — hardware gates.

    Only ioctl. No open/close — that's DeviceTable's job.
    Subclass this and implement ioctl.
    """

    def __init__(self, name: str, device_type: int = DeviceType.CUSTOM):
        self._name = name
        self._device_type = device_type
        self._state = DeviceState.CLOSED

    @property
    def name(self) -> str:
        return self._name

    @property
    def device_type(self) -> int:
        return self._device_type

    @property
    def state(self) -> DeviceState:
        return self._state

    def ioctl(self, command: str, *args: Any) -> SyscallResult:
        """Execute command on hardware. Subclass this."""
        raise NotImplementedError(f"{self._name}: ioctl '{command}' not implemented")

    def call(self, method: str, *args: Any) -> Any:
        """Name-based dispatch — the DeviceBus face of the ioctl contract.

        A driver speaks ioctl; the VM's DeviceBus speaks call(). They are the
        same operation with opposite conventions, so a driver gets both: this
        unwraps ioctl() so the device is reachable from either seam instead of
        only from the fd-based one.

        The dialects differ in case — IoctlCommand and every kernel driver
        match uppercase ("INFO", "LOAD"), the VM reaches devices by lowercase
        name ("info") — so the name is translated here, once, rather than each
        driver growing its own alias table.

        Success returns the value; failure raises, because the caller of call()
        is expected to act on it — handing back a SyscallResult nobody inspects
        would turn a device fault into silence.
        """
        from .vm import DeviceFault  # deferred: vm.py imports this module

        cmd = method.upper() if isinstance(method, str) else method
        try:
            result = self.ioctl(cmd, *args)
        except DeviceFault:
            raise
        except Exception as exc:  # noqa: BLE001 - a leaking ioctl becomes a fault
            raise DeviceFault(f"{self._name}: {method} raised: {exc}") from exc
        if not result.success:
            raise DeviceFault(f"{self._name}: {method} failed: {result.error}")
        return result.value

    def info(self) -> dict:
        return {
            "name": self._name,
            "type": self._device_type,
            "state": self._state.name,
        }


# ── Device table (request handler) ────────────────────────────────────────


class DeviceTable:
    """
    Kernel device table — routes ioctls to devices.

    Bit-based fd management:
    - 64 fds in one integer (bitmap)
    - O(1) alloc/free via bit scan
    - Direct index for fd → device lookup
    """

    def __init__(self, max_fds: int = 64):
        # Device registry
        self._devices: dict[str, DeviceDriver] = {}
        self._device_types: dict[str, int] = {}

        # FD bitmap — 64 fds in one integer
        self._fd_bitmap: int = 0
        self._max_fds: int = max_fds

        # FD tables — direct index
        self._fd_device: list[DeviceDriver | None] = [None] * max_fds
        self._fd_mode: list[int] = [0] * max_fds
        self._fd_offset: list[int] = [0] * max_fds

        self._lock = threading.Lock()

    # ── FD allocation (bit operations) ────────────────────────────────────

    def _alloc_fd(self) -> int:
        """Find first free fd using bit scan."""
        for i in range(self._max_fds):
            if not (self._fd_bitmap >> i) & 1:
                self._fd_bitmap |= 1 << i
                return i
        return -1

    def _free_fd(self, fd: int):
        """Free fd using bit clear."""
        self._fd_bitmap &= ~(1 << fd)
        self._fd_device[fd] = None
        self._fd_mode[fd] = 0
        self._fd_offset[fd] = 0

    def _fd_is_open(self, fd: int) -> bool:
        """Check if fd is open using bit test."""
        return (self._fd_bitmap >> fd) & 1 == 1

    # ── Device registration ───────────────────────────────────────────────

    def register(self, device: DeviceDriver, device_type: int = 0) -> bool:
        """Register device. Returns False if name taken."""
        with self._lock:
            if device.name in self._devices:
                return False
            self._devices[device.name] = device
            self._device_types[device.name] = device_type
            return True

    def alias(self, name: str, target: str) -> bool:
        """Point an extra name at an already-registered device.

        One driver instance, several node names — both open as fds and both
        answer ioctls, so an alias is indistinguishable from the original.
        """
        with self._lock:
            dev = self._devices.get(target)
            if dev is None:
                return False
            if name in self._devices and self._devices[name] is not dev:
                return False
            self._devices[name] = dev
            self._device_types[name] = self._device_types.get(target, 0)
            return True

    def unregister(self, name: str) -> bool:
        """Unregister device. Returns False if the name is not registered.

        Open fds bound to the device are freed with it: a dangling fd must
        never dispatch ioctls into a device the table no longer manages (the
        pre-baseboard implementation closed the device here for the same
        reason). Aliases share one driver instance, so fds stay open while
        any surviving name still maps to it.
        """
        with self._lock:
            dev = self._devices.pop(name, None)
            if dev is None:
                return False
            self._device_types.pop(name, None)
            if not any(d is dev for d in self._devices.values()):
                for fd in range(self._max_fds):
                    if self._fd_device[fd] is dev:
                        self._free_fd(fd)
            return True

    def get(self, name: str) -> DeviceDriver | None:
        """Get device by name."""
        return self._devices.get(name)

    # ── File descriptor operations ────────────────────────────────────────

    def open(self, name: str, mode: int = 0) -> int:
        """Open device, return fd. Returns -1 on error."""
        with self._lock:
            device = self._devices.get(name)
            if device is None:
                return -1
            fd = self._alloc_fd()
            if fd < 0:
                return -1
            self._fd_device[fd] = device
            self._fd_mode[fd] = mode
            self._fd_offset[fd] = 0
            return fd

    def close(self, fd: int) -> bool:
        """Close fd."""
        with self._lock:
            if fd < 0 or fd >= self._max_fds:
                return False
            if not self._fd_is_open(fd):
                return False
            self._free_fd(fd)
            return True

    def device_for_fd(self, fd: int) -> DeviceDriver | None:
        """Resolve an open fd to its device, or None if it is not open.

        The fd table's own lookup, exposed so a caller can dispatch by name on
        an open fd without reaching into ``_fd_device`` and re-checking bounds
        and the open bitmap itself.
        """
        if fd < 0 or fd >= self._max_fds or not self._fd_is_open(fd):
            return None
        return self._fd_device[fd]

    # ── ioctl dispatch ────────────────────────────────────────────────────

    def ioctl(self, fd: int, command: str, *args: Any) -> SyscallResult:
        """Route ioctl to device."""
        if fd < 0 or fd >= self._max_fds:
            return SyscallResult.fail("bad fd")
        if not self._fd_is_open(fd):
            return SyscallResult.fail("fd not open")
        device = self._fd_device[fd]
        if device is None:
            return SyscallResult.fail("no device")
        return device.ioctl(command, *args)

    # ── Info ──────────────────────────────────────────────────────────────

    def list_devices(self) -> list[dict]:
        """List registered devices — aliases collapse to one entry."""
        seen: set[int] = set()
        devices: list[dict] = []
        for device in self._devices.values():
            if id(device) in seen:
                continue
            seen.add(id(device))
            devices.append(device.info())
        return devices

    def stats(self) -> dict:
        """Get table stats."""
        return {
            "total_devices": len(self._devices),
            "open_fds": bin(self._fd_bitmap).count("1"),
            "fd_bitmap": self._fd_bitmap,
            "devices": [d.info() for d in self._devices.values()],
        }

    def capabilities(self, name: str) -> list[str]:
        """Get device capabilities (list of commands)."""
        dev = self.get(name)
        if dev is None:
            return []
        if hasattr(dev, "list_commands"):
            return dev.list_commands()
        return []


# ── Device manager (backward compat) ──────────────────────────────────────


class DeviceManager:
    """High-level device manager — wraps DeviceTable.

    Compatible with old interface (names, get) and new (open, close, ioctl).
    """

    def __init__(self):
        self.table = DeviceTable()
        self._devices: dict[str, DeviceDriver] = {}

    def register(self, device: DeviceDriver, device_type: int = 0) -> bool:
        """Register device — compatible with old interface.

        The table decides: mirror into ``_devices`` only on success, or a
        rejected duplicate would clobber the entry ``get()`` reports while the
        table keeps serving the original driver.
        """
        if not self.table.register(device, device_type):
            return False
        self._devices[device.name] = device
        return True

    def alias(self, name: str, target: str) -> bool:
        """Register an alternate node name for an existing device.

        ``llm`` and ``embedding`` resolve to the same driver instance as
        ``ai``: one registry entry, several names.
        """
        dev = self._devices.get(target)
        if dev is None:
            return False
        if name in self._devices and self._devices[name] is not dev:
            return False
        self._devices[name] = dev
        self.table.alias(name, target)
        add_alias = getattr(dev, "add_alias", None)
        if callable(add_alias):
            add_alias(name)
        return True

    def unregister(self, name: str) -> bool:
        """Unregister device."""
        self._devices.pop(name, None)
        return self.table.unregister(name)

    def get(self, name: str) -> DeviceDriver | None:
        """Get device by name — compatible with old interface."""
        return self._devices.get(name)

    @property
    def names(self) -> list[str]:
        """List device names — compatible with old interface."""
        return sorted(self._devices.keys())

    def open(self, name: str, mode: int = 0) -> int:
        """Open device, return fd."""
        return self.table.open(name, mode)

    def close(self, fd: int) -> bool:
        """Close fd."""
        return self.table.close(fd)

    def ioctl(self, fd: int, command: str, *args) -> SyscallResult:
        """Issue ioctl on fd."""
        return self.table.ioctl(fd, command, *args)

    def call(self, fd: int, method: str, *args) -> Any:
        """Dispatch by name on an open fd — DeviceManager's DeviceBus face.

        Mirrors DeviceBus.call(device, method, *args) in name and arity while
        keeping this seam's handle (an fd, like ioctl(fd, ...)), so both seams
        are driven the same way. Returns the value and raises DeviceFault on
        failure, where ioctl() hands back a SyscallResult and never raises.
        """
        from .vm import DeviceFault  # deferred: vm.py imports this module

        device = self.table.device_for_fd(fd)
        if device is None:
            raise DeviceFault(f"bad fd: {fd}")
        return device.call(method, *args)

    def info(self, fd: int) -> dict:
        """Device info for an open fd — mirrors DeviceBus.info(handle)."""
        device = self.table.device_for_fd(fd)
        return device.info() if device is not None else {}

    def list_devices(self) -> list[dict]:
        """List all registered devices."""
        return self.table.list_devices()

    def stats(self) -> dict:
        """Get table stats."""
        return self.table.stats()

    def capabilities(self, name: str) -> list[str]:
        """Get device capabilities (list of commands)."""
        dev = self.get(name)
        if dev is None:
            return []
        if hasattr(dev, "list_commands"):
            return dev.list_commands()
        return []

    def hotplug(self, name: str, device: DeviceDriver, device_type: int = 0) -> bool:
        """Hot-plug a device (register/unregister at runtime)."""
        if name in self._devices:
            return self.unregister(name)
        return self.register(device, device_type)


# ── Null device ───────────────────────────────────────────────────────────


class NullDevice(DeviceDriver):
    """A null /dev/null device that discards writes and returns empty on read."""

    def __init__(self):
        super().__init__("null", DeviceType.CUSTOM)

    def ioctl(self, command: str, *args: Any) -> SyscallResult:
        return SyscallResult.ok(b"")
