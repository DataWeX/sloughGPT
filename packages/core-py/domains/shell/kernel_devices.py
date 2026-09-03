"""
AI-Native Device Driver Framework — inference, training, storage, network.

Devices are not hardware registers. They are Python objects that expose
a standard interface: ioctl. The kernel discovers and manages them through
a device table with bit-based fd management.
"""

from __future__ import annotations

import threading
from enum import IntEnum
from dataclasses import dataclass
from typing import Any

from .kernel_syscall import SyscallResult


# ── Device types as bit flags ─────────────────────────────────────────────

class DeviceType(IntEnum):
    """Device categories as bit positions."""
    INFERENCE = 1 << 0  # 0b000001
    TRAINING  = 1 << 1  # 0b000010
    STORAGE   = 1 << 2  # 0b000100
    NETWORK   = 1 << 3  # 0b001000
    DISPLAY   = 1 << 4  # 0b010000
    INPUT     = 1 << 5  # 0b100000
    CUSTOM    = 0       # no type


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
                self._fd_bitmap |= (1 << i)
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

    def unregister(self, name: str) -> bool:
        """Unregister device."""
        with self._lock:
            self._devices.pop(name, None)
            self._device_types.pop(name, None)
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
        """List all registered devices."""
        return [d.info() for d in self._devices.values()]

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
        if hasattr(dev, 'list_commands'):
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
        """Register device — compatible with old interface."""
        self._devices[device.name] = device
        return self.table.register(device, device_type)

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
        if hasattr(dev, 'list_commands'):
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
