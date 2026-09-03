"""
Baseboard — device framework.

DeviceDriver = flip-flops (holds state, only ioctl)
DeviceTable = request handler (routes to flip-flops)
"""

from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any

from .kernel_syscall import SyscallResult


class DeviceType(IntEnum):
    """Device categories."""
    INFERENCE = 0
    TRAINING = 1
    STORAGE = 2
    NETWORK = 3
    DISPLAY = 4
    INPUT = 5
    CUSTOM = 6


class DeviceDriver:
    """Flip-flops — holds device state.

    No open/close. Only ioctl.
    Commands read/write registers.
    """

    def __init__(self, name: str, device_type: DeviceType = DeviceType.CUSTOM):
        self._name = name
        self._device_type = device_type
        self._registers: dict[str, Any] = {}  # flip-flop state

    @property
    def name(self) -> str:
        return self._name

    @property
    def device_type(self) -> DeviceType:
        return self._device_type

    def ioctl(self, command: str, *args: Any) -> SyscallResult:
        """Read/write flip-flops. That's it."""
        raise NotImplementedError(f"{self._name}: ioctl '{command}' not implemented")

    def read_register(self, name: str) -> Any:
        """Read a flip-flop."""
        return self._registers.get(name)

    def write_register(self, name: str, value: Any) -> None:
        """Write a flip-flop."""
        self._registers[name] = value

    def info(self) -> dict:
        """Read status registers."""
        return {
            "name": self._name,
            "type": self._device_type.name,
            "registers": list(self._registers.keys()),
        }


@dataclass
class FileHandle:
    """File descriptor handle — maps fd to device."""
    fd: int
    device: DeviceDriver
    mode: str  # "r", "w", "rw"


class DeviceTable:
    """Request handler — routes to flip-flops.

    Manages file descriptors.
    Routes ioctl to correct device.
    """

    def __init__(self):
        self._devices: dict[str, DeviceDriver] = {}  # name → device
        self._handles: dict[int, FileHandle] = {}     # fd → handle
        self._next_fd: int = 1

    def register(self, device: DeviceDriver) -> bool:
        """Register a device."""
        if device.name in self._devices:
            return False
        self._devices[device.name] = device
        return True

    def unregister(self, name: str) -> bool:
        """Unregister a device."""
        return self._devices.pop(name, None) is not None

    def get(self, name: str) -> DeviceDriver | None:
        """Get device by name."""
        return self._devices.get(name)

    def open(self, name: str, mode: str = "r") -> SyscallResult:
        """Open device → returns fd."""
        device = self._devices.get(name)
        if device is None:
            return SyscallResult.fail(f"device not found: {name}")

        fd = self._next_fd
        self._next_fd += 1
        self._handles[fd] = FileHandle(fd=fd, device=device, mode=mode)
        return SyscallResult.ok({"fd": fd})

    def close(self, fd: int) -> SyscallResult:
        """Close fd."""
        handle = self._handles.pop(fd, None)
        if handle is None:
            return SyscallResult.fail(f"bad fd: {fd}")
        return SyscallResult.ok({"closed": fd})

    def ioctl(self, fd: int, command: str, *args: Any) -> SyscallResult:
        """Route ioctl to device."""
        handle = self._handles.get(fd)
        if handle is None:
            return SyscallResult.fail(f"bad fd: {fd}")
        return handle.device.ioctl(command, *args)

    def info(self) -> dict:
        """List all devices."""
        return {
            "devices": list(self._devices.keys()),
            "open_fds": len(self._handles),
            "next_fd": self._next_fd,
        }
