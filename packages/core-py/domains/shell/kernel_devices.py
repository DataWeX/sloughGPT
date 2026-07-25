"""
AI-Native Device Driver Framework — inference, training, storage, network.

Devices are not hardware registers. They are Python objects that expose
a standard interface: open/close/read/write/ioctl. The kernel discovers
and manages them through a device table.
"""

from __future__ import annotations

import threading
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Protocol


class DeviceType(IntEnum):
    """Device categories."""
    INFERENCE = 0    # model inference engine
    TRAINING = 1     # model training
    STORAGE = 2      # disk/file storage
    NETWORK = 3      # network I/O
    DISPLAY = 4      # output/display
    INPUT = 5        # user input
    CUSTOM = 6       # user-defined


class DeviceState(IntEnum):
    CLOSED = 0
    OPEN = 1
    ERROR = 2


class Device(Protocol):
    """Protocol that all devices must satisfy."""

    @property
    def name(self) -> str: ...
    @property
    def device_type(self) -> DeviceType: ...
    def open(self) -> bool: ...
    def close(self) -> None: ...
    def read(self, offset: int = 0, size: int = -1) -> Any: ...
    def write(self, data: Any) -> bool: ...
    def ioctl(self, command: str, *args: Any) -> Any: ...
    def info(self) -> dict: ...


@dataclass
class DeviceHandle:
    """A file-descriptor-like handle to an open device."""
    fd: int
    device_name: str
    mode: str  # "r", "w", "rw"
    offset: int = 0


class DeviceDriver:
    """
    Base device driver — template for all device implementations.

    Subclass this and override the abstract methods.
    """

    def __init__(self, name: str, device_type: DeviceType = DeviceType.CUSTOM):
        self._name = name
        self._device_type = device_type
        self._state = DeviceState.CLOSED
        self._lock = threading.Lock()
        self._open_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def device_type(self) -> DeviceType:
        return self._device_type

    @property
    def state(self) -> DeviceState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == DeviceState.OPEN

    def open(self) -> bool:
        with self._lock:
            if self._state == DeviceState.ERROR:
                return False
            self._state = DeviceState.OPEN
            self._open_count += 1
            return True

    def close(self) -> None:
        with self._lock:
            self._open_count -= 1
            if self._open_count <= 0:
                self._state = DeviceState.CLOSED
                self._open_count = 0

    def read(self, offset: int = 0, size: int = -1) -> Any:
        raise NotImplementedError(f"{self._name}: read not implemented")

    def write(self, data: Any) -> bool:
        raise NotImplementedError(f"{self._name}: write not implemented")

    def ioctl(self, command: str, *args: Any) -> Any:
        raise NotImplementedError(f"{self._name}: ioctl '{command}' not implemented")

    def info(self) -> dict:
        return {
            "name": self._name,
            "type": self._device_type.name,
            "state": self._state.name,
            "open_count": self._open_count,
        }


@dataclass
class DeviceTable:
    """Kernel device table — maps device names to drivers."""
    _devices: dict[str, DeviceDriver] = field(default_factory=dict)
    _next_fd: int = 1
    _handles: dict[int, DeviceHandle] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def register(self, device: DeviceDriver) -> bool:
        """Register a device driver. Returns False if name taken."""
        with self._lock:
            if device.name in self._devices:
                return False
            self._devices[device.name] = device
            return True

    def unregister(self, name: str) -> bool:
        """Unregister a device driver."""
        with self._lock:
            dev = self._devices.pop(name, None)
            if dev is None:
                return False
            if dev.is_open:
                dev.close()
            return True

    def get(self, name: str) -> DeviceDriver | None:
        return self._devices.get(name)

    def open(self, name: str, mode: str = "r") -> DeviceHandle | None:
        """Open a device and return a handle (file descriptor)."""
        with self._lock:
            dev = self._devices.get(name)
            if dev is None:
                return None
            if not dev.open():
                return None
            fd = self._next_fd
            self._next_fd += 1
            handle = DeviceHandle(fd=fd, device_name=name, mode=mode)
            self._handles[fd] = handle
            return handle

    def close_fd(self, fd: int) -> bool:
        """Close a file descriptor."""
        with self._lock:
            handle = self._handles.pop(fd, None)
            if handle is None:
                return False
            dev = self._devices.get(handle.device_name)
            if dev:
                dev.close()
            return True

    def read_fd(self, fd: int, offset: int = 0, size: int = -1) -> Any:
        """Read from an open file descriptor."""
        handle = self._handles.get(fd)
        if handle is None:
            raise ValueError(f"Bad file descriptor: {fd}")
        dev = self._devices.get(handle.device_name)
        if dev is None:
            raise ValueError(f"Device disconnected: {handle.device_name}")
        return dev.read(offset=offset, size=size)

    def write_fd(self, fd: int, data: Any) -> bool:
        """Write to an open file descriptor."""
        handle = self._handles.get(fd)
        if handle is None:
            raise ValueError(f"Bad file descriptor: {fd}")
        dev = self._devices.get(handle.device_name)
        if dev is None:
            raise ValueError(f"Device disconnected: {handle.device_name}")
        return dev.write(data)

    def ioctl_fd(self, fd: int, command: str, *args: Any) -> Any:
        """Issue an ioctl on an open file descriptor."""
        handle = self._handles.get(fd)
        if handle is None:
            raise ValueError(f"Bad file descriptor: {fd}")
        dev = self._devices.get(handle.device_name)
        if dev is None:
            raise ValueError(f"Device disconnected: {handle.device_name}")
        return dev.ioctl(command, *args)

    def list_devices(self) -> list[dict]:
        return [d.info() for d in self._devices.values()]

    @property
    def device_count(self) -> int:
        return len(self._devices)

    @property
    def open_fd_count(self) -> int:
        return len(self._handles)

    def stats(self) -> dict:
        return {
            "total_devices": len(self._devices),
            "open_fds": len(self._handles),
            "devices": [d.info() for d in self._devices.values()],
        }


class DeviceManager:
    """High-level device manager — creates, registers, and manages drivers."""

    def __init__(self):
        self.table = DeviceTable()

    def register(self, device: DeviceDriver) -> bool:
        return self.table.register(device)

    def unregister(self, name: str) -> bool:
        return self.table.unregister(name)

    def get(self, name: str) -> DeviceDriver | None:
        return self.table.get(name)

    def open(self, name: str, mode: str = "r") -> DeviceHandle | None:
        return self.table.open(name, mode)

    def close(self, fd: int) -> bool:
        return self.table.close_fd(fd)

    def read(self, fd: int, **kwargs) -> Any:
        return self.table.read_fd(fd, **kwargs)

    def write(self, fd: int, data: Any) -> bool:
        return self.table.write_fd(fd, data)

    def ioctl(self, fd: int, command: str, *args) -> Any:
        return self.table.ioctl_fd(fd, command, *args)

    def list_devices(self) -> list[dict]:
        return self.table.list_devices()

    def stats(self) -> dict:
        return self.table.stats()
