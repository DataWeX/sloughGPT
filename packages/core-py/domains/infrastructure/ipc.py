"""
Rust IPC Communication Layer

Platform-agnostic inter-process communication for Python <-> Rust.
Provides typed message passing for training and inference workloads.

Architecture:
- macOS: Memory-mapped files (mmap)
- Linux: System V shared memory
- Windows: Named pipes

Falls back to file-based IPC if Rust bindings unavailable.
"""

from __future__ import annotations

import fcntl
import json
import os
import pickle
import struct
from typing import Any, Dict, List, Optional, Tuple

_RUST_AVAILABLE = False
_IpcChannel = None
_IpcConfig = None

try:
    from sloughgpt_ipc import IpcChannel as RustIpcChannel, IpcConfig as RustIpcConfig

    _RUST_AVAILABLE = True
    _IpcChannel = RustIpcChannel
    _IpcConfig = RustIpcConfig
except ImportError:
    import logging

    logging.debug("Rust IPC not available, using filesystem fallback")


class IpcConfig:
    """Configuration for an IPC channel."""

    def __init__(self, name: str, capacity_bytes: int = 1024 * 1024):
        self.name = name
        self.capacity_bytes = capacity_bytes

    def __repr__(self) -> str:
        return f"IpcConfig(name='{self.name}', capacity_bytes={self.capacity_bytes})"


class IpcChannel:
    """
    Inter-process communication channel for Python <-> Rust.

    Wraps Rust sloughgpt-ipc crate when available, with filesystem fallback.
    """

    def __init__(self, config: IpcConfig):
        self.config = config
        self._closed = False

        if _RUST_AVAILABLE:
            self._channel = _IpcChannel(_IpcConfig(config.name, config.capacity_bytes))
        else:
            self._channel = None
            self._setup_fallback()

    @staticmethod
    def connect(config: IpcConfig) -> "IpcChannel":
        """Connect to an existing IPC channel."""
        channel = IpcChannel.__new__(IpcChannel)
        channel.config = config
        channel._closed = False

        if _RUST_AVAILABLE:
            channel._channel = _IpcChannel.connect(_IpcConfig(config.name, config.capacity_bytes))
        else:
            channel._channel = None
            channel._setup_fallback()

        return channel

    def _setup_fallback(self) -> None:
        """Setup filesystem-based fallback IPC."""
        tmp_dir = os.environ.get("TMPDIR", "/tmp")
        sanitized = self.config.name.replace("/", "_").replace("\\", "_").replace(":", "_")
        self._data_path = os.path.join(tmp_dir, f"sloughgpt_ipc_{sanitized}.data")
        self._lock_path = os.path.join(tmp_dir, f"sloughgpt_ipc_{sanitized}.lock")

        os.makedirs(os.path.dirname(self._data_path) or "/tmp", exist_ok=True)
        if not os.path.exists(self._data_path):
            with open(self._data_path, "wb") as f:
                f.write(b"\x00" * (64 + self.config.capacity_bytes))
        # Create lock file if it doesn't exist
        if not os.path.exists(self._lock_path):
            with open(self._lock_path, "w") as f:
                pass

    def send_weights(self, weights: Dict[str, List[float]]) -> None:
        """Send model weights."""
        if self._closed:
            raise RuntimeError("Channel is closed")

        if _RUST_AVAILABLE and self._channel is not None:
            self._channel.send_weights(list(weights.items()))
        else:
            self._filesystem_send(json.dumps(weights).encode())

    def recv_weights(self) -> Dict[str, List[float]]:
        """Receive model weights."""
        if self._closed:
            raise RuntimeError("Channel is closed")

        if _RUST_AVAILABLE and self._channel is not None:
            return dict(self._channel.recv_weights())
        else:
            data = self._filesystem_recv()
            return json.loads(data.decode())

    def send_training_step(self, batch: List[int], targets: List[int]) -> None:
        """Send training batch and targets."""
        if self._closed:
            raise RuntimeError("Channel is closed")

        if _RUST_AVAILABLE and self._channel is not None:
            self._channel.send_training_step(batch, targets)
        else:
            data = json.dumps({"batch": batch, "targets": targets})
            self._filesystem_send(data.encode())

    def recv_training_step(self) -> Tuple[List[int], List[int]]:
        """Receive training batch and targets."""
        if self._closed:
            raise RuntimeError("Channel is closed")

        if _RUST_AVAILABLE and self._channel is not None:
            return self._channel.recv_training_step()
        else:
            data = self._filesystem_recv()
            parsed = json.loads(data.decode())
            return parsed["batch"], parsed["targets"]

    def send(self, data: Any) -> None:
        """Send arbitrary Python data (pickled)."""
        if self._closed:
            raise RuntimeError("Channel is closed")

        pickled = pickle.dumps(data)

        if _RUST_AVAILABLE and self._channel is not None:
            self._channel.send(pickled)
        else:
            self._filesystem_send(pickled)

    def recv(self) -> Any:
        """Receive arbitrary Python data."""
        if self._closed:
            raise RuntimeError("Channel is closed")

        if _RUST_AVAILABLE and self._channel is not None:
            return pickle.loads(self._channel.recv())
        else:
            return pickle.loads(self._filesystem_recv())

    def _filesystem_send(self, data: bytes) -> None:
        """Filesystem-based send using fcntl locking."""
        if len(data) > self.config.capacity_bytes:
            raise ValueError(f"Data too large: {len(data)} > {self.config.capacity_bytes}")

        with open(self._lock_path, "r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                with open(self._data_path, "r+b") as f:
                    header = struct.pack("I", len(data))
                    ready = struct.pack("I", 1)
                    f.write(header)
                    f.write(ready)
                    f.write(data)
                    f.flush()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _filesystem_recv(self) -> bytes:
        """Filesystem-based receive using fcntl locking."""
        with open(self._lock_path, "r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                with open(self._data_path, "r+b") as f:
                    header = f.read(4)
                    ready = struct.unpack("I", f.read(4))[0]
                    if ready == 0:
                        raise RuntimeError("No data available")
                    length = struct.unpack("I", header)[0]
                    data = f.read(length)
                    f.seek(4)
                    f.write(struct.pack("I", 0))
                    return data
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def close(self) -> None:
        """Close the IPC channel."""
        self._closed = True

    def __enter__(self) -> "IpcChannel":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"IpcChannel(name='{self.config.name}', capacity={self.config.capacity_bytes})"


def is_rust_available() -> bool:
    """Check if Rust IPC bindings are available."""
    return _RUST_AVAILABLE


__all__ = ["IpcChannel", "IpcConfig", "is_rust_available"]
