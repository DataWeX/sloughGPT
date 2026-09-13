"""
StorageDevice — standalone storage hardware.

File system operations with clean ioctl interface.
"""

from __future__ import annotations

import os
from typing import Any

from .kernel_syscall import SyscallResult


class StorageDevice:
    """Standalone storage hardware — file system operations.

    Has clean ioctl interface for assembly.
    Has function calls for direct use.
    """

    def __init__(self, name: str = "storage", base_path: str = "/tmp"):
        self._name = name
        self._base_path = base_path
        self._ops = {
            "READ": self._read,
            "WRITE": self._write,
            "OPEN": self._open,
            "CLOSE": self._close,
            "SEEK": self._seek,
            "TELL": self._tell,
            "STAT": self._stat,
            "LIST": self._list,
            "MKDIR": self._mkdir,
            "REMOVE": self._remove,
            "RENAME": self._rename,
            "EXISTS": self._exists,
            "INFO": self._info,
        }
        self._files: dict[int, Any] = {}
        self._next_fd: int = 1

    @property
    def name(self) -> str:
        return self._name

    def info(self) -> dict:
        return {
            "name": self._name,
            "type": "storage",
            "base_path": self._base_path,
            "open_files": len(self._files),
        }

    def call(self, method: str, *args: Any) -> Any:
        """VM Device interface — delegates to ioctl."""
        result = self.ioctl(method, *args)
        if result.success:
            return result.value
        raise Exception(result.error)

    # ── ioctl interface ───────────────────────────────────────────────────

    def ioctl(self, command: str, *args: Any) -> SyscallResult:
        """Clean ioctl interface — type-safe, documented."""
        try:
            fn = self._ops.get(command)
            if fn is None:
                return SyscallResult.fail(f"unknown command: {command}")
            result = fn(*args)
            return SyscallResult.ok(result)
        except Exception as e:
            return SyscallResult.fail(f"ioctl error: {e}")

    def list_commands(self) -> list[str]:
        """List all available commands."""
        return sorted(self._ops.keys())

    # ── Function calls (direct use) ───────────────────────────────────────

    def read(self, fd: int, size: int = -1) -> bytes:
        """Read from file."""
        if fd not in self._files:
            raise ValueError(f"bad fd: {fd}")
        f = self._files[fd]
        if size == -1:
            return f.read()
        return f.read(size)

    def write(self, fd: int, data: bytes) -> int:
        """Write to file."""
        if fd not in self._files:
            raise ValueError(f"bad fd: {fd}")
        f = self._files[fd]
        return f.write(data)

    def open_file(self, path: str, mode: str = "rb") -> int:
        """Open file, return fd."""
        full_path = os.path.join(self._base_path, path)
        f = open(full_path, mode)
        fd = self._next_fd
        self._next_fd += 1
        self._files[fd] = f
        return fd

    def close_file(self, fd: int) -> bool:
        """Close file."""
        if fd not in self._files:
            return False
        self._files[fd].close()
        del self._files[fd]
        return True

    def seek(self, fd: int, offset: int, whence: int = 0) -> int:
        """Seek in file."""
        if fd not in self._files:
            raise ValueError(f"bad fd: {fd}")
        return self._files[fd].seek(offset, whence)

    def tell(self, fd: int) -> int:
        """Tell file position."""
        if fd not in self._files:
            raise ValueError(f"bad fd: {fd}")
        return self._files[fd].tell()

    def stat(self, path: str) -> dict:
        """Get file stats."""
        full_path = os.path.join(self._base_path, path)
        s = os.stat(full_path)
        return {
            "size": s.st_size,
            "mode": s.st_mode,
            "mtime": s.st_mtime,
        }

    def list_dir(self, path: str = ".") -> list[str]:
        """List directory."""
        full_path = os.path.join(self._base_path, path)
        return os.listdir(full_path)

    def mkdir(self, path: str) -> bool:
        """Create directory."""
        full_path = os.path.join(self._base_path, path)
        os.makedirs(full_path, exist_ok=True)
        return True

    def remove(self, path: str) -> bool:
        """Remove file."""
        full_path = os.path.join(self._base_path, path)
        os.remove(full_path)
        return True

    def rename(self, src: str, dst: str) -> bool:
        """Rename file."""
        src_path = os.path.join(self._base_path, src)
        dst_path = os.path.join(self._base_path, dst)
        os.rename(src_path, dst_path)
        return True

    def exists(self, path: str) -> bool:
        """Check if file exists."""
        full_path = os.path.join(self._base_path, path)
        return os.path.exists(full_path)

    # ── Private methods (ioctl handlers) ──────────────────────────────────

    def _read(self, *args):
        fd = args[0]
        size = args[1] if len(args) > 1 else -1
        return self.read(fd, size)

    def _write(self, *args):
        fd, data = args[0], args[1]
        return self.write(fd, data)

    def _open(self, *args):
        path = args[0]
        mode = args[1] if len(args) > 1 else "rb"
        return self.open_file(path, mode)

    def _close(self, *args):
        return self.close_file(args[0])

    def _seek(self, *args):
        fd, offset = args[0], args[1]
        whence = args[2] if len(args) > 2 else 0
        return self.seek(fd, offset, whence)

    def _tell(self, *args):
        return self.tell(args[0])

    def _stat(self, *args):
        return self.stat(args[0])

    def _list(self, *args):
        path = args[0] if len(args) > 0 else "."
        return self.list_dir(path)

    def _mkdir(self, *args):
        return self.mkdir(args[0])

    def _remove(self, *args):
        return self.remove(args[0])

    def _rename(self, *args):
        return self.rename(args[0], args[1])

    def _exists(self, *args):
        return self.exists(args[0])

    def _info(self, *args):
        return self.info()
