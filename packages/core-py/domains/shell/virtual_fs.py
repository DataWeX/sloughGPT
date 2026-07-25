"""
VirtualFS — inode-based filesystem on VirtualDisk.

Provides file and directory operations (create, read, write, delete, mkdir,
listdir, rename) on top of a block device.

On-disk layout:
  Block 0: header + bitmap (managed by VirtualDisk)
  Block 1+: data blocks

  Inodes are stored in data blocks starting at block 1.
  Each inode: mode(2) + size(4) + blocks[10](4 each) + indirect(4) + ts(4) = 58 bytes
  Directory entry: inode_num(2) + name_len(1) + name(255) = 258 bytes

  Root directory is inode 1 (inode 0 is reserved).

Usage:
    disk = VirtualDisk("my.dsk", create=True)
    vfs = VirtualFS(disk)
    vfs.mkdir("/")
    vfs.create("/hello.txt")
    vfs.write("/hello.txt", b"Hello, World!")
    data = vfs.read("/hello.txt")
"""

from __future__ import annotations

import logging
import struct
import time
from typing import Optional

from domains.shell.virtual_disk import VirtualDisk

logger = logging.getLogger("slo.shell.virtual_fs")

# Inode constants
_INODE_SIZE = 58          # bytes per inode on disk
_INODE_MODE = 0           # offset within inode (uint16)
_INODE_SIZE_OFF = 2       # offset of file size (uint32)
_INODE_BLOCKS = 6         # offset of direct block pointers (10 × uint32)
_INODE_INDIRECT = 46      # offset of indirect block pointer (uint32)
_INODE_TIMESTAMP = 50     # offset of timestamp (uint32)
_INODE_NUM_DIRECT = 10
_INODE_BLOCK_PTR_SIZE = 4

# Directory entry constants
_DIRENTRY_SIZE = 258      # bytes per directory entry
_DIRENTRY_INODE = 0       # offset of inode number (uint16)
_DIRENTRY_NAME_LEN = 2    # offset of name length (uint8)
_DIRENTRY_NAME = 3        # offset of name (up to 255 bytes)

# File modes
S_IFREG = 0o100000
S_IFDIR = 0o040000
S_IFMT = 0o170000

# Special inode numbers
ROOT_INODE = 1


class Inode:
    """In-memory representation of an inode."""

    __slots__ = ("num", "mode", "size", "direct", "indirect", "timestamp")

    def __init__(self, num: int = 0, mode: int = 0, size: int = 0,
                 direct: Optional[list[int]] = None, indirect: int = 0,
                 timestamp: int = 0):
        self.num = num
        self.mode = mode
        self.size = size
        self.direct = direct or [0] * _INODE_NUM_DIRECT
        self.indirect = indirect
        self.timestamp = timestamp

    @property
    def is_dir(self) -> bool:
        return (self.mode & S_IFMT) == S_IFDIR

    @property
    def is_file(self) -> bool:
        return (self.mode & S_IFMT) == S_IFREG


class VirtualFS:
    """Inode-based filesystem on a VirtualDisk."""

    def __init__(self, disk: VirtualDisk):
        """Open a filesystem on an existing disk.

        Args:
            disk: opened VirtualDisk instance

        Side effects:
            - reads root directory inode
            - initializes inode cache
        """
        self._disk = disk
        self._block_size = disk._block_size
        self._inode_cache: dict[int, Inode] = {}
        self._dirty_inodes: set[int] = set()

        # Inodes start at block 1
        self._inode_blocks_start = 1
        # Calculate how many blocks inodes can use
        # Reserve some blocks for directory entries and file data
        max_inodes = (self._block_size // _INODE_SIZE) * 100  # ~100 blocks for inodes
        self._dir_blocks_start = self._inode_blocks_start + min(max_inodes, 50)

        # Ensure root inode exists
        if not self._inode_exists(ROOT_INODE):
            self._init_root()

    def _inode_exists(self, inode_num: int) -> bool:
        """Check if an inode number has been allocated."""
        try:
            data = self._read_inode_raw(inode_num)
            return data[0] != 0 or data[1] != 0  # mode != 0 means allocated
        except (ValueError, RuntimeError):
            return False

    def _init_root(self) -> None:
        """Initialize root directory inode."""
        inode = Inode(num=ROOT_INODE, mode=S_IFDIR | 0o755, timestamp=int(time.time()))
        self._write_inode(inode)
        self._inode_cache[ROOT_INODE] = inode

    def _read_inode_raw(self, inode_num: int) -> bytes:
        """Read raw inode bytes from disk."""
        # Inodes are packed into blocks starting at block 1
        bytes_per_block = self._block_size
        inodes_per_block = bytes_per_block // _INODE_SIZE
        block_num = self._inode_blocks_start + (inode_num // inodes_per_block)
        offset_in_block = (inode_num % inodes_per_block) * _INODE_SIZE
        raw = self._disk.read_block(block_num)
        return raw[offset_in_block:offset_in_block + _INODE_SIZE]

    def _write_inode_raw(self, inode_num: int, data: bytes) -> None:
        """Write raw inode bytes to disk."""
        bytes_per_block = self._block_size
        inodes_per_block = bytes_per_block // _INODE_SIZE
        block_num = self._inode_blocks_start + (inode_num // inodes_per_block)
        offset_in_block = (inode_num % inodes_per_block) * _INODE_SIZE
        block = bytearray(self._disk.read_block(block_num))
        block[offset_in_block:offset_in_block + _INODE_SIZE] = data
        self._disk.write_block(block_num, bytes(block))

    def _read_inode(self, inode_num: int) -> Inode:
        """Read an inode from disk (with caching)."""
        if inode_num in self._inode_cache:
            return self._inode_cache[inode_num]

        raw = self._read_inode_raw(inode_num)
        mode = struct.unpack_from("<H", raw, _INODE_MODE)[0]
        size = struct.unpack_from("<I", raw, _INODE_SIZE_OFF)[0]
        direct = []
        for i in range(_INODE_NUM_DIRECT):
            off = _INODE_BLOCKS + i * _INODE_BLOCK_PTR_SIZE
            direct.append(struct.unpack_from("<I", raw, off)[0])
        indirect = struct.unpack_from("<I", raw, _INODE_INDIRECT)[0]
        ts = struct.unpack_from("<I", raw, _INODE_TIMESTAMP)[0]

        inode = Inode(inode_num, mode, size, direct, indirect, ts)
        self._inode_cache[inode_num] = inode
        return inode

    def _write_inode(self, inode: Inode) -> None:
        """Write an inode to disk."""
        buf = bytearray(_INODE_SIZE)
        struct.pack_into("<H", buf, _INODE_MODE, inode.mode)
        struct.pack_into("<I", buf, _INODE_SIZE_OFF, inode.size)
        for i, blk in enumerate(inode.direct):
            off = _INODE_BLOCKS + i * _INODE_BLOCK_PTR_SIZE
            struct.pack_into("<I", buf, off, blk)
        struct.pack_into("<I", buf, _INODE_INDIRECT, inode.indirect)
        struct.pack_into("<I", buf, _INODE_TIMESTAMP, inode.timestamp)
        self._write_inode_raw(inode.num, bytes(buf))
        self._inode_cache[inode.num] = inode

    def _alloc_inode(self) -> int:
        """Allocate a new inode number."""
        for i in range(2, 10000):  # skip 0 (reserved) and 1 (root)
            if not self._inode_exists(i):
                return i
        raise RuntimeError("no free inodes available")

    def _path_to_parts(self, path: str) -> list[str]:
        """Split a path into parts, filtering empty strings."""
        return [p for p in path.strip("/").split("/") if p]

    def _find_inode(self, path: str) -> Optional[int]:
        """Walk a path and return the inode number of the final component.

        Returns None if any component along the path doesn't exist.
        """
        parts = self._path_to_parts(path)
        if not parts:
            return ROOT_INODE

        current_inode = ROOT_INODE
        for part in parts:
            inode = self._read_inode(current_inode)
            if not inode.is_dir:
                return None
            entry = self._dir_lookup(current_inode, part)
            if entry is None:
                return None
            current_inode = entry
        return current_inode

    def _dir_lookup(self, dir_inode_num: int, name: str) -> Optional[int]:
        """Look up a name in a directory. Returns inode number or None."""
        inode = self._read_inode(dir_inode_num)
        data = self._read_file_data(inode)
        name_bytes = name.encode("utf-8")[:255]
        offset = 0
        while offset + _DIRENTRY_SIZE <= len(data):
            entry_inode = struct.unpack_from("<H", data, offset + _DIRENTRY_INODE)[0]
            entry_name_len = data[offset + _DIRENTRY_NAME_LEN]
            entry_name = data[offset + _DIRENTRY_NAME:offset + _DIRENTRY_NAME + entry_name_len]
            if entry_inode != 0 and entry_name == name_bytes:
                return entry_inode
            offset += _DIRENTRY_SIZE
        return None

    def _dir_add_entry(self, dir_inode_num: int, name: str, inode_num: int) -> None:
        """Add a directory entry."""
        inode = self._read_inode(dir_inode_num)
        name_bytes = name.encode("utf-8")[:255]
        entry = bytearray(_DIRENTRY_SIZE)
        struct.pack_into("<H", entry, _DIRENTRY_INODE, inode_num)
        entry[_DIRENTRY_NAME_LEN] = len(name_bytes)
        entry[_DIRENTRY_NAME:_DIRENTRY_NAME + len(name_bytes)] = name_bytes

        # Append to existing data
        existing = self._read_file_data(inode)
        new_data = existing + bytes(entry)
        self._write_file_data(inode, new_data)

    def _dir_remove_entry(self, dir_inode_num: int, name: str) -> bool:
        """Remove a directory entry by name."""
        inode = self._read_inode(dir_inode_num)
        data = bytearray(self._read_file_data(inode))
        name_bytes = name.encode("utf-8")[:255]
        offset = 0
        while offset + _DIRENTRY_SIZE <= len(data):
            entry_inode = struct.unpack_from("<H", data, offset + _DIRENTRY_INODE)[0]
            entry_name_len = data[offset + _DIRENTRY_NAME_LEN]
            entry_name = data[offset + _DIRENTRY_NAME:offset + _DIRENTRY_NAME + entry_name_len]
            if entry_inode != 0 and entry_name == name_bytes:
                # Zero out the inode number to mark as deleted
                struct.pack_into("<H", data, offset + _DIRENTRY_INODE, 0)
                self._write_file_data(inode, bytes(data))
                return True
            offset += _DIRENTRY_SIZE
        return False

    def _read_file_data(self, inode: Inode) -> bytes:
        """Read all data blocks of an inode."""
        result = bytearray()
        remaining = inode.size
        for blk in inode.direct:
            if blk == 0 or remaining <= 0:
                break
            block_data = self._disk.read_block(blk)
            take = min(len(block_data), remaining)
            result.extend(block_data[:take])
            remaining -= take

        # Handle indirect block
        if remaining > 0 and inode.indirect > 0:
            indirect_data = self._disk.read_block(inode.indirect)
            ptrs_per_block = self._block_size // _INODE_BLOCK_PTR_SIZE
            for i in range(ptrs_per_block):
                if remaining <= 0:
                    break
                ptr = struct.unpack_from("<I", indirect_data, i * _INODE_BLOCK_PTR_SIZE)[0]
                if ptr == 0:
                    break
                block_data = self._disk.read_block(ptr)
                take = min(len(block_data), remaining)
                result.extend(block_data[:take])
                remaining -= take

        return bytes(result)

    def _write_file_data(self, inode: Inode, data: bytes) -> None:
        """Write data to an inode, allocating/freeing blocks as needed."""
        blocks_needed = (len(data) + self._block_size - 1) // self._block_size

        # Free excess blocks
        for i in range(blocks_needed, _INODE_NUM_DIRECT):
            if inode.direct[i] != 0:
                self._disk.free_block(inode.direct[i])
                inode.direct[i] = 0

        # Write direct blocks
        offset = 0
        for i in range(min(blocks_needed, _INODE_NUM_DIRECT)):
            chunk = data[offset:offset + self._block_size]
            if inode.direct[i] == 0:
                inode.direct[i] = self._disk.alloc_block()
            self._disk.write_block(inode.direct[i], chunk.ljust(self._block_size, b"\x00"))
            offset += self._block_size

        # Handle overflow via indirect block
        if blocks_needed > _INODE_NUM_DIRECT:
            if inode.indirect == 0:
                inode.indirect = self._disk.alloc_block()
            remaining = data[offset:]
            ptrs_per_block = self._block_size // _INODE_BLOCK_PTR_SIZE
            indirect_data = bytearray(self._disk.read_block(inode.indirect))
            for i in range(min(len(remaining) // self._block_size + 1, ptrs_per_block)):
                chunk = remaining[i * self._block_size:(i + 1) * self._block_size]
                if not chunk:
                    break
                ptr = struct.unpack_from("<I", indirect_data, i * _INODE_BLOCK_PTR_SIZE)[0]
                if ptr == 0:
                    ptr = self._disk.alloc_block()
                    struct.pack_into("<I", indirect_data, i * _INODE_BLOCK_PTR_SIZE, ptr)
                self._disk.write_block(ptr, chunk.ljust(self._block_size, b"\x00"))
            self._disk.write_block(inode.indirect, bytes(indirect_data))
        elif inode.indirect != 0:
            # Free indirect block and its data blocks
            indirect_data = self._disk.read_block(inode.indirect)
            ptrs_per_block = self._block_size // _INODE_BLOCK_PTR_SIZE
            for i in range(ptrs_per_block):
                ptr = struct.unpack_from("<I", indirect_data, i * _INODE_BLOCK_PTR_SIZE)[0]
                if ptr != 0:
                    self._disk.free_block(ptr)
            self._disk.free_block(inode.indirect)
            inode.indirect = 0

        inode.size = len(data)
        inode.timestamp = int(time.time())
        self._write_inode(inode)

    def _free_all_blocks(self, inode: Inode) -> None:
        """Free all data blocks owned by an inode."""
        for blk in inode.direct:
            if blk != 0:
                self._disk.free_block(blk)
                inode.direct[blk] = 0  # not a valid assignment, just reset below
        inode.direct = [0] * _INODE_NUM_DIRECT

        if inode.indirect != 0:
            indirect_data = self._disk.read_block(inode.indirect)
            ptrs_per_block = self._block_size // _INODE_BLOCK_PTR_SIZE
            for i in range(ptrs_per_block):
                ptr = struct.unpack_from("<I", indirect_data, i * _INODE_BLOCK_PTR_SIZE)[0]
                if ptr != 0:
                    self._disk.free_block(ptr)
            self._disk.free_block(inode.indirect)
            inode.indirect = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def create(self, path: str, mode: int = 0o100644) -> int:
        """Create a file. Returns inode number.

        Args:
            path: absolute path (e.g. "/data/hello.txt")
            mode: file mode (default: regular file, rw-r--r--)

        Returns:
            inode number of the created file

        Raises:
            FileExistsError: if file already exists
            FileNotFoundError: if parent directory doesn't exist
        """
        parts = self._path_to_parts(path)
        if not parts:
            raise ValueError("cannot create root directory")

        name = parts[-1]
        parent_path = "/".join(parts[:-1])
        parent_inode_num = self._find_inode(f"/{parent_path}" if parent_path else "/")
        if parent_inode_num is None:
            raise FileNotFoundError(f"parent directory not found: /{parent_path}")

        # Check if already exists
        existing = self._dir_lookup(parent_inode_num, name)
        if existing is not None:
            raise FileExistsError(f"file already exists: {path}")

        # Allocate inode
        inode_num = self._alloc_inode()
        inode = Inode(num=inode_num, mode=mode | S_IFREG, timestamp=int(time.time()))
        self._write_inode(inode)

        # Add directory entry
        self._dir_add_entry(parent_inode_num, name, inode_num)
        return inode_num

    def mkdir(self, path: str, mode: int = 0o40755) -> int:
        """Create a directory. Returns inode number.

        Args:
            path: absolute path (e.g. "/data")
            mode: directory mode

        Returns:
            inode number of the created directory

        Raises:
            FileExistsError: if directory already exists
            FileNotFoundError: if parent directory doesn't exist
        """
        parts = self._path_to_parts(path)
        if not parts:
            raise ValueError("root directory already exists")

        name = parts[-1]
        parent_path = "/".join(parts[:-1])
        parent_inode_num = self._find_inode(f"/{parent_path}" if parent_path else "/")
        if parent_inode_num is None:
            raise FileNotFoundError(f"parent directory not found: /{parent_path}")

        existing = self._dir_lookup(parent_inode_num, name)
        if existing is not None:
            raise FileExistsError(f"directory already exists: {path}")

        inode_num = self._alloc_inode()
        inode = Inode(num=inode_num, mode=mode | S_IFDIR, timestamp=int(time.time()))
        self._write_inode(inode)

        self._dir_add_entry(parent_inode_num, name, inode_num)
        return inode_num

    def read(self, path: str) -> bytes:
        """Read entire file content.

        Args:
            path: absolute path to file

        Returns:
            file content as bytes

        Raises:
            FileNotFoundError: if path doesn't exist
            IsADirectoryError: if path is a directory
        """
        inode_num = self._find_inode(path)
        if inode_num is None:
            raise FileNotFoundError(f"file not found: {path}")
        inode = self._read_inode(inode_num)
        if inode.is_dir:
            raise IsADirectoryError(f"is a directory: {path}")
        return self._read_file_data(inode)

    def write(self, path: str, data: bytes) -> None:
        """Write data to a file. Creates if doesn't exist.

        Args:
            path: absolute path to file
            data: bytes to write

        Side effects:
            - creates file if it doesn't exist
            - overwrites existing content
            - allocates/frees blocks as needed
        """
        parts = self._path_to_parts(path)
        if not parts:
            raise ValueError("cannot write to root directory")

        inode_num = self._find_inode(path)
        if inode_num is None:
            # Create the file first
            self.create(path)
            inode_num = self._find_inode(path)

        inode = self._read_inode(inode_num)
        if inode.is_dir:
            raise IsADirectoryError(f"is a directory: {path}")
        self._write_file_data(inode, data)

    def delete(self, path: str) -> bool:
        """Remove a file or empty directory.

        Args:
            path: absolute path to remove

        Returns:
            True if removed, False if not found
        """
        parts = self._path_to_parts(path)
        if not parts:
            return False  # cannot delete root

        name = parts[-1]
        parent_path = "/".join(parts[:-1])
        parent_inode_num = self._find_inode(f"/{parent_path}" if parent_path else "/")
        if parent_inode_num is None:
            return False

        inode_num = self._dir_lookup(parent_inode_num, name)
        if inode_num is None:
            return False

        inode = self._read_inode(inode_num)

        # Non-empty directory check
        if inode.is_dir:
            data = self._read_file_data(inode)
            if len(data) > 0:
                raise OSError(f"directory not empty: {path}")

        # Remove directory entry
        self._dir_remove_entry(parent_inode_num, name)

        # Free blocks
        self._free_all_blocks(inode)
        self._inode_cache.pop(inode_num, None)
        return True

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        return self._find_inode(path) is not None

    def listdir(self, path: str) -> list[str]:
        """List directory contents.

        Args:
            path: absolute path to directory

        Returns:
            sorted list of entry names

        Raises:
            FileNotFoundError: if path doesn't exist
            NotADirectoryError: if path is not a directory
        """
        inode_num = self._find_inode(path)
        if inode_num is None:
            raise FileNotFoundError(f"directory not found: {path}")
        inode = self._read_inode(inode_num)
        if not inode.is_dir:
            raise NotADirectoryError(f"not a directory: {path}")

        data = self._read_file_data(inode)
        entries = []
        offset = 0
        while offset + _DIRENTRY_SIZE <= len(data):
            entry_inode = struct.unpack_from("<H", data, offset + _DIRENTRY_INODE)[0]
            entry_name_len = data[offset + _DIRENTRY_NAME_LEN]
            if entry_inode != 0 and entry_name_len > 0:
                entry_name = data[offset + _DIRENTRY_NAME:offset + _DIRENTRY_NAME + entry_name_len]
                entries.append(entry_name.decode("utf-8", errors="replace"))
            offset += _DIRENTRY_SIZE
        return sorted(entries)

    def rename(self, old_path: str, new_path: str) -> None:
        """Rename a file or directory.

        Args:
            old_path: current absolute path
            new_path: new absolute path

        Raises:
            FileNotFoundError: if old path doesn't exist
            FileExistsError: if new path already exists
        """
        old_parts = self._path_to_parts(old_path)
        new_parts = self._path_to_parts(new_path)
        if not old_parts or not new_parts:
            raise ValueError("cannot rename root directory")

        old_name = old_parts[-1]
        new_name = new_parts[-1]
        old_parent_path = "/".join(old_parts[:-1])
        new_parent_path = "/".join(new_parts[:-1])

        old_parent = self._find_inode(f"/{old_parent_path}" if old_parent_path else "/")
        new_parent = self._find_inode(f"/{new_parent_path}" if new_parent_path else "/")

        if old_parent is None:
            raise FileNotFoundError(f"source directory not found: /{old_parent_path}")
        if new_parent is None:
            raise FileNotFoundError(f"target directory not found: /{new_parent_path}")

        inode_num = self._dir_lookup(old_parent, old_name)
        if inode_num is None:
            raise FileNotFoundError(f"file not found: {old_path}")

        # Check new name doesn't exist
        if self._dir_lookup(new_parent, new_name) is not None:
            raise FileExistsError(f"destination already exists: {new_path}")

        # Remove from old parent, add to new parent
        self._dir_remove_entry(old_parent, old_name)
        self._dir_add_entry(new_parent, new_name, inode_num)

    def stat(self, path: str) -> dict:
        """Get file/directory stats.

        Returns:
            dict with mode, size, inode, is_dir, is_file, timestamp
        """
        inode_num = self._find_inode(path)
        if inode_num is None:
            raise FileNotFoundError(f"file not found: {path}")
        inode = self._read_inode(inode_num)
        return {
            "mode": inode.mode,
            "size": inode.size,
            "inode": inode.num,
            "is_dir": inode.is_dir,
            "is_file": inode.is_file,
            "timestamp": inode.timestamp,
        }

    def read_text(self, path: str) -> str:
        """Read file content as UTF-8 text."""
        return self.read(path).decode("utf-8")

    def write_text(self, path: str, data: str) -> None:
        """Write text as UTF-8 to a file."""
        self.write(path, data.encode("utf-8"))
