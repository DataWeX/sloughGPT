"""
VirtualDisk — flat block device emulation with bitmap-based allocation.

Binary format (learned from .slnc and .sou patterns):
  ┌──────────────────────────────────────────┐
  │ Block 0: Header (4096 bytes)             │
  │   magic (4): b"DSK\x00"                  │
  │   version (4): uint32 LE, currently 1    │
  │   block_size (4): uint32 LE              │
  │   total_blocks (4): uint32 LE            │
  │   free_blocks (4): uint32 LE             │
  │   bitmap_offset (4): uint32 LE           │
  │   data_offset (4): uint32 LE             │
  │   reserved (4096 - 36): zeros            │
  ├──────────────────────────────────────────┤
  │ Bitmap: ceil(total_blocks / 8) bytes     │
  │   1 bit per block: 0=free, 1=allocated  │
  ├──────────────────────────────────────────┤
  │ Data blocks: block_size bytes each       │
  │   block_num 0 → file offset data_offset  │
  │   block_num 1 → data_offset + block_size │
  │   ...                                    │
  └──────────────────────────────────────────┘

Usage:
    disk = VirtualDisk("data/disks/mywork.dsk", size_mb=64, create=True)
    disk.write_block(0, b"hello" + b"\x00" * 4091)
    data = disk.read_block(0)
    block_num = disk.alloc_block()
    disk.free_block(block_num)
    disk.close()
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import Optional

logger = logging.getLogger("slo.shell.virtual_disk")

DISK_MAGIC = b"DSK\x00"
DISK_VERSION = 1
DEFAULT_BLOCK_SIZE = 4096
DEFAULT_DISK_SIZE_MB = 64

# Header field offsets (all uint32 LE except noted)
_HDR_MAGIC = 0
_HDR_VERSION = 4
_HDR_BLOCK_SIZE = 8
_HDR_TOTAL_BLOCKS = 12
_HDR_FREE_BLOCKS = 16
_HDR_BITMAP_OFFSET = 20
_HDR_DATA_OFFSET = 24
_HDR_SIZE = 4096  # first block is the header


class VirtualDisk:
    """Flat block device with bitmap-based allocation.

    Creates a fixed-size file with a bitmap tracking free/allocated blocks.
    Block 0 is the header. Data blocks start at block 1.
    """

    def __init__(self, path: str, size_mb: int = DEFAULT_DISK_SIZE_MB,
                 block_size: int = DEFAULT_BLOCK_SIZE, create: bool = False):
        """Open or create a virtual disk.

        Args:
            path: file path for the disk image
            size_mb: disk size in megabytes (only used when create=True)
            block_size: bytes per block (default 4096)
            create: if True, create a new disk image (overwrites existing)

        Side effects:
            - creates disk file if create=True
            - opens the file for read/write
        """
        self._path = os.path.expanduser(path)
        self._block_size = block_size
        self._total_blocks = (size_mb * 1024 * 1024) // block_size
        self._data_offset = _HDR_SIZE
        self._bitmap_offset = _HDR_SIZE
        self._bitmap_size = (self._total_blocks + 7) // 8
        self._file: Optional[object] = None

        if create:
            self._create_disk()
        self._open_disk()

    def _create_disk(self) -> None:
        """Initialize disk file with header + zeroed bitmap + zeroed data blocks."""
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

        total_size = self._data_offset + self._bitmap_size + (self._total_blocks * self._block_size)

        with open(self._path, "wb") as f:
            # Header (36 bytes of fields + rest zeros)
            header = bytearray(self._data_offset)
            struct.pack_into("<I", header, _HDR_MAGIC, int.from_bytes(DISK_MAGIC, "little"))
            struct.pack_into("<I", header, _HDR_VERSION, DISK_VERSION)
            struct.pack_into("<I", header, _HDR_BLOCK_SIZE, self._block_size)
            struct.pack_into("<I", header, _HDR_TOTAL_BLOCKS, self._total_blocks)
            struct.pack_into("<I", header, _HDR_FREE_BLOCKS, self._total_blocks - 1)  # block 0 is header
            struct.pack_into("<I", header, _HDR_BITMAP_OFFSET, self._bitmap_offset)
            struct.pack_into("<I", header, _HDR_DATA_OFFSET, self._data_offset)
            f.write(header)

            # Bitmap (all zeros = all blocks free)
            f.write(b"\x00" * self._bitmap_size)

            # Data blocks (all zeros)
            f.write(b"\x00" * (self._total_blocks * self._block_size))

        logger.info("created disk %s: %d blocks, %d bytes each", self._path, self._total_blocks, self._block_size)

    def _open_disk(self) -> None:
        """Open existing disk and validate header."""
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"disk not found: {self._path}")

        self._file = open(self._path, "r+b" if os.path.getsize(self._path) > 0 else "w+b")

        # Validate header
        magic = self._file.read(4)
        if magic != DISK_MAGIC:
            raise ValueError(f"invalid disk magic: {magic!r} (expected {DISK_MAGIC!r})")

        self._file.seek(_HDR_VERSION)
        version = struct.unpack("<I", self._file.read(4))[0]
        if version != DISK_VERSION:
            raise ValueError(f"unsupported disk version: {version}")

        self._file.seek(_HDR_BLOCK_SIZE)
        self._block_size = struct.unpack("<I", self._file.read(4))[0]

        self._file.seek(_HDR_TOTAL_BLOCKS)
        self._total_blocks = struct.unpack("<I", self._file.read(4))[0]

        self._file.seek(_HDR_FREE_BLOCKS)
        self._free_blocks = struct.unpack("<I", self._file.read(4))[0]

    def _seek_block(self, block_num: int) -> None:
        """Seek file pointer to the start of a data block."""
        offset = self._data_offset + (block_num * self._block_size)
        self._file.seek(offset)

    def read_block(self, block_num: int) -> bytes:
        """Read a single block from disk.

        Args:
            block_num: block number (0 = header, 1+ = data blocks)

        Returns:
            block data as bytes (exactly block_size bytes)

        Raises:
            ValueError: if block_num out of range
        """
        if block_num < 0 or block_num >= self._total_blocks:
            raise ValueError(f"block number out of range: {block_num}")
        self._seek_block(block_num)
        return self._file.read(self._block_size)

    def write_block(self, block_num: int, data: bytes) -> None:
        """Write a single block to disk.

        Args:
            block_num: block number (must be > 0 for data blocks)
            data: data to write (must be exactly block_size bytes)

        Raises:
            ValueError: if block_num out of range or data wrong size
        """
        if block_num <= 0 or block_num >= self._total_blocks:
            raise ValueError(f"block number out of range: {block_num}")
        if len(data) != self._block_size:
            raise ValueError(f"data must be exactly {self._block_size} bytes, got {len(data)}")
        self._seek_block(block_num)
        self._file.write(data)
        self._file.flush()

    def _get_bitmap(self) -> bytearray:
        """Read the allocation bitmap from disk."""
        self._file.seek(self._bitmap_offset)
        return bytearray(self._file.read(self._bitmap_size))

    def _set_bitmap(self, bitmap: bytearray) -> None:
        """Write the allocation bitmap to disk."""
        self._file.seek(self._bitmap_offset)
        self._file.write(bitmap)
        self._file.flush()

    def _update_free_count(self, delta: int) -> None:
        """Update the free block count in the header."""
        self._free_blocks = max(0, self._free_blocks + delta)
        self._file.seek(_HDR_FREE_BLOCKS)
        self._file.write(struct.pack("<I", self._free_blocks))
        self._file.flush()

    def alloc_block(self) -> int:
        """Allocate a free block. Returns block number.

        Returns:
            block number of the newly allocated block

        Raises:
            RuntimeError: if no free blocks available
        """
        bitmap = self._get_bitmap()
        for i in range(1, self._total_blocks):  # skip block 0 (header)
            byte_idx = i // 8
            bit_idx = i % 8
            if not (bitmap[byte_idx] & (1 << bit_idx)):
                # Found free block
                bitmap[byte_idx] |= (1 << bit_idx)
                self._set_bitmap(bitmap)
                self._update_free_count(-1)
                return i
        raise RuntimeError("no free blocks available on disk")

    def free_block(self, block_num: int) -> None:
        """Free a block. Updates bitmap.

        Args:
            block_num: block number to free (must be > 0)

        Raises:
            ValueError: if block_num is 0 (header) or out of range
        """
        if block_num <= 0 or block_num >= self._total_blocks:
            raise ValueError(f"block number out of range: {block_num}")
        bitmap = self._get_bitmap()
        byte_idx = block_num // 8
        bit_idx = block_num % 8
        if bitmap[byte_idx] & (1 << bit_idx):
            bitmap[byte_idx] &= ~(1 << bit_idx)
            self._set_bitmap(bitmap)
            self._update_free_count(1)

    def is_block_free(self, block_num: int) -> bool:
        """Check if a block is free."""
        bitmap = self._get_bitmap()
        byte_idx = block_num // 8
        bit_idx = block_num % 8
        return not (bitmap[byte_idx] & (1 << bit_idx))

    def stat(self) -> dict:
        """Return disk stats.

        Returns:
            dict with total_blocks, free_blocks, block_size, used_pct, path
        """
        return {
            "total_blocks": self._total_blocks,
            "free_blocks": self._free_blocks,
            "block_size": self._block_size,
            "used_pct": round((1 - self._free_blocks / self._total_blocks) * 100, 1),
            "path": self._path,
        }

    def close(self) -> None:
        """Close the disk file."""
        if self._file and not self._file.closed:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"VirtualDisk({self._path!r}, blocks={self._total_blocks}, free={self._free_blocks})"
