"""
BlobDisk — compressed, linked-sector, read-consume disk.

Unlike VirtualDisk (raw fixed-size blocks), BlobDisk stores data as compressed
blobs in sectors. Sectors are linked by logical address into version chains.
Large writes span multiple sectors via data chains.

Design:
  - New writes APPEND — old data at the same address is not deleted
  - Reading a logical address returns the NEWEST value and consumes ALL sectors
  - Old data persists until overwritten (GC reclaims orphaned sectors)
  - Sectors are compressed on write (zlib), decompressed on read
  - Sectors are read-only once written (immutable)

On-disk format (4096-byte sectors):
  Sector 0: Header (4096 bytes)
  Sectors 1..N: Head pointer array (4 bytes per logical address)
  Next 1 sector: Allocation bitmap (1 bit per data sector)
  Next M sectors: Sector table (8 bytes per entry: addr + link)
  Remaining: Data area

  Data sector layout (4096 bytes):
    addr (4): uint32 LE — logical address
    compressed_size (4): uint32 LE
    original_size (4): uint32 LE
    link (4): uint32 LE — previous version (version chain)
    next (4): uint32 LE — next sector in this write (data chain)
    data: compressed bytes (up to 4076 bytes)
"""

from __future__ import annotations

import logging
import os
import struct
import zlib
from typing import Optional

logger = logging.getLogger("slo.shell.blob_disk")

BLOB_MAGIC = b"BLB\x00"
BLOB_VERSION = 2
SECTOR_SIZE = 4096
SECTOR_HDR_SIZE = 20  # addr + compressed_size + original_size + link + next
DATA_CAPACITY = SECTOR_SIZE - SECTOR_HDR_SIZE  # 4076 bytes per sector

# Header field offsets
_HDR_MAGIC = 0
_HDR_VERSION = 4
_HDR_SECTOR_SIZE = 8
_HDR_TOTAL_SECTORS = 12
_HDR_FREE_SECTORS = 16
_HDR_HEAD_COUNT = 20
_HDR_TABLE_COUNT = 24
_HDR_DATA_OFFSET = 28
_HDR_SIZE = SECTOR_SIZE


class BlobDisk:
    """Compressed, linked-sector, read-consume disk.

    Sectors are immutable compressed blobs, linked by logical address into
    version chains. Large writes span multiple sectors via data chains.
    Reading consumes ALL sectors in the chain (returns data, deletes sectors).
    """

    def __init__(self, path: str, max_addrs: int = 4096,
                 create: bool = False):
        self._path = os.path.expanduser(path)
        self._max_addrs = max_addrs
        self._file: Optional[object] = None

        if create:
            self._create_disk()
        self._open_disk()

    def _create_disk(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

        head_sectors = (self._max_addrs * 4 + SECTOR_SIZE - 1) // SECTOR_SIZE
        bitmap_sectors = 1
        table_sectors = (self._max_addrs * 8 + SECTOR_SIZE - 1) // SECTOR_SIZE
        meta_sectors = 1 + head_sectors + bitmap_sectors + table_sectors
        data_sectors = 128
        total_sectors = meta_sectors + data_sectors
        data_offset = meta_sectors * SECTOR_SIZE

        with open(self._path, "wb") as f:
            header = bytearray(_HDR_SIZE)
            struct.pack_into("<I", header, _HDR_MAGIC,
                             int.from_bytes(BLOB_MAGIC, "little"))
            struct.pack_into("<I", header, _HDR_VERSION, BLOB_VERSION)
            struct.pack_into("<I", header, _HDR_SECTOR_SIZE, SECTOR_SIZE)
            struct.pack_into("<I", header, _HDR_TOTAL_SECTORS, total_sectors)
            struct.pack_into("<I", header, _HDR_FREE_SECTORS, data_sectors)
            struct.pack_into("<I", header, _HDR_HEAD_COUNT, self._max_addrs)
            struct.pack_into("<I", header, _HDR_TABLE_COUNT, self._max_addrs)
            struct.pack_into("<I", header, _HDR_DATA_OFFSET, data_offset)
            f.write(header)
            f.write(b"\x00" * (head_sectors * SECTOR_SIZE))
            f.write(b"\x00" * (bitmap_sectors * SECTOR_SIZE))
            f.write(b"\x00" * (table_sectors * SECTOR_SIZE))
            f.write(b"\x00" * (data_sectors * SECTOR_SIZE))

        logger.info("created blob disk %s: %d sectors (%d data), %d addrs",
                     self._path, total_sectors, data_sectors, self._max_addrs)

    def _open_disk(self) -> None:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"disk not found: {self._path}")

        self._file = open(self._path, "r+b" if os.path.getsize(self._path) > 0 else "w+b")

        magic = self._file.read(4)
        if magic != BLOB_MAGIC:
            raise ValueError(f"invalid blob disk magic: {magic!r}")

        self._file.seek(_HDR_VERSION)
        version = struct.unpack("<I", self._file.read(4))[0]
        if version != BLOB_VERSION:
            raise ValueError(f"unsupported blob disk version: {version}")

        self._file.seek(_HDR_TOTAL_SECTORS)
        self._total_sectors = struct.unpack("<I", self._file.read(4))[0]
        self._file.seek(_HDR_FREE_SECTORS)
        self._free_sectors = struct.unpack("<I", self._file.read(4))[0]
        self._file.seek(_HDR_HEAD_COUNT)
        self._head_count = struct.unpack("<I", self._file.read(4))[0]
        self._file.seek(_HDR_DATA_OFFSET)
        self._data_offset = struct.unpack("<I", self._file.read(4))[0]

        head_sectors = (self._head_count * 4 + SECTOR_SIZE - 1) // SECTOR_SIZE
        self._head_base = SECTOR_SIZE
        self._bitmap_base = self._head_base + head_sectors * SECTOR_SIZE
        self._bitmap_size = SECTOR_SIZE
        self._table_base = self._bitmap_base + self._bitmap_size
        self._data_start_sector = 1 + head_sectors + 1

    # ── Bitmap ──────────────────────────────────────────────────────────

    def _bitmap_read(self) -> bytearray:
        self._file.seek(self._bitmap_base)
        return bytearray(self._file.read(self._bitmap_size))

    def _bitmap_write(self, bmp: bytearray) -> None:
        self._file.seek(self._bitmap_base)
        self._file.write(bmp)
        self._file.flush()

    def _sector_is_allocated(self, sector: int) -> bool:
        bmp = self._bitmap_read()
        rel = sector - self._data_start_sector
        if rel < 0:
            return True
        byte_idx = rel // 8
        bit_idx = rel % 8
        if byte_idx >= len(bmp):
            return False
        return bool(bmp[byte_idx] & (1 << bit_idx))

    def _bitmap_set(self, sector: int, allocated: bool) -> None:
        bmp = self._bitmap_read()
        rel = sector - self._data_start_sector
        byte_idx = rel // 8
        bit_idx = rel % 8
        if allocated:
            bmp[byte_idx] |= (1 << bit_idx)
        else:
            bmp[byte_idx] &= ~(1 << bit_idx)
        self._bitmap_write(bmp)

    # ── Head pointers ───────────────────────────────────────────────────

    def _read_head(self, addr: int) -> int:
        self._file.seek(self._head_base + addr * 4)
        return struct.unpack("<I", self._file.read(4))[0]

    def _write_head(self, addr: int, sector: int) -> None:
        self._file.seek(self._head_base + addr * 4)
        self._file.write(struct.pack("<I", sector))
        self._file.flush()

    # ── Sector table ────────────────────────────────────────────────────

    def _read_table(self, sector: int) -> tuple:
        rel = sector - self._data_start_sector
        self._file.seek(self._table_base + rel * 8)
        data = self._file.read(8)
        if len(data) < 8:
            return (0, 0)
        return struct.unpack("<II", data)

    def _write_table(self, sector: int, addr: int, link: int) -> None:
        rel = sector - self._data_start_sector
        self._file.seek(self._table_base + rel * 8)
        self._file.write(struct.pack("<II", addr, link))
        self._file.flush()

    # ── Sector data I/O ─────────────────────────────────────────────────

    def _sector_offset(self, sector: int) -> int:
        return self._data_offset + (sector - self._data_start_sector) * SECTOR_SIZE

    def _write_one_sector(self, sector: int, addr: int, compressed: bytes,
                          original_size: int, link: int, next_sector: int) -> None:
        header = struct.pack("<IIIII", addr, len(compressed), original_size,
                             link, next_sector)
        self._file.seek(self._sector_offset(sector))
        self._file.write(header)
        self._file.write(compressed)
        remaining = max(0, DATA_CAPACITY - len(compressed))
        if remaining > 0:
            self._file.write(b"\x00" * remaining)
        self._file.flush()

    def _read_one_sector(self, sector: int) -> tuple:
        self._file.seek(self._sector_offset(sector))
        hdr = self._file.read(SECTOR_HDR_SIZE)
        if len(hdr) < SECTOR_HDR_SIZE:
            return (0, 0, 0, 0, 0, b"")
        addr, comp_size, orig_size, link, next_sec = struct.unpack("<IIIII", hdr)
        data = b""
        if comp_size > 0:
            raw = self._file.read(comp_size)
            data = zlib.decompress(raw)
        return (addr, comp_size, orig_size, link, next_sec, data)

    # ── Allocation ──────────────────────────────────────────────────────

    def _alloc_sector(self) -> int:
        bmp = self._bitmap_read()
        num_data = self._total_sectors - self._data_start_sector
        for i in range(num_data):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx >= len(bmp):
                break
            if not (bmp[byte_idx] & (1 << bit_idx)):
                bmp[byte_idx] |= (1 << bit_idx)
                self._bitmap_write(bmp)
                self._free_sectors = max(0, self._free_sectors - 1)
                self._file.seek(_HDR_FREE_SECTORS)
                self._file.write(struct.pack("<I", self._free_sectors))
                self._file.flush()
                return self._data_start_sector + i
        raise RuntimeError("no free sectors on blob disk")

    def _alloc_sectors(self, count: int) -> list:
        """Allocate multiple contiguous sectors. Returns list of sector numbers."""
        return [self._alloc_sector() for _ in range(count)]

    def _free_sector(self, sector: int) -> None:
        self._bitmap_set(sector, False)
        self._write_table(sector, 0, 0)
        self._file.seek(self._sector_offset(sector))
        self._file.write(b"\x00" * SECTOR_SIZE)
        self._file.flush()
        self._free_sectors = min(self._total_sectors - 1, self._free_sectors + 1)
        self._file.seek(_HDR_FREE_SECTORS)
        self._file.write(struct.pack("<I", self._free_sectors))
        self._file.flush()

    def _free_chain(self, head_sector: int) -> None:
        """Free all sectors in a data chain starting from head_sector."""
        current = head_sector
        visited = set()
        while current != 0 and current not in visited:
            visited.add(current)
            _, _, _, _, next_sec, _ = self._read_one_sector(current)
            self._free_sector(current)
            current = next_sec

    # ── Public API ──────────────────────────────────────────────────────

    def write(self, addr: int, data: bytes) -> None:
        """Write data to a logical address. Appends a new version.

        Data is compressed and split across multiple sectors if needed.
        Each sector holds up to DATA_CAPACITY (4076) compressed bytes.
        """
        if addr < 0 or addr >= self._head_count:
            raise ValueError(f"address out of range: {addr}")

        old_head = self._read_head(addr)
        compressed = zlib.compress(data, level=6)

        if len(compressed) <= DATA_CAPACITY:
            # Single sector
            sector = self._alloc_sector()
            self._write_one_sector(sector, addr, compressed, len(data),
                                   old_head, 0)
            self._write_table(sector, addr, old_head)
            self._write_head(addr, sector)
        else:
            # Multi-sector: split raw data into chunks, compress each independently
            chunks = []
            offset = 0
            while offset < len(data):
                end = min(offset + DATA_CAPACITY, len(data))
                chunks.append(zlib.compress(data[offset:end], level=6))
                offset = end

            sectors = self._alloc_sectors(len(chunks))

            # Write sectors in reverse order (last sector first, so link is known)
            for i in range(len(sectors) - 1, -1, -1):
                next_sec = sectors[i + 1] if i + 1 < len(sectors) else 0
                is_last_version = (i == 0)
                link = old_head if is_last_version else 0
                self._write_one_sector(sectors[i], addr, chunks[i],
                                       len(data) if i == 0 else 0,
                                       link, next_sec)
                self._write_table(sectors[i], addr,
                                  link if is_last_version else sectors[i - 1])

            # Head points to first sector
            self._write_head(addr, sectors[0])

    def _read_chain(self, head_sector: int) -> bytes:
        """Read all sectors in a data chain and reassemble."""
        parts = []
        current = head_sector
        visited = set()
        while current != 0 and current not in visited:
            visited.add(current)
            _, _, _, _, next_sec, data = self._read_one_sector(current)
            parts.append(data)
            current = next_sec
        return b"".join(parts)

    def read(self, addr: int) -> Optional[bytes]:
        """Read data at a logical address and consume ALL sectors in the chain."""
        if addr < 0 or addr >= self._head_count:
            raise ValueError(f"address out of range: {addr}")

        head = self._read_head(addr)
        if head == 0:
            return None

        # Read the data chain
        data = self._read_chain(head)

        # Find the version link from the head sector
        _, _, _, link, _, _ = self._read_one_sector(head)

        # Free all sectors in the data chain
        self._free_chain(head)

        # Re-link version chain
        self._write_head(addr, link)

        return data

    def peek(self, addr: int) -> Optional[bytes]:
        """Read data without consuming sectors."""
        if addr < 0 or addr >= self._head_count:
            raise ValueError(f"address out of range: {addr}")

        head = self._read_head(addr)
        if head == 0:
            return None

        return self._read_chain(head)

    def peek_chain(self, addr: int) -> list:
        """Return full version chain at an address (newest first).

        Each entry is (sector_num, data_bytes).
        """
        if addr < 0 or addr >= self._head_count:
            raise ValueError(f"address out of range: {addr}")

        chain = []
        current = self._read_head(addr)
        visited = set()
        while current != 0 and current not in visited:
            visited.add(current)
            _, _, _, link, _, _ = self._read_one_sector(current)
            data = self._read_chain(current)
            chain.append((current, data))
            current = link
        return chain

    def gc(self) -> int:
        """Free sectors not reachable from any head pointer chain.

        Walks all version chains AND data chains to find reachable sectors.
        """
        reachable = set()
        for addr in range(self._head_count):
            current = self._read_head(addr)
            visited = set()
            while current != 0 and current not in visited:
                visited.add(current)
                reachable.add(current)
                _, _, _, link, next_sec, _ = self._read_one_sector(current)
                # Also follow data chain
                dc = next_sec
                dc_visited = set()
                while dc != 0 and dc not in dc_visited:
                    dc_visited.add(dc)
                    reachable.add(dc)
                    _, _, _, _, dc, _ = self._read_one_sector(dc)
                current = link

        freed = 0
        for i in range(self._data_start_sector, self._total_sectors):
            if i not in reachable and self._sector_is_allocated(i):
                self._free_sector(i)
                freed += 1
        return freed

    def stat(self) -> dict:
        addrs_used = sum(1 for a in range(self._head_count) if self._read_head(a) != 0)
        return {
            "total_sectors": self._total_sectors,
            "free_sectors": self._free_sectors,
            "sector_size": SECTOR_SIZE,
            "used_pct": round((1 - self._free_sectors / max(1, self._total_sectors)) * 100, 1),
            "max_addrs": self._head_count,
            "addresses_with_data": addrs_used,
            "path": self._path,
        }

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (f"BlobDisk({self._path!r}, sectors={self._total_sectors}, "
                f"free={self._free_sectors}, addrs={self._head_count})")
