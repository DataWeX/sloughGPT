"""Tests for VirtualDisk — flat block device emulation."""

import os
import pytest
import tempfile
from domains.shell.virtual_disk import VirtualDisk, DISK_MAGIC, DISK_VERSION, DEFAULT_BLOCK_SIZE


class TestVirtualDisk:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._disk_path = os.path.join(self._tmpdir, "test.dsk")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_create_disk(self):
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        stat = disk.stat()
        assert stat["total_blocks"] > 0
        assert stat["block_size"] == DEFAULT_BLOCK_SIZE
        assert stat["free_blocks"] == stat["total_blocks"] - 1  # block 0 is header
        disk.close()

    def test_open_existing_disk(self):
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        stat1 = disk.stat()
        disk.close()

        disk2 = VirtualDisk(self._disk_path)
        stat2 = disk2.stat()
        assert stat1["total_blocks"] == stat2["total_blocks"]
        disk2.close()

    def test_open_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            VirtualDisk("/nonexistent/path.dsk")

    def test_invalid_magic_raises(self):
        with open(self._disk_path, "wb") as f:
            f.write(b"XXXX" + b"\x00" * 4096)
        with pytest.raises(ValueError, match="invalid disk magic"):
            VirtualDisk(self._disk_path)

    def test_read_write_block(self):
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        data = b"hello" + b"\x00" * (DEFAULT_BLOCK_SIZE - 5)
        disk.write_block(1, data)
        read_back = disk.read_block(1)
        assert read_back[:5] == b"hello"
        disk.close()

    def test_write_wrong_size_raises(self):
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        with pytest.raises(ValueError, match="must be exactly"):
            disk.write_block(1, b"short")
        disk.close()

    def test_write_block_0_raises(self):
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        with pytest.raises(ValueError, match="out of range"):
            disk.write_block(0, b"\x00" * DEFAULT_BLOCK_SIZE)
        disk.close()

    def test_alloc_block(self):
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        block_num = disk.alloc_block()
        assert block_num >= 1
        assert not disk.is_block_free(block_num)
        disk.close()

    def test_free_block(self):
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        block_num = disk.alloc_block()
        disk.free_block(block_num)
        assert disk.is_block_free(block_num)
        disk.close()

    def test_free_block_0_raises(self):
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        with pytest.raises(ValueError, match="out of range"):
            disk.free_block(0)
        disk.close()

    def test_stat(self):
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        stat = disk.stat()
        assert "total_blocks" in stat
        assert "free_blocks" in stat
        assert "block_size" in stat
        assert "used_pct" in stat
        assert "path" in stat
        disk.close()

    def test_context_manager(self):
        with VirtualDisk(self._disk_path, size_mb=1, create=True) as disk:
            stat = disk.stat()
            assert stat["total_blocks"] > 0
        # File should be closed after context exit
        assert disk._file is None

    def test_persistence(self):
        # Create and write
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        block = disk.alloc_block()
        disk.write_block(block, b"persist" + b"\x00" * (DEFAULT_BLOCK_SIZE - 7))
        disk.close()

        # Reopen and read
        disk2 = VirtualDisk(self._disk_path)
        data = disk2.read_block(block)
        assert data[:7] == b"persist"
        disk2.close()

    def test_repr(self):
        disk = VirtualDisk(self._disk_path, size_mb=1, create=True)
        r = repr(disk)
        assert "VirtualDisk" in r
        assert self._disk_path in r
        disk.close()
