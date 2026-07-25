"""Tests for BlobDisk — compressed, linked-sector, read-consume disk."""

import os
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from domains.shell.blob_disk import BlobDisk, SECTOR_SIZE, DATA_CAPACITY


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def disk_path(tmp_dir):
    return os.path.join(tmp_dir, "test.blk")


@pytest.fixture
def disk(disk_path):
    d = BlobDisk(disk_path, max_addrs=64, create=True)
    yield d
    d.close()


class TestBlobDiskCreate:
    def test_create(self, disk_path):
        d = BlobDisk(disk_path, max_addrs=64, create=True)
        stat = d.stat()
        assert stat["total_sectors"] > 0
        assert stat["sector_size"] == SECTOR_SIZE
        assert stat["max_addrs"] == 64
        d.close()

    def test_open_existing(self, disk_path):
        d = BlobDisk(disk_path, max_addrs=64, create=True)
        d.close()
        d2 = BlobDisk(disk_path)
        assert d2.stat()["total_sectors"] > 0
        d2.close()

    def test_open_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            BlobDisk("/nonexistent/path.blk")

    def test_invalid_magic(self, disk_path):
        with open(disk_path, "wb") as f:
            f.write(b"BAD\x00" + b"\x00" * 4092)
        with pytest.raises(ValueError, match="magic"):
            BlobDisk(disk_path)


class TestBlobDiskWriteRead:
    def test_write_read(self, disk):
        disk.write(0, b"hello world")
        assert disk.read(0) == b"hello world"

    def test_read_returns_none_for_empty(self, disk):
        assert disk.read(0) is None

    def test_read_consumes_sector(self, disk):
        disk.write(0, b"data")
        disk.read(0)
        assert disk.read(0) is None

    def test_write_multiple_addresses(self, disk):
        disk.write(0, b"zero")
        disk.write(1, b"one")
        disk.write(2, b"two")
        assert disk.read(0) == b"zero"
        assert disk.read(1) == b"one"
        assert disk.read(2) == b"two"

    def test_peek_does_not_consume(self, disk):
        disk.write(0, b"peeked")
        assert disk.peek(0) == b"peeked"
        assert disk.peek(0) == b"peeked"
        assert disk.read(0) == b"peeked"

    def test_peek_returns_none_for_empty(self, disk):
        assert disk.peek(0) is None


class TestBlobDiskMultiSector:
    def test_large_compressible_data(self, disk):
        data = b"hello" * 2000  # ~10KB, compresses well
        disk.write(0, data)
        result = disk.read(0)
        assert result == data

    def test_multi_sector_read_write(self, disk):
        # Data that compresses to > 4076 bytes (one sector)
        data = os.urandom(500) * 20  # 10KB with repetition
        disk.write(0, data)
        assert disk.peek(0) == data
        assert disk.read(0) == data

    def test_max_single_sector(self, disk):
        data = b"X" * 4076
        disk.write(0, data)
        assert disk.read(0) == data

    def test_multi_sector_peek_chain(self, disk):
        disk.write(0, b"v1_short")
        data = b"A" * 5000  # multi-sector
        disk.write(0, data)
        chain = disk.peek_chain(0)
        assert len(chain) == 2
        assert chain[0][1] == data  # newest
        assert chain[1][1] == b"v1_short"

    def test_large_binary_data(self, disk):
        data = os.urandom(8000)
        disk.write(0, data)
        assert disk.read(0) == data

    def test_small_data_still_works(self, disk):
        disk.write(0, b"tiny")
        assert disk.read(0) == b"tiny"

    def test_empty_data(self, disk):
        disk.write(0, b"")
        assert disk.read(0) == b""


class TestBlobDiskVersionChains:
    def test_overwrite_creates_version(self, disk):
        disk.write(0, b"v1")
        disk.write(0, b"v2")
        assert disk.read(0) == b"v2"
        chain = disk.peek_chain(0)
        assert len(chain) == 1
        assert chain[0][1] == b"v1"

    def test_three_versions(self, disk):
        disk.write(0, b"v1")
        disk.write(0, b"v2")
        disk.write(0, b"v3")
        assert disk.read(0) == b"v3"
        assert disk.read(0) == b"v2"
        assert disk.read(0) == b"v1"
        assert disk.read(0) is None

    def test_read_then_write(self, disk):
        disk.write(0, b"old")
        disk.read(0)
        disk.write(0, b"new")
        assert disk.read(0) == b"new"


class TestBlobDiskPersistence:
    def test_survives_close_reopen(self, disk_path):
        d = BlobDisk(disk_path, max_addrs=64, create=True)
        d.write(0, b"persistent")
        d.write(1, b"data")
        d.close()

        d2 = BlobDisk(disk_path)
        assert d2.peek(0) == b"persistent"
        assert d2.peek(1) == b"data"
        d2.close()

    def test_read_after_reopen(self, disk_path):
        d = BlobDisk(disk_path, max_addrs=64, create=True)
        d.write(0, b"saved")
        d.close()

        d2 = BlobDisk(disk_path)
        assert d2.read(0) == b"saved"
        assert d2.read(0) is None
        d2.close()

    def test_multi_sector_survives_reopen(self, disk_path):
        data = b"B" * 5000
        d = BlobDisk(disk_path, max_addrs=64, create=True)
        d.write(0, data)
        d.close()

        d2 = BlobDisk(disk_path)
        assert d2.peek(0) == data
        d2.close()


class TestBlobDiskEdgeCases:
    def test_address_out_of_range(self, disk):
        with pytest.raises(ValueError, match="address"):
            disk.write(100, b"data")
        with pytest.raises(ValueError, match="address"):
            disk.read(100)

    def test_context_manager(self, disk_path):
        with BlobDisk(disk_path, max_addrs=64, create=True) as d:
            d.write(0, b"ctx")
            assert d.peek(0) == b"ctx"

    def test_stat(self, disk):
        stat = disk.stat()
        assert "total_sectors" in stat
        assert "free_sectors" in stat
        assert "used_pct" in stat
        assert stat["addresses_with_data"] == 0
        disk.write(0, b"x")
        stat = disk.stat()
        assert stat["addresses_with_data"] == 1

    def test_repr(self, disk):
        r = repr(disk)
        assert "BlobDisk" in r
