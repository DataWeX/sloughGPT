"""Tests for BlockDevice — unified block device with Linux-compatible interface."""

from __future__ import annotations

import os
import tempfile
import pytest

from domains.shell.vm import (
    BlockDevice,
    DiskDevice,
    FlatFS,
    BlockCompressor,
    BlockMapEntry,
    CompressionAlgo,
    BlockFlags,
    crc8,
    MAGIC,
    DeviceFault,
)


# ── CRC8 ─────────────────────────────────────────────────────────────────────

class TestCRC8:
    def test_empty_data(self):
        assert crc8(b"") == 0

    def test_known_value(self):
        # CRC8 of "hello" with poly 0x07
        result = crc8(b"hello")
        assert isinstance(result, int)
        assert 0 <= result <= 255

    def test_deterministic(self):
        assert crc8(b"test") == crc8(b"test")

    def test_different_data_different_crc(self):
        assert crc8(b"aaa") != crc8(b"bbb")


# ── BlockCompressor ──────────────────────────────────────────────────────────

class TestBlockCompressor:
    def test_gzip_always_available(self):
        comp = BlockCompressor(CompressionAlgo.GZIP)
        assert comp.available is True

    def test_gzip_compress_decompress(self):
        comp = BlockCompressor(CompressionAlgo.GZIP)
        data = b"Hello, world! " * 100
        compressed = comp.compress(data)
        assert len(compressed) < len(data)
        decompressed = comp.decompress(compressed)
        assert decompressed == data

    def test_none_algo_passthrough(self):
        comp = BlockCompressor(CompressionAlgo.NONE)
        data = b"no compression"
        assert comp.compress(data) == data
        assert comp.decompress(data) == data

    def test_estimate_ratio(self):
        comp = BlockCompressor(CompressionAlgo.GZIP)
        data = b"compressible data " * 1000
        ratio = comp.estimate_ratio(data)
        assert ratio < 1.0

    def test_lz4_fallback_if_unavailable(self):
        # Test with algo that might not be installed
        comp = BlockCompressor(CompressionAlgo.LZ4)
        data = b"test data"
        result = comp.compress(data)
        assert len(result) > 0


# ── BlockMapEntry ────────────────────────────────────────────────────────────

class TestBlockMapEntry:
    def test_pack_unpack(self):
        entry = BlockMapEntry(offset=1024, compressed_size=512, flags=0x01, crc=0xAB)
        packed = entry.pack()
        assert len(packed) == 8
        unpacked = BlockMapEntry.unpack(packed)
        assert unpacked.offset == 1024
        assert unpacked.compressed_size == 512
        assert unpacked.flags == 0x01
        assert unpacked.crc == 0xAB

    def test_default_values(self):
        entry = BlockMapEntry()
        assert entry.offset == 0
        assert entry.compressed_size == 0
        assert entry.flags == 0
        assert entry.crc == 0


# ── BlockDevice (in-memory) ──────────────────────────────────────────────────

class TestBlockDevice:
    def test_read_write_sector(self):
        dev = BlockDevice(num_sectors=16)
        data = b"sector data"
        dev.write_sector(0, data)
        result = dev.read_sector(0)
        assert bytes(result[:len(data)]) == data

    def test_sector_out_of_range(self):
        dev = BlockDevice(num_sectors=4)
        with pytest.raises(DeviceFault):
            dev.read_sector(10)
        with pytest.raises(DeviceFault):
            dev.write_sector(10, b"data")

    def test_info(self):
        dev = BlockDevice(num_sectors=8)
        info = dev.info()
        assert info["type"] == "in_memory"
        assert info["sectors"] == 8
        assert info["sector_size"] == 512


# ── BlockDevice (persistent compressed) ─────────────────────────────────

class TestBlockDevice:
    def test_create_and_open(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            assert dev._total_blocks == 0
            dev.close()

            dev2 = BlockDevice(path)
            assert dev2._block_size == 4096
            dev2.close()

    def test_write_read_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(10)

            data = b"Hello, compressed world!" + b"\x00" * (4096 - len(b"Hello, compressed world!"))
            dev.write_block(0, data)

            result = dev.read_block(0)
            assert result == data

            dev.close()

    def test_write_read_sector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(10)

            sector_data = b"sector" + b"\x00" * (512 - 6)
            dev.write_sector(0, sector_data)

            result = dev.read_sector(0)
            assert result == sector_data

            dev.close()

    def test_multiple_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(100)

            for i in range(10):
                data = f"block {i}".encode() + b"\x00" * (4096 - len(f"block {i}"))
                dev.write_block(i, data)

            for i in range(10):
                result = dev.read_block(i)
                assert result.startswith(f"block {i}".encode())

            dev.close()

    def test_read_unwritten_block_returns_zeros(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(10)

            result = dev.read_block(5)
            assert result == b"\x00" * 4096

            dev.close()

    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(100)

            stats = dev.get_stats()
            assert stats["total_blocks"] == 100
            assert stats["block_size"] == 4096
            assert stats["type"] == "persistent"

            dev.close()

    def test_ioctl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(100)

            # BLKGETSIZE
            assert dev.ioctl(0x1200) == 100 * 4096 // 512

            # BLKSSZGET
            assert dev.ioctl(0x1201) == 512

            # BLKFLSBUF
            assert dev.ioctl(0x1202) == 0

            dev.close()

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            with BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True) as dev:
                dev.allocate_blocks(10)
                dev.write_block(0, b"test" + b"\x00" * 4092)
            # Device should be closed after context exit
            assert dev._file is None

    def test_compression_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(10)

            # Write highly compressible data
            data = b"A" * 4096
            dev.write_block(0, data)

            stats = dev.get_stats()
            assert stats["compression_ratio"] < 1.0  # Should be compressed

            dev.close()

    def test_device_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            info = dev.info()
            assert info["type"] == "persistent"
            assert info["block_size"] == 4096
            assert info["sector_size"] == 512
            dev.close()

    def test_read_sectors_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(10)

            for i in range(4):
                data = f"sector {i}".encode() + b"\x00" * (512 - len(f"sector {i}"))
                dev.write_sector(i, data)

            result = dev.read_sectors(0, 4)
            assert len(result) == 4 * 512
            assert result[:512].startswith(b"sector 0")
            assert result[512:1024].startswith(b"sector 1")

            dev.close()

    def test_persistence_across_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")

            # Write data
            with BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True) as dev:
                dev.allocate_blocks(10)
                dev.write_block(0, b"persistent data" + b"\x00" * (4096 - 15))

            # Reopen and verify
            with BlockDevice(path) as dev:
                result = dev.read_block(0)
                assert result.startswith(b"persistent data")

    def test_file_not_found(self):
        with pytest.raises(DeviceFault):
            BlockDevice("/nonexistent/path/test.img")


# ── FlatFS Integration ───────────────────────────────────────────────────────

class TestFlatFSWithBlockDevice:
    def test_write_read_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(100)

            fs = FlatFS(dev)
            content = b"Hello, FlatFS!"
            fs.write("hello.txt", content)

            assert fs.exists("hello.txt")
            # FlatFS reads full sectors, so data may have trailing zeros
            result = fs.read("hello.txt")
            assert result[:len(content)] == content

            dev.close()

    def test_list_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(100)

            fs = FlatFS(dev)
            fs.write("a.txt", b"file a")
            fs.write("b.txt", b"file b")

            files = fs.list_files()
            assert "a.txt" in files
            assert "b.txt" in files

            dev.close()

    def test_delete_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(100)

            fs = FlatFS(dev)
            fs.write("delete_me.txt", b"bye bye")
            assert fs.exists("delete_me.txt")

            fs.delete("delete_me.txt")
            assert not fs.exists("delete_me.txt")

            dev.close()

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")

            # Write file
            with BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True) as dev:
                dev.allocate_blocks(100)
                fs = FlatFS(dev)
                content = b"persistent data"
                fs.write("persist.txt", content)

            # Reopen and read
            with BlockDevice(path) as dev:
                fs = FlatFS(dev)
                result = fs.read("persist.txt")
                assert result[:len(content)] == content


# ── DiskDevice Integration ──────────────────────────────────────────────────

class TestDiskDeviceWithBlockDevice:
    def test_read_write_sectors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.img")
            dev = BlockDevice(path, block_size=4096, algo=CompressionAlgo.GZIP, create=True)
            dev.allocate_blocks(10)

            disk = DiskDevice(block_device=dev)
            assert disk._total_sectors == dev.get_sectors()

            # Write sector
            data = b"Test sector data" + b"\x00" * (512 - 16)
            disk.write_sectors(0, data)

            # Read sector
            result = disk.read_sectors(0, 1)
            assert result[:16] == b"Test sector data"

            dev.close()
