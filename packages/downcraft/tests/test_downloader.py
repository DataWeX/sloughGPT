"""Tests for downcraft.downloader — HTTP download with resume."""

import os
import tempfile
from pathlib import Path

import pytest

from downcraft.downloader import (
    _part_path,
    _resolve_range_start,
    _expected_size_bytes,
    download_file,
    DownloadError,
)
from conftest import RangeHandler, _range_url


class TestUtilityFunctions:
    def test_part_path_appends_sgpart(self):
        assert _part_path(Path("/tmp/file.zip")) == Path("/tmp/file.zip.sgpart")

    def test_part_path_nested(self):
        assert _part_path(Path("/a/b/c.bin")) == Path("/a/b/c.bin.sgpart")

    def test_resolve_range_start_nonexistent(self):
        p = Path("/tmp/nonexistent_file_12345.sgpart")
        assert not p.exists()
        assert _resolve_range_start(p) == 0

    def test_resolve_range_start_existing(self):
        with tempfile.TemporaryDirectory() as td:
            part = Path(td) / "test.bin.sgpart"
            part.write_bytes(b"x" * 1024)
            assert _resolve_range_start(part) == 1024

    def test_resolve_range_start_empty(self):
        with tempfile.TemporaryDirectory() as td:
            part = Path(td) / "empty.sgpart"
            part.write_text("")
            assert _resolve_range_start(part) == 0

    def test_expected_size_bytes_none(self):
        assert _expected_size_bytes(None) == 0

    def test_expected_size_bytes_zero(self):
        assert _expected_size_bytes(0) == 0

    def test_expected_size_bytes_positive(self):
        assert _expected_size_bytes(1.0) == 1073741824  # 1 GB in bytes

    def test_expected_size_bytes_fraction(self):
        assert _expected_size_bytes(0.5) == 536870912


class TestDownloadFile:
    """Tests for download_file using a local HTTP server."""

    def _write_file(self, path: Path, size: int) -> bytes:
        data = os.urandom(size)
        path.write_bytes(data)
        return data

    def test_download_small_file(self, range_server):
        content = b"hello world, this is a test file for download"
        RangeHandler.payloads["/test.bin"] = content

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.bin"
            result = download_file(_range_url(range_server, "/test.bin"), dest)
            assert result == dest
            assert dest.read_bytes() == content

    def test_download_with_expected_size(self, range_server):
        content = b"x" * 10000
        RangeHandler.payloads["/test.bin"] = content

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.bin"
            result = download_file(_range_url(range_server, "/test.bin"), dest, expected_size=10000)
            assert result == dest
            assert dest.stat().st_size == 10000

    def test_download_resume_from_partial(self, range_server):
        full_content = b"x" * 50000
        RangeHandler.payloads["/test.bin"] = full_content

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.bin"
            part = dest.with_suffix(dest.suffix + ".sgpart")
            part.write_bytes(b"x" * 20000)

            result = download_file(_range_url(range_server, "/test.bin"), dest)
            assert result == dest
            assert dest.read_bytes() == full_content
            assert not part.exists()

    def test_download_checksum_match(self, range_server):
        import hashlib
        content = b"verify me please"
        checksum = hashlib.sha256(content).hexdigest()
        RangeHandler.payloads["/test.bin"] = content

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.bin"
            download_file(_range_url(range_server, "/test.bin"), dest, checksum=checksum)
            assert dest.read_bytes() == content

    def test_download_checksum_mismatch(self, range_server):
        content = b"some content"
        RangeHandler.payloads["/test.bin"] = content

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.bin"
            with pytest.raises(DownloadError, match="Checksum mismatch"):
                download_file(_range_url(range_server, "/test.bin"), dest, checksum="wrongchecksum")

    def test_download_creates_parent_dirs(self, range_server):
        content = b"hello"
        RangeHandler.payloads["/test.bin"] = content

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "subdir" / "nested" / "test.bin"
            result = download_file(_range_url(range_server, "/test.bin"), dest)
            assert result == dest
            assert dest.read_bytes() == content

    def test_download_invalid_url(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.bin"
            with pytest.raises(DownloadError):
                download_file("http://localhost:1/nonexistent", dest)

    def test_download_calls_on_chunk(self, range_server):
        content = b"x" * 100000
        RangeHandler.payloads["/test.bin"] = content
        chunks = []

        def _cb(done, total):
            chunks.append((done, total))

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.bin"
            download_file(_range_url(range_server, "/test.bin"), dest, on_chunk=_cb)
            assert len(chunks) > 0
            assert chunks[-1][0] == len(content)

    def test_download_calls_on_complete(self, range_server):
        content = b"done"
        RangeHandler.payloads["/test.bin"] = content
        completed = []

        def _cb(path):
            completed.append(path)

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.bin"
            download_file(_range_url(range_server, "/test.bin"), dest, on_complete=_cb)
            assert len(completed) == 1
            assert completed[0] == dest

    def test_download_http_404(self, range_server):
        # No payload registered for /missing -> the server returns 404.
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.bin"
            with pytest.raises(DownloadError):
                download_file(_range_url(range_server, "/missing"), dest)
