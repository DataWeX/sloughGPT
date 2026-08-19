"""Tests for downcraft.__init__ — top-level download API."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from downcraft import download
from conftest import RangeHandler, _range_url


class TestDownload:
    def test_download_already_complete(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("downcraft.download.state.get_state") as mock_state:
                st = MagicMock()
                existing = MagicMock()
                existing.status = "complete"
                st.get.return_value = existing
                mock_state.return_value = st

                result = download("https://example.com/file", str(Path(td) / "file.bin"))
                assert result["status"] == "already_downloaded"

    def test_small_file_download(self, range_server):
        content = b"test content for download"
        RangeHandler.payloads["/test.bin"] = content

        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "test.bin")
            result = download(_range_url(range_server, "/test.bin"), dest)
            assert result["status"] == "complete"
            assert Path(dest).read_bytes() == content

    def test_download_with_progress(self, range_server):
        content = b"x" * 50000
        RangeHandler.payloads["/dlprogress.bin"] = content
        progress_values = []

        def _cb(done, total, speed):
            progress_values.append((done, total, speed))

        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "dlprogress.bin")
            download(_range_url(range_server, "/dlprogress.bin"), dest, on_progress=_cb)
            assert len(progress_values) > 0
            assert progress_values[-1][0] > 0

    def test_download_with_checksum(self, range_server):
        import hashlib
        content = b"checksum test data"
        checksum = hashlib.sha256(content).hexdigest()
        RangeHandler.payloads["/checksum.bin"] = content

        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "checksum.bin")
            result = download(_range_url(range_server, "/checksum.bin"), dest, checksum=checksum)
            assert result["status"] == "complete"

    def test_download_with_checksum_mismatch(self, range_server):
        from downcraft.download.http import DownloadError
        content = b"data"
        RangeHandler.payloads["/badchecksum.bin"] = content

        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "badchecksum.bin")
            with pytest.raises(DownloadError, match="Checksum mismatch"):
                download(_range_url(range_server, "/badchecksum.bin"), dest, checksum="wrong")

    def test_download_persists_state_on_complete(self, range_server):
        content = b"state check"
        RangeHandler.payloads["/statecheck.bin"] = content

        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "statecheck.bin")
            result = download(_range_url(range_server, "/statecheck.bin"), dest)
            assert result["status"] == "complete"
            assert result["dest"] == dest
            assert result["total_bytes"] == len(content)
            assert result["elapsed"] >= 0

    def test_download_invalid_url_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "test.bin")
            with pytest.raises(Exception):
                download("http://localhost:1/missing", dest)
