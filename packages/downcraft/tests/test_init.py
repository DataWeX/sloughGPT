"""Tests for downcraft.__init__ — top-level download API."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from downcraft import download, download_hf_model


def _unique_url(httpserver, path, suffix=""):
    """Create a unique URL with a random query param to avoid state collisions."""
    import random
    return httpserver.url_for(path) + f"?t={random.randint(0, 2**32)}{suffix}"


class TestDownload:
    def test_download_already_complete(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("downcraft.state.get_state") as mock_state:
                st = MagicMock()
                existing = MagicMock()
                existing.status = "complete"
                st.get.return_value = existing
                mock_state.return_value = st

                result = download("https://example.com/file", str(Path(td) / "file.bin"))
                assert result["status"] == "already_downloaded"

    def test_small_file_download(self, httpserver):
        content = b"test content for download"
        httpserver.expect_request("/test.bin").respond_with_data(content)

        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "test.bin")
            result = download(_unique_url(httpserver, "/test.bin"), dest)
            assert result["status"] == "complete"
            assert Path(dest).read_bytes() == content

    def test_download_with_progress(self, httpserver):
        content = b"x" * 50000
        httpserver.expect_request("/dlprogress.bin").respond_with_data(content)
        progress_values = []

        def _cb(done, total, speed):
            progress_values.append((done, total, speed))

        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "dlprogress.bin")
            download(_unique_url(httpserver, "/dlprogress.bin"), dest, on_progress=_cb)
            assert len(progress_values) > 0
            assert progress_values[-1][0] > 0

    def test_download_with_checksum(self, httpserver):
        import hashlib
        content = b"checksum test data"
        checksum = hashlib.sha256(content).hexdigest()
        httpserver.expect_request("/checksum.bin").respond_with_data(content)

        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "checksum.bin")
            result = download(_unique_url(httpserver, "/checksum.bin"), dest, checksum=checksum)
            assert result["status"] == "complete"

    def test_download_with_checksum_mismatch(self, httpserver):
        from downcraft.downloader import DownloadError
        content = b"data"
        httpserver.expect_request("/badchecksum.bin").respond_with_data(content)

        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "badchecksum.bin")
            with pytest.raises(DownloadError, match="Checksum mismatch"):
                download(_unique_url(httpserver, "/badchecksum.bin"), dest, checksum="wrong")

    def test_download_persists_state_on_complete(self, httpserver):
        content = b"state check"
        httpserver.expect_request("/statecheck.bin").respond_with_data(content)

        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "statecheck.bin")
            result = download(_unique_url(httpserver, "/statecheck.bin"), dest)
            assert result["status"] == "complete"
            assert result["dest"] == dest
            assert result["total_bytes"] == len(content)
            assert result["elapsed"] >= 0

    def test_download_invalid_url_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = str(Path(td) / "test.bin")
            with pytest.raises(Exception):
                download("http://localhost:1/missing", dest)
