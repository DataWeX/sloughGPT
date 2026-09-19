"""Tests for download_range — partial byte-slice download to save data.

Uses a self-contained local HTTP server with full ``bytes=start-end``
Range support. (The shared ``range_server`` fixture in ``conftest.py`` is
not used here: under ``--import-mode=importlib`` its ``RangeHandler`` class
object differs from the one tests import, so per-test payloads never reach
the server — a pre-existing issue also breaking ``test_downloader.py``.)
"""

import hashlib
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from downcraft.download.http import DownloadError, download_range

# Deterministic 1024-byte payload: byte i == i % 256
PAYLOAD = bytes(range(256)) * 4


class SliceHandler(BaseHTTPRequestHandler):
    """Serves one payload with full bytes=start-end Range support."""

    payload: bytes = b""
    ignore_range: bool = False

    def do_GET(self):
        if self.path.split("?")[0] != "/big.bin":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        total = len(type(self).payload)
        rng = self.headers.get("Range")
        if type(self).ignore_range or not rng:
            self.send_response(200)
            self.send_header("Content-Length", str(total))
            self.end_headers()
            self.wfile.write(type(self).payload)
            return
        m = re.match(r"bytes=(\d+)-(\d+)$", rng)
        if not m:
            self.send_response(400)
            self.end_headers()
            return
        start, end = int(m.group(1)), int(m.group(2))
        if start >= total:
            self.send_response(416)
            self.end_headers()
            return
        end = min(end, total - 1)
        data = type(self).payload[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def slice_server():
    SliceHandler.payload = PAYLOAD
    SliceHandler.ignore_range = False
    server = HTTPServer(("127.0.0.1", 0), SliceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _url(server) -> str:
    return f"http://127.0.0.1:{server.server_port}/big.bin"


class TestDownloadRange:
    def test_middle_slice(self, slice_server, tmp_path):
        dest = tmp_path / "slice.bin"
        result = download_range(_url(slice_server), dest, 10, 29)
        assert result == dest
        assert dest.read_bytes() == PAYLOAD[10:30]

    def test_slice_to_end(self, slice_server, tmp_path):
        dest = tmp_path / "tail.bin"
        download_range(_url(slice_server), dest, 1000, 1023)
        assert dest.read_bytes() == PAYLOAD[1000:1024]

    def test_single_byte(self, slice_server, tmp_path):
        dest = tmp_path / "one.bin"
        download_range(_url(slice_server), dest, 500, 500)
        assert dest.read_bytes() == PAYLOAD[500:501]

    def test_on_chunk_reports_slice_progress(self, slice_server, tmp_path):
        seen = []
        download_range(
            _url(slice_server),
            tmp_path / "s.bin",
            100,
            199,
            on_chunk=lambda done, total: seen.append((done, total)),
        )
        assert seen, "on_chunk was never called"
        assert seen[-1] == (100, 100)

    def test_slice_checksum_verified(self, slice_server, tmp_path):
        digest = hashlib.sha256(PAYLOAD[50:75]).hexdigest()
        dest = tmp_path / "verified.bin"
        download_range(_url(slice_server), dest, 50, 74, checksum=digest)
        assert dest.read_bytes() == PAYLOAD[50:75]

    def test_checksum_mismatch_errors(self, slice_server, tmp_path):
        with pytest.raises(DownloadError):
            download_range(_url(slice_server), tmp_path / "c.bin", 10, 29, checksum="0" * 64)

    def test_invalid_range_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            download_range("http://127.0.0.1:1/x", tmp_path / "a.bin", 30, 10)
        with pytest.raises(ValueError):
            download_range("http://127.0.0.1:1/x", tmp_path / "a.bin", -1, 10)

    def test_server_ignoring_range_errors(self, slice_server, tmp_path):
        # Server answers 200 with the full body instead of 206 — must error,
        # never silently download the whole file.
        SliceHandler.ignore_range = True
        with pytest.raises(DownloadError):
            download_range(_url(slice_server), tmp_path / "s.bin", 10, 29)
        assert not (tmp_path / "s.bin").exists()

    def test_missing_file_errors(self, slice_server, tmp_path):
        with pytest.raises(DownloadError):
            download_range(
                f"http://127.0.0.1:{slice_server.server_port}/missing.bin",
                tmp_path / "m.bin",
                5,
                15,
            )
