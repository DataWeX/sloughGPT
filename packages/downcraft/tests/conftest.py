"""pytest fixtures for downcraft tests.

Provides a real local HTTP server with ``Range`` support (stdlib only —
no pytest-httpserver dependency) plus per-test isolation of the
persistent download state and retry settings.
"""

import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Real local HTTP server with Range support
# ---------------------------------------------------------------------------

class RangeHandler(BaseHTTPRequestHandler):
    """Serves per-path payloads with HTTP Range support.

    Class attributes (set per test):
        payloads: ``{path: bytes}`` — any other path returns 404.
    """

    payloads: dict = {}

    def _payload_for(self):
        return self.payloads.get(self.path.split("?")[0])

    def do_GET(self):
        payload = self._payload_for()
        if payload is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes="):].split("-")[0]
            if spec.isdigit():
                start = int(spec)
        data = payload[start:]
        if start > 0:
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(payload) - 1}/{len(payload)}",
            )
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("ETag", '"static"')
        self.end_headers()
        for i in range(0, len(data), 2048):
            self.wfile.write(data[i:i + 2048])

    def log_message(self, *args):
        pass


def _range_url(server, path: str) -> str:
    """Build a URL for a path on the running range server."""
    return f"http://127.0.0.1:{server.server_port}{path}"


@pytest.fixture
def range_server():
    """A local HTTP server serving per-path payloads with Range support.

    Set ``RangeHandler.payloads[path] = bytes`` inside the test to choose
    what each URL path returns; any unregistered path yields a 404.
    """
    RangeHandler.payloads = {}
    server = HTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Per-test isolation of state + retry settings
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch, tmp_path):
    """Give each test a private download state so tests never share/collide."""
    from downcraft import state as state_mod
    monkeypatch.setattr(
        state_mod,
        "get_state",
        lambda: state_mod.PersistentState(state_dir=tmp_path / "state"),
    )


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """Keep failure-path tests fast: never wait real backoff."""
    from downcraft import downloader as downloader_mod
    monkeypatch.setattr(downloader_mod, "MAX_RETRIES", 1)


# ---------------------------------------------------------------------------
# Legacy fixtures (kept for compatibility)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state_dir():
    """Create a temporary state directory and set up a clean PersistentState."""
    with tempfile.TemporaryDirectory() as td:
        old_home = Path.home()
        # Trick: we can't easily change Path.home(), so we'll just pass state_dir
        # directly in tests
        yield Path(td)


@pytest.fixture
def sample_state_data():
    """Sample state JSON data for testing deserialization."""
    return {
        "models": {
            "test-model": {
                "status": "downloading",
                "files": [
                    {
                        "path": "model.safetensors",
                        "url": "https://example.com/model.safetensors",
                        "bytes_downloaded": 500,
                        "total_bytes": 1000,
                        "checksum": "abc123",
                        "complete": False,
                    }
                ],
                "started_at": 1000.0,
                "completed_at": None,
                "error": "",
                "cache_dir": "/tmp/cache",
            }
        },
        "updated_at": 2000.0,
    }


@pytest.fixture
def sample_file(tmp_path):
    """Create a small sample file with known content."""
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello world this is test content for sha256 verification")
    return f
