"""pytest fixtures for downcraft tests.

Provides a real local HTTP server with ``Range`` support (stdlib only —
no pytest-httpserver dependency) plus per-test isolation of the
persistent download state and retry settings.
"""

import sys
import tempfile
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

# Single-instance server helpers — see helpers.py for why these must not
# be defined in this file under --import-mode=importlib.
from helpers import RangeHandler, _range_url  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Local HTTP server fixture (handler implementation in helpers.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def range_server():
    """A local HTTP server serving per-path payloads with Range support.

    Set ``RangeHandler.payloads[path] = bytes`` inside the test to choose
    what each URL path returns; any unregistered path yields a 404.
    """
    RangeHandler.payloads = {}
    RangeHandler.content_types = {}
    RangeHandler.head_responses = {}
    RangeHandler.encodings = {}
    RangeHandler.lz4_paths = {}
    RangeHandler.lz4_naive = {}
    RangeHandler.lz4_bad_header = {}
    RangeHandler.lz4_bad_resume_sha = {}
    RangeHandler.truncate_once = {}
    RangeHandler.requests_log = []
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
    from downcraft.download import multipart as multipart_mod
    from downcraft.download import state as state_mod

    def _fresh() -> state_mod.PersistentState:
        return state_mod.PersistentState(state_dir=tmp_path / "state")

    monkeypatch.setattr(state_mod, "get_state", _fresh)
    # multipart does `from .state import get_state` — patch the bound name
    # too, otherwise download_parts() writes to the real ~/.downcraft.
    monkeypatch.setattr(multipart_mod, "get_state", _fresh)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """Keep failure-path tests fast: never wait real backoff."""
    from downcraft.download import http as http_mod

    monkeypatch.setattr(http_mod, "MAX_RETRIES", 1)


# ---------------------------------------------------------------------------
# Legacy fixtures (kept for compatibility)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_dir():
    """Create a temporary state directory and set up a clean PersistentState."""
    with tempfile.TemporaryDirectory() as td:
        Path.home()
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
