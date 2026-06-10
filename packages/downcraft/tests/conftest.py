"""pytest fixtures for downcraft tests."""

import json
import os
import tempfile
from pathlib import Path

import pytest


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
