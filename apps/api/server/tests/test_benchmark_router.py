"""
Tests for benchmark router endpoints, especially history clear.
"""

from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_clear_history_endpoint():
    """POST /benchmark/history/clear returns ok."""
    from main import app
    client = TestClient(app)
    resp = client.post("/benchmark/history/clear")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["cleared"] is True
