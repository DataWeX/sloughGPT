"""Tests for benchmark router endpoints, especially history clear."""

from test_support import get_test_client


def test_clear_history_endpoint():
    """POST /benchmark/history/clear returns ok."""
    client = get_test_client()
    resp = client.post("/benchmark/history/clear")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["cleared"] is True
