"""Tests for the /benchmark router — metrics, quality, responses, stats, clear."""

from test_support import _data, get_test_client


def test_clear_history_endpoint():
    client = get_test_client()
    resp = client.post("/benchmark/history/clear")
    assert resp.status_code == 200
    data = _data(resp)
    assert data["status"] == "ok"
    assert data["cleared"] is True


def test_model_metrics_no_model_loaded():
    client = get_test_client()
    resp = client.get("/benchmark/metrics", params={"model": "gpt2"})
    assert resp.status_code == 200
    body = _data(resp)
    assert "model" in body
    assert body["model"] == "gpt2"


def test_tracker_stats():
    client = get_test_client()
    resp = client.get("/benchmark/stats")
    assert resp.status_code == 200
    body = _data(resp)
    assert isinstance(body, dict)


def test_quality_metrics():
    client = get_test_client()
    resp = client.get("/benchmark/quality")
    assert resp.status_code == 200
    body = _data(resp)
    assert isinstance(body, dict)


def test_logged_responses():
    client = get_test_client()
    resp = client.get("/benchmark/responses")
    assert resp.status_code == 200
    body = _data(resp)
    assert "responses" in body
    assert "count" in body
    assert isinstance(body["responses"], list)


def test_logged_responses_limit():
    client = get_test_client()
    resp = client.get("/benchmark/responses", params={"limit": 5})
    assert resp.status_code == 200
    body = _data(resp)
    assert len(body["responses"]) <= 5


def test_benchmark_by_nonexistent_id():
    client = get_test_client()
    resp = client.get("/benchmark/nonexistent_id_123")
    assert resp.status_code in (200, 404)
