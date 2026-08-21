"""Tests for the /rate-limit router (status + check)."""

from test_support import get_test_client


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


class TestRateLimitStatus:
    def test_status_returns_config(self):
        client = get_test_client()
        resp = client.get("/rate-limit/status")
        assert resp.status_code == 200
        data = _data(resp)
        assert "requests_per_minute" in data
        assert "burst_size" in data
        assert "enabled" in data
        assert isinstance(data["requests_per_minute"], int)
        assert isinstance(data["burst_size"], int)
        assert data["enabled"] is True

    def test_status_defaults(self):
        client = get_test_client()
        resp = client.get("/rate-limit/status")
        data = _data(resp)
        assert data["requests_per_minute"] == 60
        assert data["burst_size"] == 10


class TestRateLimitCheck:
    def test_check_allowed(self):
        client = get_test_client()
        resp = client.get("/rate-limit/check")
        assert resp.status_code == 200
        data = _data(resp)
        assert "allowed" in data
        assert "wait_time" in data
        assert isinstance(data["allowed"], bool)

    def test_check_first_request_allowed(self):
        client = get_test_client()
        resp = client.get("/rate-limit/check")
        data = _data(resp)
        assert data["allowed"] is True
        assert data["wait_time"] == 0

    def test_rate_limiting_kicks_in(self):
        client = get_test_client()
        for _ in range(65):
            client.get("/rate-limit/check")
        resp = client.get("/rate-limit/check")
        data = _data(resp)
        assert data["allowed"] is False
        assert data["wait_time"] > 0
