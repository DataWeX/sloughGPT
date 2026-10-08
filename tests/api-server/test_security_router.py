"""Tests for the /security router (audit logs + API key info)."""

from test_support import get_test_client


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


class TestAuditLogs:
    def test_get_audit_logs_empty(self):
        client = get_test_client()
        resp = client.get("/security/audit")
        assert resp.status_code == 200
        data = _data(resp)
        assert "logs" in data
        assert "count" in data
        assert isinstance(data["logs"], list)

    def test_audit_logs_limit_parameter(self):
        client = get_test_client()
        resp = client.get("/security/audit?limit=10")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["count"] <= 10

    def test_audit_logs_limit_min(self):
        client = get_test_client()
        resp = client.get("/security/audit?limit=0")
        assert resp.status_code == 422  # validation error: ge=1

    def test_audit_logs_limit_max(self):
        client = get_test_client()
        resp = client.get("/security/audit?limit=10000")
        assert resp.status_code == 200

    def test_audit_logs_limit_over_max(self):
        client = get_test_client()
        resp = client.get("/security/audit?limit=99999")
        assert resp.status_code == 422

    def test_audit_logs_event_type_filter(self):
        client = get_test_client()
        resp = client.get("/security/audit?event_type=nonexistent")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["count"] == 0

    def test_audit_logs_history_mode(self):
        client = get_test_client()
        resp = client.get("/security/audit?history=true")
        assert resp.status_code == 200
        data = _data(resp)
        assert "logs" in data

    def test_audit_logs_before_cursor(self):
        client = get_test_client()
        resp = client.get("/security/audit?before=2026-01-01T00:00:00")
        assert resp.status_code == 200


class TestAPIKeys:
    def test_get_keys(self):
        client = get_test_client()
        resp = client.get("/security/keys")
        assert resp.status_code == 200
        data = _data(resp)
        assert "count" in data
        assert "configured" in data
        assert isinstance(data["count"], int)
        assert isinstance(data["configured"], bool)

    def test_keys_configured_matches_count(self):
        client = get_test_client()
        resp = client.get("/security/keys")
        data = _data(resp)
        assert data["configured"] == (data["count"] > 0)
