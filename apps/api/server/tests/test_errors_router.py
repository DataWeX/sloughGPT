"""Tests for errors router endpoints."""
import pytest

from tests.test_support import get_test_client
from routers.errors import _error_buffer, _error_count_since_clear, _dedup_map, clear_errors

client = get_test_client()


@pytest.fixture(autouse=True)
def _clear_error_state():
    """Clear error buffer and dedup map before and after each test."""
    _error_buffer.clear()
    _dedup_map.clear()
    yield
    _error_buffer.clear()
    _dedup_map.clear()


class TestLogErrors:
    def test_log_single_error(self):
        resp = client.post("/errors/log", json={
            "errors": [{"message": "test error", "source": "web"}]
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["logged"] == 1

    def test_log_multiple_errors(self):
        resp = client.post("/errors/log", json={
            "errors": [
                {"message": "error 1"},
                {"message": "error 2"},
                {"message": "error 3"},
            ]
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["logged"] == 3

    def test_log_error_with_metadata(self):
        resp = client.post("/errors/log", json={
            "errors": [{
                "message": "detailed error",
                "url": "https://example.com/page",
                "line": 42,
                "col": 10,
                "stack": "Error at line 42",
                "metadata": {"component": "ChatInput"},
            }]
        })
        assert resp.status_code == 200

    def test_log_empty_batch(self):
        resp = client.post("/errors/log", json={"errors": []})
        assert resp.status_code == 200
        assert resp.json()["data"]["logged"] == 0


class TestRecentErrors:
    def test_recent_empty(self):
        resp = client.get("/errors/recent")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["errors"] == []
        assert body["data"]["total"] == 0

    def test_recent_after_logging(self):
        client.post("/errors/log", json={"errors": [{"message": "alpha error"}]})
        client.post("/errors/log", json={"errors": [{"message": "bravo error"}]})
        resp = client.get("/errors/recent")
        body = resp.json()
        assert body["data"]["total"] == 2
        assert len(body["data"]["errors"]) == 2

    def test_recent_pagination(self):
        msgs = ["alpha", "bravo", "charlie", "delta", "echo"]
        for msg in msgs:
            client.post("/errors/log", json={"errors": [{"message": msg}]})
        resp = client.get("/errors/recent?limit=2&offset=0")
        body = resp.json()
        assert len(body["data"]["errors"]) == 2

    def test_recent_returns_newest_first(self):
        client.post("/errors/log", json={"errors": [{"message": "first"}]})
        client.post("/errors/log", json={"errors": [{"message": "second"}]})
        resp = client.get("/errors/recent")
        errors = resp.json()["data"]["errors"]
        assert errors[0]["message"] == "second"
        assert errors[1]["message"] == "first"


class TestGroupedErrors:
    def test_grouped_empty(self):
        resp = client.get("/errors/grouped")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["groups"] == []
        assert body["data"]["total_groups"] == 0

    def test_grouped_deduplicates(self):
        for _ in range(3):
            client.post("/errors/log", json={
                "errors": [{"message": "Same error message"}]
            })
        resp = client.get("/errors/grouped")
        groups = resp.json()["data"]["groups"]
        assert len(groups) == 1
        assert groups[0]["count"] == 3


class TestErrorTrends:
    def test_trends_returns_hours(self):
        resp = client.get("/errors/trends")
        assert resp.status_code == 200
        body = resp.json()
        assert "trends" in body["data"]
        assert len(body["data"]["trends"]) == 24

    def test_trends_custom_hours(self):
        resp = client.get("/errors/trends?hours=6")
        assert len(resp.json()["data"]["trends"]) == 6


class TestExportErrors:
    def test_export_empty(self):
        resp = client.get("/errors/export")
        assert resp.status_code == 200
        body = resp.json()
        assert body["errors"] == []
        assert body["total"] == 0

    def test_export_with_errors(self):
        client.post("/errors/log", json={"errors": [{"message": "export me"}]})
        resp = client.get("/errors/export")
        body = resp.json()
        assert len(body["errors"]) == 1
        assert body["exported"] == 1


class TestClearErrors:
    def test_clear_resets_buffer(self):
        client.post("/errors/log", json={"errors": [{"message": "to clear"}]})
        resp = client.delete("/errors/clear")
        assert resp.status_code == 200
        assert resp.json()["data"]["cleared"] is True
        resp2 = client.get("/errors/recent")
        assert resp2.json()["data"]["total"] == 0

    def test_clear_resets_unread_count(self):
        client.post("/errors/log", json={"errors": [{"message": "err"}]})
        client.delete("/errors/clear")
        resp = client.get("/errors/unread")
        assert resp.json()["data"]["unread_count"] == 0


class TestUnreadCount:
    def test_unread_after_log(self):
        client.post("/errors/log", json={"errors": [
            {"message": "a"}, {"message": "b"}
        ]})
        resp = client.get("/errors/unread")
        assert resp.json()["data"]["unread_count"] == 2


class TestFrontendLogIngest:
    def test_ingest_logs(self):
        resp = client.post("/errors/logs/ingest", json={
            "logs": [
                {"level": "error", "logger": "web", "message": "frontend error"},
                {"level": "info", "logger": "chat", "message": "chat info"},
            ]
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["ingested"] == 2


class TestOpenCodeLog:
    def test_get_opencode_log(self):
        resp = client.get("/errors/log")
        assert resp.status_code == 200
        body = resp.json()
        data = body.get("data", body)
        assert "entries" in data
        assert isinstance(data["entries"], list)
