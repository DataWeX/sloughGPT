"""
Tests for the errors router — log, recent, grouped, trends, export, clear, unread.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.errors import ErrorsRouter


@pytest.fixture
def router_instance():
    return ErrorsRouter()


@pytest.fixture
def app(router_instance):
    _app = FastAPI()
    _app.include_router(router_instance.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestLogErrors:
    """POST /errors/log"""

    def test_logs_single_error(self, client):
        resp = client.post("/errors/log", json={
            "errors": [{"message": "test error", "source": "web"}],
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "ok"
        assert data["logged"] == 1

    def test_logs_multiple_errors(self, client):
        resp = client.post("/errors/log", json={
            "errors": [
                {"message": "err1"},
                {"message": "err2"},
                {"message": "err3"},
            ],
        })
        assert resp.json()["data"]["logged"] == 3

    def test_logs_error_with_metadata(self, client):
        resp = client.post("/errors/log", json={
            "errors": [{
                "message": "auth failed",
                "source": "web",
                "url": "https://example.com/page",
                "line": 42,
                "col": 10,
                "metadata": {"user_agent": "test"},
            }],
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["logged"] == 1

    def test_log_empty_message(self, client):
        resp = client.post("/errors/log", json={
            "errors": [{"message": "", "source": "web"}],
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["logged"] == 1


class TestRecentErrors:
    """GET /errors/recent"""

    def test_returns_recent_errors(self, client, router_instance):
        router_instance._error_buffer.clear()
        router_instance._error_buffer.append({"id": "e1", "message": "test", "source": "web", "fingerprint": "fp1"})
        resp = client.get("/errors/recent")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "errors" in data
        assert "unread_count" in data
        assert "total" in data

    def test_pagination(self, client, router_instance):
        router_instance._error_buffer.clear()
        for i in range(10):
            router_instance._error_buffer.append({
                "id": f"e{i}", "message": f"err{i}", "source": "web", "fingerprint": f"fp{i}",
            })
        resp = client.get("/errors/recent?limit=3&offset=0")
        assert len(resp.json()["data"]["errors"]) == 3

    def test_recent_empty_buffer(self, client, router_instance):
        router_instance._error_buffer.clear()
        resp = client.get("/errors/recent")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["errors"] == []
        assert data["total"] == 0

    def test_recent_with_large_limit(self, client, router_instance):
        router_instance._error_buffer.clear()
        for i in range(5):
            router_instance._error_buffer.append({
                "id": f"e{i}", "message": f"err{i}", "source": "web", "fingerprint": f"fp{i}",
            })
        resp = client.get("/errors/recent?limit=1000")
        assert resp.status_code == 200
        assert len(resp.json()["data"]["errors"]) == 5

    def test_recent_offset(self, client, router_instance):
        router_instance._error_buffer.clear()
        for i in range(5):
            router_instance._error_buffer.append({
                "id": f"e{i}", "message": f"err{i}", "source": "web", "fingerprint": f"fp{i}",
            })
        resp = client.get("/errors/recent?limit=2&offset=2")
        data = resp.json()["data"]
        assert len(data["errors"]) == 2


class TestGroupedErrors:
    """GET /errors/grouped"""

    def test_groups_by_fingerprint(self, client, router_instance):
        router_instance._error_buffer.clear()
        for _ in range(3):
            router_instance._error_buffer.append({
                "id": "e1", "message": "TypeError: x is undefined", "source": "web", "fingerprint": "fp_abc",
            })
        router_instance._error_buffer.append({
            "id": "e2", "message": "Other error", "source": "web", "fingerprint": "fp_xyz",
        })
        resp = client.get("/errors/grouped")
        assert resp.status_code == 200
        groups = resp.json()["data"]["groups"]
        counts = {g["fingerprint"]: g["count"] for g in groups}
        assert counts.get("fp_abc") == 3
        assert counts.get("fp_xyz") == 1

    def test_grouped_empty(self, client, router_instance):
        router_instance._error_buffer.clear()
        resp = client.get("/errors/grouped")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["groups"] == []
        assert data["total_groups"] == 0


class TestErrorTrends:
    """GET /errors/trends"""

    def test_returns_trends(self, client):
        resp = client.get("/errors/trends?hours=4")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "trends" in data
        assert len(data["trends"]) == 4

    def test_trends_default_hours(self, client):
        resp = client.get("/errors/trends")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["trends"]) == 24

    def test_trends_count_accumulates(self, client, router_instance):
        router_instance._error_buffer.clear()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for _ in range(5):
            router_instance._error_buffer.append({
                "id": "e1", "message": "err", "source": "web",
                "fingerprint": "fp", "timestamp": now,
            })
        resp = client.get("/errors/trends?hours=1")
        data = resp.json()["data"]
        total_count = sum(t["count"] for t in data["trends"])
        assert total_count >= 5


class TestExportErrors:
    """GET /errors/export"""

    def test_returns_errors_as_json(self, client):
        resp = client.get("/errors/export")
        assert resp.status_code == 200
        assert "errors" in resp.json()

    def test_export_includes_metadata(self, client, router_instance):
        router_instance._error_buffer.clear()
        router_instance._error_buffer.append({
            "id": "e1", "message": "test", "source": "web",
            "fingerprint": "fp", "url": "https://example.com",
        })
        resp = client.get("/errors/export")
        body = resp.json()
        assert body["total"] >= 1
        assert body["exported"] >= 1


class TestClearErrors:
    """DELETE /errors/clear"""

    def test_clears_error_buffer(self, client, router_instance):
        router_instance._error_buffer.clear()
        router_instance._error_buffer.append({"id": "e1", "message": "test", "source": "web", "fingerprint": "fp"})
        resp = client.delete("/errors/clear")
        assert resp.status_code == 200
        assert resp.json()["data"]["cleared"] is True
        assert len(router_instance._error_buffer) == 0

    def test_clear_resets_unread_count(self, client, router_instance):
        router_instance._error_buffer.clear()
        router_instance._error_count_since_clear = 10
        resp = client.delete("/errors/clear")
        assert resp.status_code == 200
        assert router_instance._error_count_since_clear == 0

    def test_clear_already_empty(self, client, router_instance):
        router_instance._error_buffer.clear()
        resp = client.delete("/errors/clear")
        assert resp.status_code == 200
        assert resp.json()["data"]["cleared"] is True


class TestUnreadCount:
    """GET /errors/unread"""

    def test_returns_unread_count(self, client, router_instance):
        router_instance._error_buffer.clear()
        router_instance._error_count_since_clear = 5
        resp = client.get("/errors/unread")
        assert resp.status_code == 200
        assert resp.json()["data"]["unread_count"] == 5

    def test_unread_zero_after_clear(self, client, router_instance):
        router_instance._error_buffer.clear()
        router_instance._error_count_since_clear = 0
        resp = client.get("/errors/unread")
        assert resp.json()["data"]["unread_count"] == 0


class TestIngestFrontendLogs:
    """POST /errors/logs/ingest"""

    @patch("domains.infrastructure.output_buffer.get_server_buffer")
    def test_ingests_single_log(self, mock_get_buf, client):
        resp = client.post("/errors/logs/ingest", json={
            "logs": [{"level": "info", "logger": "chat", "message": "hello"}],
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["ingested"] == 1

    @patch("domains.infrastructure.output_buffer.get_server_buffer")
    def test_ingests_multiple_logs(self, mock_get_buf, client):
        resp = client.post("/errors/logs/ingest", json={
            "logs": [
                {"level": "debug", "message": "a"},
                {"level": "error", "message": "b"},
                {"level": "critical", "message": "c"},
            ],
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["ingested"] == 3

    @patch("logging.getLogger")
    def test_ingest_maps_levels(self, mock_get_logger, client):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        resp = client.post("/errors/logs/ingest", json={
            "logs": [{"level": "warning", "logger": "models", "message": "warn"}],
        })
        assert resp.status_code == 200
        mock_logger.log.assert_called_once()
        args, kwargs = mock_logger.log.call_args
        assert args[0] == logging.WARNING
        assert args[1] == "warn"
        assert kwargs["extra"]["source"] == "web.models"

    @patch("logging.getLogger")
    def test_ingest_unknown_level_defaults_info(self, mock_get_logger, client):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        resp = client.post("/errors/logs/ingest", json={
            "logs": [{"level": "bogus", "message": "x"}],
        })
        assert resp.status_code == 200
        mock_logger.log.assert_called_once()
        args, _ = mock_logger.log.call_args
        assert args[0] == logging.INFO

    @patch("logging.getLogger")
    def test_ingest_with_exception_context(self, mock_get_logger, client):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        resp = client.post("/errors/logs/ingest", json={
            "logs": [{
                "level": "error",
                "logger": "store",
                "message": "boom",
                "exception": "TypeError: x",
                "context": {"page": "/settings"},
            }],
        })
        assert resp.status_code == 200
        _, kwargs = mock_logger.log.call_args
        ctx = kwargs["extra"]["context"]
        assert ctx["page"] == "/settings"
        assert ctx["exception"] == "TypeError: x"

    @patch("logging.getLogger")
    def test_ingest_empty_batch(self, mock_get_logger, client):
        resp = client.post("/errors/logs/ingest", json={"logs": []})
        assert resp.status_code == 200
        assert resp.json()["data"]["ingested"] == 0


class TestOpencodeLog:
    """GET /errors/log — opencode error log."""

    def test_returns_entries(self, client):
        resp = client.get("/errors/log")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "entries" in data
        assert "total" in data


class TestErrorsValidation:
    """Validation bounds and method mismatches."""

    def test_log_message_too_long_422(self, client):
        resp = client.post("/errors/log", json={
            "errors": [{"message": "x" * 5001}],
        })
        assert resp.status_code == 422

    def test_log_too_many_errors_422(self, client):
        resp = client.post("/errors/log", json={
            "errors": [{"message": f"e{i}"} for i in range(101)],
        })
        assert resp.status_code == 422

    def test_log_missing_errors_field_422(self, client):
        resp = client.post("/errors/log", json={})
        assert resp.status_code == 422

    def test_ingest_missing_logs_field_422(self, client):
        resp = client.post("/errors/logs/ingest", json={})
        assert resp.status_code == 422

    def test_recent_wrong_method_405(self, client):
        resp = client.post("/errors/recent")
        assert resp.status_code == 405

    def test_grouped_wrong_method_405(self, client):
        resp = client.post("/errors/grouped")
        assert resp.status_code == 405

    def test_trends_wrong_method_405(self, client):
        resp = client.put("/errors/trends")
        assert resp.status_code == 405

    def test_export_wrong_method_405(self, client):
        resp = client.post("/errors/export")
        assert resp.status_code == 405

    def test_unread_wrong_method_405(self, client):
        resp = client.post("/errors/unread")
        assert resp.status_code == 405

    def test_log_wrong_method_405(self, client):
        resp = client.delete("/errors/log")
        assert resp.status_code == 405

    def test_ingest_wrong_method_405(self, client):
        resp = client.get("/errors/logs/ingest")
        assert resp.status_code == 405

    def test_clear_wrong_method_405(self, client):
        resp = client.post("/errors/clear")
        assert resp.status_code == 405
