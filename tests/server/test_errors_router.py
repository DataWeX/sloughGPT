"""
Tests for the errors router — log, recent, grouped, trends, export, clear, unread.
"""

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


class TestErrorTrends:
    """GET /errors/trends"""

    def test_returns_trends(self, client):
        resp = client.get("/errors/trends?hours=4")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "trends" in data
        assert len(data["trends"]) == 4


class TestExportErrors:
    """GET /errors/export"""

    def test_returns_errors_as_json(self, client):
        resp = client.get("/errors/export")
        assert resp.status_code == 200
        assert "errors" in resp.json()


class TestClearErrors:
    """DELETE /errors/clear"""

    def test_clears_error_buffer(self, client, router_instance):
        router_instance._error_buffer.clear()
        router_instance._error_buffer.append({"id": "e1", "message": "test", "source": "web", "fingerprint": "fp"})
        resp = client.delete("/errors/clear")
        assert resp.status_code == 200
        assert resp.json()["data"]["cleared"] is True
        assert len(router_instance._error_buffer) == 0


class TestUnreadCount:
    """GET /errors/unread"""

    def test_returns_unread_count(self, client, router_instance):
        router_instance._error_buffer.clear()
        router_instance._error_count_since_clear = 5
        resp = client.get("/errors/unread")
        assert resp.status_code == 200
        assert resp.json()["data"]["unread_count"] == 5
