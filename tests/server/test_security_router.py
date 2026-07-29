"""
Tests for the security router — GET /security/audit and GET /security/keys.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.security import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestSecurityAudit:
    """GET /security/audit"""

    @patch("infrastructure.auth.get_audit_logger")
    def test_returns_audit_logs(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "auth_success", "timestamp": "2024-01-01T00:00:00"}]
        resp = client.get("/security/audit")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 1
        assert data["logs"][0]["event_type"] == "auth_success"

    @patch("infrastructure.auth.get_audit_logger")
    def test_filters_by_event_type(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [
            {"event_type": "auth_success", "timestamp": "1"},
            {"event_type": "auth_failed", "timestamp": "2"},
        ]
        resp = client.get("/security/audit?event_type=auth_failed")
        assert resp.json()["data"]["count"] == 1
        assert resp.json()["data"]["logs"][0]["event_type"] == "auth_failed"


class TestSecurityKeys:
    """GET /security/keys"""

    @patch("settings.get_security_settings")
    def test_returns_key_info(self, mock_get_sec, client):
        sec = mock_get_sec.return_value
        sec.valid_api_keys = ["key1", "key2"]
        resp = client.get("/security/keys")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 2
        assert data["configured"] is True
