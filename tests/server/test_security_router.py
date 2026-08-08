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

    @patch("infrastructure.auth.get_audit_logger")
    def test_empty_logs(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = []
        resp = client.get("/security/audit")
        assert resp.json()["data"]["count"] == 0
        assert resp.json()["data"]["logs"] == []

    @patch("infrastructure.auth.get_audit_logger")
    def test_limit_parameter(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "a", "timestamp": str(i)} for i in range(20)]
        resp = client.get("/security/audit?limit=5")
        assert resp.json()["data"]["count"] == 5

    @patch("infrastructure.auth.get_audit_logger")
    def test_no_matching_event_type(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "auth_success", "timestamp": "1"}]
        resp = client.get("/security/audit?event_type=nonexistent")
        assert resp.json()["data"]["count"] == 0

    @patch("infrastructure.auth.get_audit_logger")
    def test_limit_one(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "a", "timestamp": "1"}, {"event_type": "b", "timestamp": "2"}]
        resp = client.get("/security/audit?limit=1")
        assert resp.json()["data"]["count"] == 1

    @patch("infrastructure.auth.get_audit_logger")
    def test_limit_larger_than_logs(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "a", "timestamp": "1"}]
        resp = client.get("/security/audit?limit=100")
        assert resp.json()["data"]["count"] == 1

    @patch("infrastructure.auth.get_audit_logger")
    def test_multiple_event_types(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [
            {"event_type": "auth_success", "timestamp": "1"},
            {"event_type": "auth_failed", "timestamp": "2"},
            {"event_type": "auth_success", "timestamp": "3"},
        ]
        resp = client.get("/security/audit?event_type=auth_success")
        assert resp.json()["data"]["count"] == 2

    @patch("infrastructure.auth.get_audit_logger")
    def test_limit_zero_returns_all(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "a", "timestamp": str(i)} for i in range(4)]
        resp = client.get("/security/audit?limit=0")
        assert resp.json()["data"]["count"] == 4

    @patch("infrastructure.auth.get_audit_logger")
    def test_combined_limit_and_filter(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [
            {"event_type": "auth_success", "timestamp": "1"},
            {"event_type": "auth_failed", "timestamp": "2"},
            {"event_type": "auth_success", "timestamp": "3"},
            {"event_type": "auth_failed", "timestamp": "4"},
        ]
        resp = client.get("/security/audit?limit=1&event_type=auth_failed")
        data = resp.json()["data"]
        assert data["count"] == 1
        assert data["logs"][0]["timestamp"] == "4"

    def test_invalid_limit_rejected(self, client):
        resp = client.get("/security/audit?limit=not-a-number")
        assert resp.status_code == 422

    @patch("infrastructure.auth.get_audit_logger")
    def test_negative_limit_slices_from_end(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "a", "timestamp": str(i)} for i in range(6)]
        resp = client.get("/security/audit?limit=-2")
        assert resp.json()["data"]["count"] == 4

    def test_wrong_method_returns_405(self, client):
        resp = client.post("/security/audit")
        assert resp.status_code == 405

    @patch("infrastructure.auth.get_audit_logger")
    def test_logs_with_missing_event_type_excluded_when_filtering(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [
            {"timestamp": "1"},
            {"event_type": "auth_success", "timestamp": "2"},
        ]
        resp = client.get("/security/audit?event_type=auth_success")
        data = resp.json()["data"]
        assert data["count"] == 1
        assert data["logs"][0]["timestamp"] == "2"

    @patch("infrastructure.auth.get_audit_logger")
    def test_extra_fields_passthrough(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "model_loaded", "timestamp": "1", "model_id": "gpt2", "tag": "REQ"}]
        resp = client.get("/security/audit")
        log = resp.json()["data"]["logs"][0]
        assert log["model_id"] == "gpt2"
        assert log["tag"] == "REQ"

    def test_audit_logger_error_returns_500(self, client):
        with patch("infrastructure.auth.get_audit_logger", side_effect=RuntimeError("broken")):
            resp = client.get("/security/audit")
        assert resp.status_code == 500

    @patch("infrastructure.auth.get_audit_logger")
    def test_audit_data_keys(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "auth_success", "timestamp": "1"}]
        resp = client.get("/security/audit")
        assert set(resp.json()["data"].keys()) == {"logs", "count"}

    @patch("infrastructure.auth.get_audit_logger")
    def test_large_limit(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "a", "timestamp": str(i)} for i in range(200)]
        resp = client.get("/security/audit?limit=150")
        assert resp.json()["data"]["count"] == 150


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

    @patch("settings.get_security_settings")
    def test_no_keys_configured(self, mock_get_sec, client):
        sec = mock_get_sec.return_value
        sec.valid_api_keys = []
        resp = client.get("/security/keys")
        data = resp.json()["data"]
        assert data["count"] == 0
        assert data["configured"] is False

    @patch("settings.get_security_settings")
    def test_single_key(self, mock_get_sec, client):
        sec = mock_get_sec.return_value
        sec.valid_api_keys = ["only-one"]
        resp = client.get("/security/keys")
        assert resp.json()["data"]["count"] == 1

    @patch("settings.get_security_settings")
    def test_keys_structure(self, mock_get_sec, client):
        sec = mock_get_sec.return_value
        sec.valid_api_keys = ["k1"]
        resp = client.get("/security/keys")
        data = resp.json()["data"]
        assert "count" in data
        assert "configured" in data

    def test_wrong_method_returns_405(self, client):
        resp = client.post("/security/keys")
        assert resp.status_code == 405

    def test_keys_error_returns_500(self, client):
        with patch("settings.get_security_settings", side_effect=RuntimeError("broken")):
            resp = client.get("/security/keys")
        assert resp.status_code == 500

    @patch("settings.get_security_settings")
    def test_keys_exact_data_keys(self, mock_get_sec, client):
        sec = mock_get_sec.return_value
        sec.valid_api_keys = ["k1", "k2", "k3"]
        resp = client.get("/security/keys")
        assert set(resp.json()["data"].keys()) == {"count", "configured"}
