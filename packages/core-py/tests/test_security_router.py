"""Tests for the security API router (routers/security.py).

Covers: get_audit_logs (memory + file history), get_keys, edge cases.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.security import SecurityRouter  # noqa: E402


def _app(sr: SecurityRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(sr.router)
    return app


class TestAuditLogs:
    def test_memory_logs_default(self):
        sr = SecurityRouter()
        mock_audit = MagicMock()
        mock_audit.logs = [
            {"event_type": "login", "ts": 1},
            {"event_type": "logout", "ts": 2},
        ]
        with patch("infrastructure.auth.get_audit_logger", return_value=mock_audit):
            client = TestClient(_app(sr))
            resp = client.get("/security/audit")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 2

    def test_memory_logs_filter_event_type(self):
        sr = SecurityRouter()
        mock_audit = MagicMock()
        mock_audit.logs = [
            {"event_type": "login", "ts": 1},
            {"event_type": "logout", "ts": 2},
        ]
        with patch("infrastructure.auth.get_audit_logger", return_value=mock_audit):
            client = TestClient(_app(sr))
            resp = client.get("/security/audit?event_type=login")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1

    def test_file_history(self):
        sr = SecurityRouter()
        mock_audit = MagicMock()
        mock_audit.file_query.return_value = [{"event_type": "login"}]
        with patch("infrastructure.auth.get_audit_logger", return_value=mock_audit):
            client = TestClient(_app(sr))
            resp = client.get("/security/audit?history=true")
        assert resp.status_code == 200
        mock_audit.file_query.assert_called_once()

    def test_empty_logs(self):
        sr = SecurityRouter()
        mock_audit = MagicMock()
        mock_audit.logs = []
        with patch("infrastructure.auth.get_audit_logger", return_value=mock_audit):
            client = TestClient(_app(sr))
            resp = client.get("/security/audit")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 0

    def test_filter_no_match(self):
        sr = SecurityRouter()
        mock_audit = MagicMock()
        mock_audit.logs = [
            {"event_type": "login", "ts": 1},
        ]
        with patch("infrastructure.auth.get_audit_logger", return_value=mock_audit):
            client = TestClient(_app(sr))
            resp = client.get("/security/audit?event_type=nonexistent")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 0

    def test_logs_returned_in_response(self):
        sr = SecurityRouter()
        mock_audit = MagicMock()
        mock_audit.logs = [
            {"event_type": "login", "ts": 100, "user": "alice"},
            {"event_type": "logout", "ts": 200, "user": "bob"},
        ]
        with patch("infrastructure.auth.get_audit_logger", return_value=mock_audit):
            client = TestClient(_app(sr))
            resp = client.get("/security/audit")
        data = resp.json()["data"]
        assert len(data.get("logs", data.get("entries", []))) == 2


class TestKeys:
    def test_keys_configured(self):
        sr = SecurityRouter()
        mock_sec = MagicMock()
        mock_sec.valid_api_keys = ["key1", "key2"]
        with patch("settings.get_security_settings", return_value=mock_sec):
            client = TestClient(_app(sr))
            resp = client.get("/security/keys")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 2
        assert data["configured"] is True

    def test_keys_not_configured(self):
        sr = SecurityRouter()
        mock_sec = MagicMock()
        mock_sec.valid_api_keys = []
        with patch("settings.get_security_settings", return_value=mock_sec):
            client = TestClient(_app(sr))
            resp = client.get("/security/keys")
        assert resp.status_code == 200
        assert resp.json()["data"]["configured"] is False

    def test_single_key(self):
        sr = SecurityRouter()
        mock_sec = MagicMock()
        mock_sec.valid_api_keys = ["only-one-key"]
        with patch("settings.get_security_settings", return_value=mock_sec):
            client = TestClient(_app(sr))
            resp = client.get("/security/keys")
        data = resp.json()["data"]
        assert data["count"] == 1
        assert data["configured"] is True

    def test_many_keys(self):
        sr = SecurityRouter()
        mock_sec = MagicMock()
        mock_sec.valid_api_keys = [f"key_{i}" for i in range(50)]
        with patch("settings.get_security_settings", return_value=mock_sec):
            client = TestClient(_app(sr))
            resp = client.get("/security/keys")
        assert resp.json()["data"]["count"] == 50
