"""
Tests for the security router — GET /security/audit and GET /security/keys.
"""

import json
import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.security import router


@pytest.fixture
def app():
    _app = FastAPI()
    register_all_handlers(_app)
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

    @patch("infrastructure.auth.get_audit_logger")
    def test_ordering_newest_last(self, mock_get_logger, client):
        """logs[-limit:] keeps source order — last N entries, not reversed."""
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "a", "timestamp": str(i)} for i in range(5)]
        resp = client.get("/security/audit?limit=2")
        logs = resp.json()["data"]["logs"]
        assert [l["timestamp"] for l in logs] == ["3", "4"]

    @patch("infrastructure.auth.get_audit_logger")
    def test_filter_applied_after_slice(self, mock_get_logger, client):
        """event_type filters the sliced window, not the full list."""
        logger = mock_get_logger.return_value
        logs = [{"event_type": "a", "timestamp": str(i)} for i in range(10)]
        logs[0]["event_type"] = "target"
        logger.logs = logs
        resp = client.get("/security/audit?limit=3&event_type=target")
        assert resp.json()["data"]["count"] == 0

    @patch("infrastructure.auth.get_audit_logger")
    def test_limit_rejects_all_then_filters(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logs = [{"event_type": "auth_success", "timestamp": str(i)} for i in range(10)]
        logs[7]["event_type"] = "auth_failed"
        logger.logs = logs
        resp = client.get("/security/audit?limit=3&event_type=auth_failed")
        assert resp.json()["data"]["count"] == 1

    def test_wrong_methods_return_405(self, client):
        assert client.put("/security/audit").status_code == 405
        assert client.delete("/security/audit").status_code == 405
        assert client.patch("/security/audit").status_code == 405


class TestAuditLoggerReal:
    """Real AuditLogger — logs exposed via .logs, no mocks."""

    def test_log_appends_record_with_event_type(self, tmp_path):
        from infrastructure.auth import AuditLogger
        logger = AuditLogger(log_path=str(tmp_path / "audit.log"))
        logger.log("auth_success", user="u1", resource="/auth/token", detail="ok", extra={"action": "token_create"})
        assert len(logger.logs) == 1
        rec = logger.logs[0]
        assert rec["event_type"] == "auth_success"
        assert rec["user"] == "u1"
        assert rec["resource"] == "/auth/token"
        assert rec["detail"] == "ok"
        assert rec["extra"] == {"action": "token_create"}
        assert "timestamp" in rec

    def test_logs_property_returns_copy(self, tmp_path):
        from infrastructure.auth import AuditLogger
        logger = AuditLogger(log_path=str(tmp_path / "audit.log"))
        logger.log("a", user="u")
        snapshot = logger.logs
        snapshot.clear()
        assert len(logger.logs) == 1

    def test_logs_ring_buffer_caps_at_maxlen(self, tmp_path):
        from infrastructure.auth import AuditLogger
        logger = AuditLogger(log_path=str(tmp_path / "audit.log"))
        for i in range(1005):
            logger.log(f"e{i}", user="u")
        assert len(logger.logs) == 1000
        assert logger.logs[0]["event_type"] == "e5"

    def test_get_audit_logger_returns_singleton(self, tmp_path, monkeypatch):
        from infrastructure import auth
        from infrastructure.auth import get_audit_logger
        monkeypatch.setattr(auth, "_audit_logger_instance", None)
        first = get_audit_logger()
        second = get_audit_logger()
        assert first is second
        monkeypatch.setattr(auth, "_audit_logger_instance", None)


class TestAuditLoggerFileQuery:
    """AuditLogger.file_query() — persisted audit.log readback."""

    def _write(self, path, lines):
        with open(path, "w") as f:
            for ln in lines:
                f.write(ln + "\n")

    def _ev(self, ts, etype):
        return json.dumps({"timestamp": ts, "event_type": etype, "user": "u"})

    def test_reads_tail_newest_last(self, tmp_path):
        from infrastructure.auth import AuditLogger
        p = str(tmp_path / "audit.log")
        self._write(p, [
            self._ev("2024-01-01T00:00:00+00:00", "a"),
            self._ev("2024-01-01T00:00:01+00:00", "b"),
            self._ev("2024-01-01T00:00:02+00:00", "c"),
        ])
        logger = AuditLogger(log_path=p)
        assert [e["event_type"] for e in logger.file_query()] == ["c", "b", "a"]

    def test_limit_positive_zero_negative(self, tmp_path):
        from infrastructure.auth import AuditLogger
        p = str(tmp_path / "audit.log")
        self._write(p, [self._ev(f"2024-01-01T00:00:{i:02d}+00:00", f"e{i}") for i in range(20)])
        logger = AuditLogger(log_path=p)
        assert len(logger.file_query(limit=5)) == 5
        assert logger.file_query(limit=5)[0]["event_type"] == "e19"
        assert len(logger.file_query(limit=0)) == 20
        assert len(logger.file_query(limit=-5)) == 15

    def test_event_type_filter(self, tmp_path):
        from infrastructure.auth import AuditLogger
        p = str(tmp_path / "audit.log")
        self._write(p, [
            self._ev("2024-01-01T00:00:00+00:00", "auth_success"),
            self._ev("2024-01-01T00:00:01+00:00", "auth_failed"),
            self._ev("2024-01-01T00:00:02+00:00", "auth_success"),
        ])
        logger = AuditLogger(log_path=p)
        res = logger.file_query(event_type="auth_success")
        assert [e["timestamp"] for e in res] == [
            "2024-01-01T00:00:02+00:00", "2024-01-01T00:00:00+00:00",
        ]

    def test_before_cursor_excludes_newer(self, tmp_path):
        from infrastructure.auth import AuditLogger
        p = str(tmp_path / "audit.log")
        self._write(p, [
            self._ev("2024-01-01T00:00:00+00:00", "a"),
            self._ev("2024-01-01T00:00:01+00:00", "b"),
            self._ev("2024-01-01T00:00:02+00:00", "c"),
        ])
        logger = AuditLogger(log_path=p)
        res = logger.file_query(before="2024-01-01T00:00:02+00:00")
        assert [e["timestamp"] for e in res] == [
            "2024-01-01T00:00:01+00:00", "2024-01-01T00:00:00+00:00",
        ]

    def test_malformed_lines_skipped(self, tmp_path):
        from infrastructure.auth import AuditLogger
        p = str(tmp_path / "audit.log")
        self._write(p, ["not json", "", "{broken", self._ev("2024-01-01T00:00:00+00:00", "a")])
        logger = AuditLogger(log_path=p)
        res = logger.file_query()
        assert [e["event_type"] for e in res] == ["a"]

    def test_missing_file_falls_back_to_buffer(self, tmp_path):
        from infrastructure.auth import AuditLogger
        logger = AuditLogger(log_path=str(tmp_path / "nope.log"))
        logger.log("auth_success", user="u")
        res = logger.file_query()
        assert len(res) == 1
        assert res[0]["event_type"] == "auth_success"

    def test_limit_caps_return(self, tmp_path):
        from infrastructure.auth import AuditLogger
        p = str(tmp_path / "audit.log")
        self._write(p, [self._ev(f"2024-01-01T00:00:{i:02d}+00:00", "x") for i in range(10)])
        logger = AuditLogger(log_path=p)
        assert len(logger.file_query(limit=3, event_type="x")) == 3

    @patch("infrastructure.auth.get_audit_logger")
    def test_router_history_true_uses_file_query(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.file_query.return_value = [{"event_type": "auth_success", "timestamp": "1"}]
        resp = client.get("/security/audit?history=true")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 1
        logger.file_query.assert_called_once_with(limit=100, event_type=None, before=None)

    @patch("infrastructure.auth.get_audit_logger")
    def test_router_history_before_and_filter_passthrough(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.file_query.return_value = []
        resp = client.get(
            "/security/audit?history=true&before=2024-01-01T00:00:00%2B00:00&event_type=auth_failed&limit=5"
        )
        assert resp.status_code == 200
        logger.file_query.assert_called_once_with(
            limit=5, event_type="auth_failed", before="2024-01-01T00:00:00+00:00",
        )

    @patch("infrastructure.auth.get_audit_logger")
    def test_router_history_false_ignores_before(self, mock_get_logger, client):
        logger = mock_get_logger.return_value
        logger.logs = [{"event_type": "auth_success", "timestamp": "1"}]
        resp = client.get("/security/audit?before=2024-01-01T00:00:00%2B00:00")
        assert resp.json()["data"]["count"] == 1


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

    def test_keys_wrong_methods_return_405(self, client):
        assert client.put("/security/keys").status_code == 405
        assert client.delete("/security/keys").status_code == 405
