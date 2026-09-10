"""Tests for the API Keys router — 6 endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler


_AUTH_USER = {"sub": "user1", "tenant_id": "t1", "workspace_id": "ws1"}


def _build_app(manager=None, auth_user_dict=_AUTH_USER):
    from routers.api_keys import ApiKeysRouter

    router_obj = ApiKeysRouter(key_manager=manager)
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app


def _mock_manager():
    mgr = MagicMock()
    mgr.create.return_value = {
        "id": "key1",
        "name": "test-key",
        "key": "slo_test123",
        "key_hash": "abc123",
        "scopes": ["*"],
        "created_at": 1000000,
        "revoked": False,
        "workspace_id": "ws1",
        "user_id": "user1",
    }
    mgr.list.return_value = [
        {
            "id": "key1",
            "name": "test-key",
            "key_hash": "abc123",
            "scopes": ["*"],
            "created_at": 1000000,
            "revoked": False,
            "workspace_id": "ws1",
            "user_id": "user1",
        }
    ]
    mgr.get.return_value = {
        "id": "key1",
        "name": "test-key",
        "key": "slo_test123",
        "key_hash": "abc123",
        "scopes": ["*"],
        "created_at": 1000000,
        "revoked": False,
        "workspace_id": "ws1",
        "user_id": "user1",
    }
    mgr.validate.return_value = True
    mgr.rotate.return_value = {
        "id": "key2",
        "name": "test-key",
        "key": "slo_new456",
        "key_hash": "def456",
        "scopes": ["*"],
        "created_at": 1000001,
        "revoked": False,
        "workspace_id": "ws1",
        "user_id": "user1",
    }
    return mgr


# ── Create key ───────────────────────────────────────────────────────────────


class TestCreateKey:
    def test_create_key(self):
        mgr = _mock_manager()
        _app = _build_app(mgr)
        resp = TestClient(_app).post(
            "/security/keys",
            json={"name": "test-key", "scopes": ["*"]},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "test-key"
        assert "key" in data


# ── List keys ────────────────────────────────────────────────────────────────


class TestListKeys:
    def test_list_keys(self):
        mgr = _mock_manager()
        _app = _build_app(mgr)
        resp = TestClient(_app).get("/security/keys")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "keys" in data
        assert data["count"] == 1


# ── Get key ──────────────────────────────────────────────────────────────────


class TestGetKey:
    def test_get_key_found(self):
        mgr = _mock_manager()
        _app = _build_app(mgr)
        resp = TestClient(_app).get("/security/keys/key1")
        assert resp.status_code == 200

    def test_get_key_not_found(self):
        mgr = _mock_manager()
        mgr.get.return_value = None
        _app = _build_app(mgr)
        resp = TestClient(_app).get("/security/keys/nonexistent")
        assert resp.status_code == 404


# ── Delete key ───────────────────────────────────────────────────────────────


class TestDeleteKey:
    def test_delete_key(self):
        mgr = _mock_manager()
        _app = _build_app(mgr)
        resp = TestClient(_app).delete("/security/keys/key1")
        assert resp.status_code == 200
        assert resp.json()["data"]["revoked"] is True

    def test_delete_key_not_found(self):
        mgr = _mock_manager()
        mgr.revoke.side_effect = ValueError("API key not found: nonexistent")
        _app = _build_app(mgr)
        resp = TestClient(_app).delete("/security/keys/nonexistent")
        assert resp.status_code == 404


# ── Rotate key ───────────────────────────────────────────────────────────────


class TestRotateKey:
    def test_rotate_key(self):
        mgr = _mock_manager()
        _app = _build_app(mgr)
        resp = TestClient(_app).post("/security/keys/key1/rotate")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "key" in data

    def test_rotate_key_not_found(self):
        mgr = _mock_manager()
        mgr.rotate.side_effect = ValueError("API key not found: nonexistent")
        _app = _build_app(mgr)
        resp = TestClient(_app).post("/security/keys/nonexistent/rotate")
        assert resp.status_code == 404


# ── Validate key ─────────────────────────────────────────────────────────────


class TestValidateKey:
    def test_validate_key_valid(self):
        mgr = _mock_manager()
        _app = _build_app(mgr)
        resp = TestClient(_app).post(
            "/security/keys/validate",
            json={"key": "slo_test123"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["valid"] is True

    def test_validate_key_invalid(self):
        mgr = _mock_manager()
        mgr.validate.return_value = False
        _app = _build_app(mgr)
        resp = TestClient(_app).post(
            "/security/keys/validate",
            json={"key": "invalid_key"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["valid"] is False
