"""
API Keys Router Tests
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from infrastructure.auth import require_auth_if_enabled
from routers.api_keys import ApiKeysRouter, ApiKeyManager


@pytest.fixture
def app_with_manager():
    app = FastAPI()
    from infrastructure.exception_handlers import register_app_error_handler
    register_app_error_handler(app)

    mock_manager = MagicMock(spec=ApiKeyManager)
    router_obj = ApiKeysRouter(key_manager=mock_manager)
    app.include_router(router_obj.router)
    app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "u1", "workspace_id": "ws1"}

    return app, mock_manager


def _make_client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestCreateKey:
    def test_creates_key(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.create.return_value = {
            "id": "k1", "name": "test-key", "key": "slo_abc123", "key_hash": "hash",
            "scopes": ["*"], "created_at": 1000, "revoked": False,
            "workspace_id": "ws1", "user_id": "u1",
        }

        client = _make_client(app)
        resp = client.post("/security/keys", json={"name": "test-key"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["name"] == "test-key"
        mgr.create.assert_called_once()


class TestListKeys:
    def test_lists_keys(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.list.return_value = [
            {"id": "k1", "name": "key-1", "revoked": False},
            {"id": "k2", "name": "key-2", "revoked": True},
        ]

        client = _make_client(app)
        resp = client.get("/security/keys")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["count"] == 2
        assert len(body["data"]["keys"]) == 2

    def test_empty_list(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.list.return_value = []

        client = _make_client(app)
        resp = client.get("/security/keys")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 0


class TestGetKey:
    def test_returns_key(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.get.return_value = {"id": "k1", "name": "test-key", "key": "slo_secret"}

        client = _make_client(app)
        resp = client.get("/security/keys/k1")
        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == "k1"

    def test_not_found(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.get.return_value = None

        client = _make_client(app)
        resp = client.get("/security/keys/nonexistent")
        assert resp.status_code == 404


class TestDeleteKey:
    def test_revokes_key(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.revoke.return_value = None

        client = _make_client(app)
        resp = client.delete("/security/keys/k1")
        assert resp.status_code == 200
        assert resp.json()["data"]["revoked"] is True
        mgr.revoke.assert_called_once_with("k1")

    def test_not_found(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.revoke.side_effect = ValueError("API key not found: k1")

        client = _make_client(app)
        resp = client.delete("/security/keys/k1")
        assert resp.status_code == 404


class TestRotateKey:
    def test_rotates_key(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.rotate.return_value = {"id": "k2", "name": "test-key", "key": "slo_new123"}

        client = _make_client(app)
        resp = client.post("/security/keys/k1/rotate")
        assert resp.status_code == 200
        assert resp.json()["data"]["key"] == "slo_new123"
        mgr.rotate.assert_called_once_with("k1")

    def test_not_found(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.rotate.side_effect = ValueError("API key not found: k1")

        client = _make_client(app)
        resp = client.post("/security/keys/k1/rotate")
        assert resp.status_code == 404


class TestValidateKey:
    def test_valid_key(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.validate.return_value = True

        client = _make_client(app)
        resp = client.post("/security/keys/validate", json={"key": "slo_abc123"})
        assert resp.status_code == 200
        assert resp.json()["data"]["valid"] is True

    def test_invalid_key(self, app_with_manager):
        app, mgr = app_with_manager
        mgr.validate.return_value = False

        client = _make_client(app)
        resp = client.post("/security/keys/validate", json={"key": "bad_key"})
        assert resp.status_code == 200
        assert resp.json()["data"]["valid"] is False
