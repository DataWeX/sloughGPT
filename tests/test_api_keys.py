"""Tests for API key management (CRUD, rotation, expiry, revocation)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[2] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.api_keys import ApiKeyManager, ApiKeysRouter
from tests.conftest import build_test_app


@pytest.fixture
def tmp_mogdb(tmp_path):
    from mogdb import MogDB

    db_path = str(tmp_path / "api_keys_mogdb")
    sync_path = str(tmp_path / "api_keys_json")
    return MogDB(db_path, sync_dir=sync_path)


@pytest.fixture
def manager(tmp_mogdb):
    return ApiKeyManager(db=tmp_mogdb)


@pytest.fixture
def client(manager):
    router = ApiKeysRouter(key_manager=manager)
    return TestClient(build_test_app(router.router))


class TestApiKeyManager:
    def test_create_key(self, manager):
        key = manager.create("test-key", scopes=["*"])
        assert key["name"] == "test-key"
        assert key["key"].startswith("slo_")
        assert key["scopes"] == ["*"]
        assert key["revoked"] is False
        assert "expires_at" not in key

    def test_create_key_with_expiry(self, manager):
        expires = int(time.time()) + 3600
        key = manager.create("expiring", scopes=["*"], expires_at=expires)
        assert key["expires_at"] == expires

    def test_list_keys_hides_full_key(self, manager):
        manager.create("k1", scopes=["*"])
        keys = manager.list()
        assert len(keys) == 1
        assert "key" not in keys[0]
        assert "key_hash" in keys[0]

    def test_get_key_by_id(self, manager):
        created = manager.create("k1", scopes=["*"])
        found = manager.get(created["id"])
        assert found["name"] == "k1"

    def test_get_nonexistent_returns_none(self, manager):
        assert manager.get("nonexistent") is None

    def test_revoke_key(self, manager):
        created = manager.create("k1", scopes=["*"])
        manager.revoke(created["id"])
        found = manager.get(created["id"])
        assert found["revoked"] is True

    def test_revoke_nonexistent_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.revoke("nonexistent")

    def test_rotate_key(self, manager):
        old = manager.create("k1", scopes=["*"])
        new = manager.rotate(old["id"])
        assert new["key"] != old["key"]
        assert new["key"].startswith("slo_")
        assert manager.get(old["id"])["revoked"] is True
        assert manager.get(new["id"])["revoked"] is False

    def test_rotate_nonexistent_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.rotate("nonexistent")

    def test_validate_key_valid(self, manager):
        created = manager.create("k1", scopes=["*"])
        assert manager.validate(created["key"]) is True

    def test_validate_revoked_key(self, manager):
        created = manager.create("k1", scopes=["*"])
        manager.revoke(created["id"])
        assert manager.validate(created["key"]) is False

    def test_validate_expired_key(self, manager):
        expires = int(time.time()) - 1
        created = manager.create("k1", scopes=["*"], expires_at=expires)
        assert manager.validate(created["key"]) is False

    def test_validate_unknown_key(self, manager):
        assert manager.validate("slo_unknown") is False


class TestApiKeysEndpoints:
    def test_create_key(self, client):
        resp = client.post("/security/keys", json={"name": "test", "scopes": ["*"]})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "test"
        assert data["key"].startswith("slo_")

    def test_list_keys(self, client):
        client.post("/security/keys", json={"name": "k1", "scopes": ["*"]})
        client.post("/security/keys", json={"name": "k2", "scopes": ["*"]})
        resp = client.get("/security/keys")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 2

    def test_get_key(self, client):
        create_resp = client.post("/security/keys", json={"name": "k1", "scopes": ["*"]})
        key_id = create_resp.json()["data"]["id"]
        resp = client.get(f"/security/keys/{key_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "k1"

    def test_get_nonexistent_key(self, client):
        resp = client.get("/security/keys/nonexistent")
        assert resp.status_code == 404

    def test_delete_key(self, client):
        create_resp = client.post("/security/keys", json={"name": "k1", "scopes": ["*"]})
        key_id = create_resp.json()["data"]["id"]
        resp = client.delete(f"/security/keys/{key_id}")
        assert resp.status_code == 200

    def test_rotate_key(self, client):
        create_resp = client.post("/security/keys", json={"name": "k1", "scopes": ["*"]})
        key_id = create_resp.json()["data"]["id"]
        resp = client.post(f"/security/keys/{key_id}/rotate")
        assert resp.status_code == 200
        assert resp.json()["data"]["key"].startswith("slo_")

    def test_validate_key_endpoint(self, client):
        create_resp = client.post("/security/keys", json={"name": "k1", "scopes": ["*"]})
        key = create_resp.json()["data"]["key"]
        resp = client.post("/security/keys/validate", json={"key": key})
        assert resp.status_code == 200
        assert resp.json()["data"]["valid"] is True

    def test_validate_invalid_key(self, client):
        resp = client.post("/security/keys/validate", json={"key": "slo_bad"})
        assert resp.status_code == 200
        assert resp.json()["data"]["valid"] is False
