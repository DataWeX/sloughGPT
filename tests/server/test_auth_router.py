"""
Tests for the auth router — login, register, me, token, verify, refresh.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.auth import AuthRouter


@pytest.fixture
def router():
    return AuthRouter()


@pytest.fixture
def app(router):
    _app = FastAPI()
    _app.include_router(router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _make_jwt_mock():
    jwt = MagicMock()
    jwt.create_token.return_value = "fake_jwt"
    return jwt


class TestRegister:
    def test_registers_user(self, router, client):
        router._load_users = MagicMock(return_value={})
        router._save_users = MagicMock()
        router._hash_password = MagicMock(return_value="v1:abc:def")
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, _make_jwt_mock(), MagicMock()))
        resp = client.post("/auth/register", json={
            "username": "alice", "email": "a@b.com", "password": "secret",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"]["username"] == "alice"

    def test_rejects_duplicate_username(self, router, client):
        router._load_users = MagicMock(return_value={
            "uid1": {"username": "alice", "email": "a@b.com", "password_hash": "x"},
        })
        resp = client.post("/auth/register", json={
            "username": "alice", "email": "a@b.com", "password": "secret",
        })
        assert resp.status_code == 409

    def test_rejects_duplicate_email(self, router, client):
        router._load_users = MagicMock(return_value={
            "uid1": {"username": "alice", "email": "a@b.com", "password_hash": "x"},
        })
        resp = client.post("/auth/register", json={
            "username": "bob", "email": "a@b.com", "password": "secret",
        })
        assert resp.status_code == 200  # email not gated; username is the unique key

    def test_missing_fields_422(self, router, client):
        resp = client.post("/auth/register", json={"username": "alice"})
        assert resp.status_code == 422

    def test_register_returns_token(self, router, client):
        jwt = _make_jwt_mock()
        jwt.create_token.return_value = "reg_token"
        router._load_users = MagicMock(return_value={})
        router._save_users = MagicMock()
        router._hash_password = MagicMock(return_value="v1:abc:def")
        router._get_auth_deps = MagicMock(return_value=(["key"], 24, jwt, MagicMock()))
        resp = client.post("/auth/register", json={
            "username": "carol", "email": "c@b.com", "password": "p",
        })
        assert resp.json()["token"] == "reg_token"


class TestLogin:
    def test_login_success(self, router, client):
        router._load_users = MagicMock(return_value={
            "uid1": {
                "username": "alice", "email": "a@b.com",
                "password_hash": "v1:salt:" + "ab" * 32,
            },
        })
        router._verify_password = MagicMock(return_value=True)
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, _make_jwt_mock(), MagicMock()))
        resp = client.post("/auth/login", json={
            "username": "alice", "password": "secret",
        })
        assert resp.status_code == 200
        assert resp.json()["user"]["username"] == "alice"

    def test_login_wrong_password(self, router, client):
        router._load_users = MagicMock(return_value={
            "uid1": {"username": "alice", "email": "a@b.com", "password_hash": "v1:salt:xx"},
        })
        router._verify_password = MagicMock(return_value=False)
        resp = client.post("/auth/login", json={
            "username": "alice", "password": "wrong",
        })
        assert resp.status_code == 401

    def test_login_unknown_user(self, router, client):
        router._load_users = MagicMock(return_value={
            "uid1": {"username": "alice", "email": "a@b.com", "password_hash": "v1:salt:xx"},
        })
        resp = client.post("/auth/login", json={"username": "ghost", "password": "x"})
        assert resp.status_code == 401

    def test_login_migrates_legacy_hash(self, router, client):
        router._load_users = MagicMock(return_value={
            "uid1": {"username": "alice", "email": "a@b.com", "password_hash": "legacy"},
        })
        router._verify_password = MagicMock(return_value=True)
        router._hash_password = MagicMock(return_value="v1:migrated")
        router._save_users = MagicMock()
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, _make_jwt_mock(), MagicMock()))
        resp = client.post("/auth/login", json={"username": "alice", "password": "secret"})
        assert resp.status_code == 200
        router._save_users.assert_called_once()

    def test_login_missing_fields_422(self, router, client):
        resp = client.post("/auth/login", json={"username": "alice"})
        assert resp.status_code == 422


class TestToken:
    def test_creates_token_for_valid_key(self, router, client):
        jwt = _make_jwt_mock()
        router._get_auth_deps = MagicMock(return_value=
            (["valid-key"], 24, jwt, MagicMock()))
        resp = client.post("/auth/token", json={"api_key": "valid-key"})
        assert resp.status_code == 200
        assert resp.json()["access_token"] == "fake_jwt"

    def test_rejects_invalid_key(self, router, client):
        jwt = _make_jwt_mock()
        router._get_auth_deps = MagicMock(return_value=
            (["valid-key"], 24, jwt, MagicMock()))
        resp = client.post("/auth/token", json={"api_key": "bad-key"})
        assert resp.status_code == 401

    def test_missing_api_key_422(self, router, client):
        resp = client.post("/auth/token", json={})
        assert resp.status_code == 422

    def test_expires_in_is_hours_seconds(self, router, client):
        jwt = _make_jwt_mock()
        router._get_auth_deps = MagicMock(return_value=
            (["valid-key"], 2, jwt, MagicMock()))
        resp = client.post("/auth/token", json={"api_key": "valid-key"})
        assert resp.json()["expires_in"] == 2 * 3600

    def test_failed_key_is_audited(self, router, client):
        jwt = _make_jwt_mock()
        audit = MagicMock()
        router._get_auth_deps = MagicMock(return_value=(["valid-key"], 24, jwt, audit))
        client.post("/auth/token", json={"api_key": "bad-key"})
        audit.log.assert_called_once()


class TestGetMe:
    def test_requires_auth(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code in (401, 403)

    def test_me_with_valid_token(self, router, client):
        jwt = _make_jwt_mock()
        jwt.verify_token.return_value = {"sub": "uid1"}
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, jwt, MagicMock()))
        router._load_users = MagicMock(return_value={
            "uid1": {"username": "alice", "email": "a@b.com", "password_hash": "x"},
        })
        resp = client.get("/auth/me", headers={"Authorization": "Bearer t"})
        assert resp.status_code == 200
        assert resp.json()["username"] == "alice"

    def test_me_unknown_user_401(self, router, client):
        jwt = _make_jwt_mock()
        jwt.verify_token.return_value = {"sub": "nope"}
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, jwt, MagicMock()))
        router._load_users = MagicMock(return_value={})
        resp = client.get("/auth/me", headers={"Authorization": "Bearer t"})
        assert resp.status_code == 401


class TestVerifyToken:
    def test_verify_valid_token(self, router, client):
        jwt = _make_jwt_mock()
        jwt.verify_token.return_value = {"user_id": "u1", "username": "alice"}
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, jwt, MagicMock()))
        resp = client.post("/auth/verify", headers={"Authorization": "Bearer fake_jwt"})
        assert resp.status_code == 200

    def test_verify_no_header(self, router, client):
        jwt = _make_jwt_mock()
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, jwt, MagicMock()))
        resp = client.post("/auth/verify")
        assert resp.status_code == 401

    def test_verify_invalid_token(self, router, client):
        jwt = _make_jwt_mock()
        jwt.verify_token.return_value = None
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, jwt, MagicMock()))
        resp = client.post("/auth/verify", headers={"Authorization": "Bearer bad"})
        assert resp.status_code == 401

    def test_verify_bad_scheme(self, router, client):
        jwt = _make_jwt_mock()
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, jwt, MagicMock()))
        resp = client.post("/auth/verify", headers={"Authorization": "Basic xyz"})
        assert resp.status_code == 401


class TestRefreshToken:
    def test_refresh_valid_token(self, router, client):
        jwt = _make_jwt_mock()
        jwt.verify_token.return_value = {"user_id": "u1"}
        jwt.refresh_token.return_value = "new_jwt"
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, jwt, MagicMock()))
        resp = client.post("/auth/refresh", headers={"Authorization": "Bearer old_jwt"})
        assert resp.status_code == 200

    def test_refresh_no_header(self, router, client):
        jwt = _make_jwt_mock()
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, jwt, MagicMock()))
        resp = client.post("/auth/refresh")
        assert resp.status_code == 401

    def test_refresh_invalid_token(self, router, client):
        jwt = _make_jwt_mock()
        jwt.refresh_token.return_value = None
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 24, jwt, MagicMock()))
        resp = client.post("/auth/refresh", headers={"Authorization": "Bearer bad"})
        assert resp.status_code == 401

    def test_refresh_returns_expiry(self, router, client):
        jwt = _make_jwt_mock()
        jwt.refresh_token.return_value = "new_jwt"
        router._get_auth_deps = MagicMock(return_value=
            (["key"], 3, jwt, MagicMock()))
        resp = client.post("/auth/refresh", headers={"Authorization": "Bearer old"})
        assert resp.json()["expires_in"] == 3 * 3600
