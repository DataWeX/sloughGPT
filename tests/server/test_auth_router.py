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
