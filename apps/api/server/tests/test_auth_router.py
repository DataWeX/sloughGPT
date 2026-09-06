from infrastructure.exception_handlers import register_app_error_handler

"""
Tests for auth router — login, register, me, token, verify, refresh.

Uses a temporary MogDB instance per test for user storage isolation.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers.auth import AuthRouter, _get_auth_deps, reset_auth_router, set_auth_router


@pytest.fixture(autouse=True)
def _isolated_auth(tmp_path):
    """Create an AuthRouter backed by a temp MogDB, inject it, clean up."""
    db_path = str(tmp_path / "auth_mogdb")
    auth = AuthRouter(db_path=db_path)
    set_auth_router(auth)
    app = FastAPI()
    register_app_error_handler(app)
    app.include_router(auth.router)
    test_client = TestClient(app)
    yield test_client, auth
    reset_auth_router()


@pytest.fixture(autouse=True)
def mock_auth_deps():
    """Mock _get_auth_deps so no real import from main.py happens."""
    mock_jwt = MagicMock()
    mock_jwt.create_token.return_value = "mock-jwt-token"
    mock_jwt.verify_token.return_value = {"sub": "test-user-id", "exp": 9999999999}
    mock_jwt.refresh_token.return_value = "mock-refreshed-jwt-token"

    mock_audit = MagicMock()

    with patch("routers.auth.AuthRouter._get_auth_deps") as mock:
        mock.return_value = ({"test-api-key"}, 24, mock_jwt, mock_audit)
        yield


# ── POST /auth/register ────────────────────────────────────────────────────


class TestRegister:
    def test_register_creates_user(self, _isolated_auth):
        client, _ = _isolated_auth
        resp = client.post(
            "/auth/register",
            json={
                "username": "alice",
                "email": "alice@example.com",
                "password": "secret123",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["token"] == "mock-jwt-token"
        assert data["user"]["username"] == "alice"
        assert data["user"]["email"] == "alice@example.com"
        assert "id" in data["user"]

    def test_register_duplicate_username_returns_409(self, _isolated_auth):
        client, _ = _isolated_auth
        client.post(
            "/auth/register",
            json={
                "username": "bob",
                "email": "bob@example.com",
                "password": "secret123",
            },
        )
        resp = client.post(
            "/auth/register",
            json={
                "username": "bob",
                "email": "bob2@example.com",
                "password": "other456",
            },
        )
        assert resp.status_code == 409
        assert "already exists" in resp.text

    def test_register_persists_password_hash(self, _isolated_auth):
        client, auth = _isolated_auth
        resp = client.post(
            "/auth/register",
            json={
                "username": "carol",
                "email": "carol@example.com",
                "password": "mypassword",
            },
        )
        uid = resp.json()["user"]["id"]
        user = auth.users_collection.find_one({"_id": uid})
        assert user is not None
        assert user["password_hash"].startswith("v1:")


# ── POST /auth/login ───────────────────────────────────────────────────────


class TestLogin:
    def test_login_valid_credentials(self, _isolated_auth):
        client, _ = _isolated_auth
        client.post(
            "/auth/register",
            json={
                "username": "dave",
                "email": "dave@example.com",
                "password": "pass123",
            },
        )
        resp = client.post(
            "/auth/login",
            json={
                "username": "dave",
                "password": "pass123",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["token"] == "mock-jwt-token"
        assert data["user"]["username"] == "dave"

    def test_login_wrong_password(self, _isolated_auth):
        client, _ = _isolated_auth
        client.post(
            "/auth/register",
            json={
                "username": "dave",
                "email": "dave@example.com",
                "password": "pass123",
            },
        )
        resp = client.post(
            "/auth/login",
            json={
                "username": "dave",
                "password": "wrongpass",
            },
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.text

    def test_login_nonexistent_user(self, _isolated_auth):
        client, _ = _isolated_auth
        resp = client.post(
            "/auth/login",
            json={
                "username": "nonexistent",
                "password": "pass123",
            },
        )
        assert resp.status_code == 401


# ── GET /auth/me ───────────────────────────────────────────────────────────


class TestMe:
    def test_me_with_valid_token(self, _isolated_auth):
        client, auth = _isolated_auth
        reg = client.post(
            "/auth/register",
            json={
                "username": "eve",
                "email": "eve@example.com",
                "password": "pass",
            },
        ).json()

        uid = reg["user"]["id"]
        user = auth.users_collection.find_one({"_id": uid})
        auth.users_collection.delete_one({"_id": uid})
        user["_id"] = "test-user-id"
        auth.users_collection.insert_one(user)

        resp = client.get("/auth/me", headers={"Authorization": "Bearer fake-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "eve"
        assert data["email"] == "eve@example.com"

    def test_me_without_token_returns_401(self, _isolated_auth):
        client, _ = _isolated_auth
        resp = client.get("/auth/me")
        assert resp.status_code == 401
        assert "Missing" in resp.text

    def test_me_with_invalid_token_returns_401(self, _isolated_auth):
        client, _ = _isolated_auth
        _, _, jwt_auth, _ = _get_auth_deps()
        jwt_auth.verify_token.return_value = None

        resp = client.get("/auth/me", headers={"Authorization": "Bearer bad-token"})
        assert resp.status_code == 401

    def test_me_with_missing_user_returns_401(self, _isolated_auth):
        client, _ = _isolated_auth
        _, _, jwt_auth, _ = _get_auth_deps()
        jwt_auth.verify_token.return_value = {"sub": "nonexistent-user-id", "exp": 999}

        resp = client.get("/auth/me", headers={"Authorization": "Bearer valid-looking"})
        assert resp.status_code == 401


# ── POST /auth/token ───────────────────────────────────────────────────────


class TestCreateToken:
    def test_token_with_valid_api_key(self, _isolated_auth):
        client, _ = _isolated_auth
        resp = client.post("/auth/token", json={"api_key": "test-api-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["access_token"] == "mock-jwt-token"
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 24 * 3600

    def test_token_with_invalid_api_key_returns_401(self, _isolated_auth):
        client, _ = _isolated_auth
        resp = client.post("/auth/token", json={"api_key": "bad-key"})
        assert resp.status_code == 401

    def test_token_audit_logged_on_success(self, _isolated_auth):
        client, _ = _isolated_auth
        _, _, _, audit_logger = _get_auth_deps_helper()
        audit_logger.log.reset_mock()

        client.post("/auth/token", json={"api_key": "test-api-key"})
        audit_logger.log.assert_called_with(
            "auth_success",
            "testclient",
            resource="/auth/token",
            extra={"action": "token_create", "status": "success"},
        )

    def test_token_audit_logged_on_failure(self, _isolated_auth):
        client, _ = _isolated_auth
        _, _, _, audit_logger = _get_auth_deps_helper()
        audit_logger.log.reset_mock()

        client.post("/auth/token", json={"api_key": "bad-key"})
        audit_logger.log.assert_called_with(
            "auth_failed",
            "testclient",
            resource="/auth/token",
            extra={"action": "token_create", "status": "failure"},
        )


def _get_auth_deps_helper():
    """Get the mocked auth deps for assertion purposes."""
    return _get_auth_deps()


# ── POST /auth/verify ──────────────────────────────────────────────────────


class TestVerify:
    def test_verify_valid_token(self, _isolated_auth):
        client, _ = _isolated_auth
        resp = client.post("/auth/verify", headers={"Authorization": "Bearer valid-token"})
        assert resp.status_code == 200
        data = resp.json()
        inner = data.get("data", data)
        assert inner.get("valid") is True
        assert inner.get("subject") == "test-user-id"

    def test_verify_missing_header_returns_401(self, _isolated_auth):
        client, _ = _isolated_auth
        resp = client.post("/auth/verify")
        assert resp.status_code == 401

    def test_verify_invalid_token_returns_401(self, _isolated_auth):
        client, _ = _isolated_auth
        _, _, jwt_auth, _ = _get_auth_deps()
        jwt_auth.verify_token.return_value = None

        resp = client.post("/auth/verify", headers={"Authorization": "Bearer bad"})
        assert resp.status_code == 401


# ── POST /auth/refresh ────────────────────────────────────────────────────


class TestRefresh:
    def test_refresh_valid_token(self, _isolated_auth):
        client, _ = _isolated_auth
        resp = client.post("/auth/refresh", headers={"Authorization": "Bearer valid-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["access_token"] == "mock-refreshed-jwt-token"
        assert data["token_type"] == "bearer"

    def test_refresh_missing_header_returns_401(self, _isolated_auth):
        client, _ = _isolated_auth
        resp = client.post("/auth/refresh")
        assert resp.status_code == 401

    def test_refresh_failed_returns_401(self, _isolated_auth):
        client, _ = _isolated_auth
        _, _, jwt_auth, _ = _get_auth_deps()
        jwt_auth.refresh_token.return_value = None

        resp = client.post("/auth/refresh", headers={"Authorization": "Bearer bad"})
        assert resp.status_code == 401
