"""
Tests for auth router — login, register, me, token, verify, refresh.

Only registers the auth router to avoid pulling in heavy dependencies
(transformers, torch, peft) that would slow test startup.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from routers.auth import router as auth_router

app = FastAPI()
app.include_router(auth_router)
client = TestClient(app)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_auth_deps():
    """Mock _get_auth_deps so no real import from main.py happens."""
    mock_jwt = MagicMock()
    mock_jwt.create_token.return_value = "mock-jwt-token"
    mock_jwt.verify_token.return_value = {"sub": "test-user-id", "exp": 9999999999}
    mock_jwt.refresh_token.return_value = "mock-refreshed-jwt-token"

    mock_audit = MagicMock()

    with patch("routers.auth._get_auth_deps") as mock:
        mock.return_value = ({"test-api-key"}, 24, mock_jwt, mock_audit)
        yield


@pytest.fixture(autouse=True)
def mock_users_file(tmp_path):
    """Redirect USERS_FILE to a temp path so tests don't touch real data."""
    users_path = tmp_path / "users.json"
    users_path.parent.mkdir(parents=True, exist_ok=True)
    users_path.write_text("{}")
    with patch("routers.auth.USERS_FILE", str(users_path)):
        yield


# ── POST /auth/register ────────────────────────────────────────────────────

class TestRegister:

    def test_register_creates_user(self):
        resp = client.post("/auth/register", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "secret123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["token"] == "mock-jwt-token"
        assert data["user"]["username"] == "alice"
        assert data["user"]["email"] == "alice@example.com"
        assert "id" in data["user"]

    def test_register_duplicate_username_returns_409(self):
        client.post("/auth/register", json={
            "username": "bob",
            "email": "bob@example.com",
            "password": "secret123",
        })
        resp = client.post("/auth/register", json={
            "username": "bob",
            "email": "bob2@example.com",
            "password": "other456",
        })
        assert resp.status_code == 409
        assert "already exists" in resp.text

    def test_register_persists_password_hash(self):
        client.post("/auth/register", json={
            "username": "carol",
            "email": "carol@example.com",
            "password": "mypassword",
        })
        # Read back the users file
        import hashlib
        hashed = hashlib.sha256("mypassword".encode()).hexdigest()

        with patch("routers.auth._load_users") as mock_load:
            mock_load.return_value = {
                "some-id": {
                    "username": "carol",
                    "email": "carol@example.com",
                    "password_hash": hashed,
                }
            }
            users = mock_load()
            assert users["some-id"]["password_hash"] == hashed


# ── POST /auth/login ───────────────────────────────────────────────────────

class TestLogin:

    @pytest.fixture(autouse=True)
    def _register_user(self):
        client.post("/auth/register", json={
            "username": "dave",
            "email": "dave@example.com",
            "password": "pass123",
        })

    def test_login_valid_credentials(self):
        resp = client.post("/auth/login", json={
            "username": "dave",
            "password": "pass123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["token"] == "mock-jwt-token"
        assert data["user"]["username"] == "dave"

    def test_login_wrong_password(self):
        resp = client.post("/auth/login", json={
            "username": "dave",
            "password": "wrongpass",
        })
        assert resp.status_code == 401
        assert "Invalid" in resp.text

    def test_login_nonexistent_user(self):
        resp = client.post("/auth/login", json={
            "username": "nonexistent",
            "password": "pass123",
        })
        assert resp.status_code == 401


# ── GET /auth/me ───────────────────────────────────────────────────────────

class TestMe:

    def test_me_with_valid_token(self):
        # Register to create a real user in the file
        reg = client.post("/auth/register", json={
            "username": "eve",
            "email": "eve@example.com",
            "password": "pass",
        }).json()

        # The mock jwt returns sub="test-user-id" regardless. We need to
        # add that user ID to the users file.
        uid = reg["user"]["id"]
        from routers.auth import _load_users, _save_users, _hash_password
        users = _load_users()
        users["test-user-id"] = users.pop(uid, {
            "username": "eve",
            "email": "eve@example.com",
            "password_hash": _hash_password("pass"),
        })
        _save_users(users)

        resp = client.get("/auth/me", headers={"Authorization": "Bearer fake-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "eve"
        assert data["email"] == "eve@example.com"

    def test_me_without_token_returns_401(self):
        resp = client.get("/auth/me")
        assert resp.status_code == 401
        assert "Missing" in resp.text

    def test_me_with_invalid_token_returns_401(self):
        from routers.auth import _get_auth_deps
        _, _, jwt_auth, _ = _get_auth_deps()
        jwt_auth.verify_token.return_value = None

        resp = client.get("/auth/me", headers={"Authorization": "Bearer bad-token"})
        assert resp.status_code == 401

    def test_me_with_missing_user_returns_401(self):
        from routers.auth import _get_auth_deps
        _, _, jwt_auth, _ = _get_auth_deps()
        jwt_auth.verify_token.return_value = {"sub": "nonexistent-user-id", "exp": 999}

        resp = client.get("/auth/me", headers={"Authorization": "Bearer valid-looking"})
        assert resp.status_code == 401


# ── POST /auth/token ───────────────────────────────────────────────────────

class TestCreateToken:

    def test_token_with_valid_api_key(self):
        resp = client.post("/auth/token", json={"api_key": "test-api-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["access_token"] == "mock-jwt-token"
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 24 * 3600

    def test_token_with_invalid_api_key_returns_401(self):
        resp = client.post("/auth/token", json={"api_key": "bad-key"})
        assert resp.status_code == 401

    def test_token_audit_logged_on_success(self):
        _, _, _, audit_logger = _get_auth_deps_helper()
        audit_logger.log.reset_mock()

        client.post("/auth/token", json={"api_key": "test-api-key"})
        audit_logger.log.assert_called_with(
            "auth_success",
            "testclient",
            resource="/auth/token",
            action="token_create",
            status="success",
        )

    def test_token_audit_logged_on_failure(self):
        _, _, _, audit_logger = _get_auth_deps_helper()
        audit_logger.log.reset_mock()

        client.post("/auth/token", json={"api_key": "bad-key"})
        audit_logger.log.assert_called_with(
            "auth_failed",
            "testclient",
            resource="/auth/token",
            action="token_create",
            status="failure",
        )


def _get_auth_deps_helper():
    """Get the mocked auth deps for assertion purposes."""
    from routers.auth import _get_auth_deps
    return _get_auth_deps()


# ── POST /auth/verify ──────────────────────────────────────────────────────

class TestVerify:

    def test_verify_valid_token(self):
        resp = client.post("/auth/verify", headers={"Authorization": "Bearer valid-token"})
        assert resp.status_code == 200
        data = resp.json()
        inner = data.get("data", data)
        assert inner.get("valid") is True
        assert inner.get("subject") == "test-user-id"

    def test_verify_missing_header_returns_401(self):
        resp = client.post("/auth/verify")
        assert resp.status_code == 401

    def test_verify_invalid_token_returns_401(self):
        from routers.auth import _get_auth_deps
        _, _, jwt_auth, _ = _get_auth_deps()
        jwt_auth.verify_token.return_value = None

        resp = client.post("/auth/verify", headers={"Authorization": "Bearer bad"})
        assert resp.status_code == 401


# ── POST /auth/refresh ────────────────────────────────────────────────────

class TestRefresh:

    def test_refresh_valid_token(self):
        resp = client.post("/auth/refresh", headers={"Authorization": "Bearer valid-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["access_token"] == "mock-refreshed-jwt-token"
        assert data["token_type"] == "bearer"

    def test_refresh_missing_header_returns_401(self):
        resp = client.post("/auth/refresh")
        assert resp.status_code == 401

    def test_refresh_failed_returns_401(self):
        from routers.auth import _get_auth_deps
        _, _, jwt_auth, _ = _get_auth_deps()
        jwt_auth.refresh_token.return_value = None

        resp = client.post("/auth/refresh", headers={"Authorization": "Bearer bad"})
        assert resp.status_code == 401
