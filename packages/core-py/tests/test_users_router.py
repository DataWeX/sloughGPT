"""
Users Router Tests
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from infrastructure.auth import require_auth_if_enabled


def _make_user(uid="u1", username="alice", email="alice@test.com", role="admin", status="active"):
    u = MagicMock()
    u.id = uid
    u.username = username
    u.email = email
    u.role.value = role
    u.status.value = status
    u.display_name = username.title()
    u.tenant_id = "t1"
    u.created_at = "2026-01-01T00:00:00Z"
    u.last_login_at = "2026-09-10T00:00:00Z"
    u.is_admin = role in ("admin", "owner")
    return u


@pytest.fixture
def admin_user():
    return _make_user("admin1", role="admin")


@pytest.fixture
def regular_user():
    return _make_user("u2", "bob", "bob@test.com", role="user")


@pytest.fixture
def app_with_repo():
    """Create app with mocked UserRepository and configurable auth."""
    app = FastAPI()

    from infrastructure.exception_handlers import register_app_error_handler
    register_app_error_handler(app)

    mock_repo = MagicMock()

    with patch("routers.users.UserRepository", return_value=mock_repo):
        from routers.users import UsersRouter
        router_obj = UsersRouter()
        app.include_router(router_obj.router)

    app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "u2", "tenant_id": "t1"}

    return app, mock_repo


def _make_client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListUsers:
    def test_admin_can_list(self, app_with_repo, admin_user):
        app, repo = app_with_repo
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1", "tenant_id": "t1"}
        repo.get.return_value = admin_user
        repo.list_by_tenant.return_value = [admin_user, _make_user("u2", "bob")]

        client = _make_client(app)
        resp = client.get("/users")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert len(body["data"]) == 2

    def test_non_admin_rejected(self, app_with_repo):
        app, repo = app_with_repo
        user = _make_user("u2", "bob", role="user")
        repo.get.return_value = user

        client = _make_client(app)
        resp = client.get("/users")
        assert resp.status_code == 403


class TestGetUser:
    def test_admin_can_get_any(self, app_with_repo, admin_user):
        app, repo = app_with_repo
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1", "tenant_id": "t1"}
        target = _make_user("u2", "bob")
        repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else target

        client = _make_client(app)
        resp = client.get("/users/u2")
        assert resp.status_code == 200
        assert resp.json()["data"]["username"] == "bob"

    def test_user_can_get_own(self, app_with_repo):
        app, repo = app_with_repo
        user = _make_user("u2", "bob", role="user")
        repo.get.return_value = user

        client = _make_client(app)
        resp = client.get("/users/u2")
        assert resp.status_code == 200

    def test_user_cannot_get_other(self, app_with_repo):
        app, repo = app_with_repo
        user = _make_user("u2", "bob", role="user")
        other = _make_user("u3", "charlie")
        repo.get.side_effect = lambda uid: user if uid == "u2" else other

        client = _make_client(app)
        resp = client.get("/users/u3")
        assert resp.status_code == 403

    def test_not_found(self, app_with_repo, admin_user):
        app, repo = app_with_repo
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1", "tenant_id": "t1"}
        repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else None

        client = _make_client(app)
        resp = client.get("/users/nonexistent")
        assert resp.status_code == 404


class TestCreateUser:
    def test_admin_creates_user(self, app_with_repo, admin_user):
        app, repo = app_with_repo
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1", "tenant_id": "t1"}
        repo.get.return_value = admin_user
        repo.get_by_username.return_value = None
        repo.get_by_email.return_value = None

        client = _make_client(app)
        resp = client.post("/users", json={
            "username": "newuser",
            "email": "new@test.com",
            "password": "securepass123",
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["username"] == "newuser"
        repo.create.assert_called_once()

    def test_duplicate_username(self, app_with_repo, admin_user):
        app, repo = app_with_repo
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1", "tenant_id": "t1"}
        repo.get.return_value = admin_user
        repo.get_by_username.return_value = _make_user("u2", "existing")

        client = _make_client(app)
        resp = client.post("/users", json={
            "username": "existing",
            "email": "new@test.com",
            "password": "securepass123",
        })
        assert resp.status_code == 409


class TestDeleteUser:
    def test_admin_deletes_user(self, app_with_repo, admin_user):
        app, repo = app_with_repo
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1", "tenant_id": "t1"}
        target = _make_user("u2", "bob")
        repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else target

        client = _make_client(app)
        resp = client.delete("/users/u2")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True
        repo.delete.assert_called_once_with("u2")

    def test_admin_cannot_delete_self(self, app_with_repo, admin_user):
        app, repo = app_with_repo
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1", "tenant_id": "t1"}
        repo.get.return_value = admin_user

        client = _make_client(app)
        resp = client.delete("/users/admin1")
        assert resp.status_code == 400


class TestChangePassword:
    def test_changes_password(self, app_with_repo):
        app, repo = app_with_repo
        user = _make_user("u2", "bob", role="user")
        user.password_hash = "old_hash"
        repo.get.return_value = user

        with patch("routers.auth.AuthRouter") as mock_auth_cls:
            mock_auth_cls._verify_password.return_value = True
            mock_auth_cls._hash_password.return_value = "new_hash"

            client = _make_client(app)
            resp = client.post("/users/me/password", json={
                "current_password": "oldpass123",
                "new_password": "newpass456",
            })
            assert resp.status_code == 200
            assert resp.json()["data"]["changed"] is True
            assert user.password_hash == "new_hash"

    def test_wrong_password(self, app_with_repo):
        app, repo = app_with_repo
        user = _make_user("u2", "bob", role="user")
        user.password_hash = "old_hash"
        repo.get.return_value = user

        with patch("routers.auth.AuthRouter") as mock_auth_cls:
            mock_auth_cls._verify_password.return_value = False

            client = _make_client(app)
            resp = client.post("/users/me/password", json={
                "current_password": "wrongpass",
                "new_password": "newpass456",
            })
            assert resp.status_code == 401


class TestGetOwnProfile:
    def test_returns_own_profile(self, app_with_repo):
        app, repo = app_with_repo
        user = _make_user("u2", "bob", role="user")
        repo.get.return_value = user

        client = _make_client(app)
        resp = client.get("/users/me/profile")
        assert resp.status_code == 200
        assert resp.json()["data"]["username"] == "bob"
