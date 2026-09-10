"""Tests for the Users admin API router — RBAC, CRUD, self-service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from domains.auth.models import Role, User, UserRole
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_user(
    user_id: str = "u1",
    username: str = "alice",
    email: str = "alice@example.com",
    role: Role = Role.ADMIN,
    status: UserRole = UserRole.ACTIVE,
    tenant_id: str = "t1",
) -> User:
    return User(
        id=user_id,
        username=username,
        email=email,
        password_hash="hashed",
        display_name=username.title(),
        role=role,
        status=status,
        tenant_id=tenant_id,
    )


_AUTH_ADMIN = {"sub": "admin1", "tenant_id": "t1"}
_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def admin_user():
    return _make_user("admin1", "admin", role=Role.ADMIN)


@pytest.fixture
def regular_user():
    return _make_user("user1", "alice", role=Role.USER)


def _build_app(mock_repo, auth_user_dict=_AUTH_ADMIN):
    from routers.users import UsersRouter

    router_obj = UsersRouter()
    router_obj._repo = mock_repo

    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app, router_obj


# ── List users ───────────────────────────────────────────────────────────────

class TestListUsers:
    def test_list_returns_all(self, mock_repo, admin_user):
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else None
        mock_repo.list_by_tenant.return_value = [
            _make_user("u1", "alice"),
            _make_user("u2", "bob", role=Role.USER),
        ]
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).get("/users")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 2

    def test_non_admin_rejected(self, mock_repo, regular_user):
        mock_repo.get.side_effect = lambda uid: regular_user if uid == "user1" else None
        _app, _ = _build_app(mock_repo, auth_user_dict=_AUTH_USER)
        resp = TestClient(_app).get("/users")
        assert resp.status_code == 403


# ── Get user ─────────────────────────────────────────────────────────────────

class TestGetUser:
    def test_admin_can_get_any_user(self, mock_repo, admin_user):
        target = _make_user("u2", "bob")
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else target
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).get("/users/u2")
        assert resp.status_code == 200
        assert resp.json()["data"]["username"] == "bob"

    def test_user_cannot_use_admin_endpoint(self, mock_repo, regular_user):
        # GET /users/{id} is admin-only for OTHER users — regular users must use /users/me/profile
        other = _make_user("u2", "bob")
        mock_repo.get.side_effect = lambda uid: regular_user if uid == "user1" else other
        _app, _ = _build_app(mock_repo, auth_user_dict=_AUTH_USER)
        resp = TestClient(_app).get("/users/u2")
        assert resp.status_code == 403

    def test_user_cannot_get_other_profile(self, mock_repo, regular_user):
        other = _make_user("u2", "bob")
        mock_repo.get.side_effect = lambda uid: regular_user if uid == "user1" else other
        _app, _ = _build_app(mock_repo, auth_user_dict=_AUTH_USER)
        resp = TestClient(_app).get("/users/u2")
        assert resp.status_code == 403

    def test_user_not_found(self, mock_repo, admin_user):
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else None
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).get("/users/nonexistent")
        assert resp.status_code == 404


# ── Create user ──────────────────────────────────────────────────────────────

class TestCreateUser:
    def test_admin_creates_user(self, mock_repo, admin_user):
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else None
        mock_repo.get_by_username.return_value = None
        mock_repo.get_by_email.return_value = None
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).post("/users", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "password123",
            "role": "user",
        })
        assert resp.status_code == 200
        mock_repo.create.assert_called_once()

    def test_duplicate_username_rejected(self, mock_repo, admin_user):
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else None
        mock_repo.get_by_username.return_value = _make_user("x", "newuser")
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).post("/users", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "password123",
        })
        assert resp.status_code == 409

    def test_duplicate_email_rejected(self, mock_repo, admin_user):
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else None
        mock_repo.get_by_username.return_value = None
        mock_repo.get_by_email.return_value = _make_user("x", "other", email="new@example.com")
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).post("/users", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "password123",
        })
        assert resp.status_code == 409

    def test_non_admin_cannot_create(self, mock_repo, regular_user):
        mock_repo.get.side_effect = lambda uid: regular_user if uid == "user1" else None
        _app, _ = _build_app(mock_repo, auth_user_dict=_AUTH_USER)
        resp = TestClient(_app).post("/users", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "password123",
        })
        assert resp.status_code == 403


# ── Update user ──────────────────────────────────────────────────────────────

class TestUpdateUser:
    def test_admin_updates_email(self, mock_repo, admin_user):
        target = _make_user("u2", "bob")
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else target
        mock_repo.get_by_email.return_value = None
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).put("/users/u2", json={"email": "bob@new.com"})
        assert resp.status_code == 200
        assert resp.json()["data"]["email"] == "bob@new.com"

    def test_update_duplicate_email_rejected(self, mock_repo, admin_user):
        target = _make_user("u2", "bob")
        other = _make_user("u3", "carol", email="taken@example.com")
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else target
        mock_repo.get_by_email.return_value = other
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).put("/users/u2", json={"email": "taken@example.com"})
        assert resp.status_code == 409

    def test_update_nonexistent_rejected(self, mock_repo, admin_user):
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else None
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).put("/users/nope", json={"email": "x@y.com"})
        assert resp.status_code == 404


# ── Delete user ──────────────────────────────────────────────────────────────

class TestDeleteUser:
    def test_admin_deletes_user(self, mock_repo, admin_user):
        target = _make_user("u2", "bob")
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else target
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).delete("/users/u2")
        assert resp.status_code == 200
        mock_repo.delete.assert_called_once_with("u2")

    def test_cannot_delete_self(self, mock_repo, admin_user):
        mock_repo.get.side_effect = lambda uid: admin_user
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).delete("/users/admin1")
        assert resp.status_code == 400

    def test_delete_nonexistent_rejected(self, mock_repo, admin_user):
        mock_repo.get.side_effect = lambda uid: admin_user if uid == "admin1" else None
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).delete("/users/nope")
        assert resp.status_code == 404


# ── Change password ──────────────────────────────────────────────────────────

class TestChangePassword:
    def test_success(self, mock_repo, admin_user):
        mock_repo.get.return_value = admin_user
        _app, _ = _build_app(mock_repo)
        with patch("routers.auth.AuthRouter._verify_password", return_value=True), \
             patch("routers.auth.AuthRouter._hash_password", return_value="new_hash"):
            resp = TestClient(_app).post("/users/me/password", json={
                "current_password": "oldpass123",
                "new_password": "newpass123",
            })
        assert resp.status_code == 200
        assert resp.json()["data"]["changed"] is True

    def test_wrong_password_rejected(self, mock_repo, admin_user):
        mock_repo.get.return_value = admin_user
        _app, _ = _build_app(mock_repo)
        with patch("routers.auth.AuthRouter._verify_password", return_value=False):
            resp = TestClient(_app).post("/users/me/password", json={
                "current_password": "wrongpass",
                "new_password": "newpass123",
            })
        assert resp.status_code == 401

    def test_user_not_found(self, mock_repo):
        mock_repo.get.return_value = None
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).post("/users/me/password", json={
            "current_password": "oldpass123",
            "new_password": "newpass123",
        })
        assert resp.status_code == 404


# ── Update profile ───────────────────────────────────────────────────────────

class TestUpdateProfile:
    def test_update_own_display_name(self, mock_repo, regular_user):
        mock_repo.get.return_value = regular_user
        _app, _ = _build_app(mock_repo, auth_user_dict=_AUTH_USER)
        resp = TestClient(_app).put("/users/me/profile", json={"display_name": "Alice A."})
        assert resp.status_code == 200
        assert resp.json()["data"]["display_name"] == "Alice A."

    def test_update_own_email(self, mock_repo, regular_user):
        mock_repo.get.return_value = regular_user
        mock_repo.get_by_email.return_value = None
        _app, _ = _build_app(mock_repo, auth_user_dict=_AUTH_USER)
        resp = TestClient(_app).put("/users/me/profile", json={"email": "alice@new.com"})
        assert resp.status_code == 200
        assert resp.json()["data"]["email"] == "alice@new.com"

    def test_duplicate_email_rejected(self, mock_repo, regular_user):
        mock_repo.get.return_value = regular_user
        mock_repo.get_by_email.return_value = _make_user("x", "other", email="taken@example.com")
        _app, _ = _build_app(mock_repo, auth_user_dict=_AUTH_USER)
        resp = TestClient(_app).put("/users/me/profile", json={"email": "taken@example.com"})
        assert resp.status_code == 409

    def test_user_not_found(self, mock_repo):
        mock_repo.get.return_value = None
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).put("/users/me/profile", json={"display_name": "X"})
        assert resp.status_code == 404


# ── Get profile ──────────────────────────────────────────────────────────────

class TestGetProfile:
    def test_get_own_profile(self, mock_repo, regular_user):
        mock_repo.get.return_value = regular_user
        _app, _ = _build_app(mock_repo, auth_user_dict=_AUTH_USER)
        resp = TestClient(_app).get("/users/me/profile")
        assert resp.status_code == 200
        assert resp.json()["data"]["username"] == "alice"

    def test_user_not_found(self, mock_repo):
        mock_repo.get.return_value = None
        _app, _ = _build_app(mock_repo)
        resp = TestClient(_app).get("/users/me/profile")
        assert resp.status_code == 404
