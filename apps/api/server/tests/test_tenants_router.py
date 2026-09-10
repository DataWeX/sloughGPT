"""Tests for the Tenants router — 6 endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

from domains.auth.models import Role, Tenant, User, UserRole
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler


_AUTH_ADMIN = {"sub": "admin1", "tenant_id": "t1"}
_AUTH_OWNER = {"sub": "owner1", "tenant_id": "t1"}


def _make_user(
    user_id: str = "admin1",
    username: str = "admin",
    role: Role = Role.ADMIN,
    tenant_id: str = "t1",
) -> User:
    return User(
        id=user_id,
        username=username,
        email=f"{username}@example.com",
        password_hash="hashed",
        display_name=username.title(),
        role=role,
        status=UserRole.ACTIVE,
        tenant_id=tenant_id,
    )


def _make_tenant(
    tenant_id: str = "t1",
    name: str = "Test Tenant",
    slug: str = "test-tenant",
) -> Tenant:
    return Tenant(
        id=tenant_id,
        name=name,
        slug=slug,
        plan="free",
        max_users=5,
        max_workspaces=3,
    )


def _build_app(user_repo=None, tenant_repo=None, auth_user_dict=_AUTH_ADMIN):
    from routers.tenants import TenantsRouter

    router_obj = TenantsRouter()
    if user_repo is not None:
        router_obj._user_repo = user_repo
    if tenant_repo is not None:
        router_obj._repo = tenant_repo

    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app


# ── List tenants ─────────────────────────────────────────────────────────────


class TestListTenants:
    def test_list_as_admin(self):
        user = _make_user()
        tenant = _make_tenant()
        user_repo = MagicMock()
        user_repo.get.return_value = user
        tenant_repo = MagicMock()
        # Admin (non-owner) sees only their own tenant via get()
        tenant_repo.get.return_value = tenant

        _app = _build_app(user_repo, tenant_repo)
        resp = TestClient(_app).get("/tenants")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1

    def test_list_as_owner_sees_all(self):
        user = _make_user(role=Role.OWNER)
        tenant = _make_tenant()
        user_repo = MagicMock()
        user_repo.get.return_value = user
        tenant_repo = MagicMock()
        tenant_repo.list_all.return_value = [tenant]

        _app = _build_app(user_repo, tenant_repo, auth_user_dict=_AUTH_OWNER)
        resp = TestClient(_app).get("/tenants")
        assert resp.status_code == 200


# ── Get tenant ───────────────────────────────────────────────────────────────


class TestGetTenant:
    def test_get_tenant_found(self):
        user = _make_user()
        tenant = _make_tenant()
        user_repo = MagicMock()
        user_repo.get.return_value = user
        tenant_repo = MagicMock()
        tenant_repo.get.return_value = tenant

        _app = _build_app(user_repo, tenant_repo)
        resp = TestClient(_app).get("/tenants/t1")
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "Test Tenant"

    def test_get_tenant_not_found(self):
        user = _make_user()
        user_repo = MagicMock()
        user_repo.get.return_value = user
        tenant_repo = MagicMock()
        tenant_repo.get.return_value = None

        _app = _build_app(user_repo, tenant_repo)
        resp = TestClient(_app).get("/tenants/nonexistent")
        # Returns 403 because admin check fails (admin.tenant_id != "nonexistent")
        assert resp.status_code in (403, 404)


# ── Create tenant ────────────────────────────────────────────────────────────


class TestCreateTenant:
    def test_create_tenant(self):
        user = _make_user()
        user_repo = MagicMock()
        user_repo.get.return_value = user
        tenant_repo = MagicMock()
        tenant_repo.get_by_slug.return_value = None
        tenant_repo.create.return_value = None

        _app = _build_app(user_repo, tenant_repo)
        resp = TestClient(_app).post(
            "/tenants",
            json={"name": "New Tenant", "slug": "new-tenant"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "New Tenant"

    def test_create_tenant_duplicate_slug(self):
        user = _make_user()
        existing = _make_tenant()
        user_repo = MagicMock()
        user_repo.get.return_value = user
        tenant_repo = MagicMock()
        tenant_repo.get_by_slug.return_value = existing

        _app = _build_app(user_repo, tenant_repo)
        resp = TestClient(_app).post(
            "/tenants",
            json={"name": "New Tenant", "slug": "test-tenant"},
        )
        assert resp.status_code == 409


# ── Update tenant ────────────────────────────────────────────────────────────


class TestUpdateTenant:
    def test_update_tenant(self):
        user = _make_user()
        tenant = _make_tenant()
        user_repo = MagicMock()
        user_repo.get.return_value = user
        tenant_repo = MagicMock()
        tenant_repo.get.return_value = tenant
        tenant_repo.update.return_value = None

        _app = _build_app(user_repo, tenant_repo)
        resp = TestClient(_app).put(
            "/tenants/t1",
            json={"name": "Updated Tenant"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "Updated Tenant"


# ── Delete tenant ────────────────────────────────────────────────────────────


class TestDeleteTenant:
    def test_delete_as_owner(self):
        user = _make_user(role=Role.OWNER)
        tenant = _make_tenant()
        user_repo = MagicMock()
        user_repo.get.return_value = user
        tenant_repo = MagicMock()
        tenant_repo.get.return_value = tenant
        tenant_repo.delete.return_value = None

        _app = _build_app(user_repo, tenant_repo, auth_user_dict=_AUTH_OWNER)
        resp = TestClient(_app).delete("/tenants/t1")
        assert resp.status_code == 200

    def test_delete_as_admin_denied(self):
        user = _make_user(role=Role.ADMIN)
        tenant = _make_tenant()
        user_repo = MagicMock()
        user_repo.get.return_value = user
        tenant_repo = MagicMock()
        tenant_repo.get.return_value = tenant

        _app = _build_app(user_repo, tenant_repo)
        resp = TestClient(_app).delete("/tenants/t1")
        assert resp.status_code == 403


# ── Stats ────────────────────────────────────────────────────────────────────


class TestTenantStats:
    def test_get_stats(self):
        user = _make_user()
        tenant = _make_tenant()
        user_repo = MagicMock()
        user_repo.get.return_value = user
        user_repo.count.return_value = 3
        tenant_repo = MagicMock()
        tenant_repo.get.return_value = tenant

        _app = _build_app(user_repo, tenant_repo)
        resp = TestClient(_app).get("/tenants/t1/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["user_count"] == 3
        assert data["plan"] == "free"
