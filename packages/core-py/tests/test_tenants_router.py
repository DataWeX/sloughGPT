"""
Tenants Router Tests
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


def _make_user(uid="admin1", role="admin", tenant_id="t1"):
    u = MagicMock()
    u.id = uid
    u.role.value = role
    u.role = role
    u.tenant_id = tenant_id
    u.is_admin = role in ("admin", "owner")
    return u


def _make_tenant(tid="t1", name="Acme", slug="acme", plan="free"):
    t = MagicMock()
    t.id = tid
    t.name = name
    t.slug = slug
    t.plan = plan
    t.max_users = 5
    t.max_workspaces = 3
    t.created_at = "2026-01-01T00:00:00Z"
    t.updated_at = "2026-01-01T00:00:00Z"
    return t


@pytest.fixture
def app_with_repos():
    app = FastAPI()
    from infrastructure.exception_handlers import register_app_error_handler
    register_app_error_handler(app)

    mock_tenant_repo = MagicMock()
    mock_user_repo = MagicMock()

    with patch("routers.tenants.TenantRepository", return_value=mock_tenant_repo), \
         patch("routers.tenants.UserRepository", return_value=mock_user_repo):
        from routers.tenants import TenantsRouter
        router_obj = TenantsRouter()
        app.include_router(router_obj.router)

    return app, mock_tenant_repo, mock_user_repo


def _make_client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListTenants:
    def test_owner_sees_all(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        owner = _make_user("owner1", "owner")
        user_repo.get.return_value = owner
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "owner1"}

        t1 = _make_tenant("t1", "Acme", "acme")
        t2 = _make_tenant("t2", "Globex", "globex", "pro")
        tenant_repo.list_all.return_value = [t1, t2]

        client = _make_client(app)
        resp = client.get("/tenants")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert len(body["data"]) == 2

    def test_admin_sees_own_tenant(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin", "t1")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}

        t1 = _make_tenant("t1", "Acme", "acme")
        tenant_repo.get.return_value = t1

        client = _make_client(app)
        resp = client.get("/tenants")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 1


class TestGetTenant:
    def test_owner_can_get_any(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        owner = _make_user("owner1", "owner")
        user_repo.get.return_value = owner
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "owner1"}

        t2 = _make_tenant("t2", "Globex", "globex")
        tenant_repo.get.return_value = t2

        client = _make_client(app)
        resp = client.get("/tenants/t2")
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "Globex"

    def test_admin_cannot_get_other_tenant(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin", "t1")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}

        client = _make_client(app)
        resp = client.get("/tenants/t2")
        assert resp.status_code == 403

    def test_not_found(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        owner = _make_user("owner1", "owner")
        user_repo.get.return_value = owner
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "owner1"}
        tenant_repo.get.return_value = None

        client = _make_client(app)
        resp = client.get("/tenants/nonexistent")
        assert resp.status_code == 404


class TestCreateTenant:
    def test_creates_tenant(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}
        tenant_repo.get_by_slug.return_value = None

        client = _make_client(app)
        resp = client.post("/tenants", json={
            "name": "New Org",
            "slug": "new-org",
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "New Org"
        tenant_repo.create.assert_called_once()

    def test_duplicate_slug(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}
        tenant_repo.get_by_slug.return_value = _make_tenant("t1", "Existing", "existing")

        client = _make_client(app)
        resp = client.post("/tenants", json={
            "name": "New Org",
            "slug": "existing",
        })
        assert resp.status_code == 409


class TestUpdateTenant:
    def test_updates_name(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}

        t1 = _make_tenant("t1", "Acme", "acme")
        tenant_repo.get.return_value = t1

        client = _make_client(app)
        resp = client.put("/tenants/t1", json={"name": "Acme Corp"})
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "Acme Corp"
        tenant_repo.update.assert_called_once()

    def test_not_found(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}
        tenant_repo.get.return_value = None

        client = _make_client(app)
        resp = client.put("/tenants/nonexistent", json={"name": "X"})
        assert resp.status_code == 404


class TestDeleteTenant:
    def test_owner_can_delete(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        owner = _make_user("owner1", "owner")
        user_repo.get.return_value = owner
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "owner1"}

        t1 = _make_tenant("t1", "Acme", "acme")
        tenant_repo.get.return_value = t1

        client = _make_client(app)
        resp = client.delete("/tenants/t1")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True
        tenant_repo.delete.assert_called_once_with("t1")

    def test_admin_cannot_delete(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}

        client = _make_client(app)
        resp = client.delete("/tenants/t1")
        assert resp.status_code == 403


class TestTenantStats:
    def test_returns_stats(self, app_with_repos):
        app, tenant_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin", "t1")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}

        t1 = _make_tenant("t1", "Acme", "acme")
        tenant_repo.get.return_value = t1
        user_repo.count.return_value = 3

        client = _make_client(app)
        resp = client.get("/tenants/t1/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["user_count"] == 3
        assert data["max_users"] == 5
        assert data["plan"] == "free"
