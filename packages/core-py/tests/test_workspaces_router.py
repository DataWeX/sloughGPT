"""
Workspaces Router Tests
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


def _make_user(uid="u1", role="admin", tenant_id="t1"):
    u = MagicMock()
    u.id = uid
    u.role.value = role
    u.tenant_id = tenant_id
    u.is_admin = role in ("admin", "owner")
    u.username = uid
    return u


def _make_workspace(wid="ws1", name="My Workspace", tenant_id="t1"):
    ws = MagicMock()
    ws.id = wid
    ws.name = name
    ws.tenant_id = tenant_id
    ws.description = f"Desc for {name}"
    ws.created_at = "2026-01-01T00:00:00Z"
    ws.updated_at = "2026-01-01T00:00:00Z"
    return ws


def _make_member(ws_id="ws1", user_id="u1", role="admin"):
    m = MagicMock()
    m.id = f"m_{user_id}"
    m.workspace_id = ws_id
    m.user_id = user_id
    m.role = role
    m.created_at = "2026-01-01T00:00:00Z"
    return m


@pytest.fixture
def app_with_repos():
    app = FastAPI()
    from infrastructure.exception_handlers import register_app_error_handler
    register_app_error_handler(app)

    mock_ws_repo = MagicMock()
    mock_user_repo = MagicMock()

    with patch("routers.workspaces.WorkspaceRepository", return_value=mock_ws_repo), \
         patch("routers.workspaces.UserRepository", return_value=mock_user_repo):
        from routers.workspaces import WorkspacesRouter
        router_obj = WorkspacesRouter()
        app.include_router(router_obj.router)

    return app, mock_ws_repo, mock_user_repo


def _make_client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListWorkspaces:
    def test_lists_user_workspaces(self, app_with_repos):
        app, ws_repo, user_repo = app_with_repos
        user = _make_user("u1", "user")
        user_repo.get.return_value = user
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "u1"}

        ws1 = _make_workspace("ws1", "Workspace 1")
        member = _make_member("ws1", "u1")
        ws_repo.list_user_workspaces.return_value = [member]
        ws_repo.get.return_value = ws1
        ws_repo.list_members.return_value = [member]

        client = _make_client(app)
        resp = client.get("/workspaces")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert len(body["data"]) == 1

    def test_admin_sees_tenant_workspaces(self, app_with_repos):
        app, ws_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin", "t1")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}

        # No user memberships
        ws_repo.list_user_workspaces.return_value = []
        # Tenant has workspaces
        ws1 = _make_workspace("ws1", "Tenant WS")
        ws_repo.list_by_tenant.return_value = [ws1]
        ws_repo.list_members.return_value = []

        client = _make_client(app)
        resp = client.get("/workspaces")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 1


class TestGetWorkspace:
    def test_member_can_get(self, app_with_repos):
        app, ws_repo, user_repo = app_with_repos
        user = _make_user("u1", "user")
        user_repo.get.return_value = user
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "u1"}

        ws1 = _make_workspace("ws1", "My Workspace")
        ws_repo.get.return_value = ws1
        member = _make_member("ws1", "u1")
        ws_repo.get_member.return_value = member
        ws_repo.list_members.return_value = [member]

        client = _make_client(app)
        resp = client.get("/workspaces/ws1")
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "My Workspace"

    def test_non_member_rejected(self, app_with_repos):
        app, ws_repo, user_repo = app_with_repos
        user = _make_user("u1", "user")
        user_repo.get.return_value = user
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "u1"}

        ws1 = _make_workspace("ws1")
        ws_repo.get.return_value = ws1
        ws_repo.get_member.return_value = None  # not a member

        client = _make_client(app)
        resp = client.get("/workspaces/ws1")
        assert resp.status_code == 403

    def test_not_found(self, app_with_repos):
        app, ws_repo, user_repo = app_with_repos
        user = _make_user("u1", "user")
        user_repo.get.return_value = user
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "u1"}
        ws_repo.get.return_value = None

        client = _make_client(app)
        resp = client.get("/workspaces/nonexistent")
        assert resp.status_code == 404


class TestCreateWorkspace:
    def test_creates_workspace(self, app_with_repos):
        app, ws_repo, user_repo = app_with_repos
        user = _make_user("u1", "admin", "t1")
        user_repo.get.return_value = user
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "u1"}

        client = _make_client(app)
        resp = client.post("/workspaces", json={"name": "New WS"})
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "New WS"
        ws_repo.create.assert_called_once()
        ws_repo.add_member.assert_called_once()


class TestUpdateWorkspace:
    def test_updates_name(self, app_with_repos):
        app, ws_repo, user_repo = app_with_repos
        user = _make_user("u1", "admin")
        user_repo.get.return_value = user
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "u1"}

        ws1 = _make_workspace("ws1", "Old Name")
        ws_repo.get.return_value = ws1
        member = _make_member("ws1", "u1", "admin")
        ws_repo.get_member.return_value = member
        ws_repo.list_members.return_value = [member]

        client = _make_client(app)
        resp = client.put("/workspaces/ws1", json={"name": "New Name"})
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "New Name"
        ws_repo.update.assert_called_once()


class TestDeleteWorkspace:
    def test_admin_can_delete(self, app_with_repos):
        app, ws_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}

        ws1 = _make_workspace("ws1")
        ws_repo.get.return_value = ws1
        ws_repo.get_member.return_value = None  # admin doesn't need membership

        client = _make_client(app)
        resp = client.delete("/workspaces/ws1")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True
        ws_repo.delete.assert_called_once_with("ws1")

    def test_not_found(self, app_with_repos):
        app, ws_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin")
        user_repo.get.return_value = admin
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}
        ws_repo.get.return_value = None

        client = _make_client(app)
        resp = client.delete("/workspaces/nonexistent")
        assert resp.status_code == 404


class TestAddMember:
    def test_adds_member(self, app_with_repos):
        app, ws_repo, user_repo = app_with_repos
        admin = _make_user("admin1", "admin")
        target = _make_user("u2", "user")
        user_repo.get.side_effect = lambda uid: admin if uid == "admin1" else target
        app.dependency_overrides[require_auth_if_enabled] = lambda: {"sub": "admin1"}

        ws1 = _make_workspace("ws1")
        ws_repo.get.return_value = ws1
        # First call: caller is admin member; second call: target is not a member
        caller_member = _make_member("ws1", "admin1", "admin")
        ws_repo.get_member.side_effect = lambda ws_id, uid: caller_member if uid == "admin1" else None

        client = _make_client(app)
        resp = client.post("/workspaces/ws1/members", json={"user_id": "u2", "role": "user"})
        assert resp.status_code == 200
        ws_repo.add_member.assert_called_once()
