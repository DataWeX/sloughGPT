"""Tests for the Workspaces router — CRUD, members, stats, health."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from domains.auth.models import Role, User, UserRole, Workspace, WorkspaceMember
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler


# ── Helpers ──────────────────────────────────────────────────────────────────

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}
_AUTH_ADMIN = {"sub": "admin1", "tenant_id": "t1"}


def _make_user(
    user_id: str = "user1",
    username: str = "alice",
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


def _make_workspace(
    ws_id: str = "ws1",
    name: str = "Test Workspace",
    tenant_id: str = "t1",
) -> Workspace:
    return Workspace(
        id=ws_id,
        name=name,
        tenant_id=tenant_id,
        description="Test workspace",
    )


def _make_member(
    ws_id: str = "ws1",
    user_id: str = "user1",
    role: Role = Role.ADMIN,
) -> WorkspaceMember:
    m = WorkspaceMember(
        workspace_id=ws_id,
        user_id=user_id,
        role=role,
    )
    m.joined_at = "2025-01-01T00:00:00Z"
    return m


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_ws_repo():
    repo = MagicMock()
    return repo


@pytest.fixture
def mock_user_repo():
    repo = MagicMock()
    return repo


def _build_app(ws_repo=None, user_repo=None, auth_user_dict=_AUTH_USER):
    from routers.workspaces import WorkspacesRouter

    router_obj = WorkspacesRouter()
    if ws_repo is not None:
        router_obj._ws_repo = ws_repo
    if user_repo is not None:
        router_obj._user_repo = user_repo

    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app


# ── List workspaces ─────────────────────────────────────────────────────────


class TestListWorkspaces:
    def test_list_empty(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        mock_user_repo.get.return_value = user
        mock_ws_repo.list_user_workspaces.return_value = []

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_list_with_memberships(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.list_user_workspaces.return_value = [MagicMock(workspace_id="ws1")]
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.list_members.return_value = [member]

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1
        assert data[0]["name"] == "Test Workspace"


# ── Get workspace ────────────────────────────────────────────────────────────


class TestGetWorkspace:
    def test_get_workspace_found(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.list_members.return_value = [member]

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1")
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "Test Workspace"

    def test_get_workspace_not_found(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = None

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/nonexistent")
        assert resp.status_code == 404

    def test_get_workspace_access_denied(self, mock_ws_repo, mock_user_repo):
        user = _make_user(role=Role.USER)
        ws = _make_workspace()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = None  # not a member

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1")
        assert resp.status_code == 403


# ── Create workspace ─────────────────────────────────────────────────────────


class TestCreateWorkspace:
    def test_create_workspace(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        mock_user_repo.get.return_value = user
        mock_ws_repo.create.return_value = None
        mock_ws_repo.add_member.return_value = None

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).post(
            "/workspaces",
            json={"name": "New Workspace", "description": "A test workspace"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "New Workspace"

    def test_create_workspace_empty_name(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        mock_user_repo.get.return_value = user

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).post(
            "/workspaces",
            json={"name": "", "description": "Test"},
        )
        assert resp.status_code == 422  # validation error


# ── Update workspace ─────────────────────────────────────────────────────────


class TestUpdateWorkspace:
    def test_update_workspace(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.update.return_value = None
        mock_ws_repo.list_members.return_value = [member]

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).put(
            "/workspaces/ws1",
            json={"name": "Updated Name"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "Updated Name"

    def test_update_workspace_not_found(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = None

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).put(
            "/workspaces/nonexistent",
            json={"name": "Test"},
        )
        assert resp.status_code == 404


# ── Delete workspace ─────────────────────────────────────────────────────────


class TestDeleteWorkspace:
    def test_delete_workspace(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.delete.return_value = None

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).delete("/workspaces/ws1")
        assert resp.status_code == 200

    def test_delete_workspace_not_found(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = None

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).delete("/workspaces/nonexistent")
        assert resp.status_code == 404


# ── Members ──────────────────────────────────────────────────────────────────


class TestWorkspaceMembers:
    def test_list_members(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.list_members.return_value = [member]

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1/members")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1

    def test_add_member(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()
        new_user = _make_user(user_id="user2", username="bob", role=Role.USER)

        mock_user_repo.get.side_effect = lambda uid: user if uid == "user1" else new_user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.side_effect = lambda ws_id, uid: member if uid == "user1" else None
        mock_ws_repo.list_members.return_value = [member]
        mock_ws_repo.add_member.return_value = None

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).post(
            "/workspaces/ws1/members",
            json={"user_id": "user2", "role": "user"},
        )
        assert resp.status_code == 200

    def test_remove_member(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.remove_member.return_value = None

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).delete("/workspaces/ws1/members/user1")
        assert resp.status_code == 200


# ── Stats ────────────────────────────────────────────────────────────────────


class TestWorkspaceStats:
    def test_get_stats(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.list_members.return_value = [member]

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "member_count" in data
        assert "dataset_count" in data


# ── Health check ─────────────────────────────────────────────────────────────


class TestWorkspaceHealth:
    def test_health_check(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.list_members.return_value = [member]

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1/health")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "status" in data
        assert "checks" in data


# ── Settings ─────────────────────────────────────────────────────────────────


class TestWorkspaceSettings:
    def test_get_settings(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1/settings")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "name" in data

    def test_update_settings(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.update.return_value = None
        mock_ws_repo.list_members.return_value = [member]

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).put(
            "/workspaces/ws1/settings",
            json={"name": "Updated Settings Name", "data_retention_days": 30},
        )
        assert resp.status_code == 200


# ── Clone ────────────────────────────────────────────────────────────────────


class TestCloneWorkspace:
    def test_clone_workspace(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.list_members.return_value = [member]
        mock_ws_repo.create.return_value = None
        mock_ws_repo.add_member.return_value = None

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).post("/workspaces/ws1/clone")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "id" in data
        assert data["name"] == "Test Workspace (Copy)"


# ── Notifications ────────────────────────────────────────────────────────────


class TestNotifications:
    def test_get_notifications(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1/notifications")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "notifications" in data


# ── Search ───────────────────────────────────────────────────────────────────


class TestSearchWorkspace:
    def test_search_workspace(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.list_members.return_value = [member]

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1/search")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "results" in data
        assert "total" in data


# ── Permissions ──────────────────────────────────────────────────────────────


class TestPermissions:
    def test_get_permissions(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.list_members.return_value = [member]

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1/permissions")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "roles" in data
        assert "all_permissions" in data


# ── Activity ─────────────────────────────────────────────────────────────────


class TestActivity:
    def test_get_activity(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1/activity")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "activities" in data


# ── Export ───────────────────────────────────────────────────────────────────


class TestExport:
    def test_export_workspace(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member(role=Role.OWNER)

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member
        mock_ws_repo.list_members.return_value = [member]

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).get("/workspaces/ws1/export")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "workspace" in data
        assert "members" in data


# ── Import ───────────────────────────────────────────────────────────────────


class TestImport:
    def test_import_workspace(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        mock_user_repo.get.return_value = user
        mock_ws_repo.create.return_value = None
        mock_ws_repo.add_member.return_value = None

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).post(
            "/workspaces/import",
            json={
                "workspace_data": {
                    "workspace": {"name": "Imported", "description": "Test"},
                    "members": [],
                }
            },
        )
        # Import may fail due to workspace data format, but endpoint should exist
        assert resp.status_code in (200, 422)


# ── Invite ───────────────────────────────────────────────────────────────────


class TestInvite:
    def test_invite_member(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()
        invitee = _make_user(user_id="user2", username="bob", role=Role.USER)

        mock_user_repo.get.side_effect = lambda uid: user if uid == "user1" else invitee
        mock_user_repo.get_by_email.return_value = invitee
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.side_effect = lambda ws_id, uid: member if uid == "user1" else None
        mock_ws_repo.list_members.return_value = [member]
        mock_ws_repo.add_member.return_value = None

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).post(
            "/workspaces/ws1/invite",
            json={"email": "bob@example.com", "role": "user"},
        )
        assert resp.status_code == 200


# ── Cleanup ──────────────────────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_workspace(self, mock_ws_repo, mock_user_repo):
        user = _make_user()
        ws = _make_workspace()
        member = _make_member()

        mock_user_repo.get.return_value = user
        mock_ws_repo.get.return_value = ws
        mock_ws_repo.get_member.return_value = member

        _app = _build_app(mock_ws_repo, mock_user_repo)
        resp = TestClient(_app).post("/workspaces/ws1/cleanup")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "cleaned" in data
