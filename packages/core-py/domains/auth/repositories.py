"""User, Tenant, Workspace repositories — persistence layer."""

from __future__ import annotations

import logging
import os
from typing import Any

from domains.auth.models import (
    Role,
    User,
    UserRole,
    Tenant,
    Workspace,
    WorkspaceMember,
)

logger = logging.getLogger("slo.auth.repo")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _get_mogdb(db_path: str | None = None):
    from mogdb import MogDB
    if db_path is None:
        db_path = os.path.join(_REPO_ROOT, "data", "auth_mogdb")
    return MogDB(db_path)


class UserRepository:
    """MogDB-backed user repository."""

    def __init__(self, db_path: str | None = None):
        self._db = _get_mogdb(db_path)
        self._col = self._db.collection("users")
        self._col.create_index("username")
        self._col.create_index("email")
        self._col.create_index("tenant_id")

    def get(self, user_id: str) -> User | None:
        doc = self._col.find_one({"_id": user_id})
        return User.from_dict(doc) if doc else None

    def get_by_username(self, username: str) -> User | None:
        doc = self._col.find_one({"username": username})
        return User.from_dict(doc) if doc else None

    def get_by_email(self, email: str) -> User | None:
        doc = self._col.find_one({"email": email})
        return User.from_dict(doc) if doc else None

    def list_by_tenant(self, tenant_id: str) -> list[User]:
        return [
            User.from_dict(d)
            for d in self._col.find({"tenant_id": tenant_id})
        ]

    def create(self, user: User) -> User:
        existing = self.get(user.id)
        if existing:
            self._col.update_one({"_id": user.id}, {"$set": user.to_dict()})
        else:
            self._col.insert_one(user.to_dict())
        return user

    def update(self, user: User) -> User:
        self._col.update_one({"_id": user.id}, {"$set": user.to_dict()})
        return user

    def delete(self, user_id: str) -> bool:
        return self._col.delete_one({"_id": user_id})

    def count(self, tenant_id: str | None = None) -> int:
        query = {"tenant_id": tenant_id} if tenant_id else {}
        return len(self._col.find(query))


class TenantRepository:
    """MogDB-backed tenant repository."""

    def __init__(self, db_path: str | None = None):
        self._db = _get_mogdb(db_path)
        self._col = self._db.collection("tenants")
        self._col.create_index("slug")

    def get(self, tenant_id: str) -> Tenant | None:
        doc = self._col.find_one({"_id": tenant_id})
        return Tenant.from_dict(doc) if doc else None

    def get_by_slug(self, slug: str) -> Tenant | None:
        doc = self._col.find_one({"slug": slug})
        return Tenant.from_dict(doc) if doc else None

    def list_all(self) -> list[Tenant]:
        return [Tenant.from_dict(d) for d in self._col.find()]

    def create(self, tenant: Tenant) -> Tenant:
        self._col.insert_one(tenant.to_dict())
        return tenant

    def update(self, tenant: Tenant) -> Tenant:
        self._col.update_one({"_id": tenant.id}, {"$set": tenant.to_dict()})
        return tenant

    def delete(self, tenant_id: str) -> bool:
        return self._col.delete_one({"_id": tenant_id})


class WorkspaceRepository:
    """MogDB-backed workspace repository."""

    def __init__(self, db_path: str | None = None):
        self._db = _get_mogdb(db_path)
        self._col = self._db.collection("workspaces")
        self._members_col = self._db.collection("workspace_members")
        self._col.create_index("tenant_id")
        self._members_col.create_index("workspace_id")
        self._members_col.create_index("user_id")

    def get(self, workspace_id: str) -> Workspace | None:
        doc = self._col.find_one({"_id": workspace_id})
        return Workspace.from_dict(doc) if doc else None

    def list_by_tenant(self, tenant_id: str) -> list[Workspace]:
        return [Workspace.from_dict(d) for d in self._col.find({"tenant_id": tenant_id})]

    def create(self, workspace: Workspace) -> Workspace:
        self._col.insert_one(workspace.to_dict())
        return workspace

    def update(self, workspace: Workspace) -> Workspace:
        self._col.update_one({"_id": workspace.id}, {"$set": workspace.to_dict()})
        return workspace

    def delete(self, workspace_id: str) -> bool:
        self._members_col.delete_one({"workspace_id": workspace_id})
        return self._col.delete_one({"_id": workspace_id})

    # ─── Members ────────────────────────────────────────────────

    def add_member(self, member: WorkspaceMember) -> WorkspaceMember:
        self._members_col.insert_one(member.to_dict())
        return member

    def remove_member(self, workspace_id: str, user_id: str) -> bool:
        return self._members_col.delete_one(
            {"workspace_id": workspace_id, "user_id": user_id}
        )

    def get_member(self, workspace_id: str, user_id: str) -> WorkspaceMember | None:
        doc = self._members_col.find_one(
            {"workspace_id": workspace_id, "user_id": user_id}
        )
        return WorkspaceMember.from_dict(doc) if doc else None

    def list_members(self, workspace_id: str) -> list[WorkspaceMember]:
        return [
            WorkspaceMember.from_dict(d)
            for d in self._members_col.find({"workspace_id": workspace_id})
        ]

    def list_user_workspaces(self, user_id: str) -> list[WorkspaceMember]:
        return [
            WorkspaceMember.from_dict(d)
            for d in self._members_col.find({"user_id": user_id})
        ]

    # ─── Invitations ────────────────────────────────────────────

    def create_invitation(
        self,
        workspace_id: str,
        email: str,
        role: str,
        invited_by: str,
    ) -> dict:
        """Create a workspace invitation."""
        import uuid
        import time

        inv_id = str(uuid.uuid4())
        doc = {
            "_id": inv_id,
            "workspace_id": workspace_id,
            "email": email,
            "role": role,
            "invited_by": invited_by,
            "created_at": int(time.time()),
            "accepted": False,
        }
        self._db.collection("workspace_invitations").insert_one(doc)
        return {
            "id": inv_id,
            "workspace_id": workspace_id,
            "email": email,
            "role": role,
            "invited_by": invited_by,
            "created_at": doc["created_at"],
            "accepted": False,
        }

    def list_invitations(self, workspace_id: str) -> list[dict]:
        """List invitations for a workspace."""
        col = self._db.collection("workspace_invitations")
        return [
            {
                "id": d["_id"],
                "workspace_id": d["workspace_id"],
                "email": d["email"],
                "role": d["role"],
                "invited_by": d["invited_by"],
                "created_at": d["created_at"],
                "accepted": d.get("accepted", False),
            }
            for d in col.find({"workspace_id": workspace_id})
        ]

    def revoke_invitation(self, workspace_id: str, invitation_id: str) -> bool:
        """Revoke a pending invitation."""
        col = self._db.collection("workspace_invitations")
        return col.delete_one({"_id": invitation_id, "workspace_id": workspace_id})
