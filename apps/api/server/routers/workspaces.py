"""Workspaces API — workspace management + member endpoints.

Provides CRUD operations for workspaces and their member assignments.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from domains.auth.models import Role, User, Workspace, WorkspaceMember
from domains.auth.repositories import UserRepository, WorkspaceRepository
from infrastructure.auth import require_auth_if_enabled
from schemas.common import classify_and_raise, raise_error, success_response

logger = logging.getLogger("slo.workspaces")


# ─── Request / Response models ─────────────────────────────────


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    tenant_id: str = Field(default="", description="Tenant ID (auto-assigned if empty)")


class WorkspaceUpdateRequest(BaseModel):
    name: str | None = Field(None, max_length=200)
    description: str | None = Field(None, max_length=1000)


class MemberAddRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    role: str = Field(default="user", description="Role: viewer, user, admin")


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    tenant_id: str
    description: str
    created_at: str
    member_count: int = 0


class MemberResponse(BaseModel):
    id: str
    workspace_id: str
    user_id: str
    username: str = ""
    role: str
    created_at: str


# ─── Router ────────────────────────────────────────────────────


class WorkspacesRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/workspaces", tags=["workspaces"])
        self._ws_repo = WorkspaceRepository()
        self._user_repo = UserRepository()
        self._register_routes()

    def _get_user(self, auth_user: dict | None) -> User:
        if not auth_user:
            raise_error("Authentication required", "E_AUTH_MISSING", status_code=401)
        user = self._user_repo.get(auth_user.get("sub", ""))
        if not user:
            raise_error("User not found", "E_NOT_FOUND", status_code=404)
        return user

    def _to_response(self, ws: Workspace, member_count: int = 0) -> WorkspaceResponse:
        return WorkspaceResponse(
            id=ws.id,
            name=ws.name,
            tenant_id=ws.tenant_id,
            description=ws.description,
            created_at=ws.created_at,
            member_count=member_count,
        )

    def _register_routes(self):
        router = self.router
        auth_dep = Depends(require_auth_if_enabled)

        # ─── List workspaces ───────────────────────────────────
        async def list_workspaces(auth_user: dict = auth_dep) -> dict:
            user = self._get_user(auth_user)
            # List workspaces the user is a member of
            memberships = self._ws_repo.list_user_workspaces(user.id)
            ws_ids = [m.workspace_id for m in memberships]
            workspaces = []
            for ws_id in ws_ids:
                ws = self._ws_repo.get(ws_id)
                if ws:
                    members = self._ws_repo.list_members(ws_id)
                    workspaces.append(self._to_response(ws, len(members)).model_dump())
            # Also include tenant workspaces if user is admin
            if user.is_admin and user.tenant_id:
                tenant_ws = self._ws_repo.list_by_tenant(user.tenant_id)
                existing_ids = set(ws_ids)
                for ws in tenant_ws:
                    if ws.id not in existing_ids:
                        members = self._ws_repo.list_members(ws.id)
                        workspaces.append(self._to_response(ws, len(members)).model_dump())
            return success_response(
                data=workspaces,
                meta={"total": len(workspaces)},
            )

        # ─── Get workspace ─────────────────────────────────────
        async def get_workspace(workspace_id: str, auth_user: dict = auth_dep) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            # Check membership
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)
            members = self._ws_repo.list_members(workspace_id)
            return success_response(data=self._to_response(ws, len(members)).model_dump())

        # ─── Create workspace ──────────────────────────────────
        async def create_workspace(req: WorkspaceCreateRequest, auth_user: dict = auth_dep) -> dict:
            user = self._get_user(auth_user)
            tenant_id = req.tenant_id or user.tenant_id
            if not tenant_id:
                raise_error("Tenant ID required", "E_INVALID_INPUT", status_code=400)

            ws = Workspace(
                id=str(uuid.uuid4()),
                name=req.name,
                tenant_id=tenant_id,
                description=req.description,
            )
            self._ws_repo.create(ws)

            # Auto-add creator as admin member
            member = WorkspaceMember(
                id=str(uuid.uuid4()),
                workspace_id=ws.id,
                user_id=user.id,
                role=Role.ADMIN,
            )
            self._ws_repo.add_member(member)

            logger.info("User %s created workspace %s", user.username, req.name)
            return success_response(data=self._to_response(ws, 1).model_dump())

        # ─── Update workspace ──────────────────────────────────
        async def update_workspace(
            workspace_id: str, req: WorkspaceUpdateRequest, auth_user: dict = auth_dep
        ) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            # Check membership or admin
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member or member.role not in (Role.ADMIN, Role.OWNER):
                if not user.is_admin:
                    raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)

            if req.name is not None:
                ws.name = req.name
            if req.description is not None:
                ws.description = req.description

            ws.updated_at = datetime.now(timezone.utc).isoformat()
            self._ws_repo.update(ws)
            members = self._ws_repo.list_members(workspace_id)
            return success_response(data=self._to_response(ws, len(members)).model_dump())

        # ─── Delete workspace ──────────────────────────────────
        async def delete_workspace(workspace_id: str, auth_user: dict = auth_dep) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            # Only owner or tenant admin can delete
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member or member.role != Role.OWNER:
                if not user.is_admin:
                    raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)
            self._ws_repo.delete(workspace_id)
            logger.info("User %s deleted workspace %s", user.username, workspace_id)
            return success_response(data={"deleted": True})

        # ─── Add member ────────────────────────────────────────
        async def add_member(
            workspace_id: str, req: MemberAddRequest, auth_user: dict = auth_dep
        ) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            # Check if caller is admin or workspace admin
            caller_member = self._ws_repo.get_member(workspace_id, user.id)
            if not caller_member or caller_member.role not in (Role.ADMIN, Role.OWNER):
                if not user.is_admin:
                    raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)

            target_user = self._user_repo.get(req.user_id)
            if not target_user:
                raise_error("User not found", "E_NOT_FOUND", status_code=404)

            existing = self._ws_repo.get_member(workspace_id, req.user_id)
            if existing:
                raise_error("User already a member", "E_INFRA_BUSY", status_code=409)

            role = Role(req.role) if req.role in [r.value for r in Role] else Role.USER
            member = WorkspaceMember(
                id=str(uuid.uuid4()),
                workspace_id=workspace_id,
                user_id=req.user_id,
                role=role,
            )
            self._ws_repo.add_member(member)
            return success_response(
                data=MemberResponse(
                    id=member.id,
                    workspace_id=member.workspace_id,
                    user_id=member.user_id,
                    username=target_user.username,
                    role=member.role.value,
                    created_at=member.created_at,
                ).model_dump()
            )

        # ─── Remove member ─────────────────────────────────────
        async def remove_member(
            workspace_id: str, user_id: str, auth_user: dict = auth_dep
        ) -> dict:
            user = self._get_user(auth_user)
            # Can remove self (leave) or others if admin
            if user.id != user_id:
                caller_member = self._ws_repo.get_member(workspace_id, user.id)
                if not caller_member or caller_member.role not in (Role.ADMIN, Role.OWNER):
                    if not user.is_admin:
                        raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)
            self._ws_repo.remove_member(workspace_id, user_id)
            return success_response(data={"removed": True})

        # ─── List members ──────────────────────────────────────
        async def list_members(workspace_id: str, auth_user: dict = auth_dep) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)

            members = self._ws_repo.list_members(workspace_id)
            result = []
            for m in members:
                u = self._user_repo.get(m.user_id)
                result.append(
                    MemberResponse(
                        id=m.id,
                        workspace_id=m.workspace_id,
                        user_id=m.user_id,
                        username=u.username if u else "",
                        role=m.role.value,
                        created_at=m.created_at,
                    ).model_dump()
                )
            return success_response(data=result, meta={"total": len(result)})

        router.add_api_route("", list_workspaces, methods=["GET"])
        router.add_api_route("/{workspace_id}", get_workspace, methods=["GET"])
        router.add_api_route("", create_workspace, methods=["POST"])
        router.add_api_route("/{workspace_id}", update_workspace, methods=["PUT"])
        router.add_api_route("/{workspace_id}", delete_workspace, methods=["DELETE"])
        router.add_api_route("/{workspace_id}/members", list_members, methods=["GET"])
        router.add_api_route("/{workspace_id}/members", add_member, methods=["POST"])
        router.add_api_route(
            "/{workspace_id}/members/{user_id}", remove_member, methods=["DELETE"]
        )


# ─── Singleton ─────────────────────────────────────────────────

_workspaces_router: WorkspacesRouter | None = None


def get_workspaces_router() -> WorkspacesRouter:
    global _workspaces_router
    if _workspaces_router is None:
        _workspaces_router = WorkspacesRouter()
    return _workspaces_router
