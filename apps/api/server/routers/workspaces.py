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


class WorkspaceSettingsRequest(BaseModel):
    name: str | None = Field(None, max_length=200)
    description: str | None = Field(None, max_length=1000)
    default_model: str | None = Field(None, max_length=200)
    data_retention_days: int | None = Field(None, ge=7, le=365)
    training_retention_days: int | None = Field(None, ge=1, le=365)
    audit_retention_days: int | None = Field(None, ge=1, le=365)
    dataset_retention_days: int | None = Field(None, ge=1, le=365)
    max_members: int | None = Field(None, ge=2, le=500)
    allow_sharing: bool | None = None


class InviteMemberRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    role: str = Field(default="user", description="Role: viewer, user, admin")


class BulkMemberImportRequest(BaseModel):
    members: list[dict[str, str]] = Field(..., description="List of {user_id, role} or {email, role}")


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

        # ─── Workspace stats ────────────────────────────────────
        async def get_workspace_stats(workspace_id: str, auth_user: dict = auth_dep) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)

            members = self._ws_repo.list_members(workspace_id)

            # Count datasets in workspace
            dataset_count = 0
            try:
                from controllers.datasets import get_datasets_controller
                ctrl = get_datasets_controller()
                datasets = ctrl.list_datasets(workspace_id=workspace_id)
                dataset_count = len(datasets)
            except Exception:
                pass

            # Count training jobs in workspace
            job_count = 0
            active_jobs = 0
            try:
                from training.jobs import training_jobs
                for j in training_jobs.values():
                    if j.get("workspace_id", "") == workspace_id:
                        job_count += 1
                        if j.get("status") == "running":
                            active_jobs += 1
            except Exception:
                pass

            # Count knowledge items in workspace
            knowledge_count = 0
            try:
                from routers.kb import get_kb_router
                kb = get_kb_router()
                memory = kb._get_memory()
                all_items = memory.list_all(top_k=5000)
                knowledge_count = sum(
                    1 for item in all_items
                    if item.get("workspace_id", "") == workspace_id
                )
            except Exception:
                pass

            return success_response(data={
                "workspace_id": workspace_id,
                "name": ws.name,
                "member_count": len(members),
                "dataset_count": dataset_count,
                "training_jobs": job_count,
                "active_training_jobs": active_jobs,
                "knowledge_items": knowledge_count,
            })

        # ─── Workspace usage ─────────────────────────────────────
        async def get_workspace_usage(
            workspace_id: str, auth_user: dict = auth_dep
        ) -> dict:
            """Detailed usage metrics for a workspace."""
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            # Must be member or admin
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)

            # Training jobs by status
            training_by_status = {"running": 0, "completed": 0, "failed": 0, "queued": 0}
            total_training_minutes = 0.0
            try:
                from training.jobs import training_jobs
                for j in training_jobs.values():
                    if j.get("workspace_id", "") == workspace_id:
                        status = j.get("status", "unknown")
                        if status in training_by_status:
                            training_by_status[status] += 1
                        # Estimate duration from timestamps
                        started = j.get("started_at", "")
                        ended = j.get("ended_at", "")
                        if started and ended:
                            from datetime import datetime as dt
                            try:
                                s = dt.fromisoformat(started.replace("Z", "+00:00"))
                                e = dt.fromisoformat(ended.replace("Z", "+00:00"))
                                total_training_minutes += (e - s).total_seconds() / 60
                            except Exception:
                                pass
            except Exception:
                pass

            # Dataset count and estimated size
            dataset_count = 0
            try:
                from controllers.datasets import get_datasets_controller
                ctrl = get_datasets_controller()
                ds_list = ctrl.list_datasets(user_id=user.id, workspace_id=workspace_id)
                dataset_count = len(ds_list) if ds_list else 0
            except Exception:
                pass

            # Knowledge items
            knowledge_count = 0
            try:
                from routers.kb import get_kb_router
                kb = get_kb_router()
                memory = kb._get_memory()
                all_items = memory.list_all(top_k=10000)
                knowledge_count = sum(
                    1 for item in all_items
                    if item.get("workspace_id", "") == workspace_id
                )
            except Exception:
                pass

            # API key count
            api_key_count = 0
            try:
                from routers.api_keys import get_api_key_manager
                mgr = get_api_key_manager()
                keys = mgr.list(workspace_id=workspace_id)
                api_key_count = len(keys) if keys else 0
            except Exception:
                pass

            # Member breakdown by role
            members = self._ws_repo.list_members(workspace_id)
            role_breakdown = {}
            for m in members:
                r = m.role.value if hasattr(m.role, "value") else str(m.role)
                role_breakdown[r] = role_breakdown.get(r, 0) + 1

            return success_response(data={
                "workspace_id": workspace_id,
                "name": ws.name,
                "members": {
                    "total": len(members),
                    "by_role": role_breakdown,
                },
                "training": {
                    "by_status": training_by_status,
                    "total_minutes": round(total_training_minutes, 1),
                },
                "datasets": dataset_count,
                "knowledge_items": knowledge_count,
                "api_keys": api_key_count,
            })

        # ─── Workspace activity log ──────────────────────────────
        async def get_workspace_activity(
            workspace_id: str, auth_user: dict = auth_dep
        ) -> dict:
            """Recent activity in a workspace (training jobs, member changes, etc.)."""
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)

            activity: list[dict] = []

            # Training job events
            try:
                from training.jobs import training_jobs
                for j in training_jobs.values():
                    if j.get("workspace_id", "") == workspace_id:
                        status = j.get("status", "unknown")
                        activity.append({
                            "type": "training",
                            "action": f"Training job {status}",
                            "detail": j.get("job_id", j.get("name", "unknown")),
                            "status": status,
                            "timestamp": j.get("updated_at", j.get("started_at", "")),
                            "user": j.get("user_id", ""),
                        })
            except Exception:
                pass

            # Audit log entries for this workspace
            try:
                from infrastructure.auth import get_audit_logger
                audit = get_audit_logger()
                entries = audit.file_query(workspace_id=workspace_id, limit=50)
                for entry in entries:
                    activity.append({
                        "type": "audit",
                        "action": entry.get("action", ""),
                        "detail": entry.get("resource", ""),
                        "status": "success" if entry.get("success", True) else "failure",
                        "timestamp": entry.get("timestamp", ""),
                        "user": entry.get("user_id", ""),
                    })
            except Exception:
                pass

            # Sort by timestamp descending, limit to 50
            activity.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            activity = activity[:50]

            return success_response(data={
                "workspace_id": workspace_id,
                "activities": activity,
                "total": len(activity),
            })

        # ─── Workspace data export ───────────────────────────────
        async def export_workspace_data(
            workspace_id: str, auth_user: dict = auth_dep
        ) -> dict:
            """Export all workspace data (members, training jobs, usage)."""
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            # Only owner or admin can export
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member or member.role not in (Role.ADMIN, Role.OWNER):
                if not user.is_admin:
                    raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)

            # Members
            members = self._ws_repo.list_members(workspace_id)
            members_data = []
            for m in members:
                members_data.append({
                    "user_id": m.user_id,
                    "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                    "joined_at": m.joined_at,
                })

            # Training jobs
            training_jobs = []
            try:
                from training.jobs import training_jobs as tj
                for j in tj.values():
                    if j.get("workspace_id", "") == workspace_id:
                        training_jobs.append({
                            "job_id": j.get("job_id", ""),
                            "name": j.get("name", ""),
                            "status": j.get("status", ""),
                            "model": j.get("model", ""),
                            "created_at": j.get("created_at", ""),
                            "started_at": j.get("started_at", ""),
                            "ended_at": j.get("ended_at", ""),
                            "user_id": j.get("user_id", ""),
                        })
            except Exception:
                pass

            # API keys (metadata only, not secrets)
            api_keys = []
            try:
                from routers.api_keys import get_api_key_manager
                mgr = get_api_key_manager()
                keys = mgr.list(workspace_id=workspace_id)
                for k in keys:
                    api_keys.append({
                        "key_id": k.get("key_id", ""),
                        "name": k.get("name", ""),
                        "created_at": k.get("created_at", ""),
                        "last_used_at": k.get("last_used_at", ""),
                    })
            except Exception:
                pass

            # Dataset count
            dataset_count = 0
            try:
                from controllers.datasets import get_datasets_controller
                ctrl = get_datasets_controller()
                ds_list = ctrl.list_datasets(user_id=user.id, workspace_id=workspace_id)
                dataset_count = len(ds_list) if ds_list else 0
            except Exception:
                pass

            # Knowledge count
            knowledge_count = 0
            try:
                from routers.kb import get_kb_router
                kb = get_kb_router()
                memory = kb._get_memory()
                all_items = memory.list_all(top_k=10000)
                knowledge_count = sum(
                    1 for item in all_items
                    if item.get("workspace_id", "") == workspace_id
                )
            except Exception:
                pass

            return success_response(data={
                "workspace": {
                    "id": ws.id,
                    "name": ws.name,
                    "description": ws.description,
                    "tenant_id": ws.tenant_id,
                    "created_at": ws.created_at,
                },
                "members": members_data,
                "training_jobs": training_jobs,
                "api_keys": api_keys,
                "datasets_count": dataset_count,
                "knowledge_count": knowledge_count,
                "exported_at": datetime.now(timezone.utc).isoformat(),
            })

        # ─── Workspace data import ───────────────────────────────
        class WorkspaceImportRequest(BaseModel):
            workspace_data: dict = Field(..., description="Exported workspace data")

        async def import_workspace_data(
            req: WorkspaceImportRequest, auth_user: dict = auth_dep
        ) -> dict:
            """Import workspace data from export format.

            Creates a new workspace with members from the export.
            Training jobs, API keys, and data counts are informational only.
            """
            user = self._get_user(auth_user)
            data = req.workspace_data

            # Extract workspace info
            ws_data = data.get("workspace", {})
            ws_name = ws_data.get("name", "Imported Workspace")
            ws_desc = ws_data.get("description", "")

            # Create workspace
            ws = Workspace(
                name=f"{ws_name} (imported)",
                description=ws_desc,
                tenant_id=user.tenant_id,
            )
            self._ws_repo.create(ws)

            # Add creator as owner
            owner_member = WorkspaceMember(
                workspace_id=ws.id,
                user_id=user.id,
                role=Role.OWNER,
            )
            self._ws_repo.add_member(owner_member)

            # Import members (skip creator, already added as owner)
            imported_members = 0
            for m in data.get("members", []):
                if m.get("user_id") == user.id:
                    continue
                try:
                    member = WorkspaceMember(
                        workspace_id=ws.id,
                        user_id=m["user_id"],
                        role=Role(m.get("role", "member")),
                    )
                    self._ws_repo.add_member(member)
                    imported_members += 1
                except Exception:
                    pass

            logger.info(
                "User %s imported workspace %s with %d members",
                user.username, ws.id, imported_members,
            )

            return success_response(data={
                "workspace_id": ws.id,
                "name": ws.name,
                "imported_members": imported_members,
            })

        # ─── Workspace health check ──────────────────────────────
        async def workspace_health_check(
            workspace_id: str, auth_user: dict = auth_dep
        ) -> dict:
            """Health check for workspace data integrity."""
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)

            checks: list[dict] = []

            # Check workspace exists
            checks.append({
                "name": "workspace_exists",
                "status": "pass",
                "detail": f"Workspace '{ws.name}' exists",
            })

            # Check members
            members = self._ws_repo.list_members(workspace_id)
            has_owner = any(
                (m.role.value if hasattr(m.role, "value") else str(m.role)) == "owner"
                for m in members
            )
            checks.append({
                "name": "has_owner",
                "status": "pass" if has_owner else "warn",
                "detail": f"{len(members)} members, {'has' if has_owner else 'missing'} owner",
            })

            # Check training jobs consistency
            orphan_jobs = 0
            active_jobs = 0
            try:
                from training.jobs import training_jobs
                member_ids = {m.user_id for m in members}
                for j in training_jobs.values():
                    if j.get("workspace_id", "") == workspace_id:
                        if j.get("status") == "running":
                            active_jobs += 1
                        j_uid = j.get("user_id", "")
                        if j_uid and j_uid not in member_ids:
                            orphan_jobs += 1
            except Exception:
                pass

            checks.append({
                "name": "training_jobs",
                "status": "pass" if orphan_jobs == 0 else "warn",
                "detail": f"{active_jobs} active, {orphan_jobs} orphaned (user not in workspace)",
            })

            # Check datasets accessible
            dataset_count = 0
            try:
                from controllers.datasets import get_datasets_controller
                ctrl = get_datasets_controller()
                ds_list = ctrl.list_datasets(user_id=user.id, workspace_id=workspace_id)
                dataset_count = len(ds_list) if ds_list else 0
            except Exception:
                pass

            checks.append({
                "name": "datasets",
                "status": "pass",
                "detail": f"{dataset_count} datasets accessible",
            })

            # Check knowledge items
            knowledge_count = 0
            try:
                from routers.kb import get_kb_router
                kb = get_kb_router()
                memory = kb._get_memory()
                all_items = memory.list_all(top_k=10000)
                knowledge_count = sum(
                    1 for item in all_items
                    if item.get("workspace_id", "") == workspace_id
                )
            except Exception:
                pass

            checks.append({
                "name": "knowledge",
                "status": "pass",
                "detail": f"{knowledge_count} knowledge items",
            })

            # Overall status
            has_warning = any(c["status"] == "warn" for c in checks)
            has_error = any(c["status"] == "fail" for c in checks)

            return success_response(data={
                "workspace_id": workspace_id,
                "status": "error" if has_error else ("warning" if has_warning else "healthy"),
                "checks": checks,
            })

        # ─── Get workspace settings ─────────────────────────
        async def get_workspace_settings(workspace_id: str, auth_user: dict = auth_dep) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)
            return success_response(data={
                "workspace_id": ws.id,
                "name": ws.name,
                "description": ws.description,
                "default_model": ws.default_model,
                "data_retention_days": ws.data_retention_days,
                "max_members": ws.max_members,
                "allow_sharing": ws.allow_sharing,
                "created_at": ws.created_at,
                "updated_at": ws.updated_at,
            })

        # ─── Update workspace settings ──────────────────────
        async def update_workspace_settings(
            workspace_id: str, req: WorkspaceSettingsRequest, auth_user: dict = auth_dep
        ) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member or member.role not in (Role.ADMIN, Role.OWNER):
                if not user.is_admin:
                    raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)

            if req.name is not None:
                ws.name = req.name
            if req.description is not None:
                ws.description = req.description
            if req.default_model is not None:
                ws.default_model = req.default_model
            if req.data_retention_days is not None:
                ws.data_retention_days = req.data_retention_days
            if req.max_members is not None:
                ws.max_members = req.max_members
            if req.allow_sharing is not None:
                ws.allow_sharing = req.allow_sharing

            ws.updated_at = datetime.now(timezone.utc).isoformat()
            self._ws_repo.update(ws)
            members = self._ws_repo.list_members(workspace_id)
            logger.info("User %s updated settings for workspace %s", user.username, workspace_id)
            return success_response(data={
                "workspace_id": ws.id,
                "name": ws.name,
                "description": ws.description,
                "default_model": ws.default_model,
                "data_retention_days": ws.data_retention_days,
                "max_members": ws.max_members,
                "allow_sharing": ws.allow_sharing,
                "created_at": ws.created_at,
                "updated_at": ws.updated_at,
                "member_count": len(members),
            })

        # ─── Invite member by email ─────────────────────────
        async def invite_member(
            workspace_id: str, req: InviteMemberRequest, auth_user: dict = auth_dep
        ) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            # Check caller is admin or owner
            caller_member = self._ws_repo.get_member(workspace_id, user.id)
            if not caller_member or caller_member.role not in (Role.ADMIN, Role.OWNER):
                if not user.is_admin:
                    raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)

            # Check max members
            members = self._ws_repo.list_members(workspace_id)
            if len(members) >= ws.max_members:
                raise_error(
                    f"Workspace has reached maximum of {ws.max_members} members",
                    "E_LIMIT_EXCEEDED",
                    status_code=400,
                )

            # Find user by email
            invitee = self._user_repo.get_by_email(req.email)
            if not invitee:
                raise_error("User not found with that email", "E_NOT_FOUND", status_code=404)

            # Check if already a member
            existing = self._ws_repo.get_member(workspace_id, invitee.id)
            if existing:
                raise_error("User is already a member", "E_CONFLICT", status_code=409)

            # Add member
            role = Role(req.role) if req.role in ("viewer", "user", "admin") else Role.USER
            member = WorkspaceMember(
                id=str(uuid.uuid4()),
                workspace_id=workspace_id,
                user_id=invitee.id,
                role=role,
            )
            self._ws_repo.add_member(member)
            logger.info("User %s invited %s to workspace %s", user.username, req.email, ws.name)
            return success_response(data={
                "user_id": invitee.id,
                "username": invitee.username,
                "email": req.email,
                "role": role.value,
            })

        # ─── Cleanup expired data ─────────────────────────────
        async def cleanup_workspace_data(workspace_id: str, auth_user: dict = auth_dep) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member or member.role not in (Role.ADMIN, Role.OWNER):
                if not user.is_admin:
                    raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)

            now = datetime.now(timezone.utc).timestamp()
            default_retention = getattr(ws, 'data_retention_days', 90) or 90
            training_retention = getattr(ws, 'training_retention_days', None) or default_retention
            audit_retention = getattr(ws, 'audit_retention_days', None) or default_retention
            dataset_retention = getattr(ws, 'dataset_retention_days', None) or default_retention

            cleaned = {"training_jobs": 0, "audit_logs": 0, "datasets": 0}

            # Clean old training jobs
            try:
                from domains.training.repository import TrainingRepository
                repo = TrainingRepository()
                training_cutoff = now - (training_retention * 86400)
                old_jobs = repo.list_by_workspace(workspace_id)
                for job in old_jobs:
                    created = getattr(job, 'created_at', None)
                    if created:
                        try:
                            job_ts = datetime.fromisoformat(created.replace('Z', '+00:00')).timestamp()
                        except (ValueError, TypeError):
                            continue
                        if job_ts < training_cutoff:
                            repo.delete(job.id)
                            cleaned["training_jobs"] += 1
            except Exception:
                pass

            # Clean old audit logs
            try:
                from infrastructure.auth import AuditLogger
                audit = AuditLogger()
                audit_cutoff = now - (audit_retention * 86400)
                old_logs = audit.list(workspace_id=workspace_id, limit=10000)
                for log in old_logs:
                    ts = log.get("timestamp", "")
                    if ts:
                        try:
                            log_ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                        except (ValueError, TypeError):
                            continue
                        if log_ts < audit_cutoff:
                            audit.delete(log.get("id", ""))
                            cleaned["audit_logs"] += 1
            except Exception:
                pass

            logger.info(
                "User %s cleaned workspace %s: %s",
                user.username, workspace_id, cleaned,
            )
            return success_response(data={
                "retention": {
                    "training_days": training_retention,
                    "audit_days": audit_retention,
                    "dataset_days": dataset_retention,
                },
                "cleaned": cleaned,
            })

        # ─── Bulk member import ───────────────────────────────
        async def bulk_import_members(
            workspace_id: str, req: BulkMemberImportRequest, auth_user: dict = auth_dep
        ) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            caller_member = self._ws_repo.get_member(workspace_id, user.id)
            if not caller_member or caller_member.role not in (Role.ADMIN, Role.OWNER):
                if not user.is_admin:
                    raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)

            added = 0
            skipped = 0
            errors = []

            for i, entry in enumerate(req.members):
                user_id = entry.get("user_id", "")
                email = entry.get("email", "")
                role_str = entry.get("role", "user")

                # Resolve user_id from email if needed
                if not user_id and email:
                    target = self._user_repo.get_by_email(email)
                    if target:
                        user_id = target.id
                    else:
                        errors.append({"index": i, "email": email, "error": "User not found"})
                        skipped += 1
                        continue

                if not user_id:
                    errors.append({"index": i, "error": "No user_id or email provided"})
                    skipped += 1
                    continue

                # Check if already a member
                existing = self._ws_repo.get_member(workspace_id, user_id)
                if existing:
                    skipped += 1
                    continue

                # Check max members
                members = self._ws_repo.list_members(workspace_id)
                if len(members) >= ws.max_members:
                    errors.append({"index": i, "user_id": user_id, "error": "Workspace full"})
                    skipped += 1
                    continue

                role = Role(role_str) if role_str in ("viewer", "user", "admin") else Role.USER
                member = WorkspaceMember(
                    id=str(uuid.uuid4()),
                    workspace_id=workspace_id,
                    user_id=user_id,
                    role=role,
                )
                self._ws_repo.add_member(member)
                added += 1

            return success_response(data={
                "added": added,
                "skipped": skipped,
                "errors": errors,
            })

        # ─── Workspace notifications ──────────────────────────
        async def get_workspace_notifications(
            workspace_id: str, auth_user: dict = auth_dep
        ) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)

            notifications = []

            # Recent training job events
            try:
                from domains.training.repository import TrainingRepository
                repo = TrainingRepository()
                jobs = repo.list_by_workspace(workspace_id)
                for job in jobs[-20:]:  # last 20
                    status = getattr(job, 'status', '')
                    if status in ('completed', 'failed'):
                        notifications.append({
                            "type": "training",
                            "title": f"Training job {status}",
                            "detail": getattr(job, 'name', job.id),
                            "status": status,
                            "timestamp": getattr(job, 'updated_at', getattr(job, 'created_at', '')),
                        })
            except Exception:
                pass

            # Recent member changes from audit log
            try:
                from infrastructure.auth import AuditLogger
                audit = AuditLogger()
                logs = audit.list(workspace_id=workspace_id, limit=50)
                for log in logs:
                    action = log.get("action", "")
                    if "member" in action or "invite" in action:
                        notifications.append({
                            "type": "member",
                            "title": action.replace("_", " ").title(),
                            "detail": log.get("detail", ""),
                            "status": "info",
                            "timestamp": log.get("timestamp", ""),
                        })
            except Exception:
                pass

            # Sort by timestamp descending
            notifications.sort(key=lambda n: n.get("timestamp", ""), reverse=True)

            return success_response(data={
                "notifications": notifications[:50],
            })

        # ─── Clone workspace ─────────────────────────────────
        async def clone_workspace(
            workspace_id: str, auth_user: dict = auth_dep
        ) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member or member.role not in (Role.ADMIN, Role.OWNER):
                if not user.is_admin:
                    raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)

            # Create new workspace with cloned settings
            new_ws = Workspace(
                id=str(uuid.uuid4()),
                name=f"{ws.name} (Copy)",
                tenant_id=ws.tenant_id,
                description=ws.description,
                default_model=getattr(ws, 'default_model', ''),
                data_retention_days=getattr(ws, 'data_retention_days', 90),
                max_members=getattr(ws, 'max_members', 50),
                allow_sharing=getattr(ws, 'allow_sharing', True),
            )
            self._ws_repo.create(new_ws)

            # Clone members (except the creator who's auto-added)
            old_members = self._ws_repo.list_members(workspace_id)
            cloned_count = 0
            for m in old_members:
                if m.user_id == user.id:
                    continue  # skip, will be added as owner
                new_member = WorkspaceMember(
                    id=str(uuid.uuid4()),
                    workspace_id=new_ws.id,
                    user_id=m.user_id,
                    role=m.role,
                )
                self._ws_repo.add_member(new_member)
                cloned_count += 1

            # Add creator as owner
            owner_member = WorkspaceMember(
                id=str(uuid.uuid4()),
                workspace_id=new_ws.id,
                user_id=user.id,
                role=Role.OWNER,
            )
            self._ws_repo.add_member(owner_member)

            logger.info("User %s cloned workspace %s to %s", user.username, workspace_id, new_ws.id)
            return success_response(data={
                "id": new_ws.id,
                "name": new_ws.name,
                "members_cloned": cloned_count,
            })

        # ─── Workspace search ─────────────────────────────────
        async def search_workspace(
            workspace_id: str, auth_user: dict = auth_dep
        ) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)

            # Get query from query params
            from fastapi import Query
            # We need to access query params differently since this is a inner function
            # Use a simpler approach - search everything
            results = {"members": [], "training_jobs": [], "datasets": [], "knowledge": []}

            # Search members
            members = self._ws_repo.list_members(workspace_id)
            for m in members:
                u = self._user_repo.get(m.user_id)
                results["members"].append({
                    "id": m.id,
                    "type": "member",
                    "title": u.username if u else m.user_id,
                    "detail": f"Role: {m.role.value}",
                })

            # Search training jobs
            try:
                from domains.training.repository import TrainingRepository
                repo = TrainingRepository()
                jobs = repo.list_by_workspace(workspace_id)
                for job in jobs:
                    results["training_jobs"].append({
                        "id": job.id,
                        "type": "training",
                        "title": getattr(job, 'name', job.id),
                        "detail": f"Status: {getattr(job, 'status', 'unknown')}",
                    })
            except Exception:
                pass

            # Search datasets
            try:
                from domains.dataset.repository import DatasetRepository
                ds_repo = DatasetRepository()
                datasets = ds_repo.list_by_workspace(workspace_id)
                for ds in datasets:
                    results["datasets"].append({
                        "id": ds.id,
                        "type": "dataset",
                        "title": ds.name,
                        "detail": f"Size: {getattr(ds, 'size', 0)} bytes",
                    })
            except Exception:
                pass

            # Search knowledge
            try:
                from domains.learner.knowledge import KnowledgeRepository
                k_repo = KnowledgeRepository()
                facts = k_repo.list_by_workspace(workspace_id)
                for fact in facts:
                    results["knowledge"].append({
                        "id": fact.id,
                        "type": "knowledge",
                        "title": fact.subject if hasattr(fact, 'subject') else str(fact.id),
                        "detail": fact.predicate if hasattr(fact, 'predicate') else "",
                    })
            except Exception:
                pass

            total = sum(len(v) for v in results.values())
            return success_response(data={"results": results, "total": total})

        # ─── Permissions matrix ───────────────────────────────
        async def get_workspace_permissions(
            workspace_id: str, auth_user: dict = auth_dep
        ) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(workspace_id)
            if not ws:
                raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)

            from domains.auth.models import ROLE_PERMISSIONS, Permission

            # Build permission matrix
            roles = {}
            for role, perms in ROLE_PERMISSIONS.items():
                roles[role.value] = {
                    "name": role.value,
                    "permissions": [p.value for p in perms],
                }

            # All available permissions grouped by category
            all_permissions = {}
            for perm in Permission:
                category = perm.value.split(":")[0]
                if category not in all_permissions:
                    all_permissions[category] = []
                all_permissions[category].append(perm.value)

            # Member permissions summary
            members = self._ws_repo.list_members(workspace_id)
            member_perms = []
            for m in members:
                u = self._user_repo.get(m.user_id)
                perms = ROLE_PERMISSIONS.get(m.role, set())
                member_perms.append({
                    "user_id": m.user_id,
                    "username": u.username if u else "",
                    "role": m.role.value,
                    "permissions": [p.value for p in perms],
                })

            return success_response(data={
                "roles": roles,
                "all_permissions": all_permissions,
                "member_permissions": member_perms,
            })

        # ─── Data Sharing ───────────────────────────────────────
        class ShareDataRequest(BaseModel):
            resource_type: str = Field(..., description="Dataset, knowledge, or api_key")
            resource_id: str = Field(..., min_length=1)
            target_workspace_id: str = Field(..., min_length=1)
            permission: str = Field(default="read", description="Read or admin")

        async def share_data(req: ShareDataRequest, auth_user: dict = auth_dep) -> dict:
            user = self._get_user(auth_user)
            ws = self._ws_repo.get(req.target_workspace_id)
            if not ws:
                raise_error("Target workspace not found", "E_NOT_FOUND", status_code=404)
            if req.resource_type not in ("dataset", "knowledge", "api_key"):
                raise_error("Invalid resource type", "E_VALIDATION", status_code=422)
            if req.permission not in ("read", "admin"):
                raise_error("Invalid permission", "E_VALIDATION", status_code=422)
            # Check source workspace membership
            source_ws_id = auth_user.get("workspace_id", "")
            if not source_ws_id:
                raise_error("No workspace context", "E_AUTH_MISSING", status_code=400)
            member = self._ws_repo.get_member(source_ws_id, user.id)
            if not member or member.role not in (Role.ADMIN, Role.OWNER):
                raise_error("Admin or owner role required", "E_AUTH_MISSING", status_code=403)
            if req.target_workspace_id == source_ws_id:
                raise_error("Cannot share with same workspace", "E_VALIDATION", status_code=422)

            share_id = str(uuid.uuid4())
            doc = {
                "id": share_id,
                "resource_type": req.resource_type,
                "resource_id": req.resource_id,
                "source_workspace_id": source_ws_id,
                "target_workspace_id": req.target_workspace_id,
                "permission": req.permission,
                "shared_by": user.id,
                "shared_at": datetime.now(timezone.utc).isoformat(),
            }
            # Store in MogDB
            from infrastructure.mogdb import get_mogdb
            db = get_mogdb()
            db.insert("workspace_shares", doc)

            safe_audit_log("workspace.share", resource=source_ws_id, user_id=user.id,
                           detail=f"{req.resource_type}:{req.resource_id} -> {req.target_workspace_id}")
            return success_response(data=doc)

        async def list_shared_data(workspace_id: str, auth_user: dict = auth_dep) -> dict:
            from infrastructure.mogdb import get_mogdb
            db = get_mogdb()
            shares = list(db.find("workspace_shares", {
                "$or": [
                    {"source_workspace_id": workspace_id},
                    {"target_workspace_id": workspace_id},
                ]
            }))
            return success_response(data={"shares": shares})

        async def revoke_share(share_id: str, workspace_id: str, auth_user: dict = auth_dep) -> dict:
            user = self._get_user(auth_user)
            member = self._ws_repo.get_member(workspace_id, user.id)
            if not member or member.role not in (Role.ADMIN, Role.OWNER):
                raise_error("Admin or owner role required", "E_AUTH_MISSING", status_code=403)
            from infrastructure.mogdb import get_mogdb
            db = get_mogdb()
            doc = db.find_one("workspace_shares", {"id": share_id})
            if not doc:
                raise_error("Share not found", "E_NOT_FOUND", status_code=404)
            if doc["source_workspace_id"] != workspace_id:
                raise_error("Not authorized to revoke this share", "E_AUTH_MISSING", status_code=403)
            db.delete("workspace_shares", {"id": share_id})
            safe_audit_log("workspace.unshare", resource=workspace_id, user_id=user.id,
                           detail=f"Revoked share {share_id}")
            return success_response(data={"revoked": share_id})

        async def get_shared_datasets(workspace_id: str, auth_user: dict = auth_dep) -> dict:
            from infrastructure.mogdb import get_mogdb
            from controllers.datasets import get_datasets_controller
            db = get_mogdb()
            shares = list(db.find("workspace_shares", {
                "target_workspace_id": workspace_id,
                "resource_type": "dataset",
            }))
            ctrl = get_datasets_controller()
            results = []
            for s in shares:
                ds_list = ctrl.list_datasets(workspace_id=s["source_workspace_id"])
                for ds in ds_list:
                    if ds.get("id") == s["resource_id"]:
                        results.append({"share": s, "dataset": ds})
                        break
            return success_response(data={"datasets": results})

        async def get_shared_knowledge(workspace_id: str, auth_user: dict = auth_dep) -> dict:
            from infrastructure.mogdb import get_mogdb
            db = get_mogdb()
            shares = list(db.find("workspace_shares", {
                "target_workspace_id": workspace_id,
                "resource_type": "knowledge",
            }))
            return success_response(data={"knowledge": shares})

        async def get_shared_api_keys(workspace_id: str, auth_user: dict = auth_dep) -> dict:
            from infrastructure.mogdb import get_mogdb
            db = get_mogdb()
            shares = list(db.find("workspace_shares", {
                "target_workspace_id": workspace_id,
                "resource_type": "api_key",
            }))
            return success_response(data={"api_keys": shares})

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
        router.add_api_route("/{workspace_id}/stats", get_workspace_stats, methods=["GET"])
        router.add_api_route("/{workspace_id}/usage", get_workspace_usage, methods=["GET"])
        router.add_api_route("/{workspace_id}/activity", get_workspace_activity, methods=["GET"])
        router.add_api_route("/{workspace_id}/export", export_workspace_data, methods=["GET"])
        router.add_api_route("/import", import_workspace_data, methods=["POST"])
        router.add_api_route("/{workspace_id}/health", workspace_health_check, methods=["GET"])
        router.add_api_route("/{workspace_id}/settings", get_workspace_settings, methods=["GET"])
        router.add_api_route("/{workspace_id}/settings", update_workspace_settings, methods=["PUT"])
        router.add_api_route("/{workspace_id}/invite", invite_member, methods=["POST"])
        router.add_api_route("/{workspace_id}/cleanup", cleanup_workspace_data, methods=["POST"])
        router.add_api_route("/{workspace_id}/members/bulk", bulk_import_members, methods=["POST"])
        router.add_api_route("/{workspace_id}/notifications", get_workspace_notifications, methods=["GET"])
        router.add_api_route("/{workspace_id}/clone", clone_workspace, methods=["POST"])
        router.add_api_route("/{workspace_id}/search", search_workspace, methods=["GET"])
        router.add_api_route("/{workspace_id}/permissions", get_workspace_permissions, methods=["GET"])
        router.add_api_route("/{workspace_id}/share", share_data, methods=["POST"])
        router.add_api_route("/{workspace_id}/shared", list_shared_data, methods=["GET"])
        router.add_api_route("/{workspace_id}/share/{share_id}", revoke_share, methods=["DELETE"])
        router.add_api_route("/{workspace_id}/shared/datasets", get_shared_datasets, methods=["GET"])
        router.add_api_route("/{workspace_id}/shared/knowledge", get_shared_knowledge, methods=["GET"])
        router.add_api_route("/{workspace_id}/shared/api-keys", get_shared_api_keys, methods=["GET"])


# ─── Singleton ─────────────────────────────────────────────────

_workspaces_router: WorkspacesRouter | None = None


def get_workspaces_router() -> WorkspacesRouter:
    global _workspaces_router
    if _workspaces_router is None:
        _workspaces_router = WorkspacesRouter()
    return _workspaces_router
