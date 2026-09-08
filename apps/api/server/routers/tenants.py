"""Tenants API — organization management endpoints.

Provides CRUD operations for managing tenants (organizations).
Only admins and owners can manage tenants.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from domains.auth.models import Role, Tenant, User, UserRole
from domains.auth.repositories import TenantRepository, UserRepository
from infrastructure.auth import require_auth_if_enabled
from schemas.common import classify_and_raise, raise_error, success_response

logger = logging.getLogger("slo.tenants")


# ─── Request / Response models ─────────────────────────────────


class TenantCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9-]+$")
    plan: str = Field(default="free", description="Plan: free, pro, enterprise")
    max_users: int = Field(default=5, ge=1, le=1000)
    max_workspaces: int = Field(default=3, ge=1, le=100)


class TenantUpdateRequest(BaseModel):
    name: str | None = Field(None, max_length=200)
    plan: str | None = Field(None, description="Plan: free, pro, enterprise")
    max_users: int | None = Field(None, ge=1, le=1000)
    max_workspaces: int | None = Field(None, ge=1, le=100)


class TenantResponse(BaseModel):
    id: str
    name: str
    slug: str
    plan: str
    max_users: int
    max_workspaces: int
    created_at: str


# ─── Router ────────────────────────────────────────────────────


class TenantsRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/tenants", tags=["tenants"])
        self._repo = TenantRepository()
        self._user_repo = UserRepository()
        self._register_routes()

    def _require_admin(self, auth_user: dict | None) -> User:
        if not auth_user:
            raise_error("Authentication required", "E_AUTH_MISSING", status_code=401)
        user_id = auth_user.get("sub", "")
        user = self._user_repo.get(user_id)
        if not user:
            raise_error("User not found", "E_NOT_FOUND", status_code=404)
        if not user.is_admin:
            raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)
        return user

    def _to_response(self, tenant: Tenant) -> TenantResponse:
        return TenantResponse(
            id=tenant.id,
            name=tenant.name,
            slug=tenant.slug,
            plan=tenant.plan,
            max_users=tenant.max_users,
            max_workspaces=tenant.max_workspaces,
            created_at=tenant.created_at,
        )

    def _register_routes(self):
        router = self.router
        auth_dep = Depends(require_auth_if_enabled)

        # ─── List tenants ──────────────────────────────────────
        async def list_tenants(auth_user: dict = auth_dep) -> dict:
            admin = self._require_admin(auth_user)
            if admin.role == Role.OWNER:
                tenants = self._repo.list_all()
            else:
                # Non-owners can only see their own tenant
                tenant = self._repo.get(admin.tenant_id)
                tenants = [tenant] if tenant else []
            return success_response(
                data=[self._to_response(t).model_dump() for t in tenants],
                meta={"total": len(tenants)},
            )

        # ─── Get tenant ────────────────────────────────────────
        async def get_tenant(tenant_id: str, auth_user: dict = auth_dep) -> dict:
            admin = self._require_admin(auth_user)
            if admin.role != Role.OWNER and admin.tenant_id != tenant_id:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)
            tenant = self._repo.get(tenant_id)
            if not tenant:
                raise_error("Tenant not found", "E_NOT_FOUND", status_code=404)
            return success_response(data=self._to_response(tenant).model_dump())

        # ─── Create tenant ─────────────────────────────────────
        async def create_tenant(req: TenantCreateRequest, auth_user: dict = auth_dep) -> dict:
            self._require_admin(auth_user)
            existing = self._repo.get_by_slug(req.slug)
            if existing:
                raise_error("Slug already taken", "E_INFRA_BUSY", status_code=409)

            tenant = Tenant(
                id=str(uuid.uuid4()),
                name=req.name,
                slug=req.slug,
                plan=req.plan,
                max_users=req.max_users,
                max_workspaces=req.max_workspaces,
            )
            self._repo.create(tenant)
            logger.info("Admin created tenant %s (slug=%s)", req.name, req.slug)
            return success_response(data=self._to_response(tenant).model_dump())

        # ─── Update tenant ─────────────────────────────────────
        async def update_tenant(
            tenant_id: str, req: TenantUpdateRequest, auth_user: dict = auth_dep
        ) -> dict:
            self._require_admin(auth_user)
            tenant = self._repo.get(tenant_id)
            if not tenant:
                raise_error("Tenant not found", "E_NOT_FOUND", status_code=404)

            if req.name is not None:
                tenant.name = req.name
            if req.plan is not None:
                tenant.plan = req.plan
            if req.max_users is not None:
                tenant.max_users = req.max_users
            if req.max_workspaces is not None:
                tenant.max_workspaces = req.max_workspaces

            tenant.updated_at = datetime.now(timezone.utc).isoformat()
            self._repo.update(tenant)
            return success_response(data=self._to_response(tenant).model_dump())

        # ─── Delete tenant ─────────────────────────────────────
        async def delete_tenant(tenant_id: str, auth_user: dict = auth_dep) -> dict:
            admin = self._require_admin(auth_user)
            if admin.role != Role.OWNER:
                raise_error("Only owners can delete tenants", "E_AUTH_MISSING", status_code=403)
            tenant = self._repo.get(tenant_id)
            if not tenant:
                raise_error("Tenant not found", "E_NOT_FOUND", status_code=404)
            self._repo.delete(tenant_id)
            logger.info("Admin deleted tenant %s", tenant.name)
            return success_response(data={"deleted": True})

        # ─── Get tenant stats ──────────────────────────────────
        async def get_tenant_stats(tenant_id: str, auth_user: dict = auth_dep) -> dict:
            admin = self._require_admin(auth_user)
            if admin.role != Role.OWNER and admin.tenant_id != tenant_id:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)
            tenant = self._repo.get(tenant_id)
            if not tenant:
                raise_error("Tenant not found", "E_NOT_FOUND", status_code=404)
            user_count = self._user_repo.count(tenant_id=tenant_id)
            return success_response(
                data={
                    "tenant_id": tenant_id,
                    "user_count": user_count,
                    "max_users": tenant.max_users,
                    "plan": tenant.plan,
                }
            )

        router.add_api_route("", list_tenants, methods=["GET"])
        router.add_api_route("/{tenant_id}", get_tenant, methods=["GET"])
        router.add_api_route("", create_tenant, methods=["POST"])
        router.add_api_route("/{tenant_id}", update_tenant, methods=["PUT"])
        router.add_api_route("/{tenant_id}", delete_tenant, methods=["DELETE"])
        router.add_api_route("/{tenant_id}/stats", get_tenant_stats, methods=["GET"])


# ─── Singleton ─────────────────────────────────────────────────

_tenants_router: TenantsRouter | None = None


def get_tenants_router() -> TenantsRouter:
    global _tenants_router
    if _tenants_router is None:
        _tenants_router = TenantsRouter()
    return _tenants_router
