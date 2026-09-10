"""Users admin API — user management endpoints.

Provides CRUD operations for managing users, with role-based access control.
Only admins and owners can manage users.
"""
from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from domains.auth.models import Role, User, UserRole
from domains.auth.repositories import UserRepository
from infrastructure.auth import get_jwt_auth, require_auth_if_enabled
from schemas.common import classify_and_raise, endpoint, raise_error, success_response

logger = logging.getLogger("slo.users")


# ─── Request / Response models ─────────────────────────────────


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=500)
    role: str = Field(default="user", description="Role: viewer, user, admin, owner")
    display_name: str = Field(default="", max_length=200)


class UserUpdateRequest(BaseModel):
    email: str | None = Field(None, max_length=254)
    role: str | None = Field(None, description="Role: viewer, user, admin, owner")
    status: str | None = Field(None, description="Status: active, inactive, suspended")
    display_name: str | None = Field(None, max_length=200)


class ProfileUpdateRequest(BaseModel):
    email: str | None = Field(None, max_length=254)
    display_name: str | None = Field(None, max_length=200)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=8, max_length=500)
    new_password: str = Field(..., min_length=8, max_length=500)


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    status: str
    display_name: str
    tenant_id: str
    created_at: str
    last_login_at: str


# ─── Router ────────────────────────────────────────────────────


class UsersRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/users", tags=["users"])
        self._repo = UserRepository()
        self._register_routes()

    def _require_admin(self, auth_user: dict | None) -> User:
        """Verify the caller is an admin or owner."""
        if not auth_user:
            raise_error("Authentication required", "E_AUTH_MISSING", status_code=401)
        user_id = auth_user.get("sub", "")
        user = self._repo.get(user_id)
        if not user:
            raise_error("User not found", "E_NOT_FOUND", status_code=404)
        if not user.is_admin:
            raise_error("Admin access required", "E_AUTH_MISSING", status_code=403)
        return user

    def _to_response(self, user: User) -> UserResponse:
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role.value,
            status=user.status.value,
            display_name=user.display_name,
            tenant_id=user.tenant_id,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )

    def _register_routes(self):
        router = self.router
        auth_dep = Depends(require_auth_if_enabled)

        # ─── List users ────────────────────────────────────────
        @endpoint("users.list")
        async def list_users(auth_user: dict = auth_dep) -> dict:
            self._require_admin(auth_user)
            users = self._repo.list_by_tenant(auth_user.get("tenant_id", ""))
            return success_response(
                data=[self._to_response(u).model_dump() for u in users],
                meta={"total": len(users)},
            )

        # ─── Get user ──────────────────────────────────────────
        @endpoint("users.get")
        async def get_user(user_id: str, auth_user: dict = auth_dep) -> dict:
            if not auth_user:
                raise_error("Authentication required", "E_AUTH_MISSING", status_code=401)
            # Users can read their own profile without admin
            is_self = auth_user.get("sub") == user_id
            if not is_self:
                caller = self._require_admin(auth_user)
                if not caller.is_admin:
                    raise_error("Access denied", "E_AUTH_MISSING", status_code=403)
            user = self._repo.get(user_id)
            if not user:
                raise_error("User not found", "E_NOT_FOUND", status_code=404)
            return success_response(data=self._to_response(user).model_dump())

        # ─── Create user ───────────────────────────────────────
        @endpoint("users.create")
        async def create_user(req: UserCreateRequest, auth_user: dict = auth_dep) -> dict:
            self._require_admin(auth_user)
            existing = self._repo.get_by_username(req.username)
            if existing:
                raise_error("Username already exists", "E_INFRA_BUSY", status_code=409)
            existing_email = self._repo.get_by_email(req.email)
            if existing_email:
                raise_error("Email already registered", "E_INFRA_BUSY", status_code=409)

            from routers.auth import AuthRouter
            password_hash = AuthRouter._hash_password(req.password)

            role = Role(req.role) if req.role in [r.value for r in Role] else Role.USER

            user = User(
                id=str(uuid.uuid4()),
                username=req.username,
                email=req.email,
                password_hash=password_hash,
                display_name=req.display_name,
                role=role,
                status=UserRole.ACTIVE,
                tenant_id=auth_user.get("tenant_id", ""),
            )
            self._repo.create(user)
            logger.info("Admin created user %s (role=%s)", req.username, role.value)
            return success_response(data=self._to_response(user).model_dump())

        # ─── Update user ───────────────────────────────────────
        @endpoint("users.update")
        async def update_user(
            user_id: str, req: UserUpdateRequest, auth_user: dict = auth_dep
        ) -> dict:
            self._require_admin(auth_user)
            user = self._repo.get(user_id)
            if not user:
                raise_error("User not found", "E_NOT_FOUND", status_code=404)

            if req.email is not None:
                existing_email = self._repo.get_by_email(req.email)
                if existing_email and existing_email.id != user_id:
                    raise_error("Email already registered", "E_INFRA_BUSY", status_code=409)
                user.email = req.email
            if req.role is not None:
                user.role = Role(req.role) if req.role in [r.value for r in Role] else user.role
            if req.status is not None:
                user.status = (
                    UserRole(req.status) if req.status in [s.value for s in UserRole] else user.status
                )
            if req.display_name is not None:
                user.display_name = req.display_name

            user.updated_at = datetime.now(timezone.utc).isoformat()
            self._repo.update(user)
            return success_response(data=self._to_response(user).model_dump())

        # ─── Delete user ───────────────────────────────────────
        @endpoint("users.delete")
        async def delete_user(user_id: str, auth_user: dict = auth_dep) -> dict:
            admin = self._require_admin(auth_user)
            if admin.id == user_id:
                raise_error("Cannot delete yourself", "E_INVALID_INPUT", status_code=400)
            user = self._repo.get(user_id)
            if not user:
                raise_error("User not found", "E_NOT_FOUND", status_code=404)
            self._repo.delete(user_id)
            logger.info("Admin deleted user %s", user.username)
            return success_response(data={"deleted": True})

        # ─── Change password (self-service) ────────────────────
        @endpoint("users.change_password")
        async def change_password(
            req: PasswordChangeRequest, auth_user: dict = auth_dep
        ) -> dict:
            if not auth_user:
                raise_error("Authentication required", "E_AUTH_MISSING", status_code=401)
            user = self._repo.get(auth_user.get("sub", ""))
            if not user:
                raise_error("User not found", "E_NOT_FOUND", status_code=404)

            from routers.auth import AuthRouter
            if not AuthRouter._verify_password(req.current_password, user.password_hash):
                raise_error("Current password is incorrect", "E_AUTH_MISSING", status_code=401)

            user.password_hash = AuthRouter._hash_password(req.new_password)
            user.updated_at = datetime.now(timezone.utc).isoformat()
            self._repo.update(user)
            return success_response(data={"changed": True})

        # ─── Update own profile (self-service) ─────────────────
        @endpoint("users.update_profile")
        async def update_profile(
            req: ProfileUpdateRequest, auth_user: dict = auth_dep
        ) -> dict:
            if not auth_user:
                raise_error("Authentication required", "E_AUTH_MISSING", status_code=401)
            user = self._repo.get(auth_user.get("sub", ""))
            if not user:
                raise_error("User not found", "E_NOT_FOUND", status_code=404)

            if req.email is not None:
                existing_email = self._repo.get_by_email(req.email)
                if existing_email and existing_email.id != user.id:
                    raise_error("Email already registered", "E_INFRA_BUSY", status_code=409)
                user.email = req.email
            if req.display_name is not None:
                user.display_name = req.display_name

            user.updated_at = datetime.now(timezone.utc).isoformat()
            self._repo.update(user)
            return success_response(data=self._to_response(user).model_dump())

        # ─── Get own profile (self-service) ────────────────────
        @endpoint("users.get_profile")
        async def get_profile(auth_user: dict = auth_dep) -> dict:
            if not auth_user:
                raise_error("Authentication required", "E_AUTH_MISSING", status_code=401)
            user = self._repo.get(auth_user.get("sub", ""))
            if not user:
                raise_error("User not found", "E_NOT_FOUND", status_code=404)
            return success_response(data=self._to_response(user).model_dump())

        router.add_api_route("", list_users, methods=["GET"])
        router.add_api_route("/{user_id}", get_user, methods=["GET"])
        router.add_api_route("", create_user, methods=["POST"])
        router.add_api_route("/{user_id}", update_user, methods=["PUT"])
        router.add_api_route("/{user_id}", delete_user, methods=["DELETE"])
        router.add_api_route("/me/password", change_password, methods=["POST"])
        router.add_api_route("/me/profile", update_profile, methods=["PUT"])
        router.add_api_route("/me/profile", get_profile, methods=["GET"])


# ─── Singleton ─────────────────────────────────────────────────

_users_router: UsersRouter | None = None


def get_users_router() -> UsersRouter:
    global _users_router
    if _users_router is None:
        _users_router = UsersRouter()
    return _users_router
