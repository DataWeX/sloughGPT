"""Backward-compatibility shim — imports from the new ``services.auth`` package."""

from services.auth import (
    RBAC,
    Permission,
    Role,
    Tenant,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
    get_rbac,
)

__all__ = [
    "Role",
    "Permission",
    "UserRole",
    "User",
    "Tenant",
    "Workspace",
    "WorkspaceMember",
    "RBAC",
    "get_rbac",
]
