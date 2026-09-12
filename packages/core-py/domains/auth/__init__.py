"""Backward-compatibility shim — imports from the new ``domain.auth`` package."""

from domain.auth import (
    Role,
    Permission,
    UserRole,
    User,
    Tenant,
    Workspace,
    WorkspaceMember,
    RBAC,
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
