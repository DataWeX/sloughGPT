"""Backward-compatibility shim — ``domains.auth.models`` → ``domain.auth``."""

from domain.auth._internal.models import (  # noqa: F401
    Role,
    Permission,
    ROLE_PERMISSIONS,
    UserRole,
    User,
    Tenant,
    Workspace,
    WorkspaceMember,
)

__all__ = [
    "Role",
    "Permission",
    "ROLE_PERMISSIONS",
    "UserRole",
    "User",
    "Tenant",
    "Workspace",
    "WorkspaceMember",
]
