"""auth — User, Tenant, Workspace, RBAC models.

Public API:
    Role, Permission, UserRole, User, Tenant, Workspace, WorkspaceMember
    RBAC, get_rbac
"""

from domain.auth._internal.models import (
    ROLE_PERMISSIONS,
    Permission,
    Role,
    Tenant,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
)
from domain.auth._internal.rbac import RBAC, get_rbac
from domain.auth._internal.repositories import (
    TenantRepository,
    UserRepository,
    WorkspaceRepository,
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
    "ROLE_PERMISSIONS",
    "TenantRepository",
    "UserRepository",
    "WorkspaceRepository",
]
