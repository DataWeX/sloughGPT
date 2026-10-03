"""auth — User, Tenant, Workspace, RBAC models.

Public API:
    Role, Permission, UserRole, User, Tenant, Workspace, WorkspaceMember
    ROLE_PERMISSIONS, UserRepository, TenantRepository, WorkspaceRepository
    RBAC, get_rbac
"""

from services.auth._internal.models import (
    ROLE_PERMISSIONS,
    Permission,
    Role,
    Tenant,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
)
from services.auth._internal.rbac import RBAC, get_rbac
from services.auth._internal.repositories import (
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
    "ROLE_PERMISSIONS",
    "UserRepository",
    "TenantRepository",
    "WorkspaceRepository",
    "RBAC",
    "get_rbac",
]
