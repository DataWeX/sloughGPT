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

_LAZY_IMPORTS = {
    "TenantRepository": ("._internal.repositories", "TenantRepository"),
    "UserRepository": ("._internal.repositories", "UserRepository"),
    "WorkspaceRepository": ("._internal.repositories", "WorkspaceRepository"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Role",
    "Permission",
    "ROLE_PERMISSIONS",
    "UserRole",
    "User",
    "Tenant",
    "Workspace",
    "WorkspaceMember",
    "RBAC",
    "get_rbac",
    "TenantRepository",
    "UserRepository",
    "WorkspaceRepository",
]
