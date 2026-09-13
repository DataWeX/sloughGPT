"""Backward-compatibility shim — ``domains.auth.repositories`` → ``domain.auth``."""

from domain.auth._internal.repositories import (  # noqa: F401
    UserRepository,
    TenantRepository,
    WorkspaceRepository,
)

__all__ = [
    "UserRepository",
    "TenantRepository",
    "WorkspaceRepository",
]
