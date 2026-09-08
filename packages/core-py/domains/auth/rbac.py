"""RBAC — Role-Based Access Control enforcement."""

from __future__ import annotations

import logging
from typing import Any

from domains.auth.models import (
    Permission,
    Role,
    ROLE_PERMISSIONS,
    User,
    WorkspaceMember,
)

logger = logging.getLogger("slo.auth.rbac")


class RBAC:
    """Role-Based Access Control engine.

    Checks permissions against:
    1. User's global role (tenant-level)
    2. User's workspace role (workspace-level)
    """

    def check_user_permission(self, user: User, permission: Permission) -> bool:
        """Check if a user has a permission based on their global role.

        Args:
            user: The user to check.
            permission: The permission to verify.

        Returns:
            True if the user's role grants the permission.
        """
        if not user.is_active:
            return False
        allowed = ROLE_PERMISSIONS.get(user.role, set())
        return permission in allowed

    def check_workspace_permission(
        self,
        user: User,
        workspace_member: WorkspaceMember | None,
        permission: Permission,
    ) -> bool:
        """Check if a user has a permission in a workspace context.

        Uses the HIGHER of the user's global role and workspace role.

        Args:
            user: The user to check.
            workspace_member: The user's membership in the workspace (if any).
            permission: The permission to verify.

        Returns:
            True if either the global or workspace role grants the permission.
        """
        # Check global role first
        if self.check_user_permission(user, permission):
            return True

        # Check workspace role
        if workspace_member and workspace_member.user_id == user.id:
            workspace_allowed = ROLE_PERMISSIONS.get(workspace_member.role, set())
            if permission in workspace_allowed:
                return True

        return False

    def get_effective_role(self, user: User, workspace_member: WorkspaceMember | None) -> Role:
        """Get the effective (higher) role between global and workspace.

        Args:
            user: The user.
            workspace_member: The user's workspace membership.

        Returns:
            The higher-privilege role.
        """
        role_hierarchy = [Role.VIEWER, Role.USER, Role.ADMIN, Role.OWNER]

        global_idx = role_hierarchy.index(user.role)
        workspace_idx = (
            role_hierarchy.index(workspace_member.role)
            if workspace_member
            else -1
        )

        return role_hierarchy[max(global_idx, workspace_idx)]

    def filter_permissions(self, user: User) -> set[Permission]:
        """Get all permissions granted to a user.

        Args:
            user: The user.

        Returns:
            Set of permissions the user has.
        """
        return ROLE_PERMISSIONS.get(user.role, set())


# Singleton
_rbac_instance: RBAC | None = None


def get_rbac() -> RBAC:
    """Get the RBAC singleton."""
    global _rbac_instance
    if _rbac_instance is None:
        _rbac_instance = RBAC()
    return _rbac_instance
