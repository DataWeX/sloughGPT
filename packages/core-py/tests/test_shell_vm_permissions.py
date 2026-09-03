"""Tests for X86RBAC — role-based access control for x86 VM."""
from __future__ import annotations

from domains.shell.vm_permissions import Permission, Role, X86RBAC


class TestRole:
    def test_ordering(self):
        assert Role.USER < Role.ADMIN < Role.KERNEL

    def test_all_roles(self):
        assert len(Role) == 3


class TestPermission:
    def test_auto_values(self):
        assert Permission.FILE_READ.value == 1
        assert Permission.FILE_WRITE.value == 2

    def test_count(self):
        assert len(Permission) == 14


class TestX86RBAC:
    def test_default_role_is_user(self):
        rbac = X86RBAC()
        assert rbac.role_of(1) == Role.USER

    def test_assign(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.ADMIN)
        assert rbac.role_of(1) == Role.ADMIN

    def test_user_file_read(self):
        rbac = X86RBAC()
        assert rbac.check(1, Permission.FILE_READ) is True

    def test_user_no_device(self):
        rbac = X86RBAC()
        assert rbac.check(1, Permission.DEVICE_DISK) is False

    def test_admin_device(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.ADMIN)
        assert rbac.check(1, Permission.DEVICE_DISK) is True

    def test_kernel_all_permissions(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        for perm in Permission:
            assert rbac.check(1, perm) is True

    def test_inherit(self):
        rbac = X86RBAC()
        rbac.assign(100, Role.ADMIN)
        rbac.inherit(101, 100)
        assert rbac.role_of(101) == Role.ADMIN

    def test_escalate_by_kernel(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        result = rbac.escalate(2, Role.ADMIN, caller_pid=1)
        assert result is True
        assert rbac.role_of(2) == Role.ADMIN

    def test_escalate_reject_non_kernel(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.ADMIN)
        result = rbac.escalate(2, Role.KERNEL, caller_pid=1)
        assert result is False
        assert rbac.role_of(2) == Role.USER
