"""Tests for domains.shell.vm_permissions — Role, Permission, X86RBAC."""

import pytest
from domains.shell.vm_permissions import Role, Permission, X86RBAC, _ROLE_PERMISSIONS


class TestRole:
    def test_user_is_zero(self):
        assert Role.USER == 0

    def test_admin_is_one(self):
        assert Role.ADMIN == 1

    def test_kernel_is_two(self):
        assert Role.KERNEL == 2

    def test_ordering(self):
        assert Role.USER < Role.ADMIN < Role.KERNEL


class TestPermission:
    def test_auto_values(self):
        assert Permission.FILE_READ.value == 1
        assert Permission.FILE_WRITE.value == 2
        assert Permission.TRAINING.value == 14

    def test_all_permissions_unique(self):
        values = [p.value for p in Permission]
        assert len(values) == len(set(values))


class TestRolePermissions:
    def test_user_has_file_ops(self):
        assert Permission.FILE_READ in _ROLE_PERMISSIONS[Role.USER]
        assert Permission.FILE_WRITE in _ROLE_PERMISSIONS[Role.USER]
        assert Permission.FILE_META in _ROLE_PERMISSIONS[Role.USER]

    def test_user_has_process_self_and_spawn(self):
        assert Permission.PROCESS_SELF in _ROLE_PERMISSIONS[Role.USER]
        assert Permission.PROCESS_SPAWN in _ROLE_PERMISSIONS[Role.USER]

    def test_user_no_devices(self):
        assert Permission.DEVICE_SERIAL not in _ROLE_PERMISSIONS[Role.USER]
        assert Permission.DEVICE_MOUSE not in _ROLE_PERMISSIONS[Role.USER]
        assert Permission.DEVICE_NET not in _ROLE_PERMISSIONS[Role.USER]

    def test_user_no_training(self):
        assert Permission.TRAINING not in _ROLE_PERMISSIONS[Role.USER]

    def test_admin_has_devices(self):
        assert Permission.DEVICE_SERIAL in _ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.DEVICE_MOUSE in _ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.DEVICE_DISK in _ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.DEVICE_RTC in _ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.DEVICE_NET in _ROLE_PERMISSIONS[Role.ADMIN]

    def test_admin_has_training(self):
        assert Permission.TRAINING in _ROLE_PERMISSIONS[Role.ADMIN]

    def test_admin_has_kill(self):
        assert Permission.PROCESS_KILL in _ROLE_PERMISSIONS[Role.ADMIN]

    def test_kernel_has_all(self):
        kernel_perms = _ROLE_PERMISSIONS[Role.KERNEL]
        for perm in Permission:
            assert perm in kernel_perms

    def test_admin_subset_of_kernel(self):
        admin_perms = _ROLE_PERMISSIONS[Role.ADMIN]
        kernel_perms = _ROLE_PERMISSIONS[Role.KERNEL]
        assert admin_perms.issubset(kernel_perms)


class TestX86RBAC:
    def test_default_role_is_user(self):
        rbac = X86RBAC()
        assert rbac.role_of(1) == Role.USER

    def test_assign(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.ADMIN)
        assert rbac.role_of(1) == Role.ADMIN

    def test_check_granted(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.USER)
        assert rbac.check(1, Permission.FILE_READ) is True

    def test_check_denied(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.USER)
        assert rbac.check(1, Permission.DEVICE_NET) is False

    def test_inherit(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.ADMIN)
        rbac.inherit(2, 1)
        assert rbac.role_of(2) == Role.ADMIN
        assert rbac.check(2, Permission.DEVICE_NET) is True

    def test_inherit_default_user(self):
        rbac = X86RBAC()
        rbac.inherit(1, 999)  # parent not assigned → default USER
        assert rbac.role_of(1) == Role.USER

    def test_escalate_by_kernel(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        result = rbac.escalate(2, Role.ADMIN, caller_pid=1)
        assert result is True
        assert rbac.role_of(2) == Role.ADMIN

    def test_escalate_denied_for_non_kernel(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.ADMIN)
        result = rbac.escalate(2, Role.KERNEL, caller_pid=1)
        assert result is False
        assert rbac.role_of(2) == Role.USER  # unchanged

    def test_escalate_denied_for_user(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.USER)
        result = rbac.escalate(2, Role.ADMIN, caller_pid=1)
        assert result is False

    def test_kernel_can_do_anything(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        for perm in Permission:
            assert rbac.check(1, perm) is True
