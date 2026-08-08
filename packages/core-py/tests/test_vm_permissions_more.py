"""Coverage tests for the x86 VM RBAC layer (domains.shell.vm_permissions)."""

from domains.shell.vm_permissions import Permission, Role, X86RBAC


class TestX86RBAC:
    def test_default_role_is_user(self):
        rbac = X86RBAC()
        assert rbac.role_of(99) == Role.USER

    def test_assign_and_role_of(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        assert rbac.role_of(1) == Role.KERNEL

    def test_user_can_read_files_not_kill(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.FILE_READ) is True
        assert rbac.check(10, Permission.PROCESS_KILL) is False
        assert rbac.check(10, Permission.DEVICE_DISK) is False
        assert rbac.check(10, Permission.TRAINING) is False

    def test_admin_has_device_and_training(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        for perm in (Permission.DEVICE_DISK, Permission.DEVICE_NET,
                     Permission.DEVICE_SERIAL, Permission.TRAINING,
                     Permission.PROCESS_KILL):
            assert rbac.check(20, perm) is True
        assert rbac.check(20, Permission.RAW_MEMORY) is False
        assert rbac.check(20, Permission.RAW_CPU) is False

    def test_kernel_has_every_permission(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        for perm in Permission:
            assert rbac.check(1, perm) is True

    def test_unknown_role_falls_back_to_user(self):
        rbac = X86RBAC()
        assert rbac.check(777, Permission.DEVICE_DISK) is False
        assert rbac.check(777, Permission.FILE_READ) is True

    def test_inherit_child_gets_parent_role(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.inherit(2, 1)
        assert rbac.role_of(2) == Role.KERNEL
        assert rbac.check(2, Permission.RAW_MEMORY) is True

    def test_inherit_unknown_parent_means_user(self):
        rbac = X86RBAC()
        rbac.inherit(2, 999)
        assert rbac.role_of(2) == Role.USER

    def test_escalate_requires_kernel_caller(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        rbac.assign(30, Role.USER)
        assert rbac.escalate(30, Role.ADMIN, caller_pid=20) is False
        assert rbac.role_of(30) == Role.USER

    def test_escalate_by_kernel_succeeds(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.assign(30, Role.USER)
        assert rbac.escalate(30, Role.KERNEL, caller_pid=1) is True
        assert rbac.role_of(30) == Role.KERNEL

    def test_escalate_unknown_caller_is_user(self):
        rbac = X86RBAC()
        assert rbac.escalate(30, Role.ADMIN, caller_pid=999) is False

    def test_escalate_same_role_by_kernel(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.assign(40, Role.ADMIN)
        assert rbac.escalate(40, Role.ADMIN, caller_pid=1) is True
