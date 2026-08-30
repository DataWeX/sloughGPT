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

    def test_role_enum_order(self):
        assert Role.USER < Role.ADMIN < Role.KERNEL

    def test_role_enum_values(self):
        assert Role.USER.value == 0
        assert Role.ADMIN.value == 1
        assert Role.KERNEL.value == 2

    def test_role_enum_count(self):
        assert len(Role) == 3

    def test_permission_enum_count(self):
        assert len(Permission) == 14

    def test_permission_auto_values(self):
        assert Permission.FILE_READ.value == 1
        assert Permission.FILE_WRITE.value == 2
        assert Permission.FILE_META.value == 3
        assert Permission.PROCESS_SPAWN.value == 4

    def test_user_file_write(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.FILE_WRITE) is True

    def test_user_file_meta(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.FILE_META) is True

    def test_user_process_spawn(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.PROCESS_SPAWN) is True

    def test_user_process_self(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.PROCESS_SELF) is True

    def test_user_no_device_mouse(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.DEVICE_MOUSE) is False

    def test_user_no_device_rtc(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.DEVICE_RTC) is False

    def test_user_no_raw_memory(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.RAW_MEMORY) is False

    def test_user_no_raw_cpu(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.RAW_CPU) is False

    def test_admin_all_user_permissions(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        user_perms = [Permission.FILE_READ, Permission.FILE_WRITE, Permission.FILE_META,
                      Permission.PROCESS_SPAWN, Permission.PROCESS_SELF]
        for perm in user_perms:
            assert rbac.check(20, perm) is True

    def test_admin_has_process_kill(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        assert rbac.check(20, Permission.PROCESS_KILL) is True

    def test_admin_has_device_serial(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        assert rbac.check(20, Permission.DEVICE_SERIAL) is True

    def test_admin_has_device_mouse(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        assert rbac.check(20, Permission.DEVICE_MOUSE) is True

    def test_admin_has_device_rtc(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        assert rbac.check(20, Permission.DEVICE_RTC) is True

    def test_admin_has_device_net(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        assert rbac.check(20, Permission.DEVICE_NET) is True

    def test_admin_has_training(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        assert rbac.check(20, Permission.TRAINING) is True

    def test_admin_no_raw_memory(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        assert rbac.check(20, Permission.RAW_MEMORY) is False

    def test_admin_no_raw_cpu(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        assert rbac.check(20, Permission.RAW_CPU) is False

    def test_escalate_user_to_admin_by_kernel(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.assign(50, Role.USER)
        assert rbac.escalate(50, Role.ADMIN, caller_pid=1) is True
        assert rbac.role_of(50) == Role.ADMIN
        assert rbac.check(50, Permission.DEVICE_DISK) is True

    def test_escalate_user_to_kernel_by_kernel(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.assign(50, Role.USER)
        assert rbac.escalate(50, Role.KERNEL, caller_pid=1) is True
        assert rbac.role_of(50) == Role.KERNEL
        assert rbac.check(50, Permission.RAW_MEMORY) is True

    def test_escalate_admin_to_kernel_by_kernel(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.assign(50, Role.ADMIN)
        assert rbac.escalate(50, Role.KERNEL, caller_pid=1) is True
        assert rbac.role_of(50) == Role.KERNEL

    def test_admin_cannot_escalate(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        rbac.assign(30, Role.USER)
        assert rbac.escalate(30, Role.ADMIN, caller_pid=20) is False

    def test_user_cannot_escalate(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        rbac.assign(30, Role.USER)
        assert rbac.escalate(30, Role.ADMIN, caller_pid=10) is False

    def test_assign_overwrites_role(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.role_of(10) == Role.USER
        rbac.assign(10, Role.ADMIN)
        assert rbac.role_of(10) == Role.ADMIN

    def test_inherit_from_admin(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        rbac.inherit(30, 20)
        assert rbac.role_of(30) == Role.ADMIN
        assert rbac.check(30, Permission.DEVICE_DISK) is True

    def test_inherit_chain(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.inherit(2, 1)
        rbac.inherit(3, 2)
        assert rbac.role_of(3) == Role.KERNEL

    def test_permission_all_values(self):
        expected = {
            "FILE_READ", "FILE_WRITE", "FILE_META",
            "PROCESS_SPAWN", "PROCESS_KILL", "PROCESS_SELF",
            "DEVICE_SERIAL", "DEVICE_MOUSE", "DEVICE_DISK", "DEVICE_RTC", "DEVICE_NET",
            "RAW_MEMORY", "RAW_CPU", "TRAINING",
        }
        actual = {p.name for p in Permission}
        assert expected == actual

    def test_role_all_values(self):
        expected = {"USER", "ADMIN", "KERNEL"}
        actual = {r.name for r in Role}
        assert expected == actual

    def test_multiple_processes_independent(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.USER)
        rbac.assign(2, Role.ADMIN)
        rbac.assign(3, Role.KERNEL)
        assert rbac.role_of(1) == Role.USER
        assert rbac.role_of(2) == Role.ADMIN
        assert rbac.role_of(3) == Role.KERNEL

    def test_inherit_then_check_permissions(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.ADMIN)
        rbac.inherit(2, 1)
        assert rbac.check(2, Permission.DEVICE_DISK) is True
        assert rbac.check(2, Permission.RAW_MEMORY) is False

    def test_escalate_then_check_permissions(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.assign(10, Role.USER)
        rbac.escalate(10, Role.ADMIN, caller_pid=1)
        assert rbac.check(10, Permission.DEVICE_DISK) is True
        assert rbac.check(10, Permission.RAW_MEMORY) is False

    def test_role_comparison_ge(self):
        assert Role.KERNEL >= Role.ADMIN
        assert Role.ADMIN >= Role.USER
        assert Role.USER >= Role.USER

    def test_role_comparison_le(self):
        assert Role.USER <= Role.ADMIN
        assert Role.ADMIN <= Role.KERNEL
        assert Role.KERNEL <= Role.KERNEL

    def test_permission_auto_values_sequential(self):
        assert Permission.PROCESS_KILL.value == 5
        assert Permission.PROCESS_SELF.value == 6
        assert Permission.DEVICE_SERIAL.value == 7
        assert Permission.DEVICE_MOUSE.value == 8
        assert Permission.DEVICE_DISK.value == 9
        assert Permission.DEVICE_RTC.value == 10
        assert Permission.DEVICE_NET.value == 11
        assert Permission.RAW_MEMORY.value == 12
        assert Permission.RAW_CPU.value == 13
        assert Permission.TRAINING.value == 14

    def test_pid_zero(self):
        rbac = X86RBAC()
        rbac.assign(0, Role.KERNEL)
        assert rbac.role_of(0) == Role.KERNEL
        assert rbac.check(0, Permission.RAW_MEMORY) is True

    def test_negative_pid(self):
        rbac = X86RBAC()
        rbac.assign(-1, Role.ADMIN)
        assert rbac.role_of(-1) == Role.ADMIN

    def test_escalate_from_user_role_denied(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        rbac.assign(20, Role.USER)
        assert rbac.escalate(20, Role.ADMIN, caller_pid=10) is False
        assert rbac.role_of(20) == Role.USER

    def test_escalate_from_admin_role_denied(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.ADMIN)
        rbac.assign(20, Role.USER)
        assert rbac.escalate(20, Role.KERNEL, caller_pid=10) is False
        assert rbac.role_of(20) == Role.USER

    def test_assign_overwrites_multiple_times(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        rbac.assign(10, Role.ADMIN)
        rbac.assign(10, Role.KERNEL)
        rbac.assign(10, Role.USER)
        assert rbac.role_of(10) == Role.USER

    def test_inherit_does_not_affect_parent(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.inherit(2, 1)
        rbac.assign(2, Role.USER)
        assert rbac.role_of(1) == Role.KERNEL
        assert rbac.role_of(2) == Role.USER

    def test_admin_has_device_disk(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        assert rbac.check(20, Permission.DEVICE_DISK) is True

    def test_admin_has_device_rtc(self):
        rbac = X86RBAC()
        rbac.assign(20, Role.ADMIN)
        assert rbac.check(20, Permission.DEVICE_RTC) is True

    def test_kernel_raw_cpu(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        assert rbac.check(1, Permission.RAW_CPU) is True

    def test_kernel_training(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        assert rbac.check(1, Permission.TRAINING) is True

    def test_user_no_training(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.TRAINING) is False

    def test_user_no_device_net(self):
        rbac = X86RBAC()
        rbac.assign(10, Role.USER)
        assert rbac.check(10, Permission.DEVICE_NET) is False

    def test_role_members(self):
        assert list(Role) == [Role.USER, Role.ADMIN, Role.KERNEL]

    def test_permission_is_intenum(self):
        from enum import IntEnum
        assert issubclass(Permission, IntEnum)

    def test_role_is_intenum(self):
        from enum import IntEnum
        assert issubclass(Role, IntEnum)
