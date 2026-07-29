"""Tests for x86 VM RBAC layer."""

from __future__ import annotations

import pytest

from domains.shell.vm import X86VirtualSystem, X86Assembler, ProcessState
from domains.shell.vm_permissions import X86RBAC, Role, Permission


class TestX86RBAC:
    def test_user_cannot_access_devices(self):
        rbac = X86RBAC()
        rbac.assign(2, Role.USER)
        assert rbac.check(2, Permission.FILE_READ)
        assert not rbac.check(2, Permission.DEVICE_SERIAL)
        assert not rbac.check(2, Permission.DEVICE_DISK)
        assert not rbac.check(2, Permission.DEVICE_NET)

    def test_admin_can_access_devices(self):
        rbac = X86RBAC()
        rbac.assign(3, Role.ADMIN)
        assert rbac.check(3, Permission.DEVICE_SERIAL)
        assert rbac.check(3, Permission.DEVICE_DISK)
        assert rbac.check(3, Permission.DEVICE_NET)
        assert rbac.check(3, Permission.PROCESS_KILL)

    def test_kernel_has_all_permissions(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        for perm in Permission:
            assert rbac.check(1, perm)

    def test_unassigned_process_defaults_to_user(self):
        rbac = X86RBAC()
        assert rbac.role_of(99) == Role.USER
        assert not rbac.check(99, Permission.DEVICE_SERIAL)

    def test_inherit_child_gets_parent_role(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.inherit(2, 1)
        assert rbac.role_of(2) == Role.KERNEL
        rbac.assign(3, Role.USER)
        rbac.inherit(4, 3)
        assert rbac.role_of(4) == Role.USER

    def test_escalate_only_by_kernel(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.KERNEL)
        rbac.assign(2, Role.USER)
        rbac.assign(3, Role.ADMIN)
        assert not rbac.escalate(2, Role.ADMIN, 2)
        assert rbac.escalate(2, Role.ADMIN, 1)
        assert rbac.role_of(2) == Role.ADMIN


class TestX86VMIntegration:
    def test_kernel_has_kernel_role(self):
        vs = X86VirtualSystem()
        assert vs._syscall._rbac.role_of(vs._kernel.pid) == Role.KERNEL

    def test_user_process_starts_as_user(self):
        vs = X86VirtualSystem()
        pcb = vs._ptable.create(name="user_prog", priority=3)
        assert vs._syscall._rbac.role_of(pcb.pid) == Role.USER

    def _write_at(self, vs, addr: int, data: bytes):
        vs.cpu._mem[addr:addr + len(data)] = data

    def _run_process(self, vs, source: str, role=Role.USER):
        pid = vs.spawn("test", source)
        assert pid is not None
        vs._syscall._rbac.assign(pid, role)
        vs.scheduler.start(vs.cpu)
        current = vs.scheduler.current
        assert current is not None
        current.restore_to_cpu(vs.cpu)
        vs.cpu.run(max_steps=200)
        return vs.cpu._read32(0x5000)

    def test_serial_denied_for_user(self):
        vs = X86VirtualSystem()
        result = self._run_process(vs, """
            [BITS 32]
            mov eax, 19
            mov ebx, 65
            int 0x80
            mov [0x5000], eax
            hlt
        """, role=Role.USER)
        assert result == 0xFFFFFFFE, f"expected -2 (denied), got {result}"

    def test_user_can_open_file(self):
        vs = X86VirtualSystem()
        vs.filesystem.write("readme.txt", b"hello world")
        fname_addr = 0x80000
        self._write_at(vs, fname_addr, b"readme.txt\x00")
        result = self._run_process(vs, f"""
            [BITS 32]
            mov eax, 4
            mov ebx, {hex(fname_addr)}
            mov ecx, 0
            int 0x80
            mov [0x5000], eax
            hlt
        """, role=Role.USER)
        assert result not in (0xFFFFFFFF, 0xFFFFFFFE), f"open should succeed, got {result}"

    def test_disk_denied_for_user(self):
        vs = X86VirtualSystem()
        result = self._run_process(vs, """
            [BITS 32]
            mov eax, 23
            mov ebx, 0
            mov ecx, 1
            mov edx, 1
            int 0x80
            mov [0x5000], eax
            hlt
        """, role=Role.USER)
        assert result == 0xFFFFFFFE, f"expected -2 (denied), got {result}"

    def test_kernel_can_access_disk(self):
        vs = X86VirtualSystem()
        result = self._run_process(vs, """
            [BITS 32]
            mov eax, 23
            mov ebx, 0
            mov ecx, 1
            mov edx, 1
            int 0x80
            mov [0x5000], eax
            hlt
        """, role=Role.KERNEL)
        assert result != 0xFFFFFFFE, "kernel should be able to access disk"

    def test_escalate_thread_from_kernel(self):
        vs = X86VirtualSystem()
        pid = vs.spawn("worker", "hlt")
        assert pid is not None
        result = vs._syscall._rbac.escalate(pid, Role.ADMIN, vs._kernel.pid)
        assert result
        assert vs._syscall._rbac.role_of(pid) == Role.ADMIN

    def test_escalate_from_user_fails(self):
        vs = X86VirtualSystem()
        user_pid = vs.spawn("user_prog", "hlt")
        admin_pid = vs.spawn("helper", "hlt")
        assert user_pid is not None and admin_pid is not None
        result = vs._syscall._rbac.escalate(admin_pid, Role.ADMIN, user_pid)
        assert not result

    def test_getrole_syscall_user(self):
        vs = X86VirtualSystem()
        result = self._run_process(vs, """
            [BITS 32]
            mov eax, 27
            int 0x80
            mov [0x5000], eax
            hlt
        """, role=Role.USER)
        assert result == 0, f"expected 0 (USER), got {result}"

    def test_getrole_syscall_kernel(self):
        vs = X86VirtualSystem()
        result = self._run_process(vs, """
            [BITS 32]
            mov eax, 27
            int 0x80
            mov [0x5000], eax
            hlt
        """, role=Role.KERNEL)
        assert result == 2, f"expected 2 (KERNEL), got {result}"

    def _setup_scheduler_with_role(self, vs, role=Role.ADMIN):
        """Spawn a process with given role and make it the current scheduler process."""
        pid = vs.spawn("runner", "[BITS 32]\nhlt")
        assert pid is not None
        vs._syscall._rbac.assign(pid, role)
        vs.scheduler.start(vs.cpu)
        current = vs.scheduler.current
        assert current is not None
        assert current.pid == pid
        current.restore_to_cpu(vs.cpu)

    def test_admin_can_kill_user_process(self):
        vs = X86VirtualSystem()
        self._setup_scheduler_with_role(vs, Role.ADMIN)
        target = vs.spawn("victim", "[BITS 32]\nhlt")
        assert target is not None
        vs._syscall._rbac.assign(target, Role.USER)
        result = vs._syscall._sys_kill(target, 9)
        assert result == 0
        assert vs._ptable.get(target) is None

    def test_admin_cannot_kill_kernel_process(self):
        vs = X86VirtualSystem()
        kernel_pid = vs._kernel.pid
        self._setup_scheduler_with_role(vs, Role.ADMIN)
        result = vs._syscall._sys_kill(kernel_pid, 9)
        assert result == -1  # admin cannot kill kernel

    def test_user_cannot_kill_process_through_syscall(self):
        vs = X86VirtualSystem()
        target = vs.spawn("victim", "[BITS 32]\nhlt")
        assert target is not None
        result = self._run_process(vs, f"""
            [BITS 32]
            mov eax, 24
            mov ebx, {target}
            mov ecx, 9
            int 0x80
            mov [0x5000], eax
            hlt
        """, role=Role.USER)
        assert result == 0xFFFFFFFE, f"expected -2 (denied), got {result}"

    def test_fork_inherits_rbac_role(self):
        vs = X86VirtualSystem()
        self._setup_scheduler_with_role(vs, Role.ADMIN)
        child_pid = vs._syscall._sys_fork()
        assert child_pid > 0
        assert vs._syscall._rbac.role_of(child_pid) == Role.ADMIN

    def test_exec_preserves_rbac_role(self):
        vs = X86VirtualSystem()
        vs.filesystem.write("prog.asm", b"[BITS 32]\nhlt")
        fname_addr = 0x80000
        self._write_at(vs, fname_addr, b"prog.asm\x00")
        self._setup_scheduler_with_role(vs, Role.ADMIN)
        result = vs._syscall._sys_exec(fname_addr)
        assert result == 0
        assert vs._syscall._rbac.role_of(vs.scheduler.current.pid) == Role.ADMIN

    def test_all_syscalls_have_permission_map(self):
        vs = X86VirtualSystem()
        handler = vs._syscall
        handler._build_perm_map()
        syscall_nums = set()
        for attr in dir(handler):
            if attr.startswith("SYS_"):
                val = getattr(handler, attr)
                if isinstance(val, int):
                    syscall_nums.add(val)
        mapped = set(handler._perm_map.keys())
        unmapped = syscall_nums - mapped
        assert unmapped == set(), f"syscalls without permission map: {sorted(unmapped)}"

    def test_raw_memory_and_cpu_are_reserved(self):
        vs = X86VirtualSystem()
        vs._syscall._build_perm_map()
        for perm_name in ("RAW_MEMORY", "RAW_CPU"):
            perm = getattr(vs._syscall._Permission, perm_name)
            assert perm not in vs._syscall._perm_map.values(), \
                f"{perm_name} should not be mapped to any syscall"

    def test_kernel_can_kill_self(self):
        vs = X86VirtualSystem()
        self._setup_scheduler_with_role(vs, Role.KERNEL)
        current = vs.scheduler.current
        assert current is not None
        result = vs._syscall._sys_kill(current.pid, 9)
        assert result == 0

    def test_vmrun_user_write_stdout_allowed(self):
        vs = X86VirtualSystem()
        hello_addr = 0x100000 + 40
        self._write_at(vs, hello_addr, b"Hello from x86!\n\x00")
        pid = vs.spawn("test", f"""
            [BITS 32]
            mov eax, 3
            mov ebx, 1
            mov ecx, {hex(hello_addr)}
            mov edx, 16
            int 0x80
            mov [0x5000], eax
            hlt
        """)
        assert pid is not None
        vs._syscall._rbac.assign(pid, Role.USER)
        vs.scheduler.start(vs.cpu)
        current = vs.scheduler.current
        assert current is not None
        current.restore_to_cpu(vs.cpu)
        vs.cpu.run(max_steps=200)
        result = vs.cpu._read32(0x5000)
        assert result == 16, f"expected 16 bytes written, got {result}"
