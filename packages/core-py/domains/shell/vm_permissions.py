"""RBAC for the x86 virtual PC.

Three tiers:
  KERNEL — PID 1 only, unrestricted
  ADMIN  — device I/O, raw memory, network
  USER   — file I/O, basic computation, no device access

Principle: AI-native kernel processes can read any state.
x86 VM guests must be RBAC-gated.
"""

from __future__ import annotations

from enum import IntEnum, auto


class Role(IntEnum):
    USER = 0
    ADMIN = 1
    KERNEL = 2


class Permission(IntEnum):
    FILE_READ = auto()
    FILE_WRITE = auto()
    FILE_META = auto()
    PROCESS_SPAWN = auto()
    PROCESS_KILL = auto()
    PROCESS_SELF = auto()
    DEVICE_SERIAL = auto()
    DEVICE_MOUSE = auto()
    DEVICE_DISK = auto()
    DEVICE_RTC = auto()
    DEVICE_NET = auto()
    RAW_MEMORY = auto()
    RAW_CPU = auto()
    TRAINING = auto()


_ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.USER: frozenset({
        Permission.FILE_READ,
        Permission.FILE_WRITE,
        Permission.FILE_META,
        Permission.PROCESS_SELF,
        Permission.PROCESS_SPAWN,
    }),
    Role.ADMIN: frozenset({
        Permission.FILE_READ,
        Permission.FILE_WRITE,
        Permission.FILE_META,
        Permission.PROCESS_SPAWN,
        Permission.PROCESS_KILL,
        Permission.PROCESS_SELF,
        Permission.DEVICE_SERIAL,
        Permission.DEVICE_MOUSE,
        Permission.DEVICE_DISK,
        Permission.DEVICE_RTC,
        Permission.DEVICE_NET,
        Permission.TRAINING,
    }),
    Role.KERNEL: frozenset(Permission),
}


class X86RBAC:
    def __init__(self) -> None:
        self._process_role: dict[int, Role] = {}

    def assign(self, pid: int, role: Role) -> None:
        self._process_role[pid] = role

    def role_of(self, pid: int) -> Role:
        return self._process_role.get(pid, Role.USER)

    def check(self, pid: int, permission: Permission) -> bool:
        role = self.role_of(pid)
        return permission in _ROLE_PERMISSIONS[role]

    def inherit(self, child_pid: int, parent_pid: int) -> None:
        parent_role = self.role_of(parent_pid)
        self._process_role[child_pid] = parent_role

    def escalate(self, pid: int, target_role: Role, caller_pid: int) -> bool:
        caller_role = self.role_of(caller_pid)
        if caller_role != Role.KERNEL:
            return False
        self._process_role[pid] = target_role
        return True
