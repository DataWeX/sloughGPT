"""
AI-Native System Call Interface — kernel services for processes.

Syscalls are how processes request kernel services: memory allocation,
device access, scheduling, inference, training, etc.
"""

from __future__ import annotations

import time
import logging
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Callable

from .kernel_process import Process, ProcessState

logger = logging.getLogger("slo.kernel.syscall")


class SyscallNumber(IntEnum):
    """System call numbers."""
    NONE = 0
    # Process management
    FORK = 1
    EXIT = 2
    WAIT = 3
    KILL = 4
    GETPID = 5
    GETPPID = 6
    SET_PRIORITY = 7
    # Memory
    MALLOC = 20
    TENSOR_ALLOC = 20  # alias for MALLOC
    FREE = 21
    READ_MEM = 22
    WRITE_MEM = 23
    # Devices
    OPEN_DEVICE = 40
    CLOSE_DEVICE = 41
    READ_DEVICE = 42
    WRITE_DEVICE = 43
    IOCTL_DEVICE = 44
    # Inference
    INFERENCE_START = 60
    INFERENCE_CANCEL = 61
    INFERENCE_RESULT = 62
    # Training
    TRAIN_START = 70
    TRAIN_STEP = 71
    TRAIN_STOP = 72
    TRAIN_STATUS = 73
    # Interrupts
    INT_ENABLE = 80
    INT_DISABLE = 81
    INT_FIRE = 82
    INT_WAIT = 83
    # Scheduler
    SCHED_YIELD = 90
    SCHED_SLEEP = 91
    SCHED_WAKE = 92
    # Neural
    NEURAL_FORWARD = 100
    NEURAL_BACKWARD = 101
    NEURAL_EMBED = 102
    NEURAL_ATTENTION = 103
    # I/O
    CONSOLE_READ = 110
    CONSOLE_WRITE = 111
    FILE_OPEN = 120
    FILE_READ = 121
    FILE_WRITE = 122
    FILE_CLOSE = 123
    # Misc
    UPTIME = 200
    STATS = 201
    YIELD = 202
    NOP = 255


@dataclass
class SyscallResult:
    """Result of a system call."""
    success: bool
    value: Any = None
    error: str | None = None
    errno: int = 0
    elapsed_ms: float = 0.0

    def __bool__(self) -> bool:
        return self.success

    @classmethod
    def ok(cls, value: Any = None) -> SyscallResult:
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, error: str = "", errno: int = 1) -> SyscallResult:
        return cls(success=False, error=error, errno=errno)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, None)


@dataclass
class SyscallEntry:
    """A registered syscall handler."""
    number: SyscallNumber
    name: str
    handler: Callable[..., Any]
    min_args: int = 0
    description: str = ""


class SyscallTable:
    """
    Syscall dispatch table — maps syscall numbers to handlers.

    Processes call syscalls by number. The table validates arguments,
    dispatches to the handler, and wraps the result.
    """

    def __init__(self):
        self._entries: dict[int, SyscallEntry] = {}
        self._call_count: dict[int, int] = {}
        self._last_call: dict[int, float] = {}
        self._pid_open_fds: dict[int, list[int]] = {}

    def register(self, number: SyscallNumber, name_or_handler: Any = None,
                 handler: Callable[..., Any] | None = None, min_args: int = 0,
                 description: str = "") -> None:
        """Register a syscall handler. Supports both 2-arg and 3-arg forms."""
        if handler is None:
            # 2-arg form: register(number, handler)
            handler = name_or_handler
            name = SyscallNumber(number).name if isinstance(number, SyscallNumber) else str(number)
        else:
            # 3-arg form: register(number, name, handler)
            name = name_or_handler
        entry = SyscallEntry(
            number=number,
            name=name,
            handler=handler,
            min_args=min_args,
            description=description,
        )
        self._entries[int(number)] = entry
        logger.debug("Registered syscall %d: %s", int(number), name)

    def unregister(self, number: SyscallNumber) -> None:
        self._entries.pop(int(number), None)

    def get_entry(self, number: SyscallNumber) -> SyscallEntry | None:
        return self._entries.get(int(number))

    def has(self, number: SyscallNumber) -> bool:
        return int(number) in self._entries

    def dispatch(self, first_arg: Any, second_arg: Any = None,
                 *args: Any, **kwargs: Any) -> SyscallResult:
        """
        Dispatch a syscall. Supports both calling conventions:
          - dispatch(caller, number) — test/old style (caller can be Kernel or Process)
          - dispatch(number, caller, *args, **kwargs) — new style
        """
        start = time.time()

        # Detect calling convention: if second arg is an int/SyscallNumber, first is caller
        if isinstance(second_arg, (int, SyscallNumber)):
            caller, number = first_arg, second_arg
        else:
            number, caller = first_arg, second_arg

        # Validate caller state (skip for Kernel objects which don't have .state)
        if hasattr(caller, 'state') and caller.state == ProcessState.STOPPED:
            return SyscallResult(
                success=False,
                error="Process is stopped",
                errno=1,
            )

        entry = self._entries.get(int(number))
        if entry is None:
            return SyscallResult(
                success=False,
                error=f"Unknown syscall: {number}",
                errno=2,
            )

        if len(args) < entry.min_args:
            return SyscallResult(
                success=False,
                error=f"Syscall {entry.name} requires {entry.min_args} args, got {len(args)}",
                errno=3,
            )

        try:
            raw = entry.handler(caller, *args, **kwargs)
            if isinstance(raw, SyscallResult):
                result = raw
            elif isinstance(raw, tuple) and len(raw) == 2:
                result = SyscallResult(success=raw[0], value=raw[1])
            elif isinstance(raw, tuple) and len(raw) == 3:
                result = SyscallResult(success=raw[0], value=raw[1], error=raw[2])
            else:
                result = SyscallResult(success=True, value=raw)
        except Exception as exc:
            result = SyscallResult(
                success=False,
                error=str(exc),
                errno=4,
            )
            logger.exception("Syscall %s failed", entry.name)

        result.elapsed_ms = (time.time() - start) * 1000
        self._call_count[int(number)] = self._call_count.get(int(number), 0) + 1
        self._last_call[int(number)] = time.time()
        return result

    def stats(self) -> dict:
        return {
            "registered": len(self._entries),
            "total_calls": sum(self._call_count.values()),
            "calls_by_number": {
                SyscallNumber(k).name: v
                for k, v in sorted(self._call_count.items())
            },
        }

    def list_syscalls(self) -> list[dict]:
        entries = []
        for num in sorted(self._entries):
            entry = self._entries[num]
            entries.append({
                "number": int(entry.number),
                "name": entry.name,
                "description": entry.description,
                "min_args": entry.min_args,
                "call_count": self._call_count.get(num, 0),
            })
        return entries


def _syscall_fork(caller: Process, *args: Any) -> tuple[bool, int]:
    """Fork: create a child process (returns child PID)."""
    return True, caller.pid + 1


def _syscall_exit(caller: Process, *args: Any) -> tuple[bool, None]:
    """Exit: stop the calling process."""
    caller.transition(ProcessState.STOPPED)
    return True, None


def _syscall_wait(caller: Process, *args: Any) -> tuple[bool, int]:
    """Wait: yield until child completes."""
    return True, 0


def _syscall_getpid(caller: Process, *args: Any) -> tuple[bool, int]:
    """Get PID of calling process."""
    return True, caller.pid


def _syscall_set_priority(caller: Process, priority: int, *args: Any) -> tuple[bool, None]:
    """Set process priority."""
    from .kernel_process import Priority
    for p in Priority:
        if int(p) == priority:
            caller.priority = p
            break
    return True, None


def _syscall_malloc(caller: Process, shape: tuple, dtype: str = "float32",
                    *args: Any) -> tuple[bool, int]:
    """Allocate tensor memory. Returns block_id."""
    return True, 0


def _syscall_free(caller: Process, block_id: int, *args: Any) -> tuple[bool, bool]:
    """Free tensor memory."""
    return True, True


def _syscall_sched_yield(caller: Process, *args: Any) -> tuple[bool, None]:
    """Yield CPU to scheduler."""
    caller.transition(ProcessState.READY)
    return True, None


def _syscall_sched_sleep(caller: Process, duration: float = 0,
                         *args: Any) -> tuple[bool, None]:
    """Put process to sleep."""
    caller.transition(ProcessState.WAITING)
    return True, None


def _syscall_uptime(caller: Process, *args: Any) -> tuple[bool, float]:
    """Get process uptime."""
    return True, caller.uptime


def _syscall_stats(caller: Process, *args: Any) -> tuple[bool, dict]:
    """Get process stats."""
    return True, {
        "pid": caller.pid,
        "name": caller.name,
        "state": caller.state.name,
        "uptime": caller.uptime,
        "memory_bytes": caller.memory_bytes,
        "inference_count": caller.inference_count,
        "tokens_generated": caller.tokens_generated,
    }


def _syscall_nop(caller: Process, *args: Any) -> tuple[bool, None]:
    """No operation."""
    return True, None


def _syscall_open_device(caller: Process, device_name: str,
                         *args: Any) -> tuple[bool, int]:
    """Open a device. Returns fd."""
    return True, 1


def _syscall_close_device(caller: Process, fd: int, *args: Any) -> tuple[bool, bool]:
    """Close a device fd."""
    return True, True


def _syscall_read_device(caller: Process, fd: int, size: int = -1,
                         *args: Any) -> tuple[bool, bytes]:
    """Read from device."""
    return True, b""


def _syscall_write_device(caller: Process, fd: int, data: Any,
                          *args: Any) -> tuple[bool, int]:
    """Write to device. Returns bytes written."""
    return True, 0


def _syscall_console_write(caller: Process, text: str, *args: Any) -> tuple[bool, int]:
    """Write to console."""
    return True, len(text)


def _syscall_console_read(caller: Process, *args: Any) -> tuple[bool, str]:
    """Read from console."""
    return True, ""


def _syscall_inference_start(caller: Process, prompt: str,
                             *args: Any) -> tuple[bool, int]:
    """Start inference. Returns job_id."""
    return True, 0


def _syscall_inference_cancel(caller: Process, job_id: int,
                              *args: Any) -> tuple[bool, bool]:
    """Cancel inference."""
    return True, True


def _syscall_train_start(caller: Process, config: dict | None = None,
                         *args: Any) -> tuple[bool, int]:
    """Start training. Returns job_id."""
    return True, 0


def _syscall_train_stop(caller: Process, job_id: int, *args: Any) -> tuple[bool, bool]:
    """Stop training."""
    return True, True


def _syscall_train_status(caller: Process, job_id: int,
                          *args: Any) -> tuple[bool, dict]:
    """Get training status."""
    return True, {"status": "idle"}


def build_default_syscall_table() -> SyscallTable:
    """Build the default syscall table with built-in handlers."""
    table = SyscallTable()

    # Process management
    table.register(SyscallNumber.FORK, "fork", _syscall_fork, description="Create child process")
    table.register(SyscallNumber.EXIT, "exit", _syscall_exit, description="Stop calling process")
    table.register(SyscallNumber.WAIT, "wait", _syscall_wait, description="Wait for child")
    table.register(SyscallNumber.GETPID, "getpid", _syscall_getpid, description="Get PID")
    table.register(SyscallNumber.SET_PRIORITY, "set_priority", _syscall_set_priority,
                   min_args=1, description="Set process priority")

    # Memory
    table.register(SyscallNumber.MALLOC, "malloc", _syscall_malloc,
                   min_args=1, description="Allocate tensor memory")
    table.register(SyscallNumber.FREE, "free", _syscall_free,
                   min_args=1, description="Free tensor memory")

    # Devices
    table.register(SyscallNumber.OPEN_DEVICE, "open_device", _syscall_open_device,
                   min_args=1, description="Open a device")
    table.register(SyscallNumber.CLOSE_DEVICE, "close_device", _syscall_close_device,
                   min_args=1, description="Close a device")
    table.register(SyscallNumber.READ_DEVICE, "read_device", _syscall_read_device,
                   min_args=1, description="Read from device")
    table.register(SyscallNumber.WRITE_DEVICE, "write_device", _syscall_write_device,
                   min_args=2, description="Write to device")

    # Scheduler
    table.register(SyscallNumber.SCHED_YIELD, "sched_yield", _syscall_sched_yield,
                   description="Yield CPU")
    table.register(SyscallNumber.SCHED_SLEEP, "sched_sleep", _syscall_sched_sleep,
                   description="Sleep for duration")

    # I/O
    table.register(SyscallNumber.CONSOLE_WRITE, "console_write", _syscall_console_write,
                   min_args=1, description="Write to console")
    table.register(SyscallNumber.CONSOLE_READ, "console_read", _syscall_console_read,
                   description="Read from console")

    # Inference
    table.register(SyscallNumber.INFERENCE_START, "inference_start", _syscall_inference_start,
                   min_args=1, description="Start inference")
    table.register(SyscallNumber.INFERENCE_CANCEL, "inference_cancel", _syscall_inference_cancel,
                   min_args=1, description="Cancel inference")

    # Training
    table.register(SyscallNumber.TRAIN_START, "train_start", _syscall_train_start,
                   description="Start training")
    table.register(SyscallNumber.TRAIN_STOP, "train_stop", _syscall_train_stop,
                   min_args=1, description="Stop training")
    table.register(SyscallNumber.TRAIN_STATUS, "train_status", _syscall_train_status,
                   min_args=1, description="Get training status")

    # Misc
    table.register(SyscallNumber.UPTIME, "uptime", _syscall_uptime, description="Get uptime")
    table.register(SyscallNumber.STATS, "stats", _syscall_stats, description="Get stats")
    table.register(SyscallNumber.NOP, "nop", _syscall_nop, description="No operation")

    return table


# Module-level convenience
_SYSCALL_TABLE: SyscallTable | None = None


def get_syscall_table() -> SyscallTable:
    global _SYSCALL_TABLE
    if _SYSCALL_TABLE is None:
        _SYSCALL_TABLE = build_default_syscall_table()
    return _SYSCALL_TABLE


def syscall(number: SyscallNumber, caller: Process, *args: Any,
            **kwargs: Any) -> SyscallResult:
    """Convenience function to dispatch a syscall using the global table."""
    return get_syscall_table().dispatch(number, caller, *args, **kwargs)


SYSCALLS = SyscallNumber
