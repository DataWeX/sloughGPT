"""
Process primitives — AI-native tasks, not threads.

A Process is an inference request, a training job, a data pipeline stage,
or any unit of work the kernel schedules and manages.
"""

from __future__ import annotations

import time
import threading
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Callable


class ProcessState(IntEnum):
    """Process lifecycle states."""
    CREATED = 0
    READY = 1
    RUNNING = 2
    WAITING = 3    # waiting on I/O, model load, or dependency
    STOPPED = 4
    ZOMBIE = 5     # completed but not yet reaped


class Priority(IntEnum):
    """Scheduling priority — lower value = higher priority."""
    CRITICAL = 0   # system services, interrupt handlers
    HIGH = 1       # real-time inference
    NORMAL = 2     # batch inference, training
    LOW = 3        # background data loading, preprocessing
    IDLE = 4       # garbage collection, cleanup


@dataclass
class TensorRef:
    """Reference to a tensor allocation in kernel memory."""
    block_id: int
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    owner_pid: int


@dataclass
class Process:
    """
    AI-native process — a unit of work the kernel manages.

    Each process has:
    - A PID and name
    - A state machine (CREATED → READY → RUNNING → WAITING/STOPPED → ZOMBIE)
    - Priority for scheduling
    - Memory ownership (tensor blocks)
    - A callable entry point and its arguments
    - Timing and resource usage stats
    """
    pid: int
    name: str
    state: ProcessState = ProcessState.CREATED
    priority: Priority = Priority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    # Entry point — the function the scheduler runs
    entry: Callable[..., Any] | None = None
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    # Memory ownership
    tensors: list[TensorRef] = field(default_factory=list)
    memory_bytes: int = 0

    # Resource usage
    cpu_time_ms: float = 0.0
    inference_count: int = 0
    tokens_generated: int = 0

    # Dependencies — process won't run until these PIDs complete
    depends_on: list[int] = field(default_factory=list)

    # Metadata — arbitrary key-value store for the process
    metadata: dict[str, Any] = field(default_factory=dict)

    # Result — set when process completes
    result: Any = None
    error: str | None = None

    # Thread handle (for actual execution)
    _thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def uptime(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def is_active(self) -> bool:
        return self.state in (ProcessState.CREATED, ProcessState.READY,
                              ProcessState.RUNNING, ProcessState.WAITING)

    @property
    def is_done(self) -> bool:
        return self.state in (ProcessState.STOPPED, ProcessState.ZOMBIE)

    def transition(self, new_state: ProcessState) -> None:
        """Transition to a new state, recording timestamps."""
        self.state = new_state
        if new_state == ProcessState.RUNNING and self.started_at is None:
            self.started_at = time.time()
        if new_state in (ProcessState.STOPPED, ProcessState.ZOMBIE):
            if self.finished_at is None:
                self.finished_at = time.time()

    def acquire_tensor(self, ref: TensorRef) -> None:
        """Register a tensor allocation owned by this process."""
        self.tensors.append(ref)
        self.memory_bytes += ref.size_bytes

    def release_tensor(self, block_id: int) -> TensorRef | None:
        """Release a tensor allocation. Returns the ref if found."""
        for i, ref in enumerate(self.tensors):
            if ref.block_id == block_id:
                self.tensors.pop(i)
                self.memory_bytes -= ref.size_bytes
                return ref
        return None

    def status_line(self) -> str:
        state_char = {
            ProcessState.CREATED: "C",
            ProcessState.READY: "R",
            ProcessState.RUNNING: "*",
            ProcessState.WAITING: "W",
            ProcessState.STOPPED: "S",
            ProcessState.ZOMBIE: "Z",
        }.get(self.state, "?")
        pri_char = {
            Priority.CRITICAL: "!",
            Priority.HIGH: "H",
            Priority.NORMAL: " ",
            Priority.LOW: "L",
            Priority.IDLE: ".",
        }.get(self.priority, " ")
        return f"[{self.pid:4d}] {state_char}{pri_char} {self.name:<30} {self.uptime:.1f}s"
