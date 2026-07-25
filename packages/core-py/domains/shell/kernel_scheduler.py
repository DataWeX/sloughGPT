"""Process scheduler for the AI-native kernel."""

from __future__ import annotations
import time
import logging
from typing import Callable, Optional

from .kernel_process import Process, ProcessState, Priority

logger = logging.getLogger("slo.kernel.scheduler")


class Scheduler:
    """Priority-based process scheduler with dependency tracking."""

    def __init__(self):
        self._processes: dict[int, Process] = {}
        self._queues: dict[str, list[int]] = {
            "realtime": [],
            "high": [],
            "normal": [],
            "low": [],
            "idle": [],
        }
        self._current_pid: Optional[int] = None
        self._callbacks: dict[str, Callable] = {}
        self._sleeping: dict[int, float] = {}

    @property
    def current_pid(self) -> Optional[int]:
        return self._current_pid

    @property
    def current_process(self) -> Optional[Process]:
        if self._current_pid is None:
            return None
        return self._processes.get(self._current_pid)

    @property
    def process_count(self) -> int:
        return len(self._processes)

    @property
    def active_count(self) -> int:
        return sum(1 for p in self._processes.values() if p.state == ProcessState.RUNNING)

    def set_callbacks(self, on_complete=None, on_interrupt=None):
        if on_complete:
            self._callbacks["complete"] = on_complete
        if on_interrupt:
            self._callbacks["interrupt"] = on_interrupt

    def add(self, process: Process) -> None:
        if process.state == ProcessState.CREATED:
            process.transition(ProcessState.READY)
        self._processes[process.pid] = process
        queue = self._priority_to_queue(process.priority if hasattr(process, "priority") else Priority.NORMAL)
        self._queues[queue].append(process.pid)
        logger.debug("Added pid=%d to queue=%s", process.pid, queue)

    def remove(self, pid: int) -> Optional[Process]:
        proc = self._processes.pop(pid, None)
        if proc is None:
            return None
        for queue in self._queues.values():
            if pid in queue:
                queue.remove(pid)
        if self._current_pid == pid:
            self._current_pid = None
        return proc

    def get(self, pid: int) -> Optional[Process]:
        return self._processes.get(pid)

    def list_all(self) -> list[Process]:
        return list(self._processes.values())

    def _pick_next(self) -> Optional[int]:
        for queue_name in ["realtime", "high", "normal", "low", "idle"]:
            queue = self._queues[queue_name]
            while queue:
                pid = queue.pop(0)
                proc = self._processes.get(pid)
                if proc is None:
                    continue
                if proc.state == ProcessState.STOPPED:
                    continue
                if pid in self._sleeping:
                    if time.time() < self._sleeping[pid]:
                        queue.append(pid)
                        continue
                    else:
                        del self._sleeping[pid]
                return pid
        return None

    def _deps_satisfied(self, proc: Process) -> bool:
        deps = proc.metadata.get("deps", []) if hasattr(proc, "metadata") else []
        for dep_pid in deps:
            dep = self._processes.get(dep_pid)
            if dep is not None and dep.state != ProcessState.STOPPED:
                return False
        return True

    def tick(self) -> Optional[Process]:
        if self._current_pid is not None:
            proc = self._processes.get(self._current_pid)
            if proc and proc.state == ProcessState.RUNNING:
                return proc

        next_pid = self._pick_next()
        if next_pid is None:
            return None

        proc = self._processes[next_pid]
        proc.state = ProcessState.RUNNING
        self._current_pid = next_pid
        return proc

    def wait_for(self, pid: int, timeout: float = 0) -> None:
        proc = self._processes.get(pid)
        if proc:
            proc.state = ProcessState.WAITING
        self._sleeping[pid] = time.time() + timeout if timeout > 0 else float("inf")

    def wake(self, pid: int) -> None:
        self._sleeping.pop(pid, None)
        proc = self._processes.get(pid)
        if proc and proc.state == ProcessState.WAITING:
            proc.state = ProcessState.READY
            queue = self._priority_to_queue(proc.priority if hasattr(proc, "priority") else Priority.NORMAL)
            self._queues[queue].append(pid)

    def complete(self, pid: int, result: dict | None = None) -> None:
        proc = self._processes.get(pid)
        if proc is None:
            return
        proc.state = ProcessState.ZOMBIE
        proc.result = result
        if self._current_pid == pid:
            self._current_pid = None
        if "complete" in self._callbacks:
            try:
                self._callbacks["complete"](proc, result)
            except Exception:
                logger.exception("Complete callback failed for pid=%d", pid)

    def reap(self, pid: int | None = None) -> list[Process] | Process | None:
        """Reap a specific zombie or all stopped zombies."""
        if pid is not None:
            proc = self._processes.get(pid)
            if proc is None or proc.state not in (ProcessState.ZOMBIE, ProcessState.STOPPED):
                return None
            self.remove(pid)
            return proc
        stopped = []
        for pid in list(self._processes.keys()):
            proc = self._processes[pid]
            if proc.state in (ProcessState.ZOMBIE, ProcessState.STOPPED):
                stopped.append(proc)
                self.remove(pid)
        return stopped

    def _finish(self, pid: int) -> None:
        proc = self._processes.get(pid)
        if proc:
            proc.state = ProcessState.STOPPED
        if self._current_pid == pid:
            self._current_pid = None

    def queue_sizes(self) -> dict[str, int]:
        return {name: len(q) for name, q in self._queues.items()}

    def stats(self) -> dict:
        return {
            "total_processes": self.process_count,
            "total": self.process_count,
            "active": self.active_count,
            "current_pid": self._current_pid,
            "queues": self.queue_sizes(),
            "sleeping": len(self._sleeping),
        }

    def _priority_to_queue(self, priority) -> str:
        if hasattr(priority, "value"):
            val = priority.value
        else:
            val = int(priority)
        if val <= 1:
            return "realtime"
        elif val <= 2:
            return "high"
        elif val <= 3:
            return "normal"
        elif val <= 4:
            return "low"
        return "idle"
