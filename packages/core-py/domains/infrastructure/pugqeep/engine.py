"""
PGQ Engine — vCPU for graph-structured infrastructure.

Spawn a single process that seeds child processes across network,
application, and protocol layers. Trees branch stems of tasks into
parallel instances. Points carry function-calling capacity.

Usage:
    from pugqeep.engine import Engine

    engine = Engine("main")

    # Spawn processes (queued for dispatch)
    proc = engine.spawn(my_function, arg1, arg2)

    # Route processes to specific trees
    engine.route("load_model", "data")
    engine.route("train", "training")

    # Run dispatch loop (auto-dispatches to trees)
    engine.run()

    # Or dispatch manually
    engine.dispatch()  # one-shot dispatch
    engine.run(poll_interval=0.5)  # continuous loop
"""

import logging
import multiprocessing
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .config import RestartPolicy

logger = logging.getLogger("slo.pugqeep.engine")

_MSG_READY = "__READY__"
_MSG_HEARTBEAT = "__HEARTBEAT__"
_MSG_ERROR = "__ERROR__"
_MSG_RESULT = "__RESULT__"


class ProcessStatus(Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StemStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TreeStatus(Enum):
    IDLE = "idle"
    BRANCHING = "branching"
    STOPPED = "stopped"


class SchedulingPolicy(Enum):
    """How the engine assigns ungrouped processes to trees."""

    ROUND_ROBIN = "round_robin"
    FIRST = "first"


@dataclass
class Process:
    """A unit of execution with lifecycle.

    A Process wraps a callable with args/kwargs and tracks its state
    through CREATED -> READY -> RUNNING -> COMPLETED/FAILED.
    """
    fn: Callable[..., Any]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    status: ProcessStatus = ProcessStatus.CREATED
    result: Any = None
    error: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    timeout: Optional[float] = None  # seconds, None = no timeout
    depends_on: List[str] = field(default_factory=list)
    _future: Optional[Future] = field(default=None, repr=False)
    _tree_name: Optional[str] = field(default=None, repr=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _priority: int = field(default=2, repr=False)
    _restart_count: int = field(default=0, repr=False)
    _pid: Optional[int] = field(default=None, repr=False)
    _last_heartbeat: Optional[float] = field(default=None, repr=False)
    _restart_policy: Optional["RestartPolicy"] = field(default=None, repr=False)

    def ready(self) -> None:
        self.status = ProcessStatus.READY

    def running(self) -> None:
        self.status = ProcessStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result: Any = None) -> None:
        self.status = ProcessStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def fail(self, error: str) -> None:
        self.status = ProcessStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def cancel(self) -> None:
        self.status = ProcessStatus.CANCELLED
        self.completed_at = time.time()
        self._cancel_event.set()

    def wait_cancel(self, timeout: float = None) -> None:
        deadline = time.time() + timeout if timeout else None
        while not self.is_cancelled:
            if deadline and time.time() > deadline:
                break
            time.sleep(0.01)

    @property
    def is_cancelled(self) -> bool:
        return self.status == ProcessStatus.CANCELLED

    @property
    def elapsed(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def is_done(self) -> bool:
        return self.status in (
            ProcessStatus.COMPLETED,
            ProcessStatus.FAILED,
            ProcessStatus.CANCELLED,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed": self.elapsed,
            "error": self.error,
            "restart_count": self._restart_count,
            "pid": self._pid,
            "timeout": self.timeout,
            "depends_on": self.depends_on,
            "is_done": self.is_done,
            "is_cancelled": self.is_cancelled,
        }


@dataclass
class Stem:
    """A branch of parallel execution from a Tree."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tree_id: str = ""
    processes: List[Process] = field(default_factory=list)
    status: StemStatus = StemStatus.CREATED
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    _done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def running(self) -> None:
        self.status = StemStatus.RUNNING

    def complete(self) -> None:
        self.status = StemStatus.COMPLETED
        self.completed_at = time.time()
        self._done_event.set()

    def fail(self) -> None:
        self.status = StemStatus.FAILED
        self.completed_at = time.time()
        self._done_event.set()

    @property
    def is_done(self) -> bool:
        return self.status in (StemStatus.COMPLETED, StemStatus.FAILED)

    @property
    def all_done(self) -> bool:
        return all(p.is_done for p in self.processes)

    def results(self) -> List[Any]:
        return [p.result for p in self.processes if p.status == ProcessStatus.COMPLETED]

    def errors(self) -> List[str]:
        return [p.error for p in self.processes if p.status == ProcessStatus.FAILED and p.error]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tree_id": self.tree_id,
            "status": self.status.value,
            "num_processes": len(self.processes),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class Tree:
    """Model instance that branches Stems of parallel tasks."""
    def __init__(self, name: str, max_stems: int = 8,
                 pool_workers: int = 4):
        self.name = name
        self.status = TreeStatus.IDLE
        self.max_stems = max_stems
        self._stems: Dict[str, Stem] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=pool_workers,
            thread_name_prefix=f"tree-{name}",
        )
        self._lock = threading.Lock()
        self._graph: Dict[str, Any] = {}

    def branch(self, processes: List[Process]) -> Stem:
        with self._lock:
            if len(self._stems) >= self.max_stems:
                raise RuntimeError(
                    f"Tree '{self.name}' at max stems ({self.max_stems})"
                )

        stem = Stem(tree_id=self.name, processes=processes)
        self._stems[stem.id] = stem
        self.status = TreeStatus.BRANCHING

        for proc in processes:
            proc.ready()
            future = self._pool.submit(self._execute, proc, stem)
            proc._future = future

        logger.debug("Tree[%s]: branched stem %s with %d processes",
                      self.name, stem.id, len(processes))
        return stem

    def _execute(self, proc: Process, stem: Stem) -> Any:
        proc.running()
        try:
            if proc.timeout is not None and proc.timeout > 0:
                result_container: List[Any] = []
                error_container: List[Optional[Exception]] = [None]

                def _target():
                    try:
                        result_container.append(proc.fn(*proc.args, **proc.kwargs))
                    except Exception as e:
                        error_container[0] = e

                worker = threading.Thread(target=_target, daemon=True)
                worker.start()
                worker.join(timeout=proc.timeout)

                if worker.is_alive():
                    proc.fail(f"timed out after {proc.timeout}s")
                elif error_container[0] is not None:
                    proc.fail(str(error_container[0]))
                else:
                    proc.complete(result_container[0])
            else:
                result = proc.fn(*proc.args, **proc.kwargs)
                proc.complete(result)
        except Exception as e:
            proc.fail(str(e))
        finally:
            if stem.all_done:
                if any(p.status == ProcessStatus.FAILED for p in stem.processes):
                    stem.fail()
                else:
                    stem.complete()
                with self._lock:
                    self._stems.pop(stem.id, None)
                if not self._stems:
                    self.status = TreeStatus.IDLE
        return proc.result

    def wait_stem(self, stem: Stem, timeout: Optional[float] = None) -> Stem:
        stem._done_event.wait(timeout=timeout)
        return stem

    def store(self, key: str, value: Any) -> None:
        self._graph[key] = value

    def recall(self, key: str) -> Optional[Any]:
        return self._graph.get(key)

    @property
    def active_stems(self) -> int:
        return len(self._stems)

    def shutdown(self) -> None:
        self.status = TreeStatus.STOPPED
        self._pool.shutdown(wait=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "active_stems": self.active_stems,
            "max_stems": self.max_stems,
            "graph_keys": list(self._graph.keys()),
        }


class GuardTree(Tree):
    """Tree that wraps processes in SubprocessProcess for subprocess isolation."""
    def __init__(self, name: str, config=None, max_stems: int = 8,
                 pool_workers: int = 4, default_timeout: float = None):
        super().__init__(name, max_stems=max_stems, pool_workers=pool_workers)
        self.subprocess_config = config
        self.default_timeout = default_timeout
        self._subprocesses: Dict[str, SubprocessProcess] = {}

    def branch(self, processes: List[Process]) -> Stem:
        for proc in processes:
            if self.subprocess_config and self.subprocess_config.enabled:
                sub = SubprocessProcess(proc, self.subprocess_config)
                self._subprocesses[proc.id] = sub
        return super().branch(processes)

    def _execute(self, proc: Process, stem: Stem) -> Any:
        sub = self._subprocesses.get(proc.id)
        if sub is not None:
            sub.start()
            sub.monitor()
            return proc.result
        return super()._execute(proc, stem)

    @property
    def subprocess_count(self) -> int:
        return len(self._subprocesses)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["subprocess_enabled"] = self.subprocess_config is not None and self.subprocess_config.enabled
        d["subprocess_count"] = self.subprocess_count
        return d

    def health(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "active_stems": self.active_stems,
            "subprocess_enabled": self.subprocess_config is not None and self.subprocess_config.enabled,
            "subprocess_count": self.subprocess_count,
        }


class EngineMetrics:
    """Track engine-wide metrics: spawned, completed, failed, etc."""
    def __init__(self):
        self._lock = threading.Lock()
        self._spawned = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._timed_out = 0
        self._restarted = 0
        self._dispatched = 0
        self._total_latency = 0.0
        self._start_time = time.monotonic()

    def record_spawn(self) -> None:
        with self._lock:
            self._spawned += 1

    def record_complete(self, proc: Process = None) -> None:
        with self._lock:
            self._completed += 1
            if proc and proc.elapsed is not None:
                self._total_latency += proc.elapsed

    def record_fail(self, proc: Process = None) -> None:
        with self._lock:
            self._failed += 1

    def record_cancel(self) -> None:
        with self._lock:
            self._cancelled += 1

    def record_timeout(self) -> None:
        with self._lock:
            self._timed_out += 1

    def record_restart(self) -> None:
        with self._lock:
            self._restarted += 1

    def record_dispatch(self, count: int = 1) -> None:
        with self._lock:
            self._dispatched += count

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            total = self._completed + self._failed
            return {
                "spawned": self._spawned,
                "completed": self._completed,
                "failed": self._failed,
                "cancelled": self._cancelled,
                "timed_out": self._timed_out,
                "restarted": self._restarted,
                "dispatched": self._dispatched,
                "avg_latency_s": self._total_latency / max(1, self._completed),
                "throughput_per_s": self._completed / max(0.001, elapsed),
                "error_rate": self._failed / max(1, total),
            }

    def reset(self) -> None:
        with self._lock:
            self._spawned = 0
            self._completed = 0
            self._failed = 0
            self._cancelled = 0
            self._timed_out = 0
            self._restarted = 0
            self._dispatched = 0
            self._total_latency = 0.0
            self._start_time = time.monotonic()


class SubprocessProcess:
    """Wraps a Process in an isolated OS subprocess.

    Runs a Process.fn in a subprocess with:
    - Memory limits via resource.setrlimit
    - CPU affinity
    - Working directory (cwd)
    - Environment variables
    - Stdout/stderr capture
    - Graceful SIGTERM->SIGKILL termination
    - Timeout watchdog thread
    """

    def __init__(self, proc: Process, config):
        self.proc = proc
        self.config = config
        self._process: Optional[multiprocessing.Process] = None
        self._parent_conn = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._lock = threading.Lock()
        self._watchdog: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._last_heartbeat: Optional[float] = None
        self._stdout: Optional[str] = None
        self._stderr: Optional[str] = None

    def start(self) -> None:
        import os
        parent_r, child_w = multiprocessing.Pipe(duplex=False)
        self._start_time = time.monotonic()
        capture = self.config.capture_output

        def _worker():
            import sys
            import io

            stdout_capture = None
            stderr_capture = None

            if capture:
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

            try:
                if self.config.memory_limit_mb is not None:
                    try:
                        import resource
                        limit_bytes = self.config.memory_limit_mb * 1024 * 1024
                        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
                    except (ImportError, ValueError, OSError):
                        pass

                if self.config.cpu_affinity is not None:
                    try:
                        os.sched_setaffinity(0, self.config.cpu_affinity)
                    except (AttributeError, OSError):
                        pass

                if self.config.cwd is not None:
                    try:
                        os.chdir(self.config.cwd)
                    except (OSError, FileNotFoundError):
                        pass

                if self.config.env is not None:
                    try:
                        os.environ.update(self.config.env)
                    except (TypeError, OSError):
                        pass

                try:
                    child_w.send(_MSG_READY)
                except Exception:
                    pass

                result = self.proc.fn(*self.proc.args, **self.proc.kwargs)

                try:
                    child_w.send(("ok", result))
                except Exception:
                    logger.debug("Failed to send result to parent", exc_info=True)
            except Exception as e:
                try:
                    child_w.send(("error", str(e)))
                except Exception:
                    logger.debug("Failed to send error to parent", exc_info=True)
            finally:
                if capture:
                    try:
                        child_w.send(("stdout", stdout_capture.getvalue() if stdout_capture else ""))
                        child_w.send(("stderr", stderr_capture.getvalue() if stderr_capture else ""))
                    except Exception:
                        pass
                try:
                    child_w.close()
                except Exception:
                    pass

        start_method = self.config.start_method or "fork"
        ctx = multiprocessing.get_context(start_method)
        self._process = ctx.Process(target=_worker, daemon=True)
        self._process.start()
        self.proc._pid = self._process.pid
        self.proc.running()

        self._reader_thread = threading.Thread(
            target=self._read_result, args=(parent_r,),
            daemon=True, name=f"reader-{self.proc.name}",
        )
        self._reader_thread.start()

        if self.proc.timeout is not None and self.proc.timeout > 0:
            self._watchdog = threading.Thread(
                target=self._watchdog_loop, daemon=True,
                name=f"watchdog-{self.proc.name}",
            )
            self._watchdog.start()

    def _read_result(self, conn) -> None:
        try:
            while True:
                if conn.poll(0.5):
                    msg = conn.recv()
                    if isinstance(msg, tuple) and len(msg) == 2:
                        status, payload = msg
                        if status == "ok":
                            self.proc.complete(payload)
                        elif status == "stdout":
                            self._stdout = payload
                        elif status == "stderr":
                            self._stderr = payload
                        elif self._cancel_event.is_set():
                            self.proc.cancel()
                        elif status == "error":
                            self.proc.fail(payload)
                        if status in ("ok", "error"):
                            return
                    elif msg == _MSG_READY:
                        self._last_heartbeat = time.monotonic()
                    elif msg == _MSG_HEARTBEAT:
                        self._last_heartbeat = time.monotonic()
                if self._process and not self._process.is_alive():
                    break
        except (EOFError, OSError):
            pass
        finally:
            if self._cancel_event.is_set() and not self.proc.is_done:
                self.proc.cancel()
            elif not self.proc.is_done:
                self.proc.fail("pipe closed unexpectedly")
            try:
                conn.close()
            except Exception:
                pass

    def monitor(self) -> None:
        if self._process is not None:
            self._process.join()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
        self._end_time = time.monotonic()
        if not self.proc.is_done:
            if self._process and self._process.exitcode == 0:
                self.proc.complete()
            else:
                self.proc.fail(f"exit code {self._process.exitcode}" if self._process else "no process")

    def _watchdog_loop(self) -> None:
        while not self._cancel_event.is_set():
            if self._process is None or not self._process.is_alive():
                break
            elapsed = time.monotonic() - (self._start_time or 0)
            if self.proc.timeout and elapsed > self.proc.timeout:
                self.terminate()
                return
            self._cancel_event.wait(0.5)

    def terminate(self) -> None:
        if self._process is None or not self._process.is_alive():
            return
        self._cancel_event.set()
        try:
            self._process.terminate()
            self._process.join(timeout=self.config.terminate_grace)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
        except Exception:
            pass
        self._end_time = time.monotonic()
        if not self.proc.is_done:
            self.proc.cancel()
        elif self.proc.status == ProcessStatus.FAILED:
            self.proc.status = ProcessStatus.CANCELLED
            self.proc.error = None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def elapsed(self) -> Optional[float]:
        end = self._end_time or time.monotonic()
        return end - (self._start_time or end) if self._start_time else None

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

    def cancel(self) -> None:
        self._cancel_event.set()

    def health(self) -> dict:
        return {
            "pid": self.proc._pid,
            "alive": self.is_alive,
            "elapsed": self.elapsed,
        }

    def resource_usage(self) -> Optional[dict]:
        if self._process is None or self._process.pid is None:
            return None
        try:
            import resource
            if self._process.is_alive():
                self._process.join(timeout=0.1)
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            return {
                "ru_maxrss": usage.ru_maxrss,
                "ru_utime": usage.ru_utime,
                "ru_stime": usage.ru_stime,
                "ru_nvcsw": usage.ru_nvcsw,
                "ru_nivcsw": usage.ru_nivcsw,
            }
        except (ImportError, OSError, TypeError):
            return None

    @property
    def stdout(self) -> Optional[str]:
        return self._stdout

    @property
    def stderr(self) -> Optional[str]:
        return self._stderr


class ProcessGroup:
    """Batch operations on a set of processes."""
    def __init__(self, name: str, engine: "Engine" = None):
        self.name = name
        self.engine = engine
        self._processes: List[Process] = []
        self._done_event = threading.Event()

    def add(self, proc: Process) -> None:
        self._processes.append(proc)

    def spawn(self, fn, *args, **kwargs) -> Process:
        if self.engine is None:
            raise RuntimeError("No engine attached to ProcessGroup")
        proc = self.engine.spawn(fn, *args, **kwargs)
        self._processes.append(proc)
        return proc

    @property
    def num_processes(self) -> int:
        return len(self._processes)

    @property
    def all_done(self) -> bool:
        return all(p.is_done for p in self._processes)

    @property
    def elapsed(self) -> Optional[float]:
        starts = [p.started_at for p in self._processes if p.started_at]
        ends = [p.completed_at or time.time() for p in self._processes]
        if not starts:
            return None
        return max(ends) - min(starts)

    def results(self) -> List[Any]:
        return [p.result for p in self._processes if p.status == ProcessStatus.COMPLETED]

    def errors(self) -> List[str]:
        return [p.error for p in self._processes if p.status == ProcessStatus.FAILED and p.error]

    def cancel(self) -> int:
        count = 0
        for p in self._processes:
            if not p.is_done:
                p.cancel()
                count += 1
        return count

    def wait(self, timeout: float = None) -> None:
        deadline = time.time() + timeout if timeout else None
        while not self.all_done:
            if deadline and time.time() > deadline:
                break
            time.sleep(0.05)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_processes": self.num_processes,
            "elapsed": self.elapsed,
            "all_done": self.all_done,
            "status_counts": {
                s.value: sum(1 for p in self._processes if p.status == s)
                for s in ProcessStatus
            },
        }


class ProcessMonitor:
    """Background thread for stall detection and restart callbacks."""
    def __init__(self, config=None, restart_policy=None,
                 poll_interval: float = 1.0, stall_timeout: float = 30.0):
        self.config = config
        self.restart_policy = restart_policy
        self.poll_interval = config.poll_interval if config else poll_interval
        self.stall_timeout = config.stall_timeout if config else stall_timeout
        self._processes: Dict[str, Process] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._on_stall: List[Callable] = []
        self._on_restart: List[Callable] = []
        self._restart_count: Dict[str, int] = {}

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._processes)

    def track(self, proc: Process) -> None:
        with self._lock:
            self._processes[proc.id] = proc

    def untrack(self, proc_id: str) -> None:
        with self._lock:
            self._processes.pop(proc_id, None)

    def on_stall(self, callback: Callable) -> None:
        self._on_stall.append(callback)

    def on_restart(self, callback: Callable) -> None:
        self._on_restart.append(callback)

    def _restart_delay(self, attempt: int) -> float:
        if self.restart_policy is None:
            return 1.0
        base = self.restart_policy.restart_delay
        if self.restart_policy.backoff == "exponential":
            delay = base * (2 ** attempt)
        elif self.restart_policy.backoff == "linear":
            delay = base * (attempt + 1)
        else:
            delay = base
        return min(delay, self.restart_policy.max_backoff)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="process-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            time.sleep(self.poll_interval)
            with self._lock:
                for proc in list(self._processes.values()):
                    if proc.status == ProcessStatus.RUNNING:
                        stalled = False
                        if proc._last_heartbeat is not None:
                            since_heartbeat = time.monotonic() - proc._last_heartbeat
                            if since_heartbeat > self.stall_timeout:
                                stalled = True
                        else:
                            elapsed = proc.elapsed
                            if elapsed is not None and elapsed > self.stall_timeout:
                                stalled = True
                        if stalled:
                            for cb in self._on_stall:
                                try:
                                    cb(proc)
                                except Exception:
                                    logger.debug("Non-critical pugqeep stall callback error", exc_info=True)
                    if proc.status == ProcessStatus.FAILED:
                        policy = proc._restart_policy or self.restart_policy
                        if policy and policy.max_restarts > 0:
                            count = proc._restart_count
                            if count < policy.max_restarts:
                                proc._restart_count = count + 1
                                self._restart_count[proc.id] = proc._restart_count
                                for cb in self._on_restart:
                                    try:
                                        cb(proc)
                                    except Exception:
                                        logger.debug("Non-critical pugqeep restart callback error", exc_info=True)

    def stats(self) -> dict:
        with self._lock:
            running = sum(1 for p in self._processes.values() if p.status == ProcessStatus.RUNNING)
            failed = sum(1 for p in self._processes.values() if p.status == ProcessStatus.FAILED)
            return {
                "monitored": len(self._processes),
                "running": running,
                "failed": failed,
                "restarts": sum(self._restart_count.values()),
            }

    def stalled_processes(self) -> list:
        stalled = []
        with self._lock:
            for proc in self._processes.values():
                if proc.status != ProcessStatus.RUNNING:
                    continue
                if proc._last_heartbeat is not None:
                    since_heartbeat = time.monotonic() - proc._last_heartbeat
                    if since_heartbeat > self.stall_timeout:
                        stalled.append(proc)
                else:
                    elapsed = proc.elapsed
                    if elapsed is not None and elapsed > self.stall_timeout:
                        stalled.append(proc)
        return stalled

    def get_restart_count(self, proc_id: str) -> int:
        with self._lock:
            return self._restart_count.get(proc_id, 0)

    def reset_restart_count(self, proc_id: str) -> None:
        with self._lock:
            self._restart_count.pop(proc_id, None)


class ResultCache:
    """LRU + TTL cache for deduplicating identical function calls."""
    def __init__(self, maxsize: int = 128, ttl: float = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _key(self, fn: Callable, args: tuple, kwargs: dict) -> str:
        return f"{fn.__name__}:{args}:{sorted(kwargs.items())}"

    def get(self, fn: Callable, args: tuple, kwargs: dict):
        key = self._key(fn, args, kwargs)
        with self._lock:
            if key in self._cache:
                if self.ttl and time.monotonic() - self._timestamps[key] > self.ttl:
                    del self._cache[key]
                    del self._timestamps[key]
                    return False, None
                return True, self._cache[key]
        return False, None

    def put(self, fn: Callable, args: tuple, kwargs: dict, result: Any) -> None:
        key = self._key(fn, args, kwargs)
        with self._lock:
            if len(self._cache) >= self.maxsize:
                oldest = min(self._timestamps, key=self._timestamps.get)
                del self._cache[oldest]
                del self._timestamps[oldest]
            self._cache[key] = result
            self._timestamps[key] = time.monotonic()

    def invalidate(self, fn: Callable = None) -> int:
        with self._lock:
            if fn is None:
                count = len(self._cache)
                self._cache.clear()
                self._timestamps.clear()
                return count
            count = 0
            to_remove = [k for k in self._cache if fn.__name__ in k]
            for k in to_remove:
                del self._cache[k]
                del self._timestamps[k]
                count += 1
            return count

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._timestamps.clear()
            return count

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "ttl": self.ttl,
            }

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class Engine:
    """Core infra engine — the vCPU.

    Spawns the main process that seeds child processes across
    network, application, and protocol layers. Trees branch stems
    of parallel tasks. Points carry function-calling capacity.
    """

    def __init__(self, name: str = "main", max_trees: int = 16, config=None):
        if config is not None:
            self.name = config.name
            self.max_trees = config.max_trees
            self._config = config
        else:
            self.name = name
            self.max_trees = max_trees
            self._config = None
        self._trees: Dict[str, Tree] = {}
        self._processes: Dict[str, Process] = {}
        self._pending: List[Process] = []
        self._running = False
        self._lock = threading.Lock()
        self._routing: Dict[str, str] = {}
        self._default_tree: Optional[str] = None
        self._on_complete: List[Callable[[Process], None]] = []
        self._completed: List[Process] = []
        self._dispatch_batch_size: int = 8
        self._round_robin_idx: int = 0
        self._scheduling_policy: SchedulingPolicy = SchedulingPolicy.ROUND_ROBIN
        self._dependents: Dict[str, List[str]] = {}
        self._spawn_queue = None
        self._metrics = EngineMetrics()
        self._cache: Optional[ResultCache] = None
        self._monitor: Optional[ProcessMonitor] = None
        self._signal_handlers_installed = False
        self._old_signal_handlers: Dict = {}

        if self._config and self._config.monitor.enabled:
            self._monitor = ProcessMonitor(
                poll_interval=self._config.monitor.poll_interval,
                stall_timeout=self._config.monitor.stall_timeout,
            )
            self._monitor.start()

        self.install_signal_handlers()

    @property
    def metrics(self) -> EngineMetrics:
        return self._metrics

    def set_scheduling(self, policy: SchedulingPolicy) -> None:
        """Set how ungrouped processes are routed to trees."""
        if not isinstance(policy, SchedulingPolicy):
            raise TypeError("policy must be a SchedulingPolicy")
        self._scheduling_policy = policy

    def spawn(self, fn: Callable[..., Any], *args: Any,
              name: str = "", tree: Optional[str] = None,
              priority: int = 2, timeout: Optional[float] = None,
              depends_on: Optional[List[str]] = None,
              subprocess: bool = False,
              register_cancel: bool = False,
              **kwargs: Any) -> Process:
        proc = Process(fn=fn, args=args, kwargs=kwargs, name=name, timeout=timeout)
        proc._priority = priority
        if tree:
            proc._tree_name = tree
        if depends_on:
            proc.depends_on = list(depends_on)
            for dep_id in depends_on:
                self._dependents.setdefault(dep_id, []).append(proc.id)
        self._processes[proc.id] = proc
        self._pending.append(proc)
        self._metrics.record_spawn()

        if register_cancel and self._monitor:
            self._monitor.track(proc)

        if self._spawn_queue is not None:
            self._spawn_queue.put(proc, priority=priority)

        logger.debug("Engine[%s]: spawned process %s (%s) -> pending",
                      self.name, proc.id, proc.name or fn.__name__)
        return proc

    def tree(self, name: str, max_stems: int = 8,
             pool_workers: int = 4, guarded: bool = False,
             default_timeout: float = None) -> Tree:
        with self._lock:
            if len(self._trees) >= self.max_trees:
                raise RuntimeError(
                    f"Engine '{self.name}' at max trees ({self.max_trees})"
                )
            if guarded:
                config = self._config.subprocess if self._config else None
                tree = GuardTree(name, config=config, max_stems=max_stems,
                                pool_workers=pool_workers, default_timeout=default_timeout)
            else:
                tree = Tree(name, max_stems=max_stems, pool_workers=pool_workers)
            self._trees[name] = tree
            if self._default_tree is None:
                self._default_tree = name
        logger.debug("Engine[%s]: created tree '%s'", self.name, name)
        return tree

    def route(self, process_name: str, tree_name: str) -> None:
        if tree_name not in self._trees:
            raise ValueError(f"Tree '{tree_name}' not found")
        self._routing[process_name] = tree_name
        logger.debug("Engine[%s]: route '%s' -> tree '%s'",
                      self.name, process_name, tree_name)

    def on_complete(self, callback: Callable[[Process], None]) -> None:
        self._on_complete.append(callback)

    def branch(self, tree_name: str, processes: List[Process]) -> Stem:
        tree = self._trees.get(tree_name)
        if tree is None:
            raise ValueError(f"Tree '{tree_name}' not found")
        stem = tree.branch(processes)
        for proc in processes:
            proc._tree_name = tree_name
        return stem

    def dispatch(self) -> int:
        if not self._pending:
            return 0

        dispatchable: List[Process] = []
        held: List[Process] = []
        for proc in self._pending:
            if proc.depends_on and not self._deps_met(proc):
                held.append(proc)
            else:
                dispatchable.append(proc)

        if not dispatchable:
            return 0

        dispatchable.sort(key=lambda p: p._priority)

        groups: Dict[str, List[Process]] = {}
        ungrouped: List[Process] = []

        for proc in dispatchable:
            tree_name = proc._tree_name or self._routing.get(proc.name)
            if tree_name:
                proc._tree_name = tree_name
                groups.setdefault(tree_name, []).append(proc)
            else:
                ungrouped.append(proc)

        if ungrouped:
            tree_names = list(self._trees.keys())
            if tree_names:
                for proc in ungrouped:
                    tree_name = tree_names[self._round_robin_idx % len(tree_names)]
                    proc._tree_name = tree_name
                    groups.setdefault(tree_name, []).append(proc)
                    self._round_robin_idx += 1

        dispatched = 0
        for tree_name, procs in groups.items():
            tree = self._trees.get(tree_name)
            if tree is None:
                logger.warning("Engine[%s]: tree '%s' not found, skipping %d processes",
                               self.name, tree_name, len(procs))
                for p in procs:
                    p.fail(f"tree '{tree_name}' not found")
                continue

            for i in range(0, len(procs), self._dispatch_batch_size):
                batch = procs[i:i + self._dispatch_batch_size]
                try:
                    tree.branch(batch)
                    dispatched += len(batch)
                except RuntimeError as e:
                    logger.error("Engine[%s]: failed to dispatch to '%s': %s",
                                 self.name, tree_name, e)
                    for p in batch:
                        p.fail(str(e))

        self._pending = held
        self._metrics.record_dispatch(dispatched)
        return dispatched

    def run(self, poll_interval: float = 0.1,
            on_progress: Optional[Callable[[dict], None]] = None) -> None:
        self._running = True
        logger.info("Engine[%s]: starting main loop", self.name)

        while self._running:
            if self._pending and self._spawn_queue is None:
                dispatched = self.dispatch()
                if dispatched > 0:
                    logger.info("Engine[%s]: dispatched %d processes",
                                self.name, dispatched)

            with self._lock:
                active = sum(t.active_stems for t in self._trees.values())

            for proc in list(self._processes.values()):
                if proc.is_done and proc not in self._completed:
                    self._completed.append(proc)
                    if proc.status == ProcessStatus.COMPLETED:
                        self._metrics.record_complete(proc)
                    elif proc.status == ProcessStatus.FAILED:
                        self._metrics.record_fail(proc)
                    for cb in self._on_complete:
                        try:
                            cb(proc)
                        except Exception as e:
                            logger.error("Engine[%s]: on_complete callback error: %s",
                                         self.name, e)

            if on_progress is not None:
                try:
                    on_progress({
                        "pending": len(self._pending),
                        "active_stems": active,
                        "completed": len(self._completed),
                        "running": sum(1 for p in self._processes.values()
                                       if p.status == ProcessStatus.RUNNING),
                        "failed": sum(1 for p in self._processes.values()
                                      if p.status == ProcessStatus.FAILED),
                    })
                except Exception as e:
                    logger.error("Engine[%s]: on_progress callback error: %s",
                                 self.name, e)

            time.sleep(poll_interval)

        logger.info("Engine[%s]: main loop stopped", self.name)

    def run_background(self, poll_interval: float = 0.1,
                       as_future: bool = False) -> Union[threading.Thread, Future]:
        if as_future:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"engine-{self.name}")
            future = executor.submit(self.run, poll_interval)
            future.add_done_callback(lambda _: executor.shutdown(wait=False))
            return future

        thread = threading.Thread(
            target=self.run,
            args=(poll_interval,),
            name=f"engine-{self.name}",
            daemon=True,
        )
        thread.start()
        return thread

    def wait(self, timeout: Optional[float] = None) -> None:
        deadline = time.time() + timeout if timeout else None
        while True:
            has_pending = bool(self._pending)
            has_running = any(
                not p.is_done
                for p in self._processes.values()
            )
            if not has_pending and not has_running:
                break
            if deadline and time.time() > deadline:
                break
            time.sleep(0.05)

    def wait_all(self, timeout: Optional[float] = None) -> List[Process]:
        self.wait(timeout=timeout)
        return [p for p in self._processes.values() if p.is_done]

    def get_completed(self) -> List[Process]:
        done = list(self._completed)
        self._completed.clear()
        return done

    def stop(self) -> None:
        self._running = False
        self.stop_workers()
        for proc in self._processes.values():
            if not proc.is_done:
                proc.cancel()
                self._metrics.record_cancel()
        if self._monitor:
            self._monitor.stop()
        for tree in self._trees.values():
            tree.shutdown()

    def get_process(self, proc_id: str) -> Optional[Process]:
        return self._processes.get(proc_id)

    def get_tree(self, name: str) -> Optional[Tree]:
        return self._trees.get(name)

    def list_trees(self) -> List[str]:
        return list(self._trees.keys())

    def list_processes(self, status: Optional[ProcessStatus] = None) -> List[Process]:
        procs = list(self._processes.values())
        if status:
            procs = [p for p in procs if p.status == status]
        return procs

    def wait_for(self, proc_id: str, timeout: float = None) -> Optional[Process]:
        proc = self._processes.get(proc_id)
        if proc is None:
            raise KeyError(f"Process '{proc_id}' not found")
        deadline = time.time() + timeout if timeout else None
        while not proc.is_done:
            if deadline and time.time() > deadline:
                return None
            time.sleep(0.05)
        return proc

    def wait_for_any(self, proc_ids: List[str], timeout: float = None) -> Optional[Process]:
        deadline = time.time() + timeout if timeout else None
        while True:
            for pid in proc_ids:
                proc = self._processes.get(pid)
                if proc and proc.is_done:
                    return proc
            if deadline and time.time() > deadline:
                return None
            time.sleep(0.05)

    def cancel_process(self, proc_id: str, propagate: bool = True) -> int:
        proc = self._processes.get(proc_id)
        if proc is None:
            return 0
        count = 0
        if not proc.is_done:
            proc.cancel()
            self._metrics.record_cancel()
            count += 1
        if propagate:
            for child_id in proc.children_ids:
                child = self._processes.get(child_id)
                if child and not child.is_done:
                    child.cancel()
                    self._metrics.record_cancel()
                    count += 1
            for dep_id in self._dependents.get(proc_id, []):
                dep = self._processes.get(dep_id)
                if dep and not dep.is_done:
                    dep.cancel()
                    self._metrics.record_cancel()
                    count += 1
        return count

    def cancel_tree(self, tree_name: str) -> int:
        count = 0
        for proc in self._processes.values():
            if proc._tree_name == tree_name and not proc.is_done:
                proc.cancel()
                self._metrics.record_cancel()
                count += 1
        return count

    def spawn_chain(self, *steps: tuple, name: str = "", tree: str = None) -> List[Process]:
        procs = []
        prev_id = None
        for i, step in enumerate(steps):
            if not step:
                continue
            fn = step[0]
            args = ()
            kwargs = {}
            if len(step) > 1:
                if isinstance(step[-1], dict):
                    args = tuple(step[1:-1])
                    kwargs = step[-1]
                else:
                    args = tuple(step[1:])
            step_name = f"{name or 'chain'}-{i}"
            if i == 0:
                p = self.spawn(fn, *args, name=step_name, tree=tree, **kwargs)
            else:
                def _make_wrapped(base_fn, base_args, base_kwargs, _prev_id=prev_id):
                    def _wrapped():
                        prev_proc = self._processes.get(_prev_id)
                        prev_result = prev_proc.result if prev_proc else None
                        return base_fn(prev_result, *base_args, **base_kwargs)
                    _wrapped.__name__ = f"chain_{base_fn.__name__}"
                    return _wrapped
                p = self.spawn(_make_wrapped(fn, args, kwargs), name=step_name, tree=tree)
                p.depends_on = [prev_id]
                self._dependents.setdefault(prev_id, []).append(p.id)
            procs.append(p)
            prev_id = p.id
        return procs

    def run_subprocess(self, fn: Callable, *args, name: str = "",
                       cwd: str = None, env: dict = None,
                       memory_limit_mb: int = None,
                       timeout: float = None,
                       capture_output: bool = False,
                       **kwargs) -> Process:
        from .config import SubprocessConfig
        sub_config = SubprocessConfig(
            enabled=True,
            cwd=cwd,
            env=env,
            memory_limit_mb=memory_limit_mb,
            capture_output=capture_output,
        )
        old_config = None
        if hasattr(self, '_subprocess_config'):
            old_config = self._subprocess_config
        self._subprocess_config = sub_config

        proc = self.spawn(fn, *args, name=name, subprocess=True, timeout=timeout, **kwargs)

        if old_config is not None:
            self._subprocess_config = old_config
        elif hasattr(self, '_subprocess_config'):
            del self._subprocess_config

        return proc

    def group(self, name: str) -> ProcessGroup:
        return ProcessGroup(name, engine=self)

    def enable_cache(self, maxsize: int = 128, ttl: float = None) -> None:
        self._cache = ResultCache(maxsize=maxsize, ttl=ttl)

    def disable_cache(self) -> None:
        self._cache = None

    def health(self) -> dict:
        return {
            "name": self.name,
            "running": self._running,
            "tree_count": len(self._trees),
            "process_count": len(self._processes),
            "pending": len(self._pending),
            "completed": len(self._completed),
            "status_counts": {
                s.value: sum(1 for p in self._processes.values() if p.status == s)
                for s in ProcessStatus
            },
            "monitor": self._monitor.stats() if self._monitor else None,
            "metrics": self._metrics.snapshot(),
            "cache": self._cache.stats() if self._cache else None,
        }

    def to_dict(self) -> dict:
        subprocess_config = None
        if self._config and hasattr(self._config, 'subprocess'):
            sc = self._config.subprocess
            subprocess_config = {
                "enabled": sc.enabled,
                "max_workers": sc.max_workers,
                "memory_limit_mb": sc.memory_limit_mb,
                "cpu_affinity": sc.cpu_affinity,
                "start_method": sc.start_method,
                "cwd": sc.cwd,
                "capture_output": sc.capture_output,
                "terminate_grace": sc.terminate_grace,
            }
        return {
            "name": self.name,
            "running": self._running,
            "trees": {n: t.to_dict() for n, t in self._trees.items()},
            "processes": len(self._processes),
            "pending": len(self._pending),
            "active_stems": sum(t.active_stems for t in self._trees.values()),
            "routing": dict(self._routing),
            "monitor": self._monitor.stats() if self._monitor else None,
            "metrics": self._metrics.snapshot(),
            "cache": self._cache.stats() if self._cache else None,
            "subprocess_config": subprocess_config,
        }

    def install_signal_handlers(self) -> None:
        import signal
        self._old_signal_handlers = {
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
            signal.SIGINT: signal.getsignal(signal.SIGINT),
        }
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        self._signal_handlers_installed = True

    def restore_signal_handlers(self) -> None:
        import signal
        for sig, handler in self._old_signal_handlers.items():
            signal.signal(sig, handler)
        self._old_signal_handlers.clear()
        self._signal_handlers_installed = False

    def _handle_signal(self, signum, frame) -> None:
        self._running = False
        self.stop()

    def _deps_met(self, proc: Process) -> bool:
        if not proc.depends_on:
            return True
        for dep_id in proc.depends_on:
            dep = self._processes.get(dep_id)
            if dep is None or dep.status != ProcessStatus.COMPLETED:
                return False
        return True

    def start_workers(self, num_workers: int = 2, max_queue: int = 128) -> None:
        if self._spawn_queue is not None:
            return

        from domains.infrastructure.producer_consumer import ProducerConsumerQueue

        self._spawn_queue = ProducerConsumerQueue[Process](
            maxsize=max_queue,
            num_consumers=num_workers,
            handler=self._dispatch_process,
            name=f"engine-{self.name}",
        )
        self._spawn_queue.start()
        logger.info("Engine[%s]: started %d workers (max_queue=%d)",
                     self.name, num_workers, max_queue,
                     extra={"tag": "INFRA"})

    def stop_workers(self, timeout: float = 5.0) -> None:
        if self._spawn_queue is not None:
            self._spawn_queue.stop(timeout=timeout)
            self._spawn_queue = None

    def _dispatch_process(self, proc: Process) -> None:
        tree_name = proc._tree_name or self._routing.get(proc.name) or self._default_tree
        if not tree_name:
            logger.warning("Engine[%s]: no tree for process '%s'", self.name, proc.name)
            return

        tree = self._trees.get(tree_name)
        if tree is None:
            logger.warning("Engine[%s]: tree '%s' not found for process '%s'",
                           self.name, tree_name, proc.name)
            proc.fail(f"tree '{tree_name}' not found")
            return

        try:
            tree.branch([proc])
        except RuntimeError as e:
            logger.error("Engine[%s]: branch failed for '%s': %s",
                         self.name, tree_name, e)
            proc.fail(str(e))
