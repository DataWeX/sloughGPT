"""
PGQ Engine — an agnostic, virtualizing vCPU for process execution.

Spawns processes and schedules them onto Pools (thread pools) or
GuardPools (subprocess isolation). Pools branch Stems of parallel
tasks; no host application semantics live in this module.

Usage:
    from pugqeep.engine import Engine

    engine = Engine("main")

    # Spawn processes (queued for dispatch)
    proc = engine.spawn(my_function, arg1, arg2)

    # Route processes to specific pools
    engine.route("load_model", "data")
    engine.route("train", "training")

    # Run dispatch loop (auto-dispatches to pools)
    engine.run()

    # Or dispatch manually
    engine.dispatch()  # one-shot dispatch
    engine.run(poll_interval=0.5)  # continuous loop
"""

import io
import logging
import os
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .config import RestartPolicy
from .frame import FrameEOF, FrameHandler, MsgType, ProtocolError
from .pipe import DEFAULT_CAPACITY, Pipe, PipeClosed

try:  # pragma: no cover - POSIX only, which is where fork exists
    import resource as _resource
except ImportError:  # pragma: no cover
    _resource = None

logger = logging.getLogger("slo.pugqeep.engine")

# TaskFuture is an alias for concurrent.futures.Future, re-exported for
# backwards compatibility with code that imports it from pugqeep.engine.
TaskFuture = Future

# Legacy string sentinels of the old multiprocessing.Pipe envelope. Nothing in
# this module reads them any more — the wire is MsgType frames now — but they
# are referenced by test docstrings, so they stay until that is cleaned up.
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


class PoolStatus(Enum):
    IDLE = "idle"
    BRANCHING = "branching"
    STOPPED = "stopped"


class SchedulingPolicy(Enum):
    """How the engine assigns ungrouped processes to pools."""

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
    # 48 bits of os.urandom, same 12-lowercase-hex shape as uuid.uuid4().hex[:12]
    # but one syscall instead of a UUID object build: 0.90 us vs 2.54 us, and ids
    # are never parsed back (grep uuid.UUID( in this package: none).
    id: str = field(default_factory=lambda: os.urandom(6).hex())
    name: str = ""
    status: ProcessStatus = ProcessStatus.CREATED
    result: Any = None
    error: str | None = None
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    timeout: float | None = None  # seconds, None = no timeout
    depends_on: list[str] = field(default_factory=list)
    _future: Future | None = field(default=None, repr=False)
    _pool_name: str | None = field(default=None, repr=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _done_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _priority: int = field(default=2, repr=False)
    _restart_count: int = field(default=0, repr=False)
    _pid: int | None = field(default=None, repr=False)
    _last_heartbeat: float | None = field(default=None, repr=False)
    _restart_policy: Optional["RestartPolicy"] = field(default=None, repr=False)
    _stream_results: list[Any] = field(default_factory=list, repr=False)
    _progress: float = field(default=0.0, repr=False)
    _progress_message: str = field(default="", repr=False)
    _on_complete: list[Callable] = field(default_factory=list, repr=False)
    _on_fail: list[Callable] = field(default_factory=list, repr=False)
    _on_cancel: list[Callable] = field(default_factory=list, repr=False)
    _on_stream: list[Callable] = field(default_factory=list, repr=False)
    _on_progress: list[Callable] = field(default_factory=list, repr=False)

    def ready(self) -> None:
        self.status = ProcessStatus.READY

    def running(self) -> None:
        self.status = ProcessStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result: Any = None) -> None:
        self.status = ProcessStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
        self._done_event.set()
        for cb in self._on_complete:
            try:
                cb(self)
            except Exception:
                pass

    def fail(self, error: str) -> None:
        self.status = ProcessStatus.FAILED
        self.error = error
        self.completed_at = time.time()
        self._done_event.set()
        for cb in self._on_fail:
            try:
                cb(self)
            except Exception:
                pass

    def cancel(self) -> None:
        self.status = ProcessStatus.CANCELLED
        self.completed_at = time.time()
        self._cancel_event.set()
        self._done_event.set()
        for cb in self._on_cancel:
            try:
                cb(self)
            except Exception:
                pass

    def wait_cancel(self, timeout: float = None) -> None:
        self._cancel_event.wait(timeout=timeout)

    def emit(self, value: Any) -> None:
        self._stream_results.append(value)
        for cb in self._on_stream:
            try:
                cb(self, value)
            except Exception:
                pass

    @property
    def stream_results(self) -> list[Any]:
        return self._stream_results

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def progress_message(self) -> str:
        return self._progress_message

    def on_complete(self, callback: Callable) -> None:
        self._on_complete.append(callback)

    def on_fail(self, callback: Callable) -> None:
        self._on_fail.append(callback)

    def on_cancel(self, callback: Callable) -> None:
        self._on_cancel.append(callback)

    def on_stream(self, callback: Callable) -> None:
        self._on_stream.append(callback)

    def on_progress(self, callback: Callable) -> None:
        self._on_progress.append(callback)

    def report_progress(self, value: float, message: str = "") -> None:
        self._progress = max(0.0, min(1.0, value))
        self._progress_message = message
        for cb in self._on_progress:
            try:
                cb(self, self._progress, message)
            except Exception:
                pass

    @property
    def is_cancelled(self) -> bool:
        return self.status == ProcessStatus.CANCELLED

    @property
    def elapsed(self) -> float | None:
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
            "progress": self._progress,
            "stream_count": len(self._stream_results),
        }


@dataclass
class Stem:
    """A branch of parallel execution from a Pool -- the unit that drives the queue.

    A Stem is execution, not accounting: while it is alive its processes
    hold capacity in the process queue, and the Stem is what completes or
    fails them so that capacity returns.
    """

    # Same 48-bit os.urandom shape as Process.id -- see that field for the why.
    id: str = field(default_factory=lambda: os.urandom(6).hex())
    pool_id: str = ""
    processes: list[Process] = field(default_factory=list)
    status: StemStatus = StemStatus.CREATED
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
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

    def results(self) -> list[Any]:
        return [p.result for p in self.processes if p.status == ProcessStatus.COMPLETED]

    def errors(self) -> list[str]:
        return [p.error for p in self.processes if p.status == ProcessStatus.FAILED and p.error]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pool_id": self.pool_id,
            "status": self.status.value,
            "num_processes": len(self.processes),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class Pool:
    """Task pooling and the resources it needs: threads, stems, limits.

    Pool decides how admitted work is laid out -- how many Stems, how many
    worker threads. It does not manage processes: the execution stack's
    only management layer is the queue behind :meth:`branch`, and Pool's
    duty there is to cross that door exactly once, before it mutates any
    of its own state.
    """

    def __init__(
        self,
        name: str,
        max_stems: int = 8,
        pool_workers: int = 4,
        pipe: Pipe | None = None,
    ):
        self.name = name
        self.status = PoolStatus.IDLE
        self.max_stems = max_stems
        self._stems: dict[str, Stem] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=pool_workers,
            thread_name_prefix=f"pool-{name}",
        )
        self._lock = threading.Lock()
        self._graph: dict[str, Any] = {}
        # A Pool escapes Engine through Engine.pool()'s return value, so it
        # needs a door even when nobody hands it one. Engine injects its own
        # pipe at creation, which keeps an Engine-wide stack on a single
        # queue; a standalone Pool mints and owns one instead.
        self._pipe = pipe if pipe is not None else Pipe(name=f"pool:{name}")
        self._owns_pipe = pipe is None

    def branch(self, processes: list[Process]) -> Stem:
        with self._lock:
            if len(self._stems) >= self.max_stems:
                raise RuntimeError(f"Pool '{self.name}' at max stems ({self.max_stems})")

        # Admit BEFORE any state change: a blocked put must not leave a
        # half-built stem behind it. Work already admitted (the Engine.spawn
        # path) passes through as a no-op, so the two doors meeting here
        # never double-count, and a refusal rolls back exactly what this
        # call added rather than leaking its capacity.
        admitted: list[Process] = []
        try:
            for proc in processes:
                fresh = not self._pipe.owns(proc)
                if not self._pipe.admit(proc):
                    raise PipeClosed(f"pipe '{self._pipe.name}' closed during admission")
                if fresh:
                    admitted.append(proc)
        except BaseException:
            for proc in admitted:
                self._pipe.release(proc)
            raise

        stem = Stem(pool_id=self.name, processes=processes)
        self._stems[stem.id] = stem
        self.status = PoolStatus.BRANCHING

        for proc in processes:
            proc.ready()
            future = self._pool.submit(self._execute, proc, stem)
            proc._future = future

        logger.debug(
            "Pool[%s]: branched stem %s with %d processes", self.name, stem.id, len(processes)
        )
        return stem

    def _execute(self, proc: Process, stem: Stem) -> Any:
        proc.running()
        try:
            if proc.timeout is not None and proc.timeout > 0:
                result_container: list[Any] = []
                error_container: list[Exception | None] = [None]

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
                    self.status = PoolStatus.IDLE
        return proc.result

    def wait_stem(self, stem: Stem, timeout: float | None = None) -> Stem:
        stem._done_event.wait(timeout=timeout)
        return stem

    def store(self, key: str, value: Any) -> None:
        self._graph[key] = value

    def recall(self, key: str) -> Any | None:
        return self._graph.get(key)

    @property
    def active_stems(self) -> int:
        return len(self._stems)

    def shutdown(self) -> None:
        self.status = PoolStatus.STOPPED
        if self._owns_pipe:
            # Only a pipe this Pool minted: an Engine-injected one is shared
            # with every other pool and must outlive any single shutdown.
            self._pipe.close()
        self._pool.shutdown(wait=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "active_stems": self.active_stems,
            "max_stems": self.max_stems,
            "graph_keys": list(self._graph.keys()),
        }


class GuardPool(Pool):
    """Pool that wraps processes in SubprocessProcess for subprocess isolation.

    Same pooling and resource responsibilities as :class:`Pool`; only the
    isolation strategy differs.
    """

    def __init__(
        self,
        name: str,
        config=None,
        max_stems: int = 8,
        pool_workers: int = 4,
        default_timeout: float = None,
        pipe: Pipe | None = None,
    ):
        super().__init__(name, max_stems=max_stems, pool_workers=pool_workers, pipe=pipe)
        self.subprocess_config = config
        self.default_timeout = default_timeout
        self._subprocesses: dict[str, SubprocessProcess] = {}

    def branch(self, processes: list[Process]) -> Stem:
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
        d["subprocess_enabled"] = (
            self.subprocess_config is not None and self.subprocess_config.enabled
        )
        d["subprocess_count"] = self.subprocess_count
        return d

    def health(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "active_stems": self.active_stems,
            "subprocess_enabled": self.subprocess_config is not None
            and self.subprocess_config.enabled,
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
        self._total_memory_bytes = 0
        self._peak_memory_bytes = 0

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

    def record_memory(self, bytes_used: int) -> None:
        with self._lock:
            self._total_memory_bytes += bytes_used
            if bytes_used > self._peak_memory_bytes:
                self._peak_memory_bytes = bytes_used

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
                "total_memory_bytes": self._total_memory_bytes,
                "peak_memory_bytes": self._peak_memory_bytes,
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
            self._total_memory_bytes = 0
            self._peak_memory_bytes = 0


# Reap poll granularity for _ForkedChild.join(). Never a blocking waitpid
# held across a lock: the reader thread calls is_alive() on the same handle
# and would otherwise stall behind it while the child keeps writing.
_REAP_POLL = 0.01


class _ForkedChild:
    """Owns one forked child: its pid, its exit status, and how it dies.

    A deliberately thin replacement for ``multiprocessing.Process``. Only
    :meth:`poll` ever calls ``waitpid``, and only ever with ``WNOHANG`` under a
    short-lived lock — so two threads (the reader's ``is_alive`` and the
    monitor's ``join``) can race without one blocking the other or double-
    reaping.
    """

    __slots__ = ("pid", "_exitcode", "_reaped", "_lock")

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._exitcode: int | None = None
        self._reaped = False
        self._lock = threading.Lock()

    @property
    def is_alive(self) -> bool:
        return self.poll() is None

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    def poll(self) -> int | None:
        """Reap without blocking; exit code, or None while still running."""
        with self._lock:
            if self._reaped:
                return self._exitcode
            try:
                pid, status = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                # Someone else reaped it (a SIGCHLD handler, a concurrent
                # monitor). Treat as done; the exit code is unknowable.
                self._reaped = True
                return self._exitcode
            if pid == 0:
                return None
            self._exitcode = os.waitstatus_to_exitcode(status)
            self._reaped = True
            return self._exitcode

    def join(self, timeout: float | None = None) -> int | None:
        """Wait for exit. ``timeout=None`` waits indefinitely."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            code = self.poll()
            if code is not None or self._reaped:
                return code
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(_REAP_POLL)

    def signal(self, signum: int) -> None:
        """Deliver ``signum``; a child that already exited is not an error."""
        if self._reaped:
            return
        try:
            os.kill(self.pid, signum)
        except ProcessLookupError:
            pass


def _send_captured(
    handler: FrameHandler, out_cap: io.StringIO | None, err_cap: io.StringIO | None
) -> None:
    """Emit captured output BEFORE the terminal frame.

    The reader stops on RESULT/ERROR, so output written after the terminal
    frame is silently dropped — the ordering bug this replaced.
    """
    if out_cap is None and err_cap is None:
        return
    handler.send_text(MsgType.STDOUT, out_cap.getvalue() if out_cap else "")
    handler.send_text(MsgType.STDERR, err_cap.getvalue() if err_cap else "")


def _subprocess_worker(
    handler: FrameHandler,
    *,
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    capture: bool,
    config,
) -> None:
    """Body of the forked child.

    Takes only plain values — no locks, no the parent's objects — and does no
    imports and no logging: this runs in a multithreaded parent's forked
    address space, where the import lock and the logging lock may be held by a
    thread that did not survive the fork.
    """
    out_cap = io.StringIO() if capture else None
    err_cap = io.StringIO() if capture else None
    if capture:
        sys.stdout = out_cap
        sys.stderr = err_cap

    handler.handshake()

    try:
        if config.memory_limit_mb is not None and _resource is not None:
            try:
                limit_bytes = config.memory_limit_mb * 1024 * 1024
                _resource.setrlimit(_resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            except (ValueError, OSError):
                pass

        if config.cpu_affinity is not None:
            try:
                os.sched_setaffinity(0, config.cpu_affinity)
            except (AttributeError, OSError):
                pass

        if config.cwd is not None:
            try:
                os.chdir(config.cwd)
            except (OSError, FileNotFoundError):
                pass

        if config.env is not None:
            try:
                os.environ.update(config.env)
            except (TypeError, OSError):
                pass

        handler.send(MsgType.READY)
        result = fn(*args, **kwargs)
        _send_captured(handler, out_cap, err_cap)
        handler.send(MsgType.RESULT, result)
    except Exception as exc:
        # A reported failure beats a bare exit code: the parent gets the text.
        try:
            _send_captured(handler, out_cap, err_cap)
            handler.send(MsgType.ERROR, str(exc))
        except Exception:
            pass


def _child_main(
    parent_sock: socket.socket,
    handler: FrameHandler,
    *,
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    capture: bool,
    config,
) -> None:
    """Child entry point after ``os.fork()``. Never returns — always ``os._exit``.

    ``os._exit`` (not ``sys.exit``) is required: the child shares the parent's
    atexit handlers and buffered streams, and running either would corrupt them.
    """
    try:
        parent_sock.close()
    except OSError:
        pass

    # The child inherits the parent's signal policy. An Engine installs a
    # SIGTERM handler in the owning process — if it survived the fork the
    # terminate() escalation would be swallowed and we would always wait out
    # terminate_grace before the SIGKILL.
    for _signum in (
        signal.SIGTERM,
        signal.SIGINT,
        signal.SIGQUIT,
        signal.SIGHUP,
        signal.SIGCHLD,
    ):
        try:
            signal.signal(_signum, signal.SIG_DFL)
        except (OSError, ValueError, RuntimeError):
            pass

    code = 0
    try:
        _subprocess_worker(handler, fn=fn, args=args, kwargs=kwargs, capture=capture, config=config)
    except BaseException as exc:  # noqa: BLE001 - the child must never return
        code = 1
        try:
            os.write(
                2,
                f"pugqeep child failed before it could report: {exc!r}\n".encode(
                    "utf-8", "replace"
                ),
            )
        except OSError:
            pass
    os._exit(code)


class SubprocessProcess:
    """Runs a Process.fn in a forked child, over a framed channel.

    Responsibilities, in order:

    1. **Fork and configure** — ``os.fork()``, then rlimits / CPU affinity /
       cwd / env applied inside the child by :func:`_subprocess_worker`.
    2. **Report** — one reader thread turns inbound frames into Process state
       transitions; it is the only writer of that state from this side.
    3. **Bound the child's life** — a timeout watchdog plus :meth:`terminate`'s
       SIGTERM -> SIGKILL escalation.

    No ``multiprocessing``: the channel is a ``socket.socketpair`` wrapped in a
    :class:`FrameHandler`, and the child handle is a :class:`_ForkedChild`.
    Fork-only — ``SubprocessConfig.start_method`` selects nothing.
    """

    def __init__(self, proc: Process, config):
        self.proc = proc
        self.config = config
        self._child: _ForkedChild | None = None
        self._handler: FrameHandler | None = None
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._lock = threading.Lock()
        self._watchdog: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._last_heartbeat: float | None = None
        self._stdout: str | None = None
        self._stderr: str | None = None

    def start(self) -> None:
        parent_sock, child_sock = socket.socketpair()
        child_handler = FrameHandler(child_sock)
        self._start_time = time.monotonic()
        capture = bool(self.config.capture_output)

        method = self.config.start_method or "fork"
        if method != "fork":
            logger.warning(
                "start_method=%r selects nothing: pugqeep forks. Spawn and "
                "forkserver were never reachable (worker was unpicklable) and "
                "left with multiprocessing.",
                method,
            )

        try:
            pid = os.fork()
        except OSError:
            parent_sock.close()
            child_sock.close()
            raise

        if pid == 0:
            # ── child: never returns into this stack ───────────────────
            _child_main(
                parent_sock,
                child_handler,
                fn=self.proc.fn,
                args=self.proc.args,
                kwargs=self.proc.kwargs,
                capture=capture,
                config=self.config,
            )
            os._exit(127)  # unreachable; _child_main always exits

        # ── parent ─────────────────────────────────────────────────────
        # Dropping our copy of the child's end is what makes EOF observable:
        # keep it and the reader blocks forever after the child exits.
        child_sock.close()
        self._child = _ForkedChild(pid)
        self.proc._pid = pid
        self.proc.running()

        self._handler = FrameHandler(parent_sock)
        self._reader_thread = threading.Thread(
            target=self._read_result,
            args=(self._handler,),
            daemon=True,
            name=f"reader-{self.proc.name}",
        )
        self._reader_thread.start()

        if self.proc.timeout is not None and self.proc.timeout > 0:
            self._watchdog = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name=f"watchdog-{self.proc.name}",
            )
            self._watchdog.start()

    def _read_result(self, handler: FrameHandler) -> None:
        """Translate inbound frames into Process state — the sole writer of it."""
        try:
            handler.handshake()
            while True:
                if not handler.poll(0.5):
                    if not self.is_alive:
                        break
                    continue
                try:
                    msg_type, value = handler.recv()
                except FrameEOF:
                    break
                if msg_type is MsgType.READY or msg_type is MsgType.HEARTBEAT:
                    self._last_heartbeat = time.monotonic()
                elif msg_type is MsgType.STDOUT:
                    # Append, not assign: send_text splits output into chunks
                    # and every chunk is a frame of its own.
                    self._stdout = (self._stdout or "") + (value or "")
                elif msg_type is MsgType.STDERR:
                    self._stderr = (self._stderr or "") + (value or "")
                elif msg_type is MsgType.RESULT:
                    self.proc.complete(value)
                    break
                elif msg_type is MsgType.ERROR:
                    # A cancel already in flight outranks the failure: the
                    # child died because we killed it, not because it broke.
                    if self._cancel_event.is_set():
                        self.proc.cancel()
                    else:
                        self.proc.fail(value)
                    break
        except (ProtocolError, OSError, ValueError) as exc:
            logger.debug("reader for %s stopped: %s", self.proc.name, exc)
            if not self._cancel_event.is_set() and not self.proc.is_done:
                self.proc.fail(f"frame protocol failure: {exc}")
        finally:
            if self._cancel_event.is_set() and not self.proc.is_done:
                self.proc.cancel()
            elif not self.proc.is_done:
                self.proc.fail("pipe closed unexpectedly")
            try:
                handler.close()
            except Exception:
                logger.debug("failed to close frame channel", exc_info=True)

    def monitor(self) -> None:
        """Block until the child is reaped, then settle anything unsettled."""
        if self._child is not None:
            self._child.join()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
        self._end_time = time.monotonic()
        if not self.proc.is_done:
            if self._child is None:
                self.proc.fail("no process")
            elif self._child.exitcode == 0:
                self.proc.complete()
            else:
                self.proc.fail(f"exit code {self._child.exitcode}")

    def _watchdog_loop(self) -> None:
        while not self._cancel_event.is_set():
            if self._child is None or not self._child.is_alive:
                break
            elapsed = time.monotonic() - (self._start_time or 0)
            if self.proc.timeout and elapsed > self.proc.timeout:
                self.terminate()
                return
            self._cancel_event.wait(0.5)

    def terminate(self) -> None:
        if self._child is None or not self._child.is_alive:
            return
        self._cancel_event.set()
        try:
            self._child.signal(signal.SIGTERM)
            self._child.join(timeout=self.config.terminate_grace)
            if self._child.is_alive:
                self._child.signal(signal.SIGKILL)
                self._child.join(timeout=1.0)
        except OSError:
            pass
        self._end_time = time.monotonic()
        if not self.proc.is_done:
            self.proc.cancel()
        elif self.proc.status == ProcessStatus.FAILED:
            self.proc.status = ProcessStatus.CANCELLED
            self.proc.error = None

    @property
    def is_alive(self) -> bool:
        return self._child is not None and self._child.is_alive

    @property
    def elapsed(self) -> float | None:
        end = self._end_time or time.monotonic()
        return end - (self._start_time or end) if self._start_time else None

    @property
    def pid(self) -> int | None:
        return self._child.pid if self._child else None

    def cancel(self) -> None:
        self._cancel_event.set()

    def health(self) -> dict:
        return {
            "pid": self.proc._pid,
            "alive": self.is_alive,
            "elapsed": self.elapsed,
        }

    def resource_usage(self) -> dict | None:
        if self._child is None or _resource is None:
            return None
        try:
            if self._child.is_alive:
                self._child.join(timeout=0.1)
            usage = _resource.getrusage(_resource.RUSAGE_CHILDREN)
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
    def stdout(self) -> str | None:
        return self._stdout

    @property
    def stderr(self) -> str | None:
        return self._stderr


class ProcessGroup:
    """Batch operations on a set of processes."""

    def __init__(self, name: str, engine: "Engine" = None):
        self.name = name
        self.engine = engine
        self._processes: list[Process] = []
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
    def elapsed(self) -> float | None:
        starts = [p.started_at for p in self._processes if p.started_at]
        ends = [p.completed_at or time.time() for p in self._processes]
        if not starts:
            return 0.0
        return max(ends) - min(starts)

    def results(self) -> list[Any]:
        return [p.result for p in self._processes if p.status == ProcessStatus.COMPLETED]

    def errors(self) -> list[str]:
        return [p.error for p in self._processes if p.status == ProcessStatus.FAILED and p.error]

    def cancel(self) -> int:
        count = 0
        for p in self._processes:
            if not p.is_done:
                p.cancel()
                count += 1
        return count

    def gather(self, timeout: float = None) -> list[Any]:
        self.wait(timeout=timeout)
        return self.results()

    def wait(self, timeout: float = None) -> None:
        deadline = time.time() + timeout if timeout else None
        for p in self._processes:
            if p.is_done:
                continue
            wait_timeout = (deadline - time.time()) if deadline else None
            if wait_timeout is not None and wait_timeout <= 0:
                break
            p._done_event.wait(timeout=wait_timeout)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_processes": self.num_processes,
            "elapsed": self.elapsed,
            "all_done": self.all_done,
            "status_counts": {
                s.value: sum(1 for p in self._processes if p.status == s) for s in ProcessStatus
            },
        }


class ProcessMonitor:
    """Background thread for stall detection and restart callbacks."""

    def __init__(
        self,
        config=None,
        restart_policy=None,
        poll_interval: float = 1.0,
        stall_timeout: float = 30.0,
    ):
        self.config = config
        self.restart_policy = restart_policy
        self.poll_interval = config.poll_interval if config else poll_interval
        self.stall_timeout = config.stall_timeout if config else stall_timeout
        self._processes: dict[str, Process] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._on_stall: list[Callable] = []
        self._on_restart: list[Callable] = []
        self._restart_count: dict[str, int] = {}

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
            delay = base * (2**attempt)
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
                                    logger.debug(
                                        "Non-critical pugqeep stall callback error", exc_info=True
                                    )
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
                                        logger.debug(
                                            "Non-critical pugqeep restart callback error",
                                            exc_info=True,
                                        )

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
        self._cache: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _key(self, fn: Callable, args: tuple, kwargs: dict) -> str:
        return f"{fn.__name__}:{args}:{sorted(kwargs.items())}"

    def get(self, fn: Callable, args: tuple, kwargs: dict):
        key = self._key(fn, args, kwargs)
        with self._lock:
            if key in self._cache:
                if self.ttl and time.monotonic() - self._timestamps[key] > self.ttl:
                    del self._cache[key]
                    del self._timestamps[key]
                    self._misses += 1
                    return False, None
                self._hits += 1
                return True, self._cache[key]
            self._misses += 1
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

    def invalidate(self, fn: Callable = None, args: tuple = None, kwargs: dict = None) -> bool:
        with self._lock:
            if fn is None and args is None and kwargs is None:
                had_items = len(self._cache) > 0
                self._cache.clear()
                self._timestamps.clear()
                return had_items
            if args is not None and kwargs is not None:
                key = self._key(fn, args, kwargs)
                if key in self._cache:
                    del self._cache[key]
                    del self._timestamps[key]
                    return True
                return False
            count = 0
            to_remove = [k for k in self._cache if fn.__name__ in k]
            for k in to_remove:
                del self._cache[k]
                del self._timestamps[k]
                count += 1
            return count > 0

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._timestamps.clear()
            return count

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(1, total),
            }

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class Engine:
    """The execution graph: points, leaves, nodes, branches and loops.

    Engine owns the *topology* of a computation, plus the single
    :class:`Pipe` every path onto the execution stack must cross -- which
    is what makes the stack's diameter knowable rather than declared. It
    spawns processes, routes them to Pools (task pooling and resources)
    and GuardPools (subprocess isolation), and branches Stems that carry
    out the work. Host-application semantics stay outside.
    """

    def __init__(self, name: str = "main", max_pools: int = 16, config=None):
        if config is not None:
            self.name = config.name
            self.max_pools = config.max_pools
            self._config = config
        else:
            self.name = name
            self.max_pools = max_pools
            self._config = None
        self._pools: dict[str, Pool] = {}
        self._processes: dict[str, Process] = {}
        self._pending: list[Process] = []
        self._running = False
        self._lock = threading.Lock()
        self._routing: dict[str, str] = {}
        self._default_pool: str | None = None
        self._on_complete: list[Callable[[Process], None]] = []
        self._completed: list[Process] = []
        self._dispatch_batch_size: int = 8
        self._round_robin_idx: int = 0
        self._scheduling_policy: SchedulingPolicy = SchedulingPolicy.ROUND_ROBIN
        self._dependents: dict[str, list[str]] = {}
        self._spawn_queue = None
        # One queue, one door. Every path onto the execution stack enters here
        # or through a Pool carrying this same pipe, which is what makes
        # diameter()/usage()/headroom() derivable rather than declared.
        capacity = self._config.queue_size if self._config else DEFAULT_CAPACITY
        self._pipe = Pipe(name=f"{self.name}:pipe", capacity=capacity)
        self._metrics = EngineMetrics()
        self._cache: ResultCache | None = None
        self._monitor: ProcessMonitor | None = None
        self._signal_handlers_installed = False
        self._old_signal_handlers: dict = {}
        self._all_done_event = threading.Event()

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
        """Set how ungrouped processes are routed to pools."""
        if not isinstance(policy, SchedulingPolicy):
            raise TypeError("policy must be a SchedulingPolicy")
        self._scheduling_policy = policy

    def spawn(
        self,
        fn: Callable[..., Any],
        *args: Any,
        name: str = "",
        pool: str | None = None,
        priority: int = 2,
        timeout: float | None = None,
        depends_on: list[str] | None = None,
        subprocess: bool = False,
        register_cancel: bool = False,
        **kwargs: Any,
    ) -> Process:
        proc = Process(fn=fn, args=args, kwargs=kwargs, name=name, timeout=timeout)

        # Admission is the FIRST mutation — ahead of _dependents, the cache and
        # _processes. Previously the blocking put ran last, after all three had
        # already changed, so a full stack left records for work that was never
        # let in and usage() disagreed with reality by exactly that window.
        # A cache hit needs no capacity: it is complete on arrival and retires
        # itself on the next read.
        if not self._pipe.admit(proc):
            raise PipeClosed(f"engine '{self.name}' is stopping; will not admit {proc.id}")

        proc._priority = priority
        if pool:
            proc._pool_name = pool
        if depends_on:
            proc.depends_on = list(depends_on)
            for dep_id in depends_on:
                self._dependents.setdefault(dep_id, []).append(proc.id)

        # Check cache for hit
        if self._cache is not None:
            hit, cached = self._cache.get(fn, args, kwargs)
            if hit:
                proc.complete(cached)
                self._processes[proc.id] = proc
                self._completed.append(proc)
                self._metrics.record_complete(proc)
                return proc

        self._processes[proc.id] = proc
        self._pending.append(proc)
        self._all_done_event.clear()
        self._metrics.record_spawn()

        if register_cancel and self._monitor:
            self._monitor.track(proc)

        if self._spawn_queue is not None:
            self._spawn_queue.put(proc, priority=priority)

        # Guarded twice: at INFO the message is never emitted, and the arguments
        # are evaluated eagerly at the call site either way -- so an unguarded
        # fn.__name__ made Engine.spawn(functools.partial(f, 3)) raise AttributeError
        # while logging was disabled entirely.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Engine[%s]: spawned process %s (%s) -> pending",
                self.name,
                proc.id,
                proc.name or getattr(fn, "__name__", type(fn).__name__),
            )
        return proc

    def pool(
        self,
        name: str,
        max_stems: int = 8,
        pool_workers: int = 4,
        guarded: bool = False,
        default_timeout: float = None,
    ) -> Pool:
        with self._lock:
            if len(self._pools) >= self.max_pools:
                raise RuntimeError(f"Engine '{self.name}' at max pools ({self.max_pools})")
            if guarded:
                config = self._config.subprocess if self._config else None
                pool = GuardPool(
                    name,
                    config=config,
                    max_stems=max_stems,
                    pool_workers=pool_workers,
                    default_timeout=default_timeout,
                    pipe=self._pipe,
                )
            else:
                pool = Pool(
                    name,
                    max_stems=max_stems,
                    pool_workers=pool_workers,
                    pipe=self._pipe,
                )
            self._pools[name] = pool
            if self._default_pool is None:
                self._default_pool = name
        logger.debug("Engine[%s]: created pool '%s'", self.name, name)
        return pool

    def route(self, process_name: str, pool_name: str) -> None:
        if pool_name not in self._pools:
            raise ValueError(f"Pool '{pool_name}' not found")
        self._routing[process_name] = pool_name
        logger.debug("Engine[%s]: route '%s' -> pool '%s'", self.name, process_name, pool_name)

    def on_complete(self, callback: Callable[[Process], None]) -> None:
        self._on_complete.append(callback)

    def branch(self, pool_name: str, processes: list[Process]) -> Stem:
        pool = self._pools.get(pool_name)
        if pool is None:
            raise ValueError(f"Pool '{pool_name}' not found")
        stem = pool.branch(processes)
        for proc in processes:
            proc._pool_name = pool_name
        return stem

    def dispatch(self) -> int:
        self._pipe.open()
        if not self._pending:
            return 0

        dispatchable: list[Process] = []
        held: list[Process] = []
        for proc in self._pending:
            if proc.depends_on and not self._deps_met(proc):
                held.append(proc)
            else:
                dispatchable.append(proc)

        if not dispatchable:
            return 0

        dispatchable.sort(key=lambda p: p._priority)

        groups: dict[str, list[Process]] = {}
        ungrouped: list[Process] = []

        for proc in dispatchable:
            pool_name = proc._pool_name or self._routing.get(proc.name)
            if pool_name:
                proc._pool_name = pool_name
                groups.setdefault(pool_name, []).append(proc)
            else:
                ungrouped.append(proc)

        if ungrouped:
            pool_names = list(self._pools.keys())
            if pool_names:
                for proc in ungrouped:
                    pool_name = pool_names[self._round_robin_idx % len(pool_names)]
                    proc._pool_name = pool_name
                    groups.setdefault(pool_name, []).append(proc)
                    self._round_robin_idx += 1

        dispatched = 0
        for pool_name, procs in groups.items():
            pool = self._pools.get(pool_name)
            if pool is None:
                logger.warning(
                    "Engine[%s]: pool '%s' not found, skipping %d processes",
                    self.name,
                    pool_name,
                    len(procs),
                )
                for p in procs:
                    p.fail(f"pool '{pool_name}' not found")
                continue

            for i in range(0, len(procs), self._dispatch_batch_size):
                batch = procs[i : i + self._dispatch_batch_size]
                try:
                    pool.branch(batch)
                    dispatched += len(batch)
                except RuntimeError as e:
                    logger.error(
                        "Engine[%s]: failed to dispatch to '%s': %s", self.name, pool_name, e
                    )
                    for p in batch:
                        p.fail(str(e))

        self._pending = held
        self._metrics.record_dispatch(dispatched)
        return dispatched

    def run(
        self, poll_interval: float = 0.1, on_progress: Callable[[dict], None] | None = None
    ) -> None:
        self._running = True
        self._pipe.open()
        logger.info("Engine[%s]: starting main loop", self.name)

        while self._running:
            if self._pending and self._spawn_queue is None:
                dispatched = self.dispatch()
                if dispatched > 0:
                    logger.info("Engine[%s]: dispatched %d processes", self.name, dispatched)

            with self._lock:
                active = sum(t.active_stems for t in self._pools.values())

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
                            logger.error("Engine[%s]: on_complete callback error: %s", self.name, e)

            if on_progress is not None:
                try:
                    on_progress(
                        {
                            "pending": len(self._pending),
                            "active_stems": active,
                            "completed": len(self._completed),
                            "running": sum(
                                1
                                for p in self._processes.values()
                                if p.status == ProcessStatus.RUNNING
                            ),
                            "failed": sum(
                                1
                                for p in self._processes.values()
                                if p.status == ProcessStatus.FAILED
                            ),
                        }
                    )
                except Exception as e:
                    logger.error("Engine[%s]: on_progress callback error: %s", self.name, e)

            if not self._pending and all(p.is_done for p in self._processes.values()):
                self._all_done_event.set()

            time.sleep(poll_interval)

        logger.info("Engine[%s]: main loop stopped", self.name)

    def run_background(
        self, poll_interval: float = 0.1, as_future: bool = False
    ) -> threading.Thread | Future:
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

    def wait(self, timeout: float | None = None) -> None:
        if not self._pending and all(p.is_done for p in self._processes.values()):
            return
        self._all_done_event.wait(timeout=timeout)

    def wait_all(self, timeout: float | None = None) -> list[Process]:
        self.wait(timeout=timeout)
        return [p for p in self._processes.values() if p.is_done]

    def get_completed(self) -> list[Process]:
        done = list(self._completed)
        self._completed.clear()
        return done

    def stop(self) -> None:
        self._running = False
        # Wake anyone parked on a full stack: a stopped engine admits nothing
        # more, so waiting for capacity that will never free is a hang rather
        # than backpressure. run()/dispatch() re-arm it.
        self._pipe.close()
        self.stop_workers()
        for proc in self._processes.values():
            if not proc.is_done:
                proc.cancel()
                self._metrics.record_cancel()
        if self._monitor:
            self._monitor.stop()
        for pool in self._pools.values():
            pool.shutdown()

    def get_process(self, proc_id: str) -> Process | None:
        return self._processes.get(proc_id)

    def get_pool(self, name: str) -> Pool | None:
        return self._pools.get(name)

    def list_pools(self) -> list[str]:
        return list(self._pools.keys())

    def list_processes(self, status: ProcessStatus | None = None) -> list[Process]:
        procs = list(self._processes.values())
        if status:
            procs = [p for p in procs if p.status == status]
        return procs

    def wait_for(self, proc_id: str, timeout: float = None) -> Process | None:
        proc = self._processes.get(proc_id)
        if proc is None:
            raise KeyError(f"Process '{proc_id}' not found")
        if proc.is_done:
            return proc
        proc._done_event.wait(timeout=timeout)
        return proc if proc.is_done else None

    def wait_for_any(self, proc_ids: list[str], timeout: float = None) -> Process | None:
        # Check if any already done
        for pid in proc_ids:
            proc = self._processes.get(pid)
            if proc and proc.is_done:
                return proc
        # Wait on all events with timeout
        events = []
        for pid in proc_ids:
            proc = self._processes.get(pid)
            if proc and not proc.is_done:
                events.append(proc._done_event)
        if not events:
            return None
        # Poll with event waits for the first one to complete
        deadline = time.time() + timeout if timeout else None
        while True:
            for pid in proc_ids:
                proc = self._processes.get(pid)
                if proc and proc.is_done:
                    return proc
            if deadline and time.time() > deadline:
                return None
            # Wait on any event with a short timeout, then re-check
            remaining = (deadline - time.time()) if deadline else 1.0
            wait_time = min(0.05, remaining) if remaining > 0 else 0.01
            events[0].wait(timeout=wait_time)

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

    def cancel_pool(self, pool_name: str) -> int:
        count = 0
        for proc in self._processes.values():
            if proc._pool_name == pool_name and not proc.is_done:
                proc.cancel()
                self._metrics.record_cancel()
                count += 1
        return count

    def spawn_chain(self, *steps: tuple, name: str = "", pool: str = None) -> list[Process]:
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
                p = self.spawn(fn, *args, name=step_name, pool=pool, **kwargs)
            else:

                def _make_wrapped(base_fn, base_args, base_kwargs, _prev_id=prev_id):
                    def _wrapped():
                        prev_proc = self._processes.get(_prev_id)
                        prev_result = prev_proc.result if prev_proc else None
                        return base_fn(prev_result, *base_args, **base_kwargs)

                    _wrapped.__name__ = f"chain_{base_fn.__name__}"
                    return _wrapped

                p = self.spawn(_make_wrapped(fn, args, kwargs), name=step_name, pool=pool)
                p.depends_on = [prev_id]
                self._dependents.setdefault(prev_id, []).append(p.id)
            procs.append(p)
            prev_id = p.id
        return procs

    def run_subprocess(
        self,
        fn: Callable,
        *args,
        name: str = "",
        cwd: str = None,
        env: dict = None,
        memory_limit_mb: int = None,
        timeout: float = None,
        capture_output: bool = False,
        **kwargs,
    ) -> Process:
        from .config import SubprocessConfig

        sub_config = SubprocessConfig(
            enabled=True,
            cwd=cwd,
            env=env,
            memory_limit_mb=memory_limit_mb,
            capture_output=capture_output,
        )
        old_config = None
        if hasattr(self, "_subprocess_config"):
            old_config = self._subprocess_config
        self._subprocess_config = sub_config

        proc = self.spawn(fn, *args, name=name, subprocess=True, timeout=timeout, **kwargs)

        if old_config is not None:
            self._subprocess_config = old_config
        elif hasattr(self, "_subprocess_config"):
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
            "pool_count": len(self._pools),
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
        if self._config and hasattr(self._config, "subprocess"):
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
            "pools": {n: t.to_dict() for n, t in self._pools.items()},
            "processes": len(self._processes),
            "pending": len(self._pending),
            "active_stems": sum(t.active_stems for t in self._pools.values()),
            "routing": dict(self._routing),
            "monitor": self._monitor.stats() if self._monitor else None,
            "metrics": self._metrics.snapshot(),
            "cache": self._cache.stats() if self._cache else None,
            "subprocess_config": subprocess_config,
        }

    def reset(self) -> None:
        with self._lock:
            self._processes.clear()
            self._pending.clear()
            self._completed.clear()
            self._dependents.clear()
            self._all_done_event.set()

    def summary(self) -> str:
        counts = {}
        for p in self._processes.values():
            counts[p.status.value] = counts.get(p.status.value, 0) + 1
        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        return f"Engine '{self.name}': processes={len(self._processes)} ({', '.join(parts) if parts else 'none'})"

    def dispatch_batch(self, max_count: int = None) -> int:
        if max_count is None:
            max_count = self._dispatch_batch_size
        if not self._pending:
            return 0
        batch = self._pending[:max_count]
        self._pending = self._pending[max_count:]
        dispatched = 0
        for proc in batch:
            self._dispatch_process(proc)
            dispatched += 1
        self._metrics.record_dispatch(dispatched)
        return dispatched

    def cancel_all(self, status: ProcessStatus = None) -> int:
        count = 0
        for proc in list(self._processes.values()):
            if proc.is_done:
                continue
            if status is not None and proc.status != status:
                continue
            proc.cancel()
            count += 1
        return count

    def dependency_graph(self) -> dict:
        nodes = []
        edges = []
        for proc in self._processes.values():
            nodes.append(proc.id)
            for dep_id in proc.depends_on:
                edges.append({"from": dep_id, "to": proc.id})
        return {"nodes": nodes, "edges": edges}

    def critical_path(self) -> list[str]:
        if not self._processes:
            return []
        # Build adjacency and find longest path via DFS
        dep_of: dict[str, list[str]] = {}
        for proc in self._processes.values():
            for dep_id in proc.depends_on:
                dep_of.setdefault(dep_id, []).append(proc.id)

        memo: dict[str, list[str]] = {}

        def _longest_path(pid: str) -> list[str]:
            if pid in memo:
                return memo[pid]
            children = dep_of.get(pid, [])
            if not children:
                memo[pid] = [pid]
                return [pid]
            best = []
            for child in children:
                path = _longest_path(child)
                if len(path) > len(best):
                    best = path
            result = [pid] + best
            memo[pid] = result
            return result

        # Find roots (processes with no dependents)
        all_children = set()
        for children in dep_of.values():
            all_children.update(children)
        roots = [p.id for p in self._processes.values() if p.id not in all_children]
        if not roots:
            roots = list(self._processes.keys())

        best_path = []
        for root in roots:
            path = _longest_path(root)
            if len(path) > len(best_path):
                best_path = path
        return best_path

    def orphan_processes(self) -> list[Process]:
        return [p for p in self._processes.values() if p.depends_on and not self._deps_met(p)]

    def spawn_batch(self, items: list) -> list[Process]:
        procs = []
        for item in items:
            if not item:
                continue
            fn = item[0]
            args = tuple(item[1:]) if len(item) > 1 else ()
            procs.append(self.spawn(fn, *args))
        return procs

    def save_state(self, path: str) -> None:
        import json

        state = {
            "name": self.name,
            "processes": {pid: p.to_dict() for pid, p in self._processes.items()},
            "metrics": self._metrics.snapshot(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def install_signal_handlers(self) -> None:
        import signal

        # Idempotent: a second install would overwrite _old_signal_handlers
        # with our own handler, losing the real previous one. That handler is
        # what _delegate_signal() hands the signal to, so losing it means the
        # owner (uvicorn, systemd, a caller's handler) can never be reached.
        if self._signal_handlers_installed:
            return
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
        try:
            self.stop()
        finally:
            self._delegate_signal(signum, frame)

    def _delegate_signal(self, signum, frame) -> None:
        """Hand the signal on instead of swallowing it.

        Engine.__init__ installs this handler process-globally, so if it never
        propagates the owning process can never be told to exit:
        uvicorn --reload sends SIGTERM to the child and then blocks forever in
        Process.join(), wedging the reloader after the first detected change,
        and plain `kill <pid>` degrades into needing SIGKILL.

        Delegation goes to the handler we replaced (uvicorn's handle_exit, a
        caller's handler, Python's SIGINT KeyboardInterrupt default). With
        nothing above us and a real signal delivery, we restore SIG_DFL and
        re-raise so the process terminates from the signal itself.

        `frame is None` means the handler was called directly rather than by
        the interpreter (tests do this to simulate delivery) — terminating
        this process would be wrong there.
        """
        import os
        import signal

        prev = self._old_signal_handlers.get(signum)
        if callable(prev) and prev is not self._handle_signal:
            prev(signum, frame)
            return
        if frame is None:
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

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

        from domain.infrastructure._internal.producer_consumer import ProducerConsumerQueue

        self._spawn_queue = ProducerConsumerQueue[Process](
            maxsize=max_queue,
            num_consumers=num_workers,
            handler=self._dispatch_process,
            name=f"engine-{self.name}",
        )
        self._spawn_queue.start()
        logger.info(
            "Engine[%s]: started %d workers (max_queue=%d)",
            self.name,
            num_workers,
            max_queue,
            extra={"tag": "INFRA"},
        )

    def stop_workers(self, timeout: float = 5.0) -> None:
        if self._spawn_queue is not None:
            self._spawn_queue.stop(timeout=timeout)
            self._spawn_queue = None

    def _dispatch_process(self, proc: Process) -> None:
        pool_name = proc._pool_name or self._routing.get(proc.name) or self._default_pool
        if not pool_name:
            logger.warning("Engine[%s]: no pool for process '%s'", self.name, proc.name)
            return

        pool = self._pools.get(pool_name)
        if pool is None:
            logger.warning(
                "Engine[%s]: pool '%s' not found for process '%s'", self.name, pool_name, proc.name
            )
            proc.fail(f"pool '{pool_name}' not found")
            return

        try:
            pool.branch([proc])
        except RuntimeError as e:
            logger.error("Engine[%s]: branch failed for '%s': %s", self.name, pool_name, e)
            proc.fail(str(e))
