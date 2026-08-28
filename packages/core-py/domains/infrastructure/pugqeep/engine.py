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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("slo.pugqeep.engine")


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


@dataclass
class Process:
    """A unit of execution with lifecycle.

    A Process wraps a callable with args/kwargs and tracks its state
    through CREATED → READY → RUNNING → COMPLETED/FAILED.
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
    _future: Optional[Future] = field(default=None, repr=False)
    _tree_name: Optional[str] = field(default=None, repr=False)

    def ready(self) -> None:
        """Mark process as ready to run."""
        self.status = ProcessStatus.READY

    def running(self) -> None:
        """Mark process as running."""
        self.status = ProcessStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result: Any = None) -> None:
        """Mark process as completed."""
        self.status = ProcessStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def fail(self, error: str) -> None:
        """Mark process as failed."""
        self.status = ProcessStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def cancel(self) -> None:
        """Cancel the process."""
        self.status = ProcessStatus.CANCELLED
        self.completed_at = time.time()

    @property
    def elapsed(self) -> Optional[float]:
        """Elapsed time in seconds, or None if not started."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def is_done(self) -> bool:
        """Whether the process is in a terminal state."""
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
        }


@dataclass
class Stem:
    """A branch of parallel execution from a Tree.

    A Stem groups processes that run concurrently on a Tree instance.
    The Tree branches (creates Stems) to parallelize work.
    """
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
        """Whether all processes in this stem are done."""
        return all(p.is_done for p in self.processes)

    def results(self) -> List[Any]:
        """Collect results from all completed processes."""
        return [p.result for p in self.processes if p.status == ProcessStatus.COMPLETED]

    def errors(self) -> List[str]:
        """Collect errors from all failed processes."""
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
    """Model instance that branches Stems of parallel tasks.

    A Tree owns a PointLibrary (graph/context) and can branch
    Stems — groups of processes that run concurrently.
    """
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
        self._graph: Dict[str, Any] = {}  # PointLibrary context

    def branch(self, processes: List[Process]) -> Stem:
        """Branch a Stem of parallel processes.

        Submits all processes to the thread pool and returns
        the Stem that tracks their execution.
        """
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
        """Execute a single process with optional timeout and update stem status."""
        proc.running()
        try:
            if proc.timeout is not None and proc.timeout > 0:
                # Run in a separate thread to enforce timeout
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
                    logger.warning("Tree[%s]: process %s timed out (%.1fs)",
                                   self.name, proc.id, proc.timeout)
                elif error_container[0] is not None:
                    proc.fail(str(error_container[0]))
                    logger.error("Tree[%s]: process %s failed: %s",
                                 self.name, proc.id, error_container[0])
                else:
                    proc.complete(result_container[0])
            else:
                result = proc.fn(*proc.args, **proc.kwargs)
                proc.complete(result)
        except Exception as e:
            proc.fail(str(e))
            logger.error("Tree[%s]: process %s failed: %s",
                         self.name, proc.id, e)
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
        """Wait for a Stem to complete (event-driven, no polling)."""
        stem._done_event.wait(timeout=timeout)
        return stem

    def store(self, key: str, value: Any) -> None:
        """Store context in the tree's graph."""
        self._graph[key] = value

    def recall(self, key: str) -> Optional[Any]:
        """Recall context from the tree's graph."""
        return self._graph.get(key)

    @property
    def active_stems(self) -> int:
        return len(self._stems)

    def shutdown(self) -> None:
        """Shutdown the tree's thread pool."""
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


class Engine:
    """Core infra engine — the vCPU.

    Spawns the main process that seeds child processes across
    network, application, and protocol layers. Manages Trees
    that branch Stems of parallel tasks.

    Two modes of operation:

    1. Direct mode: branch() called manually
        engine.branch("tree", [proc1, proc2])

    2. Dispatch mode: run() processes queue and auto-dispatches
        engine.route("load_model", "data")
        engine.spawn(load_model, weights)
        engine.run()  # auto-dispatches to "data" tree

    Usage:
        engine = Engine("main")

        # Create trees
        engine.tree("data")
        engine.tree("training")

        # Route process names to trees
        engine.route("load_model", "data")
        engine.route("train", "training")

        # Spawn and auto-dispatch
        engine.spawn(load_model, weights)
        engine.spawn(train, epochs=10)

        # Run dispatch loop
        engine.run()
    """

    def __init__(self, name: str = "main", max_trees: int = 16):
        self.name = name
        self.max_trees = max_trees
        self._trees: Dict[str, Tree] = {}
        self._processes: Dict[str, Process] = {}
        self._pending: List[Process] = []
        self._running = False
        self._lock = threading.Lock()
        self._routing: Dict[str, str] = {}  # process_name → tree_name
        self._default_tree: Optional[str] = None
        self._on_complete: List[Callable[[Process], None]] = []
        self._completed: List[Process] = []
        self._dispatch_batch_size: int = 8
        self._round_robin_idx: int = 0

        # Producer-consumer queue for spawn dispatch (replaces raw queue.Queue)
        self._spawn_queue: Optional["ProducerConsumerQueue[Process]"] = None

    def spawn(self, fn: Callable[..., Any], *args: Any,
              name: str = "", tree: Optional[str] = None,
              priority: int = 2, timeout: Optional[float] = None,
              **kwargs: Any) -> Process:
        """Spawn a new process.

        Creates a Process, adds it to the engine, and returns it.
        The process is queued for dispatch — call dispatch() or run()
        to execute it on a tree.

        Args:
            fn: Callable to execute.
            *args: Positional args forwarded to ``fn``.
            name: Optional human-readable name. Used for routing.
            tree: Optional explicit tree name (overrides routing).
            priority: Queue priority (0=highest, 3=lowest). Only used when
                workers are running via ``start_workers()``.
            timeout: Optional timeout in seconds. None = no timeout.
            **kwargs: Keyword args forwarded to ``fn``.

        Returns:
            A ``Process`` instance with a unique id.
        """
        proc = Process(fn=fn, args=args, kwargs=kwargs, name=name, timeout=timeout)
        if tree:
            proc._tree_name = tree
        self._processes[proc.id] = proc
        self._pending.append(proc)

        # Dispatch to worker queue if running
        if self._spawn_queue is not None:
            self._spawn_queue.put(proc, priority=priority)

        logger.debug("Engine[%s]: spawned process %s (%s) → pending",
                      self.name, proc.id, proc.name or fn.__name__)
        return proc

    def tree(self, name: str, max_stems: int = 8,
             pool_workers: int = 4) -> Tree:
        """Create a new Tree (model instance)."""
        with self._lock:
            if len(self._trees) >= self.max_trees:
                raise RuntimeError(
                    f"Engine '{self.name}' at max trees ({self.max_trees})"
                )
            tree = Tree(name, max_stems=max_stems, pool_workers=pool_workers)
            self._trees[name] = tree
            if self._default_tree is None:
                self._default_tree = name
        logger.debug("Engine[%s]: created tree '%s'", self.name, name)
        return tree

    def route(self, process_name: str, tree_name: str) -> None:
        """Route processes by name to a specific tree.

        Args:
            process_name: Process name to match (exact match).
            tree_name: Tree to dispatch matching processes to.
        """
        if tree_name not in self._trees:
            raise ValueError(f"Tree '{tree_name}' not found")
        self._routing[process_name] = tree_name
        logger.debug("Engine[%s]: route '%s' → tree '%s'",
                      self.name, process_name, tree_name)

    def on_complete(self, callback: Callable[[Process], None]) -> None:
        """Register a callback for when a process completes.

        The callback receives the completed Process instance.
        """
        self._on_complete.append(callback)

    def branch(self, tree_name: str, processes: List[Process]) -> Stem:
        """Branch a Stem of parallel processes on a Tree."""
        tree = self._trees.get(tree_name)
        if tree is None:
            raise ValueError(f"Tree '{tree_name}' not found")
        stem = tree.branch(processes)
        # Track completion for on_complete callbacks
        for proc in processes:
            proc._tree_name = tree_name
        return stem

    def dispatch(self) -> int:
        """Dispatch pending processes to trees.

        Groups pending processes by target tree and branches them.
        Returns the number of processes dispatched.
        """
        if not self._pending:
            return 0

        # Group by target tree
        groups: Dict[str, List[Process]] = {}
        ungrouped: List[Process] = []

        for proc in self._pending:
            tree_name = proc._tree_name or self._routing.get(proc.name)
            if tree_name:
                proc._tree_name = tree_name
                groups.setdefault(tree_name, []).append(proc)
            else:
                ungrouped.append(proc)

        # Round-robin ungrouped processes across available trees
        if ungrouped:
            tree_names = list(self._trees.keys())
            if tree_names:
                for proc in ungrouped:
                    tree_name = tree_names[self._round_robin_idx % len(tree_names)]
                    proc._tree_name = tree_name
                    groups.setdefault(tree_name, []).append(proc)
                    self._round_robin_idx += 1

        # Dispatch each group
        dispatched = 0
        for tree_name, procs in groups.items():
            tree = self._trees.get(tree_name)
            if tree is None:
                logger.warning("Engine[%s]: tree '%s' not found, skipping %d processes",
                               self.name, tree_name, len(procs))
                for p in procs:
                    p.fail(f"tree '{tree_name}' not found")
                continue

            # Batch if too many processes
            for i in range(0, len(procs), self._dispatch_batch_size):
                batch = procs[i:i + self._dispatch_batch_size]
                try:
                    tree.branch(batch)
                    dispatched += len(batch)
                    logger.debug("Engine[%s]: dispatched %d processes to tree '%s'",
                                 self.name, len(batch), tree_name)
                except RuntimeError as e:
                    logger.error("Engine[%s]: failed to dispatch to '%s': %s",
                                 self.name, tree_name, e)
                    for p in batch:
                        p.fail(str(e))

        self._pending.clear()
        return dispatched

    def run(self, poll_interval: float = 0.1,
            on_progress: Optional[Callable[[dict], None]] = None) -> None:
        """Main process loop.

        Dispatches pending processes to trees, monitors active trees,
        and fires completion callbacks. Runs until stop() is called.

        When workers are running (via ``start_workers()``), this loop
        only monitors completion — dispatch is handled by workers.

        Args:
            poll_interval: Seconds between dispatch cycles.
            on_progress: Optional callback receiving status dict each cycle:
                {"pending": int, "active_stems": int, "completed": int,
                 "running": int, "failed": int}
        """
        self._running = True
        logger.info("Engine[%s]: starting main loop", self.name)

        while self._running:
            # 1. Dispatch pending processes (only when workers are NOT running)
            if self._pending and self._spawn_queue is None:
                dispatched = self.dispatch()
                if dispatched > 0:
                    logger.info("Engine[%s]: dispatched %d processes",
                                self.name, dispatched)

            # 2. Monitor active trees and fire callbacks
            with self._lock:
                active = sum(t.active_stems for t in self._trees.values())

            # Check for completed processes (poll all tracked processes)
            for proc in list(self._processes.values()):
                if proc.is_done and proc not in self._completed:
                    self._completed.append(proc)
                    for cb in self._on_complete:
                        try:
                            cb(proc)
                        except Exception as e:
                            logger.error("Engine[%s]: on_complete callback error: %s",
                                         self.name, e)

            # 3. Fire progress callback
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

            if active > 0:
                logger.debug("Engine[%s]: %d active stems across %d trees",
                             self.name, active, len(self._trees))

            time.sleep(poll_interval)

        logger.info("Engine[%s]: main loop stopped", self.name)

    def run_background(self, poll_interval: float = 0.1,
                       as_future: bool = False) -> Union[threading.Thread, Future]:
        """Start the main loop in a background thread.

        Non-blocking — returns immediately. Use stop() to stop.

        Args:
            poll_interval: Seconds between dispatch cycles.
            as_future: If True, return a concurrent.futures.Future instead of Thread.

        Returns:
            threading.Thread if as_future=False, Future if as_future=True.
        """
        if as_future:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"engine-{self.name}")
            future = executor.submit(self.run, poll_interval)
            future.add_done_callback(lambda _: executor.shutdown(wait=False))
            logger.info("Engine[%s]: started as Future", self.name)
            return future

        thread = threading.Thread(
            target=self.run,
            args=(poll_interval,),
            name=f"engine-{self.name}",
            daemon=True,
        )
        thread.start()
        logger.info("Engine[%s]: started in background", self.name)
        return thread

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for all pending and running processes to complete.

        Args:
            timeout: Maximum seconds to wait. None = wait forever.
        """
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
        """Wait for all processes and return completed/failed/cancelled list.

        Args:
            timeout: Maximum seconds to wait. None = wait forever.

        Returns:
            List of processes in terminal state.
        """
        self.wait(timeout=timeout)
        return [p for p in self._processes.values() if p.is_done]

    def get_completed(self) -> List[Process]:
        """Return all completed processes since last call."""
        done = list(self._completed)
        self._completed.clear()
        return done

    def stop(self) -> None:
        """Stop the main process loop and workers."""
        self._running = False
        self.stop_workers()
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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "running": self._running,
            "trees": {n: t.to_dict() for n, t in self._trees.items()},
            "processes": len(self._processes),
            "pending": len(self._pending),
            "active_stems": sum(t.active_stems for t in self._trees.values()),
            "routing": dict(self._routing),
        }

    # ── Worker pool (ProducerConsumerQueue-backed) ────────────────────

    def start_workers(self, num_workers: int = 2, max_queue: int = 128) -> None:
        """Start worker threads that auto-dispatch spawned processes.

        Replaces the manual ``run()`` loop with threaded dispatch.
        Processes are dispatched to trees automatically via the routing table.

        Args:
            num_workers: Number of dispatcher threads.
            max_queue: Maximum pending spawns before backpressure.
        """
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
        """Stop worker threads gracefully."""
        if self._spawn_queue is not None:
            self._spawn_queue.stop(timeout=timeout)
            self._spawn_queue = None
            logger.info("Engine[%s]: workers stopped", self.name,
                         extra={"tag": "INFRA"})

    def _dispatch_process(self, proc: Process) -> None:
        """Worker callback: dispatch a single process to its target tree."""
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
