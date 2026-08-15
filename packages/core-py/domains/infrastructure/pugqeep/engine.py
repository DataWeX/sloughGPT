"""
PGQ Engine — vCPU for graph-structured infrastructure.

Spawn a single process that seeds child processes across network,
application, and protocol layers. Trees branch stems of tasks into
parallel instances. Points carry function-calling capacity.

Usage:
    from pugqeep.engine import Engine

    engine = Engine("main")

    # Spawn processes
    proc = engine.spawn(my_function, arg1, arg2)

    # Branch parallel stems on a tree
    stem = engine.branch("network", [proc1, proc2, proc3])

    # Run the main loop
    engine.run()
"""

import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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
    _future: Optional[Future] = field(default=None, repr=False)

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

    def running(self) -> None:
        self.status = StemStatus.RUNNING

    def complete(self) -> None:
        self.status = StemStatus.COMPLETED
        self.completed_at = time.time()

    def fail(self) -> None:
        self.status = StemStatus.FAILED
        self.completed_at = time.time()

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
        """Execute a single process and update stem status."""
        proc.running()
        try:
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
        """Wait for a Stem to complete."""
        deadline = time.time() + timeout if timeout else None
        while not stem.all_done:
            if deadline and time.time() > deadline:
                break
            time.sleep(0.01)
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

    Usage:
        engine = Engine("main")

        # Spawn a process
        proc = engine.spawn(load_config, "config.json")

        # Create a tree and branch
        engine.tree("network", pool_workers=2)
        stem = engine.branch("network", [proc1, proc2])

        # Run the main loop
        engine.run()
    """

    def __init__(self, name: str = "main", max_trees: int = 16):
        self.name = name
        self.max_trees = max_trees
        self._trees: Dict[str, Tree] = {}
        self._processes: Dict[str, Process] = {}
        self._main_queue: queue.Queue = queue.Queue()
        self._running = False
        self._lock = threading.Lock()

    def spawn(self, fn: Callable[..., Any], *args: Any,
              name: str = "", **kwargs: Any) -> Process:
        """Spawn a new process.

        Creates a Process, adds it to the engine, and returns it.
        The process is in CREATED state — call branch() to run it.
        """
        proc = Process(fn=fn, args=args, kwargs=kwargs, name=name)
        self._processes[proc.id] = proc
        self._main_queue.put(("spawn", proc))
        logger.debug("Engine[%s]: spawned process %s (%s)",
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
        logger.debug("Engine[%s]: created tree '%s'", self.name, name)
        return tree

    def branch(self, tree_name: str, processes: List[Process]) -> Stem:
        """Branch a Stem of parallel processes on a Tree."""
        tree = self._trees.get(tree_name)
        if tree is None:
            raise ValueError(f"Tree '{tree_name}' not found")
        return tree.branch(processes)

    def run(self, poll_interval: float = 0.1) -> None:
        """Main process loop.

        Processes spawn/branch events from the queue and monitors
        active trees. Runs until stop() is called.
        """
        self._running = True
        logger.info("Engine[%s]: starting main loop", self.name)

        while self._running:
            try:
                event = self._main_queue.get(timeout=poll_interval)
                event_type, data = event
                if event_type == "spawn":
                    logger.debug("Engine[%s]: processed spawn %s",
                                 self.name, data.id)
            except queue.Empty:
                pass

            # Monitor active trees
            with self._lock:
                active = sum(t.active_stems for t in self._trees.values())
            if active > 0:
                logger.debug("Engine[%s]: %d active stems across %d trees",
                             self.name, active, len(self._trees))

        logger.info("Engine[%s]: main loop stopped", self.name)

    def stop(self) -> None:
        """Stop the main process loop."""
        self._running = False
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
            "active_stems": sum(t.active_stems for t in self._trees.values()),
        }
