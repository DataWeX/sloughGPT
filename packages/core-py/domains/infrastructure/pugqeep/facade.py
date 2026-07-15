"""
PGQ — Point-Graph-Queue system facade.

Generic data structure manager with tiered caching:
  - Disk → Hot cache → In-memory
  - Compress any structured data into Points (generator functions)
  - Manage task queues with priority and routing

Quick start:
    from pugqeep import PGQ

    # Compress any data
    sys = PGQ(name="sensor-data")
    sys.put("readings", sensor_array)
    data = sys.get("readings")

    # Tiered caching
    sys = PGQ(name="ml-weights", cache_dir=Path("/tmp/cache"))
    sys.put("layer1", weights, tier="memory")
    sys.put("layer2", weights, tier="hot")

    # Task queue
    sys = PGQ(name="pipeline")
    sys.submit_task(Task(name="process", data=input))
    task = sys.next_task()
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import PointConfig, CompressorConfig, LibraryConfig, TreeConfig, QueueConfig
from .point import Point
from .compressor import PointCompressor
from .library import PointLibrary
from .model_tree import ModelTree, load_model_to_points
from .cache import TieredCache, Tier
from .store import MemoryStore, JSONStore, DirectoryStore
from .dedup import PointDeduplicator, PointLibrarySync
from .queue import ModelQueue
from .task_queue import TaskQueue, Task, TaskStatus, TaskPriority

logger = logging.getLogger("slo.pugqeep")


class PGQ:
    """Point-Graph-Queue system — generic data structure manager.

    Manages data across tiers (Disk → Hot → Memory) with compression,
    caching, and task queuing.
    """

    def __init__(self, name: str = "model",
                 storage_dir: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 n_clusters: int = 16,
                 method: str = "cluster",
                 memory_max_mb: int = 512,
                 hot_max_mb: int = 128):
        """Initialize PGQ system.

        Args:
            name: System name.
            storage_dir: Directory for persistent storage.
            cache_dir: Directory for disk cache tier. None = disk cache disabled.
            n_clusters: Number of VQ clusters for compression.
            method: Compression method ("cluster" or "function").
            memory_max_mb: Max memory cache in MB.
            hot_max_mb: Max hot cache in MB.
        """
        self.name = name
        self._config = TreeConfig(name=name, n_clusters=n_clusters, method=method)

        # Data storage
        self._library = PointLibrary(name=name, storage_dir=storage_dir)
        self._tree = ModelTree(name, self._library, n_clusters=n_clusters)
        self._compressor = PointCompressor()

        # Tiered cache
        self._cache = TieredCache(
            memory_max_mb=memory_max_mb,
            hot_max_mb=hot_max_mb,
            disk_dir=cache_dir,
        )

        # Task queue
        self._task_queue = TaskQueue(name=name, storage_dir=storage_dir)

        # Metadata
        self._shapes: Dict[str, Tuple[int, ...]] = {}
        self._dtypes: Dict[str, np.dtype] = {}

    # ── Factory methods ──

    @classmethod
    def from_model(cls, model_id: str, n_clusters: int = 16,
                   method: str = "cluster",
                   storage_dir: Optional[Path] = None) -> "PGQ":
        """Load a HuggingFace model and compress all weights."""
        tree = load_model_to_points(
            model_id,
            n_clusters=n_clusters,
            method=method,
            storage_dir=storage_dir,
        )
        sys = cls(model_id, storage_dir, n_clusters=n_clusters, method=method)
        sys._tree = tree
        sys._library = tree.library
        return sys

    @classmethod
    def from_file(cls, path: Path) -> "PGQ":
        """Load a saved library from disk."""
        library = PointLibrary.load(path)
        sys = cls(library.name)
        sys._library = library
        sys._tree = ModelTree(library.name, library)
        sys._tree._loaded = True
        return sys

    @classmethod
    def queue(cls, model_ids: List[str], n_clusters: int = 16,
              storage_dir: Optional[Path] = None) -> ModelQueue:
        """Create a queue with multiple models."""
        config = QueueConfig(
            default_n_clusters=n_clusters,
            storage_dir=storage_dir,
        )
        queue = ModelQueue(config)
        for model_id in model_ids:
            queue.load_model(model_id, n_clusters=n_clusters)
        return queue

    # ── Core data operations ──

    def put(self, name: str, data: np.ndarray,
            method: Optional[str] = None,
            tier: str = "memory",
            compress: bool = True) -> Union[Point, np.ndarray]:
        """Store data, optionally compressing into a Point.

        Args:
            name: Data identifier.
            data: Numpy array to store.
            method: Compression method. None = use default.
            tier: Storage tier ("memory", "hot", "disk").
            compress: Whether to compress (True) or store raw (False).

        Returns:
            Point if compressed, raw data if not.
        """
        if compress:
            m = method or self._config.method
            if m == "cluster":
                point = self._compressor.compress_cluster(data, name, self._config.n_clusters)
            else:
                point = self._compressor.compress_function(data, name)
            self._library.add(point)
            self._tree._weight_shapes[name] = data.shape
            self._tree._weight_dtypes[name] = data.dtype
            self._shapes[name] = data.shape
            self._dtypes[name] = data.dtype
            return point
        else:
            self._cache.put(name, data, Tier(tier), size_bytes=data.nbytes)
            self._shapes[name] = data.shape
            self._dtypes[name] = data.dtype
            return data

    def put_raw(self, name: str, data: Any, tier: str = "memory",
                size_bytes: int = 0, ttl: Optional[float] = None) -> None:
        """Store any data (not just numpy) in the cache.

        Args:
            name: Data identifier.
            data: Any serializable data.
            tier: Storage tier.
            size_bytes: Approximate size in bytes.
            ttl: Time-to-live in seconds. None = no expiration.
        """
        self._cache.put(name, data, Tier(tier), size_bytes=size_bytes, ttl=ttl)

    def get(self, name: str) -> Optional[np.ndarray]:
        """Get data by name, decompressing if needed."""
        # Try cache first
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        # Try point library
        point = self._library.get(name)
        if point is None:
            return None

        shape = self._shapes.get(name) or self._tree._weight_shapes.get(name)
        dtype = self._dtypes.get(name) or self._tree._weight_dtypes.get(name, np.float32)
        n = int(np.prod(shape)) if shape else len(point.params.get("centroids", [])) * 100
        flat = point.generate(n)
        if shape is not None:
            flat = flat.reshape(shape)
        return flat.astype(dtype)

    def get_any(self, name: str) -> Optional[Any]:
        """Get any data from cache (not just numpy)."""
        return self._cache.get(name)

    def has(self, name: str) -> bool:
        """Check if data exists."""
        return self._library.has(name) or self._cache.exists(name)

    def remove(self, name: str) -> bool:
        """Remove data from all tiers."""
        removed = self._library.remove(name)
        removed = self._cache.remove(name) or removed
        self._shapes.pop(name, None)
        self._dtypes.pop(name, None)
        return removed

    # ── Task queue operations ──

    def submit_task(self, task: Task) -> Task:
        """Submit a task to the queue."""
        return self._task_queue.submit(task)

    def next_task(self) -> Optional[Task]:
        """Get next task to process."""
        return self._task_queue.next()

    def complete_task(self, task_id: str, result: Any = None) -> Optional[Task]:
        """Mark task as completed."""
        return self._task_queue.complete(task_id, result)

    def fail_task(self, task_id: str, error: str) -> Optional[Task]:
        """Mark task as failed."""
        return self._task_queue.fail(task_id, error)

    def cancel_task(self, task_id: str) -> Optional[Task]:
        """Cancel a task."""
        return self._task_queue.cancel(task_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._task_queue.get_task(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """List tasks, optionally filtered by status."""
        return self._task_queue.list_tasks(status)

    def pause_queue(self) -> None:
        """Pause the task queue."""
        self._task_queue.pause()

    def resume_queue(self) -> None:
        """Resume the task queue."""
        self._task_queue.resume()

    # ── Training via executor (Point-Graph-Queue integration) ──

    def submit_training(
        self,
        fn: Callable[..., Any],
        job_id: str,
        tree_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Submit a training job through the shared TrainingExecutor.

        Routes the job to a specific ModelTree (``tree_id``) for isolation.
        Trained weights are automatically compressed into Points and stored
        in the tree's PointLibrary on completion.

        Args:
            fn: Training function.  Receives (job_id, tree_id, point_library,
                is_cancelled, *args, **kwargs).
            job_id: Unique job identifier.
            tree_id: ModelTree name.  None uses ``self.name``.
            **kwargs: Forwarded to the training function.

        Returns:
            The *job_id* for tracking.
        """
        from domains.training.executor import get_training_executor

        tid = tree_id or self.name
        executor = get_training_executor()
        return executor.submit_training(
            fn, job_id, tid, self._library, **kwargs,
        )

    def training_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get training job status from the executor."""
        from domains.training.executor import get_training_executor
        return get_training_executor().status(job_id)

    def cancel_training(self, job_id: str) -> bool:
        """Cancel a training job via the executor."""
        from domains.training.executor import get_training_executor
        return get_training_executor().cancel(job_id)

    # ── Search ──

    def search(self, query: str) -> List[Point]:
        """Search points by identity."""
        return self._library.search(query)

    def best(self, n: int = 10) -> List[Point]:
        """Get best points by accuracy."""
        return self._library.best_points(n)

    # ── Persistence ──

    def save(self, path: Union[Path, str]) -> Path:
        """Save to disk (library + task queue)."""
        p = Path(path)
        self._library.save(p)
        # Save task queue alongside
        task_path = p.parent / f"{p.stem}.tasks.json"
        self._task_queue.save(task_path)
        return p

    @classmethod
    def load(cls, path: Union[Path, str]) -> "PGQ":
        """Load from disk (library + task queue)."""
        p = Path(path)
        sys = cls.from_file(p)
        # Load task queue if it exists
        task_path = p.parent / f"{p.stem}.tasks.json"
        if task_path.exists():
            sys._task_queue = TaskQueue.load(task_path)
        return sys

    # ── Stats ──

    def stats(self) -> dict:
        """System statistics."""
        tree_stats = self._tree.stats()
        cache_stats = self._cache.stats()
        queue_stats = self._task_queue.stats()

        return {
            "name": self.name,
            "tree": tree_stats,
            "cache": cache_stats,
            "queue": queue_stats,
        }

    def cache_stats(self) -> dict:
        """Cache-only statistics."""
        return self._cache.stats()

    def queue_stats(self) -> dict:
        """Queue-only statistics."""
        return self._task_queue.stats()

    def cleanup_cache(self) -> int:
        """Remove expired cache entries. Returns count removed."""
        return self._cache.cleanup_expired()

    @property
    def is_loaded(self) -> bool:
        return self._tree.is_loaded

    @property
    def library(self) -> PointLibrary:
        return self._library

    @property
    def tree(self) -> ModelTree:
        return self._tree

    @property
    def cache(self) -> TieredCache:
        return self._cache

    @property
    def task_queue(self) -> TaskQueue:
        return self._task_queue

    # ── Batch operations ──

    def put_many(self, data: Dict[str, np.ndarray], compress: bool = True,
                 method: Optional[str] = None) -> dict:
        """Store multiple arrays at once.

        Args:
            data: Dict of name → numpy array.
            compress: Whether to compress.
            method: Compression method.

        Returns:
            Stats with counts and total size.
        """
        total_bytes = 0
        count = 0
        for name, arr in data.items():
            self.put(name, arr, compress=compress, method=method)
            total_bytes += arr.nbytes if isinstance(arr, np.ndarray) else 0
            count += 1
        return {"count": count, "total_bytes": total_bytes}

    def get_many(self, names: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Get multiple arrays at once.

        Args:
            names: List of data identifiers.

        Returns:
            Dict of name → array (None if not found).
        """
        return {name: self.get(name) for name in names}

    def exists_many(self, names: List[str]) -> Dict[str, bool]:
        """Check existence of multiple keys."""
        return {name: self.has(name) for name in names}

    def remove_many(self, names: List[str]) -> int:
        """Remove multiple items. Returns count removed."""
        return sum(1 for name in names if self.remove(name))

    # ── Stats export ──

    def export_stats(self) -> dict:
        """Export full system stats as a serializable dict."""
        s = self.stats()
        s["version"] = "0.1.0"
        return s
