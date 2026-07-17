"""
pugqeep — Point-Graph-Queue system.

Generic data structure manager with tiered caching:
  - Disk → Hot cache → In-memory
  - Compress any structured data into Points (generator functions)
  - Manage task queues with priority and routing

Architecture:
    Queue = Tree (model instance)
      └── Graph = PointLibrary (context — what the tree knows)
            └── Point (meaning — generator function)

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

from .point import Point
from .compressor import PointCompressor
from .library import PointLibrary
from .model_tree import ModelTree, load_model_to_points
from .queue import ModelQueue
from .cache import TieredCache, Tier, MemoryStore as CacheMemoryStore, DiskStore, HotStore
from .task_queue import TaskQueue, Task, TaskStatus, TaskPriority
from .dedup import PointDeduplicator, PointLibrarySync
from .store import MemoryStore as FunctionMemoryStore, JSONStore, DirectoryStore
from .config import PointConfig, CompressorConfig, LibraryConfig, TreeConfig, QueueConfig
from .facade import PGQ
from .generic import (
    PGQGeneric,
    CompressionStrategy,
    StorageBackend,
    FunctionType,
    registry,
    ClusterStrategy,
    FunctionStrategy,
    RawStrategy,
    AutoStrategy,
    MemoryStorage,
    JSONStorage,
    DirectoryStorage,
)

__all__ = [
    # Core types
    "Point",
    "PointCompressor",
    "PointLibrary",
    "ModelTree",
    "ModelQueue",

    # Facade
    "PGQ",

    # Generic pluggable facade
    "PGQGeneric",

    # ABCs
    "CompressionStrategy",
    "StorageBackend",
    "FunctionType",

    # Registry
    "registry",

    # Built-in strategies
    "ClusterStrategy",
    "FunctionStrategy",
    "RawStrategy",
    "AutoStrategy",

    # Built-in storage backends
    "MemoryStorage",
    "JSONStorage",
    "DirectoryStorage",

    # Cache
    "TieredCache",
    "Tier",

    # Task queue
    "TaskQueue",
    "Task",
    "TaskStatus",
    "TaskPriority",

    # Stores
    "FunctionMemoryStore",
    "CacheMemoryStore",
    "JSONStore",
    "DirectoryStore",

    # Config
    "PointConfig",
    "CompressorConfig",
    "LibraryConfig",
    "TreeConfig",
    "QueueConfig",

    # Sync/dedup
    "PointDeduplicator",
    "PointLibrarySync",

    # Helpers
    "load_model_to_points",
]

__version__ = "0.1.0"
