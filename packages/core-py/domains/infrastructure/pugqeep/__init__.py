"""
pugqeep — Point-Graph-Queue system.

Core infra engine for spawning processes, branching parallel tasks,
and managing data across tiers.

Architecture:
    Queue (core engine — main process)
      └── Tree (model instance — branches stems into parallel tasks)
            └── Graph/PointLibrary (context — what the tree knows)
                  └── Point (star — function-calling capacity)

Quick start:
    from pugqeep import PGQ

    # Spawn the core engine
    pgq = PGQ("infra")
    pgq.spawn(load_config, "config.json")
    pgq.spawn(start_server, port=8000)
    pgq.run()

    # Or use data operations
    pgq.put("weights", numpy_array)
    data = pgq.get("weights")
"""

from .point import Point
from .point_interface import PointProtocol, PointView, FunctionType
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
from .engine import Engine, Process, Stem, Tree as EngineTree, ProcessStatus, StemStatus
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
from domains.infrastructure.producer_consumer import (
    ProducerConsumerQueue,
    ShutdownMode,
)

__all__ = [
    # Core types
    "Point",
    "PointProtocol",
    "PointView",
    "FunctionType",
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

    # Producer-consumer
    "ProducerConsumerQueue",
    "ShutdownMode",
]

__version__ = "0.1.0"
