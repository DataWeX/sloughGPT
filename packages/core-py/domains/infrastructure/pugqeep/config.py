"""
Configuration for the Point compression library.

Provides dataclass configs for Point, PointCompressor, PointLibrary, ModelTree, Queue,
Engine, Subprocess, Restart, and Monitor.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PointConfig:
    """Configuration for Point creation."""
    function_type: str = "cluster"  # cluster, periodic, linear, polynomial, raw
    n_clusters: int = 16            # VQ clusters (for cluster type)
    residual_threshold: float = 0.99  # store residual if accuracy < this


@dataclass
class CompressorConfig:
    """Configuration for PointCompressor."""
    n_clusters: int = 16
    lloyd_iterations: int = 5
    gap_fill_iterations: int = 4
    gap_fill_max_elements: int = 100_000
    method: str = "cluster"  # cluster or function


@dataclass
class LibraryConfig:
    """Configuration for PointLibrary."""
    name: str = "default"
    storage_dir: Optional[Path] = None
    auto_save: bool = False  # save after every add/remove


@dataclass
class TreeConfig:
    """Configuration for ModelTree."""
    name: str = "model"
    n_clusters: int = 16
    method: str = "cluster"
    skip_embeddings: bool = True   # don't VQ embeddings (discrete, large)
    skip_biases: bool = True       # don't VQ biases (small, discrete)


@dataclass
class QueueConfig:
    """Configuration for ModelQueue."""
    max_trees: int = 10
    default_n_clusters: int = 16
    storage_dir: Optional[Path] = None
    dedup: bool = True  # deduplicate identical points across trees


@dataclass
class SubprocessConfig:
    """Configuration for subprocess execution in GuardTree."""
    enabled: bool = True
    python_exe: str = "python3"
    max_workers: int = 4
    memory_limit_mb: Optional[int] = None
    cpu_affinity: Optional[list] = None  # e.g. [0, 1, 2, 3]
    start_method: str = "fork"  # fork, spawn, forkserver
    env: Optional[dict] = None
    cwd: Optional[str] = None  # working directory for subprocess
    capture_output: bool = False
    preexec_fn: Optional[object] = None
    terminate_grace: float = 3.0  # seconds before SIGKILL


@dataclass
class RestartPolicy:
    """Restart policy for failed processes."""
    max_restarts: int = 0
    restart_delay: float = 1.0  # seconds
    backoff: str = "exponential"  # fixed, linear, exponential
    max_backoff: float = 30.0


@dataclass
class MonitorConfig:
    """Configuration for process health monitoring."""
    enabled: bool = True
    poll_interval: float = 1.0  # seconds
    stall_timeout: float = 60.0  # seconds without heartbeat
    on_stall: str = "restart"  # restart, kill, alert
    on_restart: str = "log"  # log, alert


@dataclass
class EngineConfig:
    """Configuration for the Engine."""
    name: str = "main"
    max_trees: int = 16
    tree_workers: int = 4
    max_stems: int = 8
    queue_size: int = 128
    poll_interval: float = 0.1
    subprocess: SubprocessConfig = field(default_factory=SubprocessConfig)
    restart: RestartPolicy = field(default_factory=RestartPolicy)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
