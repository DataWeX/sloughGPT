"""
Configuration for the Point compression library.

Provides dataclass configs for Point, PointCompressor, PointLibrary, Tree, Queue,
Engine, Subprocess, Restart, and Monitor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class PointConfig:
    """Configuration for Point creation."""
    function_type: str = "cluster"
    n_clusters: int = 16
    residual_threshold: float = 0.99


@dataclass(slots=True)
class CompressorConfig:
    """Configuration for PointCompressor."""
    n_clusters: int = 16
    lloyd_iterations: int = 5
    gap_fill_iterations: int = 4
    gap_fill_max_elements: int = 100_000
    method: str = "cluster"


@dataclass(slots=True)
class LibraryConfig:
    """Configuration for PointLibrary."""
    name: str = "default"
    storage_dir: Optional[Path] = None
    auto_save: bool = False


@dataclass(slots=True)
class TreeConfig:
    """Configuration for Tree."""
    name: str = "model"
    n_clusters: int = 16
    method: str = "cluster"
    skip_embeddings: bool = True
    skip_biases: bool = True


@dataclass(slots=True)
class QueueConfig:
    """Configuration for ModelQueue."""
    max_trees: int = 10
    default_n_clusters: int = 16
    storage_dir: Optional[Path] = None
    dedup: bool = True


@dataclass(slots=True)
class SubprocessConfig:
    """Configuration for subprocess execution in GuardTree."""
    enabled: bool = True
    python_exe: str = "python3"
    max_workers: int = 4
    memory_limit_mb: Optional[int] = None
    cpu_affinity: Optional[list] = None
    start_method: str = "fork"
    env: Optional[dict] = None
    cwd: Optional[str] = None
    capture_output: bool = False
    preexec_fn: Optional[object] = None
    terminate_grace: float = 3.0


@dataclass(slots=True)
class RestartPolicy:
    """Restart policy for failed processes."""
    max_restarts: int = 0
    restart_delay: float = 1.0
    backoff: str = "exponential"
    max_backoff: float = 30.0


@dataclass(slots=True)
class MonitorConfig:
    """Configuration for process health monitoring."""
    enabled: bool = True
    poll_interval: float = 1.0
    stall_timeout: float = 60.0
    on_stall: str = "restart"
    on_restart: str = "log"


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
