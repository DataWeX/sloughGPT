"""
Configuration for the Point compression library.

Provides dataclass configs for Point, PointCompressor, PointLibrary, ModelTree, and Queue.
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
