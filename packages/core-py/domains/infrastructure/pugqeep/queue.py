"""
TreeQueue — manages multiple Trees.

The Queue is the top-level entry point:
  - Queue = Tree (model instance)
    └── Graph = PointLibrary (context — what the tree knows)
          └── Point (meaning — generator function)

The Queue:
  - Registers multiple Trees (one per loaded model)
  - Manages shared PointLibraries (dedup across trees)
  - Provides model switching and weight sharing
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import QueueConfig, TreeConfig
from .library import PointLibrary
from .model_tree import ModelTree
from .dedup import PointDeduplicator

logger = logging.getLogger("slo.pdqeep")


class ModelQueue:
    """Top-level manager for multiple Trees."""

    def __init__(self, config: Optional[QueueConfig] = None):
        self.config = config or QueueConfig()
        self._trees: Dict[str, ModelTree] = {}
        self._shared_library: Optional[PointLibrary] = None

        if self.config.storage_dir:
            self._shared_library = PointLibrary(
                name="shared",
                storage_dir=self.config.storage_dir,
            )

    def add_tree(self, name: str, tree: Optional[ModelTree] = None,
                 config: Optional[TreeConfig] = None) -> ModelTree:
        """Add a ModelTree to the queue."""
        if len(self._trees) >= self.config.max_trees:
            raise ValueError(f"Queue full (max {self.config.max_trees} trees)")

        if tree is None:
            tc = config or TreeConfig(name=name, n_clusters=self.config.default_n_clusters)
            lib = self._shared_library if self._shared_library is not None else PointLibrary(name=f"{name}_points")
            tree = ModelTree(tc.name, lib, n_clusters=tc.n_clusters)

        self._trees[name] = tree
        logger.info("ModelQueue: added tree '%s'", name,
            extra={"tag": "INFRA"})
        return tree

    def get_tree(self, name: str) -> Optional[ModelTree]:
        return self._trees.get(name)

    def remove_tree(self, name: str) -> bool:
        if name in self._trees:
            del self._trees[name]
            logger.info("ModelQueue: removed tree '%s'", name,
                extra={"tag": "INFRA"})
            return True
        return False

    def list_trees(self) -> List[str]:
        return list(self._trees.keys())

    def load_model(self, model_id: str, n_clusters: int = 16,
                   method: str = "cluster") -> ModelTree:
        """Load a HuggingFace model into a new tree."""
        from .tree import load_model_to_points

        tree = load_model_to_points(
            model_id,
            library=self._shared_library,
            n_clusters=n_clusters,
            method=method,
        )
        self._trees[model_id] = tree
        return tree

    def deduplicate(self) -> dict:
        """Deduplicate identical points across all trees."""
        if not self.config.dedup:
            return {"merged": 0, "bytes_saved": 0, "groups": 0}

        dedup = PointDeduplicator()
        for tree in self._trees.values():
            dedup.add_library(tree.library)
        return dedup.deduplicate()

    def stats(self) -> dict:
        """Queue statistics."""
        total_points = 0
        total_raw = 0
        total_compressed = 0
        for tree in self._trees.values():
            s = tree.library.stats()
            total_points += s["total_points"]
            total_raw += s["total_raw_bytes"]
            total_compressed += s["total_compressed_bytes"]

        return {
            "num_trees": len(self._trees),
            "trees": list(self._trees.keys()),
            "total_points": total_points,
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "ratio": total_raw / max(total_compressed, 1),
            "shared_library": self._shared_library is not None,
        }

    def save_all(self, directory: Path) -> None:
        """Save all trees to a directory."""
        directory.mkdir(parents=True, exist_ok=True)
        for name, tree in self._trees.items():
            tree.library.save(directory / f"{name}.points.json")

    def load_all(self, directory: Path) -> None:
        """Load all trees from a directory.

        Expects files named ``{name}.points.json``.
        """
        for f in directory.glob("*.points.json"):
            name = f.name.rsplit(".points.json", 1)[0]
            lib = PointLibrary.load(f)
            tree = ModelTree(name, lib)
            tree.is_loaded = True
            self._trees[name] = tree
