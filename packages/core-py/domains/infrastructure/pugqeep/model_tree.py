"""
ModelTree — ML-specific Tree with skip logic for embeddings and biases.

Extends Tree with:
  - skip_embeddings: don't VQ embedding layers (discrete, large)
  - skip_biases: don't VQ bias tensors (small, discrete)
  - model-specific stats key

All operations delegate to Tree's generic load_data/get_data.
"""
from __future__ import annotations

import base64
from typing import Optional

import numpy as np

from .tree import (
    Tree,
    load_model_to_points,
    load_from_points,
    decompress_tree,
    save_library,
    load_library,
)
from .strategies import CompressStrategy, RawStrategy, ClusterStrategy
from .point import Point
from .compressor import PointCompressor
from .library import PointLibrary
from .config import TreeConfig


class ModelTree(Tree):
    """ML-specialized Tree that skips VQ for embeddings and biases.

    Args:
        name: Model identifier.
        library: Optional pre-existing PointLibrary.
        n_clusters: VQ cluster count.
        config: Optional TreeConfig (overrides n_clusters, method, skip_embeddings, skip_biases).
        compressor: Optional PointCompressor (overrides config's compressor settings).
    """

    def __init__(self, name: str, library: Optional[PointLibrary] = None,
                 n_clusters: int = 16, config: Optional[TreeConfig] = None,
                 compressor: Optional[PointCompressor] = None):
        super().__init__(name, library, n_clusters, config, compressor)

        if config is not None:
            self._skip_embeddings = config.skip_embeddings
            self._skip_biases = config.skip_biases
        else:
            self._skip_embeddings = True
            self._skip_biases = True

    # ── ML convenience wrappers (delegate to Tree's generic methods) ──

    def load_weights(self, weights, method=None, num_workers=0, on_progress=None):
        """Load model weights. Delegates to load_data."""
        stats = self.load_data(weights, method, num_workers, on_progress)
        stats["model"] = stats.pop("tree", self.name)
        stats["num_weights"] = stats["num_items"]
        return stats

    def get_weight(self, name):
        """Get a weight tensor. Delegates to get_data."""
        return self.get_data(name)

    def get_weights(self, names=None):
        """Get multiple weight tensors. Delegates to get_data_batch."""
        return self.get_data_batch(names)

    # ── Override compression with skip logic ──

    def _compress_item(self, name: str, raw: np.ndarray, strategy: CompressStrategy) -> tuple:
        """Compress with skip logic for embeddings and biases."""
        point_id = f"{self.name}.{name}"
        flat = raw.flatten()

        skip = False
        if self._skip_embeddings and ("embed" in name.lower() or "embedding" in name.lower()):
            skip = True
        if self._skip_biases and name.lower().endswith("bias"):
            skip = True

        if skip or (isinstance(strategy, ClusterStrategy) and len(flat) < self.n_clusters * 2):
            return RawStrategy().compress(name, raw, point_id, self.n_clusters)

        return strategy.compress(name, raw, point_id, self.n_clusters)

    def stats(self) -> dict:
        lib_stats = self.library.stats()
        return {
            "model": self.name,
            "loaded": self._loaded,
            "num_items": len(self._shapes),
            "num_weights": len(self._shapes),
            "library": lib_stats,
        }
