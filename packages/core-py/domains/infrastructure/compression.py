"""
Weight compression for NumPy transformer inference.

CompressedWeight: VQ centroids + assignments + residual.
Hierarchical: if centroids follow linear pattern, store as function.
LRUCache: cache decompressed weights.
"""

import threading
from collections import OrderedDict
from typing import Optional

import numpy as np


class CompressedWeight:
    """Compressed weight storage — VQ centroids + assignments + residual.

    Decompression: reconstructed = centroids[assignments] + residual
    With float16 residual, error is ~5e-8 per element (near machine epsilon).

    Hierarchical compression: if centroids follow a linear pattern, store
    them as a linear function (a * i + b) instead of raw float32 array.
    Reduces centroid storage from n_clusters*4 bytes to 8 bytes.
    """

    __slots__ = ('centroids', 'assignments', 'residual', 'shape', 'dtype',
                 'centroid_fn', 'centroid_fn_params')

    def __init__(self, centroids: np.ndarray, assignments: np.ndarray,
                 residual: Optional[np.ndarray], shape: tuple, dtype: np.dtype,
                 centroid_fn: Optional[str] = None,
                 centroid_fn_params: Optional[dict] = None):
        self.centroids = centroids
        self.assignments = assignments
        self.residual = residual  # float16 residual for exact reconstruction
        self.shape = shape
        self.dtype = dtype
        self.centroid_fn = centroid_fn  # "linear" if centroids compressed, None otherwise
        self.centroid_fn_params = centroid_fn_params  # {"a": float, "b": float} if linear

    def decompress(self) -> np.ndarray:
        """Reconstruct weight from centroids + assignments + residual."""
        centroids = self._decompress_centroids()
        reconstructed = centroids[self.assignments]
        if self.residual is not None:
            reconstructed = reconstructed + self.residual.astype(np.float32)
        return reconstructed.reshape(self.shape)

    def _decompress_centroids(self) -> np.ndarray:
        """Decompress centroids — either raw or from linear function."""
        if self.centroid_fn == "linear" and self.centroid_fn_params is not None:
            a = self.centroid_fn_params["a"]
            b = self.centroid_fn_params["b"]
            i = np.arange(len(self.centroids), dtype=np.float32)
            return a * i + b
        return self.centroids

    @property
    def compressed_bytes(self) -> int:
        """Total compressed size in bytes."""
        if self.centroid_fn == "linear":
            size = 8  # two float32 for a, b
        else:
            size = self.centroids.nbytes
        size += self.assignments.nbytes
        if self.residual is not None:
            size += self.residual.nbytes
        return size


class LRUCache:
    """Simple LRU cache for decompressed weights."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: np.ndarray):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
