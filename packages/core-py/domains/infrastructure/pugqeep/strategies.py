"""
Compression strategies for Tree.

ABC + implementations: ClusterStrategy, FunctionStrategy, RawStrategy,
BlockQuantStrategy (Q4_K and Q8 block quantization).
Each knows how to compress a numpy array into a Point.
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .point import Point


class CompressStrategy(ABC):
    """Abstract compression strategy."""

    __slots__ = ()

    @abstractmethod
    def compress(self, name: str, raw: np.ndarray, point_id: str,
                 n_clusters: int) -> Tuple[Point, int]:
        """Compress raw data into a Point.

        Args:
            name: Item name (for logging/metadata).
            raw: Numpy array to compress.
            point_id: Full point identity.
            n_clusters: VQ cluster count (may be ignored).

        Returns:
            Tuple of (Point, compressed_bytes).
        """
        ...


class ClusterStrategy(CompressStrategy):
    """Vector quantization compression."""

    __slots__ = ('_compressor',)

    def __init__(self, compressor):
        self._compressor = compressor

    def compress(self, name: str, raw: np.ndarray, point_id: str,
                 n_clusters: int) -> Tuple[Point, int]:
        flat = raw.flatten()
        point = self._compressor.compress_cluster(flat, point_id, n_clusters)
        centroids = point.params["centroids"]
        assignments = point.params["assignments"]
        compressed_bytes = centroids.nbytes + assignments.nbytes
        if point.residual is not None:
            compressed_bytes += point.residual.nbytes
        return point, compressed_bytes


class FunctionStrategy(CompressStrategy):
    """Function fitting compression (periodic, linear, polynomial)."""

    __slots__ = ('_compressor',)

    def __init__(self, compressor):
        self._compressor = compressor

    def compress(self, name: str, raw: np.ndarray, point_id: str,
                 n_clusters: int) -> Tuple[Point, int]:
        flat = raw.flatten()
        point = self._compressor.compress_function(flat, point_id)
        # Count actual stored bytes: 4-byte header + param floats + residual
        param_bytes = sum(4 for _ in point.params)  # float32 per param
        compressed_bytes = 4 + param_bytes
        if point.residual is not None:
            compressed_bytes += point.residual.nbytes
        return point, compressed_bytes


class RawStrategy(CompressStrategy):
    """No compression — stores raw bytes as base64."""

    __slots__ = ()

    def compress(self, name: str, raw: np.ndarray, point_id: str,
                 n_clusters: int) -> Tuple[Point, int]:
        point = Point(
            identity=point_id,
            function_type="raw",
            params={"data_b64": base64.b64encode(raw.tobytes()).decode(),
                    "shape": list(raw.shape),
                    "dtype": str(raw.dtype)},
            accuracy=1.0,
        )
        return point, raw.nbytes


class BlockQuantStrategy(CompressStrategy):
    """Block-wise quantization (Q4_K / Q8) for CPU inference.

    Each block of 32 values gets its own min/max/scale.
    Q4_K: 5.3:1 ratio, ~99% cosine.
    Q8:   3.2:1 ratio, ~99.9% cosine.
    """

    __slots__ = ('_compressor', '_bits')

    def __init__(self, compressor, bits: int = 4):
        self._compressor = compressor
        self._bits = bits

    def compress(self, name: str, raw: np.ndarray, point_id: str,
                 n_clusters: int) -> Tuple[Point, int]:
        flat = raw.flatten()
        if self._bits == 4:
            point = self._compressor.compress_block_q4(flat, point_id)
            mins = point.params["mins"]
            scales = point.params["scales"]
            packed = point.params["packed"]
            compressed_bytes = mins.nbytes + scales.nbytes + packed.nbytes
        else:
            point = self._compressor.compress_block_q8(flat, point_id)
            mins = point.params["mins"]
            scales = point.params["scales"]
            values = point.params["values"]
            compressed_bytes = mins.nbytes + scales.nbytes + values.nbytes
        return point, compressed_bytes
