"""
PointCompressor — compresses weight tensors into Points.

Supports:
  - Vector quantization (cluster-based) with Lloyd's refinement
  - Function fitting (periodic, linear, polynomial) with residual storage
"""

from typing import Optional, Tuple

import numpy as np

from .point import Point
from .config import CompressorConfig


class PointCompressor:
    """Compresses weight tensors into points (generator functions).

    Args:
        config: Optional CompressorConfig. If None, uses defaults.
        n_clusters: Override config n_clusters.
        lloyd_iterations: Override config lloyd_iterations.
        residual_threshold: Accuracy below this stores residual (0-1).
    """

    def __init__(self, config: Optional[CompressorConfig] = None, *,
                 n_clusters: int = 16, lloyd_iterations: int = 5,
                 residual_threshold: float = 0.99):
        if config is not None:
            self.n_clusters = config.n_clusters
            self.lloyd_iterations = config.lloyd_iterations
            self.gap_fill_iterations = config.gap_fill_iterations
            self.gap_fill_max_elements = config.gap_fill_max_elements
            self.method = config.method
        else:
            self.n_clusters = n_clusters
            self.lloyd_iterations = lloyd_iterations
            self.gap_fill_iterations = 4
            self.gap_fill_max_elements = 100_000
            self.method = "cluster"
        self.residual_threshold = residual_threshold

    def compress_cluster(self, weights: np.ndarray, identity: str = "unknown",
                        n_clusters: Optional[int] = None) -> Point:
        """
        Compress using vector quantization (cluster-based).

        This is the approach that works for neural network weights.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        flat = weights.flatten().astype(np.float32)
        n = len(flat)

        # Max-min init: quantile + gap-filling → Lloyd's refinement
        quantiles = np.linspace(0, 100, n_clusters + 2)[1:-1]
        centroids = np.percentile(flat, quantiles)
        centroids.sort()
        # Fill largest gaps for smaller weights only
        if n < self.gap_fill_max_elements:
            for _ in range(self.gap_fill_iterations):
                gaps = np.diff(centroids)
                biggest = np.argmax(gaps)
                new_c = (centroids[biggest] + centroids[biggest + 1]) / 2
                centroids = np.sort(np.append(centroids, new_c))

        # Lloyd's refinement
        nc = len(centroids)
        for _ in range(self.lloyd_iterations):
            assignments = np.clip(np.searchsorted(centroids, flat), 0, nc - 1).astype(np.uint8)
            sums = np.bincount(assignments, weights=flat, minlength=nc)
            counts = np.bincount(assignments, minlength=nc).astype(np.float64)
            alive = counts > 0
            centroids[alive] = (sums[alive] / counts[alive]).astype(np.float32)

        # Compute accuracy
        reconstructed = centroids[assignments]
        mse = np.mean((flat - reconstructed) ** 2)
        var = np.var(flat)
        accuracy = 1.0 - mse / (var + 1e-8)

        return Point(
            identity=identity,
            function_type="cluster",
            params={"centroids": centroids, "assignments": assignments},
            accuracy=float(accuracy),
        )

    def compress_function(self, weights: np.ndarray, identity: str = "unknown") -> Point:
        """
        Compress using function fitting (periodic, linear, polynomial).

        This works for structured weights but not random ones.
        """
        flat = weights.flatten().astype(np.float32)
        n = len(flat)
        var = np.var(flat)

        fits = [
            ("periodic", self._fit_periodic(flat)),
            ("linear", self._fit_linear(flat)),
            ("polynomial", self._fit_polynomial(flat)),
        ]

        best_type, (params, mse) = min(fits, key=lambda x: x[1][1])
        accuracy = 1.0 - mse / (var + 1e-8)

        if accuracy < self.residual_threshold:
            i = np.arange(n, dtype=np.float32)
            if best_type == "periodic":
                fitted = params["a"] * np.cos(i) + params["b"] * np.sin(i) + params["w"]
            elif best_type == "linear":
                fitted = params["a"] * i + params["b"]
            elif best_type == "polynomial":
                fitted = params["a"] * i**2 + params["b"] * i + params["c"]
            residual = flat - fitted
        else:
            residual = None

        return Point(
            identity=identity,
            function_type=best_type,
            params=params,
            residual=residual,
            accuracy=float(accuracy),
        )

    def compress(self, weights: np.ndarray, identity: str = "unknown",
                method: Optional[str] = None) -> Point:
        """Compress using specified method (defaults to self.method)."""
        if method is None:
            method = self.method
        if method == "cluster":
            return self.compress_cluster(weights, identity)
        elif method == "function":
            return self.compress_function(weights, identity)
        else:
            raise ValueError(f"Unknown method: {method}")

    def decompress(self, point: Point, n: int) -> np.ndarray:
        """Decompress a point back to weights."""
        return point.generate(n)

    def measure_compression(self, weights: np.ndarray, point: Point) -> dict:
        """Measure compression ratio and accuracy."""
        raw_size = weights.nbytes

        if point.function_type == "cluster":
            centroids = point.params["centroids"]
            assignments = point.params["assignments"]
            compressed_bytes = centroids.nbytes + assignments.nbytes
            if point.residual is not None:
                compressed_bytes += point.residual.nbytes
        elif point.function_type == "raw":
            compressed_bytes = raw_size
        else:
            compressed_bytes = 4 + len(point.params) * 4
            if point.residual is not None:
                compressed_bytes += point.residual.nbytes

        return {
            "raw_bytes": raw_size,
            "compressed_bytes": compressed_bytes,
            "ratio": raw_size / max(compressed_bytes, 1),
            "accuracy": point.accuracy,
            "function_type": point.function_type,
        }

    def _fit_periodic(self, flat: np.ndarray) -> Tuple[dict, float]:
        """Fit a * cos(i) + b * sin(i) + w."""
        n = len(flat)
        i = np.arange(n, dtype=np.float32)
        A = np.column_stack([np.cos(i), np.sin(i), np.ones(n)])
        result, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
        a, b, w = result
        fitted = a * np.cos(i) + b * np.sin(i) + w
        mse = np.mean((flat - fitted) ** 2)
        return {"a": float(a), "b": float(b), "w": float(w)}, mse

    def _fit_linear(self, flat: np.ndarray) -> Tuple[dict, float]:
        """Fit a * i + b."""
        n = len(flat)
        i = np.arange(n, dtype=np.float32)
        A = np.column_stack([i, np.ones(n)])
        result, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
        a, b = result
        fitted = a * i + b
        mse = np.mean((flat - fitted) ** 2)
        return {"a": float(a), "b": float(b)}, mse

    def _fit_polynomial(self, flat: np.ndarray) -> Tuple[dict, float]:
        """Fit a * i^2 + b * i + c."""
        n = len(flat)
        i = np.arange(n, dtype=np.float32)
        A = np.column_stack([i**2, i, np.ones(n)])
        result, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
        a, b, c = result
        fitted = a * i**2 + b * i + c
        mse = np.mean((flat - fitted) ** 2)
        return {"a": float(a), "b": float(b), "c": float(c)}, mse
