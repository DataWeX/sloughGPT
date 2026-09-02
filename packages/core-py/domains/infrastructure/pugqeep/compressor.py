"""
PointCompressor — compresses weight tensors into Points.

Supports:
  - Vector quantization (cluster-based) with Lloyd's refinement
  - Adaptive k: cluster count varies per layer by weight entropy
  - Centroid int8: quantize centroids to int8 for better ratio
  - Function fitting (periodic, linear, polynomial) with residual storage
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

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
        adaptive_k: If True, vary cluster count per layer by weight entropy.
        quantize_centroids: If True, quantize centroids to int8 when safe.
    """

    def __init__(self, config: Optional[CompressorConfig] = None, *,
                 n_clusters: int = 16, lloyd_iterations: int = 5,
                 residual_threshold: float = 0.99,
                 adaptive_k: bool = True,
                 quantize_centroids: bool = True):
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
        self.adaptive_k = adaptive_k
        self.quantize_centroids = quantize_centroids

    def compress_cluster(self, weights: np.ndarray, identity: str = "unknown",
                        n_clusters: Optional[int] = None) -> Point:
        """
        Compress using vector quantization (cluster-based).

        This is the approach that works for neural network weights.
        When adaptive_k is enabled, cluster count varies by weight entropy.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        # Input validation
        if weights.size == 0:
            raise ValueError(f"Cannot compress empty array: {identity}")
        if not np.isfinite(weights).all():
            raise ValueError(f"Array contains NaN/Inf values: {identity}")
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")

        flat = weights.flatten().astype(np.float32)
        n = len(flat)

        # Adaptive k: adjust cluster count based on weight entropy
        if self.adaptive_k:
            n_clusters = self._compute_adaptive_k(flat, n_clusters)

        # Clamp n_clusters to array size
        if n_clusters > n:
            n_clusters = n

        # Max-min init: quantile + gap-filling → Lloyd's refinement
        quantiles = np.linspace(0, 100, n_clusters + 2)[1:-1]
        centroids = np.percentile(flat, quantiles).astype(np.float32)
        centroids.sort()
        # Fill largest gaps for smaller weights only
        if n < self.gap_fill_max_elements and len(centroids) > 1:
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

        # Quantize centroids to int8 if enabled and safe
        centroid_quantized = False
        centroid_scale = None
        centroid_zero_point = None
        if self.quantize_centroids and nc >= 4:
            cmin, cmax = float(centroids.min()), float(centroids.max())
            crange = cmax - cmin
            if crange > 1e-12:
                c_scale = crange / 255.0
                c_zero_point = -cmin / c_scale
                q_centroids = np.clip(
                    np.round(centroids / c_scale + c_zero_point), 0, 255
                ).astype(np.uint8)
                # Dequantize and check accuracy loss
                recon_centroids = (q_centroids.astype(np.float32) - c_zero_point) * c_scale
                recon_via_q = recon_centroids[assignments]
                q_cos = float(np.dot(flat, recon_via_q) / (
                    np.linalg.norm(flat) * np.linalg.norm(recon_via_q) + 1e-12))
                # Original accuracy
                orig_cos = float(np.dot(flat, centroids[assignments]) / (
                    np.linalg.norm(flat) * np.linalg.norm(centroids[assignments]) + 1e-12))
                # Use int8 if >99.9% of original cosine preserved
                if q_cos > orig_cos * 0.999:
                    centroids = q_centroids  # store uint8 centroids
                    centroid_scale = c_scale
                    centroid_zero_point = c_zero_point
                    centroid_quantized = True

        # Compute accuracy
        if centroid_quantized:
            reconstructed = (centroids.astype(np.float32) - centroid_zero_point) * centroid_scale
            reconstructed = reconstructed[assignments]
        else:
            reconstructed = centroids[assignments]
        mse = np.mean((flat - reconstructed) ** 2)
        var = np.var(flat)
        accuracy = 1.0 - mse / (var + 1e-8)

        # Store residual if accuracy below threshold
        residual = None
        if accuracy < self.residual_threshold:
            residual = (flat - reconstructed).astype(np.float32)

        return Point(
            identity=identity,
            function_type="cluster",
            params={
                "centroids": centroids,
                "assignments": assignments,
                "centroid_quantized": centroid_quantized,
                "centroid_scale": centroid_scale,
                "centroid_zero_point": centroid_zero_point,
            },
            residual=residual,
            accuracy=float(accuracy),
            dtype=str(weights.dtype),
            shape=weights.shape,
        )

    def compress_function(self, weights: np.ndarray, identity: str = "unknown") -> Point:
        """
        Compress using function fitting (periodic, linear, polynomial).

        This works for structured weights but not random ones.
        """
        # Input validation
        if weights.size == 0:
            raise ValueError(f"Cannot compress empty array: {identity}")
        if not np.isfinite(weights).all():
            raise ValueError(f"Array contains NaN/Inf values: {identity}")

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
            dtype=str(weights.dtype),
            shape=weights.shape,
        )

    def compress_batch(self, weights_dict: Dict[str, np.ndarray],
                       method: Optional[str] = None,
                       prefix: str = "") -> Dict[str, Point]:
        """Compress multiple weight tensors in one call.

        Args:
            weights_dict: Dict mapping weight names to numpy arrays.
            method: Compression method (overrides self.method).
            prefix: Optional prefix for point identities.

        Returns:
            Dict mapping weight names to compressed Points.
        """
        results = {}
        for name, weights in weights_dict.items():
            identity = f"{prefix}{name}" if prefix else name
            results[name] = self.compress(weights, identity=identity, method=method)
        return results

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
            if point.params.get("centroid_quantized"):
                # Quantized: centroids are uint8 + 8 bytes for scale/zero_point
                compressed_bytes = centroids.nbytes + 8 + assignments.nbytes
            else:
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

    def _compute_adaptive_k(self, flat: np.ndarray, base_k: int) -> int:
        """Choose cluster count based on weight distribution entropy.

        High entropy (spread out weights) → more clusters.
        Low entropy (peaked weights) → fewer clusters.
        Scales k between base_k/2 and base_k*4.
        """
        # Bin into 256 bins to estimate entropy
        hist, _ = np.histogram(flat, bins=256)
        hist = hist[hist > 0].astype(np.float64)
        probs = hist / hist.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        # Normalize: uniform ~8 bits, peaked ~2 bits
        scale = entropy / 8.0  # 0..1
        k = int(base_k * (0.5 + 3.5 * scale))
        return max(4, min(k, 256))


