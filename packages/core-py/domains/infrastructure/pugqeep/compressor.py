"""
PointCompressor — compresses weight tensors into Points.

Supports:
  - Vector quantization (cluster-based) with Lloyd's refinement
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

        # Input validation
        if weights.size == 0:
            raise ValueError(f"Cannot compress empty array: {identity}")
        if not np.isfinite(weights).all():
            raise ValueError(f"Array contains NaN/Inf values: {identity}")
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")

        flat = weights.flatten().astype(np.float32)
        n = len(flat)

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

        # Compute accuracy
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
            params={"centroids": centroids, "assignments": assignments},
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
        elif method == "block_q4":
            return self.compress_block_q4(weights, identity)
        elif method == "block_q8":
            return self.compress_block_q8(weights, identity)
        else:
            raise ValueError(f"Unknown method: {method}")

    def decompress(self, point: Point, n: int) -> np.ndarray:
        """Decompress a point back to weights."""
        if point.function_type == "block_q4":
            return self.decompress_block_q4(point)
        elif point.function_type == "block_q8":
            return self.decompress_block_q8(point)
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
        elif point.function_type == "block_q4":
            compressed_bytes = point.nbytes()
        elif point.function_type == "block_q8":
            compressed_bytes = point.nbytes()
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

    # ── Block quantization (Q4_K style) ──

    BLOCK_SIZE = 32

    def compress_block_q4(self, weights: np.ndarray,
                          identity: str = "unknown") -> Point:
        """Compress using block-wise 4-bit quantization (Q4_K style).

        Each block of 32 values gets its own min/max/scale.
        Values are quantized to uint4 (0-15) and packed 2 per byte.

        Memory: (min:f32 + scale:f32 + packed:16) per 32 values = 24 bytes/block
        Ratio:  4 bytes/element → 0.75 bytes/element = 5.3:1
        """
        flat = weights.flatten().astype(np.float32)
        n = len(flat)
        bs = self.BLOCK_SIZE

        # Pad to multiple of block_size
        pad = (bs - n % bs) % bs
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        n_blocks = len(flat) // bs
        blocks = flat.reshape(n_blocks, bs)

        # Per-block min, max, scale
        bmin = blocks.min(axis=1)
        bmax = blocks.max(axis=1)
        brange = bmax - bmin
        # Avoid division by zero for constant blocks
        brange = np.maximum(brange, 1e-10)
        scale = brange / 15.0  # 15 = max uint4 value

        # Quantize: q = round((val - min) / scale), clamp to [0, 15]
        q = ((blocks - bmin[:, None]) / scale[:, None])
        q = np.clip(np.round(q), 0, 15).astype(np.uint8)

        # Pack uint4 into uint8 (2 per byte, low nibble first)
        n_values = n_blocks * bs
        q_flat = q.ravel()
        packed = np.zeros(n_values // 2, dtype=np.uint8)
        packed = q_flat[0::2].astype(np.uint8) | (q_flat[1::2].astype(np.uint8) << 4)

        # Compute accuracy
        deq = q.astype(np.float32) * scale[:, None] + bmin[:, None]
        reconstructed = deq.ravel()[:n]
        mse = np.mean((flat[:n] - reconstructed) ** 2)
        var = np.var(flat[:n])
        accuracy = 1.0 - mse / (var + 1e-8)

        return Point(
            identity=identity,
            function_type="block_q4",
            params={
                "mins": bmin.astype(np.float32),
                "scales": scale.astype(np.float32),
                "packed": packed,
                "n_elements": n,
                "n_blocks": n_blocks,
                "block_size": bs,
            },
            accuracy=float(accuracy),
            dtype=str(weights.dtype),
            shape=weights.shape,
        )

    def decompress_block_q4(self, point: Point) -> np.ndarray:
        """Decompress a block_q4 Point back to float32."""
        mins = point.params["mins"]
        scales = point.params["scales"]
        packed = point.params["packed"]
        n = point.params["n_elements"]
        n_blocks = point.params["n_blocks"]
        bs = point.params["block_size"]

        # Unpack uint4 → uint8
        unpacked = np.zeros(n_blocks * bs, dtype=np.uint8)
        unpacked[0::2] = packed & 0x0F
        unpacked[1::2] = (packed >> 4) & 0x0F

        # Dequantize: val = q * scale + min
        q = unpacked.reshape(n_blocks, bs).astype(np.float32)
        deq = q * scales[:, None] + mins[:, None]

        return deq.ravel()[:n]

    def compress_block_q8(self, weights: np.ndarray,
                          identity: str = "unknown") -> Point:
        """Compress using block-wise 8-bit quantization.

        Each block of 32 values gets its own min/max/scale.
        Values are quantized to uint8 (0-255).

        Memory: (min:f32 + scale:f32 + values:32) per 32 values = 40 bytes/block
        Ratio:  4 bytes/element → 1.25 bytes/element = 3.2:1
        Better accuracy than Q4.
        """
        flat = weights.flatten().astype(np.float32)
        n = len(flat)
        bs = self.BLOCK_SIZE

        pad = (bs - n % bs) % bs
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        n_blocks = len(flat) // bs
        blocks = flat.reshape(n_blocks, bs)

        bmin = blocks.min(axis=1)
        bmax = blocks.max(axis=1)
        brange = bmax - bmin
        brange = np.maximum(brange, 1e-10)
        scale = brange / 255.0

        q = ((blocks - bmin[:, None]) / scale[:, None])
        q = np.clip(np.round(q), 0, 255).astype(np.uint8)

        # Compute accuracy
        deq = q.astype(np.float32) * scale[:, None] + bmin[:, None]
        reconstructed = deq.ravel()[:n]
        mse = np.mean((flat[:n] - reconstructed) ** 2)
        var = np.var(flat[:n])
        accuracy = 1.0 - mse / (var + 1e-8)

        return Point(
            identity=identity,
            function_type="block_q8",
            params={
                "mins": bmin.astype(np.float32),
                "scales": scale.astype(np.float32),
                "values": q,
                "n_elements": n,
                "n_blocks": n_blocks,
                "block_size": bs,
            },
            accuracy=float(accuracy),
            dtype=str(weights.dtype),
            shape=weights.shape,
        )

    def decompress_block_q8(self, point: Point) -> np.ndarray:
        """Decompress a block_q8 Point back to float32."""
        mins = point.params["mins"]
        scales = point.params["scales"]
        values = point.params["values"]
        n = point.params["n_elements"]
        n_blocks = point.params["n_blocks"]
        bs = point.params["block_size"]

        q = values.reshape(n_blocks, bs).astype(np.float32)
        deq = q * scales[:, None] + mins[:, None]

        return deq.ravel()[:n]
