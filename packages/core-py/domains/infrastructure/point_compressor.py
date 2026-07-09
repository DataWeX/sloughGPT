"""
Point compressor — stores weights as functions, not raw values.

Two approaches:
1. Function fitting: (a + bi) + w → a * cos(i) + b * sin(i) + w
2. Cluster compression: quantize weights to centroids, compress centroids

The cluster approach works better for neural network weights (which are random-looking).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Point:
    """A weight-generating function with meaning."""
    identity: str           # what this point represents
    function_type: str      # "periodic", "linear", "polynomial", "cluster"
    params: dict            # function parameters
    residual: Optional[np.ndarray] = None  # difference from exact
    accuracy: float = 0.0   # how well it fits (0-1)

    def generate(self, n: int) -> np.ndarray:
        """Generate n weights from this point's function."""
        if self.function_type == "cluster":
            # Cluster: use centroids + assignments
            centroids = self.params["centroids"]
            assignments = self.params["assignments"]
            return centroids[assignments[:n]]

        # Function-based generation
        i = np.arange(n, dtype=np.float32)

        if self.function_type == "periodic":
            a, b, w = self.params["a"], self.params["b"], self.params["w"]
            weights = a * np.cos(i) + b * np.sin(i) + w
        elif self.function_type == "linear":
            a, b = self.params["a"], self.params["b"]
            weights = a * i + b
        elif self.function_type == "polynomial":
            a, b, c = self.params["a"], self.params["b"], self.params["c"]
            weights = a * i**2 + b * i + c
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")

        if self.residual is not None:
            weights += self.residual[:n]
        return weights

    def to_bytes(self) -> bytes:
        """Serialize point to bytes (compact storage)."""
        import struct
        type_bytes = self.function_type[:4].encode().ljust(4, b'\0')

        if self.function_type == "cluster":
            centroids = self.params["centroids"]
            assignments = self.params["assignments"]
            param_bytes = struct.pack(f'{len(centroids)}f', *centroids)
            param_bytes += assignments.tobytes()
        elif self.function_type == "periodic":
            param_bytes = struct.pack('fff', self.params["a"], self.params["b"], self.params["w"])
        elif self.function_type in ("linear", "polynomial"):
            param_bytes = struct.pack(f'{len(self.params)}f', *self.params.values())

        return type_bytes + param_bytes

    @classmethod
    def from_bytes(cls, data: bytes, identity: str = "unknown") -> "Point":
        """Deserialize point from bytes."""
        import struct
        type_bytes = data[:4]
        function_type = type_bytes.decode().rstrip('\0')
        param_bytes = data[4:]

        if function_type == "periodic":
            a, b, w = struct.unpack('fff', param_bytes[:12])
            params = {"a": a, "b": b, "w": w}
        elif function_type == "linear":
            a, b = struct.unpack('ff', param_bytes[:8])
            params = {"a": a, "b": b}
        elif function_type == "polynomial":
            a, b, c = struct.unpack('fff', param_bytes[:12])
            params = {"a": a, "b": b, "c": c}
        elif function_type == "cluster":
            # Need to know n_clusters and n_weights
            # This is a limitation — need to store metadata
            raise NotImplementedError("Cluster deserialization needs metadata")
        else:
            raise ValueError(f"Unknown function type: {function_type}")

        return cls(identity=identity, function_type=function_type, params=params)


class PointCompressor:
    """Compresses weight tensors into points (generator functions)."""

    def compress_cluster(self, weights: np.ndarray, identity: str = "unknown",
                        n_clusters: int = 16) -> Point:
        """
        Compress using vector quantization (cluster-based).

        This is the approach that works for neural network weights.
        """
        flat = weights.flatten().astype(np.float32)
        n = len(flat)

        # Initialize centroids from quantiles
        quantiles = np.linspace(0, 100, n_clusters + 2)[1:-1]
        centroids = np.percentile(flat, quantiles)

        # Assign each weight to nearest centroid
        distances = np.abs(flat[:, np.newaxis] - centroids[np.newaxis, :])
        assignments = np.argmin(distances, axis=1).astype(np.uint8)

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

        # Try all functions
        fits = [
            ("periodic", self._fit_periodic(flat)),
            ("linear", self._fit_linear(flat)),
            ("polynomial", self._fit_polynomial(flat)),
        ]

        # Pick best (lowest MSE)
        best_type, (params, mse) = min(fits, key=lambda x: x[1][1])
        accuracy = 1.0 - mse / (var + 1e-8)

        # Store residual only if accuracy < 0.99
        if accuracy < 0.99:
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
                method: str = "cluster") -> Point:
        """Compress using specified method."""
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
        else:
            compressed_bytes = 4 + len(point.params) * 4  # type + params
            if point.residual is not None:
                compressed_bytes += point.residual.nbytes

        return {
            "raw_bytes": raw_size,
            "compressed_bytes": compressed_bytes,
            "ratio": raw_size / compressed_bytes,
            "accuracy": point.accuracy,
            "function_type": point.function_type,
        }

    def _fit_periodic(self, flat: np.ndarray) -> Tuple[dict, float]:
        """Fit (a + bi) + w → a * cos(i) + b * sin(i) + w."""
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
