"""
Point compressor — stores weights as functions, not raw values.

Architecture:
  Queue = Tree (model instance)
    └── Graph = PointLibrary (context — what the tree knows)
          └── Point (meaning — generator function)

Components:
  - Point: weight-generating function with meaning (cluster, periodic, linear, polynomial)
  - PointCompressor: compresses weight tensors into Points
  - PointLibrary: stores, indexes, and retrieves Points (the Graph)
  - ModelTree: model instance that uses Points for inference (the Tree)
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger("man.point_compressor")


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
        if self.function_type == "raw":
            raw_bytes = base64.b64decode(self.params["data_b64"])
            return np.frombuffer(raw_bytes, dtype=self.params["dtype"])

        if self.function_type == "cluster":
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
            raise NotImplementedError("Cluster deserialization needs metadata")
        else:
            raise ValueError(f"Unknown function type: {function_type}")

        return cls(identity=identity, function_type=function_type, params=params)

    def to_dict(self) -> dict:
        """Serialize point to JSON-compatible dict."""
        d: dict[str, Any] = {
            "identity": self.identity,
            "function_type": self.function_type,
            "accuracy": self.accuracy,
        }
        if self.function_type == "cluster":
            centroids = self.params["centroids"]
            assignments = self.params["assignments"]
            d["params"] = {
                "centroids_b64": base64.b64encode(centroids.tobytes()).decode(),
                "centroids_shape": list(centroids.shape),
                "centroids_dtype": str(centroids.dtype),
                "assignments_b64": base64.b64encode(assignments.tobytes()).decode(),
                "assignments_shape": list(assignments.shape),
                "assignments_dtype": str(assignments.dtype),
            }
        else:
            d["params"] = {k: float(v) for k, v in self.params.items()}

        if self.residual is not None:
            d["residual_b64"] = base64.b64encode(self.residual.tobytes()).decode()
            d["residual_shape"] = list(self.residual.shape)
            d["residual_dtype"] = str(self.residual.dtype)

        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Point":
        """Deserialize point from dict."""
        func_type = d["function_type"]

        if func_type == "cluster":
            pd = d["params"]
            centroids = np.frombuffer(
                base64.b64decode(pd["centroids_b64"]),
                dtype=pd["centroids_dtype"],
            ).reshape(pd["centroids_shape"])
            assignments = np.frombuffer(
                base64.b64decode(pd["assignments_b64"]),
                dtype=pd["assignments_dtype"],
            ).reshape(pd["assignments_shape"])
            params = {"centroids": centroids, "assignments": assignments}
        else:
            params = {k: float(v) for k, v in d["params"].items()}

        residual = None
        if "residual_b64" in d:
            residual = np.frombuffer(
                base64.b64decode(d["residual_b64"]),
                dtype=d["residual_dtype"],
            ).reshape(d["residual_shape"])

        return cls(
            identity=d["identity"],
            function_type=func_type,
            params=params,
            residual=residual,
            accuracy=d.get("accuracy", 0.0),
        )


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
            if point.residual is not None:
                compressed_bytes += point.residual.nbytes
        elif point.function_type == "raw":
            compressed_bytes = raw_size  # no compression for raw
        else:
            compressed_bytes = 4 + len(point.params) * 4  # type + params
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


# ══════════════════════════════════════════════════════════════════════════════
# PointLibrary — Graph (context — what the tree knows)
# ══════════════════════════════════════════════════════════════════════════════

class PointLibrary:
    """Stores, indexes, and retrieves Points.

    This is the "Graph" in the Point-Graph-Queue architecture:
    - Points are organized by identity and function type
    - Provides lookup, add, remove, and search
    - Persists to disk as JSON with base64-encoded numpy arrays
    """

    def __init__(self, name: str = "default", storage_dir: Optional[Path] = None):
        """Initialize the library.

        Args:
            name: Library name (used for file naming).
            storage_dir: Directory for persistence. None = in-memory only.
        """
        self.name = name
        self._storage_dir = storage_dir
        self._points: Dict[str, Point] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._created_at = time.time()
        self._compressor = PointCompressor()

    # ── CRUD ──

    def add(self, point: Point) -> None:
        """Add a point to the library. Replaces if identity already exists."""
        self._points[point.identity] = point
        by_type = self._by_type.setdefault(point.function_type, [])
        if point.identity not in by_type:
            by_type.append(point.identity)
        logger.debug("PointLibrary[%s]: added %s (%s)", self.name, point.identity, point.function_type)

    def get(self, identity: str) -> Optional[Point]:
        """Get a point by identity."""
        return self._points.get(identity)

    def remove(self, identity: str) -> bool:
        """Remove a point by identity. Returns True if found and removed."""
        point = self._points.pop(identity, None)
        if point is None:
            return False
        by_type = self._by_type.get(point.function_type, [])
        if identity in by_type:
            by_type.remove(identity)
        return True

    def list_all(self) -> List[Point]:
        """List all points in the library."""
        return list(self._points.values())

    def list_by_type(self, function_type: str) -> List[Point]:
        """List all points of a given function type."""
        identities = self._by_type.get(function_type, [])
        return [self._points[i] for i in identities if i in self._points]

    def has(self, identity: str) -> bool:
        """Check if a point exists."""
        return identity in self._points

    def clear(self) -> None:
        """Remove all points."""
        self._points.clear()
        self._by_type.clear()

    # ── Compress & store ──

    def compress_and_store(self, weights: np.ndarray, identity: str,
                           method: str = "cluster", n_clusters: int = 16) -> Point:
        """Compress a weight tensor and store the resulting Point.

        Args:
            weights: Raw weight tensor.
            identity: Unique identifier for this point.
            method: "cluster" or "function".
            n_clusters: Number of clusters for VQ compression.

        Returns:
            The compressed Point (also stored in the library).
        """
        if method == "cluster":
            point = self._compressor.compress_cluster(weights, identity, n_clusters)
        else:
            point = self._compressor.compress_function(weights, identity)
        self.add(point)
        return point

    def decompress_to(self, identity: str, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        """Decompress a point back to a weight tensor.

        Args:
            identity: Point identity.
            shape: Optional shape to reshape to. If None, returns flat array.

        Returns:
            Reconstructed weight array, or None if point not found.
        """
        point = self.get(identity)
        if point is None:
            return None
        if point.function_type == "cluster":
            centroids = point.params["centroids"]
            assignments = point.params["assignments"]
            result = centroids[assignments]
        else:
            result = point.generate(len(point.params) * 100)  # estimate
        if shape is not None:
            result = result.reshape(shape)
        return result

    # ── Search ──

    def search(self, query: str) -> List[Point]:
        """Search points by identity substring."""
        q = query.lower()
        return [p for p in self._points.values() if q in p.identity.lower()]

    def best_points(self, n: int = 10) -> List[Point]:
        """Get the N points with highest accuracy."""
        return sorted(self._points.values(), key=lambda p: p.accuracy, reverse=True)[:n]

    # ── Statistics ──

    def stats(self) -> dict:
        """Return library statistics."""
        points = list(self._points.values())
        if not points:
            return {
                "name": self.name,
                "total_points": 0,
                "total_raw_bytes": 0,
                "total_compressed_bytes": 0,
                "avg_accuracy": 0.0,
                "types": {},
            }

        total_raw = 0
        total_compressed = 0
        for p in points:
            if p.function_type == "cluster":
                centroids = p.params["centroids"]
                assignments = p.params["assignments"]
                total_compressed += centroids.nbytes + assignments.nbytes
                total_raw += len(assignments) * 4  # float32 estimate
            else:
                total_compressed += 4 + len(p.params) * 4
                total_raw += len(p.params) * 100  # estimate
            if p.residual is not None:
                total_compressed += p.residual.nbytes

        return {
            "name": self.name,
            "total_points": len(points),
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "ratio": total_raw / max(total_compressed, 1),
            "avg_accuracy": sum(p.accuracy for p in points) / len(points),
            "types": {ft: len(ids) for ft, ids in self._by_type.items()},
        }

    # ── Persistence ──

    def save(self, path: Optional[Path] = None) -> Path:
        """Save library to JSON file.

        Args:
            path: File path. If None, uses storage_dir / {name}.points.json.

        Returns:
            Path to saved file.
        """
        if path is None:
            if self._storage_dir is None:
                raise ValueError("No storage_dir set and no path provided")
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            path = self._storage_dir / f"{self.name}.points.json"

        data = {
            "name": self.name,
            "created_at": self._created_at,
            "saved_at": time.time(),
            "points": [p.to_dict() for p in self._points.values()],
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("PointLibrary[%s]: saved %d points to %s", self.name, len(self._points), path)
        return path

    @classmethod
    def load(cls, path: Path) -> "PointLibrary":
        """Load library from JSON file.

        Args:
            path: Path to the .points.json file.

        Returns:
            Loaded PointLibrary.
        """
        data = json.loads(path.read_text())
        lib = cls(
            name=data.get("name", path.stem),
            storage_dir=path.parent,
        )
        lib._created_at = data.get("created_at", 0)
        for pd in data.get("points", []):
            lib.add(Point.from_dict(pd))
        logger.info("PointLibrary[%s]: loaded %d points from %s", lib.name, len(lib._points), path)
        return lib


# ══════════════════════════════════════════════════════════════════════════════
# ModelTree — Tree (model instance using Points)
# ══════════════════════════════════════════════════════════════════════════════

class ModelTree:
    """Model instance that compresses weights into Points and runs inference.

    This is the "Tree" in the Point-Graph-Queue architecture:
    - Selects points from a PointLibrary
    - Generates weights on demand from points
    - Maintains its own isolated state
    - No cross-talk with other trees unless the queue allows it

    Usage:
        tree = ModelTree("gpt2", library)
        tree.load_weights(weight_dict)
        tree.generate("Hello", max_tokens=50)
    """

    def __init__(self, name: str, library: Optional[PointLibrary] = None,
                 n_clusters: int = 16):
        """Initialize a model tree.

        Args:
            name: Model identifier (e.g., "gpt2", "qwen-0.5b").
            library: PointLibrary to store/retrieve points. Created if None.
            n_clusters: Number of VQ clusters per weight tensor.
        """
        self.name = name
        self.library = library or PointLibrary(name=f"{name}_points")
        self.n_clusters = n_clusters
        self._compressor = PointCompressor()
        self._weight_shapes: Dict[str, Tuple[int, ...]] = {}
        self._weight_dtypes: Dict[str, np.dtype] = {}
        self._loaded = False

    def load_weights(self, weights: Dict[str, np.ndarray], method: str = "cluster") -> dict:
        """Compress all weight tensors and store as Points in the library.

        Args:
            weights: Dict of weight_name → numpy array.
            method: "cluster" or "function" compression.

        Returns:
            Compression statistics.
        """
        total_raw = 0
        total_compressed = 0

        for name, raw in weights.items():
            point_id = f"{self.name}.{name}"
            flat = raw.flatten()

            if method == "cluster" and len(flat) < self.n_clusters * 2:
                # Too small to compress — store raw as a special point
                point = Point(
                    identity=point_id,
                    function_type="raw",
                    params={"data_b64": base64.b64encode(raw.tobytes()).decode(),
                            "shape": list(raw.shape),
                            "dtype": str(raw.dtype)},
                    accuracy=1.0,
                )
            elif method == "cluster":
                point = self._compressor.compress_cluster(flat, point_id, self.n_clusters)
            else:
                point = self._compressor.compress_function(flat, point_id)

            self.library.add(point)
            self._weight_shapes[name] = raw.shape
            self._weight_dtypes[name] = raw.dtype
            total_raw += raw.nbytes

            if point.function_type == "cluster":
                centroids = point.params["centroids"]
                assignments = point.params["assignments"]
                total_compressed += centroids.nbytes + assignments.nbytes
                if point.residual is not None:
                    total_compressed += point.residual.nbytes
            elif point.function_type == "raw":
                total_compressed += raw.nbytes
            else:
                total_compressed += 4 + len(point.params) * 4

        self._loaded = True
        ratio = total_raw / max(total_compressed, 1)
        logger.info(
            "ModelTree[%s]: loaded %d weights, %.1fx compression (%d → %d bytes)",
            self.name, len(weights), ratio, total_raw, total_compressed,
        )
        return {
            "model": self.name,
            "num_weights": len(weights),
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "ratio": ratio,
            "method": method,
        }

    def get_weight(self, name: str) -> Optional[np.ndarray]:
        """Decompress and return a weight tensor by name.

        Args:
            name: Weight name (e.g., "h.0.attn.c_attn.weight").

        Returns:
            Reconstructed weight array, or None if not found.
        """
        point_id = f"{self.name}.{name}"
        point = self.library.get(point_id)
        if point is None:
            return None

        shape = self._weight_shapes.get(name)
        dtype = self._weight_dtypes.get(name, np.float32)

        if point.function_type == "raw":
            raw_bytes = base64.b64decode(point.params["data_b64"])
            arr = np.frombuffer(raw_bytes, dtype=point.params["dtype"])
            return arr.reshape(point.params["shape"])

        flat = point.generate(point.params.get("n", 0) if "n" in point.params
                              else self._estimate_size(name))
        if shape is not None:
            flat = flat.reshape(shape)
        return flat.astype(dtype)

    def _estimate_size(self, name: str) -> int:
        """Estimate weight size from shape."""
        shape = self._weight_shapes.get(name)
        if shape:
            return int(np.prod(shape))
        return 1000  # fallback

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def stats(self) -> dict:
        """Return tree statistics."""
        lib_stats = self.library.stats()
        return {
            "model": self.name,
            "loaded": self._loaded,
            "num_weights": len(self._weight_shapes),
            "library": lib_stats,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Integration — wire PointLibrary + ModelTree into inference
# ══════════════════════════════════════════════════════════════════════════════

def load_model_to_points(
    model_id: str,
    library: Optional[PointLibrary] = None,
    n_clusters: int = 16,
    method: str = "cluster",
    storage_dir: Optional[Path] = None,
) -> ModelTree:
    """Load a HuggingFace model and compress its weights into Points.

    This is the main integration entry point — it:
    1. Loads model weights from HuggingFace cache
    2. Compresses each weight tensor into a Point
    3. Stores Points in a PointLibrary
    4. Returns a ModelTree for inference

    Args:
        model_id: HuggingFace model identifier (e.g., "gpt2").
        library: PointLibrary to store points. Created if None.
        n_clusters: Number of VQ clusters per weight tensor.
        method: "cluster" or "function" compression.
        storage_dir: Directory for library persistence.

    Returns:
        ModelTree ready for inference.
    """
    from domains.infrastructure.numpy_engine import _load_weights

    config, weights = _load_weights(model_id)

    if library is None:
        library = PointLibrary(
            name=model_id.replace("/", "_"),
            storage_dir=storage_dir,
        )

    tree = ModelTree(model_id, library, n_clusters=n_clusters)
    stats = tree.load_weights(weights, method=method)

    logger.info(
        "Loaded %s into PointLibrary: %d weights, %.1fx compression",
        model_id, stats["num_weights"], stats["ratio"],
    )
    return tree


def save_library(library: PointLibrary, path: Path) -> Path:
    """Save a PointLibrary to disk."""
    return library.save(path)


def load_library(path: Path) -> PointLibrary:
    """Load a PointLibrary from disk."""
    return PointLibrary.load(path)


# ══════════════════════════════════════════════════════════════════════════════
# PointDeduplicator — share identical points across models
# ══════════════════════════════════════════════════════════════════════════════

class PointDeduplicator:
    """Identifies and merges identical points across multiple libraries.

    When two models share identical weight tensors (e.g., common embeddings),
    their Points will have the same centroids and assignments. The deduplicator
    merges these into a single Point referenced by multiple identities.

    Usage:
        dedup = PointDeduplicator()
        dedup.add_library(lib_a)
        dedup.add_library(lib_b)
        stats = dedup.deduplicate()
        # stats["merged"] = number of duplicate points merged
    """

    def __init__(self, tolerance: float = 1e-6):
        """Initialize deduplicator.

        Args:
            tolerance: Maximum difference for considering two centroids equal.
        """
        self._tolerance = tolerance
        self._libraries: List[PointLibrary] = []
        self._fingerprints: Dict[str, List[str]] = {}  # fingerprint → [identity, ...]

    def add_library(self, library: PointLibrary) -> None:
        """Add a library to the deduplication pool."""
        self._libraries.append(library)
        for point in library.list_all():
            fp = self._fingerprint(point)
            self._fingerprints.setdefault(fp, []).append(point.identity)

    def find_duplicates(self) -> List[List[str]]:
        """Find groups of identical points across all libraries.

        Returns:
            List of identity groups, where each group contains points
            with identical centroids and assignments.
        """
        groups = []
        for fp, identities in self._fingerprints.items():
            if len(identities) > 1:
                groups.append(identities)
        return groups

    def deduplicate(self) -> dict:
        """Merge duplicate points, keeping the first occurrence.

        Returns:
            Statistics: {merged: int, bytes_saved: int, groups: int}
        """
        groups = self.find_duplicates()
        merged = 0
        bytes_saved = 0

        for group in groups:
            # Keep the first point, remove the rest
            keep = group[0]
            remove = group[1:]

            # Find which library has the keep point
            keep_point = None
            keep_lib = None
            for lib in self._libraries:
                p = lib.get(keep)
                if p is not None:
                    keep_point = p
                    keep_lib = lib
                    break

            if keep_point is None:
                continue

            # Remove duplicates from all libraries
            for identity in remove:
                for lib in self._libraries:
                    point = lib.get(identity)
                    if point is not None:
                        # Calculate saved bytes
                        if point.function_type == "cluster":
                            cents = point.params["centroids"]
                            assns = point.params["assignments"]
                            bytes_saved += cents.nbytes + assns.nbytes
                        lib.remove(identity)
                        merged += 1

        return {
            "merged": merged,
            "bytes_saved": bytes_saved,
            "groups": len(groups),
        }

    def _fingerprint(self, point: Point) -> str:
        """Create a fingerprint for a point (hash of its content)."""
        import hashlib

        if point.function_type == "cluster":
            cents = point.params["centroids"]
            assns = point.params["assignments"]
            data = cents.tobytes() + assns.tobytes()
        elif point.function_type == "raw":
            data = base64.b64decode(point.params["data_b64"])
        else:
            # Function-based: hash the parameters
            params = sorted(point.params.items())
            data = str(params).encode()

        return hashlib.sha256(data).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# PointLibrarySync — share PointLibraries between instances
# ══════════════════════════════════════════════════════════════════════════════

class PointLibrarySync:
    """Synchronize PointLibraries between instances.

    Provides:
    - export_bytes / import_bytes: serialize to/from bytes
    - sync_to_directory / sync_from_directory: sync to/from shared filesystem
    - merge: combine multiple libraries with dedup

    Usage:
        sync = PointLibrarySync()
        data = sync.export_bytes(library)
        # ... send data to another instance ...
        remote_lib = sync.import_bytes(data)
    """

    def __init__(self):
        self._dedup = PointDeduplicator()

    def export_bytes(self, library: PointLibrary) -> bytes:
        """Export a PointLibrary to bytes.

        Args:
            library: The library to export.

        Returns:
            JSON bytes containing all points.
        """
        data = {
            "name": library.name,
            "points": [p.to_dict() for p in library.list_all()],
            "exported_at": time.time(),
        }
        return json.dumps(data, indent=2).encode()

    def import_bytes(self, data: bytes) -> PointLibrary:
        """Import a PointLibrary from bytes.

        Args:
            data: JSON bytes from export_bytes.

        Returns:
            Imported PointLibrary.
        """
        parsed = json.loads(data)
        lib = PointLibrary(name=parsed.get("name", "imported"))
        for pd in parsed.get("points", []):
            lib.add(Point.from_dict(pd))
        return lib

    def sync_to_directory(self, library: PointLibrary, target_dir: Path) -> Path:
        """Sync a library to a shared directory.

        Args:
            library: The library to sync.
            target_dir: Target directory (e.g., NFS mount, S3 mount).

        Returns:
            Path to the synced file.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{library.name}.points.json"
        return library.save(path)

    def sync_from_directory(self, source_dir: Path, name: Optional[str] = None) -> Optional[PointLibrary]:
        """Load a library from a shared directory.

        Args:
            source_dir: Source directory.
            name: Library name. If None, loads the first .points.json found.

        Returns:
            Loaded PointLibrary, or None if not found.
        """
        if name is not None:
            path = source_dir / f"{name}.points.json"
            if path.exists():
                return PointLibrary.load(path)
            return None

        # Find first .points.json
        for f in source_dir.glob("*.points.json"):
            return PointLibrary.load(f)
        return None

    def merge(self, libraries: List[PointLibrary]) -> PointLibrary:
        """Merge multiple libraries into one, deduplicating identical points.

        Args:
            libraries: List of libraries to merge.

        Returns:
            Merged library with deduplicated points.
        """
        merged = PointLibrary(name="merged")
        for lib in libraries:
            for point in lib.list_all():
                merged.add(point)

        # Dedup within the merged library
        dedup = PointDeduplicator()
        dedup.add_library(merged)
        dedup.deduplicate()

        return merged
