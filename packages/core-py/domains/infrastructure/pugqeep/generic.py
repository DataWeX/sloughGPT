"""
Generic pluggable architecture for pugqeep.

Three registries:
  1. CompressionStrategy — custom compressors (ABC)
  2. StorageBackend — custom persistence (ABC)
  3. FunctionType — custom Point generator types

Composable facade:
  PGQGeneric — wire any strategy + storage + types together.

Usage:
    from pugqeep.generic import PGQGeneric, CompressionStrategy, registry

    # Register a custom compression strategy
    class Quantize8Bit(CompressionStrategy):
        name = "q8"
        def compress(self, data, identity):
            ...  # return Point
        def decompress(self, point, n):
            ...  # return np.ndarray

    registry.compressors.register(Quantize8Bit())

    # Use it
    sys = PGQGeneric(compressor="q8")
    sys.put("weights", array)
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from .point import Point
from .store import MemoryStore, JSONStore, DirectoryStore, Store

logger = logging.getLogger("slo.pugqeep")


# ══════════════════════════════════════════════════════════════════════════════
# ABCs
# ══════════════════════════════════════════════════════════════════════════════

class CompressionStrategy(ABC):
    """Base class for compression strategies.

    Subclass this to add custom compression algorithms.
    Each strategy has a unique name and implements compress/decompress.
    """

    name: str = "base"

    @abstractmethod
    def compress(self, data: np.ndarray, identity: str = "unknown", **kwargs) -> Point:
        """Compress a numpy array into a Point.

        Args:
            data: Numpy array to compress.
            identity: Name/path for this point.
            **kwargs: Strategy-specific options (n_clusters, threshold, etc).

        Returns:
            Compressed Point with generate() capability.
        """
        ...

    @abstractmethod
    def decompress(self, point: Point, n: int) -> np.ndarray:
        """Decompress a Point back to numpy array.

        Args:
            point: Point to decompress.
            n: Number of elements to generate.

        Returns:
            Numpy array of reconstructed values.
        """
        ...

    def nbytes(self, point: Point) -> int:
        """Estimate stored bytes for a Point. Override for accuracy."""
        return point.nbytes()


class StorageBackend(ABC):
    """Base class for storage backends.

    Subclass this to add custom persistence (S3, Redis, SQLite, etc).
    """

    name: str = "base"

    @abstractmethod
    def save(self, point: Point) -> None:
        """Persist a Point."""
        ...

    @abstractmethod
    def load(self, identity: str) -> Optional[Point]:
        """Load a Point by identity. Returns None if not found."""
        ...

    @abstractmethod
    def remove(self, identity: str) -> bool:
        """Remove a Point. Returns True if it existed."""
        ...

    @abstractmethod
    def list_all(self) -> List[Point]:
        """List all stored Points."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all Points."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Number of stored Points."""
        ...


class FunctionType(ABC):
    """Base class for custom Point function types.

    Subclass this to add new generator types beyond the built-in
    cluster/linear/polynomial/periodic/raw.
    """

    type_name: str = "custom"
    type_code: bytes = b"CUST"  # 4-byte binary header

    @abstractmethod
    def generate(self, params: dict, n: int) -> np.ndarray:
        """Generate n values from this type's params."""
        ...

    @abstractmethod
    def nbytes(self, params: dict) -> int:
        """Estimate stored bytes for params."""
        ...

    @abstractmethod
    def to_bytes(self, params: dict, residual: Optional[np.ndarray] = None) -> bytes:
        """Serialize params + residual to bytes."""
        ...

    @abstractmethod
    def from_bytes(self, data: bytes) -> Tuple[dict, Optional[np.ndarray]]:
        """Deserialize bytes to (params, residual)."""
        ...

    @abstractmethod
    def to_dict(self, params: dict) -> dict:
        """Serialize params to JSON-compatible dict."""
        ...

    @abstractmethod
    def from_dict(self, d: dict) -> dict:
        """Deserialize dict to params."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Registries
# ══════════════════════════════════════════════════════════════════════════════

class _CompressorRegistry:
    """Registry of compression strategies."""

    def __init__(self):
        self._strategies: Dict[str, CompressionStrategy] = {}

    def register(self, strategy: CompressionStrategy) -> None:
        """Register a compression strategy."""
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Optional[CompressionStrategy]:
        """Get a strategy by name."""
        return self._strategies.get(name)

    def list(self) -> List[str]:
        """List registered strategy names."""
        return list(self._strategies.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._strategies


class _StorageRegistry:
    """Registry of storage backends."""

    def __init__(self):
        self._backends: Dict[str, StorageBackend] = {}

    def register(self, backend: StorageBackend) -> None:
        """Register a storage backend."""
        self._backends[backend.name] = backend

    def get(self, name: str) -> Optional[StorageBackend]:
        """Get a backend by name."""
        return self._backends.get(name)

    def list(self) -> List[str]:
        """List registered backend names."""
        return list(self._backends.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._backends


class _FunctionTypeRegistry:
    """Registry of custom function types."""

    def __init__(self):
        self._types: Dict[str, FunctionType] = {}
        self._code_map: Dict[bytes, str] = {}

    def register(self, ft: FunctionType) -> None:
        """Register a function type."""
        self._types[ft.type_name] = ft
        self._code_map[ft.type_code] = ft.type_name

    def get(self, name: str) -> Optional[FunctionType]:
        """Get a function type by name."""
        return self._types.get(name)

    def get_by_code(self, code: bytes) -> Optional[FunctionType]:
        """Get a function type by 4-byte code."""
        type_name = self._code_map.get(code)
        return self._types.get(type_name) if type_name else None

    def list(self) -> List[str]:
        """List registered type names."""
        return list(self._types.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._types


@dataclass
class Registry:
    """Global registry for all pluggable components."""
    compressors: _CompressorRegistry = field(default_factory=_CompressorRegistry)
    storages: _StorageRegistry = field(default_factory=_StorageRegistry)
    function_types: _FunctionTypeRegistry = field(default_factory=_FunctionTypeRegistry)


# Global singleton
registry = Registry()


# ══════════════════════════════════════════════════════════════════════════════
# Built-in strategies (registered on import)
# ══════════════════════════════════════════════════════════════════════════════

class ClusterStrategy(CompressionStrategy):
    """Vector quantization with Lloyd's refinement."""

    name = "cluster"

    def __init__(self, n_clusters: int = 16, lloyd_iterations: int = 5,
                 gap_fill_iterations: int = 4, gap_fill_max_elements: int = 100_000):
        self.n_clusters = n_clusters
        self.lloyd_iterations = lloyd_iterations
        self.gap_fill_iterations = gap_fill_iterations
        self.gap_fill_max_elements = gap_fill_max_elements

    def compress(self, data: np.ndarray, identity: str = "unknown",
                 n_clusters: Optional[int] = None, **kwargs) -> Point:
        nc = n_clusters or self.n_clusters
        flat = data.flatten().astype(np.float32)
        n = len(flat)

        quantiles = np.linspace(0, 100, nc + 2)[1:-1]
        centroids = np.percentile(flat, quantiles)
        centroids.sort()

        if n < self.gap_fill_max_elements:
            for _ in range(self.gap_fill_iterations):
                gaps = np.diff(centroids)
                biggest = np.argmax(gaps)
                new_c = (centroids[biggest] + centroids[biggest + 1]) / 2
                centroids = np.sort(np.append(centroids, new_c))

        nc = len(centroids)
        for _ in range(self.lloyd_iterations):
            assignments = np.clip(np.searchsorted(centroids, flat), 0, nc - 1).astype(np.uint8)
            sums = np.bincount(assignments, weights=flat, minlength=nc)
            counts = np.bincount(assignments, minlength=nc).astype(np.float64)
            alive = counts > 0
            centroids[alive] = (sums[alive] / counts[alive]).astype(np.float32)

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

    def decompress(self, point: Point, n: int) -> np.ndarray:
        centroids = point.params["centroids"]
        assignments = point.params["assignments"]
        return centroids[assignments[:n]]


class FunctionStrategy(CompressionStrategy):
    """Function fitting (periodic, linear, polynomial)."""

    name = "function"

    def __init__(self, residual_threshold: float = 0.99):
        self.residual_threshold = residual_threshold

    def compress(self, data: np.ndarray, identity: str = "unknown", **kwargs) -> Point:
        flat = data.flatten().astype(np.float32)
        n = len(flat)
        var = np.var(flat)

        fits = [
            ("periodic", self._fit_periodic(flat)),
            ("linear", self._fit_linear(flat)),
            ("polynomial", self._fit_polynomial(flat)),
        ]

        best_type, (params, mse) = min(fits, key=lambda x: x[1][1])
        accuracy = 1.0 - mse / (var + 1e-8)

        residual = None
        if accuracy < self.residual_threshold:
            i = np.arange(n, dtype=np.float32)
            if best_type == "periodic":
                fitted = params["a"] * np.cos(i) + params["b"] * np.sin(i) + params["w"]
            elif best_type == "linear":
                fitted = params["a"] * i + params["b"]
            else:
                fitted = params["a"] * i**2 + params["b"] * i + params["c"]
            residual = flat - fitted

        return Point(
            identity=identity,
            function_type=best_type,
            params=params,
            residual=residual,
            accuracy=float(accuracy),
        )

    def decompress(self, point: Point, n: int) -> np.ndarray:
        return point.generate(n)

    def _fit_periodic(self, flat):
        n = len(flat)
        i = np.arange(n, dtype=np.float32)
        A = np.column_stack([np.cos(i), np.sin(i), np.ones(n)])
        result, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
        a, b, w = result
        fitted = a * np.cos(i) + b * np.sin(i) + w
        return {"a": float(a), "b": float(b), "w": float(w)}, float(np.mean((flat - fitted) ** 2))

    def _fit_linear(self, flat):
        n = len(flat)
        i = np.arange(n, dtype=np.float32)
        A = np.column_stack([i, np.ones(n)])
        result, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
        a, b = result
        fitted = a * i + b
        return {"a": float(a), "b": float(b)}, float(np.mean((flat - fitted) ** 2))

    def _fit_polynomial(self, flat):
        n = len(flat)
        i = np.arange(n, dtype=np.float32)
        A = np.column_stack([i**2, i, np.ones(n)])
        result, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
        a, b, c = result
        fitted = a * i**2 + b * i + c
        return {"a": float(a), "b": float(b), "c": float(c)}, float(np.mean((flat - fitted) ** 2))


class RawStrategy(CompressionStrategy):
    """No compression — store raw bytes."""

    name = "raw"

    def compress(self, data: np.ndarray, identity: str = "unknown", **kwargs) -> Point:
        return Point(
            identity=identity,
            function_type="raw",
            params={
                "data_b64": base64.b64encode(data.tobytes()).decode(),
                "shape": list(data.shape),
                "dtype": str(data.dtype),
            },
            accuracy=1.0,
        )

    def decompress(self, point: Point, n: int) -> np.ndarray:
        raw_bytes = base64.b64decode(point.params["data_b64"])
        return np.frombuffer(raw_bytes, dtype=point.params["dtype"])


class AutoStrategy(CompressionStrategy):
    """Auto-select best strategy per weight."""

    name = "auto"

    def __init__(self):
        self._cluster = ClusterStrategy()
        self._function = FunctionStrategy()

    def compress(self, data: np.ndarray, identity: str = "unknown",
                 n_clusters: int = 16, **kwargs) -> Point:
        flat = data.flatten().astype(np.float32)

        if len(flat) < n_clusters * 2:
            return RawStrategy().compress(data, identity)

        cluster = self._cluster.compress(data, identity, n_clusters=n_clusters)
        func = self._function.compress(data, identity)

        if func.accuracy > cluster.accuracy:
            return func
        return cluster

    def decompress(self, point: Point, n: int) -> np.ndarray:
        return point.generate(n)


# Built-in storage backends
class MemoryStorage(StorageBackend):
    """In-memory storage backend."""

    name = "memory"

    def __init__(self):
        self._store = MemoryStore()

    def save(self, point: Point) -> None:
        self._store.save(point)

    def load(self, identity: str) -> Optional[Point]:
        return self._store.load(identity)

    def remove(self, identity: str) -> bool:
        return self._store.remove(identity)

    def list_all(self) -> List[Point]:
        return self._store.list_all()

    def clear(self) -> None:
        self._store.clear()

    def count(self) -> int:
        return self._store.count()


class JSONStorage(StorageBackend):
    """JSON file storage backend."""

    name = "json"

    def __init__(self, path: Union[Path, str]):
        self._store = JSONStore(Path(path))

    def save(self, point: Point) -> None:
        self._store.save(point)

    def load(self, identity: str) -> Optional[Point]:
        return self._store.load(identity)

    def remove(self, identity: str) -> bool:
        return self._store.remove(identity)

    def list_all(self) -> List[Point]:
        return self._store.list_all()

    def clear(self) -> None:
        self._store.clear()

    def count(self) -> int:
        return self._store.count()


class DirectoryStorage(StorageBackend):
    """Directory storage backend (one file per Point)."""

    name = "directory"

    def __init__(self, directory: Union[Path, str]):
        self._store = DirectoryStore(Path(directory))

    def save(self, point: Point) -> None:
        self._store.save(point)

    def load(self, identity: str) -> Optional[Point]:
        return self._store.load(identity)

    def remove(self, identity: str) -> bool:
        return self._store.remove(identity)

    def list_all(self) -> List[Point]:
        return self._store.list_all()

    def clear(self) -> None:
        self._store.clear()

    def count(self) -> int:
        return self._store.count()


# ══════════════════════════════════════════════════════════════════════════════
# Generic facade
# ══════════════════════════════════════════════════════════════════════════════

class PGQGeneric:
    """Generic, pluggable pugqeep system.

    Compose any compression strategy + storage backend + custom types.

    Args:
        name: System identifier.
        compressor: CompressionStrategy instance or registered name.
        storage: StorageBackend instance or registered name.
        function_types: List of FunctionType instances to register.
    """

    def __init__(self, name: str = "generic",
                 compressor: Union[str, CompressionStrategy] = "auto",
                 storage: Union[str, StorageBackend] = "memory",
                 function_types: Optional[List[FunctionType]] = None):
        self.name = name

        # Resolve compressor
        if isinstance(compressor, str):
            self._compressor = registry.compressors.get(compressor)
            if self._compressor is None:
                raise ValueError(
                    f"Unknown compressor '{compressor}'. "
                    f"Available: {registry.compressors.list()}"
                )
        else:
            self._compressor = compressor

        # Resolve storage
        if isinstance(storage, str):
            self._storage = registry.storages.get(storage)
            if self._storage is None:
                raise ValueError(
                    f"Unknown storage '{storage}'. "
                    f"Available: {registry.storages.list()}"
                )
        else:
            self._storage = storage

        # Register custom function types
        self._custom_types: Dict[str, FunctionType] = {}
        if function_types:
            for ft in function_types:
                self._custom_types[ft.type_name] = ft
                registry.function_types.register(ft)

        # Metadata
        self._shapes: Dict[str, Tuple[int, ...]] = {}
        self._dtypes: Dict[str, np.dtype] = {}

    # ── Core operations ──

    def put(self, name: str, data: np.ndarray, **kwargs) -> Point:
        """Compress and store data.

        Args:
            name: Data identifier.
            data: Numpy array to compress.
            **kwargs: Forwarded to compressor.compress().

        Returns:
            The compressed Point.
        """
        point = self._compressor.compress(data, name, **kwargs)
        self._storage.save(point)
        self._shapes[name] = data.shape
        self._dtypes[name] = data.dtype
        return point

    def get(self, name: str) -> Optional[np.ndarray]:
        """Load and decompress data.

        Returns:
            Numpy array or None if not found.
        """
        point = self._storage.load(name)
        if point is None:
            return None

        # Check custom types first
        custom = self._custom_types.get(point.function_type)
        if custom:
            n = int(np.prod(self._shapes[name])) if name in self._shapes else len(point.params.get("centroids", [])) * 100
            flat = custom.generate(point.params, n)
        else:
            flat = point.generate(self._estimate_n(name, point))

        shape = self._shapes.get(name)
        dtype = self._dtypes.get(name, np.float32)
        if shape:
            flat = flat.reshape(shape)
        return flat.astype(dtype)

    def has(self, name: str) -> bool:
        """Check if data exists."""
        return self._storage.load(name) is not None

    def remove(self, name: str) -> bool:
        """Remove data."""
        removed = self._storage.remove(name)
        self._shapes.pop(name, None)
        self._dtypes.pop(name, None)
        return removed

    def list_all(self) -> List[Point]:
        """List all stored Points."""
        return self._storage.list_all()

    def count(self) -> int:
        """Number of stored Points."""
        return self._storage.count()

    def clear(self) -> None:
        """Remove all data."""
        self._storage.clear()
        self._shapes.clear()
        self._dtypes.clear()

    # ── Batch operations ──

    def put_many(self, data: Dict[str, np.ndarray], **kwargs) -> dict:
        """Store multiple arrays."""
        total_bytes = 0
        for name, arr in data.items():
            self.put(name, arr, **kwargs)
            total_bytes += arr.nbytes
        return {"count": len(data), "total_bytes": total_bytes}

    def get_many(self, names: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Load multiple arrays."""
        return {name: self.get(name) for name in names}

    # ── Search ──

    def search(self, query: str) -> List[Point]:
        """Search Points by identity substring."""
        return [p for p in self._storage.list_all() if query in p.identity]

    def best(self, n: int = 10) -> List[Point]:
        """Top-N Points by accuracy."""
        return sorted(self._storage.list_all(), key=lambda p: p.accuracy, reverse=True)[:n]

    # ── Stats ──

    def stats(self) -> dict:
        """System statistics."""
        points = self._storage.list_all()
        total_raw = sum(
            int(np.prod(self._shapes[p.identity])) * 4
            if p.identity in self._shapes else p.nbytes()
            for p in points
        )
        total_compressed = sum(p.nbytes() for p in points)
        accuracies = [p.accuracy for p in points if p.accuracy > 0]

        return {
            "name": self.name,
            "compressor": self._compressor.name,
            "storage": self._storage.name,
            "num_points": len(points),
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "ratio": total_raw / max(total_compressed, 1),
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "custom_types": list(self._custom_types.keys()),
        }

    # ── Internal ──

    def _estimate_n(self, name: str, point: Point) -> int:
        shape = self._shapes.get(name)
        if shape:
            return int(np.prod(shape))
        if "centroids" in point.params:
            return len(point.params["centroids"]) * 100
        return 1000


# ══════════════════════════════════════════════════════════════════════════════
# Register built-ins on import
# ══════════════════════════════════════════════════════════════════════════════

registry.compressors.register(ClusterStrategy())
registry.compressors.register(FunctionStrategy())
registry.compressors.register(RawStrategy())
registry.compressors.register(AutoStrategy())

registry.storages.register(MemoryStorage())
# JSON and Directory need paths — registered lazily via factory
