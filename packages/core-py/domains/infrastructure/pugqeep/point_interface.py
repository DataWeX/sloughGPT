"""
Point protocol and interface — the contract for compressed data.

Defines:
  - PointProtocol: interface that any Point implementation must satisfy
  - PointView: lazy decompression wrapper (holds metadata, generates numpy on-demand)
  - FunctionType: enum of supported compression types

The protocol enables:
  - Swappable compression backends (cluster, function, raw, future strategies)
  - Lazy evaluation (PointView defers decompression until .generate())
  - Type-safe function_type matching
  - Consistent serialization across implementations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np


class FunctionType(StrEnum):
    """Supported compression function types."""

    PERIODIC = "periodic"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    CLUSTER = "cluster"
    RAW = "raw"

    @classmethod
    def from_str(cls, s: str) -> FunctionType:
        try:
            return cls(s)
        except ValueError:
            raise ValueError(
                f"Unknown function type: {s!r}. Must be one of: {[ft.value for ft in cls]}"
            ) from None


class PointProtocol(ABC):
    """Abstract interface for a compressed data point.

    Required attributes (set by __init__ or as dataclass fields):
      - identity: str — unique identifier
      - function_type: str | FunctionType — compression method
      - params: dict — function parameters
      - accuracy: float — compression accuracy (0-1)
      - residual: Optional[np.ndarray] — residual array
      - dtype: str — original data dtype
      - shape: tuple — original data shape

    Required methods:
      - generate(n) -> np.ndarray
      - nbytes() -> int
      - to_dict() -> dict
      - to_bytes() -> bytes
      - from_dict(d) -> PointProtocol (classmethod)
      - from_bytes(data, identity) -> PointProtocol (classmethod)
    """

    # Validate that subclasses have the required attributes
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip validation for abstract subclasses
        if getattr(cls, "__abstractmethods__", None):
            return
        # For concrete subclasses, just ensure the class is usable
        # (full validation happens at add-time in PointLibrary)

    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """Generate n values from this point's stored function.

        For cluster points: returns centroids[assignments[:n]]
        For function points: evaluates the fitted function
        For raw points: returns the stored data
        """
        raise NotImplementedError

    @abstractmethod
    def nbytes(self) -> int:
        """Memory footprint of stored parameters (bytes)."""
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> PointProtocol:
        """Deserialize from dict."""
        raise NotImplementedError

    @abstractmethod
    def to_bytes(self) -> bytes:
        """Serialize to compact binary format."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes, identity: str = "unknown") -> PointProtocol:
        """Deserialize from binary format."""
        raise NotImplementedError

    # ── Derived properties ───────────────────────────────────────────

    @property
    def is_lossless(self) -> bool:
        """Whether this point is lossless (accuracy == 1.0)."""
        return self.accuracy >= 1.0

    @property
    def compression_ratio(self) -> float:
        """Estimated compression ratio (raw_bytes / compressed_bytes)."""
        raw = self._estimate_raw_bytes()
        comp = self.nbytes()
        if raw == 0:
            return 1.0  # unknown compression ratio
        return raw / max(comp, 1)

    def _estimate_raw_bytes(self) -> int:
        """Estimate uncompressed size (override in subclass)."""
        return self.nbytes()

    def __repr__(self) -> str:
        ft = (
            self.function_type.value
            if isinstance(self.function_type, FunctionType)
            else self.function_type
        )
        return (
            f"Point(id={self.identity!r}, type={ft}, "
            f"accuracy={self.accuracy:.4f}, nbytes={self.nbytes()})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointProtocol):
            return NotImplemented
        return self.identity == other.identity and self.function_type == other.function_type

    def __hash__(self) -> int:
        return hash((self.identity, str(self.function_type)))


class PointView:
    """Lazy decompression wrapper — holds metadata, generates numpy on-demand.

    PointView is a lightweight handle that stores the compressed Point
    plus shape/dtype info. It defers decompression until ``generate()``
    or ``__array__()`` is called, avoiding unnecessary work.

    Usage::

        view = PointView(point, shape=(768, 768), dtype="float32")
        # No numpy generated yet
        arr = view.generate()  # now decompresses
        arr = view[:]          # same as generate()
        arr = view[0:100]      # partial decompression (if supported)

    Args:
        point: The compressed Point.
        shape: Original data shape.
        dtype: Original data dtype.
    """

    __slots__ = ("_point", "_shape", "_dtype", "_cache")

    def __init__(self, point: PointProtocol, shape: tuple[int, ...] = (), dtype: str = "float32"):
        self._point = point
        self._shape = tuple(shape)
        self._dtype = dtype
        self._cache: np.ndarray | None = None

    @property
    def point(self) -> PointProtocol:
        """The underlying compressed Point."""
        return self._point

    @property
    def identity(self) -> str:
        return self._point.identity

    @property
    def function_type(self) -> str | FunctionType:
        return self._point.function_type

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._dtype)

    @property
    def accuracy(self) -> float:
        return self._point.accuracy

    @property
    def nbytes(self) -> int:
        return self._point.nbytes()

    def generate(self) -> np.ndarray:
        """Generate the full numpy array (cached after first call)."""
        if self._cache is not None:
            return self._cache

        n = int(np.prod(self._shape)) if self._shape else self._point.nbytes() // 4
        flat = self._point.generate(n)

        if self._shape:
            flat = flat.reshape(self._shape)
        self._cache = flat.astype(self._dtype)
        return self._cache

    def clear_cache(self) -> None:
        """Free the cached numpy array."""
        self._cache = None

    def __array__(self, dtype=None) -> np.ndarray:
        """numpy array protocol — enables np.array(view)."""
        arr = self.generate()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __getitem__(self, key) -> np.ndarray:
        """Slice access — lazily decompresses only the requested portion.

        Fast paths avoid materializing the full array for both cluster and
        analytic-fit (linear/periodic/polynomial) points; `raw` and strided
        slices on complex shapes fall back to the cache-aware path.
        """
        if isinstance(key, slice) and self._cache is None:
            if self._point.function_type == "cluster":
                return self._slice_cluster(key)
            if self._point.function_type in ("linear", "periodic", "polynomial"):
                return self._slice_analytic(key)

        # Fallback: full decompression then slice
        return self.generate()[key]

    def _slice_cluster(self, key: slice) -> np.ndarray:
        """Cluster fast path — look up only the requested centroids."""
        centroids = self._point.params.get("centroids")
        assignments = self._point.params.get("assignments")
        if centroids is None or assignments is None:
            return self.generate()[key]
        indices = range(*key.indices(len(assignments)))
        values = centroids[assignments[list(indices)]]
        if self._point.residual is not None:
            values = values + self._point.residual[list(indices)]
        return values.astype(self._dtype)

    def _slice_analytic(self, key: slice) -> np.ndarray:
        """Analytic fast path — evaluate the stored function over slice indices only."""
        if not self._shape:
            return self.generate()[key]
        n_total = int(np.prod(self._shape))
        start, stop, step = key.indices(n_total)
        indices = np.arange(start, stop, step, dtype=np.float32)
        if indices.size == 0:
            return np.empty((0,), dtype=self._dtype)

        p = self._point
        if p.function_type == "periodic":
            a, b, w = p.params["a"], p.params["b"], p.params["w"]
            values = a * np.cos(indices) + b * np.sin(indices) + w
        elif p.function_type == "linear":
            a, b = p.params["a"], p.params["b"]
            values = a * indices + b
        else:  # polynomial
            a, b, c = p.params["a"], p.params["b"], p.params["c"]
            values = a * indices**2 + b * indices + c

        if p.residual is not None:
            values = values + p.residual[start:stop:step]
        return values.astype(self._dtype)

    def __len__(self) -> int:
        return int(np.prod(self._shape)) if self._shape else 0

    def __repr__(self) -> str:
        cached = "cached" if self._cache is not None else "lazy"
        return (
            f"PointView(id={self.identity!r}, shape={self._shape}, dtype={self._dtype}, {cached})"
        )

    @classmethod
    def from_point_and_meta(
        cls, point: PointProtocol, shape: tuple[int, ...] = (), dtype: str = "float32"
    ) -> PointView:
        """Create a PointView from a Point and metadata."""
        return cls(point, shape=shape, dtype=dtype)
