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
        analytic-fit (linear/periodic/polynomial) points, including multi-D
        shapes and quantized centroids. `raw`, huffman-decoded clusters, and
        exotic slices (negative steps over leading axes) fall back to the
        cache-aware path.
        """
        if isinstance(key, slice) and self._cache is None:
            fast = self._try_fast_slice(key)
            if fast is not None:
                return fast

        # Fallback: full decompression then slice
        return self.generate()[key]

    def _try_fast_slice(self, key: slice) -> np.ndarray | None:
        """Compute a lazy slice, or None if the fast path cannot handle `key`."""
        ft = self._point.function_type
        if ft not in ("cluster", "linear", "periodic", "polynomial"):
            return None  # raw / unknown
        if not self._shape:
            return None
        if ft == "cluster" and self._point.params.get("huffman_data") is not None:
            return None  # huffman-cluster slices not lazy-safe

        if len(self._shape) == 1:
            n = self._shape[0]
            start, stop, step = key.indices(n)
            pos = np.arange(start, stop, step, dtype=np.int64)
            out_shape = tuple(pos.shape)
        else:
            # Leading-axis slice with full trailing axes (view[i:j] or view[i:j:k]).
            # This is the common weight-matrix case and stays O(slice).
            start, stop, step = key.indices(self._shape[0])
            if step < 0:
                return None  # reversed multi-D slice: fall back
            rows = np.arange(start, stop, step, dtype=np.int64)
            inner = int(np.prod(self._shape[1:]))
            pos = (rows[:, None] * inner + np.arange(inner, dtype=np.int64)[None, :]).ravel()
            out_shape = (rows.size,) + tuple(self._shape[1:])

        if pos.size == 0:
            return np.empty(out_shape, dtype=self._dtype)

        values = self._eval_at(pos)
        return values.reshape(out_shape).astype(self._dtype)

    def _eval_at(self, pos: np.ndarray) -> np.ndarray:
        """Evaluate the point's stored function at flat index positions `pos`.

        Matches ``Point.generate`` semantics exactly (including quantized
        centroids and residuals) so lazy slices equal ``view.generate()[key]``.
        """
        p = self._point
        if p.function_type == "cluster":
            centroids = p.params["centroids"]
            if p.params.get("centroid_quantized"):
                scale = p.params["centroid_scale"]
                zp = p.params["centroid_zero_point"]
                centroids = (centroids.astype(np.float32) - zp) * scale
            assignments = p.params["assignments"]
            values = centroids[assignments[pos]]
            if p.residual is not None:
                values = values + p.residual[pos]
            return values

        i = pos.astype(np.float32)
        if p.function_type == "periodic":
            a, b, w = p.params["a"], p.params["b"], p.params["w"]
            values = a * np.cos(i) + b * np.sin(i) + w
        elif p.function_type == "linear":
            a, b = p.params["a"], p.params["b"]
            values = a * i + b
        else:  # polynomial
            a, b, c = p.params["a"], p.params["b"], p.params["c"]
            values = a * i**2 + b * i + c

        if p.residual is not None:
            values = values + p.residual[pos]
        return values

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
