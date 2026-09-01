"""
PointLibrary — stores, indexes, and retrieves Points.

This is the "Graph" in the Point-Graph-Queue architecture:
  - Points are organized by identity and function type
  - Thread-safe CRUD with proper locking
  - Batch operations for bulk add/get/remove
  - Iterator protocol for streaming access
  - PointView for lazy decompression
  - Validation on add
  - Statistics and introspection
  - Persistence to disk as JSON with base64-encoded numpy arrays
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .point import Point
from .point_interface import PointProtocol, PointView, FunctionType
from .compressor import PointCompressor
from .config import LibraryConfig

logger = logging.getLogger("slo.pugqeep")


class PointLibrary:
    """Thread-safe store for Points with indexing, batch ops, and lazy views.

    Args:
        name: Library identifier.
        storage_dir: Directory for persistence.
        config: Optional LibraryConfig (overrides name, storage_dir, auto_save).
        validate: Whether to validate points on add (default True).
    """

    def __init__(self, name: str = "default", storage_dir: Optional[Path] = None,
                 config: Optional[LibraryConfig] = None, validate: bool = True):
        if config is not None:
            self.name = config.name
            self._storage_dir = config.storage_dir or storage_dir
            self._auto_save = config.auto_save
        else:
            self.name = name
            self._storage_dir = storage_dir
            self._auto_save = False
        self._validate = validate
        self._points: Dict[str, Point] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._views: Dict[str, PointView] = {}
        self._created_at = time.time()
        self._compressor = PointCompressor()
        self._lock = threading.RLock()
        self._stats_adds = 0
        self._stats_removes = 0
        self._stats_hits = 0
        self._stats_misses = 0

    # ── Single-item CRUD ─────────────────────────────────────────────

    def add(self, point: PointProtocol) -> bool:
        """Add a point to the library (thread-safe).

        Args:
            point: Point or PointProtocol implementation to store.

        Returns:
            True if added (new), False if replaced (existing identity).

        Raises:
            ValueError: If validation is enabled and the point is invalid.
        """
        if self._validate:
            self._validate_point(point)

        # Coerce to Point if needed
        if not isinstance(point, Point):
            point = Point(
                identity=point.identity,
                function_type=point.function_type.value if isinstance(point.function_type, FunctionType)
                             else point.function_type,
                params=dict(point.params),
                residual=point.residual,
                accuracy=point.accuracy,
                dtype=point.dtype,
                shape=point.shape,
            )

        with self._lock:
            is_new = point.identity not in self._points
            self._points[point.identity] = point
            by_type = self._by_type.setdefault(
                point.function_type if isinstance(point.function_type, str)
                else point.function_type.value, []
            )
            if point.identity not in by_type:
                by_type.append(point.identity)
            # Invalidate cached view
            self._views.pop(point.identity, None)
            self._stats_adds += 1

        if self._auto_save and self._storage_dir is not None:
            self.save()

        return is_new

    def get(self, identity: str) -> Optional[Point]:
        """Get a point by identity (thread-safe)."""
        with self._lock:
            point = self._points.get(identity)
            if point is not None:
                self._stats_hits += 1
            else:
                self._stats_misses += 1
            return point

    def remove(self, identity: str) -> bool:
        """Remove a point by identity (thread-safe). Returns True if removed."""
        with self._lock:
            point = self._points.pop(identity, None)
            if point is None:
                return False
            ft = point.function_type if isinstance(point.function_type, str) else point.function_type.value
            by_type = self._by_type.get(ft, [])
            if identity in by_type:
                by_type.remove(identity)
            self._views.pop(identity, None)
            self._stats_removes += 1

        if self._auto_save and self._storage_dir is not None:
            self.save()
        return True

    def has(self, identity: str) -> bool:
        """Check if a point exists."""
        with self._lock:
            return identity in self._points

    def clear(self) -> None:
        """Remove all points."""
        with self._lock:
            self._points.clear()
            self._by_type.clear()
            self._views.clear()

    def __contains__(self, identity: str) -> bool:
        return self.has(identity)

    def __len__(self) -> int:
        with self._lock:
            return len(self._points)

    # ── Batch operations ─────────────────────────────────────────────

    def add_many(self, points: List[PointProtocol]) -> int:
        """Add multiple points (thread-safe, single lock acquisition).

        Returns:
            Number of points added.
        """
        count = 0
        with self._lock:
            for point in points:
                if self._validate:
                    self._validate_point(point)
                if not isinstance(point, Point):
                    point = Point(
                        identity=point.identity,
                        function_type=point.function_type.value if isinstance(point.function_type, FunctionType)
                                     else point.function_type,
                        params=dict(point.params),
                        residual=point.residual,
                        accuracy=point.accuracy,
                        dtype=point.dtype,
                        shape=point.shape,
                    )
                self._points[point.identity] = point
                ft = point.function_type if isinstance(point.function_type, str) else point.function_type.value
                by_type = self._by_type.setdefault(ft, [])
                if point.identity not in by_type:
                    by_type.append(point.identity)
                self._views.pop(point.identity, None)
                self._stats_adds += 1
                count += 1

        if self._auto_save and self._storage_dir is not None:
            self.save()
        return count

    def get_many(self, identities: List[str]) -> Dict[str, Optional[Point]]:
        """Get multiple points by identity (thread-safe)."""
        with self._lock:
            result = {}
            for ident in identities:
                point = self._points.get(ident)
                if point is not None:
                    self._stats_hits += 1
                else:
                    self._stats_misses += 1
                result[ident] = point
            return result

    def remove_many(self, identities: List[str]) -> int:
        """Remove multiple points (thread-safe). Returns count removed."""
        count = 0
        with self._lock:
            for ident in identities:
                point = self._points.pop(ident, None)
                if point is not None:
                    ft = point.function_type if isinstance(point.function_type, str) else point.function_type.value
                    by_type = self._by_type.get(ft, [])
                    if ident in by_type:
                        by_type.remove(ident)
                    self._views.pop(ident, None)
                    self._stats_removes += 1
                    count += 1

        if self._auto_save and self._storage_dir is not None:
            self.save()
        return count

    def exists_many(self, identities: List[str]) -> Dict[str, bool]:
        """Check existence of multiple identities."""
        with self._lock:
            return {ident: ident in self._points for ident in identities}

    # ── Listing and iteration ────────────────────────────────────────

    def list_all(self) -> List[Point]:
        """Return all points (thread-safe copy)."""
        with self._lock:
            return list(self._points.values())

    def list_by_type(self, function_type: Union[str, FunctionType]) -> List[Point]:
        """Return all points of a given type."""
        ft = function_type.value if isinstance(function_type, FunctionType) else function_type
        with self._lock:
            identities = self._by_type.get(ft, [])
            return [self._points[i] for i in identities if i in self._points]

    def list_identities(self) -> List[str]:
        """Return all point identities."""
        with self._lock:
            return list(self._points.keys())

    def list_types(self) -> Dict[str, int]:
        """Return count of points per type."""
        with self._lock:
            return {ft: len(ids) for ft, ids in self._by_type.items()}

    def __iter__(self) -> Iterator[Point]:
        """Iterate over all points (snapshot)."""
        return iter(self.list_all())

    def iter_by_type(self, function_type: Union[str, FunctionType]) -> Iterator[Point]:
        """Iterate over points of a given type."""
        return iter(self.list_by_type(function_type))

    # ── PointView (lazy decompression) ───────────────────────────────

    def view(self, identity: str, shape: Tuple[int, ...] = (),
             dtype: str = "float32") -> Optional[PointView]:
        """Get a lazy PointView for deferred decompression.

        The view caches the numpy array after first generate() call.
        Call view.clear_cache() to free memory.

        Args:
            identity: Point identity.
            shape: Original data shape (needed for reshape).
            dtype: Original data dtype.

        Returns:
            PointView or None if point not found.
        """
        with self._lock:
            if identity in self._views:
                return self._views[identity]

            point = self._points.get(identity)
            if point is None:
                return None

            view = PointView(point, shape=shape, dtype=dtype)
            self._views[identity] = view
            return view

    def views(self, identities: List[str], shape: Tuple[int, ...] = (),
              dtype: str = "float32") -> Dict[str, Optional[PointView]]:
        """Get multiple PointViews at once."""
        return {ident: self.view(ident, shape=shape, dtype=dtype) for ident in identities}

    def clear_views(self) -> int:
        """Clear all cached PointViews. Returns count cleared."""
        with self._lock:
            count = len(self._views)
            self._views.clear()
            return count

    # ── Compress & store ─────────────────────────────────────────────

    def compress_and_store(self, weights: np.ndarray, identity: str,
                           method: str = "cluster", n_clusters: int = 16) -> Point:
        """Compress a numpy array and store the resulting Point."""
        if method == "cluster":
            point = self._compressor.compress_cluster(weights, identity, n_clusters)
        else:
            point = self._compressor.compress_function(weights, identity)
        self.add(point)
        return point

    def decompress_to(self, identity: str, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        """Decompress a point back to numpy (uses PointView internally)."""
        point = self.get(identity)
        if point is None:
            return None
        if point.function_type == "cluster":
            centroids = point.params["centroids"]
            assignments = point.params["assignments"]
            result = centroids[assignments]
        else:
            # Prefer explicit shape, then stored point.shape, else fallback
            if shape is not None:
                n = int(np.prod(shape))
            elif point.shape:
                n = int(np.prod(point.shape))
            else:
                n = 1000
            result = point.generate(n)
        if shape is not None:
            result = result.reshape(shape)
        return result

    # ── Search ───────────────────────────────────────────────────────

    def search(self, query: str) -> List[Point]:
        """Search points by identity substring (case-insensitive)."""
        q = query.lower()
        with self._lock:
            return [p for p in self._points.values() if q in p.identity.lower()]

    def search_by_type(self, function_type: Union[str, FunctionType],
                       query: str = "") -> List[Point]:
        """Search points by type and optional identity substring."""
        ft = function_type.value if isinstance(function_type, FunctionType) else function_type
        q = query.lower()
        with self._lock:
            identities = self._by_type.get(ft, [])
            if q:
                return [self._points[i] for i in identities
                        if i in self._points and q in i.lower()]
            return [self._points[i] for i in identities if i in self._points]

    def best_points(self, n: int = 10) -> List[Point]:
        """Get top-n points by accuracy."""
        with self._lock:
            return sorted(self._points.values(),
                         key=lambda p: p.accuracy, reverse=True)[:n]

    def worst_points(self, n: int = 10) -> List[Point]:
        """Get bottom-n points by accuracy."""
        with self._lock:
            return sorted(self._points.values(),
                         key=lambda p: p.accuracy)[:n]

    # ── Validation ───────────────────────────────────────────────────

    @staticmethod
    def _validate_point(point: PointProtocol) -> None:
        """Validate a point before adding.

        Raises:
            ValueError: If the point has invalid fields.
        """
        if not point.identity:
            raise ValueError("Point identity cannot be empty")

        ft = point.function_type
        if isinstance(ft, FunctionType):
            ft = ft.value
        valid_types = {e.value for e in FunctionType}
        if ft not in valid_types:
            raise ValueError(f"Invalid function_type: {ft!r}. Must be one of: {valid_types}")

        if not 0.0 <= point.accuracy <= 1.0:
            raise ValueError(f"Accuracy must be 0-1, got {point.accuracy}")

        if ft == "cluster":
            params = point.params
            if params is not None:
                if "centroids" not in params or "assignments" not in params:
                    raise ValueError("Cluster points must have 'centroids' and 'assignments' params")
                if not isinstance(params["centroids"], np.ndarray):
                    raise ValueError("centroids must be numpy array")
                if not isinstance(params["assignments"], np.ndarray):
                    raise ValueError("assignments must be numpy array")

    # ── Statistics ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """Comprehensive library statistics."""
        with self._lock:
            points = list(self._points.values())
            if not points:
                return {
                    "name": self.name,
                    "total_points": 0,
                    "total_raw_bytes": 0,
                    "total_compressed_bytes": 0,
                    "ratio": 0.0,
                    "avg_accuracy": 0.0,
                    "types": {},
                    "views_cached": 0,
                    "ops": {"adds": self._stats_adds, "removes": self._stats_removes,
                            "hits": self._stats_hits, "misses": self._stats_misses},
                }

            total_raw = 0
            total_compressed = 0
            for p in points:
                total_raw += p._estimate_raw_bytes()
                total_compressed += p.nbytes()

            return {
                "name": self.name,
                "total_points": len(points),
                "total_raw_bytes": total_raw,
                "total_compressed_bytes": total_compressed,
                "ratio": total_raw / max(total_compressed, 1),
                "avg_accuracy": sum(p.accuracy for p in points) / len(points),
                "types": {ft: len(ids) for ft, ids in self._by_type.items()},
                "views_cached": len(self._views),
                "ops": {"adds": self._stats_adds, "removes": self._stats_removes,
                        "hits": self._stats_hits, "misses": self._stats_misses},
            }

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        total = self._stats_hits + self._stats_misses
        return self._stats_hits / total if total > 0 else 0.0

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Save library to disk (thread-safe, atomic write)."""
        if path is None:
            if self._storage_dir is None:
                raise ValueError("No storage_dir set and no path provided")
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            path = self._storage_dir / f"{self.name}.points.json"

        with self._lock:
            data = {
                "name": self.name,
                "created_at": self._created_at,
                "saved_at": time.time(),
                "points": [p.to_dict() for p in self._points.values()],
            }
        # Atomic write: write to temp file then rename
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.rename(path)
        return path

    @classmethod
    def load(cls, path: Path, validate: bool = True) -> "PointLibrary":
        """Load library from disk. Handles corrupted JSON gracefully."""
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Corrupted library file %s: %s", path, e)
            return cls(name=path.stem, storage_dir=path.parent, validate=validate)
        lib = cls(
            name=data.get("name", path.stem),
            storage_dir=path.parent,
            validate=validate,
        )
        lib._created_at = data.get("created_at", 0)
        points = []
        for pd in data.get("points", []):
            try:
                points.append(Point.from_dict(pd))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("Skipping corrupted point in %s: %s", path, e)
        lib.add_many(points)
        return lib

    # ── Context manager ──────────────────────────────────────────────

    def __enter__(self) -> "PointLibrary":
        return self

    def __exit__(self, *args) -> None:
        if self._auto_save and self._storage_dir is not None:
            self.save()

    def __repr__(self) -> str:
        with self._lock:
            types = ", ".join(f"{ft}:{len(ids)}" for ft, ids in self._by_type.items() if ids)
            return f"PointLibrary(name={self.name!r}, points={len(self._points)}, types={types})"
