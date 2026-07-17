"""
PointLibrary — stores, indexes, and retrieves Points.

This is the "Graph" in the Point-Graph-Queue architecture:
  - Points are organized by identity and function type
  - Provides lookup, add, remove, and search
  - Persists to disk as JSON with base64-encoded numpy arrays
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .point import Point
from .compressor import PointCompressor
from .config import LibraryConfig

logger = logging.getLogger("slo.pdqeep")


class PointLibrary:
    """Stores, indexes, and retrieves Points.

    Args:
        name: Library identifier.
        storage_dir: Directory for persistence.
        config: Optional LibraryConfig (overrides name, storage_dir, auto_save).
    """

    def __init__(self, name: str = "default", storage_dir: Optional[Path] = None,
                 config: Optional[LibraryConfig] = None):
        if config is not None:
            self.name = config.name
            self._storage_dir = config.storage_dir or storage_dir
            self._auto_save = config.auto_save
        else:
            self.name = name
            self._storage_dir = storage_dir
            self._auto_save = False
        self._points: Dict[str, Point] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._created_at = time.time()
        self._compressor = PointCompressor()

    # ── CRUD ──

    def add(self, point: Point) -> None:
        self._points[point.identity] = point
        by_type = self._by_type.setdefault(point.function_type, [])
        if point.identity not in by_type:
            by_type.append(point.identity)
        if self._auto_save and self._storage_dir is not None:
            self.save()

    def get(self, identity: str) -> Optional[Point]:
        return self._points.get(identity)

    def remove(self, identity: str) -> bool:
        point = self._points.pop(identity, None)
        if point is None:
            return False
        by_type = self._by_type.get(point.function_type, [])
        if identity in by_type:
            by_type.remove(identity)
        if self._auto_save and self._storage_dir is not None:
            self.save()
        return True

    def list_all(self) -> List[Point]:
        return list(self._points.values())

    def list_by_type(self, function_type: str) -> List[Point]:
        identities = self._by_type.get(function_type, [])
        return [self._points[i] for i in identities if i in self._points]

    def has(self, identity: str) -> bool:
        return identity in self._points

    def clear(self) -> None:
        self._points.clear()
        self._by_type.clear()

    # ── Compress & store ──

    def compress_and_store(self, weights: np.ndarray, identity: str,
                           method: str = "cluster", n_clusters: int = 16) -> Point:
        if method == "cluster":
            point = self._compressor.compress_cluster(weights, identity, n_clusters)
        else:
            point = self._compressor.compress_function(weights, identity)
        self.add(point)
        return point

    def decompress_to(self, identity: str, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        point = self.get(identity)
        if point is None:
            return None
        if point.function_type == "cluster":
            centroids = point.params["centroids"]
            assignments = point.params["assignments"]
            result = centroids[assignments]
        else:
            result = point.generate(len(point.params) * 100)
        if shape is not None:
            result = result.reshape(shape)
        return result

    # ── Search ──

    def search(self, query: str) -> List[Point]:
        q = query.lower()
        return [p for p in self._points.values() if q in p.identity.lower()]

    def best_points(self, n: int = 10) -> List[Point]:
        return sorted(self._points.values(), key=lambda p: p.accuracy, reverse=True)[:n]

    # ── Statistics ──

    def stats(self) -> dict:
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
                total_raw += len(assignments) * 4
            else:
                total_compressed += 4 + len(p.params) * 4
                total_raw += len(p.params) * 100
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
        return path

    @classmethod
    def load(cls, path: Path) -> "PointLibrary":
        data = json.loads(path.read_text())
        lib = cls(
            name=data.get("name", path.stem),
            storage_dir=path.parent,
        )
        lib._created_at = data.get("created_at", 0)
        for pd in data.get("points", []):
            lib.add(Point.from_dict(pd))
        return lib
