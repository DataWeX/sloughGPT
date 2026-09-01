"""
Function stores — persistence layer for Points.

Provides different storage backends:
  - JSONStore: JSON file persistence (default)
  - MemoryStore: in-memory only
  - DirectoryStore: one file per point
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .point import Point

logger = logging.getLogger("slo.pugqeep")


class Store(Protocol):
    """Protocol for function stores."""
    def save(self, point: Point) -> None: ...
    def load(self, identity: str) -> Optional[Point]: ...
    def remove(self, identity: str) -> bool: ...
    def list_all(self) -> List[Point]: ...
    def clear(self) -> None: ...
    def count(self) -> int: ...


class MemoryStore:
    """In-memory store. Fast, no persistence."""

    def __init__(self):
        self._points: Dict[str, Point] = {}

    def save(self, point: Point) -> None:
        self._points[point.identity] = point

    def load(self, identity: str) -> Optional[Point]:
        return self._points.get(identity)

    def remove(self, identity: str) -> bool:
        return self._points.pop(identity, None) is not None

    def list_all(self) -> List[Point]:
        return list(self._points.values())

    def clear(self) -> None:
        self._points.clear()

    def count(self) -> int:
        return len(self._points)


class JSONStore:
    """JSON file store. Single file, atomic writes."""

    def __init__(self, path: Path):
        self._path = path
        self._points: Dict[str, Point] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            for pd in data.get("points", []):
                p = Point.from_dict(pd)
                self._points[p.identity] = p

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "points": [p.to_dict() for p in self._points.values()],
            "saved_at": time.time(),
        }
        self._path.write_text(json.dumps(data, indent=2))

    def save(self, point: Point) -> None:
        self._points[point.identity] = point
        self._save()

    def load(self, identity: str) -> Optional[Point]:
        return self._points.get(identity)

    def remove(self, identity: str) -> bool:
        if self._points.pop(identity, None) is not None:
            self._save()
            return True
        return False

    def list_all(self) -> List[Point]:
        return list(self._points.values())

    def clear(self) -> None:
        self._points.clear()
        self._save()

    def count(self) -> int:
        return len(self._points)


class DirectoryStore:
    """One file per point. Good for large clusters."""

    def __init__(self, directory: Path):
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)

    def _point_path(self, identity: str) -> Path:
        safe = identity.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.point.json"

    def save(self, point: Point) -> None:
        path = self._point_path(point.identity)
        path.write_text(json.dumps(point.to_dict(), indent=2))

    def load(self, identity: str) -> Optional[Point]:
        path = self._point_path(identity)
        if not path.exists():
            return None
        return Point.from_dict(json.loads(path.read_text()))

    def remove(self, identity: str) -> bool:
        path = self._point_path(identity)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self) -> List[Point]:
        points = []
        for f in sorted(self._dir.glob("*.point.json")):
            points.append(Point.from_dict(json.loads(f.read_text())))
        return points

    def clear(self) -> None:
        for f in self._dir.glob("*.point.json"):
            f.unlink()

    def count(self) -> int:
        return len(list(self._dir.glob("*.point.json")))
