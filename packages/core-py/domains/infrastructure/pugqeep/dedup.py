"""
PointDeduplicator + PointLibrarySync.

PointDeduplicator: shares identical points across models.
PointLibrarySync: synchronizes PointLibraries between instances.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .point import Point
from .library import PointLibrary

logger = logging.getLogger("slo.pdqeep")


class PointDeduplicator:
    """Identifies and merges identical points across multiple libraries."""

    def __init__(self, tolerance: float = 1e-6):
        self._tolerance = tolerance
        self._libraries: List[PointLibrary] = []
        self._fingerprints: Dict[str, List[str]] = {}

    def add_library(self, library: PointLibrary) -> None:
        self._libraries.append(library)
        for point in library.list_all():
            fp = self._fingerprint(point)
            self._fingerprints.setdefault(fp, []).append(point.identity)

    def find_duplicates(self) -> List[List[str]]:
        groups = []
        for fp, identities in self._fingerprints.items():
            if len(identities) > 1:
                groups.append(identities)
        return groups

    def deduplicate(self) -> dict:
        groups = self.find_duplicates()
        merged = 0
        bytes_saved = 0

        for group in groups:
            keep = group[0]
            remove = group[1:]

            keep_point = None
            for lib in self._libraries:
                p = lib.get(keep)
                if p is not None:
                    keep_point = p
                    break

            if keep_point is None:
                continue

            for identity in remove:
                for lib in self._libraries:
                    point = lib.get(identity)
                    if point is not None:
                        bytes_saved += point.nbytes()
                        lib.remove(identity)
                        merged += 1

        return {
            "merged": merged,
            "bytes_saved": bytes_saved,
            "groups": len(groups),
        }

    def _fingerprint(self, point: Point) -> str:
        if point.function_type == "cluster":
            cents = point.params["centroids"]
            assns = point.params["assignments"]
            if self._tolerance > 0:
                # Quantize centroids to tolerance before hashing
                step = self._tolerance
                cents_q = np.round(cents / step) * step
                data = cents_q.tobytes() + assns.tobytes()
            else:
                data = cents.tobytes() + assns.tobytes()
        elif point.function_type == "raw":
            raw = base64.b64decode(point.params["data_b64"])
            if self._tolerance > 0:
                arr = np.frombuffer(raw, dtype=np.float32)
                step = self._tolerance
                arr_q = np.round(arr / step) * step
                data = arr_q.tobytes()
            else:
                data = raw
        else:
            params = {}
            for k, v in point.params.items():
                if isinstance(v, float):
                    if self._tolerance > 0:
                        step = self._tolerance
                        v = round(round(v / step) * step, 10)
                    params[k] = v
                else:
                    params[k] = v
            data = str(sorted(params.items())).encode()

        return hashlib.sha256(data).hexdigest()[:16]


class PointLibrarySync:
    """Synchronize PointLibraries between instances."""

    def __init__(self):
        self._dedup = PointDeduplicator()

    def export_bytes(self, library: PointLibrary) -> bytes:
        data = {
            "name": library.name,
            "points": [p.to_dict() for p in library.list_all()],
            "exported_at": time.time(),
        }
        return json.dumps(data, indent=2).encode()

    def import_bytes(self, data: bytes) -> PointLibrary:
        parsed = json.loads(data)
        lib = PointLibrary(name=parsed.get("name", "imported"))
        for pd in parsed.get("points", []):
            lib.add(Point.from_dict(pd))
        return lib

    def sync_to_directory(self, library: PointLibrary, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{library.name}.points.json"
        return library.save(path)

    def sync_from_directory(self, source_dir: Path, name: Optional[str] = None) -> Optional[PointLibrary]:
        if name is not None:
            path = source_dir / f"{name}.points.json"
            if path.exists():
                return PointLibrary.load(path)
            return None

        for f in source_dir.glob("*.points.json"):
            return PointLibrary.load(f)
        return None

    def merge(self, libraries: List[PointLibrary]) -> PointLibrary:
        merged = PointLibrary(name="merged")
        for lib in libraries:
            for point in lib.list_all():
                merged.add(point)

        dedup = PointDeduplicator()
        dedup.add_library(merged)
        dedup.deduplicate()

        return merged
