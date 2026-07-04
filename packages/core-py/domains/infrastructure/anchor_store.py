"""
Meaning points — fixed semantic reference vectors for embedding space.

Meaning points are fixed reference vectors ("stars") that don't move
during training. New text is positioned by its distance from these
points, which determines its semantic meaning classification.

Usage:
    store = MeaningPoints(dimension=128)
    store.add("factual", [1.0, 0.0, ...])
    store.add("procedural", [-1.0, 0.0, ...])
    label = store.classify(embedding)  # → "factual"
    distances = store.distances(embedding)  # → {"factual": 0.1, "procedural": 0.9}
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_POINT_DIR = Path(__file__).resolve().parents[4] / "data" / "models"
_POINT_PATH = _POINT_DIR / "meaning_points.json"


class MeaningPoints:
    """Fixed reference vectors that define semantic meaning regions.

    Meaning points are the stars — they don't move. A text's meaning is
    determined by which meaning point it's nearest to in embedding space.

    The meaning point store is a tape recording: read from file, never
    written to by consumers. Only training writes new points.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self._points: Dict[str, np.ndarray] = {}

    def add(self, name: str, vector: List[float]) -> None:
        """Register a fixed meaning point.

        Args:
            name: semantic label (e.g., "factual", "procedural", "interrogative")
            vector: L2-normalized embedding vector
        """
        vec = np.array(vector, dtype=np.float32)
        if len(vec) != self.dimension:
            raise ValueError(f"Point '{name}' has dim {len(vec)}, expected {self.dimension}")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._points[name] = vec

    def get(self, name: str) -> Optional[np.ndarray]:
        """Get meaning point vector by name. Returns None if not found."""
        return self._points.get(name)

    def names(self) -> List[str]:
        """List all meaning point names."""
        return list(self._points.keys())

    def distances(self, vector: List[float]) -> Dict[str, float]:
        """Compute cosine distance from vector to all meaning points.

        Returns dict of {name: distance} where distance is 1 - cosine_similarity.
        0.0 = identical direction, 2.0 = opposite direction.
        """
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        result = {}
        for name, point_vec in self._points.items():
            sim = float(np.dot(vec, point_vec))
            result[name] = 1.0 - sim
        return result

    def classify(self, vector: List[float]) -> str:
        """Classify vector by nearest meaning point.

        Returns the name of the closest meaning point (semantic meaning region).
        """
        distances = self.distances(vector)
        if not distances:
            return "unknown"
        return min(distances, key=distances.get)

    def similarity(self, vector: List[float], point_name: str) -> float:
        """Cosine similarity between vector and named meaning point.

        Returns float in [-1, 1]. Higher = more similar.
        """
        point_vec = self.get(point_name)
        if point_vec is None:
            return 0.0
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return float(np.dot(vec, point_vec))

    def remove(self, name: str) -> bool:
        """Remove a meaning point. Returns True if found and removed."""
        if name in self._points:
            del self._points[name]
            return True
        return False

    def save(self, path: Optional[str] = None) -> None:
        """Save meaning points to JSON (tape recording)."""
        path = path or str(_POINT_PATH)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "dimension": self.dimension,
            "points": {
                name: vec.tolist()
                for name, vec in self._points.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "MeaningPoints":
        """Load meaning points from JSON (read the tape recording)."""
        path = path or str(_POINT_PATH)
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        store = cls(dimension=data.get("dimension", 128))
        for name, vec in data.get("points", {}).items():
            store._points[name] = np.array(vec, dtype=np.float32)
        return store


# Backward compat aliases
AnchorStore = MeaningPoints
Tags = MeaningPoints

# Default meaning points — semantic meaning regions
DEFAULT_MEANING_POINTS = {
    "factual": "declarative factual assertion statement",
    "conceptual": "abstract idea concept definition",
    "procedural": "step-by-step instruction process",
    "interrogative": "question uncertainty inquiry",
    "descriptive": "neutral observation description",
    "directive": "command directive request",
    "analytical": "reasoning analysis evaluation",
}


def _seed_point_from_text(description: str, dimension: int = 128) -> np.ndarray:
    """Generate a deterministic seed vector from a text description.

    Uses character-level hash to create a reproducible vector.
    Not learned — just a stable seed that meaning points can be refined from.
    """
    vec = np.zeros(dimension, dtype=np.float64)
    for i, ch in enumerate(description):
        h = hash(ch + str(i))
        idx = abs(h) % dimension
        vec[idx] += 1.0
        # Spread to neighbors
        idx2 = abs(h + 1) % dimension
        vec[idx2] += 0.5

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


# Backward compat aliases
_seed_anchor_from_text = _seed_point_from_text
_seed_tag_from_text = _seed_point_from_text
DEFAULT_TAGS = DEFAULT_MEANING_POINTS


def get_default_meaning_points(dimension: int = 128) -> MeaningPoints:
    """Create a MeaningPoints store with default semantic meaning points.

    These are the stars — fixed reference points that define the
    coordinate system for the embedding space.
    """
    store = MeaningPoints(dimension=dimension)
    for name, description in DEFAULT_MEANING_POINTS.items():
        vec = _seed_point_from_text(description, dimension)
        store.add(name, vec)
    return store


# Backward compat aliases
get_default_anchors = get_default_meaning_points
get_default_tags = get_default_meaning_points
