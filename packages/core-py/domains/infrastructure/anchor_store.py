"""
Anchor store — fixed reference vectors ("stars") for embedding space.

Anchors are semantic poles that don't move during training. New text
measures its distance from these poles without retraining.

Usage:
    store = AnchorStore(dimension=128)
    store.add("truth", [1.0, 0.0, ...])
    store.add("falsehood", [-1.0, 0.0, ...])
    label = store.classify(embedding)  # → "truth"
    distances = store.distances(embedding)  # → {"truth": 0.1, "falsehood": 0.9}
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ANCHOR_DIR = Path(__file__).resolve().parents[4] / "data" / "models"
_ANCHOR_PATH = _ANCHOR_DIR / "anchors.json"


class AnchorStore:
    """Fixed reference vectors that define the coordinate system.

    Anchors are the stars — they don't move. Everything else navigates
    by measuring distance from them.

    The anchor store is a tape recording: read from file, never written
    to by consumers. Only training writes new anchors.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self._anchors: Dict[str, np.ndarray] = {}

    def add(self, name: str, vector: List[float]) -> None:
        """Register a fixed anchor point.

        Args:
            name: semantic label (e.g., "truth", "falsehood", "question")
            vector: L2-normalized embedding vector
        """
        vec = np.array(vector, dtype=np.float32)
        if len(vec) != self.dimension:
            raise ValueError(f"Anchor '{name}' has dim {len(vec)}, expected {self.dimension}")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._anchors[name] = vec

    def get(self, name: str) -> Optional[np.ndarray]:
        """Get anchor vector by name. Returns None if not found."""
        return self._anchors.get(name)

    def names(self) -> List[str]:
        """List all anchor names."""
        return list(self._anchors.keys())

    def distances(self, vector: List[float]) -> Dict[str, float]:
        """Compute cosine distance from vector to all anchors.

        Returns dict of {name: distance} where distance is 1 - cosine_similarity.
        0.0 = identical direction, 2.0 = opposite direction.
        """
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        result = {}
        for name, anchor in self._anchors.items():
            sim = float(np.dot(vec, anchor))
            result[name] = 1.0 - sim
        return result

    def classify(self, vector: List[float]) -> str:
        """Classify vector by nearest anchor.

        Returns the name of the closest anchor.
        """
        distances = self.distances(vector)
        if not distances:
            return "unknown"
        return min(distances, key=distances.get)

    def similarity(self, vector: List[float], anchor_name: str) -> float:
        """Cosine similarity between vector and named anchor.

        Returns float in [-1, 1]. Higher = more similar.
        """
        anchor = self.get(anchor_name)
        if anchor is None:
            return 0.0
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return float(np.dot(vec, anchor))

    def save(self, path: Optional[str] = None) -> None:
        """Save anchors to JSON (tape recording)."""
        path = path or str(_ANCHOR_PATH)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "dimension": self.dimension,
            "anchors": {
                name: vec.tolist()
                for name, vec in self._anchors.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "AnchorStore":
        """Load anchors from JSON (read the tape recording)."""
        path = path or str(_ANCHOR_PATH)
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        store = cls(dimension=data.get("dimension", 128))
        for name, vec in data.get("anchors", {}).items():
            store._anchors[name] = np.array(vec, dtype=np.float32)
        return store


# Default anchors — the semantic poles
DEFAULT_ANCHORS = {
    "truth": "positive factual assertion",
    "falsehood": "negative denial contradiction",
    "question": "interrogative uncertainty",
    "instruction": "command directive task",
    "observation": "neutral descriptive report",
}


def _seed_anchor_from_text(description: str, dimension: int = 128) -> np.ndarray:
    """Generate a deterministic seed vector from a text description.

    Uses character-level hash to create a reproducible vector.
    Not learned — just a stable seed that anchors can be refined from.
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


def get_default_anchors(dimension: int = 128) -> AnchorStore:
    """Create an AnchorStore with default semantic poles.

    These are the stars — fixed reference points that define the
    coordinate system for the embedding space.
    """
    store = AnchorStore(dimension=dimension)
    for name, description in DEFAULT_ANCHORS.items():
        vec = _seed_anchor_from_text(description, dimension)
        store.add(name, vec)
    return store
