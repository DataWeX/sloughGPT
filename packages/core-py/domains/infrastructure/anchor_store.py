"""
Meaning tags — fixed semantic reference vectors for embedding space.

Meaning tags are fixed reference vectors ("stars") that don't move
during training. New text is positioned by its distance from these
tags, which determines its semantic meaning classification.

Usage:
    from domains.infrastructure.anchor_store import MeaningTags, get_default_meaning_tags
    store = get_default_meaning_tags(dimension=128)
"""

from __future__ import annotations
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from domains.shared import find_repo_root

_TAG_DIR = find_repo_root(Path(__file__).resolve()) / "data" / "models"
_TAG_PATH = _TAG_DIR / "meaning_tags.json"


class MeaningTags:
    """Fixed reference vectors that define semantic meaning regions.

    Meaning tags are the stars — they don't move. A text's meaning is
    determined by which meaning tag it's nearest to in embedding space.

    The meaning tag store is a tape recording: read from file, never
    written to by consumers. Only training writes new tags.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self._tags: Dict[str, np.ndarray] = {}

    def add(self, name: str, vector: List[float]) -> None:
        """Register a fixed meaning tag.

        Args:
            name: semantic label (e.g., "factual", "procedural", "interrogative")
            vector: L2-normalized embedding vector
        """
        vec = np.array(vector, dtype=np.float32)
        if len(vec) != self.dimension:
            raise ValueError(f"Tag '{name}' has dim {len(vec)}, expected {self.dimension}")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._tags[name] = vec

    def get(self, name: str) -> Optional[np.ndarray]:
        """Get meaning tag vector by name. Returns None if not found."""
        return self._tags.get(name)

    def names(self) -> List[str]:
        """List all meaning tag names."""
        return list(self._tags.keys())

    def distances(self, vector: List[float]) -> Dict[str, float]:
        """Compute cosine distance from vector to all meaning tags.

        Returns dict of {name: distance} where distance is 1 - cosine_similarity.
        0.0 = identical direction, 2.0 = opposite direction.
        """
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        result = {}
        for name, tag_vec in self._tags.items():
            sim = float(np.dot(vec, tag_vec))
            result[name] = 1.0 - sim
        return result

    def classify(self, vector: List[float]) -> str:
        """Classify vector by nearest meaning tag.

        Returns the name of the closest meaning tag (semantic meaning region).
        """
        distances = self.distances(vector)
        if not distances:
            return "unknown"
        return min(distances, key=distances.get)

    def similarity(self, vector: List[float], tag_name: str) -> float:
        """Cosine similarity between vector and named meaning tag.

        Returns float in [-1, 1]. Higher = more similar.
        """
        tag_vec = self.get(tag_name)
        if tag_vec is None:
            return 0.0
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return float(np.dot(vec, tag_vec))

    def remove(self, name: str) -> bool:
        """Remove a meaning tag. Returns True if found and removed."""
        if name in self._tags:
            del self._tags[name]
            return True
        return False

    def save(self, path: Optional[str] = None) -> None:
        """Save meaning tags to JSON (tape recording)."""
        path = path or str(_TAG_PATH)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "dimension": self.dimension,
            "tags": {
                name: vec.tolist()
                for name, vec in self._tags.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def refine(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        lr: float = 0.1,
        min_samples: int = 3,
    ) -> Dict[str, int]:
        """Refine meaning tags toward centroids of classified texts.

        After training, the seed vectors may not align well with the actual
        embedding distribution. This method moves each tag toward the mean
        embedding of texts classified under that tag.

        Args:
            texts: list of input texts
            embeddings: (N, D) L2-normalized embedding array
            lr: learning rate (0.0 = no change, 1.0 = jump to centroid)
            min_samples: minimum texts needed to move a tag

        Returns:
            Dict of {tag_name: num_samples_used} for tags that were refined.
        """
        if len(texts) == 0 or embeddings.shape[0] == 0:
            return {}

        # Classify each embedding
        label_to_indices: Dict[str, List[int]] = {}
        for i in range(len(texts)):
            label = self.classify(embeddings[i].tolist())
            label_to_indices.setdefault(label, []).append(i)

        refined = {}
        for name, indices in label_to_indices.items():
            if name not in self._tags:
                continue
            if len(indices) < min_samples:
                continue

            # Compute centroid of texts in this tag
            centroid = embeddings[indices].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            # Move tag toward centroid: tag = (1-lr)*tag + lr*centroid
            old_vec = self._tags[name]
            new_vec = (1.0 - lr) * old_vec + lr * centroid
            norm = np.linalg.norm(new_vec)
            if norm > 0:
                new_vec = new_vec / norm
            self._tags[name] = new_vec.astype(np.float32)
            refined[name] = len(indices)

        return refined

    @classmethod
    def load(cls, path: Optional[str] = None) -> "MeaningTags":
        """Load meaning tags from JSON (read the tape recording)."""
        path = path or str(_TAG_PATH)
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        store = cls(dimension=data.get("dimension", 128))
        for name, vec in data.get("tags", {}).items():
            store._tags[name] = np.array(vec, dtype=np.float32)
        return store


# Default meaning tags — semantic meaning regions
DEFAULT_MEANING_TAGS = {
    "factual": "declarative factual assertion statement",
    "conceptual": "abstract idea concept definition",
    "procedural": "step-by-step instruction process",
    "interrogative": "question uncertainty inquiry",
    "descriptive": "neutral observation description",
    "directive": "command directive request",
    "analytical": "reasoning analysis evaluation",
}


def _seed_tag_from_text(description: str, dimension: int = 128) -> np.ndarray:
    """Generate a deterministic seed vector from a text description.

    Uses character-level hash to create a reproducible vector.
    Not learned — just a stable seed that meaning tags can be refined from.
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


def get_default_meaning_tags(dimension: int = 128) -> MeaningTags:
    """Create a MeaningTags store with default semantic meaning tags.

    These are the stars — fixed reference points that define the
    coordinate system for the embedding space.
    """
    store = MeaningTags(dimension=dimension)
    for name, description in DEFAULT_MEANING_TAGS.items():
        vec = _seed_tag_from_text(description, dimension)
        store.add(name, vec)
    return store
