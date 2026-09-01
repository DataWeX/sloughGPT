"""
Embedding service — foundational base layer for vector operations.

This is the foundational base layer, not opinion. It provides:
- Embedding computation (delegates to best available backend)
- Meaning tag integration (fixed semantic reference points)
- Dimension management (padding/truncation)
- Truth verdict (distance to meaning regions)
- Model hash tagging (MD5 of checkpoint file)

Usage:
    from domains.infrastructure.embedding_service import EmbeddingService
    svc = EmbeddingService(dimension=128)
    vec = svc.embed("hello world")
    label = svc.classify("the sky is blue")
    verdict = svc.truth_verdict("the sky is blue")
"""
from __future__ import annotations

import hashlib
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from .anchor_store import MeaningTags, get_default_meaning_tags


class EmbeddingService:
    """Foundational base layer for embedding operations.

    This is the foundational base layer — it doesn't have opinions about what embeddings
    mean, only how to compute and compare them.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._meaning_tags = get_default_meaning_tags(dimension=dimension)
        self._backend = None  # lazy-loaded
        self._model_hash: Optional[str] = None

    def embed(self, text: str) -> List[float]:
        """Embed text into a vector of fixed dimension.

        Delegates to the best available backend:
        1. sentence-transformers (if installed)
        2. SloNet trained model (if checkpoint exists)
        3. Word n-gram TF-IDF (zero downloads)
        """
        from domains.inference.vector_store import simple_embed
        return simple_embed(text, dimension=self.dimension)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        return [self.embed(t) for t in texts]

    def classify(self, text: str) -> str:
        """Classify text by nearest meaning tag.

        Returns the name of the closest meaning tag (e.g., "factual", "interrogative").
        Uses embedding distance to fixed semantic meaning tags.
        """
        vec = self.embed(text)
        return self._meaning_tags.classify(vec)

    def distances(self, text: str) -> Dict[str, float]:
        """Compute distance from text to all meaning tags.

        Returns dict of {name: distance} where 0.0 = same direction.
        """
        vec = self.embed(text)
        return self._meaning_tags.distances(vec)

    def similarity(self, text: str, tag_name: str) -> float:
        """Cosine similarity between text and named meaning tag."""
        vec = self.embed(text)
        return self._meaning_tags.similarity(vec, tag_name)

    def truth_verdict(self, text: str) -> Dict:
        """Compute truth verdict for text based on distance to meaning regions.

        Returns dict with:
            - verdict: nearest meaning tag name
            - distances: dict of {name: distance} for all meaning tags
            - confidence: 1 - distance to nearest tag (0-1)
            - model_hash: MD5 hash of loaded model checkpoint (if any)
        """
        vec = self.embed(text)
        dists = self._meaning_tags.distances(vec)
        nearest = min(dists, key=dists.get)
        nearest_dist = dists[nearest]

        return {
            "verdict": nearest,
            "distances": dists,
            "confidence": max(0.0, 1.0 - nearest_dist),
            "model_hash": self._model_hash,
        }

    def set_model_hash(self, checkpoint_path: Optional[str] = None) -> Optional[str]:
        """Compute and cache MD5 hash of a model checkpoint file.

        Args:
            checkpoint_path: path to .soul checkpoint file. If None, clears hash.

        Returns:
            MD5 hex digest string, or None if no path provided or file not found.
        """
        if checkpoint_path is None:
            self._model_hash = None
            return None

        path = Path(checkpoint_path)
        if not path.exists():
            self._model_hash = None
            return None

        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        self._model_hash = h.hexdigest()
        return self._model_hash

    @property
    def meaning_tags(self) -> MeaningTags:
        """Access the meaning tag store (read-only tape recording)."""
        return self._meaning_tags

    @property
    def model_hash(self) -> Optional[str]:
        """Current model hash (MD5 of checkpoint, or None)."""
        return self._model_hash


# Module-level singleton
_service: Optional[EmbeddingService] = None
_embedding_service_lock = threading.Lock()


def get_embedding_service(dimension: int = 384) -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _service
    if _service is None or _service.dimension != dimension:
        with _embedding_service_lock:
            if _service is None or _service.dimension != dimension:
                _service = EmbeddingService(dimension=dimension)
    return _service


def reset_embedding_service() -> None:
    """Reset the singleton (for testing)."""
    global _service
    with _embedding_service_lock:
        _service = None
