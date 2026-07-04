"""
Embedding service — foundational base layer for vector operations.

This is the foundational base layer, not opinion. It provides:
- Embedding computation (delegates to best available backend)
- Meaning point integration (fixed semantic reference points)
- Dimension management (padding/truncation)

Usage:
    from domains.infrastructure.embedding_service import EmbeddingService
    svc = EmbeddingService(dimension=128)
    vec = svc.embed("hello world")
    label = svc.classify("the sky is blue")
"""
import numpy as np
from typing import Dict, List, Optional

from .anchor_store import MeaningPoints, get_default_meaning_points


class EmbeddingService:
    """Foundational base layer for embedding operations.

    This is the foundational base layer — it doesn't have opinions about what embeddings
    mean, only how to compute and compare them.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self._meaning_points = get_default_meaning_points(dimension=dimension)
        self._backend = None  # lazy-loaded

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
        """Classify text by nearest meaning point.

        Returns the name of the closest meaning point (e.g., "factual", "interrogative").
        Uses embedding distance to fixed semantic meaning points.
        """
        vec = self.embed(text)
        return self._meaning_points.classify(vec)

    def distances(self, text: str) -> Dict[str, float]:
        """Compute distance from text to all meaning points.

        Returns dict of {name: distance} where 0.0 = same direction.
        """
        vec = self.embed(text)
        return self._meaning_points.distances(vec)

    def similarity(self, text: str, point_name: str) -> float:
        """Cosine similarity between text and named meaning point."""
        vec = self.embed(text)
        return self._meaning_points.similarity(vec, point_name)

    @property
    def meaning_points(self) -> MeaningPoints:
        """Access the meaning point store (read-only tape recording)."""
        return self._meaning_points

    @property
    def anchors(self) -> MeaningPoints:
        """Backward compat alias for meaning_points."""
        return self._meaning_points


# Module-level singleton
_service: Optional[EmbeddingService] = None


def get_embedding_service(dimension: int = 128) -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _service
    if _service is None or _service.dimension != dimension:
        _service = EmbeddingService(dimension=dimension)
    return _service
