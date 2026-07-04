"""
Embedding service — foundational base layer for vector operations.

This is the foundational base layer, not opinion. It provides:
- Embedding computation (delegates to best available backend)
- Anchor store integration (fixed reference points)
- Dimension management (padding/truncation)

Usage:
    from domains.infrastructure.embedding_service import EmbeddingService
    svc = EmbeddingService(dimension=128)
    vec = svc.embed("hello world")
    label = svc.classify("the sky is blue")
"""
import numpy as np
from typing import Dict, List, Optional

from .anchor_store import AnchorStore, get_default_anchors


class EmbeddingService:
    """Foundational base layer for embedding operations.

    This is the foundational base layer — it doesn't have opinions about what embeddings
    mean, only how to compute and compare them.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self._anchor_store = get_default_anchors(dimension=dimension)
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
        """Classify text by nearest anchor point.

        Returns the name of the closest anchor (e.g., "truth", "question").
        """
        vec = self.embed(text)
        return self._anchor_store.classify(vec)

    def distances(self, text: str) -> Dict[str, float]:
        """Compute distance from text to all anchor points.

        Returns dict of {name: distance} where 0.0 = same direction.
        """
        vec = self.embed(text)
        return self._anchor_store.distances(vec)

    def similarity(self, text: str, anchor: str) -> float:
        """Cosine similarity between text and named anchor."""
        vec = self.embed(text)
        return self._anchor_store.similarity(vec, anchor)

    @property
    def anchors(self) -> AnchorStore:
        """Access the anchor store (read-only tape recording)."""
        return self._anchor_store


# Module-level singleton
_service: Optional[EmbeddingService] = None


def get_embedding_service(dimension: int = 128) -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _service
    if _service is None or _service.dimension != dimension:
        _service = EmbeddingService(dimension=dimension)
    return _service
