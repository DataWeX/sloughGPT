"""Canonical text embedder — single source of truth for text → vector.

Local-only, zero-download chain (no sentence-transformers, no torch, no
network, no API keys):

  1. SloTextEmbedder — trained on your own corpus, loaded from a ``.soul``
     checkpoint if one exists and passes its quality gate
  2. word n-gram TF-IDF (numpy + md5, L2-normalized) — zero deps

Invariant: same algorithm + dimension + normalization for every entry point,
so vectors written by one consumer are comparable with vectors read by
another. All other embedding functions — ``vector_store.simple_embed``,
``EmbeddingService``, the ``context_core``/``auto_ingest`` wrappers,
``Embedder``/``InMemoryEmbedder``, meta-weights, memory consolidation —
delegate here.
"""

from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

logger = logging.getLogger("slo.text_embedder")

DEFAULT_DIMENSION = 384


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9']+", text.lower())


def ngram_embed(text: str, dimension: int = DEFAULT_DIMENSION) -> np.ndarray:
    """Word-level n-gram TF-IDF embedding using numpy only.

    Outperforms character n-grams for semantic retrieval by operating
    on word tokens. Extracts word unigrams, bigrams, and trigrams.
    Log-frequency TF weighting, L2-normalized. Empty input → zero vector.
    """
    vec = np.zeros(dimension, dtype=np.float64)
    tokens = tokenize(text)

    if not tokens:
        return vec

    ngrams: list[str] = []
    for n in (1, 2, 3):
        for i in range(max(0, len(tokens) - n + 1)):
            ngrams.append(" ".join(tokens[i : i + n]))

    for ng in ngrams:
        h = int(hashlib.md5(ng.encode()).hexdigest()[:8], 16)
        idx = h % dimension
        vec[idx] += 1.0

    vec = np.log1p(vec)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _fit(vec: np.ndarray, dimension: int) -> np.ndarray:
    """Pad/truncate to ``dimension`` and re-normalize (L2)."""
    if len(vec) != dimension:
        if len(vec) < dimension:
            vec = np.pad(vec, (0, dimension - len(vec)))
        else:
            vec = vec[:dimension]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
    return vec


class TextEmbedder:
    """Canonical embedder: SloNet checkpoint (quality-gated) → n-gram."""

    def __init__(self, dimension: int = DEFAULT_DIMENSION):
        self.dimension = dimension
        self._slo = None
        self._slo_rejected = False

    def embed(self, text: str, dimension: int | None = None) -> list[float]:
        """Embed ``text`` into a L2-normalized ``dimension``-dim vector."""
        dim = dimension or self.dimension
        if not text or not text.strip():
            return np.zeros(dim).tolist()

        vec = self._try_slo(text)
        if vec is None:
            return ngram_embed(text, dim).tolist()

        vec = _fit(np.asarray(vec, dtype=np.float64), dim)
        return vec.tolist()

    def embed_batch(self, texts: list[str], dimension: int | None = None) -> list[list[float]]:
        return [self.embed(t, dimension) for t in texts]

    def _try_slo(self, text: str) -> np.ndarray | None:
        """SloNet-trained embedder (no downloads, trained on your corpus)."""
        if self._slo_rejected:
            return None
        if self._slo is None:
            try:
                from domain.inference._internal.slo_embedder import SloTextEmbedder

                candidate = SloTextEmbedder.load()
                if candidate is not None and not candidate.acceptable():
                    logger.info(
                        "SloNet embedder rejected by quality gate (%s), using n-gram fallback",
                        candidate.quality,
                    )
                    self._slo_rejected = True
                    return None
                self._slo = candidate
            except Exception as e:
                logger.warning(
                    "text_embedder: SloNet embedder load failed, using n-gram fallback",
                    extra={"error": str(e)},
                )
                return None
        if self._slo is None:
            return None
        try:
            vec = self._slo.embed(text)
            return np.asarray(vec, dtype=np.float64)
        except Exception:
            logger.warning("text_embedder: SloNet embed failed, using n-gram", exc_info=True)
            return None


_text_embedder: TextEmbedder | None = None


def get_text_embedder() -> TextEmbedder:
    """Process-wide canonical embedder singleton."""
    global _text_embedder
    if _text_embedder is None:
        _text_embedder = TextEmbedder()
    return _text_embedder


def embed_text(text: str, dimension: int = DEFAULT_DIMENSION) -> list[float]:
    """Embed ``text`` with the canonical embedder (see module docstring)."""
    return get_text_embedder().embed(text, dimension)


__all__ = [
    "TextEmbedder",
    "DEFAULT_DIMENSION",
    "embed_text",
    "get_text_embedder",
    "ngram_embed",
    "tokenize",
]
