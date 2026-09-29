"""Embedding utilities for memory consolidation.

Self-contained n-gram TF-IDF embedding and cosine similarity.
No external dependencies beyond numpy.
"""

from __future__ import annotations

import re

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9']+", text.lower())


def ngram_embed(text: str, dimension: int = 384) -> np.ndarray:
    """Canonical n-gram embedding — delegates to the project embedder.

    Single algorithm/dimension/normalization shared with
    ``vector_store.simple_embed`` so consolidation vectors are comparable
    with vectors stored anywhere else.
    """
    from domain.inference._internal.text_embedder import ngram_embed as canonical

    return canonical(text, dimension)
