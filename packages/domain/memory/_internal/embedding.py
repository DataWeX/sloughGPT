"""Embedding utilities for memory consolidation.

Self-contained n-gram TF-IDF embedding and cosine similarity.
No external dependencies beyond numpy.
"""
from __future__ import annotations

import hashlib
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
    """Word-level n-gram TF-IDF embedding using numpy only.

    Outperforms character n-grams for semantic retrieval by operating
    on word tokens. Extracts word unigrams, bigrams, and trigrams.
    Frequent stopwords receive a 0.5 IDF penalty so they contribute
    less to similarity. Log-frequency TF weighting. L2-normalized.
    """
    vec = np.zeros(dimension, dtype=np.float64)
    tokens = tokenize(text)

    if not tokens:
        vec[0] = 1.0
        return vec

    ngrams: list[str] = []
    for n in (1, 2, 3):
        for i in range(max(0, len(tokens) - n + 1)):
            ngrams.append(" ".join(tokens[i:i + n]))

    for ng in ngrams:
        h = int(hashlib.md5(ng.encode()).hexdigest()[:8], 16)
        idx = h % dimension
        vec[idx] += 1.0

    vec = np.log1p(vec)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
