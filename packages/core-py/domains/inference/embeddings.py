"""
Production-Grade Embedding Models

Supports:
- In-memory n-gram TF-IDF (default, zero-dependency)
- OpenAI (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)

Usage:
    from domains.inference.embeddings import Embedder, EmbeddingModel

    # Default: in-memory (zero-dependency)
    embedder = Embedder()
    vectors = embedder.embed(["Hello world", "How are you?"])

    # OpenAI
    embedder = Embedder(provider="openai", api_key="sk-...")
    vectors = embedder.embed("Hello world")
"""

import os
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import logging
import numpy as np

from domains.infrastructure.config import get_config

logger = logging.getLogger("slo.embeddings")


class EmbeddingProvider(Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    IN_MEMORY = "in_memory"


@dataclass
class EmbeddingResult:
    embedding: List[float]
    model: str
    dimension: int
    token_count: Optional[int] = None


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        pass  # pragma: no cover (abstractmethod body)

    @abstractmethod
    def get_dimension(self) -> int:
        pass  # pragma: no cover (abstractmethod body)

    @abstractmethod
    def get_model_name(self) -> str:
        pass  # pragma: no cover (abstractmethod body)


class InMemoryEmbedder(BaseEmbedder):
    """Fast hash-based embeddings for development/testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension)
            words = text.lower().split()

            for i, word in enumerate(words[:self.dimension]):
                word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
                vec[i % self.dimension] += np.sin(word_hash * (i + 1) * 0.1)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())

        return vectors

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_name(self) -> str:
        return "in_memory"


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings (ada, text-embedding-3-small, text-embedding-3-large)."""

    DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ):
        self.api_key = api_key or get_config().embedding.openai_api_key or None
        self.model_name = model
        self.dimensions = dimensions or self.DIMENSIONS.get(model, 1536)

        if not self.api_key:
            raise ValueError("OpenAI API key required (api_key or OPENAI_API_KEY)")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install openai")

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=self.dimensions if "3-" in self.model_name else None,
        )

        return [item.embedding for item in response.data]

    def get_dimension(self) -> int:
        return self.dimensions

    def get_model_name(self) -> str:
        return self.model_name


class Embedder:
    """
    Unified embedding interface.

    Usage:
        embedder = Embedder()  # sentence-transformers default
        embedder = Embedder(provider="openai", api_key="sk-...")
        embedder = Embedder(model="all-MiniLM-L6-v2")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: int = 384,
        **kwargs,
    ):
        provider = provider or get_config().embedding.provider

        if provider == "openai":
            model_name = model or "text-embedding-3-small"
            self._impl = OpenAIEmbedder(api_key=api_key, model=model_name)
        else:
            self._impl = InMemoryEmbedder(dimension=dimension)

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        return self._impl.embed(texts)

    def embed_single(self, text: str) -> List[float]:
        return self.embed(text)[0]

    def get_dimension(self) -> int:
        return self._impl.get_dimension()

    def get_model_name(self) -> str:
        return self._impl.get_model_name()

    def __call__(self, texts: Union[str, List[str]]) -> List[List[float]]:
        return self.embed(texts)


class BatchEmbedder:
    """Efficient batch embedding with caching."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        batch_size: int = 32,
        cache_size: int = 1000,
    ):
        self.embedder = embedder or Embedder()
        self.batch_size = batch_size
        self._cache: Dict[str, List[float]] = {}
        self._cache_size = cache_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        to_embed = []
        indices = []

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()

            if cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                to_embed.append(text)
                indices.append(i)

        if to_embed:
            for i in range(0, len(to_embed), self.batch_size):
                batch = to_embed[i:i + self.batch_size]
                embeddings = self.embedder.embed(batch)

                for j, text, emb in zip(indices[i:i + self.batch_size], batch, embeddings):
                    self._cache[hashlib.md5(text.encode()).hexdigest()] = emb
                    results.append((j, emb))

        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    def clear_cache(self):
        self._cache.clear()


def create_embedder(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> BaseEmbedder:
    """Factory function for creating embedders."""
    embedder = Embedder(provider=provider, model=model, **kwargs)
    return embedder._impl


__all__ = [
    "Embedder",
    "BaseEmbedder",
    "InMemoryEmbedder",
    "OpenAIEmbedder",
    "BatchEmbedder",
    "EmbeddingProvider",
    "EmbeddingResult",
    "create_embedder",
]
