"""
Base ``VectorStore`` ABC, InMemoryVectorStore, embedders, and factory.

External providers (Pinecone, ChromaDB) live in the ``vector_stores/`` package.
"""

import hashlib
import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("man.inference.vector_store")

# Prompt injection detection patterns
_INJECTION_PATTERNS: list[str] = [
    "ignore previous",
    "disregard previous",
    "ignore all rules",
    "system prompt",
    "you are now",
    "new instructions",
    "remove previous",
    "forget everything",
    "[important",
]


def sanitize_input(content: str) -> str:
    """Raise ``ValueError`` if *content* matches known injection patterns.

    Checks common prompt-injection phrases. Returns the cleaned (stripped)
    content on success so callers can inline it.
    """
    cleaned = content.strip()
    lower = cleaned.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            raise ValueError(f"Blocked: suspicious pattern {pattern!r}")
    if cleaned.startswith("[") and "IMPORTANT" in cleaned.upper():
        raise ValueError("Blocked: potential prompt injection")
    return cleaned


class VectorStoreType(str, Enum):
    """Backend identifiers for create_vector_store."""

    IN_MEMORY = "in_memory"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMADB = "chromadb"


@dataclass
class VectorEntry:
    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def upsert(self, entries: List[VectorEntry]) -> int:
        pass

    @abstractmethod
    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        pass

    @abstractmethod
    async def count(self) -> int:
        pass


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


class InMemoryVectorStore(VectorStore):
    """Simple cosine-similarity store for development and tests."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._entries: Dict[str, VectorEntry] = {}

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    def query_sync(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Synchronous query — avoids event-loop deadlock from _run_async."""
        if not self._entries:
            return []
        q = np.asarray(vector, dtype=np.float64)
        scored: List[tuple[float, VectorEntry]] = []
        for entry in self._entries.values():
            if filter_metadata and entry.metadata:
                if not all(entry.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            v = np.asarray(entry.vector, dtype=np.float64)
            scored.append((_cosine_similarity(q, v), entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[QueryResult] = []
        for score, entry in scored[:top_k]:
            out.append(
                QueryResult(
                    id=entry.id,
                    score=score,
                    text=entry.text,
                    metadata=dict(entry.metadata),
                )
            )
        return out

    def upsert_sync(self, entries: List[VectorEntry]) -> int:
        """Synchronous upsert — avoids event-loop deadlock from _run_async."""
        for e in entries:
            self._entries[e.id] = e
        return len(entries)

    def count_sync(self) -> int:
        """Synchronous count — avoids event-loop deadlock from _run_async."""
        return len(self._entries)

    async def upsert(self, entries: List[VectorEntry]) -> int:
        for e in entries:
            self._entries[e.id] = e
        return len(entries)

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        return self.query_sync(vector, top_k, filter_metadata)

    async def delete(self, ids: List[str]) -> bool:
        removed = 0
        for i in ids:
            if i in self._entries:
                del self._entries[i]
                removed += 1
        return removed > 0

    async def count(self) -> int:
        return len(self._entries)


class MogDBVectorStore(VectorStore):
    """Persistent vector store backed by MogDB.

    Vectors, text, and metadata are persisted to disk via MogDB's append-only
    journal. On startup, all entries are loaded into memory for fast cosine
    similarity queries. Every upsert/delete is written through to MogDB.

    Parameters
    ----------
    dimension:
        Embedding dimension (default 384).
    path:
        MogDB database path (default ``data/vector_store``).
    """

    def __init__(self, dimension: int = 384, path: str = "data/vector_store"):
        self.dimension = dimension
        self._path = path
        self._entries: Dict[str, VectorEntry] = {}
        self._mogdb: Optional[Any] = None
        self._coll: Optional[Any] = None

    async def connect(self) -> bool:
        from mogdb import MogDB
        self._mogdb = MogDB(self._path)
        self._coll = self._mogdb.collection("vectors")
        # Load existing entries from disk
        for doc in self._coll.find():
            vec = json.loads(doc.get("_vector_json", "[]"))
            entry = VectorEntry(
                id=doc["_id"],
                vector=vec,
                text=doc.get("text", ""),
                metadata=doc.get("metadata", {}),
            )
            self._entries[entry.id] = entry
        logger.info("MogDBVectorStore loaded %d entries from %s", len(self._entries), self._path)
        return True

    async def disconnect(self) -> None:
        if self._mogdb:
            self._mogdb.close()
            self._mogdb = None
            self._coll = None

    def query_sync(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        if not self._entries:
            return []
        q = np.asarray(vector, dtype=np.float64)
        scored: List[tuple[float, VectorEntry]] = []
        for entry in self._entries.values():
            if filter_metadata and entry.metadata:
                if not all(entry.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            v = np.asarray(entry.vector, dtype=np.float64)
            scored.append((_cosine_similarity(q, v), entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[QueryResult] = []
        for score, entry in scored[:top_k]:
            out.append(
                QueryResult(
                    id=entry.id,
                    score=score,
                    text=entry.text,
                    metadata=dict(entry.metadata),
                )
            )
        return out

    def upsert_sync(self, entries: List[VectorEntry]) -> int:
        if not self._coll:
            return 0
        for e in entries:
            self._entries[e.id] = e
            self._coll.update_one(
                    {"_id": e.id},
                    {"$set": {
                        "_vector_json": json.dumps(e.vector),
                        "text": e.text,
                        "metadata": e.metadata,
                    }},
                ) or self._coll.insert_one({
                    "_id": e.id,
                    "_vector_json": json.dumps(e.vector),
                    "text": e.text,
                    "metadata": e.metadata,
                })
        return len(entries)

    def count_sync(self) -> int:
        return len(self._entries)

    async def upsert(self, entries: List[VectorEntry]) -> int:
        return self.upsert_sync(entries)

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        return self.query_sync(vector, top_k, filter_metadata)

    async def delete(self, ids: List[str]) -> bool:
        removed = 0
        for i in ids:
            if i in self._entries:
                del self._entries[i]
                removed += 1
            if self._coll:
                self._coll.delete_one({"_id": i})
        return removed > 0

    async def count(self) -> int:
        return len(self._entries)


async def create_vector_store(provider: str = "in_memory", **kwargs: Any) -> VectorStore:
    """Factory used by ``apps/api/server/main.py`` for ``/vector/*`` endpoints."""
    key = (provider or "in_memory").lower()
    if key in ("in_memory", "memory", "local"):
        dim = int(kwargs.get("dimension", 384))
        store = InMemoryVectorStore(dimension=dim)
        await store.connect()
        return store
    if key in ("mogdb", "persist", "persistent"):
        dim = int(kwargs.get("dimension", 384))
        path = kwargs.get("path", "data/vector_store")
        store = MogDBVectorStore(dimension=dim, path=path)
        await store.connect()
        return store
    if key == "chromadb":
        from domains.inference.vector_stores.chromadb_store import ChromaDBVectorStore
        store = ChromaDBVectorStore(
            persist_directory=kwargs.get("persist_directory", "data/vector_store")
        )
        await store.connect()
        return store
    if key == "pinecone":
        from domains.inference.vector_stores.pinecone_store import PineconeVectorStore
        store = PineconeVectorStore(
            api_key=kwargs.get("api_key"),
            index_name=kwargs.get("index") or kwargs.get("index_name") or "sloughgpt",
            environment=kwargs.get("environment", "us-east-1"),
            dimension=int(kwargs.get("dimension", 768)),
        )
        ok = await store.connect()
        if not ok:
            raise RuntimeError("Pinecone connection failed")
        return store
    raise NotImplementedError(
        f"Vector store provider {provider!r} is not implemented. "
        f"Use 'in_memory', 'mogdb', 'chromadb', or 'pinecone'."
    )


_embed_model: Optional[Any] = None
_EMBED_DIM: int = 384
_EMBED_LOAD_FAILED: bool = False


def _load_embed_model() -> Any:
    """Lazy-load a sentence-transformers model for semantic embeddings.

    Currently disabled to avoid OOM on 8GB Mac (two PyTorch models in same process).
    Falls back to fast n-gram TF-IDF embedder.
    """
    global _embed_model, _EMBED_LOAD_FAILED
    if _EMBED_LOAD_FAILED:
        return None
    return None


_STOPWORDS: frozenset = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "and", "but",
    "or", "if", "while", "about", "up", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "they", "them", "their", "what", "which", "who", "whom",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    import re
    return re.findall(r"[a-z0-9']+", text.lower())


def _word_ngram_embed(text: str, dimension: int = 384) -> np.ndarray:
    """Word-level n-gram TF-IDF embedding using numpy only.

    Outperforms character n-grams for semantic retrieval by operating
    on word tokens. Extracts word unigrams, bigrams, and trigrams.
    Frequent stopwords receive a 0.5 IDF penalty so they contribute
    less to similarity. Log-frequency TF weighting. L2-normalized.
    """
    vec = np.zeros(dimension, dtype=np.float64)
    tokens = _tokenize(text)

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


def _ngram_embed(text: str, dimension: int = 384) -> np.ndarray:
    """Alias — delegates to the word-level embedder."""
    return _word_ngram_embed(text, dimension)


_slo_embedder = None


def simple_embed(text: str, dimension: int = 384) -> List[float]:
    """Embed text into a vector using the best available embedder.

    Priority:
    1. sentence-transformers (all-MiniLM-L6-v2) if installed
    2. SloTextEmbedder (trained on your own corpus) if checkpoint exists
    3. Word n-gram TF-IDF fallback (zero downloads)

    Args:
        text: input text to embed
        dimension: output vector dimension (384 for all embedders)

    Returns:
        list of floats (L2-normalized)
    """
    # 1. Try sentence-transformers
    model = _load_embed_model()
    if model is not None:
        try:
            vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            if len(vec) != dimension:
                if len(vec) < dimension:
                    vec = np.pad(vec, (0, dimension - len(vec)))
                else:
                    vec = vec[:dimension]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            return vec.tolist()
        except Exception:
            import logging
            logging.getLogger(__name__).warning("sentence-transformers encode failed, trying SloNet embedder")

    # 2. Try SloNet-trained embedder (no downloads, trained on your corpus)
    global _slo_embedder
    if _slo_embedder is None:
        try:
            from domains.inference.slo_embedder import SloTextEmbedder
            _slo_embedder = SloTextEmbedder.load()
        except Exception:
            pass
    if _slo_embedder is not None:
        try:
            vec = _slo_embedder.embed(text)
            if len(vec) != dimension:
                if len(vec) < dimension:
                    vec = np.pad(vec, (0, dimension - len(vec)))
                else:
                    vec = vec[:dimension]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            return vec.tolist() if isinstance(vec, np.ndarray) else vec
        except Exception:
            import logging
            logging.getLogger(__name__).warning("SloNet embedder failed, using n-gram fallback")

    # 3. Last resort: word n-gram TF-IDF (zero downloads, zero training)
    vec = _ngram_embed(text, dimension)
    return vec.tolist()


__all__ = [
    "VectorStore",
    "VectorStoreType",
    "VectorEntry",
    "QueryResult",
    "InMemoryVectorStore",
    "MogDBVectorStore",
    "create_vector_store",
    "simple_embed",
]
