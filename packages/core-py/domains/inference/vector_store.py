"""
Production-Grade Vector Store - Pinecone Only

Usage:
    from domains.inference.vector_store import PineconeVectorStore
    
    store = PineconeVectorStore(
        api_key="your-api-key",
        index_name="production"
    )
    await store.connect()
    await store.upsert([{"id": "1", "vector": [...], "text": "..."}])
    results = await store.query(vector=[...], top_k=5)
"""

import os
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("man.inference.vector_store")


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


class PineconeVectorStore(VectorStore):
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: str = "sloughgpt",
        environment: str = "us-east-1",
        dimension: int = 768,
        metric: str = "cosine",
        host: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.environment = environment
        self.dimension = dimension
        self.metric = metric
        self.host = host
        self.index = None
        self.serverless_spec = None
        self.pod_spec = None

    async def connect(self) -> bool:
        try:
            from pinecone import Pinecone, ServerlessSpec, PodSpec

            if not self.api_key:
                raise ValueError("PINECONE_API_KEY is required")

            self.client = Pinecone(api_key=self.api_key)

            if self.index_name not in [idx.name for idx in self.client.list_indexes()]:
                if self.environment in ["us-east-1", "us-west-2", "eu-west-1"]:
                    self.serverless_spec = ServerlessSpec(
                        cloud=self.environment.split("-")[0].upper(),
                        region=self.environment
                    )
                    self.client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=self.serverless_spec,
                    )
                else:
                    self.pod_spec = PodSpec(
                        environment=self.environment,
                        replicas=1,
                        shards=1
                    )
                    self.client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=self.pod_spec,
                    )

            self.index = self.client.Index(self.index_name)
            return True
        except ImportError:
            raise ImportError("pip install pinecone-client")
        except Exception as e:
            print(f"Pinecone connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        self.index = None

    async def upsert(self, entries: List[VectorEntry]) -> int:
        if not self.index:
            raise RuntimeError("Not connected to Pinecone")

        vectors = []
        for entry in entries:
            vectors.append({
                "id": entry.id,
                "values": entry.vector,
                "metadata": {
                    "text": entry.text,
                    **entry.metadata
                }
            })

        self.index.upsert(vectors=vectors)
        return len(entries)

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        if not self.index:
            raise RuntimeError("Not connected to Pinecone")

        query_params = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True,
        }

        if filter_metadata:
            query_params["filter"] = filter_metadata

        results = self.index.query(**query_params)

        return [
            QueryResult(
                id=match["id"],
                score=match.get("score", 0.0),
                text=match.get("metadata", {}).get("text", ""),
                metadata={k: v for k, v in match.get("metadata", {}).items() if k != "text"},
            )
            for match in results.get("matches", [])
        ]

    async def delete(self, ids: List[str]) -> bool:
        if not self.index:
            return False
        self.index.delete(ids=ids)
        return True

    async def count(self) -> int:
        if not self.index:
            return 0
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)


class ChromaDBVectorStore(VectorStore):
    """ChromaDB vector store with persistent storage."""

    def __init__(self, persist_directory: str = "data/vector_store", collection_name: str = "sloughgpt"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    async def connect(self) -> bool:
        try:
            import chromadb
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            return True
        except ImportError:
            raise ImportError("pip install chromadb")
        except Exception as e:
            print(f"ChromaDB connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        self.collection = None
        self.client = None

    async def upsert(self, entries: List[VectorEntry]) -> int:
        if not self.collection:
            raise RuntimeError("Not connected to ChromaDB")
        ids = [e.id or f"entry_{i}" for i, e in enumerate(entries)]
        embeddings = [list(e.vector) for e in entries]
        documents = [e.text for e in entries]
        metadatas = [e.metadata or {} for e in entries]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(entries)

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        if not self.collection:
            raise RuntimeError("Not connected to ChromaDB")
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter_metadata,
        )
        out: List[QueryResult] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                out.append(QueryResult(
                    id=doc_id,
                    score=float(results["distances"][0][i]) if results.get("distances") else 0.0,
                    text=results["documents"][0][i] if results.get("documents") else "",
                    metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                ))
        return out

    async def delete(self, ids: List[str]) -> bool:
        if not self.collection:
            return False
        self.collection.delete(ids=ids)
        return True

    async def count(self) -> int:
        if not self.collection:
            return 0
        return self.collection.count()


async def create_vector_store(provider: str = "in_memory", **kwargs: Any) -> VectorStore:
    """Factory used by ``apps/api/server/main.py`` for ``/vector/*`` endpoints."""
    key = (provider or "in_memory").lower()
    if key in ("in_memory", "memory", "local"):
        dim = int(kwargs.get("dimension", 384))
        store = InMemoryVectorStore(dimension=dim)
        await store.connect()
        return store
    if key == "chromadb":
        store = ChromaDBVectorStore(persist_directory=kwargs.get("persist_directory", "data/vector_store"))
        await store.connect()
        return store
    if key == "pinecone":
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
        f"Use 'in_memory', 'chromadb', or 'pinecone'."
    )


_embed_model: Optional[Any] = None
_EMBED_DIM: int = 384


def _load_embed_model() -> Any:
    """Return the sentence-transformers model if already loaded, else None.

    We do NOT lazy-load it here — loading a second PyTorch model into a
    process that already has Qwen causes OOM on 8 GB Macs.  The caller
    (simple_embed) falls back to the numpy n-gram embedder.
    """
    return _embed_model


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
    """Word-level n-gram embedding with multi-hash feature hashing.

    Extracts word unigrams, bigrams, and trigrams.  Uses 3 independent hash
    seeds per n-gram (feature hashing) to reduce collision effects.  Adds
    character-level trigrams for sub-word overlap.  Log-frequency TF weighting,
    L2-normalized.
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
        raw = ng.encode()
        for seed in range(3):
            h = int(hashlib.md5(raw + bytes([seed])).hexdigest()[:8], 16)
            idx = h % dimension
            vec[idx] += 1.0

    lower = text.lower()
    for i in range(max(0, len(lower) - 3)):
        tri = lower[i:i+3]
        h = int(hashlib.md5(tri.encode()).hexdigest()[:8], 16)
        idx = h % dimension
        vec[idx] += 0.3

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
            return _slo_embedder.embed(text)
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
    "PineconeVectorStore",
    "ChromaDBVectorStore",
    "create_vector_store",
    "simple_embed",
]
