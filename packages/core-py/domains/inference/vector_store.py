"""
Base ``VectorStore`` ABC, InMemoryVectorStore, embedders, and factory.

External providers (Pinecone, ChromaDB) live in the ``vector_stores/`` package.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("slo.inference.vector_store")

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
        pass  # pragma: no cover (abstractmethod body)

    @abstractmethod
    async def disconnect(self) -> None:
        pass  # pragma: no cover (abstractmethod body)

    @abstractmethod
    async def upsert(self, entries: List[VectorEntry]) -> int:
        pass  # pragma: no cover (abstractmethod body)

    @abstractmethod
    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        pass  # pragma: no cover (abstractmethod body)

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        pass  # pragma: no cover (abstractmethod body)

    @abstractmethod
    async def count(self) -> int:
        pass  # pragma: no cover (abstractmethod body)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def _build_matrix(entries: Dict[str, VectorEntry], ids: List[str]) -> np.ndarray:
    """Stack ``entries[eid].vector`` for every id into one (N, dim) matrix.

    A single ``np.array(list_of_lists)`` builds the 2-D array in one C pass,
    avoiding per-row ``np.asarray`` overhead.
    """
    return np.array([entries[eid].vector for eid in ids], dtype=np.float64)


def _matrix_norms(mat: np.ndarray) -> np.ndarray:
    """Row L2 norms of *mat*; O(N*dim) once, cached alongside the matrix."""
    return np.linalg.norm(mat, axis=1)


def _rank_matrix(
    mat: np.ndarray,
    ids: List[str],
    entries: Dict[str, VectorEntry],
    q: np.ndarray,
    top_k: int,
    norms: Optional[np.ndarray] = None,
) -> List[QueryResult]:
    """Score a query against an (N, dim) matrix aligned with *ids*.

    Cosine scores are ``dot(a, b) / (|a||b| + 1e-10)`` computed with a single
    matrix-vector product. Ties keep *ids* order via stable argsort, mirroring
    the old ``sorted(..., reverse=True)`` loop.

    Args:
        mat: (N, dim) float64 matrix; row i aligns with ``ids[i]``
        ids: entry ids in matrix row order
        entries: id -> entry mapping (for result payloads)
        q: query vector (float64 array)
        top_k: max results to return
        norms: precomputed row L2 norms of *mat* (from ``_matrix_norms``).
            When omitted the norm is recomputed here — callers that reuse a
            matrix across queries should pass it to skip the O(N*dim) pass.

    Returns:
        top_k QueryResults ordered by descending score
    """
    q_norm = np.linalg.norm(q)
    if norms is None:
        norms = _matrix_norms(mat)
    denom = norms * q_norm + 1e-10
    scores = (mat @ q) / denom
    order = np.argsort(-scores, kind="stable")
    out: List[QueryResult] = []
    for idx in order[:top_k]:
        i = int(idx)
        entry = entries[ids[i]]
        out.append(QueryResult(
            id=entry.id,
            score=float(scores[i]),
            text=entry.text,
            metadata=dict(entry.metadata),
        ))
    return out


class InMemoryVectorStore(VectorStore):
    """Simple cosine-similarity store for development and tests."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._entries: Dict[str, VectorEntry] = {}
        # Full-matrix cache: (version, ids, mat). _version bumps on every
        # mutation; a query rebuilds when the cached version is stale, so a
        # tuple assigned concurrently with an upsert self-heals on next read
        # (no locks needed — tuple reads are GIL-atomic).
        self._version = 0
        self._matrix_cache: Optional[tuple] = None

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    def _cached_matrix(self) -> tuple:
        """Return (ids, mat, norms) for all entries, rebuilding when stale.

        Row norms are computed once per rebuild so repeated queries skip the
        O(N*dim) ``np.linalg.norm`` pass (recomputed only on mutation).
        """
        cache = self._matrix_cache
        if cache is None or cache[0] != self._version:
            ids = list(self._entries.keys())
            mat = _build_matrix(self._entries, ids)
            cache = (self._version, ids, mat, _matrix_norms(mat))
            self._matrix_cache = cache
        return cache[1], cache[2], cache[3]

    def query_sync(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Synchronous query — avoids event-loop deadlock from _run_async.

        Unfiltered queries reuse a cached (ids, matrix, norms) so repeated
        searches skip the per-query Python matrix build and row-norm pass.
        Filtered queries build on demand. The cache is version-guarded
        against concurrent upserts.
        """
        if not self._entries:
            return []
        q = np.asarray(vector, dtype=np.float64)
        if filter_metadata:
            ids = [
                eid for eid, e in self._entries.items()
                if e.metadata and all(e.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
            if not ids:
                return []
            mat = _build_matrix(self._entries, ids)
            norms = _matrix_norms(mat)
        else:
            ids, mat, norms = self._cached_matrix()
        return _rank_matrix(mat, ids, self._entries, q, top_k, norms=norms)

    def _bump_version(self) -> None:
        """Invalidate the matrix cache after any mutation."""
        self._version += 1
        self._matrix_cache = None

    def upsert_sync(self, entries: List[VectorEntry]) -> int:
        """Synchronous upsert — avoids event-loop deadlock from _run_async."""
        for e in entries:
            self._entries[e.id] = e
        self._bump_version()
        return len(entries)

    def count_sync(self) -> int:
        """Synchronous count — avoids event-loop deadlock from _run_async."""
        return len(self._entries)

    async def upsert(self, entries: List[VectorEntry]) -> int:
        for e in entries:
            self._entries[e.id] = e
        self._bump_version()
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
        if removed:
            self._bump_version()
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
        # Full-matrix cache: (version, ids, mat). Version-guarded against
        # concurrent mutation (see InMemoryVectorStore).
        self._version = 0
        self._matrix_cache: Optional[tuple] = None

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
        logger.info("MogDBVectorStore loaded %d entries from %s", len(self._entries), self._path, extra={"tag": "INF"})
        return True

    async def disconnect(self) -> None:
        if self._mogdb:
            self._mogdb.close()
            self._mogdb = None
            self._coll = None

    def _cached_matrix(self) -> tuple:
        """Return (ids, mat, norms) for all entries, rebuilding when stale.

        Row norms are computed once per rebuild so repeated queries skip the
        O(N*dim) ``np.linalg.norm`` pass (recomputed only on mutation).
        """
        cache = self._matrix_cache
        if cache is None or cache[0] != self._version:
            ids = list(self._entries.keys())
            mat = _build_matrix(self._entries, ids)
            cache = (self._version, ids, mat, _matrix_norms(mat))
            self._matrix_cache = cache
        return cache[1], cache[2], cache[3]

    def _bump_version(self) -> None:
        """Invalidate the matrix cache after any mutation."""
        self._version += 1
        self._matrix_cache = None

    def query_sync(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Synchronous query over in-memory entries (cached matrix)."""
        if not self._entries:
            return []
        q = np.asarray(vector, dtype=np.float64)
        if filter_metadata:
            ids = [
                eid for eid, e in self._entries.items()
                if e.metadata and all(e.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
            if not ids:
                return []
            mat = _build_matrix(self._entries, ids)
            norms = _matrix_norms(mat)
        else:
            ids, mat, norms = self._cached_matrix()
        return _rank_matrix(mat, ids, self._entries, q, top_k, norms=norms)

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
        self._bump_version()
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
        if removed:
            self._bump_version()
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
_EMBED_MODEL_NAME: str = "all-MiniLM-L6-v2"
_EMBED_MIN_MEMORY_MB: int = 500


def _load_embed_model() -> Any:
    """Lazy-load a sentence-transformers model for semantic embeddings.

    Auto-downloads all-MiniLM-L6-v2 (~80 MB) on first use. Checks available
    memory before loading to avoid OOM when a larger model is already in RAM.
    Falls back to fast n-gram TF-IDF embedder on any failure.

    Returns:
        SentenceTransformer model or None (falls back to n-gram embedder).
    """
    global _embed_model, _EMBED_LOAD_FAILED
    if _EMBED_LOAD_FAILED:
        return None
    if _embed_model is not None:
        return _embed_model

    # Check available memory before loading
    try:
        import psutil
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        if avail_mb < _EMBED_MIN_MEMORY_MB:
            logger.warning(
                "Embed model skipped: only %.0f MB available (need %d MB)",
                avail_mb, _EMBED_MIN_MEMORY_MB,
                extra={"tag": "INF"},
            )
            _EMBED_LOAD_FAILED = True
            return None
    except ImportError:
        pass  # psutil not installed — proceed without memory check

    # Try importing sentence-transformers (requires torch)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info(
            "sentence-transformers not installed; using n-gram embedder. "
            "Install with: pip install sentence-transformers",
            extra={"tag": "INF"},
        )
        _EMBED_LOAD_FAILED = True
        return None

    # Load model on CPU (auto-downloads on first run)
    try:
        logger.info("Loading embedding model %s (device=cpu)...", _EMBED_MODEL_NAME, extra={"tag": "INF"})
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME, device="cpu")
        logger.info("Embedding model loaded (%d-dimensional, device=cpu)", _EMBED_DIM, extra={"tag": "INF"})
        return _embed_model
    except Exception as exc:
        logger.warning("Failed to load embedding model: %s — using n-gram fallback", exc, extra={"tag": "INF"})
        _EMBED_LOAD_FAILED = True
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
_slo_embedder_untrained = False


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
    global _slo_embedder, _slo_embedder_untrained
    if _slo_embedder is None and not _slo_embedder_untrained:
        try:
            from domains.inference.slo_embedder import SloTextEmbedder
            candidate = SloTextEmbedder.load()
            if candidate is not None and not candidate.acceptable():
                logger.info(
                    "SloNet embedder rejected by quality gate (%s), using n-gram fallback",
                    candidate.quality,
                )
                _slo_embedder_untrained = True
            elif candidate is not None:
                _slo_embedder = candidate
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
