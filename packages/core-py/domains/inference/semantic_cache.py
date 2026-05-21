"""
Semantic Cache - Instant Response Caching

Uses hyperdimensional vectors for fast semantic similarity matching.
Caches query-response pairs for instant retrieval on similar queries.

Architecture:
  Query → HD Encode → Cache Lookup → [Hit: Return cached]
                              ↓
                         [Miss: Generate → Cache → Return]
"""

import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("sloughgpt.semantic_cache")


@dataclass
class CacheEntry:
    """A cached query-response pair."""

    id: str
    query: str
    response: str
    hypervector: List[float]
    metadata: Dict[str, Any]
    timestamp: float
    hit_count: int = 0
    last_accessed: float = 0


class SemanticCache:
    """
    Fast semantic caching using hyperdimensional vectors.

    Features:
    - HD vector encoding for O(1) similarity
    - Configurable similarity threshold
    - TTL-based expiration
    - LRU eviction
    - Hit/miss statistics
    """

    def __init__(
        self,
        dim: int = 10000,
        max_entries: int = 1000,
        similarity_threshold: float = 0.30,
        ttl_seconds: float = 3600,
    ):
        self.dim = dim
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        self.entries: Dict[str, CacheEntry] = {}
        self._hyperdim = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    def _get_hyperdim(self):
        """Lazy-load hyperdimensional processor."""
        if self._hyperdim is None:
            from domains.soul.quantum import HyperdimensionalProcessor

            self._hyperdim = HyperdimensionalProcessor(dim=self.dim)
        return self._hyperdim

    def encode_query(self, query: str) -> List[float]:
        """Encode query as hypervector."""
        hd = self._get_hyperdim()
        return hd.encode_text(query)

    def get(self, query: str) -> Optional[str]:
        """
        Get cached response for query.

        Uses hybrid matching:
        - HD vector similarity (catches paraphrases)
        - Word overlap (catches topic similarity)

        Args:
            query: Input query string

        Returns:
            Cached response if hit, None if miss
        """
        if not self.entries:
            self._stats["misses"] += 1
            return None

        query_vec = self.encode_query(query)
        hd = self._get_hyperdim()
        # Strip punctuation from words
        import re

        query_words = set(re.sub(r"[^\w\s]", "", w.lower()) for w in query.split())

        best_entry: Optional[CacheEntry] = None
        best_score = 0.0
        current_time = time.time()

        for entry in self.entries.values():
            # Check TTL
            if current_time - entry.timestamp > self.ttl_seconds:
                continue

            # HD similarity
            hd_sim = hd.similarity(query_vec, entry.hypervector)

            # Word overlap (Jaccard) - filter stop words
            stop_words = {
                "what",
                "is",
                "a",
                "the",
                "an",
                "of",
                "to",
                "and",
                "or",
                "in",
                "on",
                "at",
                "for",
                "how",
                "do",
                "you",
                "your",
                "it",
                "this",
                "that",
                "can",
                "be",
                "about",
                "tell",
                "me",
                "hello",
            }
            query_content = query_words - stop_words
            entry_words = set(entry.query.lower().split())
            entry_content = entry_words - stop_words

            # Calculate content word match
            common_content = query_content & entry_content

            # Score: primarily based on content word match
            common_content = query_content & entry_content
            score = 0.0

            if common_content:
                # Query content must overlap with entry
                query_len = len(query_content)
                common_ratio = len(common_content) / query_len if query_len > 0 else 0

                # High overlap required (80%)
                if common_ratio >= 0.8:
                    score = 0.85 + common_ratio * 0.1
                elif common_ratio >= 0.5:
                    # Medium overlap - use HD to validate
                    if hd_sim > 0.5:
                        score = 0.6
                elif common_ratio >= 0.3:
                    # Low overlap - require very high HD
                    if hd_sim > 0.7:
                        score = 0.5

            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_entry = entry

        if best_entry:
            best_entry.hit_count += 1
            best_entry.last_accessed = current_time
            self._stats["hits"] += 1
            logger.debug(f"Cache hit: score={best_score:.3f}")
            return best_entry.response

        self._stats["misses"] += 1
        return None

    def put(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store query-response pair in cache.

        Args:
            query: Input query
            response: Generated response
            metadata: Optional metadata (model, params, etc.)

        Returns:
            Cache entry ID
        """
        # Evict expired entries
        self._evict_expired()

        # Evict if full
        if len(self.entries) >= self.max_entries:
            self._evict_lru()

        query_vec = self.encode_query(query)
        entry_id = hashlib.sha256(query.encode()).hexdigest()[:16]

        entry = CacheEntry(
            id=entry_id,
            query=query,
            response=response,
            hypervector=query_vec,
            metadata=metadata or {},
            timestamp=time.time(),
        )

        self.entries[entry_id] = entry
        logger.debug(f"Cached entry: {entry_id}")

        return entry_id

    def _evict_expired(self) -> int:
        """Remove expired entries."""
        current_time = time.time()
        expired = [
            entry_id
            for entry_id, entry in self.entries.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]

        for entry_id in expired:
            del self.entries[entry_id]
            self._stats["expirations"] += 1

        return len(expired)

    def _evict_lru(self) -> bool:
        """Evict least recently accessed entry."""
        if not self.entries:
            return False

        lru_entry = min(
            self.entries.values(),
            key=lambda e: e.last_accessed or e.timestamp,
        )

        del self.entries[lru_entry.id]
        self._stats["evictions"] += 1
        return True

    def invalidate(self, query: str) -> bool:
        """
        Invalidate cache entry for query.

        Args:
            query: Query to invalidate

        Returns:
            True if entry was found and removed
        """
        query_vec = self.encode_query(query)
        hd = self._get_hyperdim()

        for entry_id, entry in list(self.entries.items()):
            sim = hd.similarity(query_vec, entry.hypervector)
            if sim > 0.95:  # Exact match threshold
                del self.entries[entry_id]
                return True

        return False

    def clear(self) -> int:
        """Clear all cache entries."""
        count = len(self.entries)
        self.entries.clear()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            "enabled": True,
            "entries": len(self.entries),
            "max_entries": self.max_entries,
            "dimension": self.dim,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "evictions": self._stats["evictions"],
            "expirations": self._stats["expirations"],
        }

    def reset_stats(self) -> None:
        """Reset hit/miss statistics."""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }


class CachedSoulEngine:
    """
    SloEngine wrapper with semantic caching.

    Usage:
        engine = CachedSoulEngine(soul_engine)
        response = engine.generate("query")  # Uses cache
    """

    def __init__(
        self,
        soul_engine,
        cache: Optional[SemanticCache] = None,
        cache_responses: bool = True,
    ):
        self.engine = soul_engine
        self.cache = cache or SemanticCache()
        self.cache_responses = cache_responses

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with cache lookup and storage."""
        # Check cache first
        cached = self.cache.get(prompt)
        if cached is not None:
            return cached

        # Generate new response
        response = self.engine.generate(prompt, **kwargs)

        # Store in cache if enabled
        if self.cache_responses and response:
            self.cache.put(
                query=prompt,
                response=response,
                metadata={"kwargs": kwargs},
            )

        return response

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self) -> int:
        """Clear the cache."""
        return self.cache.clear()

    def invalidate(self, query: str) -> bool:
        """Invalidate specific cache entry."""
        return self.cache.invalidate(query)


__all__ = ["SemanticCache", "CachedSoulEngine", "CacheEntry"]
