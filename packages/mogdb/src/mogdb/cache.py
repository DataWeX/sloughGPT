"""Simple TTL-based query cache for frequent reads.

Usage::

    from mogdb.cache import QueryCache

    cache = QueryCache(ttl_seconds=5.0)

    # Cache a query result
    result = cache.get_or_set("users:all", lambda: users_col.find())

    # Invalidate specific keys or patterns
    cache.invalidate("users:all")
    cache.invalidate_pattern("users:*")

    # Clear everything
    cache.clear()
"""

import threading
import time
from typing import Any, Callable, Optional


class QueryCache:
    """Thread-safe TTL cache for query results.

    Parameters
    ----------
    ttl_seconds:
        Time-to-live for cached entries. None = no expiry.
    max_entries:
        Maximum number of cached entries (0 = unlimited).
    """

    def __init__(
        self,
        ttl_seconds: float = 5.0,
        max_entries: int = 256,
    ):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value by key. Returns None if missing or expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if self._ttl and (time.monotonic() - ts) > self._ttl:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        if self._max and len(self._cache) >= self._max:
            self._evict_oldest()
        self._cache[key] = (time.monotonic(), value)

    def get_or_set(self, key: str, factory: Callable[[], Any]) -> Any:
        """Get from cache, or compute via factory and store the result."""
        value = self.get(key)
        if value is not None:
            return value
        value = factory()
        self.set(key, value)
        return value

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        with self._lock:
            return self._cache.pop(key, None) is not None

    def invalidate_pattern(self, pattern: str) -> int:
        """Remove all keys matching a prefix (e.g. ``users:*``).

        The ``*`` suffix means "starts with the prefix before *".
        Returns the number of entries removed.
        """
        prefix = pattern.rstrip("*")
        with self._lock:
            to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in to_remove:
                del self._cache[k]
            return len(to_remove)

    def clear(self) -> int:
        """Clear all cached entries. Returns the number removed."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max,
                "ttl_seconds": self._ttl,
            }

    def _evict_oldest(self) -> None:
        """Evict the oldest entry when at capacity."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
        del self._cache[oldest_key]


# Module-level singleton for convenience
_default_cache: Optional[QueryCache] = None
_default_lock = threading.Lock()


def get_query_cache() -> QueryCache:
    """Get or create the default query cache singleton."""
    global _default_cache
    if _default_cache is None:
        with _default_lock:
            if _default_cache is None:
                _default_cache = QueryCache()
    return _default_cache
