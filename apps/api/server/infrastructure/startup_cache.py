"""Startup module cache — caches importlib results for faster subsequent starts.

Usage:
    from infrastructure.startup_cache import get_startup_cache, cached_import

    cache = get_startup_cache()
    module = cached_import("json")
    # Later...
    stats = cache.get_stats()
"""

import importlib
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached module entry."""

    module_name: str
    module: Any
    import_time_ms: float
    cached_at: float
    hit_count: int = 0

    def to_dict(self) -> dict:
        return {
            "module": self.module_name,
            "import_time_ms": round(self.import_time_ms, 2),
            "cached_at": self.cached_at,
            "hit_count": self.hit_count,
        }


@dataclass
class CacheStats:
    """Cache performance statistics."""

    total_hits: int = 0
    total_misses: int = 0
    total_import_time_ms: float = 0.0
    total_cache_time_ms: float = 0.0
    entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(self.hit_rate, 3),
            "total_import_time_ms": round(self.total_import_time_ms, 2),
            "total_cache_time_ms": round(self.total_cache_time_ms, 2),
            "entries": self.entries,
        }


class StartupModuleCache:
    """Caches module imports to speed up startup.

    When a module is imported through cached_import(), it's stored
    in the cache. Subsequent imports return the cached module directly,
    skipping the import machinery.
    """

    def __init__(self, max_size: int = 500) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._stats = CacheStats()

    def get(self, module_name: str) -> Any | None:
        """Get a module from the cache.

        Returns the cached module if available, None otherwise.
        """
        entry = self._cache.get(module_name)
        if entry is not None:
            entry.hit_count += 1
            self._stats.total_hits += 1
            return entry.module
        self._stats.total_misses += 1
        return None

    def put(self, module_name: str, module: Any, import_time_ms: float) -> None:
        """Store a module in the cache."""
        if len(self._cache) >= self._max_size:
            self._evict()

        entry = CacheEntry(
            module_name=module_name,
            module=module,
            import_time_ms=import_time_ms,
            cached_at=time.time(),
        )
        self._cache[module_name] = entry

    def _evict(self) -> None:
        """Evict least recently used entries."""
        if not self._cache:
            return

        # Remove entries with lowest hit count
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].hit_count,
        )
        to_remove = len(self._cache) - self._max_size + 1
        for name, _ in sorted_entries[:to_remove]:
            del self._cache[name]

    def has(self, module_name: str) -> bool:
        """Check if a module is cached."""
        return module_name in self._cache

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.entries = len(self._cache)
        return self._stats

    def get_entries(self) -> list[dict]:
        """Get all cached entries."""
        return [entry.to_dict() for entry in self._cache.values()]

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._stats = CacheStats()


_global_cache: StartupModuleCache | None = None


def get_startup_cache() -> StartupModuleCache:
    """Get the global startup module cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = StartupModuleCache()
    return _global_cache


def cached_import(module_name: str) -> Any:
    """Import a module, using the cache if available.

    Args:
        module_name: The module to import (e.g., "json", "os.path").

    Returns:
        The imported module.
    """
    cache = get_startup_cache()

    # Check cache first
    cached = cache.get(module_name)
    if cached is not None:
        return cached

    # Import the module
    start = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
    except Exception:
        raise
    finally:
        import_time_ms = (time.perf_counter() - start) * 1000
        cache._stats.total_import_time_ms += import_time_ms

    # Cache it
    cache.put(module_name, module, import_time_ms)

    return module
