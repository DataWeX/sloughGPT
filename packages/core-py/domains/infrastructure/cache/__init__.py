"""
Cache Manager Implementation

This module provides advanced caching capabilities with multiple backends.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...__init__ import BaseComponent, ComponentException, ICacheManager


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    ttl: Optional[int]
    created_at: float
    accessed_count: int
    last_accessed: float


class CacheManager(BaseComponent, ICacheManager):
    """Advanced cache management system"""

    def __init__(self) -> None:
        super().__init__("cache_manager")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}

        # Configuration
        self.max_size = 10000
        self.default_ttl = 3600  # 1 hour
        self.cleanup_task: Optional[asyncio.Task[Any]] = None
        self.cleanup_interval = 300  # 5 minutes

        # Statistics
        stats_dict: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "hit_rate": 0.0,
            "cache_size": 0,
            "max_size": 0,
            "utilization": 0.0,
        }
        self.stats = stats_dict

        # Background tasks
        self.cleanup_task = None

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize cache manager"""
        try:
            self.logger.info("Initializing Cache Manager...")

            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            self.is_initialized = True
            self.logger.info("Cache Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Cache Manager: {e}")
            raise ComponentException(f"Cache Manager initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown cache manager"""
        try:
            self.logger.info("Shutting down Cache Manager...")

            # Cancel cleanup task
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass

            self.is_initialized = False
            self.logger.info("Cache Manager shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Cache Manager: {e}")
            raise ComponentException(f"Cache Manager shutdown failed: {e}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            entry = self.cache[key]

            # Check TTL
            if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
                await self.delete(key)
                self.stats["misses"] += 1
                return None

            # Update access stats
            entry.accessed_count += 1
            entry.last_accessed = time.time()

            self.stats["hits"] += 1
            self.logger.debug(f"Cache hit for key: {key}")
            return entry.value

        except Exception as e:
            self.logger.error(f"Failed to get cache key {key}: {e}")
            self.stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            # Check cache size limit
            if len(self.cache) >= self.max_size:
                await self._evict_lru()

            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                created_at=time.time(),
                accessed_count=1,
                last_accessed=time.time(),
            )

            self.cache[key] = entry
            self.stats["sets"] += 1

            self.logger.debug(f"Cache set for key: {key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                self.stats["deletes"] += 1
                self.logger.debug(f"Cache delete for key: {key}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            self.cache.clear()
            self.logger.info("Cache cleared")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.stats.copy()

        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0.0

        # Add current cache info
        stats["cache_size"] = len(self.cache)
        stats["max_size"] = self.max_size
        stats["utilization"] = len(self.cache) / self.max_size

        return stats

    # Private helper methods

    async def _evict_lru(self) -> None:
        """Evict least recently used entries"""
        if not self.cache:
            return

        # Find LRU entry
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)

        del self.cache[lru_key]
        self.stats["evictions"] += 1

        self.logger.debug(f"Evicted LRU entry: {lru_key}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired entries"""
        while self.is_initialized:
            try:
                await self._cleanup_expired_entries()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_entries(self) -> None:
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.cache.items():
            if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]
            self.stats["evictions"] += 1

        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


__all__ = ["CacheEntry", "CacheManager"]

