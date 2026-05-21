"""
SloughGPT SDK - Cache
Response caching for the SloughGPT SDK.
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, Any
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """A cached response entry."""
    key: str
    value: Any
    created_at: float
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl


class InMemoryCache:
    """
    In-memory cache for responses.
    
    Example:
    
    ```python
    from sloughgpt_sdk.cache import InMemoryCache
    
    cache = InMemoryCache(ttl=3600)  # 1 hour TTL
    cache.set("key", "value")
    value = cache.get("key")
    ```
    """
    
    def __init__(self, ttl: Optional[float] = None, max_size: int = 1000):
        """
        Initialize cache.
        
        Args:
            ttl: Default TTL in seconds.
            max_size: Maximum number of entries.
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._ttl = ttl
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            self._misses += 1
            return None
        
        entry = self._cache[key]
        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None
        
        self._hits += 1
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl or self._ttl,
        )
    
    def delete(self, key: str):
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
    
    def clear(self):
        """Clear all cached values."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def _evict_oldest(self):
        """Evict oldest entry."""
        if not self._cache:
            return
        
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "hit_rate": hit_rate,
        }


class DiskCache:
    """
    Disk-based cache for responses.
    
    Example:
    
    ```python
    from sloughgpt_sdk.cache import DiskCache
    
    cache = DiskCache(cache_dir="./.cache", ttl=86400)  # 24 hour TTL
    cache.set("key", {"result": "value"})
    value = cache.get("key")
    ```
    """
    
    def __init__(
        self,
        cache_dir: str = "./.sloughgpt_cache",
        ttl: Optional[float] = None,
        max_size_mb: int = 100,
    ):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files.
            ttl: Default TTL in seconds.
            max_size_mb: Maximum cache size in MB.
        """
        import os
        import struct
        
        self._cache_dir = cache_dir
        self._ttl = ttl
        self._max_size_mb = max_size_mb
        self._hits = 0
        self._misses = 0
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_path(self, key: str) -> str:
        """Get file path for key."""
        return f"{self._cache_dir}/{key}.json"
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        import os
        
        path = self._get_path(key)
        if not os.path.exists(path):
            self._misses += 1
            return None
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            if "expires_at" in data:
                if time.time() > data["expires_at"]:
                    os.remove(path)
                    self._misses += 1
                    return None
            
            self._hits += 1
            return data.get("value")
        except (json.JSONDecodeError, IOError):
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        import os
        
        ttl = ttl or self._ttl
        data = {"value": value}
        
        if ttl:
            data["expires_at"] = time.time() + ttl
        
        path = self._get_path(key)
        with open(path, "w") as f:
            json.dump(data, f)
        
        self._check_size()
    
    def delete(self, key: str):
        """Delete value from cache."""
        import os
        path = self._get_path(key)
        if os.path.exists(path):
            os.remove(path)
    
    def clear(self):
        """Clear all cached values."""
        import os
        for filename in os.listdir(self._cache_dir):
            if filename.endswith(".json"):
                os.remove(os.path.join(self._cache_dir, filename))
        self._hits = 0
        self._misses = 0
    
    def _check_size(self):
        """Check and enforce size limit."""
        import os
        import shutil
        
        total_size = sum(
            os.path.getsize(os.path.join(self._cache_dir, f))
            for f in os.listdir(self._cache_dir)
            if f.endswith(".json")
        )
        
        if total_size > self._max_size_mb * 1024 * 1024:
            files = sorted(
                os.listdir(self._cache_dir),
                key=lambda f: os.path.getmtime(os.path.join(self._cache_dir, f))
            )
            for filename in files:
                if total_size <= self._max_size_mb * 1024 * 1024 * 0.8:
                    break
                filepath = os.path.join(self._cache_dir, filename)
                total_size -= os.path.getsize(filepath)
                os.remove(filepath)
    
    @property
    def size(self) -> int:
        """Get number of cached items."""
        import os
        return len([f for f in os.listdir(self._cache_dir) if f.endswith(".json")])
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        import os
        
        total_size = sum(
            os.path.getsize(os.path.join(self._cache_dir, f))
            for f in os.listdir(self._cache_dir)
            if f.endswith(".json")
        )
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": self.size,
            "size_bytes": total_size,
            "hit_rate": hit_rate,
        }


def cached(ttl: Optional[float] = None, cache: Optional[InMemoryCache] = None):
    """
    Decorator for caching function results.
    
    Example:
    
    ```python
    from sloughgpt_sdk.cache import cached, InMemoryCache
    
    cache = InMemoryCache(ttl=3600)
    
    @cached(ttl=3600, cache=cache)
    def expensive_operation(param):
        return do_work(param)
    ```
    """
    _cache = cache or InMemoryCache(ttl=ttl)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = _cache._generate_key(func.__name__, *args, **kwargs)
            result = _cache.get(key)
            
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            _cache.set(key, result, ttl=ttl)
            return result
        
        wrapper.cache = _cache
        wrapper.cache_clear = _cache.clear
        wrapper.cache_stats = _cache.stats
        return wrapper
    
    return decorator
