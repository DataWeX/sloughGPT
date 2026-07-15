"""
Tiered cache — Disk → Hot → Memory.

Manages data across three storage tiers:
  - Disk: cold storage (files, slow but persistent)
  - Hot: fast local cache (SSD/Redis, faster than disk)
  - Memory: in-memory (RAM, fastest but volatile)

Automatically promotes/demotes data based on access patterns.
"""

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger("slo.pugqeep")


class EvictionPolicy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used


class Tier(Enum):
    """Storage tiers from coldest to hottest."""
    DISK = "disk"
    HOT = "hot"
    MEMORY = "memory"


@dataclass
class CacheEntry:
    """A cached item with metadata."""
    key: str
    tier: Tier
    data: Any = None
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    pinned: bool = False  # don't evict pinned entries
    ttl: Optional[float] = None  # time-to-live in seconds, None = forever

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    promotions: int = 0
    demotions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class Store(Protocol):
    """Protocol for storage backends."""
    def get(self, key: str) -> Optional[Any]: ...
    def put(self, key: str, value: Any) -> None: ...
    def remove(self, key: str) -> bool: ...
    def list_keys(self) -> List[str]: ...
    def exists(self, key: str) -> bool: ...
    def size_bytes(self) -> int: ...


class MemoryStore:
    """In-memory LRU cache with size limit."""

    def __init__(self, max_size_bytes: int = 512 * 1024 * 1024):  # 512MB default
        self._max_size = max_size_bytes
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._sizes: Dict[str, int] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: Any, size_bytes: int = 0) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        self._sizes[key] = size_bytes

    def remove(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._sizes.pop(key, None)
            return True
        return False

    def list_keys(self) -> List[str]:
        return list(self._data.keys())

    def exists(self, key: str) -> bool:
        return key in self._data

    def size_bytes(self) -> int:
        return sum(self._sizes.values())

    def evict_lru(self, target_bytes: int) -> List[str]:
        """Evict least-recently-used entries to free space."""
        evicted = []
        while self.size_bytes() > self._max_size - target_bytes and self._data:
            key, _ = self._data.popitem(last=False)
            evicted.append(key)
            self._sizes.pop(key, None)
        return evicted

    def evict_lfu(self, target_bytes: int, access_counts: Dict[str, int]) -> List[str]:
        """Evict least-frequently-used entries to free space.

        Args:
            target_bytes: Target bytes to free.
            access_counts: Dict of key → access_count from CacheEntry.
        """
        if self.size_bytes() <= self._max_size - target_bytes:
            return []
        # Sort keys by access count (ascending), evict lowest first
        keys_by_freq = sorted(self._data.keys(), key=lambda k: access_counts.get(k, 0))
        evicted = []
        for key in keys_by_freq:
            if self.size_bytes() <= self._max_size - target_bytes:
                break
            self._data.pop(key, None)
            self._sizes.pop(key, None)
            evicted.append(key)
        return evicted


class DiskStore:
    """Disk-backed store using JSON + numpy arrays."""

    def __init__(self, directory: Path):
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.meta.json"

    def _data_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.data.npy"

    def get(self, key: str) -> Optional[Any]:
        meta_path = self._meta_path(key)
        data_path = self._data_path(key)
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        if data_path.exists():
            return np.load(data_path, allow_pickle=False)
        return meta.get("value")

    def put(self, key: str, value: Any, meta: Optional[dict] = None) -> None:
        if isinstance(value, np.ndarray):
            np.save(self._data_path(key), value)
            meta_data = {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            meta_data = {"type": "json", "value": value}
        if meta:
            meta_data.update(meta)
        self._meta_path(key).write_text(json.dumps(meta_data))

    def remove(self, key: str) -> bool:
        removed = False
        if self._meta_path(key).exists():
            self._meta_path(key).unlink()
            removed = True
        if self._data_path(key).exists():
            self._data_path(key).unlink()
            removed = True
        return removed

    def list_keys(self) -> List[str]:
        return [f.stem.replace(".meta", "") for f in self._dir.glob("*.meta.json")]

    def exists(self, key: str) -> bool:
        return self._meta_path(key).exists()

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self._dir.glob("*") if f.is_file())


class HotStore:
    """Hot cache — fast local storage (could back with Redis later)."""

    def __init__(self, max_size_bytes: int = 128 * 1024 * 1024):  # 128MB default
        self._inner = MemoryStore(max_size_bytes)

    def get(self, key: str) -> Optional[Any]:
        return self._inner.get(key)

    def put(self, key: str, value: Any, size_bytes: int = 0) -> None:
        self._inner.put(key, value, size_bytes)

    def remove(self, key: str) -> bool:
        return self._inner.remove(key)

    def list_keys(self) -> List[str]:
        return self._inner.list_keys()

    def exists(self, key: str) -> bool:
        return self._inner.exists(key)

    def size_bytes(self) -> int:
        return self._inner.size_bytes()


class TieredCache:
    """Three-tier cache: Disk → Hot → Memory.

    Manages promotion/demotion of data across tiers based on access patterns.
    Enforces size limits per tier with configurable eviction (LRU or LFU).
    """

    def __init__(self,
                 memory_max_mb: int = 512,
                 hot_max_mb: int = 128,
                 disk_dir: Optional[Path] = None,
                 promote_threshold: int = 3,
                 auto_promote: bool = True,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        """Initialize tiered cache.

        Args:
            memory_max_mb: Max memory cache size in MB.
            hot_max_mb: Max hot cache size in MB.
            disk_dir: Directory for disk storage. None = disk disabled.
            promote_threshold: Access count before promoting to hotter tier.
            auto_promote: Automatically promote frequently accessed data.
            eviction_policy: LRU or LFU eviction strategy.
        """
        self._memory = MemoryStore(memory_max_mb * 1024 * 1024)
        self._hot = HotStore(hot_max_mb * 1024 * 1024)
        self._disk = DiskStore(disk_dir) if disk_dir else None
        self._entries: Dict[str, CacheEntry] = {}
        self._promote_threshold = promote_threshold
        self._auto_promote = auto_promote
        self._eviction_policy = eviction_policy
        self._stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """Get data, promoting to hotter tier if needed."""
        entry = self._entries.get(key)
        if entry is None:
            self._stats.misses += 1
            return None

        # Check TTL expiration
        if entry.is_expired():
            self.remove(key)
            self._stats.misses += 1
            return None

        # Try memory first
        data = self._memory.get(key)
        if data is not None:
            entry.tier = Tier.MEMORY
            entry.touch()
            self._stats.hits += 1
            return data

        # Try hot cache
        data = self._hot.get(key)
        if data is not None:
            entry.tier = Tier.HOT
            entry.touch()
            self._stats.hits += 1
            # Auto-promote to memory if hot enough
            if self._auto_promote and entry.access_count >= self._promote_threshold:
                self._promote(key, data, Tier.MEMORY)
            return data

        # Try disk
        if self._disk:
            data = self._disk.get(key)
            if data is not None:
                entry.tier = Tier.DISK
                entry.touch()
                self._stats.hits += 1
                # Auto-promote to hot if accessed enough
                if self._auto_promote and entry.access_count >= self._promote_threshold:
                    self._promote(key, data, Tier.HOT)
                return data

        self._stats.misses += 1
        return None

    def put(self, key: str, value: Any, tier: Tier = Tier.MEMORY,
            size_bytes: int = 0, pinned: bool = False,
            ttl: Optional[float] = None) -> None:
        """Store data at the specified tier.

        Args:
            key: Cache key.
            value: Data to store.
            tier: Storage tier.
            size_bytes: Approximate size in bytes.
            pinned: If True, won't be evicted.
            ttl: Time-to-live in seconds. None = no expiration.
        """
        if size_bytes == 0 and isinstance(value, np.ndarray):
            size_bytes = value.nbytes

        entry = CacheEntry(
            key=key,
            tier=tier,
            size_bytes=size_bytes,
            pinned=pinned,
            ttl=ttl,
        )
        self._entries[key] = entry

        if tier == Tier.MEMORY:
            self._memory.put(key, value, size_bytes)
            self._maybe_evict_memory()
        elif tier == Tier.HOT:
            self._hot.put(key, value, size_bytes)
            self._maybe_evict_hot()
        elif tier == Tier.DISK and self._disk:
            self._disk.put(key, value)

    def remove(self, key: str) -> bool:
        """Remove data from all tiers."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return False

        self._memory.remove(key)
        self._hot.remove(key)
        if self._disk:
            self._disk.remove(key)
        return True

    def exists(self, key: str) -> bool:
        return key in self._entries

    def list_keys(self, tier: Optional[Tier] = None) -> List[str]:
        """List keys, optionally filtered by tier."""
        if tier is None:
            return list(self._entries.keys())
        return [k for k, e in self._entries.items() if e.tier == tier]

    def stats(self) -> dict:
        """Cache statistics."""
        tier_counts = {}
        tier_sizes = {}
        for entry in self._entries.values():
            t = entry.tier.value
            tier_counts[t] = tier_counts.get(t, 0) + 1
            tier_sizes[t] = tier_sizes.get(t, 0) + entry.size_bytes

        # Count expired entries
        expired = sum(1 for e in self._entries.values() if e.is_expired())

        return {
            "total_entries": len(self._entries),
            "expired_entries": expired,
            "eviction_policy": self._eviction_policy.value,
            "tier_counts": tier_counts,
            "tier_sizes": tier_sizes,
            "memory_size": self._memory.size_bytes(),
            "memory_max": self._memory._max_size,
            "hot_size": self._hot.size_bytes(),
            "hot_max": self._hot._inner._max_size,
            "disk_size": self._disk.size_bytes() if self._disk else 0,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": self._stats.hit_rate,
            "evictions": self._stats.evictions,
            "promotions": self._stats.promotions,
            "demotions": self._stats.demotions,
        }

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        expired_keys = [k for k, e in self._entries.items() if e.is_expired()]
        for key in expired_keys:
            self.remove(key)
        return len(expired_keys)

    def _promote(self, key: str, data: Any, target_tier: Tier) -> None:
        """Promote data to a hotter tier."""
        entry = self._entries.get(key)
        if entry is None or entry.pinned:
            return

        old_tier = entry.tier
        if target_tier == Tier.MEMORY:
            self._memory.put(key, data, entry.size_bytes)
            entry.tier = Tier.MEMORY
        elif target_tier == Tier.HOT:
            self._hot.put(key, data, entry.size_bytes)
            entry.tier = Tier.HOT

        self._stats.promotions += 1
        logger.debug("Promoted %s: %s → %s", key, old_tier.value, target_tier.value)

    def evict(self, target_tier: Tier, target_bytes: int) -> int:
        """Evict entries from a tier to free space. Returns bytes freed."""
        freed = 0

        if target_tier == Tier.MEMORY:
            evicted = self._memory.evict_lru(target_bytes)
            for key in evicted:
                entry = self._entries.get(key)
                if entry:
                    # Demote to hot
                    self._hot.put(key, None, entry.size_bytes)
                    entry.tier = Tier.HOT
                    self._stats.demotions += 1
                    freed += entry.size_bytes
                    self._stats.evictions += 1

        return freed

    def _maybe_evict_memory(self) -> None:
        """Auto-evict from memory when over capacity."""
        max_bytes = self._memory._max_size
        current = self._memory.size_bytes()
        if current <= max_bytes:
            return
        # Need to free ~10% headroom
        target = max_bytes // 10
        if self._eviction_policy == EvictionPolicy.LFU:
            counts = {k: e.access_count for k, e in self._entries.items()
                      if e.tier == Tier.MEMORY and not e.pinned}
            evicted = self._memory.evict_lfu(target, counts)
        else:
            evicted = self._memory.evict_lru(target)
        for key in evicted:
            entry = self._entries.get(key)
            if entry and not entry.pinned:
                data = None  # data was removed from MemoryStore
                if self._disk:
                    self._disk.put(key, data, meta={"demoted_from": "memory"})
                    entry.tier = Tier.DISK
                else:
                    self._hot.put(key, data, entry.size_bytes)
                    entry.tier = Tier.HOT
                self._stats.demotions += 1
                self._stats.evictions += 1

    def _maybe_evict_hot(self) -> None:
        """Auto-evict from hot when over capacity."""
        max_bytes = self._hot._inner._max_size
        current = self._hot.size_bytes()
        if current <= max_bytes:
            return
        target = max_bytes // 10
        if self._eviction_policy == EvictionPolicy.LFU:
            counts = {k: e.access_count for k, e in self._entries.items()
                      if e.tier == Tier.HOT and not e.pinned}
            evicted = self._hot._inner.evict_lfu(target, counts)
        else:
            evicted = self._hot._inner.evict_lru(target)
        for key in evicted:
            entry = self._entries.get(key)
            if entry and not entry.pinned:
                if self._disk:
                    self._disk.put(key, None, meta={"demoted_from": "hot"})
                    entry.tier = Tier.DISK
                self._stats.demotions += 1
                self._stats.evictions += 1
