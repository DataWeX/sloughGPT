"""Unified KV cache.

- KVCacheBase: session (LRU + TTL prefix cache) + paged block cache
- SessionKVManager: per-session KV state registry with LRU + TTL (lives
  on the session manager) — delegates into KVCacheBase semantics
- NativeKVCache: concatenating decode cache — C (ctypes) + numpy fallback
- NumpyKVCache: the pure-numpy concatenating backend of NativeKVCache
"""

from domain.infrastructure._internal.kv_cache.base import KVCacheBase
from domain.infrastructure._internal.kv_cache.native import NativeKVCache, NumpyKVCache
from domain.infrastructure._internal.kv_cache.session import SessionKVManager

__all__ = ["KVCacheBase", "NativeKVCache", "NumpyKVCache", "SessionKVManager"]
