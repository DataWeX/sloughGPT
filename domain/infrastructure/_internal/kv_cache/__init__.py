"""Unified KV cache.

- KVCache: session (LRU + TTL prefix cache) + paged block cache
- SessionKVManager: per-session KV state registry with LRU + TTL (lives
  on the session manager) — delegates into KVCache semantics
- NativeKVCache: concatenating decode cache — C (ctypes) + numpy fallback
- NumpyKVCache: the pure-numpy concatenating backend of NativeKVCache
"""

from domain.infrastructure._internal.kv_cache.kv_cache import KVCache
from domain.infrastructure._internal.kv_cache.native import NativeKVCache, NumpyKVCache
from domain.infrastructure._internal.kv_cache.session import SessionKVManager

__all__ = ["KVCache", "NativeKVCache", "NumpyKVCache", "SessionKVManager"]
