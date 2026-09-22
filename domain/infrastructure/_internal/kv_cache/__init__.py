"""Unified KV cache — single base with paged + session support.

Backends:
  - KVCacheBase: C-backed concat, paged blocks, session LRU — all in one
  - SessionKVManager: per-session KV state registry with LRU + TTL
  - NativeKVCache: ctypes wrapper for C transformer library
"""

from domain.infrastructure._internal.kv_cache.base import KVCacheBase
from domain.infrastructure._internal.kv_cache.native import NativeKVCache
from domain.infrastructure._internal.kv_cache.session import SessionKVManager

__all__ = ["KVCacheBase", "NativeKVCache", "SessionKVManager"]
