"""KV state — two layers: base and native.

- KVState (base): the default — paged K/V storage + generated-token state +
  session prefix cache (LRU + TTL)
- NativeKVState (native): the concatenating decode state — C (ctypes) with
  an internal numpy fallback
- SessionKVManager: per-session KV state registry with LRU + TTL — delegates
  its storage onto ``KVState`` session mode
"""

from domain.infrastructure._internal.kv_cache.kv_state import KVState
from domain.infrastructure._internal.kv_cache.native import NativeKVState
from domain.infrastructure._internal.kv_cache.session import SessionKVManager

__all__ = ["KVState", "NativeKVState", "SessionKVManager"]
