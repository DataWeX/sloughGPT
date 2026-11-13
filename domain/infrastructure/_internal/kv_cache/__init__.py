"""KV state — three layers: base, native, and the cross-session prefix table.

- KVState (base): the default — paged K/V storage + generated-token state +
  session prefix cache (LRU + TTL)
- NativeKVState (native): the concatenating decode state — C (ctypes) with
  an internal numpy fallback
- SessionKVManager: per-session KV state registry with LRU + TTL — delegates
  its storage onto ``KVState`` session mode
- PrefixCache: content-addressed, *session-independent* KV prefix table —
  the same token prefix from a different client or route shares one prefill
  (token-exact by construction; card 20261102_023531)
"""

from __future__ import annotations

from domain.infrastructure._internal.kv_cache.kv_state import KVState
from domain.infrastructure._internal.kv_cache.native import NativeKVState
from domain.infrastructure._internal.kv_cache.prefix_cache import KVSnapshot, PrefixCache
from domain.infrastructure._internal.kv_cache.session import SessionKVManager

__all__ = ["KVSnapshot", "KVState", "NativeKVState", "PrefixCache", "SessionKVManager"]
