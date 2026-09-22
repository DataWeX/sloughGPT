"""Thread-safe per-session KV cache for incremental cross-turn decoding.

Thin facade over the unified ``KVCacheBase`` session mode — all storage,
LRU eviction and TTL logic lives once in ``domain.infrastructure._internal.kv_cache``.
"""

from __future__ import annotations

from typing import Any

from domain.infrastructure._internal.kv_cache.base import KVCacheBase


class SessionKVCache:
    """Thread-safe per-session KV cache for incremental cross-turn decoding.

    Stores ``(token_ids, past_key_values)`` per session so that follow-up
    messages can skip re-encoding the shared prompt prefix.

    Cache entries expire after ``ttl`` seconds and at most ``max_sessions``
    entries are kept (LRU eviction). Backed by ``KVCacheBase``.
    """

    def __init__(self, max_sessions: int = 20, ttl: float = 600.0):
        self._max_sessions = max_sessions
        self._ttl = ttl
        self._kv = KVCacheBase(n_layers=1, max_sessions=max_sessions, ttl=ttl)

    def get(self, session_id: str, current_ids: list[int]):
        """Return the cached ``past_key_values`` if ``current_ids`` shares a
        prefix with the stored token IDs, else ``None``.

        Also returns the prefix length so the caller can build the correct
        attention mask.
        """
        return self._kv.session_get(session_id, current_ids)

    def store(self, session_id: str, token_ids: list[int], past_key_values: Any) -> None:
        """Store ``past_key_values`` keyed by session + token IDs."""
        self._kv.session_store(session_id, token_ids, past_key_values)

    def clear(self, session_id: str) -> None:
        self._kv.session_clear(session_id)

    def clear_all(self) -> int:
        """Clear all sessions. Returns the number of sessions cleared."""
        with self._kv._session_lock:
            n = len(self._kv._sessions)
            self._kv._sessions.clear()
        return n

    def evict_expired(self) -> None:
        self._kv._session_evict_expired()

    @property
    def size(self) -> int:
        with self._kv._session_lock:
            return len(self._kv._sessions)

    def stats(self) -> dict:
        with self._kv._session_lock:
            return {
                "entries": len(self._kv._sessions),
                "max_sessions": self._max_sessions,
                "ttl_seconds": self._ttl,
            }
