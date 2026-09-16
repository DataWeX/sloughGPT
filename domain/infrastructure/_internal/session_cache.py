"""Thread-safe per-session KV cache for incremental cross-turn decoding."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any


class SessionKVCache:
    """Thread-safe per-session KV cache for incremental cross-turn decoding.

    Stores ``(token_ids, past_key_values)`` per session so that follow-up
    messages can skip re-encoding the shared prompt prefix.

    Cache entries expire after ``ttl`` seconds and at most ``max_sessions``
    entries are kept (LRU eviction).
    """

    def __init__(self, max_sessions: int = 20, ttl: float = 600.0):
        self._caches: dict[str, Any] = {}
        self._max_sessions = max_sessions
        self._ttl = ttl
        self._lock = Lock()

    def get(self, session_id: str, current_ids: list[int]):
        """Return the cached ``past_key_values`` if ``current_ids`` shares a
        prefix with the stored token IDs, else ``None``.

        Also returns the prefix length so the caller can build the correct
        attention mask.
        """
        with self._lock:
            entry = self._caches.get(session_id)
        if entry is None:
            return None, 0
        cached_ids, cached_pkv, _ = entry
        prefix_len = 0
        for a, b in zip(cached_ids, current_ids, strict=False):
            if a != b:
                break
            prefix_len += 1
        if prefix_len == 0:
            return None, 0
        return cached_pkv, prefix_len

    def store(self, session_id: str, token_ids: list[int], past_key_values: Any) -> None:
        """Store ``past_key_values`` keyed by session + token IDs."""
        with self._lock:
            self._evict_expired()
            if len(self._caches) >= self._max_sessions:
                oldest = min(self._caches, key=lambda k: self._caches[k][2])
                del self._caches[oldest]
            self._caches[session_id] = (token_ids, past_key_values, time.time())

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._caches.pop(session_id, None)

    def _evict_expired(self) -> None:
        now = time.time()
        stale = [k for k, v in self._caches.items() if now - v[2] > self._ttl]
        for k in stale:
            del self._caches[k]

    def evict_expired(self) -> None:
        with self._lock:
            self._evict_expired()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._caches)

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._caches),
                "max_sessions": self._max_sessions,
                "ttl_seconds": self._ttl,
            }
