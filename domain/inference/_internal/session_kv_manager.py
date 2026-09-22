"""
SessionKVManager — Manages per-session KV cache state for inference providers.

Extracted from SloNetChatProvider to centralize KV cache logic and reduce
code duplication across construction paths.
"""

from __future__ import annotations

import threading
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SessionKVManager:
    """Manages per-session KV cache state with LRU eviction and TTL.

    This class encapsulates all KV cache management logic that was previously
    duplicated across SloNetChatProvider's construction paths.
    """

    kv_states: dict[str, Any] = field(default_factory=dict)
    kv_last_access: dict[str, float] = field(default_factory=dict)
    kv_ttl: float = 3600.0  # 1 hour default TTL for idle sessions
    kv_max_sessions: int = 64  # LRU cap on concurrent sessions
    lock: threading.RLock = field(default_factory=threading.RLock)
    state_factory: Callable[[], Any] | None = field(default=None)

    def get_session(self, session_id: str) -> Any | None:
        """Get KV state for a session, updating last access time."""
        with self.lock:
            state = self.kv_states.get(session_id)
            if state is not None:
                self.kv_states[session_id] = state
                self.kv_last_access[session_id] = _time.monotonic()
            return state

    def get_or_create(self, session_id: str) -> Any | None:
        """Get a session's KV state, creating it via ``state_factory`` on miss.

        TTL-stale sessions are evicted up front, mirroring the provider's
        original resolve-then-evict behaviour on every access. On a cache miss
        the state is created lazily via ``state_factory`` (never called on
        hit), and LRU eviction runs when the insert would exceed
        ``kv_max_sessions`` — the just-created session is excluded from
        eviction candidates. Returns ``None`` on miss when no factory is set.
        """
        if self.state_factory is None:
            return self.get_session(session_id)
        with self.lock:
            self.evict_stale_sessions()
            state = self.kv_states.get(session_id)
            if state is not None:
                self.kv_last_access[session_id] = _time.monotonic()
                return state
            state = self.state_factory()
            self.kv_states[session_id] = state
            self.kv_last_access[session_id] = _time.monotonic()
            self._evict_if_needed(session_id)
            return state

    def set_session(self, session_id: str, state: Any) -> None:
        """Set KV state for a session, evicting LRU if at capacity."""
        with self.lock:
            self.kv_states[session_id] = state
            self.kv_last_access[session_id] = _time.monotonic()
            self._evict_if_needed(session_id)

    def cached_token_count(self) -> int:
        """Total cached tokens across all sessions (sum of each state's kv_len)."""
        with self.lock:
            total = 0
            for state in self.kv_states.values():
                kv = getattr(state, "kv_len", None)
                if isinstance(kv, (list, tuple)):
                    total += sum(kv)
                elif kv is not None:
                    total += kv
            return total

    def session_summary(self) -> dict[str, Any]:
        """Compact KV cache statistics for health/surface reporting.

        Counts only — does not compute per-session K/V shapes (see
        ``get_stats``). Mirrors the provider's ``session_stats`` output.
        """
        with self.lock:
            ages = self.kv_last_access.values()
            return {
                "active_sessions": len(self.kv_states),
                "max_sessions": self.kv_max_sessions,
                "ttl_seconds": self.kv_ttl,
                "cached_tokens": self.cached_token_count(),
                "oldest_session_age": (max(ages) - min(ages)) if len(ages) > 1 else 0.0,
            }

    def remove_session(self, session_id: str) -> bool:
        """Remove KV state for a session. Returns True if it existed."""
        with self.lock:
            existed = self.kv_states.pop(session_id, None) is not None
            self.kv_last_access.pop(session_id, None)
            return existed

    def clear_all(self) -> int:
        """Clear all KV states. Returns number of sessions cleared."""
        with self.lock:
            n = len(self.kv_states)
            self.kv_states.clear()
            self.kv_last_access.clear()
            return n

    def get_stats(self) -> dict[str, Any]:
        """Get KV cache statistics."""
        with self.lock:
            n_sessions = len(self.kv_states)
            state_sizes = {}
            for sid, state in self.kv_states.items():
                if hasattr(state, "k") and hasattr(state, "v"):
                    state_sizes[sid] = {
                        "k_shape": list(state.k.shape) if hasattr(state.k, "shape") else None,
                        "v_shape": list(state.v.shape) if hasattr(state.v, "shape") else None,
                    }
                else:
                    state_sizes[sid] = {"type": type(state).__name__}

            return {
                "active_sessions": n_sessions,
                "session_sizes": state_sizes,
                "max_sessions": self.kv_max_sessions,
                "ttl_seconds": self.kv_ttl,
                "oldest_session_age": (
                    max(self.kv_last_access.values()) - min(self.kv_last_access.values())
                    if len(self.kv_last_access) > 1
                    else 0.0
                ),
            }

    def evict_stale_sessions(self) -> int:
        """Remove KV states for sessions idle longer than kv_ttl seconds."""
        now = _time.monotonic()
        with self.lock:
            stale = [sid for sid, ts in self.kv_last_access.items() if now - ts > self.kv_ttl]
            for sid in stale:
                self.kv_states.pop(sid, None)
                self.kv_last_access.pop(sid, None)
            if stale:
                from domain.infrastructure._internal.structured_log import StructuredLogger

                logger = StructuredLogger("slo.inference.kv_cache")
                logger.info(
                    "Evicted %d stale KV sessions (TTL=%.0fs)",
                    len(stale),
                    self.kv_ttl,
                    extra={"tag": "INF"},
                )
            return len(stale)

    def _evict_if_needed(self, current_session_id: str) -> None:
        """Evict LRU session if at capacity.

        Must be called with self.lock held. The session being resolved is
        excluded from eviction candidates (it just got a write).
        """
        if len(self.kv_states) <= self.kv_max_sessions:
            return

        evictable = {
            sid: ts for sid, ts in self.kv_last_access.items() if sid != current_session_id
        }
        if not evictable:
            return

        lru_id = min(evictable, key=evictable.get)
        self.kv_states.pop(lru_id, None)
        self.kv_last_access.pop(lru_id, None)
        from domain.infrastructure._internal.structured_log import StructuredLogger

        logger = StructuredLogger("slo.inference.kv_cache")
        logger.info(
            "Evicted LRU session %s (max=%d)",
            lru_id,
            self.kv_max_sessions,
            extra={"tag": "INF"},
        )
