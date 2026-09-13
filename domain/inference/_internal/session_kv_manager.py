"""
SessionKVManager — Manages per-session KV cache state for inference providers.

Extracted from SloNetChatProvider to centralize KV cache logic and reduce
code duplication across construction paths.
"""

from __future__ import annotations

import threading
import time as _time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SessionKVManager:
    """Manages per-session KV cache state with LRU eviction and TTL.

    This class encapsulates all KV cache management logic that was previously
    duplicated across SloNetChatProvider's construction paths.
    """

    kv_states: Dict[str, Any] = field(default_factory=dict)
    kv_last_access: Dict[str, float] = field(default_factory=dict)
    kv_ttl: float = 3600.0  # 1 hour default TTL for idle sessions
    kv_max_sessions: int = 64  # LRU cap on concurrent sessions
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_session(self, session_id: str) -> Optional[Any]:
        """Get KV state for a session, updating last access time."""
        with self.lock:
            state = self.kv_states.get(session_id)
            if state is not None:
                self.kv_states[session_id] = state
                self.kv_last_access[session_id] = _time.monotonic()
            return state

    def set_session(self, session_id: str, state: Any) -> None:
        """Set KV state for a session, evicting LRU if at capacity."""
        with self.lock:
            self.kv_states[session_id] = state
            self.kv_last_access[session_id] = _time.monotonic()
            self._evict_if_needed(session_id)

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

    def get_stats(self) -> Dict[str, Any]:
        """Get KV cache statistics."""
        with self.lock:
            n_sessions = len(self.kv_states)
            state_sizes = {}
            for sid, state in self.kv_states.items():
                if hasattr(state, 'k') and hasattr(state, 'v'):
                    state_sizes[sid] = {
                        'k_shape': list(state.k.shape) if hasattr(state.k, 'shape') else None,
                        'v_shape': list(state.v.shape) if hasattr(state.v, 'shape') else None,
                    }
                else:
                    state_sizes[sid] = {'type': type(state).__name__}

            return {
                "active_sessions": n_sessions,
                "session_sizes": state_sizes,
                "max_sessions": self.kv_max_sessions,
                "ttl_seconds": self.kv_ttl,
                "oldest_session_age": (
                    max(self.kv_last_access.values()) - min(self.kv_last_access.values())
                    if len(self.kv_last_access) > 1 else 0.0
                ),
            }

    def evict_stale_sessions(self) -> int:
        """Remove KV states for sessions idle longer than kv_ttl seconds."""
        now = _time.monotonic()
        with self.lock:
            stale = [
                sid for sid, ts in self.kv_last_access.items()
                if now - ts > self.kv_ttl
            ]
            for sid in stale:
                self.kv_states.pop(sid, None)
                self.kv_last_access.pop(sid, None)
            if stale:
                from domains.infrastructure.structured_log import StructuredLogger
                logger = StructuredLogger("slo.inference.kv_cache")
                logger.info(
                    "Evicted %d stale KV sessions (TTL=%.0fs)",
                    len(stale), self.kv_ttl,
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
            sid: ts for sid, ts in self.kv_last_access.items()
            if sid != current_session_id
        }
        if not evictable:
            return

        lru_id = min(evictable, key=evictable.get)
        self.kv_states.pop(lru_id, None)
        self.kv_last_access.pop(lru_id, None)
        from domains.infrastructure.structured_log import StructuredLogger
        logger = StructuredLogger("slo.inference.kv_cache")
        logger.info(
            "Evicted LRU session %s (max=%d)",
            lru_id, self.kv_max_sessions,
            extra={"tag": "INF"},
        )
