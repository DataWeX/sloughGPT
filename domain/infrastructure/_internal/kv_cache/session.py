"""Session KV state registry — a thin manager over ``KVState`` session mode.

This is the single home for per-session KV bookkeeping (LRU eviction + TTL).
It does not own storage: every entry lives in a ``KVState`` instance opened
in session mode (opaque entries — no prefix key), so the LRU/TTL/locking
machinery has exactly one implementation.
"""

from __future__ import annotations

import threading
import time as _time
from collections.abc import Callable, Iterator, MutableMapping
from typing import Any

from domain.infrastructure._internal.kv_cache.kv_state import KVState


class _SessionView(MutableMapping):
    """Mutable view over one ``KVState._sessions`` entry field.

    ``kv_states`` reads/writes ``entry[1]`` (the state); ``kv_last_access``
    reads/writes ``entry[2]`` (the timestamp). Both views share the same
    backing storage — no second registry.
    """

    __slots__ = ("_kv", "_field")

    def __init__(self, kv: KVState, field: int) -> None:
        self._kv = kv
        self._field = field

    def __getitem__(self, sid: str) -> Any:
        return self._kv._sessions[sid][self._field]

    def __setitem__(self, sid: str, value: Any) -> None:
        entry = self._kv._sessions.get(sid)
        if entry is None:
            base = [[], None, _time.time()]
        else:
            base = [*entry]
        base[self._field] = value
        self._kv._sessions[sid] = tuple(base)

    def __delitem__(self, sid: str) -> None:
        self._kv._sessions.pop(sid, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._kv._sessions)

    def __len__(self) -> int:
        return len(self._kv._sessions)

    def __contains__(self, sid: object) -> bool:
        return sid in self._kv._sessions


class SessionKVManager:
    """Per-session KV state register with LRU eviction + TTL.

    Storage is delegated onto ``KVState`` session mode: sessions are stored
    as opaque (prefix-free) entries, and reads/lookups touch last-access so
    LRU is by recency. The capped size and idle TTL live on the KVState.
    """

    __slots__ = ("_kv", "_state_view", "_access_view")

    def __init__(self, kv_ttl: float = 3600.0, kv_max_sessions: int = 64) -> None:
        self._kv = KVState(n_layers=0, max_sessions=kv_max_sessions, ttl=kv_ttl)
        self._state_view = _SessionView(self._kv, 1)
        self._access_view = _SessionView(self._kv, 2)

    @property
    def kv_ttl(self) -> float:
        """Idle TTL before a session is considered stale (seconds)."""
        return self._kv._ttl

    @kv_ttl.setter
    def kv_ttl(self, value: float) -> None:
        self._kv._ttl = value

    @property
    def kv_max_sessions(self) -> int:
        """LRU cap on concurrent sessions."""
        return self._kv._max_sessions

    @kv_max_sessions.setter
    def kv_max_sessions(self, value: int) -> None:
        self._kv._max_sessions = value

    @property
    def lock(self) -> threading.Lock:
        return self._kv._session_lock

    @property
    def kv_states(self) -> MutableMapping[str, Any]:
        """Session id -> KV state (mutable view over KVState storage)."""
        return self._state_view

    @property
    def kv_last_access(self) -> MutableMapping[str, float]:
        """Session id -> last-access timestamp (mutable view over storage)."""
        return self._access_view

    def get_session(self, session_id: str) -> Any | None:
        """Get KV state for a session, updating last access time."""
        return self._kv.session_get(session_id, [])[0]

    def set_session(self, session_id: str, state: Any) -> None:
        """Set KV state for a session, evicting TTL-expired + LRU if needed."""
        self._kv.session_store(session_id, [], state)

    def get_or_create(self, session_id: str, factory: Callable[[], Any]) -> Any:
        """Atomically return the state for ``session_id`` or create one.

        The check → create → store runs under one lock acquisition, so
        concurrent resolutions of the same id yield exactly one state object
        (no lost-update / double-create race from get-then-set callers).
        The factory is invoked only on a miss.
        """
        with self.lock:
            self._kv._session_evict_expired()
            entry = self._kv._sessions.get(session_id)
            if entry is not None:
                self._kv._sessions[session_id] = (entry[0], entry[1], _time.monotonic())
                return entry[1]
            if len(self._kv._sessions) >= self._kv._max_sessions:
                oldest = min(self._kv._sessions, key=lambda k: self._kv._sessions[k][2])
                del self._kv._sessions[oldest]
            state = factory()
            self._kv._sessions[session_id] = ([], state, _time.monotonic())
            return state

    def remove_session(self, session_id: str) -> bool:
        """Remove KV state for a session. Returns True if it existed."""
        with self.lock:
            return self._kv._sessions.pop(session_id, None) is not None

    def clear_all(self) -> int:
        """Clear all KV states. Returns number of sessions cleared."""
        with self.lock:
            n = len(self._kv._sessions)
            self._kv._sessions.clear()
            return n

    def get_stats(self) -> dict[str, Any]:
        """Get KV cache statistics."""
        with self.lock:
            sessions = self._kv._sessions
            n_sessions = len(sessions)
            state_sizes = {}
            for sid, entry in sessions.items():
                state = entry[1]
                if hasattr(state, "k") and hasattr(state, "v"):
                    state_sizes[sid] = {
                        "k_shape": list(state.k.shape) if hasattr(state.k, "shape") else None,
                        "v_shape": list(state.v.shape) if hasattr(state.v, "shape") else None,
                    }
                else:
                    state_sizes[sid] = {"type": type(state).__name__}

            last_access = [entry[2] for entry in sessions.values()]
            return {
                "active_sessions": n_sessions,
                "session_sizes": state_sizes,
                "max_sessions": self.kv_max_sessions,
                "ttl_seconds": self.kv_ttl,
                "oldest_session_age": (
                    max(last_access) - min(last_access) if len(last_access) > 1 else 0.0
                ),
            }

    def evict_stale_sessions(self) -> int:
        """Remove KV states for sessions idle longer than kv_ttl seconds."""
        with self.lock:
            before = len(self._kv._sessions)
            self._kv._session_evict_expired()
            n = before - len(self._kv._sessions)
        if n > 0:
            from domain.infrastructure._internal.structured_log import StructuredLogger

            logger = StructuredLogger("slo.inference.kv_cache")
            logger.info(
                "Evicted %d stale KV sessions (TTL=%.0fs)",
                n,
                self.kv_ttl,
                extra={"tag": "INF"},
            )
        return n
