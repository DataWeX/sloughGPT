"""Chat response cache — the reconnect-replay store for POST /chat.

Lives in infrastructure, not in either router: the chat surface and the
inference surface may both need it, and routers must not import routers
(pinned by ``test_chat_router_does_not_import_inference_router``).

Contract (card 8125c86e): a turn's token list is cached on client disconnect
(partial) and on completion (complete), keyed by session, with the per-request
event-counter the frames carried. A reconnect that proves retry-ness with a
``Last-Event-ID`` header replays the cached *unseen tail* from the chat
router — this module is storage only.
"""

import threading
import time

_chat_response_cache: dict[str, dict] = {}
_chat_cache_lock = threading.Lock()
_CHAT_CACHE_MAX_SESSIONS = 100
_CHAT_CACHE_MAX_AGE_S = 300  # 5 minutes


def get_chat_response_cache(session_id: str) -> dict | None:
    """Get cached response for a session."""
    with _chat_cache_lock:
        entry = _chat_response_cache.get(session_id)
        if entry and time.time() - entry.get("timestamp", 0) > _CHAT_CACHE_MAX_AGE_S:
            _chat_response_cache.pop(session_id, None)
            return None
        return entry


def set_chat_response_cache(session_id: str, response: dict) -> None:
    """Cache a partial response for a session."""
    with _chat_cache_lock:
        # Purge old entries if cache is full
        if len(_chat_response_cache) >= _CHAT_CACHE_MAX_SESSIONS:
            now = time.time()
            expired = [
                k
                for k, v in _chat_response_cache.items()
                if now - v.get("timestamp", 0) > _CHAT_CACHE_MAX_AGE_S
            ]
            for k in expired[:10]:  # Remove up to 10 expired entries
                _chat_response_cache.pop(k, None)
        _chat_response_cache[session_id] = {**response, "timestamp": time.time()}


def clear_chat_response_cache(session_id: str) -> None:
    """Clear cached response for a session."""
    with _chat_cache_lock:
        _chat_response_cache.pop(session_id, None)
