"""Session-keyed chat response replay cache — reconnect continuity for POST /chat.

Housing moved from routers/inference.py, where the retired router-level
chat_stream path (e40b376a7) left the cache functions dead: chat now owns
reconnect replay through the infrastructure seam instead of importing a sibling
router. Replay is keyed on session_id + a message-list hash — a reconnect of
the SAME turn (same messages) replays the completed text without regenerating;
a new turn misses the hash and streams fresh. Event-id resume was deliberately
NOT adopted: the SSE frames never carried ids (the frontend's lastEventId was
always undefined), so the content hash is the honest contract (card 8125c86e).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time

_chat_response_cache: dict[str, dict] = {}
_chat_cache_lock = threading.Lock()
_CHAT_CACHE_MAX_SESSIONS = 100
_CHAT_CACHE_MAX_AGE_S = 300  # 5 minutes


def chunk_replay(text: str, size: int = 40) -> list[str]:
    """Split cached text into SSE replay chunks without altering whitespace.

    Replay must concatenate back to the exact cached text — newlines and
    multi-space runs included — so reconnects render identically to the
    original stream.
    """
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def get_chat_response_cache(session_id: str) -> dict | None:
    """Get cached response for a session."""
    with _chat_cache_lock:
        entry = _chat_response_cache.get(session_id)
        if entry and time.time() - entry.get("timestamp", 0) > _CHAT_CACHE_MAX_AGE_S:
            _chat_response_cache.pop(session_id, None)
            return None
        return entry


def set_chat_response_cache(session_id: str, response: dict) -> None:
    """Cache a partial or complete response for a session."""
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


def hash_messages(messages: list[dict[str, str]]) -> str:
    """Content-address a turn by its message list.

    A reconnect re-POSTs the same messages; a new turn does not — so the hash
    is what separates "resume me" from "answer me".
    """
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
