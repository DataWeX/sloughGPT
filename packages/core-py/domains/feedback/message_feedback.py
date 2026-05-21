"""
Message Feedback – in-memory feedback, session context, and regeneration storage.

Provides the ``MessageFeedback`` class (previously defined inline in the API server)
so that core domain modules (e.g. ``session_core``) can access it without importing
from the API layer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MessageData:
    """Lightweight message with role and content – no Pydantic dependency."""
    role: str
    content: str


class MessageFeedback:
    """Stores feedback for messages (thumbs up/down)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._feedback: Dict[str, Dict[str, Any]] = {}
        self._regenerations: Dict[str, Dict[str, Any]] = {}
        self._session_contexts: Dict[str, List[MessageData]] = {}

    def record_feedback(
        self,
        message_id: str,
        rating: str,
        session_id: Optional[str] = None,
        message_content: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record feedback for a message."""
        with self._lock:
            timestamp = datetime.utcnow().isoformat()
            feedback_entry: Dict[str, Any] = {
                "message_id": message_id,
                "rating": rating,
                "timestamp": timestamp,
                "session_id": session_id,
            }
            self._feedback[message_id] = feedback_entry

            if context:
                self._feedback[message_id]["context"] = (
                    context[:1000] if len(context) > 1000 else context
                )
            return feedback_entry

    def get_feedback(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a message."""
        return self._feedback.get(message_id)

    def store_session_context(self, session_id: str, messages: List[MessageData]) -> None:
        """Store conversation context for regeneration."""
        with self._lock:
            self._session_contexts[session_id] = list(messages)

    def get_session_context(self, session_id: str) -> Optional[List[MessageData]]:
        """Get stored conversation context."""
        with self._lock:
            return self._session_contexts.get(session_id)

    def clear_session_context(self, session_id: str) -> None:
        """Clear stored context for a session."""
        with self._lock:
            if session_id in self._session_contexts:
                del self._session_contexts[session_id]

    def record_regeneration(
        self,
        original_message_id: str,
        new_message_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a regeneration event."""
        with self._lock:
            regen_entry: Dict[str, Any] = {
                "original_message_id": original_message_id,
                "new_message_id": new_message_id,
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
            }
            self._regenerations[original_message_id] = regen_entry
            return regen_entry

    def list_conversations(self) -> List[Dict[str, Any]]:
        """Return metadata for every session that has stored context."""
        with self._lock:
            return [
                {"session_id": sid, "message_count": len(msgs)}
                for sid, msgs in self._session_contexts.items()
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        with self._lock:
            thumbs_up = sum(1 for f in self._feedback.values() if f.get("rating") == "thumbs_up")
            thumbs_down = sum(1 for f in self._feedback.values() if f.get("rating") == "thumbs_down")
            return {
                "total_feedback": len(self._feedback),
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "total_regenerations": len(self._regenerations),
                "active_sessions": len(self._session_contexts),
            }


# Singleton
_feedback_instance: Optional[MessageFeedback] = None


def get_message_feedback() -> MessageFeedback:
    """Get the global MessageFeedback singleton."""
    global _feedback_instance
    if _feedback_instance is None:
        _feedback_instance = MessageFeedback()
    return _feedback_instance


__all__ = [
    "MessageData",
    "MessageFeedback",
    "get_message_feedback",
]
