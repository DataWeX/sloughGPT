"""
Session core – unified backend handling for chat session storage and retrieval.

Provides a thin wrapper around ``MessageFeedback`` (in-memory store) for
session context.  This module lives in core‑py so the API server never needs
to reverse‑import domain internals.
"""

from __future__ import annotations

from typing import Any, Dict, List

from domains.feedback.message_feedback import get_message_feedback, MessageData


class SessionCore:
    """High‑level API for session context.

    All callers should use this class instead of directly accessing the
    ``message_feedback`` singleton or any in‑process dicts.
    """

    @staticmethod
    def store_context(session_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store a list of messages for *session_id*.

        Args:
            session_id: Identifier for the conversation.
            messages:   List of dicts ``{"role": ..., "content": ...}``.
        Returns:
            Dict containing ``status`` and ``message_count`` for API response.
        """
        converted = [MessageData(
            role=m.get("role", "user"),
            content=m.get("content", ""),
        ) for m in messages]
        get_message_feedback().store_session_context(session_id, converted)
        return {"status": "stored", "session_id": session_id, "message_count": len(messages)}

    @staticmethod
    def get_messages(session_id: str) -> List[Dict[str, str]]:
        """Retrieve stored messages for *session_id*.

        Returns a list of plain ``{"role": ..., "content": ...}`` dicts.
        """
        msgs = get_message_feedback().get_session_context(session_id)
        if msgs is None:
            return []
        return [{"role": m.role, "content": m.content} for m in msgs]

    @staticmethod
    def list_sessions() -> List[Dict[str, Any]]:
        """List all conversation metadata.

        This mirrors the ``list_conversations`` method of ``FeedbackDB`` but
        formats the result for the API consumer.
        """
        return get_message_feedback().list_conversations()
