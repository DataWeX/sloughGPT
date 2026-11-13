"""
Session core – unified backend handling for chat session storage and retrieval.

Provides a thin wrapper around ``MessageFeedback`` (in-memory store) for
session context.  This module lives in core‑py so the API server never needs
to reverse‑import domain internals.
"""

from __future__ import annotations

from typing import Any

from domain.feedback._internal.message_feedback import MessageData, get_message_feedback


class SessionCore:
    """High‑level API for session context.

    All callers should use this class instead of directly accessing the
    ``message_feedback`` singleton or any in‑process dicts.
    """

    @staticmethod
    def store_context(session_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Store a list of messages for *session_id*.

        Args:
            session_id: Identifier for the conversation.
            messages:   List of dicts ``{"role": ..., "content": ...}`` with
                optional client-assigned ``"id"`` (kept so id-keyed consumers
                survive reloads).

        Returns:
            Dict containing ``status`` and ``message_count`` for API response.
        """
        converted = [
            MessageData(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                id=m.get("id"),
            )
            for m in messages
        ]
        get_message_feedback().store_session_context(session_id, converted)
        return {"status": "stored", "session_id": session_id, "message_count": len(messages)}

    @staticmethod
    def get_messages(session_id: str) -> list[dict[str, str]]:
        """Retrieve stored messages for *session_id*.

        Returns a list of ``{"role": ..., "content": ...}`` dicts; ``id`` is
        included only when the writer assigned one (legacy writes keep their
        exact legacy shape — never an ``id: null`` leak).
        """
        msgs = get_message_feedback().get_session_context(session_id)
        if msgs is None:
            return []
        out: list[dict[str, str]] = []
        for m in msgs:
            entry: dict[str, str] = {"role": m.role, "content": m.content}
            if m.id is not None:
                entry["id"] = m.id
            out.append(entry)
        return out

    @staticmethod
    def list_sessions() -> list[dict[str, Any]]:
        """List all conversation metadata.

        This mirrors the ``list_conversations`` method of ``FeedbackDB`` but
        formats the result for the API consumer.
        """
        return get_message_feedback().list_conversations()
