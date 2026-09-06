"""MogDB-backed chat session persistence.

Replaces the FileRepository-based session storage with MogDB + JSON sync.
Provides a clean API for CRUD operations on chat sessions.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("slo.session_store")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """MogDB-backed chat session store with JSON sync.

    Parameters
    ----------
    db_path:
        Directory for MogDB journal files.
    sync_dir:
        Directory for JSON sync files (human-readable backup).
    """

    def __init__(self, db_path: str | Path | None = None, sync_dir: str | Path | None = None):
        from mogdb import MogDB

        if db_path is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            db_path = repo_root / "data" / "sessions_mogdb"
        if sync_dir is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            sync_dir = repo_root / "data" / "chat_sessions"

        self._db = MogDB(str(db_path), sync_dir=str(sync_dir))
        self._col = self._db.collection("chat_sessions")

    def create(
        self,
        name: str = "",
        model: str | None = None,
        messages: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Create a new chat session.

        Returns the full session dict with auto-generated id and timestamps.
        """
        now = _now_iso()
        doc = {
            "id": str(uuid.uuid4()),
            "name": name,
            "model": model,
            "messages": messages or [],
            "created_at": now,
            "updated_at": now,
            "archived": False,
            "starred": False,
            "pinned": False,
        }
        self._col.insert_one(doc)
        logger.info("Created session '%s' (id=%s)", name, doc["id"])
        return doc

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get a session by ID. Returns None if not found."""
        return self._col.find_one({"id": session_id})

    def list(self, include_archived: bool = False) -> list[dict[str, Any]]:
        """List sessions, sorted by updated_at descending."""
        query = {} if include_archived else {"archived": {"$ne": True}}
        sessions = self._col.find(query, sort=[("updated_at", -1)])
        return sessions

    def upsert(self, session_id: str, **fields: Any) -> None:
        """Update session fields (archived, starred, pinned, name, etc.).

        Raises ValueError if session not found.
        """
        session = self._col.find_one({"id": session_id})
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        update = {k: v for k, v in fields.items() if k != "id"}
        update["updated_at"] = _now_iso()
        self._col.update_one({"id": session_id}, {"$set": update})

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        **extra: Any,
    ) -> None:
        """Append a message to a session's message list.

        Raises ValueError if session not found.
        """
        session = self._col.find_one({"id": session_id})
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        msg = {
            "role": role,
            "content": content,
            "timestamp": _now_iso(),
            **extra,
        }
        messages = session.get("messages", [])
        messages.append(msg)
        self._col.update_one(
            {"id": session_id},
            {"$set": {"messages": messages, "updated_at": _now_iso()}},
        )

    def delete(self, session_id: str) -> None:
        """Delete a session permanently.

        Raises ValueError if session not found.
        """
        session = self._col.find_one({"id": session_id})
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        self._col.delete_one({"id": session_id})
        logger.info("Deleted session (id=%s)", session_id)

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search sessions by name or message content.

        Simple substring match across session names and message content.
        """
        all_sessions = self._col.find()
        q = query.lower()
        results = []
        for s in all_sessions:
            if q in s.get("name", "").lower():
                results.append(s)
                continue
            for msg in s.get("messages", []):
                if q in msg.get("content", "").lower():
                    results.append(s)
                    break
        return results

    def count(self, include_archived: bool = True) -> int:
        """Count sessions."""
        if include_archived:
            return self._col.count()
        return self._col.count({"archived": {"$ne": True}})
