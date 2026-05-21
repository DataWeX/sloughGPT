"""
Conversation Engine - Persistent database for conversation storage.
Designed for incremental backups and permanent storage.
"""

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import threading
import hashlib


@dataclass
class ChatMessage:
    """Single message record in the conversation DB."""

    id: str
    session_id: str
    role: str  # "user" | "assistant"
    content: str
    model: str
    timestamp: str
    tokens: Optional[int] = None
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationDB:
    """
    Persistent conversation database with incremental backup support.
    """

    VERSION = "1.0"
    HEADER = "=== CONVERSATION_DB_V1 ==="
    FOOTER = "=== END_CONVERSATION_DB ==="

    def __init__(
        self,
        data_dir: str = "data",
        db_name: str = "conversations.db",
        backup_enabled: bool = True,
        max_backups: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / db_name
        self.log_path = self.data_dir / "conversations.log"
        self.backup_dir = self.data_dir / "backups"
        self.backup_enabled = backup_enabled
        self.max_backups = max_backups

        self._lock = threading.Lock()
        self._cache: Optional[List[ChatMessage]] = None

        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize storage."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.backup_enabled:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        if not self.db_path.exists():
            self._save_all([])

    def _load_all(self) -> List[ChatMessage]:
        """Load all messages from main DB."""
        if not self.db_path.exists():
            return []

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return []

            if self.HEADER in content:
                start = content.index(self.HEADER) + len(self.HEADER)
                end = content.index(self.FOOTER) if self.FOOTER in content else len(content)
                data_str = content[start:end].strip()
            else:
                data_str = content.strip()

            if not data_str:
                return []

            data = json.loads(data_str)
            return [ChatMessage(**msg) for msg in data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_all(self, messages: List[ChatMessage]) -> None:
        """Save all messages to main DB."""
        with self._lock:
            data = json.dumps([asdict(m) for m in messages], indent=2, ensure_ascii=False)

            with open(self.db_path, "w", encoding="utf-8") as f:
                f.write(self.HEADER + "\n")
                f.write(data)
                self._cache = messages

            if self.log_path.exists():
                self.log_path.unlink()

    def _append_log(self, operation: str, data: Dict) -> None:
        """Append operation to log."""
        with self._lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "data": data,
            }

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

    def _ensure_cache(self) -> List[ChatMessage]:
        """Get messages, loading from disk if needed."""
        if self._cache is None:
            self._cache = self._load_all()
        return self._cache

    def _invalidate_cache(self) -> None:
        """Invalidate cache."""
        self._cache = None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: str,
        tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Add a new message."""
        messages = self._ensure_cache()

        msg_id = f"msg_{len(messages)}_{int(datetime.now().timestamp() * 1000)}"
        msg = ChatMessage(
            id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            model=model,
            timestamp=datetime.now().isoformat(),
            tokens=tokens,
            metadata=metadata or {},
        )

        messages.append(msg)
        self._save_all(messages)
        self._append_log("add", asdict(msg))

        return msg

    def update_message(self, message_id: str, **kwargs) -> bool:
        """Update a message's fields."""
        messages = self._ensure_cache()

        for msg in messages:
            if msg.id == message_id:
                for key, value in kwargs.items():
                    if hasattr(msg, key):
                        setattr(msg, key, value)

                self._append_log("update", {"id": message_id, "updates": kwargs})
                self._save_all(messages)
                return True

        return False

    def delete_message(self, message_id: str) -> bool:
        """Delete a message by ID."""
        messages = self._ensure_cache()

        original_len = len(messages)
        messages = [m for m in messages if m.id != message_id]

        if len(messages) < original_len:
            self._append_log("delete", {"id": message_id})
            self._save_all(messages)
            return True

        return False

    def delete_session(self, session_id: str) -> int:
        """Delete all messages in a session."""
        messages = self._ensure_cache()

        original_len = len(messages)
        messages = [m for m in messages if m.session_id != session_id]

        deleted = original_len - len(messages)
        if deleted > 0:
            self._append_log("delete_session", {"session_id": session_id, "count": deleted})
            self._save_all(messages)

        return deleted

    def get_message(self, message_id: str) -> Optional[ChatMessage]:
        """Get a single message by ID."""
        messages = self._ensure_cache()
        for msg in messages:
            if msg.id == message_id:
                return msg
        return None

    def get_session(self, session_id: str) -> List[ChatMessage]:
        """Get all messages in a session."""
        messages = self._ensure_cache()
        return [m for m in messages if m.session_id == session_id]

    def get_recent(self, limit: int = 50) -> List[ChatMessage]:
        """Get most recent messages."""
        messages = self._ensure_cache()
        return messages[-limit:]

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        limit: int = 50,
    ) -> List[ChatMessage]:
        """Search messages."""
        messages = self._ensure_cache()
        query_lower = query.lower()

        results = []
        for msg in messages:
            if query_lower in msg.content.lower():
                if session_id and msg.session_id != session_id:
                    continue
                if role and msg.role != role:
                    continue
                results.append(msg)

        return results[:limit]

    def get_sessions(self) -> List[str]:
        """Get list of all session IDs."""
        messages = self._ensure_cache()
        return list(set(m.session_id for m in messages))

    def get_training_pairs(
        self,
        min_feedback: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Get (prompt, response) pairs for training."""
        messages = self._ensure_cache()

        if session_id:
            messages = [m for m in messages if m.session_id == session_id]

        pairs = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "user" and i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.role == "assistant":
                    if (
                        min_feedback is None
                        or next_msg.feedback == min_feedback
                        or next_msg.feedback is None
                    ):
                        pairs.append(
                            {
                                "prompt": msg.content,
                                "response": next_msg.content,
                            }
                        )
                    i += 1
            i += 1

        return pairs

    def to_context_string(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """Format messages as context string."""
        messages = self._ensure_cache()

        if session_id:
            messages = [m for m in messages if m.session_id == session_id]

        messages = messages[-limit:]

        lines = ["=== CONVERSATIONS ==="]
        for m in messages:
            role = "User" if m.role == "user" else "Assistant"
            content = m.content[:300] + "..." if len(m.content) > 300 else m.content
            lines.append(f"[{role}]({m.model}): {content}")

        lines.append("=== END ===")
        return "\n".join(lines)

    def export_json(self) -> str:
        """Export entire DB as JSON."""
        messages = self._ensure_cache()
        return json.dumps([asdict(m) for m in messages], indent=2)

    def export_csv(self, session_id: Optional[str] = None) -> str:
        """Export as CSV."""
        messages = self._ensure_cache()

        if session_id:
            messages = [m for m in messages if m.session_id == session_id]

        lines = ["id,session_id,role,content,model,timestamp,tokens,feedback"]
        for m in messages:
            content_escaped = m.content.replace('"', '""')
            lines.append(
                f'"{m.id}","{m.session_id}","{m.role}","{content_escaped}",'
                f'"{m.model}","{m.timestamp}","{m.tokens or ""}","{m.feedback or ""}"'
            )

        return "\n".join(lines)

    def create_backup(self) -> str:
        """Create a timestamped backup."""
        if not self.backup_enabled:
            raise RuntimeError("Backups are disabled")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"conversations_{timestamp}.db"
        backup_path = self.backup_dir / backup_name

        shutil.copy2(self.db_path, backup_path)

        if self.log_path.exists():
            log_backup = self.backup_dir / f"conversations_{timestamp}.log"
            shutil.copy2(self.log_path, log_backup)

        self._cleanup_old_backups()

        return str(backup_path)

    def _cleanup_old_backups(self) -> None:
        """Remove old backups."""
        if not self.backup_dir.exists():
            return

        backups = sorted(self.backup_dir.glob("conversations_*.db"))

        while len(backups) > self.max_backups:
            old = backups.pop(0)
            old.unlink()

            log_file = self.backup_dir / old.name.replace(".db", ".log")
            if log_file.exists():
                log_file.unlink()

    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore from a backup file."""
        if not Path(backup_path).exists():
            return False

        # Backup current state first (skip cleanup to preserve target backup)
        if self.backup_enabled and self.db_path.exists() and self.db_path.stat().st_size > 100:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            emergency_backup = self.backup_dir / f"conversations_pre_restore_{timestamp}.db"
            shutil.copy2(self.db_path, emergency_backup)

        # Now restore from the requested backup
        shutil.copy2(backup_path, self.db_path)
        self._invalidate_cache()
        return True

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        if not self.backup_dir.exists():
            return []

        backups = []
        for f in sorted(self.backup_dir.glob("conversations_*.db")):
            stat = f.stat()
            backups.append(
                {
                    "path": str(f),
                    "name": f.name,
                    "size_bytes": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                }
            )

        return backups

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        messages = self._ensure_cache()

        sessions = set(m.session_id for m in messages)
        users = [m for m in messages if m.role == "user"]
        assistants = [m for m in messages if m.role == "assistant"]

        return {
            "total_messages": len(messages),
            "total_sessions": len(sessions),
            "user_messages": len(users),
            "assistant_messages": len(assistants),
            "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "log_size_bytes": self.log_path.stat().st_size if self.log_path.exists() else 0,
            "backups_count": len(self.list_backups()),
        }

    def compact(self) -> None:
        """Compact the database."""
        messages = self._ensure_cache()

        seen_ids = set()
        unique_messages = []
        for msg in messages:
            if msg.id not in seen_ids:
                seen_ids.add(msg.id)
                unique_messages.append(msg)

        self._save_all(unique_messages)

    def clear(self) -> None:
        """Clear all messages."""
        self._append_log("clear", {"count": len(self._ensure_cache())})
        self._save_all([])

    def vacuum(self) -> None:
        """Vacuum - optimize storage."""
        self.compact()
        self._cleanup_old_backups()


_db_instance: Optional[ConversationDB] = None
_db_lock = threading.Lock()


def get_db(
    data_dir: str = "data",
    db_name: str = "conversations.db",
    backup_enabled: bool = True,
) -> ConversationDB:
    """Get or create the conversation DB singleton."""
    global _db_instance

    with _db_lock:
        if _db_instance is None:
            _db_instance = ConversationDB(
                data_dir=data_dir,
                db_name=db_name,
                backup_enabled=backup_enabled,
            )
        return _db_instance


def reset_db() -> None:
    """Reset the DB instance."""
    global _db_instance
    with _db_lock:
        _db_instance = None


__all__ = [
    "ChatMessage",
    "ConversationDB",
    "get_db",
    "reset_db",
]
