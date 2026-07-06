"""
SloughGPT Database Module
Supports SQLite (default) and PostgreSQL
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import uuid
from dataclasses import dataclass
import logging

from domains.infrastructure.repository import FileRepository, JsonSerializer

logger = logging.getLogger("man.db_manager")

_CONVERSATIONS_DIR = Path(__file__).resolve().parents[4] / "data" / "conversations"
_conversation_repo = FileRepository[dict](
    directory=str(_CONVERSATIONS_DIR),
    serializer=JsonSerializer(dict),
    key_suffix=".json",
)
_conversation_repo.enable_cache(ttl_seconds=2.0)

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./sloughgpt.db")

# Try to import database libraries
try:
    import psycopg2

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        String,
        Float,
        DateTime,
        Text,
        JSON,
        Boolean,
    )
    from sqlalchemy.orm import sessionmaker, Session, declarative_base

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


if not SQLALCHEMY_AVAILABLE:
    # Provide stub placeholders for SQLAlchemy constructs so that model definitions
    # can be imported without the actual library. These stubs accept any arguments
    # and do nothing – they exist solely for type‑checking / test import safety.
    class _Stub:
        def __init__(self, *_, **__):
            pass
    Column = String = Integer = Float = DateTime = Text = JSON = Boolean = _Stub
    # When SQLAlchemy is not available, we also need a Base class for ORM models.
    Base = _Stub
else:
    # SQLAlchemy is available – import real symbols.
    from sqlalchemy.orm import sessionmaker, Session, declarative_base
    Base = declarative_base()
    # The imports above already pulled Column etc. into the module namespace.


class DatabaseManager:
    """Simple synchronous DB manager with file‑based fallback.

    When SQLAlchemy is unavailable we store data as JSON files under ``data/conversations``.
    The implementation mirrors the original sync API used by the UI server.
    """

    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.SessionLocal = None
        self._connected = False
        # Attempt to connect (will succeed with stub engine)
        self.connect()

    def connect(self) -> bool:
        """Create a SQLAlchemy engine if the library is available.

        Returns ``True`` on success, ``False`` otherwise.
        """
        try:
            self.engine = create_engine(self.database_url)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            self._connected = True
            return True
        except Exception as e:
            logger.error("Database connection failed: %s", e)
            self._connected = False
            return False

    def get_session(self) -> Optional[Session]:
        """Return a new SQLAlchemy session or ``None`` if unavailable."""
        if self.SessionLocal:
            return self.SessionLocal()
        return None

    # ---------- Simple dataclass models for fallback ----------
    # They provide ``to_dict`` used by the UI routes.
    @dataclass
    class ConversationModel:
        id: str
        name: Optional[str] = None
        metadata: dict = None
        messages: list = None

        def __post_init__(self):
            self.metadata = self.metadata or {}
            self.messages = self.messages or []

        def to_dict(self) -> dict:
            return {"id": self.id, "name": self.name, "metadata": self.metadata, "messages": self.messages}

    @dataclass
    class MessageModel:
        id: str
        conversation_id: str
        role: str
        content: str
        metadata: dict = None

        def __post_init__(self):
            self.metadata = self.metadata or {}

        def to_dict(self) -> dict:
            return {"id": self.id, "conversation_id": self.conversation_id, "role": self.role, "content": self.content, "metadata": self.metadata}

    @dataclass
    class TrainingJobModel:
        id: str
        config: dict

        def to_dict(self) -> dict:
            return {"id": self.id, "config": self.config}

    # ---------- Conversation operations ----------
    def create_conversation(self, name: str = None, metadata: dict = None) -> dict:
        """Create a new conversation – file fallback if not connected."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            conv = self.ConversationModel(id=str(uuid.uuid4()), name=name, metadata=metadata or {})
            session.add(conv)  # type: ignore[arg-type]
            session.commit()
            result = conv.to_dict()
            session.close()
            return result
        else:
            return self._create_conversation_file(name, metadata)

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        """Retrieve a conversation by id."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            conv = session.query(self.ConversationModel).filter_by(id=conv_id).first()
            result = conv.to_dict() if conv else None
            session.close()
            return result
        else:
            return self._get_conversation_file(conv_id)

    def list_conversations(self) -> List[dict]:
        """List all conversations."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            convs = session.query(self.ConversationModel).all()
            result = [c.to_dict() for c in convs]
            session.close()
            return result
        else:
            return self._list_conversations_file()

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            conv = session.query(self.ConversationModel).filter_by(id=conv_id).first()
            if conv:
                session.delete(conv)
                session.commit()
                session.close()
                return True
            session.close()
            return False
        else:
            return self._delete_conversation_file(conv_id)

    # ---------- Message operations ----------
    def add_message(self, conversation_id: str, role: str, content: str, metadata: dict = None) -> dict:
        """Add a message to a conversation."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            msg = self.MessageModel(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata=metadata or {},
            )
            session.add(msg)  # type: ignore[arg-type]
            session.commit()
            result = msg.to_dict()
            session.close()
            return result
        else:
            return self._add_message_file(conversation_id, role, content, metadata)

    def get_messages(self, conversation_id: str) -> List[dict]:
        """Get all messages for a conversation."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            msgs = session.query(self.MessageModel).filter_by(conversation_id=conversation_id).all()
            result = [m.to_dict() for m in msgs]
            session.close()
            return result
        else:
            return self._get_messages_file(conversation_id)

    # ---------- Training job operations ----------
    def create_training_job(self, job_id: str, config: dict) -> dict:
        """Create a training job record."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            job = self.TrainingJobModel(id=job_id, config=config)
            session.add(job)  # type: ignore[arg-type]
            session.commit()
            result = job.to_dict()
            session.close()
            return result
        return {"id": job_id, "config": config}

    def update_training_job(self, job_id: str, **kwargs) -> bool:
        """Update a training job."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            job = session.query(self.TrainingJobModel).filter_by(id=job_id).first()
            if job:
                for key, value in kwargs.items():
                    setattr(job, key, value)
                session.commit()
                session.close()
                return True
            session.close()
            return False
        return False

    def get_training_job(self, job_id: str) -> Optional[dict]:
        """Retrieve a training job."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            job = session.query(self.TrainingJobModel).filter_by(id=job_id).first()
            result = job.to_dict() if job else None
            session.close()
            return result
        return None

    # ---------- File‑based fallback helpers (FileRepository) ----------
    def _create_conversation_file(self, name, metadata):
        conv_id = str(uuid.uuid4())
        conv = {"id": conv_id, "name": name, "metadata": metadata, "messages": []}
        _conversation_repo.save(conv_id, conv)
        return conv

    def _get_conversation_file(self, conv_id):
        return _conversation_repo.get(conv_id)

    def _list_conversations_file(self):
        return _conversation_repo.list()

    def _delete_conversation_file(self, conv_id):
        return _conversation_repo.delete(conv_id)

    def _add_message_file(self, conv_id, role, content, metadata):
        conv = _conversation_repo.get(conv_id)
        if conv:
            msg = {"id": str(uuid.uuid4()), "role": role, "content": content, "metadata": metadata}
            conv["messages"].append(msg)
            _conversation_repo.save(conv_id, conv)
            return msg
        return None

    def _get_messages_file(self, conv_id):
        conv = _conversation_repo.get(conv_id)
        if conv:
            return conv.get("messages", [])
        return []

# Default manager instance
db = DatabaseManager()


def init_db(database_url: str = None):
    """Initialize the database (re‑connect with optional URL)."""
    global db
    db = DatabaseManager(database_url)
    return db.connect()
