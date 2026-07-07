"""
Domain-specific Data Repositories with pluggable backends.

Provides typed repository classes for every major domain entity
(Session, Conversation, Feedback, Knowledge, Dataset, Agent, TrainingJob)
backed by either JSON files (FileRepository) or SQLite (SyncSQLiteRepository).

Usage:
    from domains.infrastructure.data_repository import RepositoryFactory, get_repository_factory

    factory = get_repository_factory(backend="file", base_dir="data")
    session_repo = factory.session_repo()
    session = session_repo.get("session_abc")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from domains.infrastructure.data_models import (
    AgentData,
    ConversationData,
    DatasetData,
    FeedbackData,
    KnowledgeFactData,
    SessionContextData,
    SessionData,
    TrainingJobData,
)
from domains.infrastructure.repository import (
    CachedRepository,
    FileRepository,
    JsonSerializer,
    Repository,
)

logger = logging.getLogger("man.data_repository")

T = TypeVar("T")


# ── Sync SQLite Backend ──


class SyncSQLiteRepository(Generic[T]):
    """
    SQLite-backed repository implementing the Repository[T] protocol (sync).

    Stores records as JSON blobs in a single table, mirroring the
    FileRepository interface but backed by sqlite3.
    """

    def __init__(
        self,
        table: str,
        db_path: str | Path,
        *,
        serializer: Any = None,
        cache_ttl: float = 0,
    ):
        self._table = table
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._serializer = serializer
        self._cache: dict[str, tuple[T, float]] = {}
        self._cache_ttl = cache_ttl
        self._init_table()

    def _init_table(self):
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self._table} (
                        id TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )"""
                )
                conn.commit()
            finally:
                conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _resolve_serializer(self) -> Any:
        if self._serializer is not None:
            if isinstance(self._serializer, type):
                return JsonSerializer(self._serializer)
            return self._serializer
        return JsonSerializer(dict)

    def _deserialize(self, data: dict) -> T:
        ser = self._resolve_serializer()
        return ser.deserialize(data)

    def _serialize(self, obj: T) -> dict:
        ser = self._resolve_serializer()
        return ser.serialize(obj)

    def _cache_get(self, key: str) -> T | None:
        if self._cache_ttl <= 0:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry[1] > self._cache_ttl:
            del self._cache[key]
            return None
        return entry[0]

    def _cache_set(self, key: str, value: T):
        if self._cache_ttl > 0:
            self._cache[key] = (value, time.monotonic())

    def _invalidate(self, key: str | None = None):
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    # ── Repository protocol ──

    def get(self, key: str) -> T | None:
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        with self._lock:
            conn = self._conn()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT data FROM {self._table} WHERE id = ?", (key,)
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                data = json.loads(row[0])
                obj = self._deserialize(data)
                self._cache_set(key, obj)
                return obj
            except Exception:
                logger.exception("Failed to read %s from SQLite %s", key, self._table)
                return None
            finally:
                conn.close()

    def list(self) -> list[T]:
        results: list[T] = []
        with self._lock:
            conn = self._conn()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT data FROM {self._table} ORDER BY updated_at DESC"
                )
                for row in cursor.fetchall():
                    try:
                        data = json.loads(row[0])
                        results.append(self._deserialize(data))
                    except Exception:
                        continue
            except Exception:
                logger.exception("Failed to list %s", self._table)
            finally:
                conn.close()
        return results

    def save(self, key: str, obj: T) -> bool:
        with self._lock:
            conn = self._conn()
            try:
                data = self._serialize(obj)
                now = time.time()
                json_data = json.dumps(data)
                conn.execute(
                    f"""INSERT OR REPLACE INTO {self._table}
                        (id, data, created_at, updated_at) VALUES (?, ?, COALESCE(
                            (SELECT created_at FROM {self._table} WHERE id = ?), ?
                        ), ?)""",
                    (key, json_data, key, now, now),
                )
                conn.commit()
                self._cache_set(key, obj)
                return True
            except Exception:
                logger.exception("Failed to save %s to SQLite %s", key, self._table)
                return False
            finally:
                conn.close()

    def delete(self, key: str) -> bool:
        with self._lock:
            conn = self._conn()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    f"DELETE FROM {self._table} WHERE id = ?", (key,)
                )
                conn.commit()
                deleted = cursor.rowcount > 0
                self._invalidate(key)
                return deleted
            except Exception:
                logger.exception("Failed to delete %s from SQLite %s", key, self._table)
                return False
            finally:
                conn.close()

    def search(self, query: str, fields: list[str] | None = None) -> list[T]:
        q = query.lower()
        results: list[T] = []
        for obj in self.list():
            data = self._serialize(obj)
            for field in fields or list(data.keys()):
                val = str(data.get(field, ""))
                if q in val.lower():
                    results.append(obj)
                    break
        return results

    def exists(self, key: str) -> bool:
        with self._lock:
            conn = self._conn()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT 1 FROM {self._table} WHERE id = ?", (key,)
                )
                return cursor.fetchone() is not None
            finally:
                conn.close()

    def count(self) -> int:
        with self._lock:
            conn = self._conn()
            try:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {self._table}")
                return cursor.fetchone()[0]
            finally:
                conn.close()


# ── Factory ──


class RepositoryFactory:
    """
    Config-driven factory for domain repositories.

    Selects backend (file or sqlite) and wires domain-specific repository
    classes with the correct serializers and file paths.
    """

    BACKEND_FILE = "file"
    BACKEND_SQLITE = "sqlite"

    def __init__(
        self,
        backend: str = BACKEND_FILE,
        base_dir: str | Path = "data",
        cache_ttl: float = 2.0,
    ):
        self._backend = backend
        self._base_dir = Path(base_dir)
        self._cache_ttl = cache_ttl

    def _file_repo(self, subdir: str, model_cls: type[T]) -> FileRepository[T]:
        """Create a FileRepository for the given subdirectory."""
        repo = FileRepository[T](
            self._base_dir / subdir,
            serializer=model_cls,
        )
        if self._cache_ttl > 0:
            repo.enable_cache(self._cache_ttl)
        return repo

    def _sqlite_repo(self, table: str, model_cls: type[T]) -> SyncSQLiteRepository[T]:
        """Create a SyncSQLiteRepository for the given table."""
        return SyncSQLiteRepository[T](
            table,
            self._base_dir / "repository.db",
            serializer=model_cls,
            cache_ttl=self._cache_ttl,
        )

    def _repo(self, subdir: str, model_cls: type[T]) -> Repository[T]:
        """Create a repository with the configured backend."""
        if self._backend == self.BACKEND_SQLITE:
            return self._sqlite_repo(subdir, model_cls)
        return self._file_repo(subdir, model_cls)

    # ── Domain repos ──

    def session_repo(self) -> SessionRepository:
        return SessionRepository(self._repo("chat_sessions", SessionData))

    def conversation_repo(self) -> ConversationRepository:
        return ConversationRepository(self._repo("conversations", ConversationData))

    def feedback_repo(self) -> FeedbackRepository:
        return FeedbackRepository(self._repo("feedback", FeedbackData))

    def knowledge_repo(self) -> KnowledgeRepository:
        return KnowledgeRepository(self._repo("knowledge", KnowledgeFactData))

    def dataset_repo(self) -> DatasetRepository:
        return DatasetRepository(self._repo("datasets", DatasetData))

    def agent_repo(self) -> AgentRepository:
        return AgentRepository(self._repo("agents", AgentData))

    def training_job_repo(self) -> TrainingJobRepository:
        return TrainingJobRepository(self._repo("training_jobs", TrainingJobData))

    def session_context_repo(self) -> SessionContextRepository:
        return SessionContextRepository(self._repo("session_contexts", SessionContextData))


# ── Domain-specific repositories ──


class SessionRepository:
    """Repository for chat sessions."""

    def __init__(self, backend: Repository[SessionData]):
        self._backend = backend

    def get(self, session_id: str) -> SessionData | None:
        return self._backend.get(session_id)

    def list(self) -> list[SessionData]:
        return self._backend.list()

    def save(self, session: SessionData) -> bool:
        return self._backend.save(session.id, session)

    def delete(self, session_id: str) -> bool:
        return self._backend.delete(session_id)

    def search(self, query: str) -> list[SessionData]:
        return self._backend.search(query, fields=["id"])

    def exists(self, session_id: str) -> bool:
        return self._backend.exists(session_id)

    def count(self) -> int:
        return self._backend.count()

    def get_messages(self, session_id: str) -> list[dict]:
        session = self.get(session_id)
        if session is None:
            return []
        return session.messages

    def append_message(self, session_id: str, role: str, content: str) -> bool:
        session = self.get(session_id)
        if session is None:
            return False
        from datetime import datetime, timezone
        session.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        session.updated_at = datetime.now(timezone.utc).isoformat()
        return self.save(session)


class ConversationRepository:
    """Repository for conversations (client-facing list items)."""

    def __init__(self, backend: Repository[ConversationData]):
        self._backend = backend

    def get(self, conv_id: str) -> ConversationData | None:
        return self._backend.get(conv_id)

    def list(self) -> list[ConversationData]:
        return self._backend.list()

    def save(self, conv: ConversationData) -> bool:
        return self._backend.save(conv.id, conv)

    def delete(self, conv_id: str) -> bool:
        return self._backend.delete(conv_id)

    def search(self, query: str) -> list[ConversationData]:
        return self._backend.search(query, fields=["name"])


class FeedbackRepository:
    """Repository for user feedback records."""

    def __init__(self, backend: Repository[FeedbackData]):
        self._backend = backend

    def get(self, feedback_id: str) -> FeedbackData | None:
        return self._backend.get(feedback_id)

    def list(self) -> list[FeedbackData]:
        return self._backend.list()

    def save(self, feedback: FeedbackData) -> bool:
        return self._backend.save(feedback.id, feedback)

    def delete(self, feedback_id: str) -> bool:
        return self._backend.delete(feedback_id)

    def count(self) -> int:
        return self._backend.count()

    def list_by_message(self, message_id: str) -> list[FeedbackData]:
        return self._backend.search(message_id, fields=["message_id"])


class KnowledgeRepository:
    """Repository for knowledge facts."""

    def __init__(self, backend: Repository[KnowledgeFactData]):
        self._backend = backend

    def get(self, fact_id: str) -> KnowledgeFactData | None:
        return self._backend.get(fact_id)

    def list(self) -> list[KnowledgeFactData]:
        return self._backend.list()

    def save(self, fact: KnowledgeFactData) -> bool:
        return self._backend.save(fact.id, fact)

    def delete(self, fact_id: str) -> bool:
        return self._backend.delete(fact_id)

    def search(self, query: str) -> list[KnowledgeFactData]:
        return self._backend.search(query, fields=["content", "topic"])

    def search_by_topic(self, topic: str) -> list[KnowledgeFactData]:
        return self._backend.search(topic, fields=["topic"])

    def count(self) -> int:
        return self._backend.count()


class DatasetRepository:
    """Repository for dataset metadata."""

    def __init__(self, backend: Repository[DatasetData]):
        self._backend = backend

    def get(self, dataset_id: str) -> DatasetData | None:
        return self._backend.get(dataset_id)

    def list(self) -> list[DatasetData]:
        return self._backend.list()

    def save(self, dataset: DatasetData) -> bool:
        return self._backend.save(dataset.id, dataset)

    def delete(self, dataset_id: str) -> bool:
        return self._backend.delete(dataset_id)

    def search(self, query: str) -> list[DatasetData]:
        return self._backend.search(query, fields=["name", "description"])


class AgentRepository:
    """Repository for AI agent definitions."""

    def __init__(self, backend: Repository[AgentData]):
        self._backend = backend

    def get(self, agent_id: str) -> AgentData | None:
        return self._backend.get(agent_id)

    def list(self) -> list[AgentData]:
        return self._backend.list()

    def save(self, agent: AgentData) -> bool:
        return self._backend.save(agent.id, agent)

    def delete(self, agent_id: str) -> bool:
        return self._backend.delete(agent_id)

    def search(self, query: str) -> list[AgentData]:
        return self._backend.search(query, fields=["name", "description"])


class TrainingJobRepository:
    """Repository for training job records."""

    def __init__(self, backend: Repository[TrainingJobData]):
        self._backend = backend

    def get(self, job_id: str) -> TrainingJobData | None:
        return self._backend.get(job_id)

    def list(self) -> list[TrainingJobData]:
        return self._backend.list()

    def save(self, job: TrainingJobData) -> bool:
        return self._backend.save(job.id, job)

    def delete(self, job_id: str) -> bool:
        return self._backend.delete(job_id)

    def list_by_status(self, status: str) -> list[TrainingJobData]:
        return self._backend.search(status, fields=["status"])


class SessionContextRepository:
    """Repository for session context (regeneration fallback data)."""

    def __init__(self, backend: Repository[SessionContextData]):
        self._backend = backend

    def get(self, session_id: str) -> SessionContextData | None:
        return self._backend.get(session_id)

    def save(self, ctx: SessionContextData) -> bool:
        return self._backend.save(ctx.session_id, ctx)

    def delete(self, session_id: str) -> bool:
        return self._backend.delete(session_id)


# ── Singleton factory ──

_factory: RepositoryFactory | None = None
_factory_lock = threading.Lock()


def get_repository_factory(
    backend: str = "file",
    base_dir: str | Path = "data",
    cache_ttl: float = 2.0,
) -> RepositoryFactory:
    """
    Get or create the singleton RepositoryFactory.

    Args:
        backend: Backend type ("file" or "sqlite").
        base_dir: Base directory for data storage.
        cache_ttl: TTL in seconds for read cache (0 = disabled).

    Returns:
        RepositoryFactory instance.
    """
    global _factory
    if _factory is None:
        with _factory_lock:
            if _factory is None:
                _factory = RepositoryFactory(
                    backend=backend,
                    base_dir=base_dir,
                    cache_ttl=cache_ttl,
                )
    return _factory


def reset_repository_factory():
    """Reset the singleton factory (for testing)."""
    global _factory
    with _factory_lock:
        _factory = None


__all__ = [
    "RepositoryFactory",
    "SessionRepository",
    "ConversationRepository",
    "FeedbackRepository",
    "KnowledgeRepository",
    "DatasetRepository",
    "AgentRepository",
    "TrainingJobRepository",
    "SessionContextRepository",
    "SyncSQLiteRepository",
    "get_repository_factory",
    "reset_repository_factory",
]
