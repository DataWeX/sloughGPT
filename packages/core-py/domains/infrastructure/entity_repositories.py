"""
Entity Repositories — typed wrappers around the base Repository for each domain entity.

Provides concrete repository implementations for:
  - KnowledgeRepository: knowledge facts and feed subscriptions
  - FeedbackRepository: conversation feedback (wraps MogDB)
  - SessionRepository: chat sessions (in-memory + optional persistence)
  - DatasetRepository: training datasets

Each repository encapsulates the storage format and provides a clean
CRUD interface matching the Repository protocol.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from domains.infrastructure.repository import (
    FileRepository,
    MemoryRepository,
)

logger = logging.getLogger("slo.repositories")


# ── Knowledge Repository ──


@dataclass
class KnowledgeEntry:
    """A single knowledge fact with metadata."""

    id: str
    content: str
    topic: str = "general"
    source: str = "manual"
    url: str = ""
    timestamp: float = 0.0
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FeedState:
    """RSS feed subscription state."""

    url: str
    title: str = ""
    last_fetched: float = 0.0
    poll_interval: float = 3600.0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class KnowledgeRepository:
    """Repository for knowledge facts with JSONL persistence.

    Replaces the ad-hoc ENTRIES_PATH / FEED_STATE_PATH JSON file usage
    in knowledge.py with a structured repository.
    """

    def __init__(self, data_dir: str | Path, *, enable_cache: bool = True, cache_ttl: float = 10.0):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._facts_repo = FileRepository[dict](
            self._data_dir / "facts",
            serializer=dict,
        )
        if enable_cache:
            self._facts_repo.enable_cache(cache_ttl)

        self._feeds_path = self._data_dir / "feeds.json"
        self._feeds: dict[str, FeedState] = {}
        self._lock = threading.Lock()
        self._load_feeds()

    def _load_feeds(self) -> None:
        if self._feeds_path.exists():
            try:
                data = json.loads(self._feeds_path.read_text())
                self._feeds = {k: FeedState.from_dict(v) for k, v in data.items()}
            except Exception:
                logger.warning("Failed to load feed state from %s", self._feeds_path)

    def _save_feeds(self) -> None:
        try:
            data = {k: v.to_dict() for k, v in self._feeds.items()}
            self._feeds_path.write_text(json.dumps(data, indent=2))
        except Exception:
            logger.warning("Failed to save feed state to %s", self._feeds_path)

    def get_fact(self, fact_id: str) -> Optional[KnowledgeEntry]:
        data = self._facts_repo.get(fact_id)
        if data is None:
            return None
        return KnowledgeEntry.from_dict(data)

    def list_facts(self, topic: str | None = None) -> list[KnowledgeEntry]:
        all_facts = [KnowledgeEntry.from_dict(d) for d in self._facts_repo.list()]
        if topic:
            return [f for f in all_facts if f.topic == topic]
        return all_facts

    def save_fact(self, entry: KnowledgeEntry) -> bool:
        return self._facts_repo.save(entry.id, entry.to_dict())

    def delete_fact(self, fact_id: str) -> bool:
        return self._facts_repo.delete(fact_id)

    def search_facts(self, query: str) -> list[KnowledgeEntry]:
        return [
            KnowledgeEntry.from_dict(d)
            for d in self._facts_repo.search(query, fields=["content", "topic", "source"])
        ]

    def count_facts(self) -> int:
        return self._facts_repo.count()

    def get_feed(self, url: str) -> Optional[FeedState]:
        return self._feeds.get(url)

    def list_feeds(self) -> list[FeedState]:
        return list(self._feeds.values())

    def save_feed(self, feed: FeedState) -> None:
        with self._lock:
            self._feeds[feed.url] = feed
            self._save_feeds()

    def delete_feed(self, url: str) -> bool:
        with self._lock:
            if url in self._feeds:
                del self._feeds[url]
                self._save_feeds()
                return True
            return False


# ── Session Repository ──


@dataclass
class SessionData:
    """Chat session metadata."""

    session_id: str
    message_count: int = 0
    created_at: float = 0.0
    last_active: float = 0.0
    title: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionData:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MessageRecord:
    """A single message in a session."""

    role: str
    content: str
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SessionRepository:
    """Repository for chat sessions with optional file persistence."""

    def __init__(self, persist_dir: str | Path | None = None):
        self._memory = MemoryRepository[dict]()
        self._messages: dict[str, list[dict]] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._lock = threading.Lock()

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self._persist_dir:
            return
        for path in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                session_id = path.stem
                self._memory.save(session_id, data.get("meta", {}))
                self._messages[session_id] = data.get("messages", [])
            except Exception:
                logger.warning("Failed to load session from %s", path)

    def _persist_session(self, session_id: str) -> None:
        if not self._persist_dir:
            return
        meta = self._memory.get(session_id)
        messages = self._messages.get(session_id, [])
        path = self._persist_dir / f"{session_id}.json"
        try:
            path.write_text(json.dumps({
                "meta": meta or {},
                "messages": messages,
            }, indent=2))
        except Exception:
            logger.warning("Failed to persist session %s", session_id)

    def get_session(self, session_id: str) -> Optional[SessionData]:
        data = self._memory.get(session_id)
        if data is None:
            return None
        return SessionData.from_dict(data)

    def list_sessions(self) -> list[SessionData]:
        return [SessionData.from_dict(d) for d in self._memory.list()]

    def save_session(self, session: SessionData) -> bool:
        with self._lock:
            ok = self._memory.save(session.session_id, session.to_dict())
            if ok and self._persist_dir:
                self._persist_session(session.session_id)
            return ok

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            self._messages.pop(session_id, None)
            ok = self._memory.delete(session_id)
            if ok and self._persist_dir:
                path = self._persist_dir / f"{session_id}.json"
                if path.exists():
                    path.unlink()
            return ok

    def get_messages(self, session_id: str) -> list[MessageRecord]:
        msgs = self._messages.get(session_id, [])
        return [MessageRecord(**m) for m in msgs]

    def add_message(self, session_id: str, message: MessageRecord) -> None:
        with self._lock:
            if session_id not in self._messages:
                self._messages[session_id] = []
            self._messages[session_id].append(message.to_dict())
            meta = self._memory.get(session_id)
            if meta:
                meta["message_count"] = len(self._messages[session_id])
                meta["last_active"] = time.time()
                self._memory.save(session_id, meta)
            if self._persist_dir:
                self._persist_session(session_id)

    def search_sessions(self, query: str) -> list[SessionData]:
        return [
            SessionData.from_dict(d)
            for d in self._memory.search(query, fields=["title", "model"])
        ]


# ── Feedback Repository ──


@dataclass
class FeedbackRecord:
    """A single feedback entry."""

    id: str
    message_id: str
    rating: str
    session_id: str = ""
    quality_score: float = 0.0
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class FeedbackRepository:
    """Repository for feedback data with in-memory fallback."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path
        self._db = None
        self._memory = MemoryRepository[dict]()
        self._lock = threading.Lock()

        if db_path:
            try:
                from mogdb import MogDB
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                self._db = MogDB(db_path)
                self._feedback = self._db.collection("feedback")
            except Exception as e:
                logger.warning("MogDB unavailable, using in-memory feedback: %s", e)

    def get_feedback(self, feedback_id: str) -> Optional[FeedbackRecord]:
        if self._db:
            data = self._feedback.find_one({"id": feedback_id})
            if data:
                return FeedbackRecord.from_dict(data)
        data = self._memory.get(feedback_id)
        return FeedbackRecord.from_dict(data) if data else None

    def list_feedback(self, session_id: str | None = None) -> list[FeedbackRecord]:
        if self._db:
            query = {"session_id": session_id} if session_id else {}
            items = self._feedback.find(query)
            return [FeedbackRecord.from_dict(d) for d in items]
        all_feedback = [FeedbackRecord.from_dict(d) for d in self._memory.list()]
        if session_id:
            return [f for f in all_feedback if f.session_id == session_id]
        return all_feedback

    def save_feedback(self, record: FeedbackRecord) -> bool:
        with self._lock:
            if self._db:
                try:
                    existing = self._feedback.find_one({"id": record.id})
                    if existing:
                        self._feedback.update_one({"id": record.id}, record.to_dict())
                    else:
                        self._feedback.insert_one(record.to_dict())
                    return True
                except Exception as e:
                    logger.warning("MogDB feedback save failed: %s", e)
            return self._memory.save(record.id, record.to_dict())

    def delete_feedback(self, feedback_id: str) -> bool:
        with self._lock:
            if self._db:
                try:
                    self._feedback.delete_one({"id": feedback_id})
                    return True
                except Exception as exc:
                    import logging
                    logging.getLogger("slo.repos").warning(
                        "MongoDB delete_feedback failed for %s: %s", feedback_id, exc)
            return self._memory.delete(feedback_id)

    def get_stats(self) -> dict[str, Any]:
        all_feedback = self.list_feedback()
        thumbs_up = sum(1 for f in all_feedback if f.rating == "thumbs_up")
        thumbs_down = sum(1 for f in all_feedback if f.rating == "thumbs_down")
        return {
            "total_feedback": len(all_feedback),
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
        }


# ── Dataset Repository ──


@dataclass
class DatasetMetadata:
    """Training dataset metadata."""

    id: str
    name: str = ""
    description: str = ""
    source: str = ""
    format: str = "jsonl"
    record_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMetadata:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DatasetRepository:
    """Repository for training dataset metadata."""

    def __init__(self, data_dir: str | Path, *, enable_cache: bool = True):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._repo = FileRepository[dict](
            self._data_dir / "registry",
            serializer=dict,
        )
        if enable_cache:
            self._repo.enable_cache(30.0)

    def get(self, dataset_id: str) -> Optional[DatasetMetadata]:
        data = self._repo.get(dataset_id)
        return DatasetMetadata.from_dict(data) if data else None

    def list(self) -> list[DatasetMetadata]:
        return [DatasetMetadata.from_dict(d) for d in self._repo.list()]

    def save(self, meta: DatasetMetadata) -> bool:
        return self._repo.save(meta.id, meta.to_dict())

    def delete(self, dataset_id: str) -> bool:
        return self._repo.delete(dataset_id)

    def search(self, query: str) -> list[DatasetMetadata]:
        return [
            DatasetMetadata.from_dict(d)
            for d in self._repo.search(query, fields=["name", "description", "tags"])
        ]


# ── Singleton accessors ──

_knowledge_repo: Optional[KnowledgeRepository] = None
_session_repo: Optional[SessionRepository] = None
_feedback_repo: Optional[FeedbackRepository] = None
_dataset_repo: Optional[DatasetRepository] = None


def get_knowledge_repository(data_dir: str | Path | None = None) -> KnowledgeRepository:
    global _knowledge_repo
    if _knowledge_repo is None:
        from domains.shared import find_repo_root
        from pathlib import Path as _Path
        base = _Path(data_dir) if data_dir else find_repo_root(_Path(__file__).resolve()) / "data" / "knowledge"
        _knowledge_repo = KnowledgeRepository(base)
    return _knowledge_repo


def get_session_repository(persist_dir: str | Path | None = None) -> SessionRepository:
    global _session_repo
    if _session_repo is None:
        from domains.shared import find_repo_root
        from pathlib import Path as _Path
        base = _Path(persist_dir) if persist_dir else find_repo_root(_Path(__file__).resolve()) / "data" / "sessions"
        _session_repo = SessionRepository(base)
    return _session_repo


def get_feedback_repository(db_path: str | None = None) -> FeedbackRepository:
    global _feedback_repo
    if _feedback_repo is None:
        from domains.shared import find_repo_root
        from pathlib import Path as _Path
        path = db_path or str(find_repo_root(_Path(__file__).resolve()) / "data" / "feedback.db")
        _feedback_repo = FeedbackRepository(path)
    return _feedback_repo


def get_dataset_repository(data_dir: str | Path | None = None) -> DatasetRepository:
    global _dataset_repo
    if _dataset_repo is None:
        from domains.shared import find_repo_root
        from pathlib import Path as _Path
        base = _Path(data_dir) if data_dir else find_repo_root(_Path(__file__).resolve()) / "data" / "datasets"
        _dataset_repo = DatasetRepository(base)
    return _dataset_repo
