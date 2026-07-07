"""
Tests for Data Repository layer — data models, domain repos, factory, SQLite backend.

Covers:
  - Data model creation (all 7+ entity types)
  - FileRepository-backed domain repos (CRUD, search, list, edge cases)
  - SyncSQLiteRepository-backed domain repos (same interface, different backend)
  - RepositoryFactory with both backends
  - Cache behavior (TTL invalidation)
  - Error handling (missing records, serializer edge cases)
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone

import pytest

from domains.infrastructure.data_models import (
    AgentData,
    ConversationData,
    DatasetData,
    FeedbackData,
    FeedbackStatsData,
    KnowledgeFactData,
    SessionContextData,
    SessionData,
    TrainingJobData,
)
from domains.infrastructure.data_repository import (
    RepositoryFactory,
    SyncSQLiteRepository,
    get_repository_factory,
    reset_repository_factory,
)
from domains.infrastructure.repository import FileRepository


# ── Fixtures ──


@pytest.fixture
def tmp_base(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def file_factory(tmp_base: Path) -> RepositoryFactory:
    return RepositoryFactory(backend="file", base_dir=tmp_base, cache_ttl=0)


@pytest.fixture
def sqlite_factory(tmp_base: Path) -> RepositoryFactory:
    return RepositoryFactory(backend="sqlite", base_dir=tmp_base, cache_ttl=0)


@pytest.fixture
def cached_factory(tmp_base: Path) -> RepositoryFactory:
    return RepositoryFactory(backend="file", base_dir=tmp_base, cache_ttl=60)


# ── Data Model Tests ──


class TestDataModels:
    def test_session_data_defaults(self):
        s = SessionData(id="s1")
        assert s.id == "s1"
        assert s.messages == []
        assert s.created_at == ""
        assert s.metadata == {}
        assert s.user_id is None

    def test_session_data_with_messages(self):
        s = SessionData(id="s1", messages=[{"role": "user", "content": "hi"}])
        assert len(s.messages) == 1
        assert s.messages[0]["role"] == "user"

    def test_conversation_data(self):
        c = ConversationData(id="c1", session_id="s1", name="test")
        assert c.message_count == 0
        assert c.pinned is False

    def test_feedback_data(self):
        f = FeedbackData(id="f1", message_id="m1", rating="thumbs_up")
        assert f.rating == "thumbs_up"
        assert f.quality_score is None

    def test_feedback_stats_data(self):
        s = FeedbackStatsData(thumbs_up=10, thumbs_down=2)
        assert s.total == 12
        assert s.up_ratio == 10 / 12

    def test_knowledge_fact_data(self):
        k = KnowledgeFactData(
            content="Paris is the capital of France",
            topic="geography",
            importance=0.9,
        )
        assert k.source == "manual"
        assert k.tags == []

    def test_dataset_data(self):
        d = DatasetData(id="ds1", name="my_dataset", size_bytes=1024)
        assert d.type == "text"
        assert d.num_samples == 0

    def test_agent_data(self):
        a = AgentData(id="a1", name="helper")
        assert a.model == ""

    def test_training_job_data(self):
        t = TrainingJobData(id="j1", name="train1", status="running", model="gpt2")
        assert t.progress == 0
        assert t.loss is None

    def test_session_context_data(self):
        sc = SessionContextData(session_id="s1")
        assert sc.system_prompt == ""
        assert sc.messages == []


# ── Domain Repo Tests (File Backend) ──


class TestFileSessionRepository:
    def test_save_and_get(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        session = SessionData(id="s1", created_at="now")
        assert repo.save(session) is True
        loaded = repo.get("s1")
        assert loaded is not None
        assert loaded.id == "s1"

    def test_get_nonexistent_returns_none(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        assert repo.get("nonexistent") is None

    def test_delete(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        repo.save(SessionData(id="s1"))
        assert repo.delete("s1") is True
        assert repo.get("s1") is None

    def test_delete_nonexistent(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        assert repo.delete("nonexistent") is True  # FileRepository returns True

    def test_list(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        repo.save(SessionData(id="s1"))
        repo.save(SessionData(id="s2"))
        sessions = repo.list()
        assert len(sessions) == 2

    def test_list_empty(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        assert repo.list() == []

    def test_exists(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        assert repo.exists("s1") is False
        repo.save(SessionData(id="s1"))
        assert repo.exists("s1") is True

    def test_count(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        assert repo.count() == 0
        repo.save(SessionData(id="s1"))
        repo.save(SessionData(id="s2"))
        assert repo.count() == 2

    def test_get_messages(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        session = SessionData(
            id="s1",
            messages=[{"role": "user", "content": "hello"}],
        )
        repo.save(session)
        msgs = repo.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"

    def test_get_messages_nonexistent(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        assert repo.get_messages("nope") == []

    def test_append_message(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        repo.save(SessionData(id="s1", created_at="now"))
        assert repo.append_message("s1", "user", "hi") is True
        msgs = repo.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_append_message_nonexistent(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        assert repo.append_message("nope", "user", "hi") is False

    def test_search(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        repo.save(SessionData(id="abc123"))
        results = repo.search("abc")
        assert len(results) == 1

    def test_save_updates_existing(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        repo.save(SessionData(id="s1", messages=[{"role": "user", "content": "a"}]))
        repo.save(SessionData(id="s1", messages=[{"role": "user", "content": "b"}]))
        loaded = repo.get("s1")
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "b"


class TestFileConversationRepository:
    def test_crud(self, file_factory: RepositoryFactory):
        repo = file_factory.conversation_repo()
        conv = ConversationData(id="c1", session_id="s1", name="Chat 1")
        assert repo.save(conv) is True
        loaded = repo.get("c1")
        assert loaded is not None
        assert loaded.name == "Chat 1"

    def test_list_sorted(self, file_factory: RepositoryFactory):
        repo = file_factory.conversation_repo()
        repo.save(ConversationData(id="c1", session_id="s1", name="A"))
        repo.save(ConversationData(id="c2", session_id="s2", name="B"))
        assert len(repo.list()) == 2


class TestFileFeedbackRepository:
    def test_crud(self, file_factory: RepositoryFactory):
        repo = file_factory.feedback_repo()
        fb = FeedbackData(id="f1", message_id="m1", rating="thumbs_up")
        assert repo.save(fb) is True
        loaded = repo.get("f1")
        assert loaded is not None
        assert loaded.rating == "thumbs_up"

    def test_count(self, file_factory: RepositoryFactory):
        repo = file_factory.feedback_repo()
        repo.save(FeedbackData(id="f1", message_id="m1", rating="thumbs_up"))
        repo.save(FeedbackData(id="f2", message_id="m2", rating="thumbs_down"))
        assert repo.count() == 2

    def test_list_by_message(self, file_factory: RepositoryFactory):
        repo = file_factory.feedback_repo()
        repo.save(FeedbackData(id="f1", message_id="m1", rating="up"))
        repo.save(FeedbackData(id="f2", message_id="m1", rating="down"))
        repo.save(FeedbackData(id="f3", message_id="m2", rating="up"))
        results = repo.list_by_message("m1")
        assert len(results) == 2


class TestFileKnowledgeRepository:
    def test_crud(self, file_factory: RepositoryFactory):
        repo = file_factory.knowledge_repo()
        fact = KnowledgeFactData(
            id="k1",
            content="Earth orbits the Sun",
            topic="astronomy",
        )
        assert repo.save(fact) is True
        loaded = repo.get("k1")
        assert loaded is not None
        assert loaded.topic == "astronomy"

    def test_search_by_topic(self, file_factory: RepositoryFactory):
        repo = file_factory.knowledge_repo()
        repo.save(KnowledgeFactData(id="k1", content="A", topic="astro"))
        repo.save(KnowledgeFactData(id="k2", content="B", topic="astro"))
        repo.save(KnowledgeFactData(id="k3", content="C", topic="geo"))
        results = repo.search_by_topic("astro")
        assert len(results) == 2

    def test_search_content(self, file_factory: RepositoryFactory):
        repo = file_factory.knowledge_repo()
        repo.save(KnowledgeFactData(id="k1", content="Paris is capital"))
        results = repo.search("paris")
        assert len(results) == 1


class TestFileDatasetRepository:
    def test_crud(self, file_factory: RepositoryFactory):
        repo = file_factory.dataset_repo()
        ds = DatasetData(id="ds1", name="test", size_bytes=500)
        assert repo.save(ds) is True
        loaded = repo.get("ds1")
        assert loaded is not None
        assert loaded.name == "test"


class TestFileAgentRepository:
    def test_crud(self, file_factory: RepositoryFactory):
        repo = file_factory.agent_repo()
        agent = AgentData(id="a1", name="helper", system_prompt="Be helpful")
        assert repo.save(agent) is True
        loaded = repo.get("a1")
        assert loaded is not None
        assert loaded.system_prompt == "Be helpful"


class TestFileTrainingJobRepository:
    def test_crud(self, file_factory: RepositoryFactory):
        repo = file_factory.training_job_repo()
        job = TrainingJobData(id="j1", name="exp1", status="running")
        assert repo.save(job) is True
        loaded = repo.get("j1")
        assert loaded is not None
        assert loaded.status == "running"

    def test_list_by_status(self, file_factory: RepositoryFactory):
        repo = file_factory.training_job_repo()
        repo.save(TrainingJobData(id="j1", name="a", status="running"))
        repo.save(TrainingJobData(id="j2", name="b", status="completed"))
        repo.save(TrainingJobData(id="j3", name="c", status="running"))
        running = repo.list_by_status("running")
        assert len(running) == 2


class TestFileSessionContextRepository:
    def test_crud(self, file_factory: RepositoryFactory):
        repo = file_factory.session_context_repo()
        ctx = SessionContextData(session_id="s1", system_prompt="You are helpful")
        assert repo.save(ctx) is True
        loaded = repo.get("s1")
        assert loaded is not None
        assert loaded.system_prompt == "You are helpful"


# ── SQLite Backend Tests ──


class TestSQLiteSessionRepository:
    def test_save_and_get(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.session_repo()
        session = SessionData(id="s1", created_at="now")
        assert repo.save(session) is True
        loaded = repo.get("s1")
        assert loaded is not None
        assert loaded.id == "s1"

    def test_get_nonexistent(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.session_repo()
        assert repo.get("nope") is None

    def test_delete(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.session_repo()
        repo.save(SessionData(id="s1"))
        assert repo.delete("s1") is True
        assert repo.get("s1") is None

    def test_list(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.session_repo()
        repo.save(SessionData(id="s1"))
        repo.save(SessionData(id="s2"))
        assert len(repo.list()) == 2

    def test_list_empty(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.session_repo()
        assert repo.list() == []

    def test_exists(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.session_repo()
        assert repo.exists("s1") is False
        repo.save(SessionData(id="s1"))
        assert repo.exists("s1") is True

    def test_count(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.session_repo()
        repo.save(SessionData(id="s1"))
        repo.save(SessionData(id="s2"))
        assert repo.count() == 2

    def test_append_message(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.session_repo()
        repo.save(SessionData(id="s1", created_at="now"))
        assert repo.append_message("s1", "user", "hi") is True
        msgs = repo.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hi"

    def test_save_updates_existing(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.session_repo()
        repo.save(SessionData(id="s1", messages=[{"role": "user", "content": "a"}]))
        repo.save(SessionData(id="s1", messages=[{"role": "user", "content": "b"}]))
        loaded = repo.get("s1")
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "b"

    def test_mixed_data_types(self, sqlite_factory: RepositoryFactory):
        """SQLite blob storage should handle complex nested data."""
        repo = sqlite_factory.session_repo()
        session = SessionData(
            id="s1",
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            metadata={"source": "web", "tags": ["test"]},
        )
        assert repo.save(session) is True
        loaded = repo.get("s1")
        assert loaded is not None
        assert len(loaded.messages) == 2
        assert loaded.metadata["source"] == "web"
        assert loaded.metadata["tags"] == ["test"]


class TestSQLiteKnowledgeRepository:
    def test_search(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.knowledge_repo()
        repo.save(KnowledgeFactData(id="k1", content="Paris is capital"))
        results = repo.search("paris")
        assert len(results) == 1

    def test_search_by_topic(self, sqlite_factory: RepositoryFactory):
        repo = sqlite_factory.knowledge_repo()
        repo.save(KnowledgeFactData(id="k1", content="A", topic="astro"))
        results = repo.search_by_topic("astro")
        assert len(results) == 1


# ── Factory Tests ──


class TestRepositoryFactory:
    def test_default_backend_is_file(self, tmp_base: Path):
        factory = RepositoryFactory(base_dir=tmp_base)
        repo = factory.session_repo()
        assert isinstance(repo._backend, FileRepository)

    def test_sqlite_backend(self, tmp_base: Path):
        factory = RepositoryFactory(backend="sqlite", base_dir=tmp_base)
        repo = factory.session_repo()
        assert isinstance(repo._backend, SyncSQLiteRepository)

    def test_all_repos_created(self, file_factory: RepositoryFactory):
        assert file_factory.session_repo() is not None
        assert file_factory.conversation_repo() is not None
        assert file_factory.feedback_repo() is not None
        assert file_factory.knowledge_repo() is not None
        assert file_factory.dataset_repo() is not None
        assert file_factory.agent_repo() is not None
        assert file_factory.training_job_repo() is not None
        assert file_factory.session_context_repo() is not None

    def test_repos_persist_across_factory(self, tmp_base: Path):
        factory = RepositoryFactory(backend="file", base_dir=tmp_base)
        repo1 = factory.session_repo()
        repo1.save(SessionData(id="persist_test"))
        repo2 = factory.session_repo()
        loaded = repo2.get("persist_test")
        assert loaded is not None

    def test_singleton_factory(self):
        reset_repository_factory()
        f1 = get_repository_factory()
        f2 = get_repository_factory()
        assert f1 is f2
        reset_repository_factory()

    def test_singleton_isolation(self):
        """Each reset creates a fresh factory."""
        reset_repository_factory()
        f1 = get_repository_factory(backend="file", base_dir=Path("/tmp/a"))
        reset_repository_factory()
        f2 = get_repository_factory(backend="file", base_dir=Path("/tmp/b"))
        assert f1 is not f2
        reset_repository_factory()


# ── Cache Behavior Tests ──


class TestCacheBehavior:
    def test_cache_returns_same_object(self, cached_factory: RepositoryFactory):
        repo = cached_factory.session_repo()
        repo.save(SessionData(id="s1", created_at="now"))
        loaded1 = repo.get("s1")
        loaded2 = repo.get("s1")
        assert loaded1 is not None
        assert loaded2 is not None
        # Same object identity when cached
        assert loaded1 is loaded2

    def test_cache_invalidated_after_delete(self, cached_factory: RepositoryFactory):
        repo = cached_factory.session_repo()
        repo.save(SessionData(id="s1"))
        repo.get("s1")  # populate cache
        repo.delete("s1")
        assert repo.get("s1") is None

    def test_cache_hit(self, cached_factory: RepositoryFactory):
        repo = cached_factory.session_repo()
        repo.save(SessionData(id="s1"))
        repo.get("s1")  # populate cache
        # Second get should hit cache (not read from disk)
        loaded = repo.get("s1")
        assert loaded is not None
        assert loaded.id == "s1"


# ── SQLite Backend Direct Tests ──


class TestSyncSQLiteRepository:
    def test_init_creates_table(self, tmp_base: Path):
        db_path = tmp_base / "test.db"
        repo = SyncSQLiteRepository[SessionData](
            "sessions", db_path, serializer=SessionData
        )
        assert db_path.exists()
        # Verify table exists
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            assert cursor.fetchone() is not None
        finally:
            conn.close()

    def test_save_and_get(self, tmp_base: Path):
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        repo.save("s1", SessionData(id="s1"))
        loaded = repo.get("s1")
        assert loaded is not None
        assert loaded.id == "s1"

    def test_save_overwrite(self, tmp_base: Path):
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        repo.save("s1", SessionData(id="s1", messages=[{"role": "user", "content": "a"}]))
        repo.save("s1", SessionData(id="s1", messages=[{"role": "user", "content": "b"}]))
        loaded = repo.get("s1")
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "b"

    def test_delete(self, tmp_base: Path):
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        repo.save("s1", SessionData(id="s1"))
        assert repo.delete("s1") is True
        assert repo.get("s1") is None

    def test_delete_nonexistent(self, tmp_base: Path):
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        assert repo.delete("nope") is False

    def test_list(self, tmp_base: Path):
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        repo.save("s1", SessionData(id="s1"))
        repo.save("s2", SessionData(id="s2"))
        assert len(repo.list()) == 2

    def test_list_empty(self, tmp_base: Path):
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        assert repo.list() == []

    def test_exists(self, tmp_base: Path):
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        assert repo.exists("s1") is False
        repo.save("s1", SessionData(id="s1"))
        assert repo.exists("s1") is True

    def test_count(self, tmp_base: Path):
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        repo.save("s1", SessionData(id="s1"))
        repo.save("s2", SessionData(id="s2"))
        assert repo.count() == 2

    def test_search(self, tmp_base: Path):
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        repo.save("s1", SessionData(id="s1", messages=[{"role": "user", "content": "hello"}]))
        results = repo.search("hello")
        assert len(results) == 1

    def test_cross_table_isolation(self, tmp_base: Path):
        """Different tables should not interfere."""
        s_repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        k_repo = SyncSQLiteRepository[KnowledgeFactData](
            "knowledge", tmp_base / "test.db", serializer=KnowledgeFactData
        )
        s_repo.save("s1", SessionData(id="s1"))
        k_repo.save("k1", KnowledgeFactData(id="k1", content="fact"))
        assert len(s_repo.list()) == 1
        assert len(k_repo.list()) == 1
        assert s_repo.get("k1") is None

    def test_cache_ttl(self, tmp_base: Path):
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData, cache_ttl=60
        )
        repo.save("s1", SessionData(id="s1"))
        # Populate cache
        cached = repo.get("s1")
        # Direct DB mutation (simulate external change)
        conn = sqlite3.connect(str(tmp_base / "test.db"))
        try:
            conn.execute(
                "UPDATE sessions SET data = ? WHERE id = ?",
                (json.dumps({"id": "s1", "messages": [], "metadata": {}, "created_at": "new", "updated_at": "new"}), "s1"),
            )
            conn.commit()
        finally:
            conn.close()
        # Cache should still return old object
        still_cached = repo.get("s1")
        # The cache doesn't know about the external change, so it returns the cached version
        # (This is expected cache behavior — TTL expires after cache_ttl seconds)
        assert still_cached is not None
        assert still_cached.created_at == ""  # original cached value

    def test_concurrent_safe(self, tmp_base: Path):
        """Basic thread-safety: concurrent writes should not corrupt."""
        import threading
        repo = SyncSQLiteRepository[SessionData](
            "sessions", tmp_base / "test.db", serializer=SessionData
        )
        errors = []
        def writer(n: int):
            try:
                repo.save(f"s{n}", SessionData(id=f"s{n}"))
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert repo.count() == 10


# ── Edge Cases ──


class TestEdgeCases:
    def test_large_messages(self, file_factory: RepositoryFactory):
        """Large message content should serialize/deserialize correctly."""
        repo = file_factory.session_repo()
        large_content = "x" * 100_000
        session = SessionData(
            id="s1",
            messages=[{"role": "user", "content": large_content}],
        )
        assert repo.save(session) is True
        loaded = repo.get("s1")
        assert loaded is not None
        assert len(loaded.messages[0]["content"]) == 100_000

    def test_special_characters(self, file_factory: RepositoryFactory):
        """Unicode and special characters should survive round-trip."""
        repo = file_factory.knowledge_repo()
        fact = KnowledgeFactData(
            id="k1",
            content="Café résumé naïve 日本語 😊",
            topic="unicode",
        )
        assert repo.save(fact) is True
        loaded = repo.get("k1")
        assert loaded is not None
        assert loaded.content == "Café résumé naïve 日本語 😊"

    def test_empty_messages_list(self, file_factory: RepositoryFactory):
        repo = file_factory.session_repo()
        repo.save(SessionData(id="s1"))
        loaded = repo.get("s1")
        assert loaded is not None
        assert loaded.messages == []

    def test_multiple_saves_same_key(self, file_factory: RepositoryFactory):
        """Re-saving the same key should overwrite."""
        repo = file_factory.session_repo()
        for i in range(5):
            repo.save(SessionData(id="s1", messages=[{"role": "user", "content": f"msg{i}"}]))
        loaded = repo.get("s1")
        assert loaded is not None
        assert loaded.messages[0]["content"] == "msg4"

    def test_backend_isolation(self, tmp_base: Path):
        """File and SQLite repos should use independent storage."""
        file_repo = RepositoryFactory(backend="file", base_dir=tmp_base / "file").session_repo()
        sqlite_repo = RepositoryFactory(backend="sqlite", base_dir=tmp_base / "sqlite").session_repo()
        file_repo.save(SessionData(id="s1"))
        assert sqlite_repo.get("s1") is None

    def test_messages_are_independent_copies(self, file_factory: RepositoryFactory):
        """Modifying loaded messages should not affect stored data."""
        repo = file_factory.session_repo()
        repo.save(SessionData(id="s1", messages=[{"role": "user", "content": "hello"}]))
        loaded = repo.get("s1")
        assert loaded is not None
        loaded.messages.append({"role": "assistant", "content": "hi"})
        loaded_again = repo.get("s1")
        assert loaded_again is not None
        assert len(loaded_again.messages) == 1  # original unchanged


import sqlite3  # noqa: E402 (needed for TestSyncSQLiteRepository)
