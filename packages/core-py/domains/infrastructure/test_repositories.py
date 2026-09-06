"""Tests for Data Repository layer (repository.py + entity_repositories.py)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from domains.infrastructure.repository import (
    CachedRepository,
    FileRepository,
    JsonSerializer,
    MemoryRepository,
    Migration,
    MigrationRunner,
)
from domains.infrastructure.entity_repositories import (
    DatasetMetadata,
    DatasetRepository,
    FeedbackRecord,
    FeedbackRepository,
    FeedState,
    KnowledgeEntry,
    KnowledgeRepository,
    MessageRecord,
    SessionData,
    SessionRepository,
)


# ── Test dataclasses ──


@dataclass
class SimpleRecord:
    id: str
    value: str = ""
    count: int = 0


# ── JsonSerializer tests ──


class TestJsonSerializer:
    def test_serialize_dataclass(self):
        s = JsonSerializer(SimpleRecord)
        obj = SimpleRecord(id="r1", value="hello", count=5)
        data = s.serialize(obj)
        assert data == {"id": "r1", "value": "hello", "count": 5}

    def test_deserialize_dataclass(self):
        s = JsonSerializer(SimpleRecord)
        data = {"id": "r1", "value": "hello", "count": 5}
        obj = s.deserialize(data)
        assert obj.id == "r1"
        assert obj.value == "hello"
        assert obj.count == 5

    def test_serialize_dict(self):
        s = JsonSerializer(dict)
        obj = {"key": "val"}
        assert s.serialize(obj) == {"key": "val"}

    def test_deserialize_unknown_class(self):
        class Custom:
            def __init__(self, data):
                self.data = data

        s = JsonSerializer(Custom)
        obj = s.deserialize({"foo": 1})
        assert obj.data == {"foo": 1}


# ── Migration tests ──


class TestMigrationRunner:
    def test_no_migrations(self):
        runner = MigrationRunner()
        assert runner.latest_version == 0
        data = {"value": 1}
        assert runner.run(data) == {"value": 1}

    def test_single_migration(self):
        m = Migration(1, "add flag", lambda d: {**d, "flagged": True})
        runner = MigrationRunner([m])
        assert runner.latest_version == 1
        result = runner.run({"value": 1})
        assert result["flagged"] is True
        assert result["_schema_version"] == 1

    def test_migrations_run_in_order(self):
        m1 = Migration(1, "v1", lambda d: {**d, "steps": d.get("steps", []) + ["v1"]})
        m2 = Migration(2, "v2", lambda d: {**d, "steps": d.get("steps", []) + ["v2"]})
        m3 = Migration(3, "v3", lambda d: {**d, "steps": d.get("steps", []) + ["v3"]})
        runner = MigrationRunner([m3, m1, m2])
        result = runner.run({})
        assert result["steps"] == ["v1", "v2", "v3"]
        assert result["_schema_version"] == 3

    def test_skips_already_applied(self):
        m1 = Migration(1, "v1", lambda d: {**d, "steps": d.get("steps", []) + ["v1"]})
        m2 = Migration(2, "v2", lambda d: {**d, "steps": d.get("steps", []) + ["v2"]})
        runner = MigrationRunner([m1, m2])
        result = runner.run({"_schema_version": 1})
        assert result["steps"] == ["v2"]
        assert result["_schema_version"] == 2

    def test_migration_failure_propagates(self):
        def fail(d):
            raise RuntimeError("migration broke")

        m = Migration(1, "bad", fail)
        runner = MigrationRunner([m])
        with pytest.raises(RuntimeError, match="migration broke"):
            runner.run({})

    def test_add_migration(self):
        runner = MigrationRunner()
        runner.add(Migration(1, "first", lambda d: d))
        assert runner.latest_version == 1
        runner.add(Migration(2, "second", lambda d: d))
        assert runner.latest_version == 2


# ── MemoryRepository tests ──


class TestMemoryRepository:
    def test_get_set(self):
        repo = MemoryRepository[SimpleRecord]()
        repo.save("r1", SimpleRecord(id="r1", value="a"))
        assert repo.get("r1").value == "a"

    def test_get_missing(self):
        repo = MemoryRepository[SimpleRecord]()
        assert repo.get("missing") is None

    def test_list(self):
        repo = MemoryRepository[SimpleRecord]()
        repo.save("a", SimpleRecord(id="a", value="x"))
        repo.save("b", SimpleRecord(id="b", value="y"))
        items = repo.list()
        assert len(items) == 2
        assert {i.id for i in items} == {"a", "b"}

    def test_delete(self):
        repo = MemoryRepository[SimpleRecord]()
        repo.save("r1", SimpleRecord(id="r1"))
        assert repo.delete("r1") is True
        assert repo.get("r1") is None
        assert repo.delete("r1") is False

    def test_search(self):
        repo = MemoryRepository[str]()
        repo.save("a", "hello world")
        repo.save("b", "goodbye")
        results = repo.search("hello")
        assert results == ["hello world"]

    def test_clear(self):
        repo = MemoryRepository[SimpleRecord]()
        repo.save("a", SimpleRecord(id="a"))
        repo.save("b", SimpleRecord(id="b"))
        repo.clear()
        assert repo.list() == []


# ── FileRepository tests ──


class TestFileRepository:
    def test_crud(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        repo.save("item1", {"id": "item1", "name": "foo"})
        obj = repo.get("item1")
        assert obj == {"id": "item1", "name": "foo"}

    def test_get_missing(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        assert repo.get("nope") is None

    def test_list(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        repo.save("a", {"id": "a"})
        repo.save("b", {"id": "b"})
        items = repo.list()
        assert len(items) == 2

    def test_delete(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        repo.save("a", {"id": "a"})
        assert repo.delete("a") is True
        assert repo.get("a") is None
        assert repo.delete("a") is False

    def test_exists(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        repo.save("a", {"id": "a"})
        assert repo.exists("a") is True
        assert repo.exists("b") is False

    def test_count(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        assert repo.count() == 0
        repo.save("a", {"id": "a"})
        repo.save("b", {"id": "b"})
        assert repo.count() == 2

    def test_keys(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        repo.save("x", {"id": "x"})
        repo.save("y", {"id": "y"})
        assert sorted(repo.keys()) == ["x", "y"]

    def test_search(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        repo.save("a", {"id": "a", "name": "alpha"})
        repo.save("b", {"id": "b", "name": "beta"})
        results = repo.search("alpha")
        assert len(results) == 1
        assert results[0]["name"] == "alpha"

    def test_search_specific_fields(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        repo.save("a", {"id": "a", "name": "alpha", "desc": "hidden"})
        results = repo.search("alpha", fields=["name"])
        assert len(results) == 1

    def test_corrupt_file_returns_none(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        (tmp_path / "repo" / "bad.json").write_text("not valid json {{{")
        assert repo.get("bad") is None

    def test_empty_file_returns_none(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        (tmp_path / "repo" / "empty.json").write_text("")
        assert repo.get("empty") is None

    def test_migration_applied_on_read(self, tmp_path: Path):
        m = Migration(1, "add field", lambda d: {**d, "migrated": True})
        runner = MigrationRunner([m])
        repo = FileRepository[dict](tmp_path / "repo", migration_runner=runner)
        repo.save("a", {"id": "a", "value": 1})
        obj = repo.get("a")
        assert obj["migrated"] is True
        assert obj.get("_schema_version") is None

    def test_atomic_write_no_partial(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        repo.save("a", {"id": "a"})
        assert not (tmp_path / "repo" / "a.json.tmp").exists()
        assert (tmp_path / "repo" / "a.json").exists()

    def test_custom_suffix(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo", key_suffix=".yaml")
        repo.save("a", {"id": "a"})
        assert (tmp_path / "repo" / "a.yaml").exists()


# ── CachedRepository tests ──


class TestCachedRepository:
    def test_caches_get(self, tmp_path: Path):
        inner = FileRepository[dict](tmp_path / "repo")
        inner.save("a", {"id": "a", "v": 1})
        cached = CachedRepository(inner, ttl=10.0)
        assert cached.get("a") == {"id": "a", "v": 1}

    def test_cache_invalidation(self, tmp_path: Path):
        inner = FileRepository[dict](tmp_path / "repo")
        inner.save("a", {"id": "a", "v": 1})
        cached = CachedRepository(inner, ttl=10.0)
        cached.get("a")
        inner.save("a", {"id": "a", "v": 2})
        cached.invalidate()
        assert cached.get("a") == {"id": "a", "v": 2}

    def test_list_caching(self, tmp_path: Path):
        inner = FileRepository[dict](tmp_path / "repo")
        inner.save("a", {"id": "a"})
        cached = CachedRepository(inner, ttl=10.0)
        assert len(cached.list()) == 1
        inner.save("b", {"id": "b"})
        assert len(cached.list()) == 1
        cached.invalidate()
        assert len(cached.list()) == 2

    def test_delete_invalidates(self, tmp_path: Path):
        inner = FileRepository[dict](tmp_path / "repo")
        inner.save("a", {"id": "a"})
        cached = CachedRepository(inner, ttl=10.0)
        cached.get("a")
        cached.delete("a")
        assert cached.get("a") is None


# ── KnowledgeRepository tests ──


class TestKnowledgeRepository:
    def test_facts_crud(self, tmp_path: Path):
        repo = KnowledgeRepository(tmp_path / "knowledge")
        entry = KnowledgeEntry(id="f1", content="Python is great", topic="tech", source="manual")
        assert repo.save_fact(entry) is True
        loaded = repo.get_fact("f1")
        assert loaded is not None
        assert loaded.content == "Python is great"
        assert loaded.topic == "tech"

    def test_facts_list_and_filter(self, tmp_path: Path):
        repo = KnowledgeRepository(tmp_path / "knowledge")
        repo.save_fact(KnowledgeEntry(id="f1", content="A", topic="tech"))
        repo.save_fact(KnowledgeEntry(id="f2", content="B", topic="food"))
        repo.save_fact(KnowledgeEntry(id="f3", content="C", topic="tech"))
        all_facts = repo.list_facts()
        assert len(all_facts) == 3
        tech_facts = repo.list_facts(topic="tech")
        assert len(tech_facts) == 2

    def test_facts_delete(self, tmp_path: Path):
        repo = KnowledgeRepository(tmp_path / "knowledge")
        repo.save_fact(KnowledgeEntry(id="f1", content="X"))
        assert repo.delete_fact("f1") is True
        assert repo.get_fact("f1") is None
        assert repo.delete_fact("f1") is False

    def test_facts_search(self, tmp_path: Path):
        repo = KnowledgeRepository(tmp_path / "knowledge")
        repo.save_fact(KnowledgeEntry(id="f1", content="Python rocks"))
        repo.save_fact(KnowledgeEntry(id="f2", content="Java is okay"))
        results = repo.search_facts("Python")
        assert len(results) == 1
        assert results[0].id == "f1"

    def test_facts_count(self, tmp_path: Path):
        repo = KnowledgeRepository(tmp_path / "knowledge")
        assert repo.count_facts() == 0
        repo.save_fact(KnowledgeEntry(id="f1", content="A"))
        repo.save_fact(KnowledgeEntry(id="f2", content="B"))
        assert repo.count_facts() == 2

    def test_feeds_crud(self, tmp_path: Path):
        repo = KnowledgeRepository(tmp_path / "knowledge")
        feed = FeedState(url="https://example.com/rss", title="Example Feed")
        repo.save_feed(feed)
        loaded = repo.get_feed("https://example.com/rss")
        assert loaded is not None
        assert loaded.title == "Example Feed"
        assert len(repo.list_feeds()) == 1
        repo.delete_feed("https://example.com/rss")
        assert repo.get_feed("https://example.com/rss") is None

    def test_feeds_persistence(self, tmp_path: Path):
        repo1 = KnowledgeRepository(tmp_path / "knowledge")
        repo1.save_feed(FeedState(url="https://a.com", title="A"))
        repo2 = KnowledgeRepository(tmp_path / "knowledge")
        assert repo2.get_feed("https://a.com") is not None

    def test_get_missing_fact(self, tmp_path: Path):
        repo = KnowledgeRepository(tmp_path / "knowledge")
        assert repo.get_fact("nope") is None


# ── SessionRepository tests ──


class TestSessionRepository:
    def test_session_crud(self, tmp_path: Path):
        repo = SessionRepository()
        session = SessionData(session_id="s1", title="Test Session", model="gpt2")
        repo.save_session(session)
        loaded = repo.get_session("s1")
        assert loaded is not None
        assert loaded.title == "Test Session"

    def test_session_list(self, tmp_path: Path):
        repo = SessionRepository()
        repo.save_session(SessionData(session_id="s1", title="A"))
        repo.save_session(SessionData(session_id="s2", title="B"))
        assert len(repo.list_sessions()) == 2

    def test_session_delete(self, tmp_path: Path):
        repo = SessionRepository()
        repo.save_session(SessionData(session_id="s1"))
        assert repo.delete_session("s1") is True
        assert repo.get_session("s1") is None
        assert repo.delete_session("s1") is False

    def test_messages(self, tmp_path: Path):
        repo = SessionRepository()
        repo.save_session(SessionData(session_id="s1"))
        msg = MessageRecord(role="user", content="hello")
        repo.add_message("s1", msg)
        msgs = repo.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0].content == "hello"

    def test_message_updates_metadata(self, tmp_path: Path):
        repo = SessionRepository()
        repo.save_session(SessionData(session_id="s1", message_count=0))
        repo.add_message("s1", MessageRecord(role="user", content="hi"))
        meta = repo.get_session("s1")
        assert meta.message_count == 1

    def test_search(self, tmp_path: Path):
        repo = SessionRepository()
        repo.save_session(SessionData(session_id="s1", title="Alpha Chat"))
        repo.save_session(SessionData(session_id="s2", title="Beta Talk"))
        results = repo.search_sessions("Alpha")
        assert len(results) == 1
        assert results[0].title == "Alpha Chat"

    def test_persistence(self, tmp_path: Path):
        dir1 = tmp_path / "sessions1"
        repo1 = SessionRepository(persist_dir=dir1)
        repo1.save_session(SessionData(session_id="s1", title="Persisted"))
        repo1.add_message("s1", MessageRecord(role="user", content="stored"))

        tmp_path / "sessions2"
        repo2 = SessionRepository(persist_dir=dir1)
        loaded = repo2.get_session("s1")
        assert loaded is not None
        assert loaded.title == "Persisted"
        msgs = repo2.get_messages("s1")
        assert len(msgs) == 1

    def test_no_persistence(self):
        repo = SessionRepository(persist_dir=None)
        repo.save_session(SessionData(session_id="s1"))
        repo.add_message("s1", MessageRecord(role="user", content="hi"))
        assert repo.get_session("s1") is not None


# ── FeedbackRepository tests ──


class TestFeedbackRepository:
    def test_crud_in_memory(self):
        repo = FeedbackRepository()
        record = FeedbackRecord(id="fb1", message_id="m1", rating="thumbs_up")
        assert repo.save_feedback(record) is True
        loaded = repo.get_feedback("fb1")
        assert loaded is not None
        assert loaded.rating == "thumbs_up"

    def test_list(self):
        repo = FeedbackRepository()
        repo.save_feedback(FeedbackRecord(id="fb1", message_id="m1", rating="thumbs_up", session_id="s1"))
        repo.save_feedback(FeedbackRecord(id="fb2", message_id="m2", rating="thumbs_down", session_id="s2"))
        all_fb = repo.list_feedback()
        assert len(all_fb) == 2
        s1_fb = repo.list_feedback(session_id="s1")
        assert len(s1_fb) == 1

    def test_delete(self):
        repo = FeedbackRepository()
        repo.save_feedback(FeedbackRecord(id="fb1", message_id="m1", rating="thumbs_up"))
        assert repo.delete_feedback("fb1") is True
        assert repo.get_feedback("fb1") is None
        assert repo.delete_feedback("fb1") is False

    def test_stats(self):
        repo = FeedbackRepository()
        repo.save_feedback(FeedbackRecord(id="fb1", message_id="m1", rating="thumbs_up"))
        repo.save_feedback(FeedbackRecord(id="fb2", message_id="m2", rating="thumbs_up"))
        repo.save_feedback(FeedbackRecord(id="fb3", message_id="m3", rating="thumbs_down"))
        stats = repo.get_stats()
        assert stats["total_feedback"] == 3
        assert stats["thumbs_up"] == 2
        assert stats["thumbs_down"] == 1

    def test_upsert(self):
        repo = FeedbackRepository()
        repo.save_feedback(FeedbackRecord(id="fb1", message_id="m1", rating="thumbs_up"))
        repo.save_feedback(FeedbackRecord(id="fb1", message_id="m1", rating="thumbs_down"))
        loaded = repo.get_feedback("fb1")
        assert loaded.rating == "thumbs_down"
        assert repo.list_feedback().__len__() == 1


# ── DatasetRepository tests ──


class TestDatasetRepository:
    def test_crud(self, tmp_path: Path):
        repo = DatasetRepository(tmp_path / "datasets")
        meta = DatasetMetadata(id="d1", name="My Dataset", record_count=100)
        assert repo.save(meta) is True
        loaded = repo.get("d1")
        assert loaded is not None
        assert loaded.name == "My Dataset"
        assert loaded.record_count == 100

    def test_list(self, tmp_path: Path):
        repo = DatasetRepository(tmp_path / "datasets")
        repo.save(DatasetMetadata(id="d1", name="Alpha"))
        repo.save(DatasetMetadata(id="d2", name="Beta"))
        assert len(repo.list()) == 2

    def test_delete(self, tmp_path: Path):
        repo = DatasetRepository(tmp_path / "datasets")
        repo.save(DatasetMetadata(id="d1"))
        assert repo.delete("d1") is True
        assert repo.get("d1") is None

    def test_search(self, tmp_path: Path):
        repo = DatasetRepository(tmp_path / "datasets")
        repo.save(DatasetMetadata(id="d1", name="Training Data"))
        repo.save(DatasetMetadata(id="d2", name="Validation Set"))
        results = repo.search("Training")
        assert len(results) == 1
        assert results[0].name == "Training Data"

    def test_persistence(self, tmp_path: Path):
        repo1 = DatasetRepository(tmp_path / "datasets")
        repo1.save(DatasetMetadata(id="d1", name="Persisted"))
        repo2 = DatasetRepository(tmp_path / "datasets")
        loaded = repo2.get("d1")
        assert loaded is not None
        assert loaded.name == "Persisted"


# ── Thread safety smoke test ──


class TestThreadSafety:
    def test_concurrent_saves(self, tmp_path: Path):
        repo = FileRepository[dict](tmp_path / "repo")
        errors: list[Exception] = []

        def save_item(i: int):
            try:
                repo.save(f"item_{i}", {"id": f"item_{i}", "value": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_item, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert repo.count() == 50

    def test_concurrent_memory_saves(self):
        repo = MemoryRepository[dict]()
        errors: list[Exception] = []

        def save_item(i: int):
            try:
                repo.save(f"item_{i}", {"id": f"item_{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_item, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(repo.list()) == 100
