"""Tests for entity_repositories — typed wrappers for domain repositories."""

import json
import time
from pathlib import Path
import pytest

from domains.infrastructure.entity_repositories import (
    KnowledgeEntry,
    FeedState,
    KnowledgeRepository,
    SessionData,
    MessageRecord,
    SessionRepository,
    FeedbackRecord,
    FeedbackRepository,
    DatasetMetadata,
    DatasetRepository,
)


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass round-trips
# ══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeEntry:
    def test_to_dict_and_back(self):
        entry = KnowledgeEntry(
            id="k1", content="test fact", topic="ai",
            source="manual", url="http://x", timestamp=1.0,
            importance=0.8, tags=["tag1"],
        )
        d = entry.to_dict()
        assert d["id"] == "k1"
        assert d["content"] == "test fact"
        assert d["tags"] == ["tag1"]
        restored = KnowledgeEntry.from_dict(d)
        assert restored.id == entry.id
        assert restored.content == entry.content
        assert restored.tags == entry.tags

    def test_from_dict_extra_keys_ignored(self):
        d = {"id": "x", "content": "y", "unknown_field": 123}
        entry = KnowledgeEntry.from_dict(d)
        assert entry.id == "x"
        assert entry.content == "y"

    def test_defaults(self):
        entry = KnowledgeEntry(id="a", content="b")
        assert entry.topic == "general"
        assert entry.source == "manual"
        assert entry.importance == 0.5
        assert entry.tags == []


class TestFeedState:
    def test_to_dict_and_back(self):
        feed = FeedState(url="http://rss", title="Feed", last_fetched=100.0, poll_interval=600.0, enabled=False)
        d = feed.to_dict()
        restored = FeedState.from_dict(d)
        assert restored.url == "http://rss"
        assert restored.enabled is False

    def test_defaults(self):
        feed = FeedState(url="http://x")
        assert feed.title == ""
        assert feed.poll_interval == 3600.0
        assert feed.enabled is True


class TestSessionData:
    def test_to_dict_and_back(self):
        s = SessionData(session_id="s1", message_count=5, title="Test", model="gpt2")
        d = s.to_dict()
        restored = SessionData.from_dict(d)
        assert restored.session_id == "s1"
        assert restored.message_count == 5

    def test_from_dict_extra_keys(self):
        d = {"session_id": "s2", "extra": True}
        s = SessionData.from_dict(d)
        assert s.session_id == "s2"


class TestMessageRecord:
    def test_to_dict(self):
        m = MessageRecord(role="user", content="hello", timestamp=1.0)
        d = m.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"
        assert d["timestamp"] == 1.0


class TestFeedbackRecord:
    def test_to_dict_and_back(self):
        f = FeedbackRecord(id="f1", message_id="m1", rating="thumbs_up", session_id="s1", quality_score=0.9)
        d = f.to_dict()
        restored = FeedbackRecord.from_dict(d)
        assert restored.id == "f1"
        assert restored.rating == "thumbs_up"

    def test_defaults(self):
        f = FeedbackRecord(id="f2", message_id="m2", rating="neutral")
        assert f.session_id == ""
        assert f.quality_score == 0.0


class TestDatasetMetadata:
    def test_to_dict_and_back(self):
        ds = DatasetMetadata(id="d1", name="train", description="desc", record_count=100, tags=["t1"])
        d = ds.to_dict()
        restored = DatasetMetadata.from_dict(d)
        assert restored.id == "d1"
        assert restored.name == "train"
        assert restored.tags == ["t1"]

    def test_defaults(self):
        ds = DatasetMetadata(id="d2")
        assert ds.format == "jsonl"
        assert ds.record_count == 0


# ══════════════════════════════════════════════════════════════════════════════
# KnowledgeRepository
# ══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeRepository:
    def test_save_and_get_fact(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        entry = KnowledgeEntry(id="k1", content="Python is great", topic="programming")
        assert repo.save_fact(entry) is True
        got = repo.get_fact("k1")
        assert got is not None
        assert got.content == "Python is great"

    def test_get_nonexistent_returns_none(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        assert repo.get_fact("missing") is None

    def test_list_facts_all(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        repo.save_fact(KnowledgeEntry(id="k1", content="a", topic="ai"))
        repo.save_fact(KnowledgeEntry(id="k2", content="b", topic="ml"))
        assert len(repo.list_facts()) == 2

    def test_list_facts_by_topic(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        repo.save_fact(KnowledgeEntry(id="k1", content="a", topic="ai"))
        repo.save_fact(KnowledgeEntry(id="k2", content="b", topic="ml"))
        repo.save_fact(KnowledgeEntry(id="k3", content="c", topic="ai"))
        ai_facts = repo.list_facts(topic="ai")
        assert len(ai_facts) == 2

    def test_delete_fact(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        repo.save_fact(KnowledgeEntry(id="k1", content="x"))
        assert repo.delete_fact("k1") is True
        assert repo.get_fact("k1") is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        assert repo.delete_fact("missing") is False

    def test_count_facts(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        assert repo.count_facts() == 0
        repo.save_fact(KnowledgeEntry(id="k1", content="a"))
        repo.save_fact(KnowledgeEntry(id="k2", content="b"))
        assert repo.count_facts() == 2

    def test_search_facts(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        repo.save_fact(KnowledgeEntry(id="k1", content="Python tutorial", topic="programming"))
        repo.save_fact(KnowledgeEntry(id="k2", content="Java basics", topic="programming"))
        results = repo.search_facts("Python")
        assert len(results) == 1
        assert results[0].id == "k1"


class TestKnowledgeFeeds:
    def test_save_and_get_feed(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        feed = FeedState(url="http://rss.example.com", title="My Feed")
        repo.save_feed(feed)
        got = repo.get_feed("http://rss.example.com")
        assert got is not None
        assert got.title == "My Feed"

    def test_list_feeds(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        repo.save_feed(FeedState(url="http://a"))
        repo.save_feed(FeedState(url="http://b"))
        assert len(repo.list_feeds()) == 2

    def test_delete_feed(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        repo.save_feed(FeedState(url="http://a"))
        assert repo.delete_feed("http://a") is True
        assert repo.get_feed("http://a") is None

    def test_delete_nonexistent_feed(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        assert repo.delete_feed("http://missing") is False

    def test_feeds_persist_to_disk(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kr")
        repo.save_feed(FeedState(url="http://a", title="A"))
        # Reload from disk
        repo2 = KnowledgeRepository(tmp_path / "kr")
        got = repo2.get_feed("http://a")
        assert got is not None
        assert got.title == "A"

    def test_corrupted_feeds_file_does_not_crash(self, tmp_path):
        feeds_path = tmp_path / "kr" / "feeds.json"
        feeds_path.parent.mkdir(parents=True, exist_ok=True)
        feeds_path.write_text("NOT JSON{{{")
        repo = KnowledgeRepository(tmp_path / "kr")
        assert repo.list_feeds() == []


# ══════════════════════════════════════════════════════════════════════════════
# SessionRepository
# ══════════════════════════════════════════════════════════════════════════════


class TestSessionRepository:
    def test_save_and_get_session(self, tmp_path):
        repo = SessionRepository()
        s = SessionData(session_id="s1", title="Chat 1", model="gpt2")
        assert repo.save_session(s) is True
        got = repo.get_session("s1")
        assert got is not None
        assert got.title == "Chat 1"

    def test_get_nonexistent_returns_none(self, tmp_path):
        repo = SessionRepository()
        assert repo.get_session("missing") is None

    def test_list_sessions(self, tmp_path):
        repo = SessionRepository()
        repo.save_session(SessionData(session_id="s1"))
        repo.save_session(SessionData(session_id="s2"))
        assert len(repo.list_sessions()) == 2

    def test_delete_session(self, tmp_path):
        repo = SessionRepository()
        repo.save_session(SessionData(session_id="s1"))
        assert repo.delete_session("s1") is True
        assert repo.get_session("s1") is None

    def test_add_message_updates_count(self, tmp_path):
        repo = SessionRepository()
        repo.save_session(SessionData(session_id="s1"))
        repo.add_message("s1", MessageRecord(role="user", content="hi"))
        repo.add_message("s1", MessageRecord(role="assistant", content="hello"))
        msgs = repo.get_messages("s1")
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_get_messages_empty(self, tmp_path):
        repo = SessionRepository()
        assert repo.get_messages("nonexistent") == []

    def test_search_sessions(self, tmp_path):
        repo = SessionRepository()
        repo.save_session(SessionData(session_id="s1", title="Python chat"))
        repo.save_session(SessionData(session_id="s2", title="Java chat"))
        results = repo.search_sessions("Python")
        assert len(results) == 1
        assert results[0].session_id == "s1"


class TestSessionPersistence:
    def test_persist_to_disk(self, tmp_path):
        persist_dir = tmp_path / "sessions"
        repo = SessionRepository(persist_dir=persist_dir)
        repo.save_session(SessionData(session_id="s1", title="Test"))
        repo.add_message("s1", MessageRecord(role="user", content="hi"))

        # Reload from disk
        repo2 = SessionRepository(persist_dir=persist_dir)
        got = repo2.get_session("s1")
        assert got is not None
        assert got.title == "Test"
        msgs = repo2.get_messages("s1")
        assert len(msgs) == 1

    def test_delete_removes_from_disk(self, tmp_path):
        persist_dir = tmp_path / "sessions"
        repo = SessionRepository(persist_dir=persist_dir)
        repo.save_session(SessionData(session_id="s1"))
        assert (persist_dir / "s1.json").exists()
        repo.delete_session("s1")
        assert not (persist_dir / "s1.json").exists()

    def test_corrupted_session_file_does_not_crash(self, tmp_path):
        persist_dir = tmp_path / "sessions"
        persist_dir.mkdir(parents=True, exist_ok=True)
        (persist_dir / "bad.json").write_text("NOT JSON{{{")
        repo = SessionRepository(persist_dir=persist_dir)
        assert repo.list_sessions() == []


# ══════════════════════════════════════════════════════════════════════════════
# FeedbackRepository (in-memory only — no MogDB)
# ══════════════════════════════════════════════════════════════════════════════


class TestFeedbackRepository:
    def test_save_and_get_feedback(self):
        repo = FeedbackRepository()
        f = FeedbackRecord(id="f1", message_id="m1", rating="thumbs_up")
        assert repo.save_feedback(f) is True
        got = repo.get_feedback("f1")
        assert got is not None
        assert got.rating == "thumbs_up"

    def test_get_nonexistent_returns_none(self):
        repo = FeedbackRepository()
        assert repo.get_feedback("missing") is None

    def test_list_feedback_all(self):
        repo = FeedbackRepository()
        repo.save_feedback(FeedbackRecord(id="f1", message_id="m1", rating="thumbs_up"))
        repo.save_feedback(FeedbackRecord(id="f2", message_id="m2", rating="thumbs_down"))
        assert len(repo.list_feedback()) == 2

    def test_list_feedback_by_session(self):
        repo = FeedbackRepository()
        repo.save_feedback(FeedbackRecord(id="f1", message_id="m1", rating="thumbs_up", session_id="s1"))
        repo.save_feedback(FeedbackRecord(id="f2", message_id="m2", rating="thumbs_down", session_id="s2"))
        s1_feedback = repo.list_feedback(session_id="s1")
        assert len(s1_feedback) == 1
        assert s1_feedback[0].session_id == "s1"

    def test_delete_feedback(self):
        repo = FeedbackRepository()
        repo.save_feedback(FeedbackRecord(id="f1", message_id="m1", rating="thumbs_up"))
        assert repo.delete_feedback("f1") is True
        assert repo.get_feedback("f1") is None

    def test_delete_nonexistent(self):
        repo = FeedbackRepository()
        assert repo.delete_feedback("missing") is False

    def test_get_stats(self):
        repo = FeedbackRepository()
        repo.save_feedback(FeedbackRecord(id="f1", message_id="m1", rating="thumbs_up"))
        repo.save_feedback(FeedbackRecord(id="f2", message_id="m2", rating="thumbs_up"))
        repo.save_feedback(FeedbackRecord(id="f3", message_id="m3", rating="thumbs_down"))
        stats = repo.get_stats()
        assert stats["total_feedback"] == 3
        assert stats["thumbs_up"] == 2
        assert stats["thumbs_down"] == 1

    def test_upsert_feedback(self):
        repo = FeedbackRepository()
        f1 = FeedbackRecord(id="f1", message_id="m1", rating="thumbs_up")
        repo.save_feedback(f1)
        # Save again with updated rating
        f2 = FeedbackRecord(id="f1", message_id="m1", rating="thumbs_down")
        repo.save_feedback(f2)
        got = repo.get_feedback("f1")
        assert got is not None
        assert got.rating == "thumbs_down"


# ══════════════════════════════════════════════════════════════════════════════
# DatasetRepository
# ══════════════════════════════════════════════════════════════════════════════


class TestDatasetRepository:
    def test_save_and_get(self, tmp_path):
        repo = DatasetRepository(tmp_path / "ds")
        ds = DatasetMetadata(id="d1", name="train", record_count=100)
        assert repo.save(ds) is True
        got = repo.get("d1")
        assert got is not None
        assert got.name == "train"
        assert got.record_count == 100

    def test_get_nonexistent_returns_none(self, tmp_path):
        repo = DatasetRepository(tmp_path / "ds")
        assert repo.get("missing") is None

    def test_list(self, tmp_path):
        repo = DatasetRepository(tmp_path / "ds")
        repo.save(DatasetMetadata(id="d1", name="train"))
        repo.save(DatasetMetadata(id="d2", name="val"))
        assert len(repo.list()) == 2

    def test_delete(self, tmp_path):
        repo = DatasetRepository(tmp_path / "ds")
        repo.save(DatasetMetadata(id="d1"))
        assert repo.delete("d1") is True
        assert repo.get("d1") is None

    def test_search(self, tmp_path):
        repo = DatasetRepository(tmp_path / "ds")
        repo.save(DatasetMetadata(id="d1", name="training data", description="Main dataset"))
        repo.save(DatasetMetadata(id="d2", name="validation set"))
        results = repo.search("training")
        assert len(results) == 1
        assert results[0].id == "d1"
