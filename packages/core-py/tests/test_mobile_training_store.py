"""Tests for MobileTrainingStore (MogDB-backed)."""

import os
import tempfile
import time

import pytest

from domains.training.mobile_training_store import MobileTrainingStore


@pytest.fixture
def store(tmp_path):
    """Create a temporary MobileTrainingStore."""
    s = MobileTrainingStore(str(tmp_path / "training_test"))
    yield s
    s.close()


class TestMobileTrainingStore:
    def test_add_pair_returns_id(self, store):
        """add_pair returns a non-empty string ID."""
        doc_id = store.add_pair("hello", "hi", "s1")
        assert doc_id, "add_pair should return a truthy ID"

    def test_add_pair_stores_fields(self, store):
        """add_pair stores all fields correctly."""
        doc_id = store.add_pair("hello", "hi", "s1", quality=1, model="gpt2")
        pair = store.get_pair(doc_id)
        assert pair is not None
        assert pair["user_msg"] == "hello"
        assert pair["assistant_msg"] == "hi"
        assert pair["session_id"] == "s1"
        assert pair["quality"] == 1
        assert pair["model"] == "gpt2"
        assert pair["synced"] is False
        assert pair["used_for_training"] is False
        assert "timestamp" in pair

    def test_add_pair_default_quality(self, store):
        """add_pair defaults quality to 0."""
        doc_id = store.add_pair("a", "b", "s1")
        pair = store.get_pair(doc_id)
        assert pair["quality"] == 0.0

    def test_add_batch(self, store):
        """add_batch inserts multiple pairs."""
        ids = store.add_batch([
            {"user_msg": "u1", "assistant_msg": "a1", "session_id": "s1"},
            {"user_msg": "u2", "assistant_msg": "a2", "session_id": "s1"},
            {"user_msg": "u3", "assistant_msg": "a3", "session_id": "s2"},
        ])
        assert len(ids) == 3
        assert store.count() == 3

    def test_add_batch_empty(self, store):
        """add_batch with empty list returns empty list."""
        ids = store.add_batch([])
        assert ids == []

    def test_get_pair(self, store):
        """get_pair returns the correct document."""
        doc_id = store.add_pair("hello", "hi", "s1")
        pair = store.get_pair(doc_id)
        assert pair["user_msg"] == "hello"

    def test_get_pair_nonexistent(self, store):
        """get_pair returns None for nonexistent ID."""
        assert store.get_pair("nonexistent") is None

    def test_get_pending_pairs(self, store):
        """get_pending_pairs returns unsynced pairs only."""
        store.add_pair("u1", "a1", "s1")
        store.add_pair("u2", "a2", "s1")
        pending = store.get_pending_pairs()
        assert len(pending) == 2

    def test_get_pending_pairs_after_sync(self, store):
        """get_pending_pairs excludes synced pairs."""
        id1 = store.add_pair("u1", "a1", "s1")
        store.add_pair("u2", "a2", "s1")
        store.mark_synced([id1])
        pending = store.get_pending_pairs()
        assert len(pending) == 1
        assert pending[0]["user_msg"] == "u2"

    def test_get_pairs_by_session(self, store):
        """get_pairs_by_session filters by session_id."""
        store.add_pair("u1", "a1", "s1")
        store.add_pair("u2", "a2", "s1")
        store.add_pair("u3", "a3", "s2")
        s1_pairs = store.get_pairs_by_session("s1")
        assert len(s1_pairs) == 2
        s2_pairs = store.get_pairs_by_session("s2")
        assert len(s2_pairs) == 1

    def test_get_pairs_by_quality(self, store):
        """get_pairs_by_quality filters by quality threshold."""
        store.add_pair("u1", "a1", "s1", quality=0.5)
        store.add_pair("u2", "a2", "s1", quality=-0.5)
        store.add_pair("u3", "a3", "s1", quality=1.0)
        high = store.get_pairs_by_quality(min_quality=0.0)
        assert len(high) == 2

    def test_get_training_ready(self, store):
        """get_training_ready returns pairs when above threshold."""
        for i in range(15):
            store.add_pair(f"u{i}", f"a{i}", "s1")
        ready = store.get_training_ready(min_pairs=10, limit=5)
        assert len(ready) == 5

    def test_get_training_ready_below_threshold(self, store):
        """get_training_ready returns empty when below threshold."""
        for i in range(3):
            store.add_pair(f"u{i}", f"a{i}", "s1")
        ready = store.get_training_ready(min_pairs=10)
        assert ready == []

    def test_mark_synced(self, store):
        """mark_synced sets synced=True on matching pairs."""
        id1 = store.add_pair("u1", "a1", "s1")
        id2 = store.add_pair("u2", "a2", "s1")
        count = store.mark_synced([id1, id2])
        assert count == 2
        pair1 = store.get_pair(id1)
        pair2 = store.get_pair(id2)
        assert pair1["synced"] is True
        assert pair2["synced"] is True

    def test_mark_used(self, store):
        """mark_used sets used_for_training=True."""
        doc_id = store.add_pair("u1", "a1", "s1")
        store.mark_used([doc_id])
        pair = store.get_pair(doc_id)
        assert pair["used_for_training"] is True

    def test_update_quality(self, store):
        """update_quality changes the quality field."""
        doc_id = store.add_pair("u1", "a1", "s1", quality=0.0)
        store.update_quality(doc_id, 1.0)
        pair = store.get_pair(doc_id)
        assert pair["quality"] == 1.0

    def test_delete_pair(self, store):
        """delete_pair removes a single document."""
        doc_id = store.add_pair("u1", "a1", "s1")
        assert store.delete_pair(doc_id) is True
        assert store.get_pair(doc_id) is None

    def test_delete_synced(self, store):
        """delete_synced removes all synced documents."""
        id1 = store.add_pair("u1", "a1", "s1")
        store.add_pair("u2", "a2", "s1")
        store.mark_synced([id1])
        count = store.delete_synced()
        assert count == 1
        assert store.count() == 1

    def test_count(self, store):
        """count returns total documents."""
        assert store.count() == 0
        store.add_pair("u1", "a1", "s1")
        assert store.count() == 1
        store.add_pair("u2", "a2", "s1")
        assert store.count() == 2

    def test_count_with_query(self, store):
        """count filters by query."""
        store.add_pair("u1", "a1", "s1", quality=1)
        store.add_pair("u2", "a2", "s2", quality=-1)
        assert store.count({"session_id": "s1"}) == 1
        assert store.count({"quality": 1}) == 1

    def test_stats(self, store):
        """stats returns correct breakdown."""
        id1 = store.add_pair("u1", "a1", "s1")
        store.add_pair("u2", "a2", "s1")
        store.mark_synced([id1])
        stats = store.stats()
        assert stats["total"] == 2
        assert stats["pending"] == 1
        assert stats["synced"] == 1
        assert stats["used"] == 0

    def test_compact(self, store):
        """compact returns without error."""
        for i in range(20):
            store.add_pair(f"u{i}", f"a{i}", "s1")
        result = store.compact()
        assert result >= 0

    def test_many_pairs(self, store):
        """Store handles 500 pairs without issues."""
        ids = []
        for i in range(500):
            ids.append(store.add_pair(f"user_{i}", f"assistant_{i}", f"s{i % 5}"))
        assert store.count() == 500
        pending = store.get_pending_pairs()
        assert len(pending) == 500
        stats = store.stats()
        assert stats["total"] == 500
        assert stats["pending"] == 500
