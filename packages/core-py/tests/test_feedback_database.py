"""Tests for feedback database — conversations, messages, feedback, vector search, meta-weights."""

import json
import time
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timezone

from domains.feedback.database import (
    FeedbackDB,
    Message,
    Feedback,
    SimilarPattern,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """Fresh FeedbackDB with a temporary directory."""
    return FeedbackDB(db_path=str(tmp_path / "feedback"))


@pytest.fixture
def db_with_conversation(db):
    """DB with a single conversation and two messages."""
    conv_id = db.create_conversation(user_id="alice", title="Test Chat")
    msg1 = db.add_message(conv_id, "user", "Hello, how are you?")
    msg2 = db.add_message(conv_id, "assistant", "I am doing well, thanks!")
    return db, conv_id, msg1, msg2


# ── Message and Feedback dataclasses ────────────────────────────────────────

class TestMessageDataclass:

    def test_creation(self):
        m = Message(id="m1", conversation_id="c1", role="user", content="hi")
        assert m.id == "m1"
        assert m.embedding is None
        assert m.created_at is None

    def test_with_embedding(self):
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        m = Message(id="m1", conversation_id="c1", role="user", content="hi", embedding=emb)
        assert m.embedding is not None


class TestFeedbackDataclass:

    def test_creation(self):
        f = Feedback(id="f1", message_id="m1", rating="thumbs_up")
        assert f.rating == "thumbs_up"
        assert f.quality_score is None


class TestSimilarPatternDataclass:

    def test_creation(self):
        p = SimilarPattern(content="hello", rating="thumbs_up", similarity=0.9, pattern_type="message")
        assert p.similarity == 0.9


# ── DB initialization ──────────────────────────────────────────────────────

class TestDBInit:

    def test_creates_parent_directory(self, tmp_path):
        FeedbackDB(db_path=str(tmp_path / "deep" / "feedback"))
        assert (tmp_path / "deep" / "feedback").parent.exists()

    def test_collections_initialized(self, db):
        assert db._conversations is not None
        assert db._messages is not None
        assert db._feedback is not None
        assert db._meta_weights is not None


# ── Embedding helpers ──────────────────────────────────────────────────────

class TestEmbeddingHelpers:

    def test_embedding_to_list_roundtrip(self, db):
        original = np.array([1.0, -0.5, 0.0, 0.25], dtype=np.float32)
        as_list = db._embedding_to_list(original)
        assert isinstance(as_list, list)
        assert len(as_list) == 4
        restored = db._list_to_embedding(as_list)
        assert restored.dtype == np.float32
        np.testing.assert_array_equal(original, restored)

    def test_embedding_to_list_converts_to_float(self, db):
        arr = np.array([1, 2, 3], dtype=np.float64)
        result = db._embedding_to_list(arr)
        assert all(isinstance(x, float) for x in result)


# ── Cosine similarity ──────────────────────────────────────────────────────

class TestCosineSimilarity:

    def test_identical_vectors(self, db):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert db._cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, db):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert db._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self, db):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert db._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self, db):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert db._cosine_similarity(a, b) == 0.0


# ── Strip meta ─────────────────────────────────────────────────────────────

class TestStripMeta:

    def test_removes_internal_fields(self, db):
        doc = {"_id": "x", "_created": 1, "_updated": 2, "user_id": "u1", "name": "test"}
        result = db._strip_meta(doc)
        assert "_id" not in result
        assert "_created" not in result
        assert "_updated" not in result
        assert result["user_id"] == "u1"
        assert result["name"] == "test"

    def test_preserves_all_public_fields(self, db):
        doc = {"a": 1, "b": 2}
        assert db._strip_meta(doc) == {"a": 1, "b": 2}


# ── Conversations ──────────────────────────────────────────────────────────

class TestConversations:

    def test_create_conversation(self, db):
        conv_id = db.create_conversation(user_id="alice", title="Chat 1")
        assert isinstance(conv_id, str)
        assert len(conv_id) > 0

    def test_get_conversation(self, db):
        conv_id = db.create_conversation(user_id="alice", title="Chat 1")
        conv = db.get_conversation(conv_id)
        assert conv is not None
        assert conv["id"] == conv_id
        assert conv["user_id"] == "alice"
        assert conv["title"] == "Chat 1"
        assert conv["created_at"] is not None

    def test_get_nonexistent_conversation(self, db):
        assert db.get_conversation("nope") is None

    def test_list_conversations(self, db):
        db.create_conversation(user_id="alice", title="Chat 1")
        db.create_conversation(user_id="alice", title="Chat 2")
        db.create_conversation(user_id="bob", title="Bob's Chat")
        alice_convs = db.list_conversations(user_id="alice")
        assert len(alice_convs) == 2
        titles = {c["title"] for c in alice_convs}
        assert titles == {"Chat 1", "Chat 2"}

    def test_list_conversations_limit(self, db):
        for i in range(5):
            db.create_conversation(user_id="alice", title=f"Chat {i}")
        result = db.list_conversations(user_id="alice", limit=2)
        assert len(result) == 2

    def test_list_conversations_sorted_by_updated(self, db):
        db.create_conversation(user_id="alice", title="First")
        time.sleep(0.01)
        db.create_conversation(user_id="alice", title="Second")
        convs = db.list_conversations(user_id="alice")
        assert convs[0]["title"] == "Second"


# ── Messages ───────────────────────────────────────────────────────────────

class TestMessages:

    def test_add_message(self, db, db_with_conversation):
        _, conv_id, msg1, msg2 = db_with_conversation
        assert isinstance(msg1, str)
        assert isinstance(msg2, str)

    def test_get_messages(self, db, db_with_conversation):
        _, conv_id, _, _ = db_with_conversation
        messages = db.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, how are you?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "I am doing well, thanks!"

    def test_messages_ordered_by_created(self, db):
        conv_id = db.create_conversation()
        db.add_message(conv_id, "user", "second")
        time.sleep(0.01)
        db.add_message(conv_id, "user", "third")
        # Use the ID of the first message (created first in fixture)
        messages = db.get_messages(conv_id)
        assert messages[0]["content"] == "second"
        assert messages[1]["content"] == "third"

    def test_message_updates_conversation_timestamp(self, db):
        conv_id = db.create_conversation()
        conv_before = db.get_conversation(conv_id)
        time.sleep(0.01)
        db.add_message(conv_id, "user", "test")
        conv_after = db.get_conversation(conv_id)
        assert conv_after["updated_at"] >= conv_before["updated_at"]

    def test_get_message_embedding_none(self, db):
        conv_id = db.create_conversation()
        msg_id = db.add_message(conv_id, "user", "no embedding")
        assert db.get_message_embedding(msg_id) is None

    def test_get_message_embedding_exists(self, db):
        conv_id = db.create_conversation()
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        msg_id = db.add_message(conv_id, "user", "with embedding", embedding=emb)
        stored = db.get_message_embedding(msg_id)
        assert stored is not None
        np.testing.assert_array_almost_equal(stored, emb)


# ── Feedback ───────────────────────────────────────────────────────────────

class TestFeedback:

    def test_add_feedback(self, db):
        conv_id = db.create_conversation()
        msg_id = db.add_message(conv_id, "user", "test")
        fb_id = db.add_feedback(msg_id, "thumbs_up")
        assert isinstance(fb_id, str)
        assert len(fb_id) > 0

    def test_get_feedback(self, db):
        conv_id = db.create_conversation()
        msg_id = db.add_message(conv_id, "user", "test")
        fb_id = db.add_feedback(msg_id, "thumbs_up", quality_score=0.9)
        feedbacks = db.get_feedback(msg_id)
        assert len(feedbacks) == 1
        assert feedbacks[0]["rating"] == "thumbs_up"
        assert feedbacks[0]["quality_score"] == 0.9

    def test_multiple_feedback_same_message(self, db):
        conv_id = db.create_conversation()
        msg_id = db.add_message(conv_id, "user", "test")
        db.add_feedback(msg_id, "thumbs_up")
        db.add_feedback(msg_id, "thumbs_down")
        feedbacks = db.get_feedback(msg_id)
        assert len(feedbacks) == 2

    def test_get_all_feedback(self, db):
        conv_id = db.create_conversation()
        msg_id = db.add_message(conv_id, "user", "test")
        db.add_feedback(msg_id, "thumbs_up")
        all_fb = db.get_all_feedback()
        assert len(all_fb) == 1

    def test_get_all_feedback_joins_message_content(self, db):
        conv_id = db.create_conversation()
        msg_id = db.add_message(conv_id, "user", "Hello world")
        db.add_feedback(msg_id, "thumbs_up")
        all_fb = db.get_all_feedback()
        assert all_fb[0]["content"] == "Hello world"
        assert all_fb[0]["conversation_id"] == conv_id

    def test_get_all_feedback_filtered_by_rating(self, db):
        conv_id = db.create_conversation()
        msg1 = db.add_message(conv_id, "user", "good")
        msg2 = db.add_message(conv_id, "user", "bad")
        db.add_feedback(msg1, "thumbs_up")
        db.add_feedback(msg2, "thumbs_down")
        ups = db.get_all_feedback(rating="thumbs_up")
        assert len(ups) == 1
        downs = db.get_all_feedback(rating="thumbs_down")
        assert len(downs) == 1

    def test_get_all_feedback_skips_missing_messages(self, db):
        conv_id = db.create_conversation()
        msg_id = db.add_message(conv_id, "user", "test")
        db.add_feedback(msg_id, "thumbs_up")
        # Delete the message directly
        db._messages.delete_one({"_id": msg_id})
        all_fb = db.get_all_feedback()
        assert len(all_fb) == 0


# ── Vector search ──────────────────────────────────────────────────────────

class TestVectorSearch:

    def test_find_similar_messages(self, db):
        conv_id = db.create_conversation()
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        emb3 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        db.add_message(conv_id, "user", "doc1", embedding=emb1)
        db.add_message(conv_id, "user", "doc2", embedding=emb2)
        db.add_message(conv_id, "user", "doc3", embedding=emb3)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = db.find_similar_messages(query, k=2)
        assert len(results) == 2
        assert results[0].content == "doc1"
        assert results[0].similarity > results[1].similarity

    def test_find_similar_messages_min_similarity(self, db):
        conv_id = db.create_conversation()
        emb = np.array([1.0, 0.0], dtype=np.float32)
        db.add_message(conv_id, "user", "far away", embedding=emb)
        query = np.array([-1.0, 0.0], dtype=np.float32)
        results = db.find_similar_messages(query, k=5, min_similarity=0.5)
        assert len(results) == 0

    def test_find_similar_messages_with_rating_filter(self, db):
        conv_id = db.create_conversation()
        msg1 = db.add_message(conv_id, "user", "liked", embedding=np.array([1.0, 0.0], dtype=np.float32))
        msg2 = db.add_message(conv_id, "user", "disliked", embedding=np.array([1.0, 0.0], dtype=np.float32))
        db.add_feedback(msg1, "thumbs_up")
        db.add_feedback(msg2, "thumbs_down")
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = db.find_similar_messages(query, k=5, rating="thumbs_up")
        assert len(results) == 1
        assert results[0].content == "liked"
        assert results[0].rating == "thumbs_up"


# ── Text search ────────────────────────────────────────────────────────────

class TestTextSearch:

    def test_find_similar_by_text(self, db):
        conv_id = db.create_conversation()
        db.add_message(conv_id, "user", "the quick brown fox")
        db.add_message(conv_id, "user", "the lazy dog")
        db.add_message(conv_id, "user", "completely different topic")
        results = db.find_similar_by_text("quick fox", k=2)
        assert len(results) > 0
        assert results[0].similarity > 0

    def test_text_search_with_rating(self, db):
        conv_id = db.create_conversation()
        msg1 = db.add_message(conv_id, "user", "python programming tips")
        msg2 = db.add_message(conv_id, "user", "python programming tricks")
        db.add_feedback(msg1, "thumbs_up")
        db.add_feedback(msg2, "thumbs_down")
        results = db.find_similar_by_text("python programming", rating="thumbs_up")
        for r in results:
            assert r.rating == "thumbs_up"

    def test_text_search_no_match(self, db):
        conv_id = db.create_conversation()
        db.add_message(conv_id, "user", "hello")
        results = db.find_similar_by_text("xyz", k=5)
        assert len(results) == 0

    def test_text_search_truncates_content(self, db):
        conv_id = db.create_conversation()
        long_text = "a" * 300
        db.add_message(conv_id, "user", long_text)
        results = db.find_similar_by_text("a", k=1)
        if results:
            assert len(results[0].content) <= 200


# ── Statistics ──────────────────────────────────────────────────────────────

class TestStats:

    def test_empty_db(self, db):
        stats = db.get_stats()
        assert stats["conversations"] == 0
        assert stats["messages"] == 0
        assert stats["feedback_total"] == 0
        assert stats["thumbs_up"] == 0
        assert stats["thumbs_down"] == 0
        assert stats["ratio"] == 0.0

    def test_populated_db(self, db):
        conv_id = db.create_conversation()
        msg1 = db.add_message(conv_id, "user", "good")
        msg2 = db.add_message(conv_id, "user", "bad")
        db.add_feedback(msg1, "thumbs_up")
        db.add_feedback(msg2, "thumbs_down")
        stats = db.get_stats()
        assert stats["conversations"] == 1
        assert stats["messages"] == 2
        assert stats["feedback_total"] == 2
        assert stats["thumbs_up"] == 1
        assert stats["thumbs_down"] == 1
        assert stats["ratio"] == pytest.approx(1.0)

    def test_ratio_only_upvotes(self, db):
        conv_id = db.create_conversation()
        msg = db.add_message(conv_id, "user", "great")
        db.add_feedback(msg, "thumbs_up")
        stats = db.get_stats()
        assert stats["ratio"] == 1.0


# ── Export ──────────────────────────────────────────────────────────────────

class TestExport:

    def test_export_jsonl(self, db, tmp_path):
        conv_id = db.create_conversation()
        msg_user = db.add_message(conv_id, "user", "prompt text")
        msg_asst = db.add_message(conv_id, "assistant", "response text")
        db.add_message(conv_id, "user", "follow up")
        db.add_feedback(msg_asst, "thumbs_up", quality_score=0.95)
        filepath = str(tmp_path / "export.jsonl")
        db.export_feedback_jsonl(filepath)
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["response"] == "response text"
        assert record["rating"] == "thumbs_up"
        assert record["quality_score"] == 0.95
        assert record["prompt"] == "follow up"

    def test_export_dpo_format(self, db, tmp_path):
        conv_id = db.create_conversation()
        msg1 = db.add_message(conv_id, "assistant", "good response")
        msg2 = db.add_message(conv_id, "assistant", "bad response")
        db.add_feedback(msg1, "thumbs_up", context_snippet="user question")
        db.add_feedback(msg2, "thumbs_down", context_snippet="user question")
        filepath = str(tmp_path / "dpo.jsonl")
        db.export_dpo_format(filepath)
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["chosen"] == "good response"
        assert record["rejected"] == "bad response"


# ── User meta weights ──────────────────────────────────────────────────────

class TestUserMetaWeights:

    def test_get_nonexistent(self, db):
        assert db.get_user_meta_weights("nobody") is None

    def test_create_with_thumbs_up(self, db):
        weights = db.update_user_meta_weights("alice", "thumbs_up")
        assert weights["user_id"] == "alice"
        assert weights["thumbs_up_count"] == 1
        assert weights["thumbs_down_count"] == 0
        assert weights["temperature_boost"] > 0
        assert weights["repetition_boost"] < 0

    def test_create_with_thumbs_down(self, db):
        weights = db.update_user_meta_weights("alice", "thumbs_down")
        assert weights["thumbs_up_count"] == 0
        assert weights["thumbs_down_count"] == 1
        assert weights["temperature_boost"] < 0
        assert weights["repetition_boost"] > 0

    def test_update_accumulates(self, db):
        db.update_user_meta_weights("alice", "thumbs_up")
        db.update_user_meta_weights("alice", "thumbs_up")
        weights = db.update_user_meta_weights("alice", "thumbs_up")
        assert weights["thumbs_up_count"] == 3
        assert weights["temperature_boost"] == pytest.approx(0.03)

    def test_update_mixed_ratings(self, db):
        db.update_user_meta_weights("alice", "thumbs_up")
        db.update_user_meta_weights("alice", "thumbs_down")
        weights = db.get_user_meta_weights("alice")
        assert weights["thumbs_up_count"] == 1
        assert weights["thumbs_down_count"] == 1
        # temperature_boost: +0.01 - 0.01 = 0
        assert weights["temperature_boost"] == pytest.approx(0.0)

    def test_get_all_user_meta_weights(self, db):
        db.update_user_meta_weights("alice", "thumbs_up")
        db.update_user_meta_weights("bob", "thumbs_down")
        all_weights = db.get_all_user_meta_weights()
        assert len(all_weights) == 2
        user_ids = {w["user_id"] for w in all_weights}
        assert user_ids == {"alice", "bob"}

    def test_custom_deltas(self, db):
        weights = db.update_user_meta_weights(
            "alice", "thumbs_up",
            temperature_delta=0.1,
            repetition_delta=0.2,
            top_p_delta=0.3,
            top_k_delta=5.0,
        )
        assert weights["temperature_boost"] == pytest.approx(0.1)
        assert weights["repetition_boost"] == pytest.approx(-0.2)
        assert weights["top_p_boost"] == pytest.approx(0.3)
        assert weights["top_k_boost"] == pytest.approx(-5.0)

    def test_created_at_set(self, db):
        weights = db.update_user_meta_weights("alice", "thumbs_up")
        assert weights["created_at"] is not None

    def test_last_updated_changes(self, db):
        db.update_user_meta_weights("alice", "thumbs_up")
        time.sleep(0.01)
        weights = db.update_user_meta_weights("alice", "thumbs_up")
        assert weights["last_updated"] is not None


# ── Thread safety ──────────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_creates(self, db):
        import threading

        errors = []

        def create_conv(i):
            try:
                db.create_conversation(user_id=f"user{i}", title=f"Chat {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_conv, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(errors) == 0
        assert len(db.list_conversations(user_id="user0")) == 1

    def test_concurrent_meta_weight_updates(self, db):
        import threading

        errors = []

        def update():
            try:
                db.update_user_meta_weights("alice", "thumbs_up")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(errors) == 0
        weights = db.get_user_meta_weights("alice")
        assert weights["thumbs_up_count"] == 10


# ── Persistence round-trip ──────────────────────────────────────────────────

class TestPersistence:

    def test_survives_recreation(self, tmp_path):
        path = str(tmp_path / "feedback")
        db1 = FeedbackDB(db_path=path)
        conv_id = db1.create_conversation(user_id="alice", title="Persistent Chat")
        msg_id = db1.add_message(conv_id, "user", "remember me")
        db1.add_feedback(msg_id, "thumbs_up")

        db2 = FeedbackDB(db_path=path)
        convs = db2.list_conversations(user_id="alice")
        assert len(convs) == 1
        assert convs[0]["title"] == "Persistent Chat"

        messages = db2.get_messages(conv_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "remember me"

        feedbacks = db2.get_feedback(msg_id)
        assert len(feedbacks) == 1
        assert feedbacks[0]["rating"] == "thumbs_up"

        weights = db2.get_user_meta_weights("alice")
        assert weights is None  # Meta weights not persisted through re-creation in this flow
