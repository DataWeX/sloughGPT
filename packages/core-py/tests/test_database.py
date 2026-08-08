"""Tests for domains.feedback.database — SQLite feedback database with vector search."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from domains.feedback.database import (
    FeedbackDB,
    Message,
    Feedback,
    SimilarPattern,
    get_feedback_db,
)


class TestDataclasses:
    def test_message(self):
        msg = Message(id="m1", conversation_id="c1", role="user", content="hi")
        assert msg.id == "m1"
        assert msg.embedding is None

    def test_feedback(self):
        fb = Feedback(id="f1", message_id="m1", rating="thumbs_up")
        assert fb.quality_score is None

    def test_similar_pattern(self):
        sp = SimilarPattern(content="hello", rating="thumbs_up", similarity=0.9, pattern_type="exact")
        assert sp.similarity == 0.9


class TestFeedbackDBInit:
    def test_creates_db_file(self, tmp_path):
        db = FeedbackDB(db_path=str(tmp_path / "test.db"))
        assert Path(tmp_path / "test.db").exists()

    def test_creates_parent_dir(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "feedback.db")
        db = FeedbackDB(db_path=db_path)
        assert Path(db_path).exists()


class TestConversations:
    def test_create_conversation(self, db):
        conv_id = db.create_conversation(user_id="u1", title="Test Chat")
        assert conv_id is not None
        assert len(conv_id) > 0

    def test_get_conversation(self, db):
        conv_id = db.create_conversation(user_id="u1", title="My Chat")
        conv = db.get_conversation(conv_id)
        assert conv is not None
        assert conv["title"] == "My Chat"
        assert conv["user_id"] == "u1"

    def test_get_nonexistent_conversation(self, db):
        assert db.get_conversation("nonexistent") is None

    def test_list_conversations_empty(self, db):
        assert db.list_conversations(user_id="u1") == []

    def test_list_conversations(self, db):
        db.create_conversation(user_id="u1", title="Chat 1")
        db.create_conversation(user_id="u1", title="Chat 2")
        convos = db.list_conversations(user_id="u1")
        assert len(convos) == 2

    def test_list_conversations_by_user(self, db):
        db.create_conversation(user_id="u1", title="Chat 1")
        db.create_conversation(user_id="u2", title="Chat 2")
        assert len(db.list_conversations(user_id="u1")) == 1
        assert len(db.list_conversations(user_id="u2")) == 1

    def test_list_conversations_limit(self, db):
        for i in range(5):
            db.create_conversation(user_id="u1", title=f"Chat {i}")
        assert len(db.list_conversations(user_id="u1", limit=2)) == 2


class TestMessages:
    def test_add_message(self, db):
        conv_id = db.create_conversation(user_id="u1")
        msg_id = db.add_message(conv_id, "user", "Hello")
        assert msg_id is not None

    def test_get_messages(self, db):
        conv_id = db.create_conversation(user_id="u1")
        db.add_message(conv_id, "user", "Hello")
        db.add_message(conv_id, "assistant", "Hi there")
        msgs = db.get_messages(conv_id)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "Hi there"

    def test_add_message_with_embedding(self, db):
        conv_id = db.create_conversation(user_id="u1")
        emb = np.random.randn(384).astype(np.float32)
        msg_id = db.add_message(conv_id, "user", "Hello", embedding=emb)
        retrieved = db.get_message_embedding(msg_id)
        assert retrieved is not None
        assert np.allclose(retrieved, emb)

    def test_get_message_embedding_nonexistent(self, db):
        assert db.get_message_embedding("nonexistent") is None


class TestFeedbackCRUD:
    def test_add_feedback(self, db):
        conv_id = db.create_conversation(user_id="u1")
        msg_id = db.add_message(conv_id, "user", "Hello")
        fb_id = db.add_feedback(msg_id, "thumbs_up", quality_score=0.9)
        assert fb_id is not None

    def test_get_feedback(self, db):
        conv_id = db.create_conversation(user_id="u1")
        msg_id = db.add_message(conv_id, "user", "Hello")
        db.add_feedback(msg_id, "thumbs_up")
        db.add_feedback(msg_id, "thumbs_down")
        fbs = db.get_feedback(msg_id)
        assert len(fbs) == 2

    def test_get_all_feedback(self, db):
        conv_id = db.create_conversation(user_id="u1")
        msg_id = db.add_message(conv_id, "user", "Hello")
        db.add_feedback(msg_id, "thumbs_up")
        db.add_feedback(msg_id, "thumbs_down")
        all_up = db.get_all_feedback(rating="thumbs_up")
        assert len(all_up) == 1
        all_fb = db.get_all_feedback()
        assert len(all_fb) == 2


class TestSimilarSearch:
    def test_find_similar_messages(self, db):
        conv_id = db.create_conversation(user_id="u1")
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        db.add_message(conv_id, "user", "Hello", embedding=emb1)
        db.add_message(conv_id, "user", "Goodbye", embedding=emb2)
        results = db.find_similar_messages(emb1, k=5)
        assert len(results) >= 1
        assert results[0].content == "Hello"

    def test_find_similar_messages_with_rating_filter(self, db):
        conv_id = db.create_conversation(user_id="u1")
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        msg_id1 = db.add_message(conv_id, "user", "Hello", embedding=emb1)
        db.add_message(conv_id, "user", "Goodbye", embedding=emb2)
        db.add_feedback(msg_id1, "thumbs_up")
        results = db.find_similar_messages(emb1, k=5, rating="thumbs_up")
        assert len(results) >= 1

    def test_find_similar_by_text(self, db):
        conv_id = db.create_conversation(user_id="u1")
        emb1 = np.random.randn(384).astype(np.float32)
        db.add_message(conv_id, "user", "Hello world", embedding=emb1)
        results = db.find_similar_by_text("Hello", k=5)
        assert isinstance(results, list)

    def test_find_similar_no_embeddings(self, db):
        conv_id = db.create_conversation(user_id="u1")
        db.add_message(conv_id, "user", "Hello")
        results = db.find_similar_messages(np.zeros(384, dtype=np.float32), k=5)
        assert isinstance(results, list)


class TestStats:
    def test_get_stats_empty(self, db):
        stats = db.get_stats()
        assert stats["conversations"] == 0
        assert stats["messages"] == 0
        assert stats["feedback_total"] == 0

    def test_get_stats_with_data(self, db):
        conv_id = db.create_conversation(user_id="u1")
        msg_id = db.add_message(conv_id, "user", "Hello")
        db.add_feedback(msg_id, "thumbs_up")
        stats = db.get_stats()
        assert stats["conversations"] == 1
        assert stats["messages"] == 1
        assert stats["feedback_total"] == 1


class TestExport:
    def test_export_feedback_jsonl(self, db, tmp_path):
        conv_id = db.create_conversation(user_id="u1")
        msg_id = db.add_message(conv_id, "user", "Hello")
        db.add_feedback(msg_id, "thumbs_up")
        export_path = str(tmp_path / "export.jsonl")
        db.export_feedback_jsonl(export_path)
        assert Path(export_path).exists()

    def test_export_dpo_format(self, db, tmp_path):
        conv_id = db.create_conversation(user_id="u1")
        msg_id = db.add_message(conv_id, "user", "Hello")
        db.add_feedback(msg_id, "thumbs_up")
        export_path = str(tmp_path / "dpo.jsonl")
        db.export_dpo_format(export_path)
        assert Path(export_path).exists()


class TestMetaWeights:
    def test_get_user_meta_weights_none(self, db):
        assert db.get_user_meta_weights("u1") is None

    def test_update_and_get_user_meta_weights(self, db):
        weights = {"warmth": 0.7, "creativity": 0.5}
        db.update_user_meta_weights("u1", weights)
        result = db.get_user_meta_weights("u1")
        assert result is not None
        assert result["user_id"] == "u1"

    def test_get_all_user_meta_weights(self, db):
        db.update_user_meta_weights("u1", {"warmth": 0.7})
        db.update_user_meta_weights("u2", {"warmth": 0.3})
        all_weights = db.get_all_user_meta_weights()
        assert len(all_weights) == 2


class TestConcurrency:
    def test_concurrent_writes(self, tmp_path):
        db = FeedbackDB(db_path=str(tmp_path / "concurrent.db"))
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    conv_id = db.create_conversation(user_id=f"u{n}", title=f"Chat {n}_{i}")
                    db.add_message(conv_id, "user", f"Message {i}")
            except Exception as e:
                errors.append(e)

        import threading
        threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        stats = db.get_stats()
        assert stats["conversations"] == 30
        assert stats["messages"] == 30


class TestSingleton:
    def test_same_instance(self):
        a = get_feedback_db()
        b = get_feedback_db()
        assert a is b


@pytest.fixture()
def db(tmp_path):
    return FeedbackDB(db_path=str(tmp_path / "test.db"))
