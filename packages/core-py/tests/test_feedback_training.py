"""Tests for feedback/training.py — TrainingExample, DPOPair, FeedbackTrainer."""

import pytest
from mogdb import MogDB
from domains.feedback.training import TrainingExample, DPOPair, FeedbackTrainer


class TestTrainingExample:
    def test_creation(self):
        ex = TrainingExample(prompt="hello", response="world", rating="thumbs_up")
        assert ex.prompt == "hello"
        assert ex.response == "world"
        assert ex.rating == "thumbs_up"
        assert ex.quality_score is None

    def test_with_quality_score(self):
        ex = TrainingExample(prompt="q", response="a", rating="ok", quality_score=0.8)
        assert ex.quality_score == 0.8

    def test_defaults(self):
        ex = TrainingExample(prompt="", response="", rating="")
        assert ex.quality_score is None


class TestDPOPair:
    def test_creation(self):
        pair = DPOPair(chosen="good", rejected="bad", prompt="q")
        assert pair.chosen == "good"
        assert pair.rejected == "bad"
        assert pair.prompt == "q"


def _seed(db_path, messages, feedback):
    db = MogDB(str(db_path))
    msgs = db.collection("messages")
    fb = db.collection("feedback")
    for m in messages:
        doc = dict(m)
        doc.setdefault("_id", m["id"])
        msgs.insert_one(doc)
    for f in feedback:
        doc = dict(f)
        doc.setdefault("_id", f["id"])
        fb.insert_one(doc)
    return FeedbackTrainer(db_path=str(db_path))


class TestFeedbackTrainer:
    @pytest.fixture
    def trainer(self, tmp_path):
        db = tmp_path / "feedback.db"
        return _seed(
            db,
            [
                {
                    "id": "m1",
                    "conversation_id": "c1",
                    "role": "user",
                    "content": "What is 2+2?",
                    "created_at": "2024-01-01 00:00:00",
                },
                {
                    "id": "m2",
                    "conversation_id": "c1",
                    "role": "assistant",
                    "content": "4",
                    "created_at": "2024-01-01 00:00:01",
                },
                {
                    "id": "m3",
                    "conversation_id": "c1",
                    "role": "assistant",
                    "content": "I don't know",
                    "created_at": "2024-01-01 00:00:02",
                },
            ],
            [
                {
                    "id": "f1",
                    "message_id": "m2",
                    "rating": "thumbs_up",
                    "quality_score": 0.9,
                    "created_at": "2024-01-01 00:00:03",
                },
                {
                    "id": "f2",
                    "message_id": "m3",
                    "rating": "thumbs_down",
                    "quality_score": 0.2,
                    "created_at": "2024-01-01 00:00:04",
                },
            ],
        )

    def test_get_training_examples(self, trainer):
        examples = trainer.get_training_examples()
        assert len(examples) == 2

    def test_get_training_examples_min_quality(self, trainer):
        examples = trainer.get_training_examples(min_quality=0.5)
        assert len(examples) == 1
        assert examples[0].rating == "thumbs_up"

    def test_get_training_examples_limit(self, trainer):
        examples = trainer.get_training_examples(limit=1)
        assert len(examples) == 1

    def test_prepare_dpo_pairs(self, trainer):
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) == 1
        assert pairs[0].chosen == "4"
        assert pairs[0].rejected == "I don't know"

    def test_prepare_sft_data(self, trainer):
        sft = trainer.prepare_sft_data()
        assert len(sft) == 1
        assert sft[0]["response"] == "4"

    def test_empty_db(self, tmp_path):
        db = tmp_path / "empty.db"
        mog = MogDB(str(db))
        mog.collection("messages")
        mog.collection("feedback")
        t = FeedbackTrainer(db_path=str(db))
        assert t.get_training_examples() == []
        assert t.prepare_dpo_pairs() == []
        assert t.prepare_sft_data() == []

    def test_sqlite_db_is_migrated_on_init(self, tmp_path):
        import sqlite3

        db = tmp_path / "feedback.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT,
                rating TEXT,
                quality_score REAL,
                created_at TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            ("m1", "c1", "user", "prompt text", "2024-01-01 00:00:00"),
        )
        conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            ("m2", "c1", "assistant", "good answer", "2024-01-01 00:00:01"),
        )
        conn.execute(
            "INSERT INTO feedback (message_id, rating, quality_score, created_at) VALUES (?, ?, ?, ?)",
            ("m2", "thumbs_up", 0.9, "2024-01-01 00:00:02"),
        )
        conn.commit()
        conn.close()

        t = FeedbackTrainer(db_path=str(db))
        examples = t.get_training_examples()
        assert len(examples) == 1
        assert examples[0].response == "good answer"
        assert examples[0].prompt == "prompt text"
