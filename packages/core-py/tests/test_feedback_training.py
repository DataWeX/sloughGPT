"""Tests for feedback/training.py — TrainingExample, DPOPair, FeedbackTrainer."""

import sqlite3
import pytest
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


class TestFeedbackTrainer:
    @pytest.fixture
    def trainer(self, tmp_path):
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
            ("m1", "c1", "user", "What is 2+2?", "2024-01-01 00:00:00"),
        )
        conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            ("m2", "c1", "assistant", "4", "2024-01-01 00:00:01"),
        )
        conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            ("m3", "c1", "assistant", "I don't know", "2024-01-01 00:00:02"),
        )
        conn.execute(
            "INSERT INTO feedback (message_id, rating, quality_score, created_at) VALUES (?, ?, ?, ?)",
            ("m2", "thumbs_up", 0.9, "2024-01-01 00:00:03"),
        )
        conn.execute(
            "INSERT INTO feedback (message_id, rating, quality_score, created_at) VALUES (?, ?, ?, ?)",
            ("m3", "thumbs_down", 0.2, "2024-01-01 00:00:04"),
        )
        conn.commit()
        conn.close()
        return FeedbackTrainer(db_path=str(db))

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
        conn.commit()
        conn.close()
        t = FeedbackTrainer(db_path=str(db))
        assert t.get_training_examples() == []
        assert t.prepare_dpo_pairs() == []
        assert t.prepare_sft_data() == []
