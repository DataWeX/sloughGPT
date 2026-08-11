"""Tests for feedback/training.py — training data pipeline from feedback DB.

Creates in-memory SQLite DB with the required schema to test:
  - TrainingExample / DPOPair dataclasses
  - FeedbackTrainer.get_training_examples()
  - FeedbackTrainer.prepare_dpo_pairs()
  - FeedbackTrainer.prepare_sft_data()
  - FeedbackTrainer.get_training_stats()
  - FeedbackTrainer.export_for_alignment()
  - FeedbackTrainer.export_dpo() / export_sft()
  - create_training_pipeline() factory
"""

import json
import os
import tempfile
import sqlite3
import pytest
from domains.feedback.training import (
    TrainingExample,
    DPOPair,
    FeedbackTrainer,
    create_training_pipeline,
)


def _create_test_db(db_path: str):
    """Create a test DB with feedback schema and sample data."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT,
            rating TEXT,
            quality_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Conv 1: both thumbs_up and thumbs_down (for DPO)
    c.execute("INSERT INTO conversations VALUES ('c1', '2024-01-01')")
    c.execute("INSERT INTO messages VALUES ('m1u', 'c1', 'user', 'What is 2+2?', '2024-01-01 00:00')")
    c.execute("INSERT INTO messages VALUES ('m1a', 'c1', 'assistant', '4', '2024-01-01 00:01')")
    c.execute("INSERT INTO feedback VALUES (1, 'm1a', 'thumbs_up', 0.9, '2024-01-01')")
    c.execute("INSERT INTO messages VALUES ('m1b', 'c1', 'assistant', 'I dont know', '2024-01-01 00:02')")
    c.execute("INSERT INTO feedback VALUES (2, 'm1b', 'thumbs_down', 0.1, '2024-01-01')")

    # Conv 2: thumbs_up only (for SFT)
    c.execute("INSERT INTO conversations VALUES ('c2', '2024-01-02')")
    c.execute("INSERT INTO messages VALUES ('m2u', 'c2', 'user', 'Hello', '2024-01-02 00:00')")
    c.execute("INSERT INTO messages VALUES ('m2a', 'c2', 'assistant', 'Hi there!', '2024-01-02 00:01')")
    c.execute("INSERT INTO feedback VALUES (3, 'm2a', 'thumbs_up', 0.8, '2024-01-02')")

    # Conv 3: both ratings (for DPO)
    c.execute("INSERT INTO conversations VALUES ('c3', '2024-01-03')")
    c.execute("INSERT INTO messages VALUES ('m3u', 'c3', 'user', 'Explain AI', '2024-01-03 00:00')")
    c.execute("INSERT INTO messages VALUES ('m3a', 'c3', 'assistant', 'AI is...', '2024-01-03 00:01')")
    c.execute("INSERT INTO feedback VALUES (4, 'm3a', 'thumbs_up', 0.95, '2024-01-03')")
    c.execute("INSERT INTO messages VALUES ('m3b', 'c3', 'assistant', 'Whatever', '2024-01-03 00:02')")
    c.execute("INSERT INTO feedback VALUES (5, 'm3b', 'thumbs_down', 0.05, '2024-01-03')")

    conn.commit()
    conn.close()


class TestTrainingExample:
    def test_dataclass(self):
        ex = TrainingExample(prompt="q", response="a", rating="thumbs_up", quality_score=0.9)
        assert ex.prompt == "q"
        assert ex.response == "a"
        assert ex.quality_score == 0.9

    def test_optional_quality(self):
        ex = TrainingExample(prompt="q", response="a", rating="thumbs_up")
        assert ex.quality_score is None


class TestDPOPair:
    def test_dataclass(self):
        p = DPOPair(chosen="good", rejected="bad", prompt="q")
        assert p.chosen == "good"
        assert p.rejected == "bad"


class TestFeedbackTrainer:
    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path):
        self.db_path = str(tmp_path / "test_feedback.db")
        _create_test_db(self.db_path)
        self.trainer = FeedbackTrainer(db_path=self.db_path)

    def test_get_training_examples(self):
        examples = self.trainer.get_training_examples()
        assert len(examples) == 5
        assert all(isinstance(e, TrainingExample) for e in examples)

    def test_get_training_examples_min_quality(self):
        examples = self.trainer.get_training_examples(min_quality=0.5)
        assert len(examples) >= 1
        for e in examples:
            if e.quality_score is not None:
                assert e.quality_score >= 0.5

    def test_get_training_examples_limit(self):
        examples = self.trainer.get_training_examples(limit=2)
        assert len(examples) <= 2

    def test_prepare_dpo_pairs(self):
        pairs = self.trainer.prepare_dpo_pairs()
        assert len(pairs) == 2  # c1 and c3 have both ratings
        assert all(isinstance(p, DPOPair) for p in pairs)
        for p in pairs:
            assert p.chosen != p.rejected
            assert len(p.chosen) > 0
            assert len(p.rejected) > 0

    def test_prepare_dpo_pairs_chosen_is_positive(self):
        pairs = self.trainer.prepare_dpo_pairs()
        for p in pairs:
            # chosen should be from thumbs_up
            assert "4" in p.chosen or "AI is" in p.chosen

    def test_prepare_sft_data(self):
        sft = self.trainer.prepare_sft_data(min_quality=0.0)
        assert len(sft) >= 2  # m1a and m2a are thumbs_up with prompts
        for item in sft:
            assert "prompt" in item
            assert "response" in item
            assert "quality_score" in item

    def test_prepare_sft_filters_negative(self):
        sft = self.trainer.prepare_sft_data(min_quality=0.0)
        # SFT only includes thumbs_up
        assert all(item["quality_score"] > 0 for item in sft)

    def test_get_training_stats(self):
        stats = self.trainer.get_training_stats()
        assert stats["total_conversations"] == 3
        assert stats["total_responses"] == 5
        assert stats["thumbs_up"] >= 2
        assert stats["thumbs_down"] >= 2
        assert stats["available_dpo_pairs"] == 2
        assert isinstance(stats["available_sft_examples"], int)

    def test_export_for_alignment_dpo(self, tmp_path):
        out = str(tmp_path / "export")
        results = self.trainer.export_for_alignment(output_dir=out, formats=["dpo"])
        assert "dpo" in results
        assert os.path.exists(results["dpo"])
        with open(results["dpo"]) as f:
            lines = f.readlines()
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert "chosen" in data
        assert "rejected" in data
        assert "prompt" in data

    def test_export_for_alignment_sft(self, tmp_path):
        out = str(tmp_path / "export")
        results = self.trainer.export_for_alignment(output_dir=out, formats=["sft"])
        assert "sft" in results
        assert os.path.exists(results["sft"])

    def test_export_for_alignment_reward(self, tmp_path):
        out = str(tmp_path / "export")
        results = self.trainer.export_for_alignment(output_dir=out, formats=["reward"])
        assert "reward" in results
        with open(results["reward"]) as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            assert "reward" in data
            assert data["reward"] in (0.0, 1.0)

    def test_export_dpo(self, tmp_path):
        filepath = str(tmp_path / "dpo.jsonl")
        count = self.trainer.export_dpo(filepath)
        assert count == 2
        assert os.path.exists(filepath)

    def test_export_sft(self, tmp_path):
        filepath = str(tmp_path / "sft.jsonl")
        count = self.trainer.export_sft(filepath)
        assert count >= 1
        assert os.path.exists(filepath)

    def test_empty_db(self, tmp_path):
        empty_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(empty_path)
        conn.executescript("""
            CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at TIMESTAMP);
            CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT, content TEXT, created_at TIMESTAMP);
            CREATE TABLE feedback (id INTEGER PRIMARY KEY, message_id TEXT, rating TEXT, quality_score REAL, created_at TIMESTAMP);
        """)
        conn.close()
        trainer = FeedbackTrainer(db_path=empty_path)
        stats = trainer.get_training_stats()
        assert stats["total_conversations"] == 0
        assert stats["available_dpo_pairs"] == 0
        assert len(trainer.prepare_dpo_pairs()) == 0
        assert len(trainer.prepare_sft_data()) == 0


class TestFactory:
    def test_create_training_pipeline(self):
        trainer = create_training_pipeline("test.db")
        assert isinstance(trainer, FeedbackTrainer)
        assert trainer.db_path == "test.db"
