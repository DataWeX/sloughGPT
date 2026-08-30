"""Tests for feedback/training.py — TrainingExample, DPOPair, FeedbackTrainer."""

import json
import pytest
from pathlib import Path
from mogdb import MogDB
from domains.feedback.training import TrainingExample, DPOPair, FeedbackTrainer, create_training_pipeline


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------

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

    def test_equality(self):
        a = TrainingExample(prompt="p", response="r", rating="thumbs_up")
        b = TrainingExample(prompt="p", response="r", rating="thumbs_up")
        assert a == b

    def test_inequality(self):
        a = TrainingExample(prompt="p1", response="r", rating="thumbs_up")
        b = TrainingExample(prompt="p2", response="r", rating="thumbs_up")
        assert a != b

    def test_quality_score_zero(self):
        ex = TrainingExample(prompt="p", response="r", rating="ok", quality_score=0.0)
        assert ex.quality_score == 0.0

    def test_quality_score_one(self):
        ex = TrainingExample(prompt="p", response="r", rating="ok", quality_score=1.0)
        assert ex.quality_score == 1.0

    def test_negative_quality_score(self):
        ex = TrainingExample(prompt="p", response="r", rating="ok", quality_score=-0.5)
        assert ex.quality_score == -0.5

    def test_long_text(self):
        ex = TrainingExample(prompt="x" * 10000, response="y" * 10000, rating="ok")
        assert len(ex.prompt) == 10000
        assert len(ex.response) == 10000

    def test_empty_strings(self):
        ex = TrainingExample(prompt="", response="", rating="")
        assert ex.prompt == ""
        assert ex.response == ""
        assert ex.rating == ""


# ---------------------------------------------------------------------------
# DPOPair
# ---------------------------------------------------------------------------

class TestDPOPair:
    def test_creation(self):
        pair = DPOPair(chosen="good", rejected="bad", prompt="q")
        assert pair.chosen == "good"
        assert pair.rejected == "bad"
        assert pair.prompt == "q"

    def test_equality(self):
        a = DPOPair(chosen="g", rejected="b", prompt="q")
        b = DPOPair(chosen="g", rejected="b", prompt="q")
        assert a == b

    def test_inequality(self):
        a = DPOPair(chosen="g1", rejected="b", prompt="q")
        b = DPOPair(chosen="g2", rejected="b", prompt="q")
        assert a != b

    def test_empty_strings(self):
        pair = DPOPair(chosen="", rejected="", prompt="")
        assert pair.chosen == ""
        assert pair.rejected == ""
        assert pair.prompt == ""

    def test_long_text(self):
        pair = DPOPair(chosen="a" * 5000, rejected="b" * 5000, prompt="q")
        assert len(pair.chosen) == 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _seed_multi_conv(tmp_path):
    """Seed a database with two conversations for cross-conversation DPO testing."""
    db = tmp_path / "multi.db"
    messages = [
        {"id": "u1", "conversation_id": "c1", "role": "user", "content": "prompt A", "created_at": "2024-01-01 00:00:00"},
        {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "good answer A", "created_at": "2024-01-01 00:00:01"},
        {"id": "a2", "conversation_id": "c1", "role": "assistant", "content": "bad answer A", "created_at": "2024-01-01 00:00:02"},
        {"id": "u2", "conversation_id": "c2", "role": "user", "content": "prompt B", "created_at": "2024-01-01 00:00:03"},
        {"id": "a3", "conversation_id": "c2", "role": "assistant", "content": "good answer B", "created_at": "2024-01-01 00:00:04"},
        {"id": "a4", "conversation_id": "c2", "role": "assistant", "content": "bad answer B", "created_at": "2024-01-01 00:00:05"},
    ]
    feedback = [
        {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:06"},
        {"id": "f2", "message_id": "a2", "rating": "thumbs_down", "quality_score": 0.2, "created_at": "2024-01-01 00:00:07"},
        {"id": "f3", "message_id": "a3", "rating": "thumbs_up", "quality_score": 0.8, "created_at": "2024-01-01 00:00:08"},
        {"id": "f4", "message_id": "a4", "rating": "thumbs_down", "quality_score": 0.1, "created_at": "2024-01-01 00:00:09"},
    ]
    return _seed(db, messages, feedback)


# ---------------------------------------------------------------------------
# FeedbackTrainer — basic operations
# ---------------------------------------------------------------------------

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

    def test_get_training_examples_returns_newest_first(self, tmp_path):
        db = tmp_path / "order.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p1", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r1", "created_at": "2024-01-01 00:00:01"},
            {"id": "u2", "conversation_id": "c2", "role": "user", "content": "p2", "created_at": "2024-01-01 00:00:02"},
            {"id": "a2", "conversation_id": "c2", "role": "assistant", "content": "r2", "created_at": "2024-01-01 00:00:03"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.8, "created_at": "2024-01-01 00:00:04"},
            {"id": "f2", "message_id": "a2", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:05"},
        ]
        trainer = _seed(db, messages, feedback)
        examples = trainer.get_training_examples()
        assert len(examples) == 2
        assert examples[0].response == "r2"
        assert examples[1].response == "r1"

    def test_get_training_examples_filters_none_rating(self, tmp_path):
        db = tmp_path / "none_rating.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r", "created_at": "2024-01-01 00:00:01"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": None, "quality_score": 0.5, "created_at": "2024-01-01 00:00:02"},
        ]
        trainer = _seed(db, messages, feedback)
        assert trainer.get_training_examples() == []

    def test_get_training_examples_filters_user_role(self, tmp_path):
        db = tmp_path / "user_role.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
        ]
        feedback = [
            {"id": "f1", "message_id": "u1", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:01"},
        ]
        trainer = _seed(db, messages, feedback)
        assert trainer.get_training_examples() == []

    def test_get_training_examples_filters_missing_message(self, tmp_path):
        db = tmp_path / "missing.db"
        feedback = [
            {"id": "f1", "message_id": "nonexistent", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:01"},
        ]
        trainer = _seed(db, [], feedback)
        assert trainer.get_training_examples() == []

    def test_get_training_examples_quality_boundary(self, tmp_path):
        db = tmp_path / "boundary.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r1", "created_at": "2024-01-01 00:00:01"},
            {"id": "a2", "conversation_id": "c1", "role": "assistant", "content": "r2", "created_at": "2024-01-01 00:00:02"},
            {"id": "a3", "conversation_id": "c1", "role": "assistant", "content": "r3", "created_at": "2024-01-01 00:00:03"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.4, "created_at": "2024-01-01 00:00:04"},
            {"id": "f2", "message_id": "a2", "rating": "thumbs_up", "quality_score": 0.5, "created_at": "2024-01-01 00:00:05"},
            {"id": "f3", "message_id": "a3", "rating": "thumbs_up", "quality_score": 0.6, "created_at": "2024-01-01 00:00:06"},
        ]
        trainer = _seed(db, messages, feedback)
        examples = trainer.get_training_examples(min_quality=0.5)
        assert len(examples) == 2

    def test_get_training_examples_limit_boundary(self, tmp_path):
        db = tmp_path / "limit.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r1", "created_at": "2024-01-01 00:00:01"},
            {"id": "a2", "conversation_id": "c1", "role": "assistant", "content": "r2", "created_at": "2024-01-01 00:00:02"},
            {"id": "a3", "conversation_id": "c1", "role": "assistant", "content": "r3", "created_at": "2024-01-01 00:00:03"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.8, "created_at": "2024-01-01 00:00:04"},
            {"id": "f2", "message_id": "a2", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:05"},
            {"id": "f3", "message_id": "a3", "rating": "thumbs_up", "quality_score": 1.0, "created_at": "2024-01-01 00:00:06"},
        ]
        trainer = _seed(db, messages, feedback)
        examples = trainer.get_training_examples(limit=2)
        assert len(examples) == 2
        assert examples[0].response == "r3"
        assert examples[1].response == "r2"


# ---------------------------------------------------------------------------
# FeedbackTrainer — DPO pairs
# ---------------------------------------------------------------------------

class TestFeedbackTrainerDPO:
    def test_dpo_multi_conversation(self, tmp_path):
        trainer = _seed_multi_conv(tmp_path)
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) == 2
        chosen_set = {p.chosen for p in pairs}
        assert "good answer A" in chosen_set
        assert "good answer B" in chosen_set

    def test_dpo_requires_both_ratings(self, tmp_path):
        db = tmp_path / "one_rating.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r1", "created_at": "2024-01-01 00:00:01"},
            {"id": "a2", "conversation_id": "c1", "role": "assistant", "content": "r2", "created_at": "2024-01-01 00:00:02"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:03"},
            {"id": "f2", "message_id": "a2", "rating": "thumbs_up", "quality_score": 0.8, "created_at": "2024-01-01 00:00:04"},
        ]
        trainer = _seed(db, messages, feedback)
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) == 0

    def test_dpo_conversation_with_only_downvotes(self, tmp_path):
        db = tmp_path / "down.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r1", "created_at": "2024-01-01 00:00:01"},
            {"id": "a2", "conversation_id": "c1", "role": "assistant", "content": "r2", "created_at": "2024-01-01 00:00:02"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_down", "quality_score": 0.1, "created_at": "2024-01-01 00:00:03"},
            {"id": "f2", "message_id": "a2", "rating": "thumbs_down", "quality_score": 0.2, "created_at": "2024-01-01 00:00:04"},
        ]
        trainer = _seed(db, messages, feedback)
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) == 0

    def test_dpo_prompt_from_prior_user_message(self, tmp_path):
        db = tmp_path / "prompt.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "the real prompt", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "good", "created_at": "2024-01-01 00:00:01"},
            {"id": "a2", "conversation_id": "c1", "role": "assistant", "content": "bad", "created_at": "2024-01-01 00:00:02"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:03"},
            {"id": "f2", "message_id": "a2", "rating": "thumbs_down", "quality_score": 0.1, "created_at": "2024-01-01 00:00:04"},
        ]
        trainer = _seed(db, messages, feedback)
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) == 1
        assert pairs[0].prompt == "the real prompt"

    def test_dpo_empty_when_no_feedback(self, tmp_path):
        db = tmp_path / "nofb.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r", "created_at": "2024-01-01 00:00:01"},
        ]
        trainer = _seed(db, messages, [])
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# FeedbackTrainer — SFT data
# ---------------------------------------------------------------------------

class TestFeedbackTrainerSFT:
    def test_sft_filters_thumbs_down(self, tmp_path):
        db = tmp_path / "sft_filter.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r1", "created_at": "2024-01-01 00:00:01"},
            {"id": "a2", "conversation_id": "c1", "role": "assistant", "content": "r2", "created_at": "2024-01-01 00:00:02"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:03"},
            {"id": "f2", "message_id": "a2", "rating": "thumbs_down", "quality_score": 0.1, "created_at": "2024-01-01 00:00:04"},
        ]
        trainer = _seed(db, messages, feedback)
        sft = trainer.prepare_sft_data()
        assert len(sft) == 1
        assert sft[0]["response"] == "r1"

    def test_sft_includes_quality_score(self, tmp_path):
        db = tmp_path / "sft_qs.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r", "created_at": "2024-01-01 00:00:01"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.85, "created_at": "2024-01-01 00:00:02"},
        ]
        trainer = _seed(db, messages, feedback)
        sft = trainer.prepare_sft_data()
        assert len(sft) == 1
        assert sft[0]["quality_score"] == 0.85

    def test_sft_default_quality_score(self, tmp_path):
        db = tmp_path / "sft_def.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r", "created_at": "2024-01-01 00:00:01"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": None, "created_at": "2024-01-01 00:00:02"},
        ]
        trainer = _seed(db, messages, feedback)
        sft = trainer.prepare_sft_data()
        assert sft[0]["quality_score"] == 1.0

    def test_sft_filters_empty_prompt(self, tmp_path):
        db = tmp_path / "sft_empty.db"
        messages = [
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r", "created_at": "2024-01-01 00:00:01"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:02"},
        ]
        trainer = _seed(db, messages, feedback)
        sft = trainer.prepare_sft_data()
        assert len(sft) == 0

    def test_sft_with_min_quality(self, tmp_path):
        db = tmp_path / "sft_mq.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r1", "created_at": "2024-01-01 00:00:01"},
            {"id": "a2", "conversation_id": "c1", "role": "assistant", "content": "r2", "created_at": "2024-01-01 00:00:02"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.3, "created_at": "2024-01-01 00:00:03"},
            {"id": "f2", "message_id": "a2", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:04"},
        ]
        trainer = _seed(db, messages, feedback)
        sft = trainer.prepare_sft_data(min_quality=0.5)
        assert len(sft) == 1
        assert sft[0]["response"] == "r2"


# ---------------------------------------------------------------------------
# FeedbackTrainer — export_for_alignment
# ---------------------------------------------------------------------------

class TestFeedbackTrainerExport:
    @pytest.fixture
    def trainer(self, tmp_path):
        db = tmp_path / "export.db"
        return _seed(
            db,
            [
                {"id": "u1", "conversation_id": "c1", "role": "user", "content": "What is Python?", "created_at": "2024-01-01 00:00:00"},
                {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "A language.", "created_at": "2024-01-01 00:00:01"},
                {"id": "a2", "conversation_id": "c1", "role": "assistant", "content": "No idea.", "created_at": "2024-01-01 00:00:02"},
            ],
            [
                {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:03"},
                {"id": "f2", "message_id": "a2", "rating": "thumbs_down", "quality_score": 0.1, "created_at": "2024-01-01 00:00:04"},
            ],
        )

    def test_export_dpo_format(self, trainer, tmp_path):
        out = tmp_path / "out"
        results = trainer.export_for_alignment(output_dir=str(out), formats=["dpo"])
        assert "dpo" in results
        path = Path(results["dpo"])
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "chosen" in record
        assert "rejected" in record
        assert "prompt" in record

    def test_export_sft_format(self, trainer, tmp_path):
        out = tmp_path / "out"
        results = trainer.export_for_alignment(output_dir=str(out), formats=["sft"])
        assert "sft" in results
        path = Path(results["sft"])
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "prompt" in record
        assert "response" in record
        assert "quality_score" in record

    def test_export_reward_format(self, trainer, tmp_path):
        out = tmp_path / "out"
        results = trainer.export_for_alignment(output_dir=str(out), formats=["reward"])
        assert "reward" in results
        path = Path(results["reward"])
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "reward" in record
            assert record["reward"] in (0.0, 1.0)

    def test_export_reward_up_is_one(self, trainer, tmp_path):
        out = tmp_path / "out"
        results = trainer.export_for_alignment(output_dir=str(out), formats=["reward"])
        lines = Path(results["reward"]).read_text().strip().split("\n")
        up_reward = [json.loads(l) for l in lines if json.loads(l)["response"] == "A language."]
        assert up_reward[0]["reward"] == 1.0

    def test_export_reward_down_is_zero(self, trainer, tmp_path):
        out = tmp_path / "out"
        results = trainer.export_for_alignment(output_dir=str(out), formats=["reward"])
        lines = Path(results["reward"]).read_text().strip().split("\n")
        down_reward = [json.loads(l) for l in lines if json.loads(l)["response"] == "No idea."]
        assert down_reward[0]["reward"] == 0.0

    def test_export_all_formats(self, trainer, tmp_path):
        out = tmp_path / "out"
        results = trainer.export_for_alignment(output_dir=str(out), formats=["dpo", "sft", "reward"])
        assert set(results.keys()) == {"dpo", "sft", "reward"}

    def test_export_creates_directory(self, trainer, tmp_path):
        out = tmp_path / "new" / "dir"
        results = trainer.export_for_alignment(output_dir=str(out), formats=["dpo"])
        assert Path(results["dpo"]).parent.exists()

    def test_export_empty_db(self, tmp_path):
        db = tmp_path / "empty.db"
        mog = MogDB(str(db))
        mog.collection("messages")
        mog.collection("feedback")
        trainer = FeedbackTrainer(db_path=str(db))
        out = tmp_path / "out"
        results = trainer.export_for_alignment(output_dir=str(out))
        for path_str in results.values():
            assert Path(path_str).exists()
            assert Path(path_str).read_text().strip() == ""


# ---------------------------------------------------------------------------
# FeedbackTrainer — export_dpo / export_sft
# ---------------------------------------------------------------------------

class TestFeedbackTrainerExportDirect:
    @pytest.fixture
    def trainer(self, tmp_path):
        db = tmp_path / "direct.db"
        return _seed(
            db,
            [
                {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
                {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "good", "created_at": "2024-01-01 00:00:01"},
                {"id": "a2", "conversation_id": "c1", "role": "assistant", "content": "bad", "created_at": "2024-01-01 00:00:02"},
            ],
            [
                {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:03"},
                {"id": "f2", "message_id": "a2", "rating": "thumbs_down", "quality_score": 0.1, "created_at": "2024-01-01 00:00:04"},
            ],
        )

    def test_export_dpo_returns_count(self, trainer, tmp_path):
        filepath = str(tmp_path / "dpo.jsonl")
        count = trainer.export_dpo(filepath)
        assert count == 1

    def test_export_dpo_writes_jsonl(self, trainer, tmp_path):
        filepath = str(tmp_path / "dpo.jsonl")
        trainer.export_dpo(filepath)
        lines = Path(filepath).read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["chosen"] == "good"
        assert record["rejected"] == "bad"

    def test_export_sft_returns_count(self, trainer, tmp_path):
        filepath = str(tmp_path / "sft.jsonl")
        count = trainer.export_sft(filepath)
        assert count == 1

    def test_export_sft_writes_jsonl(self, trainer, tmp_path):
        filepath = str(tmp_path / "sft.jsonl")
        trainer.export_sft(filepath)
        lines = Path(filepath).read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["response"] == "good"


# ---------------------------------------------------------------------------
# FeedbackTrainer — get_training_stats
# ---------------------------------------------------------------------------

class TestFeedbackTrainerStats:
    def test_stats_empty_db(self, tmp_path):
        db = tmp_path / "empty.db"
        mog = MogDB(str(db))
        mog.collection("messages")
        mog.collection("feedback")
        trainer = FeedbackTrainer(db_path=str(db))
        stats = trainer.get_training_stats()
        assert stats["total_conversations"] == 0
        assert stats["total_responses"] == 0
        assert stats["thumbs_up"] == 0
        assert stats["thumbs_down"] == 0
        assert stats["available_dpo_pairs"] == 0
        assert stats["available_sft_examples"] == 0

    def test_stats_populated(self, tmp_path):
        trainer = _seed_multi_conv(tmp_path)
        stats = trainer.get_training_stats()
        assert stats["total_conversations"] == 2
        assert stats["total_responses"] == 4
        assert stats["thumbs_up"] == 2
        assert stats["thumbs_down"] == 2
        assert stats["available_dpo_pairs"] == 2
        assert stats["available_sft_examples"] == 2

    def test_stats_single_conversation(self, tmp_path):
        db = tmp_path / "single.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "p", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r", "created_at": "2024-01-01 00:00:01"},
        ]
        feedback = [
            {"id": "f1", "message_id": "a1", "rating": "thumbs_up", "quality_score": 0.9, "created_at": "2024-01-01 00:00:02"},
        ]
        trainer = _seed(db, messages, feedback)
        stats = trainer.get_training_stats()
        assert stats["total_conversations"] == 1
        assert stats["total_responses"] == 1
        assert stats["thumbs_up"] == 1
        assert stats["thumbs_down"] == 0


# ---------------------------------------------------------------------------
# FeedbackTrainer — _message_by_id / _prompt_for
# ---------------------------------------------------------------------------

class TestFeedbackTrainerInternals:
    def test_message_by_id(self, tmp_path):
        db = tmp_path / "internal.db"
        messages = [
            {"id": "m1", "conversation_id": "c1", "role": "assistant", "content": "hello", "created_at": "2024-01-01 00:00:00"},
        ]
        trainer = _seed(db, messages, [])
        msg = trainer._message_by_id("m1")
        assert msg is not None
        assert msg["content"] == "hello"

    def test_message_by_id_not_found(self, tmp_path):
        db = tmp_path / "internal.db"
        trainer = _seed(db, [], [])
        msg = trainer._message_by_id("nonexistent")
        assert msg is None

    def test_prompt_for(self, tmp_path):
        db = tmp_path / "prompt.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "the prompt", "created_at": "2024-01-01 00:00:00"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "response", "created_at": "2024-01-01 00:00:01"},
        ]
        trainer = _seed(db, messages, [])
        prompt = trainer._prompt_for({"conversation_id": "c1", "created_at": "2024-01-01 00:00:01"})
        assert prompt == "the prompt"

    def test_prompt_for_no_prior_message(self, tmp_path):
        db = tmp_path / "noprior.db"
        messages = [
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r", "created_at": "2024-01-01 00:00:01"},
        ]
        trainer = _seed(db, messages, [])
        prompt = trainer._prompt_for({"conversation_id": "c1", "created_at": "2024-01-01 00:00:01"})
        assert prompt == ""

    def test_latest_prior_user_message(self, tmp_path):
        db = tmp_path / "multi_user.db"
        messages = [
            {"id": "u1", "conversation_id": "c1", "role": "user", "content": "first", "created_at": "2024-01-01 00:00:00"},
            {"id": "u2", "conversation_id": "c1", "role": "user", "content": "second", "created_at": "2024-01-01 00:00:02"},
            {"id": "a1", "conversation_id": "c1", "role": "assistant", "content": "r", "created_at": "2024-01-01 00:00:03"},
        ]
        trainer = _seed(db, messages, [])
        prompt = trainer._latest_prior_user_message("c1", "2024-01-01 00:00:03")
        assert prompt == "second"


# ---------------------------------------------------------------------------
# create_training_pipeline factory
# ---------------------------------------------------------------------------

class TestCreateTrainingPipeline:
    def test_factory_returns_trainer(self, tmp_path):
        db = tmp_path / "factory.db"
        trainer = create_training_pipeline(db_path=str(db))
        assert isinstance(trainer, FeedbackTrainer)

    def test_factory_default_path(self):
        trainer = create_training_pipeline()
        assert isinstance(trainer, FeedbackTrainer)
