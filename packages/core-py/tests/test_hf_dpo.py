"""
Tests for HFDPOTrainer — Direct Preference Optimization on feedback pairs.

The trainer is intentionally torch-free (SloNet numpy autograd), so these
tests run without torch. Pair building is exercised against the real
FeedbackDB SQLite schema via a patched get_feedback_db; the training path
is tested end-to-end with a small real SloTransformer.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

from domains.feedback.hf_dpo import DEFAULT_EPOCHS, HFDPOTrainer


@pytest.fixture
def feedback_db():
    """A temporary feedback SQLite DB with the real schema and mixed data."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, user_id TEXT, title TEXT,
            created_at TEXT, updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT,
            content TEXT, embedding BLOB, created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY, message_id TEXT, rating TEXT,
            quality_score REAL, context_snippet TEXT, created_at TEXT
        )
    """)

    conn.execute(
        "INSERT INTO messages VALUES ('m1', 'conv1', 'user', 'What is 2+2?', NULL, '2024-01-01T00:00:00')"
    )
    conn.execute(
        "INSERT INTO messages VALUES ('m2', 'conv1', 'assistant', '4', NULL, '2024-01-01T00:01:00')"
    )
    conn.execute(
        "INSERT INTO messages VALUES ('m3', 'conv1', 'assistant', '5', NULL, '2024-01-01T00:02:00')"
    )
    conn.execute(
        "INSERT INTO messages VALUES ('m4', 'conv2', 'assistant', 'Paris', NULL, '2024-01-01T00:03:00')"
    )
    conn.execute(
        "INSERT INTO messages VALUES ('m6', 'conv2', 'assistant', 'Lyon', NULL, '2024-01-01T00:04:00')"
    )
    conn.execute(
        "INSERT INTO feedback VALUES ('f1', 'm2', 'thumbs_up', 1.0, NULL, '2024-01-01T00:01:00')"
    )
    conn.execute(
        "INSERT INTO feedback VALUES ('f2', 'm3', 'thumbs_down', 0.1, NULL, '2024-01-01T00:02:00')"
    )
    conn.execute(
        "INSERT INTO feedback VALUES ('f3', 'm4', 'thumbs_up', 1.0, NULL, '2024-01-01T00:03:00')"
    )
    conn.execute(
        "INSERT INTO feedback VALUES ('f4', 'm6', 'thumbs_down', 0.1, NULL, '2024-01-01T00:04:00')"
    )

    conn.commit()
    conn.close()
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def real_db(feedback_db):
    """Patch get_feedback_db to point at the temp DB."""
    from domains.feedback.database import FeedbackDB

    with patch("domains.feedback.hf_dpo.get_feedback_db", return_value=FeedbackDB(feedback_db)):
        yield feedback_db


class StubTokenizer:
    """A tokenizer stub mapping ASCII chars to in-vocab ids."""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, **kw):
        return [max(1, ord(ch) % self.vocab_size) for ch in text if ord(ch) < 128]


def _small_transformer():
    from domains.training.slonet import SloTransformer

    return SloTransformer(
        vocab_size=128, n_embed=32, n_layer=1, n_head=2,
        block_size=64, max_seq_len=64, dropout=0.0,
    )


class TestHFDPOTrainer:
    """Tests for HFDPOTrainer."""

    def test_import(self):
        """Verify the torch-free module imports cleanly."""
        from domains.feedback.hf_dpo import DPO_BETA, DEFAULT_LR, HFDPOTrainer
        assert HFDPOTrainer is not None
        assert DPO_BETA == 0.1
        assert DEFAULT_LR == 1e-4

    def test_prepare_dpo_pairs_prefers_same_conversation(self, real_db):
        """A thumbs-down pairs with the same-conversation thumbs-up."""
        trainer = HFDPOTrainer(model=MagicMock(), tokenizer=MagicMock())
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) >= 2
        by_rejected = {p["rejected"]: p["chosen"] for p in pairs}
        assert by_rejected["5"] == "4"  # same-conversation preference (m2/m3)
        assert by_rejected["Lyon"] == "Paris"  # same-conversation preference (m4/m6)

    def test_prepare_dpo_pairs_insufficient(self):
        """Empty result when no mixed-feedback conversations exist."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT,
                content TEXT, embedding BLOB, created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY, message_id TEXT, rating TEXT,
                quality_score REAL, context_snippet TEXT, created_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO messages VALUES ('m1', 'conv1', 'user', 'hi', NULL, '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO messages VALUES ('m2', 'conv1', 'assistant', 'hello', NULL, '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO feedback VALUES ('f1', 'm2', 'thumbs_up', 1.0, NULL, '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO feedback VALUES ('f5', 'm6', 'thumbs_down', 0.1, NULL, '2024-01-01')"
        )
        conn.commit()
        conn.close()

        from domains.feedback.database import FeedbackDB

        with patch(
            "domains.feedback.hf_dpo.get_feedback_db", return_value=FeedbackDB(tmp.name)
        ):
            trainer = HFDPOTrainer(model=MagicMock(), tokenizer=MagicMock())
            pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) == 0
        os.unlink(tmp.name)

    def test_prepare_dpo_pairs_max_pairs_cap(self, real_db):
        """max_pairs caps the returned pair list."""
        trainer = HFDPOTrainer(model=MagicMock(), tokenizer=MagicMock())
        full = trainer.prepare_dpo_pairs()
        assert len(full) >= 1
        capped = trainer.prepare_dpo_pairs(max_pairs=1)
        assert len(capped) == 1
        assert capped[0] == full[0]

    def test_prepare_dpo_pairs_ignores_empty_content_feedback(self, feedback_db):
        """Blank assistant content is not used to build pairs."""
        from domains.feedback.database import FeedbackDB

        conn = sqlite3.connect(feedback_db)
        conn.execute(
            "INSERT INTO messages VALUES ('m7', 'conv1', 'assistant', '   ', NULL, '2024-01-01T00:04:00')"
        )
        conn.execute(
            "INSERT INTO feedback VALUES ('f7', 'm7', 'thumbs_down', 0.1, NULL, '2024-01-01T00:04:00')"
        )
        conn.commit()
        conn.close()

        with patch(
            "domains.feedback.hf_dpo.get_feedback_db", return_value=FeedbackDB(feedback_db)
        ):
            trainer = HFDPOTrainer(model=MagicMock(), tokenizer=MagicMock())
            pairs = trainer.prepare_dpo_pairs()
        contents = [p["rejected"] for p in pairs]
        assert "   " not in contents

    def test_is_trainable_rejects_none(self):
        """None model is not trainable."""
        assert HFDPOTrainer._is_trainable(None) is False

    def test_is_trainable_rejects_missing_parameters(self):
        """A model without parameters() is not trainable."""
        class NoParams:
            def forward(self, x):
                return x

        assert HFDPOTrainer._is_trainable(NoParams()) is False

    def test_is_trainable_accepts_real_slonet(self):
        """A real SloTransformer is trainable."""
        from domains.training.slonet import Tensor

        assert HFDPOTrainer._is_trainable(_small_transformer()) is True

    def test_encode_via_tokenizer(self):
        """Tokenized text returns integer ids."""
        trainer = HFDPOTrainer(model=MagicMock(), tokenizer=StubTokenizer())
        ids = trainer._encode("hello")
        assert ids.dtype.kind == "i"
        assert len(ids) == 5
        assert (ids >= 1).all()

    def test_encode_falls_back_to_char_ords(self):
        """Without a tokenizer, ASCII ords are used."""
        trainer = HFDPOTrainer(model=MagicMock(), tokenizer=None)
        ids = trainer._encode("abc")
        assert ids.tolist() == [ord("a"), ord("b"), ord("c")]

    def test_train_rejects_zero_pairs(self, real_db):
        """train returns rejected when no pairs were provided."""
        trainer = HFDPOTrainer(model=MagicMock(), tokenizer=MagicMock())
        result = trainer.train(pairs=[])
        assert result["status"] == "rejected"
        assert "at least 2" in result["reason"]
        assert result["steps"] == 0
        assert result["pairs_trained"] == 0

    def test_train_rejects_single_pair(self, real_db):
        """train returns rejected with only 1 pair (need >=2)."""
        trainer = HFDPOTrainer(model=MagicMock(), tokenizer=MagicMock())
        pairs = [
            {"chosen": "good answer", "rejected": "bad answer"},
        ]
        result = trainer.train(pairs=pairs)
        assert result["status"] == "rejected"
        assert result["pairs_trained"] == 0

    def test_train_rejects_mock_model(self, real_db):
        """A MagicMock model is not SloNet-trainable -> honest rejection."""
        trainer = HFDPOTrainer(model=MagicMock(), tokenizer=StubTokenizer())
        pairs = [
            {"chosen": "good answer", "rejected": "bad answer"},
            {"chosen": "better answer", "rejected": "worse answer"},
        ]
        result = trainer.train(pairs=pairs)
        assert result["status"] == "rejected"
        assert "trainable SloNet" in result["reason"]
        assert result["steps"] == 0

    def test_train_accepted_with_real_slonet(self, real_db):
        """train runs a real DPO gradient update on a SloTransformer."""
        model = _small_transformer()
        tokenizer = StubTokenizer()
        trainer = HFDPOTrainer(model=model, tokenizer=tokenizer, learning_rate=1e-3)
        pairs = [
            {"chosen": "the cat sat on the mat", "rejected": "the dog ran away fast"},
            {"chosen": "a sunny day is nice", "rejected": "a rainy day is sad now"},
        ]
        result = trainer.train(pairs=pairs)
        assert result["status"] == "accepted"
        assert result["reason"] is None
        assert result["steps"] == len(pairs) * DEFAULT_EPOCHS
        assert result["pairs_trained"] == len(pairs)
        assert result["avg_loss"] >= 0
        assert result["ppl_before"] > 0
        assert result["ppl_after"] > 0

    def test_train_accepted_uses_feedback_pairs(self, real_db):
        """train builds pairs from the feedback store when none supplied."""
        model = _small_transformer()
        trainer = HFDPOTrainer(model=model, tokenizer=StubTokenizer(), learning_rate=1e-3)
        result = trainer.train()
        assert result["status"] == "accepted"
        assert result["pairs_trained"] >= 1

    def test_export_pairs(self, real_db):
        """Export DPO pairs to JSONL."""

        trainer = HFDPOTrainer(model=MagicMock(), tokenizer=MagicMock())
        pairs = trainer.prepare_dpo_pairs()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = f.name
            with open(path, "w") as out:
                for p in pairs:
                    out.write(json.dumps(p) + "\n")
        with open(path) as f:
            line = json.loads(f.readline())
            assert "chosen" in line
            assert "rejected" in line
        os.unlink(path)

    def test_dpo_pair_dict_contract(self, real_db):
        """DPO pair records expose the documented dict shape."""
        trainer = HFDPOTrainer(model=MagicMock(), tokenizer=MagicMock())
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) >= 1
        assert isinstance(pairs[0], dict)
        assert set(["chosen", "rejected"]).issubset(pairs[0].keys())
        assert isinstance(pairs[0]["chosen"], str)
        assert isinstance(pairs[0]["rejected"], str)