"""
Tests for HFDPOTrainer — Direct Preference Optimization for HF models.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def mock_model():
    """A tiny mock HF causal LM for testing DPO logic."""
    model = MagicMock()
    model.config.hidden_size = 64

    # Mock forward pass — return a MagicMock with .loss
    def forward(x, labels=None, **kw):
        out = MagicMock()
        out.loss = torch.tensor(0.5, requires_grad=True)
        out.logits = torch.randn(1, x.shape[1], 1000)
        return out

    model.side_effect = forward
    model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]

    # Named parameters for snapshot
    model.named_parameters.return_value = [
        ("lora_A", torch.nn.Parameter(torch.randn(10, 10))),
    ]
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns simple IDs."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.vocab_size = 1000

    def encode(text, **kw):
        class MockEncoding:
            input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])

            def to(self, device):
                return self

        return MockEncoding()

    tokenizer.side_effect = encode
    tokenizer.encode = encode
    tokenizer.__call__ = encode
    tokenizer.apply_chat_template = lambda messages, **kw: f"{messages[0]['content']}"
    return tokenizer


@pytest.fixture
def feedback_db():
    """Create a temporary feedback SQLite DB with test data."""
    import sqlite3

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            message_id TEXT,
            rating TEXT,
            created_at TIMESTAMP
        )
    """)

    # Insert test data: 2 conversations with mixed feedback
    conn.execute(
        "INSERT INTO messages VALUES ('m1', 'conv1', 'user', 'What is 2+2?', '2024-01-01T00:00:00')"
    )
    conn.execute(
        "INSERT INTO messages VALUES ('m2', 'conv1', 'assistant', '4', '2024-01-01T00:01:00')"
    )
    conn.execute(
        "INSERT INTO messages VALUES ('m3', 'conv1', 'assistant', '5', '2024-01-01T00:02:00')"
    )
    conn.execute(
        "INSERT INTO feedback VALUES ('f1', 'm2', 'thumbs_up', '2024-01-01T00:01:00')"
    )
    conn.execute(
        "INSERT INTO feedback VALUES ('f2', 'm3', 'thumbs_down', '2024-01-01T00:02:00')"
    )

    conn.commit()
    conn.close()
    yield db_path
    os.unlink(db_path)


class TestHFDPOTrainer:
    """Tests for HFDPOTrainer."""

    def test_import(self):
        """Verify the module imports correctly."""
        from domains.feedback.hf_dpo import HFDPOTrainer, DPOPair
        assert HFDPOTrainer is not None
        assert DPOPair is not None

    def test_prepare_dpo_pairs(self, feedback_db):
        """Test DPO pair extraction from feedback DB."""
        from domains.feedback.hf_dpo import HFDPOTrainer

        trainer = HFDPOTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            db_path=feedback_db,
        )
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) >= 1
        pair = pairs[0]
        assert pair.chosen == "4"
        assert pair.rejected == "5"
        assert pair.prompt == "What is 2+2?"

    def test_prepare_dpo_pairs_insufficient(self):
        """Test empty result when no mixed-feedback conversations exist."""
        import sqlite3
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, conversation_id TEXT,
                role TEXT, content TEXT, created_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY, message_id TEXT,
                rating TEXT, created_at TIMESTAMP
            )
        """)
        # Only thumbs-up (no mixed feedback)
        conn.execute(
            "INSERT INTO messages VALUES ('m1', 'conv1', 'user', 'hi', '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO messages VALUES ('m2', 'conv1', 'assistant', 'hello', '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO feedback VALUES ('f1', 'm2', 'thumbs_up', '2024-01-01')"
        )
        conn.commit()
        conn.close()

        from domains.feedback.hf_dpo import HFDPOTrainer
        trainer = HFDPOTrainer(
            model=MagicMock(), tokenizer=MagicMock(), db_path=tmp.name,
        )
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) == 0
        os.unlink(tmp.name)

    def test_benchmark_ppl(self, mock_model, mock_tokenizer):
        """Test PPL computation runs without error."""
        from domains.feedback.hf_dpo import HFDPOTrainer

        trainer = HFDPOTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        ppl = trainer._compute_ppl(["Hello world"])
        assert isinstance(ppl, float)
        assert ppl > 0

    def test_snapshot_and_restore(self):
        """Test weight snapshot/restore preserves original values."""
        from domains.feedback.hf_dpo import HFDPOTrainer

        model = torch.nn.Linear(4, 4)
        orig_weight = model.weight.data.clone()

        trainer = HFDPOTrainer(
            model=model,
            tokenizer=MagicMock(),
        )

        snapshot = trainer._take_snapshot()
        # Modify weights
        with torch.no_grad():
            model.weight.data += 1.0
        assert not torch.allclose(model.weight.data, orig_weight)

        # Restore
        trainer._restore_snapshot(snapshot)
        assert torch.allclose(model.weight.data, orig_weight)

    def test_train_skipped_no_pairs(self, mock_model, mock_tokenizer):
        """Test train returns skipped when no pairs available."""
        from domains.feedback.hf_dpo import HFDPOTrainer

        trainer = HFDPOTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        result = trainer.train(pairs=[])
        assert result["status"] == "skipped"

    def test_train_skipped_single_pair(self, mock_model, mock_tokenizer):
        """Test train returns skipped with only 1 pair (need >=2)."""
        from domains.feedback.hf_dpo import HFDPOTrainer, DPOPair

        trainer = HFDPOTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        pair = DPOPair(chosen="good answer", rejected="bad answer", prompt="test")
        result = trainer.train(pairs=[pair])
        assert result["status"] == "skipped"

    def test_export_pairs(self, feedback_db, mock_model, mock_tokenizer):
        """Test exporting DPO pairs to JSONL."""
        from domains.feedback.hf_dpo import HFDPOTrainer

        trainer = HFDPOTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            db_path=feedback_db,
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            count = trainer.export_pairs(f.name)
        assert count >= 1
        with open(f.name) as f:
            line = json.loads(f.readline())
            assert "chosen" in line
            assert "rejected" in line
            assert "prompt" in line
        os.unlink(f.name)

    def test_dpo_pair_dataclass(self):
        """Test DPOPair dataclass works."""
        from domains.feedback.hf_dpo import DPOPair
        p = DPOPair(chosen="a", rejected="b", prompt="c")
        assert p.chosen == "a"
        assert p.rejected == "b"
        assert p.prompt == "c"
