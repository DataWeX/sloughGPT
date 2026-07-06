"""
Tests for VideoCaptionTrainer — creation, vocab, checkpoint save/load.
"""

from pathlib import Path
import json
import tempfile
import numpy as np
import pytest

pytestmark = pytest.mark.slow

from domains.training.video_trainer import VideoCaptionTrainer, list_video_checkpoints


def test_video_trainer_creation():
    """Verify trainer initializes with default parameters."""
    trainer = VideoCaptionTrainer()
    assert trainer.embed_dim == 256
    assert trainer.hidden_dim == 512
    assert trainer.max_frames == 8
    assert trainer.lr == 3e-4
    assert trainer.vision_encoder is not None
    assert trainer.temporal_encoder is not None
    assert trainer.decoder is not None
    assert not trainer._trained


def test_vocab_building():
    """Test character-level vocabulary from captions."""
    trainer = VideoCaptionTrainer(vocab_size=512)
    captions = ["hello world", "cat video", "test"]
    trainer.build_vocab(captions)
    assert "<PAD>" in trainer._vocab
    assert "<BOS>" in trainer._vocab
    assert "<EOS>" in trainer._vocab
    assert "h" in trainer._vocab
    assert "w" in trainer._vocab
    assert len(trainer._vocab) >= 6  # PAD + BOS + EOS + at least 3 chars


def test_encode_decode_roundtrip():
    """Test text encoding and decoding are inverses (minus EOS)."""
    trainer = VideoCaptionTrainer(vocab_size=512)
    trainer.build_vocab(["hello world"])
    encoded = trainer.encode_text("hello")
    decoded = trainer.decode_text(encoded)
    assert decoded == "hello"


def test_decode_stops_at_eos():
    """Test that decode stops at EOS token."""
    trainer = VideoCaptionTrainer(vocab_size=512)
    trainer.build_vocab(["abc"])
    eos = trainer._vocab["<EOS>"]
    # Manually construct tokens: BOS, 'a', EOS, extra
    bos = trainer._vocab["<BOS>"]
    a_id = trainer._vocab.get("a", 0)
    decoded = trainer.decode_text([bos, a_id, eos, a_id, a_id])
    assert decoded == "a"


def test_list_checkpoints_empty():
    """Test listing checkpoints when none exist."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpts = list_video_checkpoints(tmp)
        assert ckpts == []


def test_save_and_list_checkpoint():
    """Test saving a checkpoint and listing it."""
    trainer = VideoCaptionTrainer(vocab_size=512)
    trainer.build_vocab(["test caption"])

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer._save_checkpoint(ckpt_dir, "test_ckpt", epoch=1, step=10, loss=0.5)

        assert (ckpt_dir / "test_ckpt.npz").exists()
        assert (ckpt_dir / "test_ckpt_meta.json").exists()

        ckpts = list_video_checkpoints(tmp)
        assert len(ckpts) == 1
        assert ckpts[0]["name"] == "test_ckpt"
        assert ckpts[0]["epoch"] == 1
        assert ckpts[0]["step"] == 10
        assert ckpts[0]["loss"] == 0.5


def test_generate_returns_untrained_message():
    """Test that generate returns placeholder message when untrained."""
    trainer = VideoCaptionTrainer()
    result = trainer.generate("/nonexistent/video.mp4")
    assert result == "[model untrained — train on videos first]"


def test_dataset_load():
    """Test loading a JSONL dataset."""
    trainer = VideoCaptionTrainer()
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "dataset.jsonl"
        with open(data_path, "w") as f:
            f.write(json.dumps({"video_path": "/videos/a.mp4", "caption": "cat walking"}) + "\n")
            f.write(json.dumps({"video_path": "/videos/b.mp4", "caption": "dog running"}) + "\n")

        entries = trainer.load_dataset(str(data_path))
        assert len(entries) == 2
        assert entries[0]["caption"] == "cat walking"


def test_dataset_load_skips_invalid():
    """Test that entries without required fields are skipped."""
    trainer = VideoCaptionTrainer()
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "dataset.jsonl"
        with open(data_path, "w") as f:
            f.write(json.dumps({"video_path": "/v/a.mp4", "caption": "test"}) + "\n")
            f.write(json.dumps({"missing": "fields"}) + "\n")
            f.write(json.dumps({"video_path": "/v/b.mp4", "caption": "test2"}) + "\n")

        entries = trainer.load_dataset(str(data_path))
        assert len(entries) == 2
        assert entries[1]["caption"] == "test2"
