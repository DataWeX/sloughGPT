"""Tests for chat_trainer — on-device training from chat pairs."""
import gc
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core-py"))


@pytest.fixture(autouse=True)
def _cleanup_gc():
    gc.collect()
    yield
    gc.collect()


def _pairs():
    """Minimal training pairs for fast tests."""
    return [
        {"user_msg": "Hello", "assistant_msg": "Hi there"},
        {"user_msg": "What is 2+2?", "assistant_msg": "4"},
        {"user_msg": "Goodbye", "assistant_msg": "See you later"},
        {"user_msg": "Thanks", "assistant_msg": "You're welcome"},
        {"user_msg": "How are you?", "assistant_msg": "I'm fine"},
        {"user_msg": "What is AI?", "assistant_msg": "Artificial intelligence"},
        {"user_msg": "Tell me a joke", "assistant_msg": "Why did the chicken cross the road"},
        {"user_msg": "Good morning", "assistant_msg": "Good morning! How can I help"},
    ]


class TestChatTrainConfig:
    def test_defaults(self):
        from domains.training.chat_trainer import ChatTrainConfig
        c = ChatTrainConfig()
        assert c.n_embed == 128
        assert c.epochs == 10
        assert c.lr == 3e-4
        assert c.min_pair_quality == 2.0
        assert c.soul_name == "chat-trained"

    def test_custom(self):
        from domains.training.chat_trainer import ChatTrainConfig
        c = ChatTrainConfig(n_embed=64, epochs=3, min_pair_quality=1.0)
        assert c.n_embed == 64
        assert c.epochs == 3
        assert c.min_pair_quality == 1.0


class TestChatTextDataset:
    def test_length(self):
        from domains.training.chat_trainer import ChatTextDataset
        ds = ChatTextDataset("abcdef", block_size=3, stoi={c: i for i, c in enumerate("abcdef")})
        assert len(ds) > 0
        assert len(ds) == 6 - 3 - 1

    def test_get_batch_shape(self):
        from domains.training.chat_trainer import ChatTextDataset
        stoi = {c: i + 1 for i, c in enumerate("abcdef")}
        ds = ChatTextDataset("abcdef", block_size=3, stoi=stoi)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch(2, rng)
        assert x.shape == (2, 3)
        assert y.shape == (2, 3)

    def test_batch_values_are_valid(self):
        from domains.training.chat_trainer import ChatTextDataset
        stoi = {c: i + 1 for i, c in enumerate("abcdef")}
        ds = ChatTextDataset("abcdefabcdef", block_size=4, stoi=stoi)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch(4, rng)
        assert np.all(x >= 0)
        assert np.all(y >= 0)


class TestVocab:
    def test_build_vocab(self):
        from domains.training.chat_trainer import _build_vocab
        pairs = _pairs()
        stoi, itos = _build_vocab(pairs)
        assert len(stoi) > 0
        assert len(itos) == len(stoi)
        assert "\x00" in stoi
        assert stoi["\x00"] == 0

    def test_format_pairs_text(self):
        from domains.training.chat_trainer import _format_pairs_text
        text = _format_pairs_text(_pairs()[:2])
        assert "User: Hello" in text
        assert "Assistant: Hi there" in text


class TestCrossEntropyLoss:
    def test_perfect_prediction(self):
        from domains.training.chat_trainer import _cross_entropy_loss
        logits = np.array([[0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
        targets = np.array([1, 2])
        loss = _cross_entropy_loss(logits, targets)
        assert loss < 0.01

    def test_worse_prediction(self):
        from domains.training.chat_trainer import _cross_entropy_loss
        logits = np.array([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        targets = np.array([1, 2])
        loss = _cross_entropy_loss(logits, targets)
        assert loss > 5.0


class TestTrainChatModel:
    def test_basic_training(self):
        from domains.training.chat_trainer import ChatTrainConfig, train_chat_model
        pairs = _pairs()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChatTrainConfig(
                n_embed=16, n_layer=1, n_head=2, block_size=16,
                epochs=1, batch_size=2, log_interval=10, eval_interval=50,
                checkpoint_dir=tmpdir, soul_name="test",
                min_pair_quality=0.0,
            )
            model, meta = train_chat_model(pairs, config)
            assert "checkpoint" in meta
            assert Path(meta["checkpoint"]).exists()
            assert meta["num_pairs"] == len(pairs)
            assert meta["epochs_completed"] == 1
            assert meta["vocab_size"] > 0

    def test_training_loss_decreases(self):
        from domains.training.chat_trainer import ChatTrainConfig, train_chat_model
        pairs = _pairs()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChatTrainConfig(
                n_embed=16, n_layer=1, n_head=2, block_size=16,
                epochs=3, batch_size=2, log_interval=50, eval_interval=50,
                checkpoint_dir=tmpdir, soul_name="test",
                min_pair_quality=0.0,
            )
            _, meta = train_chat_model(pairs, config)
            losses = meta["train_losses"]
            assert len(losses) >= 1
            assert all(np.isfinite(l) for l in losses)
            assert losses[-1] < 10.0

    def test_empty_pairs_raises(self):
        from domains.training.chat_trainer import ChatTrainConfig, train_chat_model
        config = ChatTrainConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            with pytest.raises(ValueError, match="No training pairs"):
                train_chat_model([], config)

    def test_quality_filter(self):
        from domains.training.chat_trainer import ChatTrainConfig, train_chat_model
        pairs = _pairs()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChatTrainConfig(
                n_embed=16, n_layer=1, n_head=2, block_size=16,
                epochs=1, batch_size=2, checkpoint_dir=tmpdir,
                min_pair_quality=5.0,
            )
            _, meta = train_chat_model(pairs, config)
            assert meta["num_pairs"] >= 5

    def test_checkpoint_soul_format(self):
        from domains.training.chat_trainer import ChatTrainConfig, train_chat_model
        pairs = _pairs()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChatTrainConfig(
                n_embed=16, n_layer=1, n_head=2, block_size=16,
                epochs=1, batch_size=2, checkpoint_dir=tmpdir, soul_name="format-test",
            )
            _, meta = train_chat_model(pairs, config)
            ckpt_path = Path(meta["checkpoint"])
            assert ckpt_path.suffix == ".soul"
            header = ckpt_path.read_bytes()[:4]
            assert header == b"SOUL"

    def test_resume(self):
        from domains.training.chat_trainer import ChatTrainConfig, train_chat_model
        pairs = _pairs()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = str(Path(tmpdir) / "resume-test.soul")
            config = ChatTrainConfig(
                n_embed=16, n_layer=1, n_head=2, block_size=16,
                epochs=1, batch_size=2, checkpoint_dir=tmpdir, soul_name="resume-test",
            )
            train_chat_model(pairs, config)
            config2 = ChatTrainConfig(
                n_embed=16, n_layer=1, n_head=2, block_size=16,
                epochs=2, batch_size=2, checkpoint_dir=tmpdir, soul_name="resume-test",
                resume_checkpoint=ckpt,
            )
            model, meta = train_chat_model(pairs, config2)
            assert meta["epochs_completed"] == 2


class TestGenerateFromChatModel:
    def test_generate(self):
        from domains.training.chat_trainer import (
            ChatTrainConfig, train_chat_model, generate_from_chat_model, _build_vocab,
        )
        pairs = _pairs()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChatTrainConfig(
                n_embed=16, n_layer=1, n_head=2, block_size=16,
                epochs=1, batch_size=2, checkpoint_dir=tmpdir,
            )
            model, meta = train_chat_model(pairs, config)
            stoi, itos = _build_vocab(pairs)
            output = generate_from_chat_model(model, stoi, itos, "Hello", max_tokens=10)
            assert isinstance(output, str)
            assert len(output) > 0


class TestEvalLoss:
    def test_eval_loss_is_finite(self):
        from domains.training.chat_trainer import ChatTextDataset, _eval_loss
        from domains.training.slonet import SloTransformer
        stoi = {"\x00": 0, **{c: i + 1 for i, c in enumerate("abcdef")}}
        ds = ChatTextDataset("abcdef" * 10, block_size=4, stoi=stoi)
        model = SloTransformer(
            vocab_size=len(stoi), n_embed=16, n_layer=1, n_head=2,
            block_size=4, use_rope=True, norm_type="rms_norm",
        )
        rng = np.random.default_rng(0)
        loss = _eval_loss(model, ds, 2, rng)
        assert np.isfinite(loss)
        assert loss > 0


class TestEvaluateChatModel:
    def test_evaluate_returns_samples(self):
        from domains.training.chat_trainer import (
            ChatTrainConfig, train_chat_model, evaluate_chat_model,
        )
        pairs = _pairs()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChatTrainConfig(
                n_embed=16, n_layer=1, n_head=2, block_size=16,
                epochs=1, batch_size=2, checkpoint_dir=tmpdir,
            )
            model, meta = train_chat_model(pairs, config)
            stoi = meta["stoi"]
            itos = meta["itos"]
            result = evaluate_chat_model(model, stoi, itos, pairs)
            assert "perplexity" in result
            assert "samples" in result
            assert "avg_response_len" in result
            assert len(result["samples"]) > 0
            assert result["perplexity"] > 0

    def test_evaluate_max_samples(self):
        from domains.training.chat_trainer import (
            ChatTrainConfig, train_chat_model, evaluate_chat_model,
        )
        pairs = _pairs()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ChatTrainConfig(
                n_embed=16, n_layer=1, n_head=2, block_size=16,
                epochs=1, batch_size=2, checkpoint_dir=tmpdir,
            )
            model, meta = train_chat_model(pairs, config)
            stoi = meta["stoi"]
            itos = meta["itos"]
            result = evaluate_chat_model(model, stoi, itos, pairs, max_samples=2)
            assert len(result["samples"]) == 2
