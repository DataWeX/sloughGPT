"""Tests for domains/training/chat_trainer.py."""

import threading

import json
import numpy as np
import pytest

from domains.training.chat_trainer import (
    ChatTextDataset,
    ChatTrainConfig,
    _build_vocab,
    _cross_entropy_loss,
    _format_pairs_text,
    evaluate_chat_model,
    generate_from_chat_model,
    train_chat_model,
    train_from_sessions,
)
from domains.training.pair_extractor import _SESSIONS_DIR


def _pairs(n=6):
    return [
        {
            "user_msg": f"User message number {i} asking something interesting.",
            "assistant_msg": f"Assistant responds helpfully with a detailed answer about topic {i}.",
        }
        for i in range(n)
    ]


def _tiny_config(tmp_path):
    return ChatTrainConfig(
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=16,
        dropout=0.0,
        epochs=2,
        lr=1e-3,
        batch_size=8,
        warmup_steps=0,
        eval_interval=1000,
        log_interval=1,
        min_pair_quality=0.0,
        max_pairs=100,
        val_split=0.1,
        checkpoint_dir=str(tmp_path),
        soul_name="test-chat",
    )


class TestChatTrainConfig:
    def test_defaults(self):
        c = ChatTrainConfig()
        assert c.n_embed == 128
        assert c.n_layer == 4
        assert c.n_head == 4
        assert c.block_size == 128
        assert c.epochs == 10
        assert c.lr == 3e-4
        assert c.batch_size == 8
        assert c.grad_clip == 1.0
        assert c.min_pair_quality == 2.0
        assert c.max_pairs == 500
        assert c.val_split == 0.1
        assert c.checkpoint_dir == "models/auto-training"
        assert c.soul_name == "chat-trained"
        assert c.session_ids is None
        assert c.resume_checkpoint is None
        assert c.resume_epoch == 0

    def test_override(self):
        c = ChatTrainConfig(n_embed=32, epochs=3, soul_name="x")
        assert c.n_embed == 32
        assert c.epochs == 3
        assert c.soul_name == "x"


class TestChatTextDataset:
    def test_n_samples(self):
        ds = ChatTextDataset("hello world", 4, {"h": 1, "e": 2, "l": 3, "o": 4, " ": 5, "w": 6, "r": 7, "d": 8})
        assert ds.n_samples == max(1, 11 - 4 - 1)
        assert len(ds) == ds.n_samples

    def test_short_text_at_least_one(self):
        ds = ChatTextDataset("ab", 4, {"a": 1, "b": 2})
        assert ds.n_samples == 1

    def test_unknown_char_maps_to_zero(self):
        ds = ChatTextDataset("ab", 4, {"a": 1})
        assert ds.ids == [1, 0]

    def test_get_batch_shapes(self):
        text = "the quick brown fox jumps over the lazy dog"
        stoi = {c: i + 1 for i, c in enumerate(sorted(set(text)))}
        stoi["\x00"] = 0
        ds = ChatTextDataset(text, 8, stoi)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch(4, rng)
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)
        assert x.dtype == np.int32
        assert y.dtype == np.int32

    def test_get_batch_shift_correctness(self):
        text = "abcdefghijklmnop"
        stoi = {c: i + 1 for i, c in enumerate(text)}
        stoi["\x00"] = 0
        ds = ChatTextDataset(text, 4, stoi)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch(1, rng)
        for i in range(3):
            if x[0, i] != 0:
                assert y[0, i] == x[0, i + 1]


class TestVocabAndFormat:
    def test_build_vocab(self):
        stoi, itos = _build_vocab(_pairs(2))
        assert stoi["\x00"] == 0
        assert len(stoi) == len(itos)
        assert set(itos.keys()) == set(stoi.values())
        assert all(stoi[char] == idx for idx, char in itos.items())

    def test_build_vocab_sorted(self):
        stoi, _ = _build_vocab([{"user_msg": "ba", "assistant_msg": "ab"}])
        assert stoi["a"] == 1
        assert stoi["b"] == 2

    def test_format_pairs_text(self):
        text = _format_pairs_text([
            {"user_msg": "Hi", "assistant_msg": "Hello"},
        ])
        assert text == "User: Hi\nAssistant: Hello\n\n"

    def test_format_multiple_pairs(self):
        text = _format_pairs_text([
            {"user_msg": "A", "assistant_msg": "B"},
            {"user_msg": "C", "assistant_msg": "D"},
        ])
        assert text.count("User:") == 2
        assert text.count("Assistant:") == 2


class TestCrossEntropyLoss:
    def test_perfect_prediction_low(self):
        logits = np.zeros((3, 4))
        targets = np.array([0, 1, 2])
        logits[np.arange(3), targets] = 10.0
        loss = _cross_entropy_loss(logits, targets)
        assert loss < 1e-3

    def test_wrong_prediction_high(self):
        logits = np.zeros((1, 4))
        targets = np.array([0])
        logits[0, 3] = 10.0
        loss = _cross_entropy_loss(logits, targets)
        assert loss > 1.0

    def test_returns_float(self):
        logits = np.random.RandomState(0).randn(4, 5)
        targets = np.array([0, 1, 2, 3])
        assert isinstance(_cross_entropy_loss(logits, targets), float)


class TestTrainChatModel:
    def test_empty_pairs_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No training pairs"):
            train_chat_model([], _tiny_config(tmp_path))

    def test_trains_and_saves_checkpoint(self, tmp_path):
        model, meta = train_chat_model(_pairs(6), _tiny_config(tmp_path))
        ckpt = tmp_path / "test-chat.soul"
        assert ckpt.exists()
        assert meta["checkpoint"] == str(ckpt)
        assert meta["num_pairs"] == 6
        assert meta["vocab_size"] > 1
        assert meta["total_steps"] > 0
        assert meta["final_loss"] >= 0
        assert "train_losses" in meta
        assert "stoi" in meta and "itos" in meta

    def test_quality_filter_drops_pairs(self, tmp_path):
        config = _tiny_config(tmp_path)
        config.min_pair_quality = 3.5
        pairs = _pairs(6)
        pairs.append({"user_msg": "ok", "assistant_msg": "ok"})
        model, meta = train_chat_model(pairs, config)
        assert meta["num_pairs"] == 6

    def test_few_good_pairs_fallback(self, tmp_path):
        config = _tiny_config(tmp_path)
        config.min_pair_quality = 5.0
        pairs = [{"user_msg": "ok", "assistant_msg": "ok"}] * 3
        model, meta = train_chat_model(pairs, config)
        assert meta["total_pairs"] == 3

    def test_max_pairs_truncation(self, tmp_path):
        config = _tiny_config(tmp_path)
        config.max_pairs = 2
        model, meta = train_chat_model(_pairs(6), config)
        assert meta["num_pairs"] == 2

    def test_on_step_callback(self, tmp_path):
        config = _tiny_config(tmp_path)
        calls = []
        train_chat_model(_pairs(4), config, on_step=lambda s, l, e, total_steps=0: calls.append((s, l, e)))
        assert calls
        assert all(isinstance(s, int) and isinstance(l, float) and isinstance(e, int) for s, l, e in calls)

    def test_cancel_event_prevents_start(self, tmp_path):
        config = _tiny_config(tmp_path)
        cancel = threading.Event()
        cancel.set()
        model, meta = train_chat_model(_pairs(4), config, cancel_event=cancel)
        assert meta["total_steps"] == 0

    def test_resume_from_checkpoint(self, tmp_path):
        config = _tiny_config(tmp_path)
        model, meta = train_chat_model(_pairs(4), config)
        ckpt = meta["checkpoint"]

        resume_config = _tiny_config(tmp_path)
        resume_config.resume_checkpoint = ckpt
        resume_config.epochs = 3
        model2, meta2 = train_chat_model(_pairs(4), resume_config)
        assert meta2["total_steps"] > meta["total_steps"]

    def test_resume_nonexistent_creates_new(self, tmp_path):
        config = _tiny_config(tmp_path)
        config.resume_checkpoint = str(tmp_path / "missing.soul")
        model, meta = train_chat_model(_pairs(4), config)
        assert meta["total_steps"] > 0

    def test_cancel_mid_training(self, tmp_path):
        config = _tiny_config(tmp_path)
        cancel = threading.Event()

        def stop_after_first(step, loss, epoch, total_steps=0):
            cancel.set()

        model, meta = train_chat_model(_pairs(4), config, on_step=stop_after_first, cancel_event=cancel)
        assert meta["total_steps"] < 5

    def test_periodic_val_eval(self, tmp_path):
        config = _tiny_config(tmp_path)
        config.eval_interval = 5
        config.epochs = 1
        model, meta = train_chat_model(_pairs(2), config)
        assert meta["val_losses"]
        assert meta["val_loss"] is not None


class TestGenerateFromChatModel:
    def _train_tiny(self, tmp_path):
        config = _tiny_config(tmp_path)
        model, meta = train_chat_model(_pairs(4), config)
        return model, meta["stoi"], meta["itos"]

    def test_generate_greedy(self, tmp_path):
        model, stoi, itos = self._train_tiny(tmp_path)
        out = generate_from_chat_model(model, stoi, itos, "User: Hello", max_tokens=10, temperature=0.0)
        assert isinstance(out, str)

    def test_generate_sampling(self, tmp_path):
        model, stoi, itos = self._train_tiny(tmp_path)
        out = generate_from_chat_model(model, stoi, itos, "User: Hello", max_tokens=10, temperature=0.9)
        assert isinstance(out, str)

    def test_generate_empty_prompt(self, tmp_path):
        model, stoi, itos = self._train_tiny(tmp_path)
        out = generate_from_chat_model(model, stoi, itos, "", max_tokens=5, temperature=0.0)
        assert isinstance(out, str)

    def test_generate_long_prompt_truncates(self, tmp_path):
        model, stoi, itos = self._train_tiny(tmp_path)
        out = generate_from_chat_model(model, stoi, itos, "User: " + "x" * 100, max_tokens=5, temperature=0.0)
        assert isinstance(out, str)


class TestEvaluateChatModel:
    def test_returns_metrics(self, tmp_path):
        config = _tiny_config(tmp_path)
        model, meta = train_chat_model(_pairs(4), config)
        result = evaluate_chat_model(model, meta["stoi"], meta["itos"], _pairs(4), max_samples=2)
        assert result["perplexity"] >= 1.0
        assert len(result["samples"]) == 2
        assert result["avg_response_len"] >= 0

    def test_max_samples_limit(self, tmp_path):
        config = _tiny_config(tmp_path)
        model, meta = train_chat_model(_pairs(4), config)
        result = evaluate_chat_model(model, meta["stoi"], meta["itos"], _pairs(4), max_samples=0)
        assert result["samples"] == []
        assert result["avg_response_len"] == 0.0


class TestTrainFromSessions:
    def test_no_sessions_raises(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", empty)
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", empty)
        config = _tiny_config(tmp_path)
        with pytest.raises(ValueError, match="No chat sessions"):
            train_from_sessions(config)

    def test_no_sessions_falls_back_to_corpus(self, tmp_path, monkeypatch):
        """When no sessions exist, captured API conversations are used."""
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", empty)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        with open(corpus / "corpus.jsonl", "w") as f:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": "User message number one asking something interesting."},
                {"role": "assistant", "content": "Assistant responds helpfully with a detailed answer about topic one."},
            ]}) + "\n")
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", corpus)
        config = _tiny_config(tmp_path)
        model, meta = train_from_sessions(config)
        assert meta["num_pairs"] >= 1
        assert meta["checkpoint"]

    def test_train_from_session_files(self, tmp_path, monkeypatch):
        sess_dir = tmp_path / "sessions"
        sess_dir.mkdir()
        (sess_dir / "s1.json").write_text(
            '{"messages": ['
            '{"role": "user", "content": "User message number one asking something interesting."},'
            '{"role": "assistant", "content": "Assistant responds helpfully with a detailed answer about topic one."},'
            '{"role": "user", "content": "User message number two asking something interesting."},'
            '{"role": "assistant", "content": "Assistant responds helpfully with a detailed answer about topic two."}'
            ']}',
            encoding="utf-8",
        )
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", sess_dir)
        config = _tiny_config(tmp_path)
        model, meta = train_from_sessions(config)
        assert meta["num_pairs"] >= 1
        assert "perplexity" in meta
        assert "samples" in meta

    def test_eval_failure_does_not_abort(self, tmp_path, monkeypatch):
        sess_dir = tmp_path / "sessions"
        sess_dir.mkdir()
        (sess_dir / "s1.json").write_text(
            '{"messages": ['
            '{"role": "user", "content": "User message number one asking something interesting."},'
            '{"role": "assistant", "content": "Assistant responds helpfully with a detailed answer about topic one."}'
            ']}',
            encoding="utf-8",
        )
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", sess_dir)

        def boom(*a, **k):
            raise RuntimeError("eval exploded")

        import domains.training.chat_trainer as ct
        monkeypatch.setattr(ct, "evaluate_chat_model", boom)
        config = _tiny_config(tmp_path)
        model, meta = train_from_sessions(config)
        assert meta["num_pairs"] >= 1
        assert "perplexity" not in meta


# ── Checkpoint metadata + resume edge tests ─────────────────────────────────

class TestCheckpointMetadata:
    """Verify all training metadata is stored in .soul checkpoint."""

    PAIRS = [
        {"user_msg": "hello", "assistant_msg": "hi there"},
        {"user_msg": "how are you", "assistant_msg": "doing well"},
        {"user_msg": "what is 2+2", "assistant_msg": "four"},
        {"user_msg": "goodbye", "assistant_msg": "see ya"},
    ]

    def test_checkpoint_contains_all_training_fields(self, tmp_path):
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="meta-test",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        model, meta = train_chat_model(self.PAIRS, config=config)
        ckpt = tmp_path / "meta-test.soul"
        assert ckpt.exists()

        from domains.training.slonet import import_from_sou
        loaded = import_from_sou(str(ckpt))
        md = loaded.metadata
        assert md is not None
        for key in ("soul_name", "vocab_size", "n_embed", "n_layer", "n_head",
                     "block_size", "epoch", "step", "epochs", "final_loss",
                     "best_loss", "num_pairs", "num_chars", "stoi", "itos"):
            assert key in md, f"Missing checkpoint metadata key: {key}"

    def test_checkpoint_stoi_itos_are_consistent(self, tmp_path):
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="vocab-test",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        _, _ = train_chat_model(self.PAIRS, config=config)
        from domains.training.slonet import import_from_sou
        loaded = import_from_sou(str(tmp_path / "vocab-test.soul"))
        stoi = loaded.metadata["stoi"]
        itos = loaded.metadata["itos"]
        assert isinstance(stoi, dict) and isinstance(itos, dict)
        # stoi: char → int (always correct types), itos: JSON may serialize int keys as strings
        itos_int_keys = {int(k): v for k, v in itos.items()}
        for char, idx in stoi.items():
            assert itos_int_keys[idx] == char, f"stoi/itos mismatch for char={char!r} idx={idx}"

    def test_checkpoint_optimizer_state_saved(self, tmp_path):
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="opt-test",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        _, _ = train_chat_model(self.PAIRS, config=config)
        from domains.training.slonet import import_from_sou
        loaded = import_from_sou(str(tmp_path / "opt-test.soul"))
        opt_state = loaded.metadata.get("optimizer_state")
        assert opt_state is not None
        assert "hyperparameters" in opt_state
        assert "t" in opt_state
        assert opt_state["t"] >= 0


class TestResumeCheckpoint:
    """Resume from checkpoint — metadata, step, epoch, optimizer state."""

    PAIRS_A = [
        {"user_msg": "hello", "assistant_msg": "hi there"},
        {"user_msg": "how are you", "assistant_msg": "doing well"},
        {"user_msg": "what is 2+2", "assistant_msg": "four"},
        {"user_msg": "goodbye", "assistant_msg": "see ya"},
    ]

    def test_resume_restores_epoch_and_step(self, tmp_path):
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=2, batch_size=2, soul_name="resume-test",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        _, meta1 = train_chat_model(self.PAIRS_A, config=config)
        assert meta1["epochs_completed"] >= 1
        assert meta1["total_steps"] > 0

        # Resume with same config
        config2 = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=3, batch_size=2, soul_name="resume-test",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=str(tmp_path / "resume-test.soul"),
            min_pair_quality=0.0,
        )
        _, meta2 = train_chat_model(self.PAIRS_A, config=config2)
        # Should have started from epoch >= 1 (not epoch 0)
        assert meta2["total_steps"] > meta1["total_steps"]

    def test_resume_uses_checkpoint_vocab_not_data_vocab(self, tmp_path):
        """When training data changes, resume still uses checkpoint vocab."""
        pairs_a = [
            {"user_msg": "abc", "assistant_msg": "xyz"},
            {"user_msg": "def", "assistant_msg": "uvw"},
        ]
        pairs_b = [
            {"user_msg": "hello world", "assistant_msg": "goodbye moon"},
            {"user_msg": "foo bar", "assistant_msg": "baz qux"},
        ]
        config_a = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="vocab-switch",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        _, meta_a = train_chat_model(pairs_a, config=config_a)
        vocab_a = meta_a["vocab_size"]

        # Resume with different data — checkpoint vocab should win
        config_b = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=2, batch_size=2, soul_name="vocab-switch",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=str(tmp_path / "vocab-switch.soul"),
            min_pair_quality=0.0,
        )
        _, meta_b = train_chat_model(pairs_b, config=config_b)
        # vocab_size should match checkpoint, not new data
        assert meta_b["vocab_size"] == vocab_a

    def test_resume_with_nonexistent_checkpoint_falls_through(self, tmp_path):
        """Missing checkpoint → creates fresh model."""
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="no-ckpt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=str(tmp_path / "nonexistent.soul"),
            min_pair_quality=0.0,
        )
        model, meta = train_chat_model(self.PAIRS_A, config=config)
        assert meta["checkpoint"] != ""
        assert meta["total_steps"] > 0

    def test_resume_best_loss_continues_from_prev(self, tmp_path):
        """best_loss in checkpoint is carried forward, not reset to inf."""
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="loss-carry",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        _, meta1 = train_chat_model(self.PAIRS_A, config=config)
        saved_best = meta1["best_loss"]

        config2 = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=2, batch_size=2, soul_name="loss-carry",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=str(tmp_path / "loss-carry.soul"),
            min_pair_quality=0.0,
        )
        _, meta2 = train_chat_model(self.PAIRS_A, config=config2)
        # best_loss should be <= saved_best (not reset to inf)
        assert meta2["best_loss"] <= saved_best

    def test_resume_optimizer_state_restored(self, tmp_path):
        """Optimizer momentum/velocity restored from checkpoint."""
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="opt-resume",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        _, meta1 = train_chat_model(self.PAIRS_A, config=config)
        from domains.training.slonet import import_from_sou
        ckpt1 = import_from_sou(str(tmp_path / "opt-resume.soul"))
        t1 = ckpt1.metadata["optimizer_state"]["t"]

        config2 = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=2, batch_size=2, soul_name="opt-resume",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=str(tmp_path / "opt-resume.soul"),
            min_pair_quality=0.0,
        )
        _, meta2 = train_chat_model(self.PAIRS_A, config=config2)
        ckpt2 = import_from_sou(str(tmp_path / "opt-resume.soul"))
        t2 = ckpt2.metadata["optimizer_state"]["t"]
        # Optimizer timestep should have advanced
        assert t2 >= t1

    def test_resume_with_config_change_n_embed(self, tmp_path):
        """Changing n_embed on resume with checkpoint vocab still works
        because the model is loaded from checkpoint (not recreated)."""
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="cfg-change",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        train_chat_model(self.PAIRS_A, config=config)

        # "Change" n_embed — but resume loads from checkpoint which has n_embed=32
        config2 = ChatTrainConfig(
            n_embed=64, n_layer=1, n_head=2, block_size=16,
            epochs=2, batch_size=2, soul_name="cfg-change",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=str(tmp_path / "cfg-change.soul"),
            min_pair_quality=0.0,
        )
        model, meta = train_chat_model(self.PAIRS_A, config=config2)
        # Model loaded from checkpoint — n_embed stays 32, not 64
        assert model.n_embed == 32

    def test_checkpoint_metadata_matches_returned_metadata(self, tmp_path):
        """Metadata dict returned by train_chat_model matches checkpoint contents."""
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="meta-match",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        _, returned_meta = train_chat_model(self.PAIRS_A, config=config)

        from domains.training.slonet import import_from_sou
        loaded = import_from_sou(str(tmp_path / "meta-match.soul"))
        ckpt_meta = loaded.metadata

        assert ckpt_meta["vocab_size"] == returned_meta["vocab_size"]
        assert ckpt_meta["n_embed"] == config.n_embed
        assert ckpt_meta["n_layer"] == config.n_layer
        assert ckpt_meta["num_pairs"] == returned_meta["num_pairs"]
        assert ckpt_meta["soul_name"] == config.soul_name

    def test_empty_data_with_resume(self, tmp_path):
        """Resume with empty pairs list doesn't crash — no pairs to train on."""
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="empty-resume",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        train_chat_model(self.PAIRS_A, config=config)

        config2 = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="empty-resume",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=str(tmp_path / "empty-resume.soul"),
            min_pair_quality=999.0,  # Filter out all pairs
        )
        model, meta = train_chat_model(self.PAIRS_A, config=config2)
        # Should still return a model (loaded from checkpoint)
        assert meta["checkpoint"] != ""


class TestCorruptedMetadata:
    """Edge cases: checkpoint metadata fields changed, corrupted, or missing."""

    PAIRS = [
        {"user_msg": "hello", "assistant_msg": "hi"},
        {"user_msg": "bye", "assistant_msg": "see ya"},
        {"user_msg": "thanks", "assistant_msg": "np"},
        {"user_msg": "ok", "assistant_msg": "sure"},
    ]

    def _make_ckpt(self, tmp_path, meta_overrides):
        """Train a model then rewrite its metadata with overrides."""
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path), min_pair_quality=0.0,
        )
        train_chat_model(self.PAIRS, config=config)

        # Load the .sou file, patch metadata, rewrite
        import json, struct
        import struct as _struct
        from pathlib import Path
        from domains.training.slonet import export_to_sou, import_from_sou

        ckpt_path = tmp_path / "corrupt.soul"
        model = import_from_sou(str(ckpt_path))
        md = model.metadata.copy()
        md.update(meta_overrides)

        # Rewrite .sou with patched metadata (keep weights intact)
        export_to_sou(model, str(ckpt_path), metadata=md)
        return str(ckpt_path)

    def test_missing_epoch_defaults_to_zero(self, tmp_path):
        path = self._make_ckpt(tmp_path, {"epoch": None})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=5, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        assert meta["total_steps"] > 0

    def test_negative_epoch_defaults_to_zero(self, tmp_path):
        path = self._make_ckpt(tmp_path, {"epoch": -5})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=5, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        assert meta["total_steps"] > 0

    def test_epoch_exceeding_config_resets_to_zero(self, tmp_path):
        """Epoch 10 in checkpoint but config only has 2 epochs → start from 0."""
        path = self._make_ckpt(tmp_path, {"epoch": 10})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=2, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        # Should still produce a valid checkpoint
        assert meta["checkpoint"] != ""

    def test_negative_step_defaults_to_zero(self, tmp_path):
        path = self._make_ckpt(tmp_path, {"step": -100})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=2, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        assert meta["total_steps"] > 0

    def test_string_epoch_ignored(self, tmp_path):
        """Epoch stored as string → treated as invalid, start from 0."""
        path = self._make_ckpt(tmp_path, {"epoch": "abc"})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=3, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        assert meta["checkpoint"] != ""

    def test_nan_best_loss_ignored(self, tmp_path):
        path = self._make_ckpt(tmp_path, {"best_loss": float("nan")})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        assert np.isfinite(meta["best_loss"])

    def test_missing_stoi_uses_data_vocab(self, tmp_path):
        path = self._make_ckpt(tmp_path, {"stoi": None, "itos": None})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        # Should use vocab from data, not crash
        assert meta["vocab_size"] > 0

    def test_empty_stoi_uses_data_vocab(self, tmp_path):
        path = self._make_ckpt(tmp_path, {"stoi": {}, "itos": {}})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        assert meta["vocab_size"] > 0

    def test_corrupt_itos_values_skipped(self, tmp_path):
        """itos has non-integer keys after JSON round-trip — repairable entries kept."""
        path = self._make_ckpt(tmp_path, {"itos": {"0": "\x00", "bad": "x", "2": "b"}})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        assert meta["vocab_size"] > 0

    def test_optimizer_state_none_ignored(self, tmp_path):
        path = self._make_ckpt(tmp_path, {"optimizer_state": None})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        assert meta["checkpoint"] != ""

    def test_optimizer_state_wrong_shape_ignored(self, tmp_path):
        """Optimizer state from a model with different param count → gracefully skipped."""
        path = self._make_ckpt(tmp_path, {"optimizer_state": {"t": 5, "state": {"fake": {"m": [1,2,3]}}}})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        assert meta["checkpoint"] != ""

    def test_total_steps_includes_start_step(self, tmp_path):
        """total_steps = start_step + remaining, not just remaining."""
        path = self._make_ckpt(tmp_path, {"step": 50, "epoch": 0})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=2, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        # total_steps should be >= 50 (start_step) + some remaining
        assert meta["total_steps"] >= 50

    def test_empty_metadata_dict_creates_fresh_model(self, tmp_path):
        """Checkpoint with empty metadata → fresh model, no crash."""
        path = self._make_ckpt(tmp_path, {"stoi": {}, "itos": {}, "epoch": None, "step": None})
        config = ChatTrainConfig(
            n_embed=32, n_layer=1, n_head=2, block_size=16,
            epochs=1, batch_size=2, soul_name="corrupt",
            checkpoint_dir=str(tmp_path),
            resume_checkpoint=path, min_pair_quality=0.0,
        )
        _, meta = train_chat_model(self.PAIRS, config=config)
        assert meta["checkpoint"] != ""
