"""Tests for CLI train commands."""
import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add cli src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDistillConfig:
    def test_config_defaults(self):
        from domains.training.distill_gpt2 import DistillConfig
        c = DistillConfig()
        assert c.n_embed == 128
        assert c.n_layer == 4
        assert c.n_head == 4
        assert c.block_size == 128
        assert c.epochs == 10
        assert c.lr == 3e-4
        assert c.temperature == 4.0
        assert c.resume_checkpoint is None

    def test_config_resume(self):
        from domains.training.distill_gpt2 import DistillConfig
        c = DistillConfig(resume_checkpoint="test.soul", resume_epoch=5)
        assert c.resume_checkpoint == "test.soul"
        assert c.resume_epoch == 5

    def test_config_custom_values(self):
        from domains.training.distill_gpt2 import DistillConfig
        c = DistillConfig(n_embed=64, n_layer=2, n_head=2, epochs=3, lr=1e-3)
        assert c.n_embed == 64
        assert c.n_layer == 2
        assert c.epochs == 3
        assert c.lr == 1e-3

    def test_config_temperature(self):
        from domains.training.distill_gpt2 import DistillConfig
        c = DistillConfig(temperature=2.0)
        assert c.temperature == 2.0


class TestCmdDistill:
    def test_missing_text_returns_early(self):
        from commands.train import cmd_distill
        args = MagicMock()
        args.text_source = None
        args.file = None
        args.api = False
        cmd_distill(args)

    def test_file_not_found_returns_early(self):
        from commands.train import cmd_distill
        args = MagicMock()
        args.text_source = None
        args.file = "/nonexistent/file.txt"
        args.api = False
        cmd_distill(args)

    def test_empty_text_returns_early(self):
        from commands.train import cmd_distill
        args = MagicMock()
        args.text_source = ""
        args.file = None
        args.api = False
        cmd_distill(args)


class TestCmdTrainNative:
    def _args(self, **overrides):
        base = dict(
            dataset="datasets/tinyshakespeare/input.txt",
            steps=3,
            embed=32,
            layers=1,
            heads=2,
            block=64,
            batch=4,
            epochs=1,
            lr=3e-3,
            weight_decay=0.01,
            scheduler="cosine",
            warmup=2,
            min_lr=1e-5,
            grad_norm=1.0,
            dropout=0.1,
            checkpoint_dir="/tmp/opencode/cli-native-test",
            checkpoint_interval=50,
            max_checkpoints=2,
            save_best_only=False,
            eval_interval=50,
            log_interval=50,
            soul_name="test-native",
            save_stem=None,
            save_format="sou",
            resume=None,
            resume_latest=False,
            device="cpu",
            host="localhost",
            port=8000,
        )
        base.update(overrides)
        return MagicMock(**base)

    def test_missing_dataset_exits(self):
        from commands.train import cmd_train_native
        with pytest.raises(SystemExit) as exc:
            cmd_train_native(self._args(dataset=None))
        assert exc.value.code == 2

    def test_resume_and_resume_latest_conflict(self):
        from commands.train import cmd_train_native
        args = self._args(resume="/x.soul", resume_latest=True)
        with patch("domains.training.train_pipeline.SloughGPTTrainer") as mock_trainer:
            mock_trainer.return_value.training_model.num_parameters.return_value = 1000
            with pytest.raises(SystemExit) as exc:
                cmd_train_native(args)
        assert exc.value.code == 2

    def test_bad_save_format_warns_and_uses_sou(self):
        from commands.train import cmd_train_native
        args = self._args(save_format="pt")
        with patch("domains.training.train_pipeline.SloughGPTTrainer") as mock_trainer:
            instance = mock_trainer.return_value
            instance.training_model.num_parameters.return_value = 1000
            cmd_train_native(args)
        assert instance.save.called
        save_args = instance.save.call_args
        assert save_args.kwargs.get("format") == "sou"

    def test_full_native_train_pipeline(self):
        from commands.train import cmd_train_native
        with patch("domains.training.train_pipeline.SloughGPTTrainer") as mock_trainer:
            instance = mock_trainer.return_value
            instance.training_model.num_parameters.return_value = 1000
            cmd_train_native(self._args())
            assert instance.train.called
            assert instance.save.called
            assert instance.save.call_args[1]["format"] == "sou"

    def test_save_stem_overrides_soul_name(self):
        from commands.train import cmd_train_native
        args = self._args(save_stem="my_model")
        with patch("domains.training.train_pipeline.SloughGPTTrainer") as mock_trainer:
            instance = mock_trainer.return_value
            instance.training_model.num_parameters.return_value = 1000
            cmd_train_native(args)
        assert instance.save.called
        save_path = instance.save.call_args[0][0]
        assert save_path.endswith("/my_model")
        assert "/sloughgpt-native" not in save_path

    def test_completed_run_leaves_single_model_file(self, tmp_path):
        from commands.train import cmd_train_native
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        for name in ("tinyshakespeare_1.soul", "tinyshakespeare_1.soul.meta.json",
                     "tinyshakespeare_2.soul", "tinyshakespeare_2.soul.meta.json",
                     "test-native.soul", "test-native.soul.meta.json"):
            (ckpt_dir / name).write_text("x", encoding="utf-8")
        args = self._args(checkpoint_dir=str(ckpt_dir))
        with patch("domains.training.train_pipeline.SloughGPTTrainer") as mock_trainer:
            instance = mock_trainer.return_value
            instance.training_model.num_parameters.return_value = 1000
            cmd_train_native(args)
        remaining = sorted(p.name for p in ckpt_dir.iterdir())
        assert remaining == ["test-native.soul", "test-native.soul.meta.json"]


class _FakeEmbedder:
    """Deterministic one-hot embedder: identical texts get identical vectors."""

    def __init__(self, dim=16):
        self.dim = dim
        self._idx = {}

    def _vec(self, text):
        if text not in self._idx:
            self._idx[text] = len(self._idx)
        v = [0.0] * self.dim
        v[self._idx[text] % self.dim] = 1.0
        return v

    def embed(self, text):
        return self._vec(text)

    def embed_batch(self, texts):
        return [self._vec(t) for t in texts]


class TestSplitCorpusText:
    def test_paragraphs(self):
        from commands.train import _split_corpus_text
        text = "first paragraph has enough words.\n\nsecond paragraph has words too.\n\nthird one has plenty of words as well."
        chunks = _split_corpus_text(text, min_len=20)
        assert len(chunks) >= 3

    def test_single_block_windows(self):
        from commands.train import _split_corpus_text
        blob = "word " * 500
        chunks = _split_corpus_text(blob, min_len=40)
        assert len(chunks) >= 2
        assert all(len(c) >= 40 for c in chunks)

    def test_word_fallback_tiny(self):
        from commands.train import _split_corpus_text
        text = "short text that is not enough for paragraphs at all."
        chunks = _split_corpus_text(text, min_len=20)
        assert len(chunks) >= 1


class TestCmdTrainEmbed:
    def _args(self, tmp_path, **overrides):
        base = dict(
            corpus=None,
            epochs=2,
            lr=3e-4,
            batch_size=8,
            embed_dim=64,
            vocab_size=256,
            output=None,
            test=None,
        )
        base.update(overrides)
        return MagicMock(**base)

    def test_single_file_corpus_is_chunked(self, tmp_path):
        import numpy as np
        from commands.train import cmd_train_embed
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            "neural networks learn from examples by adjusting weights.\n\n"
            "convolutional layers scan images with small filters.\n\n"
            "recurrent networks keep hidden state across time steps.\n\n"
            "transformer attention relates every token to every other token.\n",
            encoding="utf-8",
        )
        args = self._args(tmp_path, corpus=str(corpus))
        fake = _FakeEmbedder()
        with patch("domains.inference.slo_embedder.train_embedder") as mock_train, \
             patch("domains.inference.slo_embedder.SloTextEmbedder.load", return_value=fake) as mock_load:
            mock_train.return_value = {
                "save_path": "/tmp/x.sou", "final_loss": 0.5, "vocab_size": 10, "n_params": 100,
            }
            cmd_train_embed(args)
        assert mock_train.called
        texts = mock_train.call_args[1]["texts"]
        assert len(texts) >= 2
        assert mock_load.called

    def test_post_train_runs_retrieval_check(self, tmp_path):
        from commands.train import cmd_train_embed
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            "neural networks learn from examples by adjusting weights.\n\n"
            "convolutional layers scan images with small filters.\n\n"
            "recurrent networks keep hidden state across time steps.\n\n"
            "transformer attention relates every token to every other token.\n",
            encoding="utf-8",
        )
        args = self._args(tmp_path, corpus=str(corpus))
        fake = _FakeEmbedder()
        with patch("domains.inference.slo_embedder.train_embedder") as mock_train, \
             patch("domains.inference.slo_embedder.SloTextEmbedder.load", return_value=fake) as mock_load, \
             patch("commands.train._embedder_retrieval_check") as mock_check:
            mock_train.return_value = {
                "save_path": "/tmp/x.sou", "final_loss": 0.5, "vocab_size": 10, "n_params": 100,
            }
            cmd_train_embed(args)
        assert mock_check.called
        assert mock_check.call_args[0][0] is fake
        assert mock_load.call_args[0][0] == "/tmp/x.sou"

    def test_test_mode_retrieves_top_matches(self, tmp_path):
        from commands.train import cmd_train_embed
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            "neural networks learn from examples by adjusting weights.\n\n"
            "recurrent networks keep hidden state across time steps.\n\n"
            "transformer attention relates every token to every other token.\n",
            encoding="utf-8",
        )
        args = self._args(tmp_path, corpus=str(corpus), test="recurrent networks")
        fake = _FakeEmbedder()
        with patch("domains.inference.slo_embedder.SloTextEmbedder.load", return_value=fake) as mock_load, \
             patch("commands.train._embedder_retrieval_check") as mock_check:
            cmd_train_embed(args)
        assert mock_load.called
        assert mock_check.called
        assert mock_check.call_args[1]["query"] == "recurrent networks"

    def test_retrieval_check_self_rank(self):
        from commands.train import _embedder_retrieval_check
        texts = ["alpha bravo charlie delta", "echo foxtrot golf hotel", "india juliet kilo lima"]
        _embedder_retrieval_check(_FakeEmbedder(), texts)
