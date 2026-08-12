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
