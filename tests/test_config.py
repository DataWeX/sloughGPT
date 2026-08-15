"""
SloughGPT Config Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_cli_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "apps", "cli", "src")
if os.path.isdir(_cli_src) and _cli_src not in sys.path:
    sys.path.insert(0, _cli_src)

from types import SimpleNamespace

import pytest
import tempfile
import yaml


class TestConfigLoader:
    """Tests for configuration loader."""

    def test_load_default_config(self):
        """Test loading default config."""
        from config_loader import load_config, Config

        config = load_config("nonexistent.yaml")

        assert isinstance(config, Config)
        assert config.model.n_embed == 256
        assert config.training.epochs == 10
        assert config.lora.enabled is False

    def test_load_yaml_config(self):
        """Test loading from YAML file."""
        from config_loader import load_config

        config_data = {
            'model': {'n_embed': 128, 'n_layer': 4},
            'training': {'epochs': 5, 'batch_size': 32},
            'lora': {'enabled': True, 'rank': 4},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = load_config(config_path)

            assert config.model.n_embed == 128
            assert config.model.n_layer == 4
            assert config.training.epochs == 5
            assert config.training.batch_size == 32
            assert config.lora.enabled is True
            assert config.lora.rank == 4
        finally:
            os.unlink(config_path)

    def test_model_config(self):
        """Test model config defaults."""
        from config_loader import ModelConfig

        cfg = ModelConfig()

        assert cfg.name == "sloughgpt"
        assert cfg.soul_name is None
        assert cfg.n_embed == 256
        assert cfg.n_layer == 6
        assert cfg.n_head == 8
        assert cfg.block_size == 128
        assert cfg.dropout == 0.1

    def test_training_config(self):
        """Test training config defaults."""
        from config_loader import TrainingConfig

        cfg = TrainingConfig()

        assert cfg.epochs == 10
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 1e-3
        assert cfg.weight_decay == 0.01
        assert cfg.scheduler == "cosine"
        assert cfg.log_interval == 10
        assert cfg.eval_interval == 100
        assert cfg.max_steps is None
        assert cfg.gradient_accumulation_steps == 1
        assert cfg.min_lr == 1e-5
        assert cfg.gradient_clip == 1.0
        assert cfg.use_mixed_precision is False
        assert cfg.mixed_precision_dtype == "bf16"

    def test_training_config_mixed_precision_deprecation_warns(self):
        from config_loader import TrainingConfig

        with pytest.warns(DeprecationWarning):
            TrainingConfig(use_mixed_precision=True)

    def test_lora_config(self):
        """Test LoRA config defaults."""
        from config_loader import LoRAConfig

        cfg = LoRAConfig()

        assert cfg.enabled is False
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert "c_attn" in cfg.target_modules

    def test_tracking_config(self):
        """Test tracking config defaults."""
        from config_loader import TrackingConfig

        cfg = TrackingConfig()

        assert cfg.enabled is False
        assert cfg.backend == "wandb"
        assert cfg.project == "sloughgpt"
        assert cfg.log_every == 10

    def test_merge_preserves_max_steps_from_training_config(self):
        from config_loader import Config, TrainingConfig, merge_args_with_config

        cfg = Config(training=TrainingConfig(max_steps=500))
        args = SimpleNamespace(
            dataset=None,
            epochs=None,
            batch_size=None,
            lr=None,
            use_lora=False,
            resume=None,
            max_steps=None,
            log_interval=None,
            eval_interval=None,
        )
        merge_args_with_config(cfg, args)
        assert cfg.training.max_steps == 500

    def test_merge_max_steps_cli_overrides_yaml(self):
        from config_loader import Config, TrainingConfig, merge_args_with_config

        cfg = Config(training=TrainingConfig(max_steps=500))
        args = SimpleNamespace(
            dataset=None,
            epochs=None,
            batch_size=None,
            lr=None,
            use_lora=False,
            resume=None,
            max_steps=42,
            log_interval=None,
            eval_interval=None,
        )
        merge_args_with_config(cfg, args)
        assert cfg.training.max_steps == 42

    def test_merge_scheduler_warmup_weight_decay_from_args(self):
        from config_loader import Config, TrainingConfig, merge_args_with_config

        cfg = Config(
            training=TrainingConfig(
                scheduler="constant",
                warmup_steps=999,
                weight_decay=0.2,
            )
        )
        args = SimpleNamespace(
            dataset=None,
            epochs=None,
            batch_size=None,
            lr=None,
            scheduler="polynomial",
            warmup_steps=50,
            weight_decay=0.05,
            use_lora=False,
            resume=None,
            max_steps=None,
            log_interval=None,
            eval_interval=None,
        )
        merge_args_with_config(cfg, args)
        assert cfg.training.scheduler == "polynomial"
        assert cfg.training.warmup_steps == 50
        assert cfg.training.weight_decay == 0.05

    def test_merge_warmup_steps_zero_updates_config(self):
        from config_loader import Config, TrainingConfig, merge_args_with_config

        cfg = Config(training=TrainingConfig(warmup_steps=500))
        args = SimpleNamespace(
            dataset=None,
            epochs=None,
            batch_size=None,
            lr=None,
            scheduler=None,
            warmup_steps=0,
            weight_decay=None,
            use_lora=False,
            resume=None,
            max_steps=None,
            log_interval=None,
            eval_interval=None,
        )
        merge_args_with_config(cfg, args)
        assert cfg.training.warmup_steps == 0

    def test_merge_min_lr_grad_clip_grad_accum_from_args(self):
        from config_loader import Config, TrainingConfig, merge_args_with_config

        cfg = Config(
            training=TrainingConfig(
                min_lr=1e-3,
                gradient_clip=2.0,
                gradient_accumulation_steps=4,
            )
        )
        args = SimpleNamespace(
            dataset=None,
            epochs=None,
            batch_size=None,
            lr=None,
            scheduler=None,
            warmup_steps=None,
            weight_decay=None,
            min_lr=1e-6,
            max_grad_norm=0.5,
            gradient_accumulation_steps=2,
            use_lora=False,
            resume=None,
            max_steps=None,
            log_interval=None,
            eval_interval=None,
        )
        merge_args_with_config(cfg, args)
        assert cfg.training.min_lr == 1e-6
        assert cfg.training.gradient_clip == 0.5
        assert cfg.training.gradient_accumulation_steps == 2

    def test_merge_lora_rank_alpha_from_args(self):
        from config_loader import Config, LoRAConfig, merge_args_with_config

        cfg = Config(lora=LoRAConfig(rank=32, alpha=64))
        args = SimpleNamespace(
            dataset=None,
            epochs=None,
            batch_size=None,
            lr=None,
            scheduler=None,
            warmup_steps=None,
            weight_decay=None,
            min_lr=None,
            max_grad_norm=None,
            gradient_accumulation_steps=None,
            lora_rank=16,
            lora_alpha=32,
            use_lora=False,
            resume=None,
            max_steps=None,
            log_interval=None,
            eval_interval=None,
        )
        merge_args_with_config(cfg, args)
        assert cfg.lora.rank == 16
        assert cfg.lora.alpha == 32

    def test_checkpoint_config_defaults(self):
        from config_loader import CheckpointConfig

        c = CheckpointConfig()
        assert c.trainer_dir == "checkpoints"
        assert c.trainer_interval == 1000
        assert c.save_best_only is False
        assert c.max_checkpoints == 5
        assert c.export_format == "soul"

    def test_merge_trainer_checkpoint_from_args(self):
        from config_loader import CheckpointConfig, Config, merge_args_with_config

        cfg = Config(
            checkpoint=CheckpointConfig(
                trainer_dir="old_dir",
                trainer_interval=50,
                save_best_only=False,
                max_checkpoints=2,
            )
        )
        args = SimpleNamespace(
            checkpoint_dir="custom_ckpt",
            checkpoint_interval=125,
            save_best_only=True,
            max_checkpoints=11,
        )
        merge_args_with_config(cfg, args)
        assert cfg.checkpoint.trainer_dir == "custom_ckpt"
        assert cfg.checkpoint.trainer_interval == 125
        assert cfg.checkpoint.save_best_only is True
        assert cfg.checkpoint.max_checkpoints == 11

    def test_merge_soul_name_from_args(self):
        from config_loader import Config, ModelConfig, merge_args_with_config

        cfg = Config(model=ModelConfig(name="m1", soul_name="orig"))
        args = SimpleNamespace(soul_name="cli soul")
        merge_args_with_config(cfg, args)
        assert cfg.model.soul_name == "cli soul"

    def test_merge_save_format_is_ignored(self):
        from config_loader import Config, ModelConfig, CheckpointConfig, merge_args_with_config

        cfg = Config(
            model=ModelConfig(name="m1", soul_name="orig"),
            checkpoint=CheckpointConfig(),
        )
        args = SimpleNamespace(soul_name=None, save_format="safetensors")
        merge_args_with_config(cfg, args)
        assert cfg.checkpoint.export_format == "soul"

    def test_checkpoint_export_format_deprecation_warns(self):
        from config_loader import CheckpointConfig

        with pytest.warns(DeprecationWarning):
            CheckpointConfig(export_format="sou")

    def test_merge_dropout_from_args(self):
        from config_loader import Config, ModelConfig, merge_args_with_config

        cfg = Config(model=ModelConfig(dropout=0.05))
        args = SimpleNamespace(dropout=0.25)
        merge_args_with_config(cfg, args)
        assert cfg.model.dropout == 0.25

    def test_merge_train_device_from_args(self):
        from config_loader import Config, DeviceConfig, merge_args_with_config

        cfg = Config(device=DeviceConfig(type="auto"))
        args = SimpleNamespace(train_device="cpu")
        merge_args_with_config(cfg, args)
        assert cfg.device.type == "cpu"


class TestGetDevice:
    """Sanity checks for ``config_loader.get_device`` (see ``cli.py train`` local path)."""

    def test_get_device_respects_explicit_type(self):
        from config_loader import DeviceConfig, get_device

        assert get_device(DeviceConfig(type="cpu")) == "cpu"
        assert get_device(DeviceConfig(type="cuda")) == "cuda"
        assert get_device(DeviceConfig(type="mps")) == "mps"
