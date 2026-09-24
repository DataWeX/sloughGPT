"""Tests for training.hf_lora_finetune — HF LoRA finetune."""

from __future__ import annotations

# HFLoraConfig.__post_init__ requires existing model_path + data_path files.
import tempfile as _tempfile
from pathlib import Path as _Path

from domain.training._internal.hf_lora_finetune import (
    HFLoraConfig,
    HFLoraTrainer,
)

_CFG_DIR = _Path(_tempfile.mkdtemp(prefix="hf_lora_cfg_"))


def _realize(path_str: str) -> str:
    """Return an existing file path, preserving the original stem under a temp dir."""
    p = _Path(path_str)
    if p.is_file():
        return str(p.resolve())
    dest = _CFG_DIR / p.name
    if not dest.is_file():
        dest.write_bytes(b"x")
    return str(dest)


def _cfg(model_path: str = "model.slnc", data_path: str = "data.txt", **kwargs) -> HFLoraConfig:
    """HFLoraConfig with required real files; overrides keep their filename/stem."""
    kwargs["model_path"] = _realize(model_path)
    kwargs["data_path"] = _realize(data_path)
    return HFLoraConfig(**kwargs)



# ── HFLoraConfig ────────────────────────────────────────────────────────────


class TestHFLoraConfig:
    def test_default(self):
        config = _cfg()
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.epochs == 3

    def test_custom(self):
        config = _cfg(rank=4, alpha=8.0, epochs=10)
        assert config.rank == 4
        assert config.alpha == 8.0
        assert config.epochs == 10

    def test_adapter_name(self):
        config = _cfg(model_path="models/gpt2.slnc")
        assert "gpt2" in config.adapter_name
        assert "r8" in config.adapter_name

    def test_adapter_name_custom(self):
        config = _cfg(model_path="models/test.slnc", adapter_name="custom")
        assert config.adapter_name == "custom"


# ── HFLoraTrainer ──────────────────────────────────────────────────────────


class TestHFLoraTrainer:
    def test_init(self):
        config = _cfg()
        trainer = HFLoraTrainer(config)
        assert trainer.model is None
        assert trainer.config.rank == 8
