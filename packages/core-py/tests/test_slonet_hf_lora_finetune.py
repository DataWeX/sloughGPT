"""Tests for training.hf_lora_finetune — HF LoRA finetune."""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.training._internal.hf_lora_finetune import (
    HFLoraConfig,
    HFLoraTrainer,
)

# Real fixtures tracked in the repo: a model and a data file, so path
# validation in HFLoraConfig.__post_init__ can be exercised honestly.
MODEL = "models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/model.slnc"
DATA = "data/user_adapters/user_adapters.journal.jsonl"


def _assert_valid(config: HFLoraConfig) -> None:
    assert Path(config.model_path).is_file()
    assert Path(config.data_path).is_file()


# ── HFLoraConfig ────────────────────────────────────────────────────────────


class TestHFLoraConfig:
    def test_default(self):
        config = HFLoraConfig(model_path=MODEL, data_path=DATA)
        _assert_valid(config)
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.epochs == 3

    def test_custom(self):
        config = HFLoraConfig(model_path=MODEL, data_path=DATA, rank=4, alpha=8.0, epochs=10)
        _assert_valid(config)
        assert config.rank == 4
        assert config.alpha == 8.0
        assert config.epochs == 10

    def test_adapter_name(self):
        config = HFLoraConfig(model_path=MODEL, data_path=DATA)
        # model stem "model" + rank 8
        assert "model" in config.adapter_name
        assert "r8" in config.adapter_name

    def test_adapter_name_custom(self):
        config = HFLoraConfig(model_path=MODEL, data_path=DATA, adapter_name="custom")
        _assert_valid(config)
        assert config.adapter_name == "custom"

    def test_model_path_required(self):
        with pytest.raises(ValueError, match="model_path is required"):
            HFLoraConfig(model_path="", data_path=DATA)

    def test_model_file_must_exist(self):
        with pytest.raises(ValueError, match="Model file not found"):
            HFLoraConfig(model_path="models/nonexistent.slnc", data_path=DATA)

    def test_rank_must_be_positive(self):
        with pytest.raises(ValueError, match="rank must be >= 1"):
            HFLoraConfig(model_path=MODEL, data_path=DATA, rank=0)


# ── HFLoraTrainer ──────────────────────────────────────────────────────────


class TestHFLoraTrainer:
    def test_init(self):
        config = HFLoraConfig(model_path=MODEL, data_path=DATA)
        trainer = HFLoraTrainer(config)
        assert trainer.model is None
        assert trainer.config.rank == 8
