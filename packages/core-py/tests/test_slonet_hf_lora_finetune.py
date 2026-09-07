"""Tests for training.hf_lora_finetune — HF LoRA finetune."""

from __future__ import annotations

import pytest
from domains.training.hf_lora_finetune import (
    HFLoraConfig,
    HFLoraTrainer,
)


# ── HFLoraConfig ────────────────────────────────────────────────────────────


class TestHFLoraConfig:

    def test_default(self):
        config = HFLoraConfig()
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.epochs == 3

    def test_custom(self):
        config = HFLoraConfig(rank=4, alpha=8.0, epochs=10)
        assert config.rank == 4
        assert config.alpha == 8.0
        assert config.epochs == 10

    def test_adapter_name(self):
        config = HFLoraConfig(model_path="models/gpt2.slnc")
        assert "gpt2" in config.adapter_name
        assert "r8" in config.adapter_name

    def test_adapter_name_custom(self):
        config = HFLoraConfig(model_path="models/test.slnc", adapter_name="custom")
        assert config.adapter_name == "custom"


# ── HFLoraTrainer ──────────────────────────────────────────────────────────


class TestHFLoraTrainer:

    def test_init(self):
        config = HFLoraConfig()
        trainer = HFLoraTrainer(config)
        assert trainer.model is None
        assert trainer.config.rank == 8
