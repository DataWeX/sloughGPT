"""Tests for training.hf_lora_finetune — HF LoRA finetune."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from domain.training._internal.hf_lora_finetune import (
    HFLoraConfig,
    HFLoraTrainer,
)

# Repo-root-relative fixtures so the tests run from any cwd/worktree.
# HFLoraConfig is a pure spec (cb971c824): construction never touches disk,
# file-existence checks live at the consumption points (load_model,
# _prepare_data) — covered below. MODEL is the gitignored 2.5 GB cache model
# and is only used as a path value, never asserted to exist.
REPO = Path(__file__).resolve().parents[3]
MODEL = str(REPO / "models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/model.slnc")
DATA = str(REPO / "data/user_adapters/user_adapters.journal.jsonl")


# ── HFLoraConfig ────────────────────────────────────────────────────────────


class TestHFLoraConfig:
    def test_default(self):
        config = HFLoraConfig(model_path=MODEL, data_path=DATA)
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.epochs == 3

    def test_custom(self):
        config = HFLoraConfig(model_path=MODEL, data_path=DATA, rank=4, alpha=8.0, epochs=10)
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
        assert config.adapter_name == "custom"

    def test_missing_model_raises_at_load(self):
        # Pure spec: empty/missing paths are legal at init; the error surfaces
        # at the consumption point instead (cb971c824).
        trainer = HFLoraTrainer(HFLoraConfig(model_path="models/nonexistent.slnc"))
        with pytest.raises(FileNotFoundError, match="Model not found"):
            trainer.load_model()

    def test_missing_data_raises_at_prepare(self):
        trainer = HFLoraTrainer(HFLoraConfig(model_path=MODEL, data_path="missing.jsonl"))
        with pytest.raises(FileNotFoundError, match="Data not found"):
            trainer._prepare_data()

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

    def test_load_model_wires_tokenizer(self, monkeypatch, tmp_path):
        """load_model must hand the provider's tokenizer to the model.

        Without this, _prepare_data sees model._tokenizer is None and silently
        trains on char-level ids — a different tokenization than serving.
        """
        slnc = tmp_path / "m.slnc"
        slnc.write_bytes(b"stub")
        stub_model = SimpleNamespace(vocab_size=10, n_embed=4, n_layer=1)
        stub_tokenizer = SimpleNamespace()
        stub_provider = SimpleNamespace(_model=stub_model, _tokenizer=stub_tokenizer)

        from domain.inference._internal import slonet_provider

        monkeypatch.setattr(
            slonet_provider.SloNetChatProvider,
            "from_slnc",
            lambda *args, **kwargs: stub_provider,
        )
        trainer = HFLoraTrainer(HFLoraConfig(model_path=str(slnc)))
        model = trainer.load_model()
        assert model._tokenizer is stub_tokenizer
