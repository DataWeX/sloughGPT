"""Tests for training.hf_lora_finetune — HF LoRA finetune."""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.training._internal.hf_lora_finetune import (
    HFLoraConfig,
    HFLoraTrainer,
)

# NOTE: the original MODEL/DATA paths (models/hf-cache/..., data/user_adapters/...)
# are NOT git-tracked — they existed only on a dev box, so CI failed on
# Path.is_file(). Tests that exercise __post_init__'s value validation use the
# hermetic `realish_paths` fixture; MODEL stays for tests that never touch disk.
MODEL = "models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/model.slnc"
DATA = "data/user_adapters/user_adapters.journal.jsonl"


def _assert_valid(config: HFLoraConfig) -> None:
    assert Path(config.model_path).is_file()
    assert Path(config.data_path).is_file()


@pytest.fixture()
def realish_paths(tmp_path):
    """Hermetic stand-ins for the machine-local model/data fixtures."""
    model = tmp_path / "model.slnc"
    model.write_bytes(b"")
    data = tmp_path / "adapters.journal.jsonl"
    data.write_text("{}\n")
    return str(model), str(data)


# ── HFLoraConfig ────────────────────────────────────────────────────────────


class TestHFLoraConfig:
    def test_default(self, realish_paths):
        model, data = realish_paths
        config = HFLoraConfig(model_path=model, data_path=data)
        _assert_valid(config)
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.epochs == 3

    def test_custom(self, realish_paths):
        model, data = realish_paths
        config = HFLoraConfig(model_path=model, data_path=data, rank=4, alpha=8.0, epochs=10)
        _assert_valid(config)
        assert config.rank == 4
        assert config.alpha == 8.0
        assert config.epochs == 10

    def test_adapter_name(self):
        config = HFLoraConfig(model_path=MODEL, data_path=DATA)
        # model stem "model" + rank 8
        assert "model" in config.adapter_name
        assert "r8" in config.adapter_name

    def test_adapter_name_custom(self, realish_paths):
        model, data = realish_paths
        config = HFLoraConfig(model_path=model, data_path=data, adapter_name="custom")
        _assert_valid(config)
        assert config.adapter_name == "custom"

    def test_model_path_empty_accepted_at_init(self):
        # cb971c824 scope: init validates VALUE ranges (rank/alpha/epochs/...),
        # never path presence — 80+ tests legitimately construct configs with
        # empty paths. The empty path surfaces at CONSUMPTION instead
        # (load_model raises FileNotFoundError "Model not found").
        config = HFLoraConfig(model_path="", data_path=DATA)
        assert config.model_path == ""

    def test_model_file_must_exist(self):
        # Contract (cb971c824): init validates VALUES; file existence is
        # checked at CONSUMPTION — construction succeeds, load_model raises.
        config = HFLoraConfig(model_path="models/nonexistent.slnc", data_path=DATA)
        trainer = HFLoraTrainer(config)
        with pytest.raises(FileNotFoundError, match="Model not found"):
            trainer.load_model()

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
