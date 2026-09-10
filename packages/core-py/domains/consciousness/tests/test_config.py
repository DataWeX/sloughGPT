"""Tests for ConsciousnessConfig."""

import json
import pytest
from pathlib import Path
from domains.consciousness.config import ConsciousnessConfig


class TestConsciousnessConfig:
    def test_default_config(self):
        cfg = ConsciousnessConfig()
        assert cfg.level == 0
        assert cfg.max_tokens == 150
        assert cfg.training_enabled is False
        assert cfg.lora_rank == 4
        assert cfg.is_enabled() is False

    def test_enabled_at_level_1(self):
        cfg = ConsciousnessConfig(level=1)
        assert cfg.is_enabled() is True

    def test_validate_rejects_invalid_level(self):
        cfg = ConsciousnessConfig(level=5)
        with pytest.raises(ValueError, match="Level must be 0-3"):
            cfg.validate()

    def test_validate_rejects_low_max_tokens(self):
        cfg = ConsciousnessConfig(max_tokens=5)
        with pytest.raises(ValueError, match="max_tokens"):
            cfg.validate()

    def test_validate_rejects_zero_lora_rank(self):
        cfg = ConsciousnessConfig(lora_rank=0)
        with pytest.raises(ValueError, match="lora_rank"):
            cfg.validate()

    def test_validate_passes(self):
        cfg = ConsciousnessConfig(level=2, max_tokens=200, lora_rank=8)
        cfg.validate()  # should not raise

    def test_get_store_path(self):
        cfg = ConsciousnessConfig(store_path="/tmp/test")
        assert cfg.get_store_path().as_posix() == "/tmp/test"

    def test_save_and_load(self, tmp_path):
        cfg = ConsciousnessConfig(level=2, max_tokens=200, store_path=str(tmp_path))
        cfg.save()

        loaded = ConsciousnessConfig.load(store_path=str(tmp_path))
        assert loaded.level == 2
        assert loaded.max_tokens == 200
        assert loaded.lora_rank == 4  # default preserved

    def test_load_returns_defaults_when_no_file(self, tmp_path):
        loaded = ConsciousnessConfig.load(store_path=str(tmp_path))
        assert loaded.level == 0
        assert loaded.max_tokens == 150

    def test_load_returns_defaults_on_corrupt_file(self, tmp_path):
        config_file = tmp_path / "consciousness_config.json"
        config_file.write_text("not valid json {{{")
        loaded = ConsciousnessConfig.load(store_path=str(tmp_path))
        assert loaded.level == 0

    def test_save_persists_all_fields(self, tmp_path):
        cfg = ConsciousnessConfig(
            level=3, max_tokens=300, training_enabled=True,
            lora_rank=16, lora_alpha=32, store_path=str(tmp_path),
        )
        cfg.save()

        raw = json.loads((tmp_path / "consciousness_config.json").read_text())
        assert raw["level"] == 3
        assert raw["max_tokens"] == 300
        assert raw["training_enabled"] is True
        assert raw["lora_rank"] == 16
        assert raw["lora_alpha"] == 32
