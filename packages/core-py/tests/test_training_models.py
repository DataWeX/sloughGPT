"""Tests for domains.training — ModelManager, ModelConfig, ModelType, ModelArchitecture."""

import pytest
from domains.training import (
    ModelManager, ModelConfig, ModelType, ModelArchitecture,
)


class TestModelType:
    def test_all_members(self):
        assert len(ModelType) == 2
    def test_values(self):
        assert ModelType.LANGUAGE_MODEL.value == "language_model"
        assert ModelType.CHAT_MODEL.value == "chat_model"


class TestModelArchitecture:
    def test_all_members(self):
        assert len(ModelArchitecture) == 3


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(
            name="gpt2",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture=ModelArchitecture.GPT,
        )
        assert cfg.hidden_size == 768
        assert cfg.num_layers == 12

    def test_custom(self):
        cfg = ModelConfig(
            name="bert",
            model_type=ModelType.CHAT_MODEL,
            architecture=ModelArchitecture.BERT,
            hidden_size=512,
            num_layers=6,
        )
        assert cfg.hidden_size == 512


class TestModelManager:
    def test_register_and_create(self):
        mgr = ModelManager()
        cfg = ModelConfig(
            name="gpt2",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture=ModelArchitecture.GPT,
        )
        mgr.register_model(cfg)
        result = mgr.create_model("gpt2")
        assert result["name"] == "gpt2"
        assert result["ready"] is True

    def test_create_not_found(self):
        mgr = ModelManager()
        with pytest.raises(ValueError, match="not found"):
            mgr.create_model("nonexistent")
