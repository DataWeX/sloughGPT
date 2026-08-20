"""Tests for domains.training.huggingface.model_map — model registry and lookup."""

import pytest
from domains.training.huggingface.model_map import (
    ModelSize, get_model_info, search_models, get_recommended_quantization,
    get_model_requirements, map_to_sloughgpt_config, HF_MODELS,
)


class TestModelSize:
    def test_small_value(self):
        assert ModelSize.SMALL.value == "small"

    def test_all_members(self):
        assert len(ModelSize) == 4


class TestGetModelInfo:
    def test_existing_model(self):
        info = get_model_info("gpt2")
        assert info is not None
        assert info.model_id == "gpt2"

    def test_nonexistent_model(self):
        info = get_model_info("nonexistent/model")
        assert info is None


class TestSearchModels:
    def test_all_models(self):
        results = search_models()
        assert len(results) > 0

    def test_filter_by_org(self):
        results = search_models(organization="openai")
        assert len(results) >= 1
        for r in results:
            assert r.organization == "openai"

    def test_filter_by_size(self):
        results = search_models(size=ModelSize.SMALL)
        assert len(results) >= 1


class TestGetRecommendedQuantization:
    def test_small_model(self):
        q = get_recommended_quantization("gpt2")
        assert isinstance(q, str)

    def test_unknown_model(self):
        q = get_recommended_quantization("nonexistent/model")
        assert isinstance(q, str)


class TestGetModelRequirements:
    def test_known_model(self):
        req = get_model_requirements("gpt2")
        assert "memory_gb" in req
        assert req["memory_gb"] > 0

    def test_unknown_model_returns_defaults(self):
        req = get_model_requirements("nonexistent/model")
        assert "memory_gb" in req


class TestMapToSloughgptConfig:
    def test_known_model(self):
        config = map_to_sloughgpt_config("gpt2")
        assert "n_embed" in config or "embed_dim" in config

    def test_unknown_model(self):
        config = map_to_sloughgpt_config("nonexistent/model")
        assert isinstance(config, dict)


class TestHFModels:
    def test_is_dict(self):
        assert isinstance(HF_MODELS, dict)

    def test_has_gpt2(self):
        assert "gpt2" in HF_MODELS
