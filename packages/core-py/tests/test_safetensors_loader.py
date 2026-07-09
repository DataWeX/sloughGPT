"""Tests for safetensors_loader — torch-free weight loading."""

import numpy as np
import pytest
from domains.infrastructure.safetensors_loader import (
    load_model_weights,
    load_model_config,
    list_cached_models,
    _get_model_dir,
    _find_safetensors,
)
from pathlib import Path


@pytest.fixture(scope="session")
def gpt2_weights():
    """Load GPT-2 weights once for entire test session."""
    return load_model_weights("gpt2")


@pytest.fixture(scope="session")
def gpt2_config():
    """Load GPT-2 config once for entire test session."""
    return load_model_config("gpt2")


class TestGetModelDir:
    """Path resolution tests."""

    def test_gpt2_path(self):
        d = _get_model_dir("gpt2")
        assert d.name == "models--gpt2"
        assert "huggingface" in str(d)

    def test_namespaced_model(self):
        d = _get_model_dir("Qwen/Qwen2-0.5B-Instruct")
        assert d.name == "models--Qwen--Qwen2-0.5B-Instruct"


class TestLoadModelConfig:
    """Config loading tests."""

    def test_gpt2_config(self, gpt2_config):
        assert gpt2_config["model_type"] == "gpt2"
        assert gpt2_config["vocab_size"] == 50257
        assert gpt2_config["n_layer"] == 12
        assert gpt2_config["n_head"] == 12
        assert gpt2_config["n_embd"] == 768

    def test_unknown_model_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model_config("nonexistent/model-xyz")


class TestLoadModelWeights:
    """Weight loading tests."""

    def test_gpt2_weights(self, gpt2_weights):
        assert isinstance(gpt2_weights, dict)
        assert len(gpt2_weights) > 0
        assert all(isinstance(v, np.ndarray) for v in gpt2_weights.values())

    def test_gpt2_has_embed(self, gpt2_weights):
        assert "wte.weight" in gpt2_weights
        assert gpt2_weights["wte.weight"].shape == (50257, 768)

    def test_gpt2_has_attn(self, gpt2_weights):
        attn_keys = [k for k in gpt2_weights if "attn" in k]
        assert len(attn_keys) > 0

    def test_unknown_model_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model_weights("nonexistent/model-xyz")

    def test_weights_are_float32(self, gpt2_weights):
        for k, v in list(gpt2_weights.items())[:5]:
            assert v.dtype == np.float32, f"{k} has dtype {v.dtype}"


class TestListCachedModels:
    """Cached model listing tests."""

    def test_returns_list(self):
        models = list_cached_models()
        assert isinstance(models, list)

    def test_gpt2_in_list(self):
        models = list_cached_models()
        ids = [m["id"] for m in models]
        assert "gpt2" in ids or "openai-community/gpt2" in ids

    def test_each_model_has_fields(self):
        models = list_cached_models()
        for m in models:
            assert "id" in m
            assert "path" in m
            assert "size_mb" in m
            assert m["size_mb"] > 0

    def test_sorted_by_id(self):
        models = list_cached_models()
        ids = [m["id"] for m in models]
        assert ids == sorted(ids)
