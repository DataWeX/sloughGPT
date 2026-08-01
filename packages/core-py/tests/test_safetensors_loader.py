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

QWEN2_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _is_cached(model_id: str) -> bool:
    return _find_safetensors(_get_model_dir(model_id)) is not None


@pytest.fixture(scope="session")
def qwen_weights():
    """Load Qwen weights once for entire test session."""
    return load_model_weights(QWEN2_ID)


@pytest.fixture(scope="session")
def qwen_config():
    """Load Qwen config once for entire test session."""
    return load_model_config(QWEN2_ID)


class TestGetModelDir:
    """Path resolution tests."""

    def test_namespaced_model(self):
        d = _get_model_dir("Qwen/Qwen2.5-0.5B-Instruct")
        assert d.name == "models--Qwen--Qwen2.5-0.5B-Instruct"

    def test_namespaced_model_unrelated(self):
        d = _get_model_dir("Qwen/Qwen2-0.5B-Instruct")
        assert d.name == "models--Qwen--Qwen2-0.5B-Instruct"


class TestLoadModelConfig:
    """Config loading tests."""

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_qwen_config(self, qwen_config):
        assert qwen_config["model_type"] == "qwen2"
        assert qwen_config["vocab_size"] == 151936
        assert qwen_config["num_hidden_layers"] == 24
        assert qwen_config["num_attention_heads"] == 14
        assert qwen_config["hidden_size"] == 896

    def test_unknown_model_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model_config("nonexistent/model-xyz")


class TestLoadModelWeights:
    """Weight loading tests."""

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_qwen_weights(self, qwen_weights):
        assert isinstance(qwen_weights, dict)
        assert len(qwen_weights) > 0
        assert all(isinstance(v, np.ndarray) for v in qwen_weights.values())

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_qwen_has_embed(self, qwen_weights):
        assert "model.embed_tokens.weight" in qwen_weights
        assert qwen_weights["model.embed_tokens.weight"].shape == (151936, 896)

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_qwen_has_attn(self, qwen_weights):
        attn_keys = [k for k in qwen_weights if "attn" in k]
        assert len(attn_keys) > 0

    def test_unknown_model_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model_weights("nonexistent/model-xyz")

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_weights_are_float32(self, qwen_weights):
        for k, v in list(qwen_weights.items())[:5]:
            assert v.dtype == np.float32, f"{k} has dtype {v.dtype}"


class TestListCachedModels:
    """Cached model listing tests."""

    def test_returns_list(self):
        models = list_cached_models()
        assert isinstance(models, list)

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_qwen_in_list(self):
        models = list_cached_models()
        ids = [m["id"] for m in models]
        assert "Qwen/Qwen2.5-0.5B-Instruct" in ids

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
