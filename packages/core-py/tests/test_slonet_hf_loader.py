"""Tests for training.huggingface.local_loader — HFLocalConfig."""

from __future__ import annotations

import pytest
from domains.training.huggingface.local_loader import (
    HFLocalConfig,
    HuggingFaceLocalLoader,
)


# ── HFLocalConfig ──────────────────────────────────────────────────────────


class TestHFLocalConfig:

    def test_default(self):
        config = HFLocalConfig(model="test-model")
        assert config.model == "test-model"
        assert config.device == "auto"
        assert config.dtype == "auto"
        assert config.load_in_8bit is False
        assert config.load_in_4bit is False

    def test_custom(self):
        config = HFLocalConfig(
            model="test-model",
            device="cpu",
            dtype="float16",
            max_new_tokens=512,
        )
        assert config.device == "cpu"
        assert config.dtype == "float16"
        assert config.max_new_tokens == 512


# ── HuggingFaceLocalLoader ─────────────────────────────────────────────────


class TestHuggingFaceLocalLoader:

    def test_init(self):
        config = HFLocalConfig(model="test-model")
        loader = HuggingFaceLocalLoader(config)
        assert loader.model is None
        assert loader.tokenizer is None

    def test_device_auto(self):
        config = HFLocalConfig(model="test-model", device="auto")
        loader = HuggingFaceLocalLoader(config)
        assert loader.config.device == "cpu"

    def test_get_dtype(self):
        config = HFLocalConfig(model="test-model", dtype="float16")
        loader = HuggingFaceLocalLoader(config)
        assert loader._get_dtype() == "float16"

    def test_get_dtype_auto(self):
        config = HFLocalConfig(model="test-model", dtype="auto")
        loader = HuggingFaceLocalLoader(config)
        assert loader._get_dtype() == "auto"

    def test_get_dtype_invalid(self):
        config = HFLocalConfig(model="test-model", dtype="invalid")
        loader = HuggingFaceLocalLoader(config)
        assert loader._get_dtype() == "float32"
