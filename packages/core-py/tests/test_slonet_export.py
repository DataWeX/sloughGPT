"""Tests for training.gguf_export — GGUF export classes."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.gguf_export import (
    GGUFExportConfig,
    TensorMapping,
    SloughGPTMapping,
    LLaMAMapping,
    MistralMapping,
    GPT2Mapping,
    OPTMapping,
    FalconMapping,
    GPTNeoXMapping,
    QUANTIZATION_TYPES,
    MOBILE_RECOMMENDED,
)


# ── GGUFExportConfig ───────────────────────────────────────────────────────


class TestGGUFExportConfig:

    def test_default(self):
        config = GGUFExportConfig()
        assert config.model_name == "sloughgpt"
        assert config.quantization == "Q4_K_M"
        assert config.n_ctx == 2048

    def test_custom(self):
        config = GGUFExportConfig(model_name="test", quantization="Q8_0")
        assert config.model_name == "test"
        assert config.quantization == "Q8_0"


# ── QUANTIZATION_TYPES / MOBILE_RECOMMENDED ────────────────────────────────


class TestQuantizationTypes:

    def test_types_exist(self):
        assert "F32" in QUANTIZATION_TYPES
        assert "F16" in QUANTIZATION_TYPES
        assert "Q4_K_M" in QUANTIZATION_TYPES

    def test_mobile_recommended(self):
        assert "Q4_K_M" in MOBILE_RECOMMENDED


# ── SloughGPTMapping ───────────────────────────────────────────────────────


class TestSloughGPTMapping:

    def test_init(self):
        mapping = SloughGPTMapping()
        assert mapping.name == "sloughgpt"

    def test_get_tensor_map(self):
        mapping = SloughGPTMapping()
        tm = mapping.get_tensor_map()
        assert isinstance(tm, dict)

    def test_get_block_prefix(self):
        mapping = SloughGPTMapping()
        prefix = mapping.get_block_prefix()
        assert isinstance(prefix, str)

    def test_has_rope(self):
        mapping = SloughGPTMapping()
        assert mapping.has_rope() is True

    def test_has_position_embeddings(self):
        mapping = SloughGPTMapping()
        assert mapping.has_position_embeddings() is False


# ── LLaMAMapping ───────────────────────────────────────────────────────────


class TestLLaMAMapping:

    def test_init(self):
        mapping = LLaMAMapping()
        assert mapping.name == "llama"

    def test_get_tensor_map(self):
        mapping = LLaMAMapping()
        tm = mapping.get_tensor_map()
        assert isinstance(tm, dict)

    def test_has_rope(self):
        mapping = LLaMAMapping()
        assert mapping.has_rope() is True


# ── MistralMapping ─────────────────────────────────────────────────────────


class TestMistralMapping:

    def test_init(self):
        mapping = MistralMapping()
        assert mapping.name == "mistral"

    def test_get_tensor_map(self):
        mapping = MistralMapping()
        tm = mapping.get_tensor_map()
        assert isinstance(tm, dict)


# ── GPT2Mapping ────────────────────────────────────────────────────────────


class TestGPT2Mapping:

    def test_init(self):
        mapping = GPT2Mapping()
        assert mapping.name == "gpt2"

    def test_get_tensor_map(self):
        mapping = GPT2Mapping()
        tm = mapping.get_tensor_map()
        assert isinstance(tm, dict)

    def test_has_rope(self):
        mapping = GPT2Mapping()
        assert mapping.has_rope() is False

    def test_has_position_embeddings(self):
        mapping = GPT2Mapping()
        assert mapping.has_position_embeddings() is True


# ── OPTMapping ─────────────────────────────────────────────────────────────


class TestOPTMapping:

    def test_init(self):
        mapping = OPTMapping()
        assert mapping.name == "opt"

    def test_get_tensor_map(self):
        mapping = OPTMapping()
        tm = mapping.get_tensor_map()
        assert isinstance(tm, dict)


# ── FalconMapping ──────────────────────────────────────────────────────────


class TestFalconMapping:

    def test_init(self):
        mapping = FalconMapping()
        assert mapping.name == "falcon"

    def test_get_tensor_map(self):
        mapping = FalconMapping()
        tm = mapping.get_tensor_map()
        assert isinstance(tm, dict)


# ── GPTNeoXMapping ────────────────────────────────────────────────────────


class TestGPTNeoXMapping:

    def test_init(self):
        mapping = GPTNeoXMapping()
        assert mapping.name == "gpt_neox"

    def test_get_tensor_map(self):
        mapping = GPTNeoXMapping()
        tm = mapping.get_tensor_map()
        assert isinstance(tm, dict)


# ── TensorMapping abstract ─────────────────────────────────────────────────


class TestTensorMappingAbstract:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            TensorMapping("test")
