"""Tests for HuggingFace model map registry — pure data, zero dependencies.

Covers:
  - ModelSize enum values
  - HFModelInfo dataclass structure
  - HF_MODELS registry completeness and consistency
  - get_model_info() lookup
  - search_models() filtering by org, size, tags
  - get_recommended_quantization() with known and unknown models
  - get_model_requirements() for all precision modes
  - map_to_sloughgpt_config() size-to-config mapping
"""

import pytest
from domains.training.huggingface.model_map import (
    ModelSize,
    HFModelInfo,
    HF_MODELS,
    get_model_info,
    search_models,
    get_recommended_quantization,
    get_model_requirements,
    map_to_sloughgpt_config,
)


class TestModelSize:
    def test_enum_values(self):
        assert ModelSize.SMALL.value == "small"
        assert ModelSize.MEDIUM.value == "medium"
        assert ModelSize.LARGE.value == "large"
        assert ModelSize.XLARGE.value == "xlarge"

    def test_enum_members(self):
        assert len(ModelSize) == 4


class TestHFModelInfo:
    def test_dataclass_fields(self):
        m = HFModelInfo(
            model_id="test",
            name="Test",
            description="desc",
            size=ModelSize.SMALL,
            params=1000,
            context_length=512,
            recommended_quantization="fp16",
            memory_fp16_gb=0.5,
            memory_int8_gb=0.3,
            memory_q4_gb=0.2,
            organization="org",
            tags=["tag1"],
        )
        assert m.model_id == "test"
        assert m.size == ModelSize.SMALL
        assert m.tags == ["tag1"]


class TestHF_MODELS:
    def test_not_empty(self):
        assert len(HF_MODELS) > 0

    def test_all_have_required_fields(self):
        for key, m in HF_MODELS.items():
            assert m.model_id == key, f"Key mismatch: {key} vs {m.model_id}"
            assert isinstance(m.size, ModelSize)
            assert m.params > 0
            assert m.context_length > 0
            assert m.memory_fp16_gb > 0
            assert m.memory_int8_gb > 0
            assert m.memory_q4_gb > 0
            assert len(m.tags) > 0

    def test_small_models_reasonable_range(self):
        for m in HF_MODELS.values():
            if m.size == ModelSize.SMALL:
                assert m.params <= 4_000_000_000, f"{m.model_id} has {m.params} params but is SMALL"

    def test_medium_models_1b_to_7b(self):
        for m in HF_MODELS.values():
            if m.size == ModelSize.MEDIUM:
                assert 1_000_000_000 <= m.params <= 8_000_000_000

    def test_large_models_above_7b(self):
        for m in HF_MODELS.values():
            if m.size == ModelSize.LARGE:
                assert m.params >= 7_000_000_000

    def test_memory_fp16_gte_int8(self):
        for m in HF_MODELS.values():
            assert m.memory_fp16_gb >= m.memory_int8_gb, f"{m.model_id}: fp16 < int8"

    def test_memory_int8_gte_q4(self):
        for m in HF_MODELS.values():
            assert m.memory_int8_gb >= m.memory_q4_gb, f"{m.model_id}: int8 < q4"

    def test_quantization_is_valid(self):
        valid = {"fp16", "q4_k_m"}
        for m in HF_MODELS.values():
            assert m.recommended_quantization in valid


class TestGetModelInfo:
    def test_known_model(self):
        m = get_model_info("gpt2")
        assert m is not None
        assert m.name == "GPT-2"

    def test_unknown_model(self):
        assert get_model_info("nonexistent") is None

    def test_returns_correct_type(self):
        m = get_model_info("gpt2")
        assert isinstance(m, HFModelInfo)


class TestSearchModels:
    def test_no_filters(self):
        results = search_models()
        assert len(results) == len(HF_MODELS)

    def test_filter_by_organization(self):
        results = search_models(organization="openai")
        assert len(results) > 0
        assert all(m.organization == "openai" for m in results)

    def test_filter_by_size(self):
        results = search_models(size=ModelSize.LARGE)
        assert len(results) > 0
        assert all(m.size == ModelSize.LARGE for m in results)

    def test_filter_by_tags(self):
        results = search_models(tags=["code"])
        assert len(results) > 0
        assert any("code" in m.tags for m in results)

    def test_combined_filters(self):
        results = search_models(organization="qwen", size=ModelSize.MEDIUM)
        assert all(m.organization == "qwen" and m.size == ModelSize.MEDIUM for m in results)

    def test_no_match(self):
        results = search_models(organization="nonexistent")
        assert len(results) == 0


class TestGetRecommendedQuantization:
    def test_known_model(self):
        q = get_recommended_quantization("gpt2")
        assert q == "fp16"

    def test_unknown_model_defaults(self):
        q = get_recommended_quantization("nonexistent")
        assert q == "q4_k_m"


class TestGetModelRequirements:
    def test_known_model_default_precision(self):
        req = get_model_requirements("gpt2")
        assert req["model_id"] == "gpt2"
        assert req["params"] == 124_000_000
        assert req["precision"] == "bf16"
        assert req["memory_gb"] == 0.5  # fp16 == bf16
        assert req["context_length"] == 1024
        assert req["size"] == "small"

    def test_fp32_precision(self):
        req = get_model_requirements("gpt2", precision="fp32")
        assert req["memory_gb"] == 1.0  # 0.5 * 2

    def test_int8_precision(self):
        req = get_model_requirements("gpt2", precision="int8")
        assert req["memory_gb"] == 0.3

    def test_q4_precision(self):
        req = get_model_requirements("gpt2", precision="q4")
        assert req["memory_gb"] == 0.2

    def test_unknown_precision_fallback(self):
        req = get_model_requirements("gpt2", precision="bogus")
        assert req["memory_gb"] == 0.5  # falls back to fp16

    def test_unknown_model(self):
        req = get_model_requirements("nonexistent")
        assert req["memory_gb"] == "unknown"
        assert req["params"] == "unknown"


class TestMapToSloughgptConfig:
    def test_small_model(self):
        cfg = map_to_sloughgpt_config("gpt2")
        assert cfg["n_embed"] == 256
        assert cfg["n_layer"] == 6
        assert cfg["n_head"] == 8
        assert cfg["block_size"] == 512

    def test_medium_model(self):
        cfg = map_to_sloughgpt_config("mistralai/Mistral-7B-Instruct-v0.2")
        assert cfg["n_embed"] == 512
        assert cfg["n_layer"] == 12

    def test_large_model(self):
        cfg = map_to_sloughgpt_config("meta-llama/Llama-2-13b-chat-hf")
        assert cfg["n_embed"] == 768
        assert cfg["n_layer"] == 20

    def test_unknown_model_returns_empty(self):
        cfg = map_to_sloughgpt_config("nonexistent")
        assert cfg == {}
