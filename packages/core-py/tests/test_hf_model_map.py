"""Tests for domains.training.huggingface.model_map — ModelSize, HFModelInfo, lookup functions."""

from domains.training.huggingface.model_map import (
    ModelSize, HFModelInfo, get_model_info, search_models,
    get_recommended_quantization, get_model_requirements, map_to_sloughgpt_config,
)


class TestModelSize:
    def test_all_members(self):
        assert len(ModelSize) == 4

    def test_values(self):
        assert ModelSize.SMALL.value == "small"
        assert ModelSize.MEDIUM.value == "medium"
        assert ModelSize.LARGE.value == "large"
        assert ModelSize.XLARGE.value == "xlarge"


class TestHFModelInfo:
    def test_fields(self):
        info = HFModelInfo(
            model_id="gpt2", name="GPT-2", description="base",
            size=ModelSize.SMALL, params=124_000_000, context_length=1024,
            recommended_quantization="int8", memory_fp16_gb=0.5,
            memory_int8_gb=0.3, memory_q4_gb=0.2, organization="openai",
            tags=["text", "causal"],
        )
        assert info.model_id == "gpt2"
        assert info.size == ModelSize.SMALL


class TestLookupFunctions:
    def test_get_model_info_found(self):
        info = get_model_info("gpt2")
        assert info is not None
        assert info.model_id == "gpt2"

    def test_get_model_info_missing(self):
        info = get_model_info("nonexistent_model_xyz")
        assert info is None

    def test_search_models_by_org(self):
        results = search_models(organization="openai")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_models_by_size(self):
        results = search_models(size=ModelSize.SMALL)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_models_by_tags(self):
        results = search_models(tags=["gpt"])
        assert isinstance(results, list)
        assert len(results) > 0

    def test_get_recommended_quantization(self):
        q = get_recommended_quantization("gpt2")
        assert isinstance(q, str)
        assert len(q) > 0

    def test_get_model_requirements(self):
        reqs = get_model_requirements("gpt2", precision="bf16")
        assert isinstance(reqs, dict)
        assert "memory_gb" in reqs

    def test_map_to_sloughgpt_config(self):
        cfg = map_to_sloughgpt_config("gpt2")
        assert isinstance(cfg, dict)
