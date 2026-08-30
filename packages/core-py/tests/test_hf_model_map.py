"""Tests for domains.training.huggingface.model_map — ModelSize, HFModelInfo, lookup functions."""

from domains.training.huggingface.model_map import (
    ModelSize, HFModelInfo, HF_MODELS, get_model_info, search_models,
    get_recommended_quantization, get_model_requirements, map_to_sloughgpt_config,
)


# ── ModelSize ────────────────────────────────────────────────────────────────


class TestModelSize:
    def test_all_members(self):
        assert len(ModelSize) == 4

    def test_values(self):
        assert ModelSize.SMALL.value == "small"
        assert ModelSize.MEDIUM.value == "medium"
        assert ModelSize.LARGE.value == "large"
        assert ModelSize.XLARGE.value == "xlarge"

    def test_member_names(self):
        names = [m.name for m in ModelSize]
        assert "SMALL" in names
        assert "MEDIUM" in names
        assert "LARGE" in names
        assert "XLARGE" in names

    def test_enum_identity(self):
        assert ModelSize.SMALL == ModelSize.SMALL
        assert ModelSize.SMALL != ModelSize.MEDIUM

    def test_enum_from_value(self):
        assert ModelSize("small") == ModelSize.SMALL
        assert ModelSize("medium") == ModelSize.MEDIUM
        assert ModelSize("large") == ModelSize.LARGE
        assert ModelSize("xlarge") == ModelSize.XLARGE

    def test_invalid_value_raises(self):
        import pytest
        with pytest.raises(ValueError):
            ModelSize("huge")


# ── HFModelInfo ──────────────────────────────────────────────────────────────


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

    def test_all_fields_stored(self):
        info = HFModelInfo(
            model_id="test", name="Test", description="desc",
            size=ModelSize.LARGE, params=7_000_000_000, context_length=4096,
            recommended_quantization="q4_k_m", memory_fp16_gb=14.0,
            memory_int8_gb=7.0, memory_q4_gb=4.0, organization="meta",
            tags=["llama", "chat"],
        )
        assert info.model_id == "test"
        assert info.name == "Test"
        assert info.description == "desc"
        assert info.size == ModelSize.LARGE
        assert info.params == 7_000_000_000
        assert info.context_length == 4096
        assert info.recommended_quantization == "q4_k_m"
        assert info.memory_fp16_gb == 14.0
        assert info.memory_int8_gb == 7.0
        assert info.memory_q4_gb == 4.0
        assert info.organization == "meta"
        assert info.tags == ["llama", "chat"]

    def test_dataclass_type(self):
        info = HFModelInfo(
            model_id="x", name="X", description="d",
            size=ModelSize.SMALL, params=100, context_length=128,
            recommended_quantization="fp16", memory_fp16_gb=0.1,
            memory_int8_gb=0.05, memory_q4_gb=0.03, organization="org",
            tags=[],
        )
        assert isinstance(info, HFModelInfo)

    def test_tags_list(self):
        info = HFModelInfo(
            model_id="x", name="X", description="d",
            size=ModelSize.SMALL, params=100, context_length=128,
            recommended_quantization="fp16", memory_fp16_gb=0.1,
            memory_int8_gb=0.05, memory_q4_gb=0.03, organization="org",
            tags=["a", "b", "c"],
        )
        assert len(info.tags) == 3

    def test_memory_ordering(self):
        info = HFModelInfo(
            model_id="x", name="X", description="d",
            size=ModelSize.SMALL, params=100, context_length=128,
            recommended_quantization="fp16", memory_fp16_gb=2.0,
            memory_int8_gb=1.0, memory_q4_gb=0.5, organization="org",
            tags=[],
        )
        assert info.memory_fp16_gb > info.memory_int8_gb > info.memory_q4_gb


# ── HF_MODELS registry ──────────────────────────────────────────────────────


class TestHFModelsRegistry:
    def test_registry_not_empty(self):
        assert len(HF_MODELS) > 0

    def test_all_values_are_hf_model_info(self):
        for key, info in HF_MODELS.items():
            assert isinstance(info, HFModelInfo), f"{key} is not HFModelInfo"

    def test_keys_match_model_ids(self):
        for key, info in HF_MODELS.items():
            assert key == info.model_id, f"Key {key} != model_id {info.model_id}"

    def test_all_have_model_id(self):
        for info in HF_MODELS.values():
            assert info.model_id
            assert isinstance(info.model_id, str)

    def test_all_have_name(self):
        for info in HF_MODELS.values():
            assert info.name
            assert isinstance(info.name, str)

    def test_all_have_description(self):
        for info in HF_MODELS.values():
            assert info.description
            assert isinstance(info.description, str)

    def test_all_have_size(self):
        for info in HF_MODELS.values():
            assert isinstance(info.size, ModelSize)

    def test_all_params_positive(self):
        for info in HF_MODELS.values():
            assert info.params > 0

    def test_all_context_length_positive(self):
        for info in HF_MODELS.values():
            assert info.context_length > 0

    def test_all_memory_values_non_negative(self):
        for info in HF_MODELS.values():
            assert info.memory_fp16_gb >= 0
            assert info.memory_int8_gb >= 0
            assert info.memory_q4_gb >= 0

    def test_all_have_organization(self):
        for info in HF_MODELS.values():
            assert info.organization
            assert isinstance(info.organization, str)

    def test_all_have_tags(self):
        for info in HF_MODELS.values():
            assert isinstance(info.tags, list)

    def test_all_have_quantization(self):
        for info in HF_MODELS.values():
            assert info.recommended_quantization
            assert isinstance(info.recommended_quantization, str)

    def test_gpt2_exists(self):
        assert "gpt2" in HF_MODELS

    def test_gpt2_medium_exists(self):
        assert "gpt2-medium" in HF_MODELS

    def test_gpt2_large_exists(self):
        assert "gpt2-large" in HF_MODELS

    def test_phi2_exists(self):
        assert "microsoft/phi-2" in HF_MODELS

    def test_mistral_7b_exists(self):
        assert "mistralai/Mistral-7B-Instruct-v0.2" in HF_MODELS

    def test_llama2_7b_exists(self):
        assert "meta-llama/Llama-2-7b-chat-hf" in HF_MODELS

    def test_llama2_13b_exists(self):
        assert "meta-llama/Llama-2-13b-chat-hf" in HF_MODELS

    def test_codellama_exists(self):
        assert "codellama/CodeLlama-7b-Instruct-hf" in HF_MODELS

    def test_qwen2_05b_exists(self):
        assert "Qwen/Qwen2-0.5B-Instruct" in HF_MODELS

    def test_qwen2_7b_exists(self):
        assert "Qwen/Qwen2-7B-Instruct" in HF_MODELS

    def test_gemma_2b_exists(self):
        assert "google/gemma-2b-it" in HF_MODELS

    def test_gemma_7b_exists(self):
        assert "google/gemma-7b-it" in HF_MODELS

    def test_deepseek_13b_exists(self):
        assert "deepseek-ai/DeepSeek-Coder-1.3B-instruct" in HF_MODELS

    def test_tinyllama_exists(self):
        assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in HF_MODELS

    def test_total_model_count(self):
        assert len(HF_MODELS) >= 20


# ── get_model_info ───────────────────────────────────────────────────────────


class TestGetModelInfo:
    def test_get_model_info_found(self):
        info = get_model_info("gpt2")
        assert info is not None
        assert info.model_id == "gpt2"

    def test_get_model_info_missing(self):
        info = get_model_info("nonexistent_model_xyz")
        assert info is None

    def test_returns_hf_model_info(self):
        info = get_model_info("gpt2")
        assert isinstance(info, HFModelInfo)

    def test_get_medium_model(self):
        info = get_model_info("microsoft/phi-2")
        assert info is not None
        assert info.size == ModelSize.MEDIUM

    def test_get_large_model(self):
        info = get_model_info("meta-llama/Llama-2-13b-chat-hf")
        assert info is not None
        assert info.size == ModelSize.LARGE

    def test_empty_string(self):
        info = get_model_info("")
        assert info is None

    def test_case_sensitive(self):
        info = get_model_info("GPT2")
        assert info is None

    def test_all_registered_models_retrievable(self):
        for model_id in HF_MODELS:
            info = get_model_info(model_id)
            assert info is not None, f"Could not retrieve {model_id}"


# ── search_models ────────────────────────────────────────────────────────────


class TestSearchModels:
    def test_search_models_by_org(self):
        results = search_models(organization="openai")
        assert isinstance(results, list)
        assert len(results) > 0
        for m in results:
            assert m.organization == "openai"

    def test_search_models_by_size(self):
        results = search_models(size=ModelSize.SMALL)
        assert isinstance(results, list)
        assert len(results) > 0
        for m in results:
            assert m.size == ModelSize.SMALL

    def test_search_models_by_tags(self):
        results = search_models(tags=["gpt"])
        assert isinstance(results, list)
        assert len(results) > 0
        for m in results:
            assert "gpt" in m.tags

    def test_search_no_filters(self):
        results = search_models()
        assert len(results) == len(HF_MODELS)

    def test_search_by_org_and_size(self):
        results = search_models(organization="openai", size=ModelSize.SMALL)
        assert all(m.organization == "openai" and m.size == ModelSize.SMALL for m in results)

    def test_search_nonexistent_org(self):
        results = search_models(organization="nonexistent")
        assert results == []

    def test_search_nonexistent_tag(self):
        results = search_models(tags=["nonexistent_tag"])
        assert results == []

    def test_search_multiple_tags(self):
        results = search_models(tags=["chat", "instruction"])
        assert len(results) > 0
        for m in results:
            assert "chat" in m.tags or "instruction" in m.tags

    def test_search_medium_size(self):
        results = search_models(size=ModelSize.MEDIUM)
        assert len(results) > 0
        for m in results:
            assert m.size == ModelSize.MEDIUM

    def test_search_large_size(self):
        results = search_models(size=ModelSize.LARGE)
        assert len(results) > 0
        for m in results:
            assert m.size == ModelSize.LARGE

    def test_search_xlarge_size(self):
        results = search_models(size=ModelSize.XLARGE)
        assert len(results) == 0  # No XL models in registry

    def test_search_microsoft_org(self):
        results = search_models(organization="microsoft")
        assert len(results) > 0
        for m in results:
            assert m.organization == "microsoft"

    def test_search_google_org(self):
        results = search_models(organization="google")
        assert len(results) > 0
        for m in results:
            assert m.organization == "google"

    def test_search_meta_org(self):
        results = search_models(organization="meta")
        assert len(results) > 0
        for m in results:
            assert m.organization == "meta"

    def test_search_code_tag(self):
        results = search_models(tags=["code"])
        assert len(results) > 0
        for m in results:
            assert "code" in m.tags

    def test_search_multilingual_tag(self):
        results = search_models(tags=["multilingual"])
        assert len(results) > 0

    def test_search_results_are_hf_model_info(self):
        results = search_models(tags=["chat"])
        for m in results:
            assert isinstance(m, HFModelInfo)


# ── get_recommended_quantization ─────────────────────────────────────────────


class TestGetRecommendedQuantization:
    def test_get_recommended_quantization(self):
        q = get_recommended_quantization("gpt2")
        assert isinstance(q, str)
        assert len(q) > 0

    def test_gpt2_returns_fp16(self):
        q = get_recommended_quantization("gpt2")
        assert q == "fp16"

    def test_missing_model_returns_q4_k_m(self):
        q = get_recommended_quantization("nonexistent")
        assert q == "q4_k_m"

    def test_phi2_returns_q4_k_m(self):
        q = get_recommended_quantization("microsoft/phi-2")
        assert q == "q4_k_m"

    def test_tinyllama_returns_fp16(self):
        q = get_recommended_quantization("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert q == "fp16"

    def test_all_registered_models_have_quantization(self):
        for model_id in HF_MODELS:
            q = get_recommended_quantization(model_id)
            assert isinstance(q, str)
            assert len(q) > 0

    def test_empty_string_returns_default(self):
        q = get_recommended_quantization("")
        assert q == "q4_k_m"


# ── get_model_requirements ───────────────────────────────────────────────────


class TestGetModelRequirements:
    def test_get_model_requirements(self):
        reqs = get_model_requirements("gpt2", precision="bf16")
        assert isinstance(reqs, dict)
        assert "memory_gb" in reqs

    def test_all_keys_present(self):
        reqs = get_model_requirements("gpt2", precision="fp16")
        assert "model_id" in reqs
        assert "name" in reqs
        assert "params" in reqs
        assert "precision" in reqs
        assert "memory_gb" in reqs
        assert "context_length" in reqs
        assert "size" in reqs

    def test_missing_model(self):
        reqs = get_model_requirements("nonexistent")
        assert reqs["memory_gb"] == "unknown"
        assert reqs["params"] == "unknown"

    def test_precision_fp32(self):
        reqs = get_model_requirements("gpt2", precision="fp32")
        assert reqs["precision"] == "fp32"
        assert isinstance(reqs["memory_gb"], (int, float))

    def test_precision_fp16(self):
        reqs = get_model_requirements("gpt2", precision="fp16")
        assert reqs["precision"] == "fp16"

    def test_precision_int8(self):
        reqs = get_model_requirements("gpt2", precision="int8")
        assert reqs["precision"] == "int8"

    def test_precision_q4_k_m(self):
        reqs = get_model_requirements("gpt2", precision="q4_k_m")
        assert reqs["precision"] == "q4_k_m"

    def test_memory_fp32_greater_than_fp16(self):
        r_fp32 = get_model_requirements("gpt2", precision="fp32")
        r_fp16 = get_model_requirements("gpt2", precision="fp16")
        assert r_fp32["memory_gb"] > r_fp16["memory_gb"]

    def test_memory_fp16_greater_than_int8(self):
        r_fp16 = get_model_requirements("gpt2", precision="fp16")
        r_int8 = get_model_requirements("gpt2", precision="int8")
        assert r_fp16["memory_gb"] > r_int8["memory_gb"]

    def test_memory_int8_greater_than_q4(self):
        r_int8 = get_model_requirements("gpt2", precision="int8")
        r_q4 = get_model_requirements("gpt2", precision="q4_k_m")
        assert r_int8["memory_gb"] > r_q4["memory_gb"]

    def test_phi2_has_more_memory_than_gpt2(self):
        r_gpt2 = get_model_requirements("gpt2", precision="fp16")
        r_phi2 = get_model_requirements("microsoft/phi-2", precision="fp16")
        assert r_phi2["memory_gb"] > r_gpt2["memory_gb"]

    def test_missing_model_model_id(self):
        reqs = get_model_requirements("missing")
        assert reqs["model_id"] == "missing"

    def test_missing_model_precision(self):
        reqs = get_model_requirements("missing", precision="int8")
        assert reqs["precision"] == "int8"

    def test_unknown_precision_uses_fp16(self):
        reqs_fp16 = get_model_requirements("gpt2", precision="fp16")
        reqs_unknown = get_model_requirements("gpt2", precision="unknown")
        assert reqs_fp16["memory_gb"] == reqs_unknown["memory_gb"]

    def test_context_length_matches_model(self):
        reqs = get_model_requirements("gpt2")
        info = get_model_info("gpt2")
        assert reqs["context_length"] == info.context_length

    def test_name_matches_model(self):
        reqs = get_model_requirements("gpt2")
        info = get_model_info("gpt2")
        assert reqs["name"] == info.name


# ── map_to_sloughgpt_config ─────────────────────────────────────────────────


class TestMapToSloughgptConfig:
    def test_map_to_sloughgpt_config(self):
        cfg = map_to_sloughgpt_config("gpt2")
        assert isinstance(cfg, dict)

    def test_all_keys_for_small(self):
        cfg = map_to_sloughgpt_config("gpt2")
        assert "n_embed" in cfg
        assert "n_layer" in cfg
        assert "n_head" in cfg
        assert "block_size" in cfg

    def test_small_model_config(self):
        cfg = map_to_sloughgpt_config("gpt2")
        assert cfg["n_embed"] == 256
        assert cfg["n_layer"] == 6
        assert cfg["n_head"] == 8
        assert cfg["block_size"] == 512

    def test_medium_model_config(self):
        cfg = map_to_sloughgpt_config("microsoft/phi-2")
        assert cfg["n_embed"] == 512
        assert cfg["n_layer"] == 12
        assert cfg["n_head"] == 16
        assert cfg["block_size"] == 1024

    def test_large_model_config(self):
        cfg = map_to_sloughgpt_config("meta-llama/Llama-2-13b-chat-hf")
        assert cfg["n_embed"] == 768
        assert cfg["n_layer"] == 20
        assert cfg["n_head"] == 24
        assert cfg["block_size"] == 1024

    def test_missing_model_returns_empty(self):
        cfg = map_to_sloughgpt_config("nonexistent")
        assert cfg == {}

    def test_config_values_are_integers(self):
        for model_id in HF_MODELS:
            cfg = map_to_sloughgpt_config(model_id)
            assert isinstance(cfg, dict)
            for key in ["n_embed", "n_layer", "n_head", "block_size"]:
                assert isinstance(cfg[key], int), f"{model_id}.{key} is not int"

    def test_config_values_positive(self):
        for model_id in HF_MODELS:
            cfg = map_to_sloughgpt_config(model_id)
            for key in ["n_embed", "n_layer", "n_head", "block_size"]:
                assert cfg[key] > 0, f"{model_id}.{key} is not positive"

    def test_small_models_all_same_config(self):
        small_ids = [mid for mid, info in HF_MODELS.items() if info.size == ModelSize.SMALL]
        configs = [map_to_sloughgpt_config(mid) for mid in small_ids]
        for cfg in configs[1:]:
            assert cfg == configs[0]

    def test_medium_models_all_same_config(self):
        medium_ids = [mid for mid, info in HF_MODELS.items() if info.size == ModelSize.MEDIUM]
        configs = [map_to_sloughgpt_config(mid) for mid in medium_ids]
        for cfg in configs[1:]:
            assert cfg == configs[0]

    def test_large_models_all_same_config(self):
        large_ids = [mid for mid, info in HF_MODELS.items() if info.size == ModelSize.LARGE]
        configs = [map_to_sloughgpt_config(mid) for mid in large_ids]
        for cfg in configs[1:]:
            assert cfg == configs[0]

    def test_config_embedding_scales_with_size(self):
        small = map_to_sloughgpt_config("gpt2")
        medium = map_to_sloughgpt_config("microsoft/phi-2")
        large = map_to_sloughgpt_config("meta-llama/Llama-2-13b-chat-hf")
        assert small["n_embed"] < medium["n_embed"] < large["n_embed"]
        assert small["n_layer"] < medium["n_layer"] < large["n_layer"]
