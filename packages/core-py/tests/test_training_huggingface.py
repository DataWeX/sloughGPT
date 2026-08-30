"""Tests for packages/core-py/domains/training/huggingface/ modules.

Covers pure logic only: config parsing, data processing, model map,
format helpers, and error paths. No external API calls or model loading.
"""

import os
from pathlib import Path

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
from domains.training.huggingface.local_loader import (
    HFLocalConfig,
    HuggingFaceLocalLoader,
    HuggingFaceLocalClient,
)
from domains.training.huggingface.api_loader import (
    HFAPIConfig,
    HuggingFaceAPILoader,
    HFInferenceClient,
    create_api_client,
)
from domains.training.huggingface.client import (
    HFClient,
    get_model_memory,
    list_models,
)


# ---------------------------------------------------------------------------
# model_map tests
# ---------------------------------------------------------------------------

class TestModelSize:
    def test_enum_values(self):
        assert ModelSize.SMALL.value == "small"
        assert ModelSize.MEDIUM.value == "medium"
        assert ModelSize.LARGE.value == "large"
        assert ModelSize.XLARGE.value == "xlarge"

    def test_enum_member_count(self):
        assert len(ModelSize) == 4


class TestHFModelInfo:
    def test_fields(self):
        info = HFModelInfo(
            model_id="test/model",
            name="Test",
            description="desc",
            size=ModelSize.SMALL,
            params=100,
            context_length=512,
            recommended_quantization="fp16",
            memory_fp16_gb=0.5,
            memory_int8_gb=0.3,
            memory_q4_gb=0.2,
            organization="test",
            tags=["tag1"],
        )
        assert info.model_id == "test/model"
        assert info.size == ModelSize.SMALL
        assert info.params == 100
        assert "tag1" in info.tags


class TestGetModelInfo:
    def test_known_model(self):
        info = get_model_info("gpt2")
        assert info is not None
        assert info.model_id == "gpt2"
        assert info.organization == "openai"
        assert info.size == ModelSize.SMALL

    def test_unknown_model_returns_none(self):
        assert get_model_info("nonexistent/model") is None


class TestSearchModels:
    def test_no_filters_returns_all(self):
        results = search_models()
        assert len(results) == len(HF_MODELS)

    def test_filter_by_organization(self):
        results = search_models(organization="meta")
        assert len(results) == 2
        assert all(m.organization == "meta" for m in results)

    def test_filter_by_size(self):
        results = search_models(size=ModelSize.LARGE)
        assert len(results) == 1
        assert results[0].model_id == "meta-llama/Llama-2-13b-chat-hf"

    def test_filter_by_tags(self):
        results = search_models(tags=["code"])
        model_ids = [m.model_id for m in results]
        assert "microsoft/phi-2" in model_ids
        assert "codellama/CodeLlama-7b-Instruct-hf" in model_ids

    def test_combined_filters(self):
        results = search_models(organization="meta", size=ModelSize.MEDIUM)
        assert len(results) == 1
        assert results[0].model_id == "meta-llama/Llama-2-7b-chat-hf"

    def test_no_match(self):
        results = search_models(organization="nonexistent")
        assert results == []

    def test_tags_partial_match(self):
        results = search_models(tags=["chat"])
        assert len(results) > 0

    def test_tags_no_match(self):
        results = search_models(tags=["nonexistent-tag"])
        assert results == []


class TestGetRecommendedQuantization:
    def test_known_model(self):
        assert get_recommended_quantization("gpt2") == "fp16"

    def test_unknown_model_returns_default(self):
        assert get_recommended_quantization("nonexistent") == "q4_k_m"

    def test_all_models_have_quantization(self):
        for model_id in HF_MODELS:
            q = get_recommended_quantization(model_id)
            assert q in ("fp16", "q4_k_m"), f"{model_id} has unexpected quant: {q}"


class TestGetModelRequirements:
    def test_known_model_default_precision(self):
        req = get_model_requirements("gpt2")
        assert req["model_id"] == "gpt2"
        assert req["name"] == "GPT-2"
        assert req["params"] == 124_000_000
        assert req["memory_gb"] == 0.5
        assert req["precision"] == "bf16"
        assert req["context_length"] == 1024
        assert req["size"] == "small"

    def test_fp32_precision(self):
        req = get_model_requirements("gpt2", precision="fp32")
        assert req["memory_gb"] == 1.0

    def test_fp16_precision(self):
        req = get_model_requirements("gpt2", precision="fp16")
        assert req["memory_gb"] == 0.5

    def test_int8_precision(self):
        req = get_model_requirements("gpt2", precision="int8")
        assert req["memory_gb"] == 0.3

    def test_q4_k_m_precision(self):
        req = get_model_requirements("gpt2", precision="q4_k_m")
        assert req["memory_gb"] == 0.2

    def test_q4_precision(self):
        req = get_model_requirements("gpt2", precision="q4")
        assert req["memory_gb"] == 0.2

    def test_q8_precision(self):
        req = get_model_requirements("gpt2", precision="q8")
        assert req["memory_gb"] == 0.3

    def test_unknown_precision_falls_back_to_fp16(self):
        req = get_model_requirements("gpt2", precision="unknown_thing")
        assert req["memory_gb"] == 0.5

    def test_unknown_model(self):
        req = get_model_requirements("nonexistent/model")
        assert req["model_id"] == "nonexistent/model"
        assert req["precision"] == "bf16"
        assert req["memory_gb"] == "unknown"
        assert req["params"] == "unknown"

    def test_large_model_requirements(self):
        req = get_model_requirements("meta-llama/Llama-2-13b-chat-hf", precision="fp32")
        assert req["memory_gb"] == 52.0
        assert req["size"] == "large"


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
        assert cfg["n_head"] == 16
        assert cfg["block_size"] == 1024

    def test_large_model(self):
        cfg = map_to_sloughgpt_config("meta-llama/Llama-2-13b-chat-hf")
        assert cfg["n_embed"] == 768
        assert cfg["n_layer"] == 20
        assert cfg["n_head"] == 24
        assert cfg["block_size"] == 1024

    def test_xlarge_model(self):
        results = search_models(size=ModelSize.XLARGE)
        if results:
            cfg = map_to_sloughgpt_config(results[0].model_id)
            assert cfg["n_embed"] == 1024
            assert cfg["n_layer"] == 24
            assert cfg["n_head"] == 32
            assert cfg["block_size"] == 2048

    def test_unknown_model_returns_empty(self):
        cfg = map_to_sloughgpt_config("nonexistent/model")
        assert cfg == {}

    def test_all_models_have_config(self):
        for model_id in HF_MODELS:
            cfg = map_to_sloughgpt_config(model_id)
            assert "n_embed" in cfg, f"{model_id} missing n_embed"
            assert "n_layer" in cfg, f"{model_id} missing n_layer"
            assert "n_head" in cfg, f"{model_id} missing n_head"
            assert "block_size" in cfg, f"{model_id} missing block_size"


# ---------------------------------------------------------------------------
# local_loader tests (pure logic — no transformer dependency)
# ---------------------------------------------------------------------------

class TestHFLocalConfig:
    def test_defaults(self):
        cfg = HFLocalConfig(model="gpt2")
        assert cfg.model == "gpt2"
        assert cfg.device == "auto"
        assert cfg.dtype == "auto"
        assert cfg.load_in_8bit is False
        assert cfg.load_in_4bit is False
        assert cfg.cache_dir is None
        assert cfg.local_files_only is True
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.repetition_penalty == 1.0

    def test_custom_values(self):
        cfg = HFLocalConfig(
            model="mistral",
            device="cuda",
            dtype="float16",
            load_in_8bit=True,
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.95,
        )
        assert cfg.device == "cuda"
        assert cfg.dtype == "float16"
        assert cfg.load_in_8bit is True
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.5
        assert cfg.top_p == 0.95


class TestHuggingFaceLocalLoader:
    def test_device_auto_resolves_to_cpu(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.config.device == "cpu"

    def test_device_explicit_preserved(self):
        cfg = HFLocalConfig(model="gpt2", device="cuda")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.config.device == "cuda"

    def test_get_dtype_auto(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader._get_dtype() == "auto"

    def test_get_dtype_float32(self):
        cfg = HFLocalConfig(model="gpt2", dtype="float32")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader._get_dtype() == "float32"

    def test_get_dtype_float16(self):
        cfg = HFLocalConfig(model="gpt2", dtype="float16")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader._get_dtype() == "float16"

    def test_get_dtype_half(self):
        cfg = HFLocalConfig(model="gpt2", dtype="half")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader._get_dtype() == "float16"

    def test_get_dtype_bfloat16(self):
        cfg = HFLocalConfig(model="gpt2", dtype="bfloat16")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader._get_dtype() == "bfloat16"

    def test_get_dtype_unknown_defaults_to_float32(self):
        cfg = HFLocalConfig(model="gpt2", dtype="something_weird")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader._get_dtype() == "float32"

    def test_generate_raises_without_load(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.generate("hello")

    def test_unload_sets_none(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        loader.model = "fake_model"
        loader.tokenizer = "fake_tokenizer"
        loader.unload()
        assert loader.model is None
        assert loader.tokenizer is None

    def test_format_chat_prompt_user_only(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        messages = [{"role": "user", "content": "hello"}]
        result = loader._format_chat_prompt(messages)
        assert result == "User: hello\nAssistant:"

    def test_format_chat_prompt_assistant(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        messages = [{"role": "assistant", "content": "hi there"}]
        result = loader._format_chat_prompt(messages)
        assert result == "Assistant: hi there\nAssistant:"

    def test_format_chat_prompt_system(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        messages = [{"role": "system", "content": "You are helpful"}]
        result = loader._format_chat_prompt(messages)
        assert result == "System: You are helpful\nAssistant:"

    def test_format_chat_prompt_multi_turn(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        result = loader._format_chat_prompt(messages)
        assert result == (
            "System: Be helpful\n"
            "User: hi\n"
            "Assistant: hello\n"
            "User: bye\n"
            "Assistant:"
        )

    def test_format_chat_prompt_missing_role_defaults_to_user(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        messages = [{"content": "hello"}]
        result = loader._format_chat_prompt(messages)
        assert result == "User: hello\nAssistant:"

    def test_format_chat_prompt_missing_content(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        messages = [{"role": "user"}]
        result = loader._format_chat_prompt(messages)
        assert result == "User: \nAssistant:"

    def test_format_chat_prompt_empty_messages(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        result = loader._format_chat_prompt([])
        assert result == "Assistant:"


class TestHuggingFaceLocalClient:
    def test_is_subclass(self):
        assert issubclass(HuggingFaceLocalClient, HuggingFaceLocalLoader)

    def test_instantiation(self):
        cfg = HFLocalConfig(model="gpt2")
        client = HuggingFaceLocalClient(cfg)
        assert client.config.model == "gpt2"


# ---------------------------------------------------------------------------
# api_loader tests (pure logic — no network calls)
# ---------------------------------------------------------------------------

class TestHFAPIConfig:
    def test_defaults(self):
        cfg = HFAPIConfig(model="gpt2")
        assert cfg.model == "gpt2"
        assert cfg.api_key is None
        assert cfg.timeout == 60
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.repetition_penalty == 1.0

    def test_custom_values(self):
        cfg = HFAPIConfig(model="llama", api_key="k", timeout=30, max_new_tokens=512)
        assert cfg.api_key == "k"
        assert cfg.timeout == 30
        assert cfg.max_new_tokens == 512


class TestHuggingFaceAPILoader:
    def test_api_key_from_config(self):
        cfg = HFAPIConfig(model="gpt2", api_key="test_key")
        loader = HuggingFaceAPILoader(cfg)
        assert loader.api_key == "test_key"
        assert loader.headers["Authorization"] == "Bearer test_key"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("HF_API_KEY", "env_key")
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        assert loader.api_key == "env_key"
        assert loader.headers["Authorization"] == "Bearer env_key"

    def test_api_key_from_hf_token(self, monkeypatch):
        monkeypatch.delenv("HF_API_KEY", raising=False)
        monkeypatch.setenv("HF_TOKEN", "token_key")
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        assert loader.api_key == "token_key"

    def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("HF_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        assert loader.api_key is None
        assert loader.headers == {}

    def test_format_chat_prompt_user(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([{"role": "user", "content": "hi"}])
        assert result == "User: hi\nAssistant:"

    def test_format_chat_prompt_assistant(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([{"role": "assistant", "content": "hello"}])
        assert result == "Assistant: hello\nAssistant:"

    def test_format_chat_prompt_system(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([{"role": "system", "content": "sys"}])
        assert result == "System: sys\nAssistant:"

    def test_format_chat_prompt_multi_turn(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = loader._format_chat_prompt(msgs)
        assert result == "System: You are helpful\nUser: hi\nAssistant: hello\nAssistant:"

    def test_format_chat_prompt_empty(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([])
        assert result == "Assistant:"


class TestHFInferenceClient:
    def test_is_subclass(self):
        assert issubclass(HFInferenceClient, HuggingFaceAPILoader)


class TestCreateApiClient:
    def test_returns_loader(self):
        client = create_api_client("gpt2")
        assert isinstance(client, HuggingFaceAPILoader)
        assert client.config.model == "gpt2"

    def test_with_kwargs(self):
        client = create_api_client("gpt2", api_key="k", timeout=10)
        assert client.api_key == "k"
        assert client.config.timeout == 10


# ---------------------------------------------------------------------------
# client.py tests (pure logic)
# ---------------------------------------------------------------------------

class TestHFClient:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            HFClient("gpt2", mode="invalid")

    def test_api_mode_creates_client(self):
        client = HFClient("gpt2", mode="api")
        assert client.model == "gpt2"
        assert client.mode == "api"
        assert isinstance(client._client, HuggingFaceAPILoader)

    def test_repr(self):
        client = HFClient("gpt2", mode="api")
        assert "gpt2" in repr(client)
        assert "api" in repr(client)


class TestGetModelMemory:
    def test_known_model(self):
        result = get_model_memory("gpt2")
        assert result["model_id"] == "gpt2"
        assert result["params"] == 124_000_000

    def test_unknown_model(self):
        result = get_model_memory("nonexistent")
        assert result["memory_gb"] == "unknown"


class TestListModels:
    def test_no_filters(self):
        models = list_models()
        assert len(models) == len(HF_MODELS)
        assert "gpt2" in models

    def test_filter_by_org(self):
        models = list_models(organization="meta")
        assert len(models) == 2
        assert all("meta-llama" in m for m in models)

    def test_filter_by_size(self):
        models = list_models(size="large")
        assert len(models) == 1
        assert "meta-llama/Llama-2-13b-chat-hf" in models

    def test_filter_by_size_small(self):
        models = list_models(size="small")
        assert len(models) > 0

    def test_filter_by_size_medium(self):
        models = list_models(size="medium")
        assert len(models) > 0

    def test_no_match(self):
        models = list_models(organization="nonexistent")
        assert models == []


# ---------------------------------------------------------------------------
# __init__ exports
# ---------------------------------------------------------------------------

class TestInitExports:
    def test_key_exports_importable(self):
        from domains.training.huggingface import (
            HFClient,
            HFLocalConfig,
            HuggingFaceLocalLoader,
            HuggingFaceLocalClient,
            ModelSize,
            HF_MODELS,
            get_model_info,
            search_models,
            get_recommended_quantization,
            get_model_requirements,
            map_to_sloughgpt_config,
            get_model_memory,
            list_models,
        )
        assert HFClient is not None
        assert HFLocalConfig is not None
        assert HF_MODELS is not None

    def test_local_model_loader_alias(self):
        from domains.training.huggingface import LocalModelLoader
        assert LocalModelLoader is HuggingFaceLocalLoader

    def test_model_registry_alias(self):
        from domains.training.huggingface import MODEL_REGISTRY
        assert MODEL_REGISTRY is HF_MODELS

    def test_model_info_alias(self):
        from domains.training.huggingface import ModelInfo
        assert ModelInfo is HFModelInfo
