"""Tests for the HuggingFace integration package.

Covers model_map.py (offline registry), api_loader.py (requests mocked),
local_loader.py (transformers mocked), and client.py dispatch.
"""

import os
from types import SimpleNamespace

import pytest

from domains.training.huggingface import api_loader
from domains.training.huggingface.api_loader import (
    HFAPIConfig,
    HFInferenceClient,
    HuggingFaceAPILoader,
    chat_via_api,
    create_api_client,
    generate_via_api,
)
from domains.training.huggingface import local_loader
from domains.training.huggingface.local_loader import (
    HFLocalConfig,
    HuggingFaceLocalClient,
    HuggingFaceLocalLoader,
    download_model,
    generate_local,
    load_model,
)
from domains.training.huggingface import model_map
from domains.training.huggingface.model_map import (
    HFModelInfo,
    HF_MODELS,
    ModelSize,
    get_model_info,
    get_model_requirements,
    get_recommended_quantization,
    map_to_sloughgpt_config,
    search_models,
)
from domains.training.huggingface.client import (
    HFClient,
    chat,
    generate,
    get_model_memory,
    list_models,
)


# ---------------------------------------------------------------------------
# model_map
# ---------------------------------------------------------------------------

class TestModelSize:
    def test_enum_values(self):
        assert ModelSize.SMALL.value == "small"
        assert ModelSize.MEDIUM.value == "medium"
        assert ModelSize.LARGE.value == "large"
        assert ModelSize.XLARGE.value == "xlarge"


class TestHFModelInfo:
    def test_fields(self):
        info = HF_MODELS["gpt2"]
        assert info.model_id == "gpt2"
        assert info.name == "GPT-2"
        assert info.size == ModelSize.SMALL
        assert info.params == 124_000_000
        assert info.context_length == 1024
        assert info.recommended_quantization == "fp16"
        assert info.memory_fp16_gb == 0.5
        assert info.memory_int8_gb == 0.3
        assert info.memory_q4_gb == 0.2
        assert info.organization == "openai"
        assert "gpt" in info.tags


class TestRegistry:
    def test_known_models_present(self):
        for mid in [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "microsoft/phi-2",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "codellama/CodeLlama-7b-Instruct-hf",
            "Qwen/Qwen2-0.5B-Instruct",
            "google/gemma-2b-it",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ]:
            assert mid in HF_MODELS, mid

    def test_all_entries_are_hf_model_info(self):
        for info in HF_MODELS.values():
            assert isinstance(info, HFModelInfo)


class TestGetModelInfo:
    def test_known(self):
        assert get_model_info("gpt2") is HF_MODELS["gpt2"]

    def test_unknown_returns_none(self):
        assert get_model_info("nonexistent/model") is None


class TestSearchModels:
    def test_all(self):
        results = search_models()
        assert len(results) == len(HF_MODELS)

    def test_by_organization(self):
        results = search_models(organization="openai")
        assert results and all(m.organization == "openai" for m in results)
        assert {m.model_id for m in results} == {"gpt2", "gpt2-medium", "gpt2-large"}

    def test_by_size(self):
        results = search_models(size=ModelSize.MEDIUM)
        assert results and all(m.size == ModelSize.MEDIUM for m in results)

    def test_by_tags(self):
        results = search_models(tags=["code"])
        assert results and all("code" in m.tags for m in results)

    def test_combined_filters(self):
        results = search_models(organization="mistralai", size=ModelSize.MEDIUM)
        assert results and all(
            m.organization == "mistralai" and m.size == ModelSize.MEDIUM for m in results
        )

    def test_no_match(self):
        assert search_models(organization="nonexistent-org") == []


class TestGetRecommendedQuantization:
    def test_known_model(self):
        assert get_recommended_quantization("gpt2") == "fp16"
        assert get_recommended_quantization("mistralai/Mistral-7B-Instruct-v0.2") == "q4_k_m"

    def test_unknown_model_default(self):
        assert get_recommended_quantization("unknown/model") == "q4_k_m"


class TestGetModelRequirements:
    def test_known_bf16(self):
        req = get_model_requirements("gpt2", precision="bf16")
        assert req["model_id"] == "gpt2"
        assert req["name"] == "GPT-2"
        assert req["params"] == 124_000_000
        assert req["precision"] == "bf16"
        assert req["memory_gb"] == 0.5
        assert req["context_length"] == 1024
        assert req["size"] == "small"

    @pytest.mark.parametrize(
        "precision,expected",
        [
            ("fp32", 1.0),
            ("fp16", 0.5),
            ("bf16", 0.5),
            ("int8", 0.3),
            ("q4_k_m", 0.2),
            ("q4", 0.2),
            ("q8", 0.3),
        ],
    )
    def test_precision_variants(self, precision, expected):
        req = get_model_requirements("gpt2", precision=precision)
        assert req["memory_gb"] == expected

    def test_unknown_precision_falls_back_to_fp16(self):
        req = get_model_requirements("gpt2", precision="exotic")
        assert req["memory_gb"] == 0.5

    def test_unknown_model(self):
        req = get_model_requirements("unknown/model")
        assert req["model_id"] == "unknown/model"
        assert req["memory_gb"] == "unknown"
        assert req["params"] == "unknown"


class TestMapToSloughgptConfig:
    def test_small(self):
        cfg = map_to_sloughgpt_config("gpt2")
        assert cfg["n_embed"] == 256
        assert cfg["n_layer"] == 6
        assert cfg["n_head"] == 8
        assert cfg["block_size"] == 512

    def test_medium(self):
        cfg = map_to_sloughgpt_config("meta-llama/Llama-2-7b-chat-hf")
        assert cfg["n_embed"] == 512
        assert cfg["n_layer"] == 12
        assert cfg["n_head"] == 16
        assert cfg["block_size"] == 1024

    def test_large(self):
        cfg = map_to_sloughgpt_config("meta-llama/Llama-2-13b-chat-hf")
        assert cfg["n_embed"] == 768
        assert cfg["n_layer"] == 20
        assert cfg["n_head"] == 24
        assert cfg["block_size"] == 1024

    def test_xlarge(self, monkeypatch):
        info = HFModelInfo(
            model_id="x",
            name="X",
            description="",
            size=ModelSize.XLARGE,
            params=15_000_000_000,
            context_length=4096,
            recommended_quantization="q4_k_m",
            memory_fp16_gb=30.0,
            memory_int8_gb=15.0,
            memory_q4_gb=8.0,
            organization="org",
            tags=[],
        )
        monkeypatch.setattr(model_map, "get_model_info", lambda mid: info)
        cfg = map_to_sloughgpt_config("x")
        assert cfg["n_embed"] == 1024
        assert cfg["n_layer"] == 24
        assert cfg["n_head"] == 32
        assert cfg["block_size"] == 2048

    def test_unknown_model(self):
        assert map_to_sloughgpt_config("unknown/model") == {}


# ---------------------------------------------------------------------------
# api_loader
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, json_data=None, raise_error=None):
        self.status_code = status_code
        self._json_data = json_data
        self._raise_error = raise_error

    def raise_for_status(self):
        if self._raise_error:
            raise self._raise_error

    def json(self):
        return self._json_data


@pytest.fixture
def fake_api(monkeypatch):
    calls = []
    state = {"response": FakeResponse(json_data={"generated_text": "ok"})}

    def _post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return state["response"]

    monkeypatch.setattr(api_loader, "requests", SimpleNamespace(post=_post))
    return SimpleNamespace(
        calls=calls,
        set_response=lambda r: state.update(response=r),
    )


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


class TestLoaderInit:
    def test_api_key_from_config(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2", api_key="secret"))
        assert loader.api_key == "secret"
        assert loader.headers["Authorization"] == "Bearer secret"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("HF_API_KEY", "env-key")
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.api_key == "env-key"

    def test_api_key_from_hf_token_env(self, monkeypatch):
        monkeypatch.delenv("HF_API_KEY", raising=False)
        monkeypatch.setenv("HF_TOKEN", "token-key")
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.api_key == "token-key"

    def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("HF_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.api_key is None
        assert loader.headers == {}


class TestMakeRequest:
    def test_success(self, fake_api):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2", api_key="k"))
        result = loader._make_request({"inputs": "hi"})
        assert result == {"generated_text": "ok"}
        call = fake_api.calls[0]
        assert call["url"] == "https://api-inference.huggingface.co/models/gpt2"
        assert call["headers"] == {"Authorization": "Bearer k"}
        assert call["json"] == {"inputs": "hi"}
        assert call["timeout"] == 60

    def test_503_raises(self, monkeypatch):
        def _post(url, headers=None, json=None, timeout=None):
            return FakeResponse(status_code=503)

        monkeypatch.setattr(api_loader, "requests", SimpleNamespace(post=_post))
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        with pytest.raises(RuntimeError, match="loading"):
            loader._make_request({})

    def test_401_raises(self, monkeypatch):
        def _post(url, headers=None, json=None, timeout=None):
            return FakeResponse(status_code=401)

        monkeypatch.setattr(api_loader, "requests", SimpleNamespace(post=_post))
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        with pytest.raises(RuntimeError, match="token"):
            loader._make_request({})

    def test_http_error_raises(self, monkeypatch):
        def _post(url, headers=None, json=None, timeout=None):
            return FakeResponse(status_code=500, raise_error=RuntimeError("server error"))

        monkeypatch.setattr(api_loader, "requests", SimpleNamespace(post=_post))
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        with pytest.raises(RuntimeError, match="server error"):
            loader._make_request({})


class TestGenerate:
    def test_list_response(self, fake_api):
        fake_api.set_response(FakeResponse(json_data=[{"generated_text": "hello"}]))
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.generate("hi") == "hello"

    def test_dict_response(self, fake_api):
        fake_api.set_response(FakeResponse(json_data={"generated_text": "world"}))
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.generate("hi") == "world"

    def test_other_response(self, fake_api):
        fake_api.set_response(FakeResponse(json_data="plain-text"))
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.generate("hi") == "plain-text"

    def test_empty_list_returns_str(self, fake_api):
        fake_api.set_response(FakeResponse(json_data=[]))
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.generate("hi") == "[]"

    def test_payload_structure(self, fake_api):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2", max_new_tokens=10, temperature=0.3))
        loader.generate("hi", top_p=0.5)
        payload = fake_api.calls[0]["json"]
        assert payload["inputs"] == "hi"
        assert payload["options"] == {"use_cache": True}
        params = payload["parameters"]
        assert params["max_new_tokens"] == 10
        assert params["temperature"] == 0.3
        assert params["top_p"] == 0.5
        assert params["repetition_penalty"] == 1.0
        assert params["return_full_text"] is False

    def test_kwargs_merged_into_parameters(self, fake_api):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        loader.generate("hi", do_sample=False)
        assert fake_api.calls[0]["json"]["parameters"]["do_sample"] is False

    def test_overrides_defaults(self, fake_api):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        loader.generate("hi", max_new_tokens=5, temperature=0.1, repetition_penalty=2.0)
        params = fake_api.calls[0]["json"]["parameters"]
        assert params["max_new_tokens"] == 5
        assert params["temperature"] == 0.1
        assert params["repetition_penalty"] == 2.0


class TestChatApi:
    def test_chat_formats_then_generates(self, fake_api):
        fake_api.set_response(FakeResponse(json_data={"generated_text": "hey"}))
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        result = loader.chat([{"role": "user", "content": "hi"}])
        assert result == "hey"
        assert fake_api.calls[0]["json"]["inputs"] == "User: hi\nAssistant:"

    def test_format_chat_prompt(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        prompt = loader._format_chat_prompt(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"content": "default-role"},
            ]
        )
        assert prompt == "System: be brief\nUser: q1\nAssistant: a1\nUser: default-role\nAssistant:"


class TestApiHelpers:
    def test_hf_inference_client_alias(self):
        assert issubclass(HFInferenceClient, HuggingFaceAPILoader)

    def test_create_api_client(self):
        client = create_api_client("gpt2", api_key="k")
        assert isinstance(client, HuggingFaceAPILoader)
        assert client.config.model == "gpt2"
        assert client.api_key == "k"

    def test_generate_via_api(self, fake_api):
        fake_api.set_response(FakeResponse(json_data=[{"generated_text": "via-api"}]))
        assert generate_via_api("hi", model="gpt2", api_key="k") == "via-api"

    def test_chat_via_api(self, fake_api):
        fake_api.set_response(FakeResponse(json_data=[{"generated_text": "via-chat"}]))
        assert chat_via_api([{"role": "user", "content": "hi"}], model="gpt2", api_key="k") == "via-chat"

    def test_api_base_url(self):
        assert api_loader.HF_API_BASE == "https://api-inference.huggingface.co/models"


# ---------------------------------------------------------------------------
# local_loader
# ---------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, data, device="cpu"):
        self.data = data
        self.device = device

    def to(self, device):
        return FakeTensor(self.data, device)


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def __init__(self, **load_kwargs):
        self.load_kwargs = load_kwargs
        self.pad_token = None
        self.eos_token = "<eos>"
        self.last_prompt = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tok = cls(**kwargs)
        tok._init_args = args
        return tok

    def __call__(self, prompt, return_tensors="pt"):
        self.last_prompt = prompt
        return {
            "input_ids": FakeTensor([[1, 2, 3]]),
            "attention_mask": FakeTensor([[1, 1, 1]]),
        }

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded-text"


class FakeModel:
    def __init__(self, **load_kwargs):
        self.load_kwargs = load_kwargs
        self.to_called_with = None
        self.eval_called = False
        self.generate_kwargs = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(**kwargs)

    def to(self, device):
        self.to_called_with = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3, 4]]


@pytest.fixture
def fake_transformers(monkeypatch):
    monkeypatch.setattr(local_loader, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(local_loader, "AutoModelForCausalLM", FakeModel)


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


class TestDetermineDevice:
    def test_auto_becomes_cpu(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", device="auto"))
        assert loader.config.device == "cpu"

    def test_explicit_preserved(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", device="mps"))
        assert loader.config.device == "mps"


class TestGetDtype:
    def test_mapping(self):
        cases = {
            "float32": "float32",
            "float16": "float16",
            "half": "float16",
            "bfloat16": "bfloat16",
            "auto": "auto",
        }
        for key, expected in cases.items():
            loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype=key))
            assert loader._get_dtype() == expected

    def test_unknown_defaults_to_float32(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="float64"))
        assert loader._get_dtype() == "float32"


class TestLoad:
    def test_raises_without_transformers(self):
        assert local_loader.AutoTokenizer is None
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        with pytest.raises(ImportError, match="transformers"):
            loader.load()

    def test_load_success(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="float32"))
        loader.load()
        assert loader.model is not None
        assert loader.tokenizer is not None
        assert loader.model.eval_called is True
        assert loader.tokenizer.pad_token == "<eos>"
        assert loader.model.load_kwargs["dtype"] == "float32"
        assert loader.model.load_kwargs["device_map"] is None

    def test_load_preserves_existing_pad_token(self, fake_transformers, monkeypatch):
        class PadTokenizer(FakeTokenizer):
            def __init__(self, **load_kwargs):
                super().__init__(**load_kwargs)
                self.pad_token = "<pad>"

        monkeypatch.setattr(local_loader, "AutoTokenizer", PadTokenizer)
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.load()
        assert loader.tokenizer.pad_token == "<pad>"

    def test_load_8bit(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", load_in_8bit=True))
        loader.load()
        assert loader.model.load_kwargs["load_in_8bit"] is True
        assert loader.model.load_kwargs["device_map"] == "auto"
        assert "dtype" not in loader.model.load_kwargs
        assert loader.model.to_called_with is None

    def test_load_4bit(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", load_in_4bit=True))
        loader.load()
        assert loader.model.load_kwargs["load_in_4bit"] is True
        assert loader.model.load_kwargs["device_map"] == "auto"

    def test_load_cpu_default_no_to_call(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.load()
        assert loader.model.to_called_with is None

    def test_load_moves_model_to_device(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", device="mps"))
        loader.load()
        assert loader.model.to_called_with == "mps"


class TestGenerateLocal:
    def test_generate_before_load_raises(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        with pytest.raises(RuntimeError, match="load"):
            loader.generate("hi")

    def test_generate_success(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.load()
        result = loader.generate("hello", max_new_tokens=8, temperature=0.2)
        assert result == "decoded-text"
        gen = loader.model.generate_kwargs
        assert gen["max_new_tokens"] == 8
        assert gen["temperature"] == 0.2
        assert gen["top_p"] == 0.9
        assert gen["do_sample"] is True

    def test_generate_uses_config_defaults(self, fake_transformers):
        loader = HuggingFaceLocalLoader(
            HFLocalConfig(model="gpt2", max_new_tokens=42, temperature=0.5)
        )
        loader.load()
        loader.generate("hi")
        gen = loader.model.generate_kwargs
        assert gen["max_new_tokens"] == 42
        assert gen["temperature"] == 0.5

    def test_generate_moves_inputs_on_non_cpu(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", device="mps"))
        loader.load()
        result = loader.generate("hi")
        assert result == "decoded-text"
        inputs = loader.model.generate_kwargs
        assert inputs["input_ids"].device == "mps"

    def test_generate_extra_kwargs(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.load()
        loader.generate("hi", num_beams=3)
        assert loader.model.generate_kwargs["num_beams"] == 3


class TestChatLocal:
    def test_chat_formats_and_generates(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.load()
        result = loader.chat([{"role": "user", "content": "question"}])
        assert result == "decoded-text"
        assert loader.tokenizer.last_prompt == "User: question\nAssistant:"

    def test_format_chat_prompt(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "asst"},
            ]
        )
        assert prompt == "System: sys\nUser: usr\nAssistant: asst\nAssistant:"

    def test_chat_with_defaults(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.load()
        result = loader.chat([{"content": "no-role"}])
        assert result == "decoded-text"


class TestUnload:
    def test_unload_resets_state(self, fake_transformers):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.load()
        assert loader.model is not None
        loader.unload()
        assert loader.model is None
        assert loader.tokenizer is None


class TestLocalAliases:
    def test_hf_local_client_is_subclass(self):
        assert issubclass(HuggingFaceLocalClient, HuggingFaceLocalLoader)

    def test_download_model(self, fake_transformers, tmp_path):
        cache_dir = download_model("gpt2", cache_dir=str(tmp_path))
        assert cache_dir == str(tmp_path)

    def test_load_model_helper(self, fake_transformers):
        loader = load_model(HFLocalConfig(model="gpt2"))
        assert isinstance(loader, HuggingFaceLocalLoader)
        assert loader.model is not None

    def test_generate_local_helper(self, fake_transformers):
        assert generate_local("hi", model="gpt2") == "decoded-text"


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

class TestHFClientApiMode:
    def test_api_mode_uses_api_loader(self):
        client = HFClient("gpt2", mode="api")
        assert isinstance(client._client, HuggingFaceAPILoader)

    def test_call_delegates(self, monkeypatch):
        client = HFClient("gpt2", mode="api")
        monkeypatch.setattr(client._client, "generate", lambda p, **kw: f"gen:{p}")
        assert client("hello") == "gen:hello"

    def test_call_passes_kwargs(self, monkeypatch):
        client = HFClient("gpt2", mode="api")
        captured = {}

        def fake_generate(p, **kw):
            captured.update(kw)
            return "x"

        monkeypatch.setattr(client._client, "generate", fake_generate)
        client("hi", max_new_tokens=5)
        assert captured["max_new_tokens"] == 5

    def test_chat_delegates(self, monkeypatch):
        client = HFClient("gpt2", mode="api")
        monkeypatch.setattr(client._client, "chat", lambda msgs, **kw: "chat-ok")
        assert client.chat([{"role": "user", "content": "hi"}]) == "chat-ok"

    def test_repr(self):
        client = HFClient("gpt2", mode="api")
        assert repr(client) == "HFClient(model='gpt2', mode='api')"


class TestHFClientLocalMode:
    def test_local_mode_loads_model(self, fake_transformers):
        client = HFClient("gpt2", mode="local")
        assert isinstance(client._client, HuggingFaceLocalLoader)
        assert client._client.model is not None

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            HFClient("gpt2", mode="bogus")


class TestClientHelpers:
    def test_get_model_memory(self):
        result = get_model_memory("gpt2")
        assert result["model_id"] == "gpt2"
        assert result["memory_gb"] == 0.5

    def test_get_model_memory_unknown(self):
        result = get_model_memory("unknown/model")
        assert result["memory_gb"] == "unknown"

    def test_list_models_no_filter(self):
        models = list_models()
        assert "gpt2" in models
        assert len(models) == len(HF_MODELS)

    def test_list_models_by_organization(self):
        models = list_models(organization="openai")
        assert models == ["gpt2", "gpt2-medium", "gpt2-large"]

    def test_list_models_by_size(self):
        models = list_models(size="small")
        assert models
        assert all(get_model_info(m).size == ModelSize.SMALL for m in models)

    def test_list_models_invalid_size_raises(self):
        with pytest.raises(KeyError):
            list_models(size="huge")


class TestClientConvenience:
    def test_generate_convenience(self, monkeypatch):
        calls = []

        class FakeClient:
            def __init__(self, model, mode):
                calls.append((model, mode))

            def __call__(self, prompt, **kw):
                return f"out:{prompt}"

        monkeypatch.setattr("domains.training.huggingface.client.HFClient", FakeClient)
        assert generate("hello") == "out:hello"
        assert calls == [("meta-llama/Llama-2-7b-chat-hf", "api")]

    def test_chat_convenience(self, monkeypatch):
        class FakeClient:
            def __init__(self, model, mode):
                pass

            def chat(self, messages, **kw):
                return "chat-out"

        monkeypatch.setattr("domains.training.huggingface.client.HFClient", FakeClient)
        assert chat([{"role": "user", "content": "hi"}]) == "chat-out"
