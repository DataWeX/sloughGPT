"""Tests for domains.training.huggingface.api_loader — HFAPIConfig."""

from domains.training.huggingface.api_loader import (
    HFAPIConfig,
    HuggingFaceAPILoader,
    HFInferenceClient,
    create_api_client,
)


class TestHFAPIConfigDefaults:
    def test_model_required(self):
        hc = HFAPIConfig(model="gpt2")
        assert hc.model == "gpt2"

    def test_api_key_default_none(self):
        hc = HFAPIConfig(model="gpt2")
        assert hc.api_key is None

    def test_timeout_default(self):
        hc = HFAPIConfig(model="gpt2")
        assert hc.timeout == 60

    def test_max_new_tokens_default(self):
        hc = HFAPIConfig(model="gpt2")
        assert hc.max_new_tokens == 256

    def test_temperature_default(self):
        hc = HFAPIConfig(model="gpt2")
        assert hc.temperature == 0.7

    def test_top_p_default(self):
        hc = HFAPIConfig(model="gpt2")
        assert hc.top_p == 0.9

    def test_repetition_penalty_default(self):
        hc = HFAPIConfig(model="gpt2")
        assert hc.repetition_penalty == 1.0

    def test_field_count(self):
        import dataclasses
        fields = [f.name for f in dataclasses.fields(HFAPIConfig)]
        assert len(fields) == 7

    def test_field_names(self):
        import dataclasses
        names = {f.name for f in dataclasses.fields(HFAPIConfig)}
        expected = {"model", "api_key", "timeout", "max_new_tokens",
                    "temperature", "top_p", "repetition_penalty"}
        assert names == expected


class TestHFAPIConfigCustom:
    def test_model_custom(self):
        hc = HFAPIConfig(model="llama")
        assert hc.model == "llama"

    def test_api_key_custom(self):
        hc = HFAPIConfig(model="gpt2", api_key="secret")
        assert hc.api_key == "secret"

    def test_timeout_custom(self):
        hc = HFAPIConfig(model="gpt2", timeout=120)
        assert hc.timeout == 120

    def test_max_new_tokens_custom(self):
        hc = HFAPIConfig(model="gpt2", max_new_tokens=512)
        assert hc.max_new_tokens == 512

    def test_temperature_custom(self):
        hc = HFAPIConfig(model="gpt2", temperature=0.3)
        assert hc.temperature == 0.3

    def test_top_p_custom(self):
        hc = HFAPIConfig(model="gpt2", top_p=0.95)
        assert hc.top_p == 0.95

    def test_repetition_penalty_custom(self):
        hc = HFAPIConfig(model="gpt2", repetition_penalty=1.2)
        assert hc.repetition_penalty == 1.2

    def test_all_fields_custom(self):
        hc = HFAPIConfig(
            model="mistral-7b",
            api_key="key123",
            timeout=30,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.8,
            repetition_penalty=1.5,
        )
        assert hc.model == "mistral-7b"
        assert hc.api_key == "key123"
        assert hc.timeout == 30
        assert hc.max_new_tokens == 1024
        assert hc.temperature == 0.1
        assert hc.top_p == 0.8
        assert hc.repetition_penalty == 1.5

    def test_empty_model_string(self):
        hc = HFAPIConfig(model="")
        assert hc.model == ""

    def test_model_with_slash(self):
        hc = HFAPIConfig(model="meta-llama/Llama-2-7b-hf")
        assert hc.model == "meta-llama/Llama-2-7b-hf"


class TestHFAPIConfigEquality:
    def test_equal_instances(self):
        a = HFAPIConfig(model="gpt2")
        b = HFAPIConfig(model="gpt2")
        assert a == b

    def test_not_equal_model(self):
        a = HFAPIConfig(model="gpt2")
        b = HFAPIConfig(model="llama")
        assert a != b

    def test_not_equal_api_key(self):
        a = HFAPIConfig(model="gpt2", api_key="a")
        b = HFAPIConfig(model="gpt2", api_key="b")
        assert a != b

    def test_not_equal_to_non_dataclass(self):
        hc = HFAPIConfig(model="gpt2")
        assert hc != "not a config"

    def test_equal_custom(self):
        a = HFAPIConfig(model="x", temperature=0.5)
        b = HFAPIConfig(model="x", temperature=0.5)
        assert a == b


class TestHFAPIConfigRepr:
    def test_repr_contains_class_name(self):
        hc = HFAPIConfig(model="gpt2")
        assert "HFAPIConfig" in repr(hc)

    def test_repr_contains_model(self):
        hc = HFAPIConfig(model="llama")
        r = repr(hc)
        assert "model='llama'" in r

    def test_repr_contains_temperature(self):
        hc = HFAPIConfig(model="gpt2", temperature=0.3)
        r = repr(hc)
        assert "temperature=0.3" in r


class TestHFAPIConfigMutation:
    def test_can_set_model(self):
        hc = HFAPIConfig(model="gpt2")
        hc.model = "llama"
        assert hc.model == "llama"

    def test_can_set_api_key(self):
        hc = HFAPIConfig(model="gpt2")
        hc.api_key = "secret"
        assert hc.api_key == "secret"

    def test_can_set_temperature(self):
        hc = HFAPIConfig(model="gpt2")
        hc.temperature = 0.1
        assert hc.temperature == 0.1

    def test_can_overwrite_fields(self):
        hc = HFAPIConfig(model="gpt2", timeout=60)
        hc.timeout = 30
        hc.timeout = 120
        assert hc.timeout == 120

    def test_independent_instances(self):
        a = HFAPIConfig(model="a", temperature=0.1)
        b = HFAPIConfig(model="b", temperature=0.9)
        a.model = "c"
        assert b.model == "b"


class TestHFAPIConfigEdgeCases:
    def test_zero_timeout(self):
        hc = HFAPIConfig(model="gpt2", timeout=0)
        assert hc.timeout == 0

    def test_negative_temperature(self):
        hc = HFAPIConfig(model="gpt2", temperature=-1.0)
        assert hc.temperature == -1.0

    def test_zero_max_new_tokens(self):
        hc = HFAPIConfig(model="gpt2", max_new_tokens=0)
        assert hc.max_new_tokens == 0

    def test_negative_repetition_penalty(self):
        hc = HFAPIConfig(model="gpt2", repetition_penalty=-1.0)
        assert hc.repetition_penalty == -1.0

    def test_zero_top_p(self):
        hc = HFAPIConfig(model="gpt2", top_p=0.0)
        assert hc.top_p == 0.0

    def test_api_key_empty_string(self):
        hc = HFAPIConfig(model="gpt2", api_key="")
        assert hc.api_key == ""

    def test_large_timeout(self):
        hc = HFAPIConfig(model="gpt2", timeout=999999)
        assert hc.timeout == 999999

    def test_large_max_new_tokens(self):
        hc = HFAPIConfig(model="gpt2", max_new_tokens=100000)
        assert hc.max_new_tokens == 100000

    def test_copy_semantics(self):
        import dataclasses
        a = HFAPIConfig(model="gpt2", temperature=0.7)
        b = dataclasses.replace(a, temperature=0.1)
        assert a.temperature == 0.7
        assert b.temperature == 0.1
        assert a.model == b.model

    def test_copy_preserves_api_key(self):
        import dataclasses
        a = HFAPIConfig(model="gpt2", api_key="secret")
        b = dataclasses.replace(a, model="llama")
        assert b.api_key == "secret"
        assert b.model == "llama"


class TestHFAPIConfigLoaderInteraction:
    def test_loader_stores_config(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        assert loader.config is cfg

    def test_loader_api_key_from_config(self):
        cfg = HFAPIConfig(model="gpt2", api_key="key123")
        loader = HuggingFaceAPILoader(cfg)
        assert loader.api_key == "key123"

    def test_loader_api_key_none_when_not_set(self):
        import os
        old_key = os.environ.pop("HF_API_KEY", None)
        old_token = os.environ.pop("HF_TOKEN", None)
        try:
            cfg = HFAPIConfig(model="gpt2")
            loader = HuggingFaceAPILoader(cfg)
            assert loader.api_key is None
        finally:
            if old_key:
                os.environ["HF_API_KEY"] = old_key
            if old_token:
                os.environ["HF_TOKEN"] = old_token

    def test_loader_headers_with_api_key(self):
        cfg = HFAPIConfig(model="gpt2", api_key="mykey")
        loader = HuggingFaceAPILoader(cfg)
        assert loader.headers["Authorization"] == "Bearer mykey"

    def test_loader_headers_empty_without_key(self):
        import os
        old_key = os.environ.pop("HF_API_KEY", None)
        old_token = os.environ.pop("HF_TOKEN", None)
        try:
            cfg = HFAPIConfig(model="gpt2")
            loader = HuggingFaceAPILoader(cfg)
            assert "Authorization" not in loader.headers
        finally:
            if old_key:
                os.environ["HF_API_KEY"] = old_key
            if old_token:
                os.environ["HF_TOKEN"] = old_token

    def test_hf_inference_client_is_subclass(self):
        assert issubclass(HFInferenceClient, HuggingFaceAPILoader)

    def test_create_api_client_returns_loader(self):
        client = create_api_client("gpt2")
        assert isinstance(client, HuggingFaceAPILoader)

    def test_create_api_client_with_api_key(self):
        client = create_api_client("gpt2", api_key="test")
        assert client.api_key == "test"

    def test_create_api_client_with_temperature(self):
        client = create_api_client("gpt2", temperature=0.1)
        assert client.config.temperature == 0.1


class TestHFAPIConfigFormatChatPrompt:
    def test_single_user_message(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([{"role": "user", "content": "hello"}])
        assert result == "User: hello\nAssistant:"

    def test_single_assistant_message(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([{"role": "assistant", "content": "hi"}])
        assert result == "Assistant: hi\nAssistant:"

    def test_single_system_message(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([{"role": "system", "content": "be polite"}])
        assert result == "System: be polite\nAssistant:"

    def test_multi_turn_conversation(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you"},
        ]
        result = loader._format_chat_prompt(messages)
        assert result == "User: hi\nAssistant: hello\nUser: how are you\nAssistant:"

    def test_unknown_role_dropped(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([{"role": "custom", "content": "test"}])
        assert result == "Assistant:"

    def test_missing_role_defaults_to_user(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([{"content": "hello"}])
        assert "User: hello" in result

    def test_missing_content_defaults_to_empty(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([{"role": "user"}])
        assert "User: \n" in result

    def test_empty_messages(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([])
        assert result == "Assistant:"

    def test_system_user_assistant_flow(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
        ]
        result = loader._format_chat_prompt(messages)
        assert "System: You are helpful" in result
        assert "User: hi" in result
        assert "Assistant: hello!" in result
        assert result.endswith("Assistant:")

    def test_content_with_newlines(self):
        cfg = HFAPIConfig(model="gpt2")
        loader = HuggingFaceAPILoader(cfg)
        result = loader._format_chat_prompt([{"role": "user", "content": "line1\nline2"}])
        assert "User: line1\nline2" in result
