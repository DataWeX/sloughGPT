"""Tests for training/huggingface/client.py — HFClient, get_model_memory, list_models."""

import pytest
from unittest.mock import MagicMock, patch
from domains.training.huggingface.client import (
    HFClient, get_model_memory, list_models,
)


class TestHFClient:
    def test_api_mode_init(self):
        with patch("domains.training.huggingface.client._APILoader") as MockLoader, \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("meta-llama/Llama-2-7b-chat-hf", mode="api")
            assert client.model == "meta-llama/Llama-2-7b-chat-hf"
            assert client.mode == "api"
            MockLoader.assert_called_once()

    def test_local_mode_init(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader") as MockLoader:
            mock_instance = MagicMock()
            MockLoader.return_value = mock_instance
            client = HFClient("gpt2", mode="local")
            assert client.mode == "local"
            mock_instance.load.assert_called_once()

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            HFClient("gpt2", mode="gpu")

    def test_call_delegates_to_generate(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            client._client = MagicMock()
            client._client.generate.return_value = "hello"
            result = client("hi")
            assert result == "hello"
            client._client.generate.assert_called_once_with("hi")

    def test_chat_delegates_to_chat(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            client._client = MagicMock()
            client._client.chat.return_value = "response"
            messages = [{"role": "user", "content": "hi"}]
            result = client.chat(messages)
            assert result == "response"
            client._client.chat.assert_called_once_with(messages)

    def test_repr(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            r = repr(client)
            assert "gpt2" in r
            assert "api" in r

    def test_api_mode_stores_model(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("my-model", mode="api")
            assert client.model == "my-model"

    def test_api_mode_stores_mode(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            assert client.mode == "api"

    def test_local_mode_stores_model(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader") as MockLoader:
            MockLoader.return_value = MagicMock()
            client = HFClient("mistral-7b", mode="local")
            assert client.model == "mistral-7b"

    def test_call_returns_string(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            client._client = MagicMock()
            client._client.generate.return_value = "output text"
            result = client("prompt")
            assert isinstance(result, str)

    def test_chat_returns_string(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            client._client = MagicMock()
            client._client.chat.return_value = "chat response"
            result = client.chat([{"role": "user", "content": "hi"}])
            assert isinstance(result, str)

    def test_multiple_calls(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            client._client = MagicMock()
            client._client.generate.side_effect = ["first", "second"]
            assert client("a") == "first"
            assert client("b") == "second"

    def test_repr_contains_quotes(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("model-name", mode="api")
            r = repr(client)
            assert "'" in r

    def test_mode_api_only(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            assert client.mode in ("api", "local")

    def test_different_models(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            for model in ["gpt2", "llama-7b", "mistral-7b"]:
                client = HFClient(model, mode="api")
                assert client.model == model

    def test_api_mode_kwarg_passthrough(self):
        with patch("domains.training.huggingface.client._APILoader") as MockLoader, \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api", temperature=0.7)
            MockLoader.assert_called_once()

    def test_call_with_kwargs(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            client._client = MagicMock()
            client._client.generate.return_value = "result"
            result = client("prompt", max_new_tokens=100)
            assert result == "result"

    def test_chat_with_kwargs(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            client._client = MagicMock()
            client._client.chat.return_value = "reply"
            result = client.chat([{"role": "user", "content": "hi"}], temperature=0.5)
            assert result == "reply"

    def test_api_loader_receives_config(self):
        with patch("domains.training.huggingface.client._APILoader") as MockLoader, \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            call_args = MockLoader.call_args
            config = call_args[0][0]
            assert config.model == "gpt2"

    def test_repr_format(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("model-x", mode="local")
            r = repr(client)
            assert r == "HFClient(model='model-x', mode='local')"

    def test_empty_model_name(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("", mode="api")
            assert client.model == ""

    def test_special_chars_model_name(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("org/model-name_v2.0", mode="api")
            assert client.model == "org/model-name_v2.0"

    def test_call_with_empty_prompt(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            client._client = MagicMock()
            client._client.generate.return_value = ""
            result = client("")
            assert result == ""

    def test_chat_with_empty_messages(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            client._client = MagicMock()
            client._client.chat.return_value = "ok"
            result = client.chat([])
            assert result == "ok"

    def test_chat_with_multiple_messages(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            client._client = MagicMock()
            client._client.chat.return_value = "reply"
            msgs = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hi"},
            ]
            result = client.chat(msgs)
            assert result == "reply"

    def test_call_preserves_client_reference(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("gpt2", mode="api")
            original = client._client
            client._client = MagicMock()
            client._client.generate.return_value = "x"
            client("test")
            assert client._client is not original

    def test_repr_model_only(self):
        with patch("domains.training.huggingface.client._APILoader"), \
             patch("domains.training.huggingface.client._LocalLoader"):
            client = HFClient("test-model", mode="api")
            assert "HFClient" in repr(client)

    def test_unknown_mode_message(self):
        with pytest.raises(ValueError, match="api.*local"):
            HFClient("gpt2", mode="invalid")


class TestGetModelMemory:
    def test_returns_dict(self):
        with patch("domains.training.huggingface.client.get_model_requirements") as mock_req:
            mock_req.return_value = {"memory_gb": 4.0}
            result = get_model_memory("gpt2")
            assert result == {"memory_gb": 4.0}
            mock_req.assert_called_once_with("gpt2")

    def test_passes_precision(self):
        with patch("domains.training.huggingface.client.get_model_requirements") as mock_req:
            mock_req.return_value = {"memory_gb": 2.0}
            result = get_model_memory("gpt2", precision="int8")
            mock_req.assert_called_once_with("gpt2")

    def test_different_models(self):
        with patch("domains.training.huggingface.client.get_model_requirements") as mock_req:
            mock_req.return_value = {"memory_gb": 8.0}
            result = get_model_memory("llama-7b")
            mock_req.assert_called_once_with("llama-7b")

    def test_return_type(self):
        with patch("domains.training.huggingface.client.get_model_requirements") as mock_req:
            mock_req.return_value = {"memory_gb": 4.0, "params": "7b"}
            result = get_model_memory("gpt2")
            assert isinstance(result, dict)

    def test_memory_gb_value(self):
        with patch("domains.training.huggingface.client.get_model_requirements") as mock_req:
            mock_req.return_value = {"memory_gb": 16.0}
            result = get_model_memory("gpt2")
            assert result["memory_gb"] == 16.0

    def test_empty_dict_return(self):
        with patch("domains.training.huggingface.client.get_model_requirements") as mock_req:
            mock_req.return_value = {}
            result = get_model_memory("unknown-model")
            assert result == {}

    def test_multiple_keys(self):
        with patch("domains.training.huggingface.client.get_model_requirements") as mock_req:
            mock_req.return_value = {"memory_gb": 4.0, "params": "7b", "layers": 32}
            result = get_model_memory("gpt2")
            assert "params" in result
            assert "layers" in result

    def test_precision_ignored_in_delegation(self):
        with patch("domains.training.huggingface.client.get_model_requirements") as mock_req:
            mock_req.return_value = {"memory_gb": 1.0}
            get_model_memory("gpt2", precision="fp32")
            get_model_memory("gpt2", precision="bf16")
            assert mock_req.call_count == 2


class TestListModels:
    def test_list_all(self):
        mock_result1 = MagicMock()
        mock_result1.model_id = "gpt2"
        mock_result2 = MagicMock()
        mock_result2.model_id = "gpt2-medium"
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize"):
            mock_search.return_value = [mock_result1, mock_result2]
            result = list_models()
            assert result == ["gpt2", "gpt2-medium"]

    def test_list_with_org_filter(self):
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize"):
            mock_search.return_value = []
            result = list_models(organization="meta")
            mock_search.assert_called_once_with(organization="meta", size=None)

    def test_list_with_size_filter(self):
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize") as MockSize:
            MockSize.__getitem__ = MagicMock(return_value="SMALL")
            mock_search.return_value = []
            result = list_models(size="small")
            mock_search.assert_called_once()

    def test_list_empty(self):
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize"):
            mock_search.return_value = []
            result = list_models()
            assert result == []

    def test_list_returns_list(self):
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize"):
            mock_search.return_value = []
            result = list_models()
            assert isinstance(result, list)

    def test_list_with_both_filters(self):
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize") as MockSize:
            MockSize.__getitem__ = MagicMock(return_value="MEDIUM")
            mock_search.return_value = []
            result = list_models(organization="openai", size="medium")
            mock_search.assert_called_once()

    def test_list_extracts_model_ids(self):
        mock_result = MagicMock()
        mock_result.model_id = "test-model"
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize"):
            mock_search.return_value = [mock_result]
            result = list_models()
            assert result == ["test-model"]

    def test_list_none_filters(self):
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize"):
            mock_search.return_value = []
            list_models(organization=None, size=None)
            mock_search.assert_called_once_with(organization=None, size=None)

    def test_list_no_size_no_org(self):
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize"):
            mock_search.return_value = []
            list_models()
            mock_search.assert_called_once_with(organization=None, size=None)

    def test_list_preserves_order(self):
        mock_results = [MagicMock(model_id=f"model-{i}") for i in range(5)]
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize"):
            mock_search.return_value = mock_results
            result = list_models()
            assert result == [f"model-{i}" for i in range(5)]

    def test_list_with_large_org(self):
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize"):
            mock_search.return_value = []
            list_models(organization="meta-llama")
            mock_search.assert_called_once_with(organization="meta-llama", size=None)

    def test_list_size_large(self):
        with patch("domains.training.huggingface.model_map.search_models") as mock_search, \
             patch("domains.training.huggingface.model_map.ModelSize") as MockSize:
            MockSize.__getitem__ = MagicMock(return_value="LARGE")
            mock_search.return_value = []
            list_models(size="large")
            mock_search.assert_called_once()


class TestConvenienceFunctions:
    def test_generate_function(self):
        with patch("domains.training.huggingface.client.HFClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.return_value = "generated text"
            MockClient.return_value = mock_instance
            from domains.training.huggingface.client import generate
            result = generate("prompt", model="gpt2")
            MockClient.assert_called_once_with("gpt2", mode="api")
            mock_instance.assert_called_once_with("prompt")

    def test_chat_function(self):
        with patch("domains.training.huggingface.client.HFClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = "chat reply"
            MockClient.return_value = mock_instance
            from domains.training.huggingface.client import chat
            msgs = [{"role": "user", "content": "hi"}]
            result = chat(msgs, model="gpt2")
            MockClient.assert_called_once_with("gpt2", mode="api")
            mock_instance.chat.assert_called_once_with(msgs)

    def test_generate_default_model(self):
        with patch("domains.training.huggingface.client.HFClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.return_value = "output"
            MockClient.return_value = mock_instance
            from domains.training.huggingface.client import generate
            generate("prompt")
            MockClient.assert_called_once_with(
                "meta-llama/Llama-2-7b-chat-hf", mode="api"
            )

    def test_chat_default_model(self):
        with patch("domains.training.huggingface.client.HFClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = "reply"
            MockClient.return_value = mock_instance
            from domains.training.huggingface.client import chat
            chat([{"role": "user", "content": "hi"}])
            MockClient.assert_called_once_with(
                "meta-llama/Llama-2-7b-chat-hf", mode="api"
            )

    def test_generate_returns_string(self):
        with patch("domains.training.huggingface.client.HFClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.return_value = "text output"
            MockClient.return_value = mock_instance
            from domains.training.huggingface.client import generate
            result = generate("prompt")
            assert isinstance(result, str)

    def test_chat_returns_string(self):
        with patch("domains.training.huggingface.client.HFClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = "chat output"
            MockClient.return_value = mock_instance
            from domains.training.huggingface.client import chat
            result = chat([{"role": "user", "content": "hi"}])
            assert isinstance(result, str)
