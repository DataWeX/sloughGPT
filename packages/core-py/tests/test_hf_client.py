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


class TestGetModelMemory:
    def test_returns_dict(self):
        with patch("domains.training.huggingface.client.get_model_requirements") as mock_req:
            mock_req.return_value = {"memory_gb": 4.0}
            result = get_model_memory("gpt2")
            assert result == {"memory_gb": 4.0}
            mock_req.assert_called_once_with("gpt2")


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
