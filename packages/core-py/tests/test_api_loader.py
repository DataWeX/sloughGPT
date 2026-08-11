"""Tests for api_loader — HuggingFace Inference API wrapper."""

import pytest
from unittest.mock import patch, MagicMock
from domains.training.huggingface.api_loader import (
    HFAPIConfig,
    HuggingFaceAPILoader,
    HFInferenceClient,
    create_api_client,
    generate_via_api,
    chat_via_api,
)


class TestHFAPIConfig:
    def test_defaults(self):
        cfg = HFAPIConfig(model="gpt2")
        assert cfg.model == "gpt2"
        assert cfg.api_key is None
        assert cfg.timeout == 60
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.7

    def test_custom(self):
        cfg = HFAPIConfig(model="gpt2", api_key="tok", timeout=30, temperature=0.5)
        assert cfg.api_key == "tok"
        assert cfg.timeout == 30
        assert cfg.temperature == 0.5


class TestHuggingFaceAPILoader:
    def test_init_no_key(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.api_key is None
        assert "Authorization" not in loader.headers

    def test_init_with_key(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2", api_key="tok123"))
        assert loader.api_key == "tok123"
        assert loader.headers["Authorization"] == "Bearer tok123"

    @patch.dict("os.environ", {"HF_TOKEN": "envtok"})
    def test_init_env_token(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.api_key == "envtok"

    def test_format_chat_prompt_user(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user", "content": "hi"}])
        assert "User: hi" in prompt
        assert prompt.endswith("Assistant:")

    def test_format_chat_prompt_assistant(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "assistant", "content": "hello"}])
        assert "Assistant: hello" in prompt

    def test_format_chat_prompt_system(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "system", "content": "be nice"}])
        assert "System: be nice" in prompt

    def test_format_chat_prompt_multi(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        prompt = loader._format_chat_prompt(msgs)
        assert "System: sys" in prompt
        assert "User: q" in prompt
        assert "Assistant: a" in prompt
        assert prompt.endswith("Assistant:")

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_list_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "hello world"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        result = loader.generate("hi")
        assert result == "hello world"
        mock_post.assert_called_once()

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_dict_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"generated_text": "test output"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        result = loader.generate("hi")
        assert result == "test output"

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_503_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        with pytest.raises(RuntimeError, match="loading"):
            loader.generate("hi")

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_401_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        with pytest.raises(RuntimeError, match="token"):
            loader.generate("hi")

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_chat_calls_generate(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "chat reply"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        result = loader.chat([{"role": "user", "content": "hi"}])
        assert result == "chat reply"


class TestAliases:
    def test_hf_inference_client_is_loader(self):
        assert issubclass(HFInferenceClient, HuggingFaceAPILoader)

    def test_create_api_client(self):
        client = create_api_client("gpt2")
        assert isinstance(client, HuggingFaceAPILoader)
        assert client.config.model == "gpt2"

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_via_api(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "via_api"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = generate_via_api("hi", model="gpt2")
        assert result == "via_api"

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_chat_via_api(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "chat_via"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = chat_via_api([{"role": "user", "content": "hi"}], model="gpt2")
        assert result == "chat_via"
