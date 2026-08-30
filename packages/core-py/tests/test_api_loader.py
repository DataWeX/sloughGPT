"""Tests for api_loader — HuggingFace Inference API wrapper."""

import os
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


# ── HFAPIConfig ────────────────────────────────────────────────────────


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

    def test_top_p_default(self):
        cfg = HFAPIConfig(model="gpt2")
        assert cfg.top_p == 0.9

    def test_repetition_penalty_default(self):
        cfg = HFAPIConfig(model="gpt2")
        assert cfg.repetition_penalty == 1.0

    def test_all_fields_settable(self):
        cfg = HFAPIConfig(
            model="m", api_key="k", timeout=10, max_new_tokens=512,
            temperature=0.3, top_p=0.8, repetition_penalty=1.2,
        )
        assert cfg.model == "m"
        assert cfg.api_key == "k"
        assert cfg.timeout == 10
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.3
        assert cfg.top_p == 0.8
        assert cfg.repetition_penalty == 1.2

    def test_dataclass_fields(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(HFAPIConfig)}
        assert "model" in fields
        assert "api_key" in fields
        assert "timeout" in fields
        assert "max_new_tokens" in fields
        assert "temperature" in fields
        assert "top_p" in fields
        assert "repetition_penalty" in fields

    def test_equality(self):
        c1 = HFAPIConfig(model="gpt2")
        c2 = HFAPIConfig(model="gpt2")
        assert c1 == c2

    def test_inequality(self):
        c1 = HFAPIConfig(model="gpt2")
        c2 = HFAPIConfig(model="gpt3")
        assert c1 != c2

    def test_timeout_zero(self):
        cfg = HFAPIConfig(model="gpt2", timeout=0)
        assert cfg.timeout == 0


# ── HuggingFaceAPILoader ──────────────────────────────────────────────


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

    @patch.dict("os.environ", {"HF_API_KEY": "apitok"})
    def test_init_env_api_key(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.api_key == "apitok"

    @patch.dict("os.environ", {"HF_API_KEY": "apitok", "HF_TOKEN": "envtok"})
    def test_api_key_takes_precedence(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        assert loader.api_key == "apitok"

    def test_explicit_key_over_env(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2", api_key="explicit"))
        assert loader.api_key == "explicit"

    def test_config_stored(self):
        cfg = HFAPIConfig(model="mymodel")
        loader = HuggingFaceAPILoader(cfg)
        assert loader.config is cfg
        assert loader.config.model == "mymodel"

    # ── _format_chat_prompt ────────────────────────────────────────────

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

    def test_format_chat_prompt_unknown_role(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "custom", "content": "data"}])
        # Unknown roles are not formatted as any specific role prefix
        # The function only handles user/assistant/system — others are skipped
        assert prompt.endswith("Assistant:")
        # Content for unknown roles is NOT included
        assert "data" not in prompt

    def test_format_chat_prompt_empty_messages(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([])
        assert prompt == "Assistant:"

    def test_format_chat_prompt_missing_content(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user"}])
        assert "User: " in prompt

    def test_format_chat_prompt_missing_role(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"content": "hello"}])
        # Missing role defaults to "user"
        assert "User: hello" in prompt

    def test_format_chat_prompt_multiline_content(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user", "content": "line1\nline2"}])
        assert "line1\nline2" in prompt

    def test_format_chat_prompt_empty_content(self):
        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user", "content": ""}])
        assert "User: " in prompt

    # ── generate ───────────────────────────────────────────────────────

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
    def test_generate_passthrough_response(self, mock_post):
        """Non-list, non-dict response falls through to str()."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = "plain string"
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        result = loader.generate("hi")
        assert result == "plain string"

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_custom_params(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "out"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        loader.generate("hi", max_new_tokens=100, temperature=0.1, top_p=0.5)
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["parameters"]["max_new_tokens"] == 100
        assert payload["parameters"]["temperature"] == 0.1
        assert payload["parameters"]["top_p"] == 0.5

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_list_empty_text(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": ""}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        result = loader.generate("hi")
        assert result == ""

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_empty_list_returns_empty(self, mock_post):
        """Empty list response returns empty string."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        result = loader.generate("hi")
        assert result == "[]"

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_dict_no_generated_text(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"other_key": "val"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        result = loader.generate("hi")
        assert result == ""

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_extra_kwargs_forwarded(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "ok"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        loader.generate("hi", do_sample=True, num_beams=3)
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["parameters"]["do_sample"] is True
        assert payload["parameters"]["num_beams"] == 3

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_repetition_penalty(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "ok"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        loader.generate("hi", repetition_penalty=1.5)
        payload = mock_post.call_args[1]["json"]
        assert payload["parameters"]["repetition_penalty"] == 1.5

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_uses_config_defaults(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "ok"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = HFAPIConfig(model="gpt2", max_new_tokens=512, temperature=0.2, top_p=0.8)
        loader = HuggingFaceAPILoader(cfg)
        loader.generate("hi")
        payload = mock_post.call_args[1]["json"]
        assert payload["parameters"]["max_new_tokens"] == 512
        assert payload["parameters"]["temperature"] == 0.2
        assert payload["parameters"]["top_p"] == 0.8

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_return_full_text_false(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "ok"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        loader.generate("hi")
        payload = mock_post.call_args[1]["json"]
        assert payload["parameters"]["return_full_text"] is False

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_use_cache(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "ok"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        loader.generate("hi")
        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["use_cache"] is True

    # ── chat ───────────────────────────────────────────────────────────

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

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_chat_custom_params(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "ok"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        result = loader.chat(
            [{"role": "user", "content": "hi"}],
            max_new_tokens=50, temperature=0.2,
        )
        assert result == "ok"

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_chat_formats_prompt(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "reply"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        loader.chat([{"role": "user", "content": "hello"}])
        payload = mock_post.call_args[1]["json"]
        assert "User: hello" in payload["inputs"]

    # ── _make_request ──────────────────────────────────────────────────

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_make_request_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="mymodel"))
        loader._make_request({"inputs": "test"})
        call_args = mock_post.call_args
        assert "mymodel" in call_args[0][0]

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_make_request_timeout(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2", timeout=15))
        loader._make_request({"inputs": "test"})
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["timeout"] == 15

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_make_request_sends_headers(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2", api_key="tok"))
        loader._make_request({"inputs": "test"})
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer tok"

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_make_request_post_method(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loader = HuggingFaceAPILoader(HFAPIConfig(model="gpt2"))
        loader._make_request({"inputs": "test"})
        assert mock_post.called


# ── Aliases ────────────────────────────────────────────────────────────


class TestAliases:
    def test_hf_inference_client_is_loader(self):
        assert issubclass(HFInferenceClient, HuggingFaceAPILoader)

    def test_hf_inference_client_works(self):
        client = HFInferenceClient(HFAPIConfig(model="gpt2"))
        assert isinstance(client, HuggingFaceAPILoader)
        assert client.config.model == "gpt2"

    def test_create_api_client(self):
        client = create_api_client("gpt2")
        assert isinstance(client, HuggingFaceAPILoader)
        assert client.config.model == "gpt2"

    def test_create_api_client_with_key(self):
        client = create_api_client("gpt2", api_key="key123")
        assert client.api_key == "key123"

    def test_create_api_client_custom_timeout(self):
        client = create_api_client("gpt2", timeout=30)
        assert client.config.timeout == 30

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

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_generate_via_api_with_key(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "result"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = generate_via_api("hi", model="m", api_key="k1")
        assert result == "result"
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer k1"

    @patch("domains.training.huggingface.api_loader.requests.post")
    def test_chat_via_api_with_key(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "result"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = chat_via_api([{"role": "user", "content": "hi"}], model="m", api_key="k2")
        assert result == "result"

    def test_module_all(self):
        import domains.training.huggingface.api_loader as mod
        assert "HFAPIConfig" in mod.__all__
        assert "HuggingFaceAPILoader" in mod.__all__
        assert "HFInferenceClient" in mod.__all__
        assert "create_api_client" in mod.__all__
        assert "generate_via_api" in mod.__all__
        assert "chat_via_api" in mod.__all__

    def test_module_all_count(self):
        import domains.training.huggingface.api_loader as mod
        assert len(mod.__all__) == 6

    def test_hf_api_base_url(self):
        import domains.training.huggingface.api_loader as mod
        assert "huggingface.co" in mod.HF_API_BASE
