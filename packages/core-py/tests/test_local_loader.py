"""Tests for local_loader — HuggingFace local model loading."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from domains.training.huggingface.local_loader import (
    HFLocalConfig,
    HuggingFaceLocalLoader,
    HuggingFaceLocalClient,
)


class TestHFLocalConfig:
    def test_defaults(self):
        cfg = HFLocalConfig(model="gpt2")
        assert cfg.model == "gpt2"
        assert cfg.device == "auto"
        assert cfg.dtype == "auto"
        assert cfg.load_in_8bit is False
        assert cfg.load_in_4bit is False
        assert cfg.local_files_only is True

    def test_custom(self):
        cfg = HFLocalConfig(model="gpt2", device="cpu", dtype="float32", load_in_8bit=True)
        assert cfg.device == "cpu"
        assert cfg.dtype == "float32"
        assert cfg.load_in_8bit is True


class TestHuggingFaceLocalLoader:
    def test_init_auto_device(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        assert loader.config.device == "cpu"

    def test_init_explicit_device(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", device="cuda"))
        assert loader.config.device == "cuda"

    def test_get_dtype_auto(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="auto"))
        assert loader._get_dtype() == "auto"

    def test_get_dtype_float32(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="float32"))
        assert loader._get_dtype() == "float32"

    def test_get_dtype_float16(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="float16"))
        assert loader._get_dtype() == "float16"

    def test_get_dtype_half(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="half"))
        assert loader._get_dtype() == "float16"

    def test_get_dtype_bfloat16(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="bfloat16"))
        assert loader._get_dtype() == "bfloat16"

    def test_get_dtype_unknown_falls_back(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="bogus"))
        assert loader._get_dtype() == "float32"

    def test_init_state(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        assert loader.model is None
        assert loader.tokenizer is None

    def test_generate_without_model_raises(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        with pytest.raises(RuntimeError, match="not loaded"):
            loader.generate("hello")

    def test_format_chat_prompt_user(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user", "content": "hi"}])
        assert "User: hi" in prompt
        assert prompt.endswith("Assistant:")

    def test_format_chat_prompt_multi(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        prompt = loader._format_chat_prompt(msgs)
        assert "System: sys" in prompt
        assert "User: q" in prompt
        assert "Assistant: a" in prompt

    def test_unload(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.model = MagicMock()
        loader.tokenizer = MagicMock()
        loader.unload()
        assert loader.model is None
        assert loader.tokenizer is None

    def test_load_without_transformers_raises(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", None):
            loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
            with pytest.raises(ImportError, match="transformers"):
                loader.load()


class TestAliases:
    def test_client_is_loader(self):
        assert issubclass(HuggingFaceLocalClient, HuggingFaceLocalLoader)
