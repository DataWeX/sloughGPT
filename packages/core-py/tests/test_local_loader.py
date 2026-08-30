"""Tests for local_loader — HuggingFace local model loading.

Comprehensive coverage of config, device detection, dtype resolution,
chat prompt formatting, error paths, and module-level functions.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from domains.training.huggingface.local_loader import (
    HFLocalConfig,
    HuggingFaceLocalLoader,
    HuggingFaceLocalClient,
    download_model,
    load_model,
    generate_local,
)


# ── HFLocalConfig ────────────────────────────────────────────────────────────

class TestHFLocalConfig:
    def test_defaults(self):
        cfg = HFLocalConfig(model="gpt2")
        assert cfg.model == "gpt2"
        assert cfg.device == "auto"
        assert cfg.dtype == "auto"
        assert cfg.load_in_8bit is False
        assert cfg.load_in_4bit is False
        assert cfg.local_files_only is True
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.repetition_penalty == 1.0
        assert cfg.cache_dir is None

    def test_custom_device(self):
        cfg = HFLocalConfig(model="gpt2", device="cuda")
        assert cfg.device == "cuda"

    def test_custom_dtype(self):
        cfg = HFLocalConfig(model="gpt2", dtype="float32")
        assert cfg.dtype == "float32"

    def test_load_in_8bit(self):
        cfg = HFLocalConfig(model="gpt2", load_in_8bit=True)
        assert cfg.load_in_8bit is True

    def test_load_in_4bit(self):
        cfg = HFLocalConfig(model="gpt2", load_in_4bit=True)
        assert cfg.load_in_4bit is True

    def test_custom_cache_dir(self):
        cfg = HFLocalConfig(model="gpt2", cache_dir="/tmp/cache")
        assert cfg.cache_dir == "/tmp/cache"

    def test_custom_max_new_tokens(self):
        cfg = HFLocalConfig(model="gpt2", max_new_tokens=512)
        assert cfg.max_new_tokens == 512

    def test_custom_temperature(self):
        cfg = HFLocalConfig(model="gpt2", temperature=0.1)
        assert cfg.temperature == 0.1

    def test_custom_top_p(self):
        cfg = HFLocalConfig(model="gpt2", top_p=0.95)
        assert cfg.top_p == 0.95

    def test_custom_repetition_penalty(self):
        cfg = HFLocalConfig(model="gpt2", repetition_penalty=1.2)
        assert cfg.repetition_penalty == 1.2

    def test_local_files_only_false(self):
        cfg = HFLocalConfig(model="gpt2", local_files_only=False)
        assert cfg.local_files_only is False

    def test_all_fields_settable(self):
        cfg = HFLocalConfig(
            model="custom",
            device="cuda",
            dtype="float16",
            load_in_8bit=True,
            load_in_4bit=False,
            cache_dir="/tmp",
            local_files_only=False,
            max_new_tokens=128,
            temperature=0.5,
            top_p=0.8,
            repetition_penalty=1.1,
        )
        assert cfg.model == "custom"
        assert cfg.device == "cuda"
        assert cfg.dtype == "float16"
        assert cfg.load_in_8bit is True
        assert cfg.load_in_4bit is False
        assert cfg.cache_dir == "/tmp"
        assert cfg.local_files_only is False
        assert cfg.max_new_tokens == 128
        assert cfg.temperature == 0.5
        assert cfg.top_p == 0.8
        assert cfg.repetition_penalty == 1.1


# ── Device detection ─────────────────────────────────────────────────────────

class TestDeviceDetection:
    def test_auto_device(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        assert loader.config.device == "cpu"

    def test_explicit_device_preserved(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", device="cuda"))
        assert loader.config.device == "cuda"

    def test_explicit_cpu(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", device="cpu"))
        assert loader.config.device == "cpu"


# ── Dtype resolution ─────────────────────────────────────────────────────────

class TestGetDtype:
    def test_auto(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="auto"))
        assert loader._get_dtype() == "auto"

    def test_float32(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="float32"))
        assert loader._get_dtype() == "float32"

    def test_float16(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="float16"))
        assert loader._get_dtype() == "float16"

    def test_half(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="half"))
        assert loader._get_dtype() == "float16"

    def test_bfloat16(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="bfloat16"))
        assert loader._get_dtype() == "bfloat16"

    def test_unknown_falls_back(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="bogus"))
        assert loader._get_dtype() == "float32"

    def test_int64_falls_back(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="int64"))
        assert loader._get_dtype() == "float32"

    def test_empty_string_falls_back(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype=""))
        assert loader._get_dtype() == "float32"


# ── Initial state ────────────────────────────────────────────────────────────

class TestInitialState:
    def test_model_is_none(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        assert loader.model is None

    def test_tokenizer_is_none(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        assert loader.tokenizer is None

    def test_config_stored(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.config is cfg


# ── Generate without model ───────────────────────────────────────────────────

class TestGenerateWithoutModel:
    def test_generate_raises(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        with pytest.raises(RuntimeError, match="not loaded"):
            loader.generate("hello")

    def test_generate_with_params_raises(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        with pytest.raises(RuntimeError, match="not loaded"):
            loader.generate("hello", max_new_tokens=10, temperature=0.5)


# ── Chat prompt formatting ───────────────────────────────────────────────────

class TestFormatChatPrompt:
    def test_single_user(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user", "content": "hi"}])
        assert "User: hi" in prompt
        assert prompt.endswith("Assistant:")

    def test_multi_turn(self):
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

    def test_empty_messages(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([])
        assert prompt == "Assistant:"

    def test_unknown_role(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "unknown", "content": "x"}])
        assert "Assistant:" in prompt

    def test_missing_role_key(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"content": "hi"}])
        # Defaults to "user"
        assert "User: hi" in prompt

    def test_missing_content_key(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user"}])
        assert "User: " in prompt

    def test_empty_content(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user", "content": ""}])
        assert "User: " in prompt

    def test_consecutive_assistant(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        msgs = [
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ]
        prompt = loader._format_chat_prompt(msgs)
        assert "Assistant: a1" in prompt
        assert "Assistant: a2" in prompt

    def test_unicode_content(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user", "content": "こんにちは"}])
        assert "User: こんにちは" in prompt

    def test_multiline_content(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user", "content": "line1\nline2"}])
        assert "User: line1\nline2" in prompt


# ── Unload ───────────────────────────────────────────────────────────────────

class TestUnload:
    def test_unload_clears_model(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.model = MagicMock()
        loader.tokenizer = MagicMock()
        loader.unload()
        assert loader.model is None

    def test_unload_clears_tokenizer(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.model = MagicMock()
        loader.tokenizer = MagicMock()
        loader.unload()
        assert loader.tokenizer is None

    def test_unload_when_already_none(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.unload()
        assert loader.model is None
        assert loader.tokenizer is None

    def test_unload_idempotent(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.unload()
        loader.unload()
        assert loader.model is None


# ── Load without transformers ────────────────────────────────────────────────

class TestLoadWithoutTransformers:
    def test_load_raises_import_error(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", None):
            loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
            with pytest.raises(ImportError, match="transformers"):
                loader.load()

    def test_load_raises_when_model_none(self):
        with patch("domains.training.huggingface.local_loader.AutoModelForCausalLM", None):
            loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
            with pytest.raises(ImportError, match="transformers"):
                loader.load()


# ── Aliases ──────────────────────────────────────────────────────────────────

class TestAliases:
    def test_client_is_loader(self):
        assert issubclass(HuggingFaceLocalClient, HuggingFaceLocalLoader)

    def test_client_same_name(self):
        assert HuggingFaceLocalClient.__name__ == "HuggingFaceLocalClient"


# ── __all__ exports ──────────────────────────────────────────────────────────

class TestExports:
    def test_all_exports(self):
        from domains.training.huggingface import local_loader
        for name in local_loader.__all__:
            assert hasattr(local_loader, name), f"Missing export: {name}"

    def test_expected_exports(self):
        from domains.training.huggingface import local_loader
        expected = {
            "HFLocalConfig",
            "HuggingFaceLocalLoader",
            "HuggingFaceLocalClient",
            "download_model",
            "load_model",
            "generate_local",
        }
        assert set(local_loader.__all__) == expected


# ── Module-level functions (require transformers mock) ───────────────────────

class TestDownloadModel:
    def test_download_raises_without_transformers(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", None):
            with pytest.raises(AttributeError):
                download_model("gpt2")

    def test_download_custom_cache(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", None):
            with pytest.raises(AttributeError):
                download_model("gpt2", cache_dir="/tmp/cache")


class TestLoadModel:
    def test_load_model_raises_without_transformers(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", None):
            cfg = HFLocalConfig(model="gpt2")
            with pytest.raises(ImportError):
                load_model(cfg)


class TestGenerateLocal:
    def test_generate_local_raises_without_transformers(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", None):
            with pytest.raises(ImportError):
                generate_local("hello", model="gpt2")


# ── Generate with mocked model ───────────────────────────────────────────────

class TestGenerateWithMock:
    def test_generate_calls_model(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": "toks"}
        mock_model.generate.return_value = "generated_ids"
        mock_tokenizer.decode.return_value = "generated text"
        loader.model = mock_model
        loader.tokenizer = mock_tokenizer
        result = loader.generate("hello")
        assert result == "generated text"
        mock_model.generate.assert_called_once()

    def test_generate_cpu_path(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", device="cpu"))
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": "toks"}
        mock_model.generate.return_value = "ids"
        mock_tokenizer.decode.return_value = "output"
        loader.model = mock_model
        loader.tokenizer = mock_tokenizer
        result = loader.generate("hello", max_new_tokens=10, temperature=0.5)
        assert result == "output"

    def test_generate_with_kwargs(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": "toks"}
        mock_model.generate.return_value = "ids"
        mock_tokenizer.decode.return_value = "out"
        loader.model = mock_model
        loader.tokenizer = mock_tokenizer
        result = loader.generate("hi", max_new_tokens=50, temperature=0.3, top_p=0.8)
        assert result == "out"

    def test_chat_calls_generate(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": "toks"}
        mock_model.generate.return_value = "ids"
        mock_tokenizer.decode.return_value = "response"
        loader.model = mock_model
        loader.tokenizer = mock_tokenizer
        result = loader.chat([{"role": "user", "content": "hi"}])
        assert result == "response"


# ── Additional config edge cases ────────────────────────────────────────────

class TestConfigEdgeCases:
    def test_repetition_penalty_one(self):
        cfg = HFLocalConfig(model="gpt2", repetition_penalty=1.0)
        assert cfg.repetition_penalty == 1.0

    def test_temperature_zero(self):
        cfg = HFLocalConfig(model="gpt2", temperature=0.0)
        assert cfg.temperature == 0.0

    def test_top_p_zero(self):
        cfg = HFLocalConfig(model="gpt2", top_p=0.0)
        assert cfg.top_p == 0.0

    def test_max_new_tokens_zero(self):
        cfg = HFLocalConfig(model="gpt2", max_new_tokens=0)
        assert cfg.max_new_tokens == 0

    def test_model_name_with_slash(self):
        cfg = HFLocalConfig(model="meta-llama/Llama-2-7b")
        assert cfg.model == "meta-llama/Llama-2-7b"

    def test_cache_dir_none(self):
        cfg = HFLocalConfig(model="gpt2")
        assert cfg.cache_dir is None

    def test_all_devices(self):
        for dev in ["cpu", "cuda", "mps", "auto"]:
            cfg = HFLocalConfig(model="gpt2", device=dev)
            assert cfg.device == dev

    def test_all_dtypes(self):
        for dt in ["auto", "float32", "float16", "half", "bfloat16"]:
            cfg = HFLocalConfig(model="gpt2", dtype=dt)
            assert cfg.dtype == dt


# ── More dtype tests ────────────────────────────────────────────────────────

class TestGetDtypeExtended:
    def test_float32_literal(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="float32"))
        assert loader._get_dtype() == "float32"

    def test_float16_literal(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="float16"))
        assert loader._get_dtype() == "float16"

    def test_auto_literal(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="auto"))
        assert loader._get_dtype() == "auto"

    def test_unknown_multiple_chars(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="complex64"))
        assert loader._get_dtype() == "float32"

    def test_int8_string(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2", dtype="int8"))
        assert loader._get_dtype() == "float32"


# ── More chat prompt tests ──────────────────────────────────────────────────

class TestFormatChatPromptExtended:
    def test_system_only(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        msgs = [{"role": "system", "content": "You are helpful."}]
        prompt = loader._format_chat_prompt(msgs)
        assert "System: You are helpful." in prompt
        assert prompt.endswith("Assistant:")

    def test_long_conversation(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        msgs = []
        for i in range(10):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        prompt = loader._format_chat_prompt(msgs)
        assert "User: q0" in prompt
        assert "Assistant: a9" in prompt

    def test_mixed_roles_order(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        prompt = loader._format_chat_prompt(msgs)
        assert prompt.index("System:") < prompt.index("User: q1")
        assert prompt.index("User: q1") < prompt.index("Assistant: a1")
        assert prompt.index("Assistant: a1") < prompt.index("User: q2")

    def test_empty_role_string(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "", "content": "hi"}])
        assert "Assistant:" in prompt

    def test_whitespace_content(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        prompt = loader._format_chat_prompt([{"role": "user", "content": "  "}])
        assert "User:   " in prompt


# ── Loader state tests ──────────────────────────────────────────────────────

class TestLoaderState:
    def test_model_none_after_init(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        assert loader.model is None

    def test_tokenizer_none_after_init(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        assert loader.tokenizer is None

    def test_config_device_resolved(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        assert loader.config.device == "cpu"

    def test_config_preserved_after_init(self):
        cfg = HFLocalConfig(model="custom-model", device="cpu", dtype="float16")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.config.model == "custom-model"
        assert loader.config.dtype == "float16"

    def test_unload_after_unload(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        loader.unload()
        loader.unload()
        assert loader.model is None
        assert loader.tokenizer is None

    def test_multiple_loaders_independent(self):
        l1 = HuggingFaceLocalLoader(HFLocalConfig(model="model1"))
        l2 = HuggingFaceLocalLoader(HFLocalConfig(model="model2"))
        l1.model = MagicMock()
        assert l2.model is None
        assert l1.config.model == "model1"
        assert l2.config.model == "model2"


# ── Generate with mocked model (more scenarios) ─────────────────────────────

class TestGenerateExtended:
    def test_generate_empty_prompt(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": "toks"}
        mock_model.generate.return_value = "ids"
        mock_tokenizer.decode.return_value = "output"
        loader.model = mock_model
        loader.tokenizer = mock_tokenizer
        result = loader.generate("")
        assert result == "output"

    def test_generate_with_all_params(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": "toks"}
        mock_model.generate.return_value = "ids"
        mock_tokenizer.decode.return_value = "out"
        loader.model = mock_model
        loader.tokenizer = mock_tokenizer
        result = loader.generate(
            "hello",
            max_new_tokens=50,
            temperature=0.3,
            top_p=0.8,
            repetition_penalty=1.2,
            do_sample=False,
        )
        assert result == "out"

    def test_chat_with_system(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": "toks"}
        mock_model.generate.return_value = "ids"
        mock_tokenizer.decode.return_value = "response"
        loader.model = mock_model
        loader.tokenizer = mock_tokenizer
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        result = loader.chat(msgs)
        assert result == "response"

    def test_chat_multiple_turns(self):
        loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": "toks"}
        mock_model.generate.return_value = "ids"
        mock_tokenizer.decode.return_value = "reply"
        loader.model = mock_model
        loader.tokenizer = mock_tokenizer
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        result = loader.chat(msgs)
        assert result == "reply"


# ── Module-level functions (more tests) ─────────────────────────────────────

class TestModuleFunctions:
    def test_download_model_no_transformers(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", None):
            with pytest.raises(AttributeError):
                download_model("gpt2", cache_dir="/tmp")

    def test_load_model_no_transformers(self):
        with patch("domains.training.huggingface.local_loader.AutoModelForCausalLM", None):
            with pytest.raises(ImportError):
                load_model(HFLocalConfig(model="gpt2"))

    def test_generate_local_no_transformers(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", None):
            with pytest.raises(ImportError):
                generate_local("hi", model="gpt2")

    def test_all_exports_complete(self):
        from domains.training.huggingface import local_loader
        assert set(local_loader.__all__) == {
            "HFLocalConfig",
            "HuggingFaceLocalLoader",
            "HuggingFaceLocalClient",
            "download_model",
            "load_model",
            "generate_local",
        }

    def test_client_class_name(self):
        assert HuggingFaceLocalClient.__name__ == "HuggingFaceLocalClient"

    def test_client_is_subclass(self):
        assert issubclass(HuggingFaceLocalClient, HuggingFaceLocalLoader)

    def test_client_has_all_methods(self):
        assert hasattr(HuggingFaceLocalClient, "load")
        assert hasattr(HuggingFaceLocalClient, "generate")
        assert hasattr(HuggingFaceLocalClient, "chat")
        assert hasattr(HuggingFaceLocalClient, "unload")


# ── Load path edge cases ────────────────────────────────────────────────────

class TestLoadPath:
    def test_load_raises_when_both_none(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", None):
            with patch("domains.training.huggingface.local_loader.AutoModelForCausalLM", None):
                loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
                with pytest.raises(ImportError):
                    loader.load()

    def test_load_raises_when_tokenizer_none(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", None):
            loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
            with pytest.raises(ImportError, match="transformers"):
                loader.load()

    def test_load_raises_when_model_none(self):
        with patch("domains.training.huggingface.local_loader.AutoTokenizer", MagicMock()):
            with patch("domains.training.huggingface.local_loader.AutoModelForCausalLM", None):
                loader = HuggingFaceLocalLoader(HFLocalConfig(model="gpt2"))
                with pytest.raises(ImportError):
                    loader.load()
