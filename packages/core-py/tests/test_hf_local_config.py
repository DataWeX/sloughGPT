"""Tests for domains.training.huggingface.local_loader — HFLocalConfig."""

from domains.training.huggingface.local_loader import (
    HFLocalConfig,
    HuggingFaceLocalLoader,
    HuggingFaceLocalClient,
)


class TestHFLocalConfigDefaults:
    def test_model_required(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.model == "gpt2"

    def test_device_default(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.device == "auto"

    def test_dtype_default(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.dtype == "auto"

    def test_load_in_8bit_default(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.load_in_8bit is False

    def test_load_in_4bit_default(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.load_in_4bit is False

    def test_cache_dir_default_none(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.cache_dir is None

    def test_local_files_only_default(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.local_files_only is True

    def test_max_new_tokens_default(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.max_new_tokens == 256

    def test_temperature_default(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.temperature == 0.7

    def test_top_p_default(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.top_p == 0.9

    def test_repetition_penalty_default(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.repetition_penalty == 1.0

    def test_field_count(self):
        import dataclasses
        fields = [f.name for f in dataclasses.fields(HFLocalConfig)]
        assert len(fields) == 11

    def test_field_names(self):
        import dataclasses
        names = {f.name for f in dataclasses.fields(HFLocalConfig)}
        expected = {"model", "device", "dtype", "load_in_8bit", "load_in_4bit",
                    "cache_dir", "local_files_only", "max_new_tokens",
                    "temperature", "top_p", "repetition_penalty"}
        assert names == expected


class TestHFLocalConfigCustom:
    def test_device_custom(self):
        hc = HFLocalConfig(model="gpt2", device="cuda")
        assert hc.device == "cuda"

    def test_load_in_4bit_true(self):
        hc = HFLocalConfig(model="gpt2", load_in_4bit=True)
        assert hc.load_in_4bit is True

    def test_load_in_8bit_true(self):
        hc = HFLocalConfig(model="gpt2", load_in_8bit=True)
        assert hc.load_in_8bit is True

    def test_dtype_custom(self):
        hc = HFLocalConfig(model="gpt2", dtype="float16")
        assert hc.dtype == "float16"

    def test_cache_dir_custom(self):
        hc = HFLocalConfig(model="gpt2", cache_dir="/tmp/cache")
        assert hc.cache_dir == "/tmp/cache"

    def test_local_files_only_false(self):
        hc = HFLocalConfig(model="gpt2", local_files_only=False)
        assert hc.local_files_only is False

    def test_max_new_tokens_custom(self):
        hc = HFLocalConfig(model="gpt2", max_new_tokens=512)
        assert hc.max_new_tokens == 512

    def test_temperature_custom(self):
        hc = HFLocalConfig(model="gpt2", temperature=0.1)
        assert hc.temperature == 0.1

    def test_top_p_custom(self):
        hc = HFLocalConfig(model="gpt2", top_p=0.95)
        assert hc.top_p == 0.95

    def test_repetition_penalty_custom(self):
        hc = HFLocalConfig(model="gpt2", repetition_penalty=1.3)
        assert hc.repetition_penalty == 1.3

    def test_all_fields_custom(self):
        hc = HFLocalConfig(
            model="llama-7b",
            device="cuda",
            dtype="bfloat16",
            load_in_8bit=True,
            load_in_4bit=False,
            cache_dir="/models",
            local_files_only=False,
            max_new_tokens=1024,
            temperature=0.2,
            top_p=0.8,
            repetition_penalty=1.5,
        )
        assert hc.model == "llama-7b"
        assert hc.device == "cuda"
        assert hc.dtype == "bfloat16"
        assert hc.load_in_8bit is True
        assert hc.load_in_4bit is False
        assert hc.cache_dir == "/models"
        assert hc.local_files_only is False
        assert hc.max_new_tokens == 1024
        assert hc.temperature == 0.2
        assert hc.top_p == 0.8
        assert hc.repetition_penalty == 1.5

    def test_empty_model_string(self):
        hc = HFLocalConfig(model="")
        assert hc.model == ""

    def test_model_with_slash(self):
        hc = HFLocalConfig(model="meta-llama/Llama-2-7b-hf")
        assert hc.model == "meta-llama/Llama-2-7b-hf"


class TestHFLocalConfigEquality:
    def test_equal_instances(self):
        a = HFLocalConfig(model="gpt2")
        b = HFLocalConfig(model="gpt2")
        assert a == b

    def test_not_equal_model(self):
        a = HFLocalConfig(model="gpt2")
        b = HFLocalConfig(model="llama")
        assert a != b

    def test_not_equal_device(self):
        a = HFLocalConfig(model="gpt2", device="cpu")
        b = HFLocalConfig(model="gpt2", device="cuda")
        assert a != b

    def test_not_equal_load_in_4bit(self):
        a = HFLocalConfig(model="gpt2", load_in_4bit=False)
        b = HFLocalConfig(model="gpt2", load_in_4bit=True)
        assert a != b

    def test_not_equal_to_non_dataclass(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc != "not a config"

    def test_equal_custom(self):
        a = HFLocalConfig(model="x", device="cuda", temperature=0.1)
        b = HFLocalConfig(model="x", device="cuda", temperature=0.1)
        assert a == b


class TestHFLocalConfigRepr:
    def test_repr_contains_class_name(self):
        hc = HFLocalConfig(model="gpt2")
        assert "HFLocalConfig" in repr(hc)

    def test_repr_contains_model(self):
        hc = HFLocalConfig(model="llama")
        r = repr(hc)
        assert "model='llama'" in r

    def test_repr_contains_device(self):
        hc = HFLocalConfig(model="gpt2", device="cuda")
        r = repr(hc)
        assert "device='cuda'" in r

    def test_repr_contains_load_in_4bit(self):
        hc = HFLocalConfig(model="gpt2", load_in_4bit=True)
        r = repr(hc)
        assert "load_in_4bit=True" in r


class TestHFLocalConfigMutation:
    def test_can_set_device(self):
        hc = HFLocalConfig(model="gpt2")
        hc.device = "cuda"
        assert hc.device == "cuda"

    def test_can_set_dtype(self):
        hc = HFLocalConfig(model="gpt2")
        hc.dtype = "float16"
        assert hc.dtype == "float16"

    def test_can_set_load_in_4bit(self):
        hc = HFLocalConfig(model="gpt2")
        hc.load_in_4bit = True
        assert hc.load_in_4bit is True

    def test_can_set_cache_dir(self):
        hc = HFLocalConfig(model="gpt2")
        hc.cache_dir = "/new/path"
        assert hc.cache_dir == "/new/path"

    def test_can_overwrite_fields(self):
        hc = HFLocalConfig(model="gpt2", device="cpu")
        hc.device = "cuda"
        hc.device = "mps"
        assert hc.device == "mps"

    def test_independent_instances(self):
        a = HFLocalConfig(model="a", device="cpu")
        b = HFLocalConfig(model="b", device="cuda")
        a.device = "cuda"
        assert b.device == "cuda"


class TestHFLocalConfigEdgeCases:
    def test_zero_temperature(self):
        hc = HFLocalConfig(model="gpt2", temperature=0.0)
        assert hc.temperature == 0.0

    def test_negative_temperature(self):
        hc = HFLocalConfig(model="gpt2", temperature=-0.5)
        assert hc.temperature == -0.5

    def test_zero_max_new_tokens(self):
        hc = HFLocalConfig(model="gpt2", max_new_tokens=0)
        assert hc.max_new_tokens == 0

    def test_large_max_new_tokens(self):
        hc = HFLocalConfig(model="gpt2", max_new_tokens=100000)
        assert hc.max_new_tokens == 100000

    def test_zero_top_p(self):
        hc = HFLocalConfig(model="gpt2", top_p=0.0)
        assert hc.top_p == 0.0

    def test_negative_repetition_penalty(self):
        hc = HFLocalConfig(model="gpt2", repetition_penalty=-1.0)
        assert hc.repetition_penalty == -1.0

    def test_zero_repetition_penalty(self):
        hc = HFLocalConfig(model="gpt2", repetition_penalty=0.0)
        assert hc.repetition_penalty == 0.0

    def test_cache_dir_empty_string(self):
        hc = HFLocalConfig(model="gpt2", cache_dir="")
        assert hc.cache_dir == ""

    def test_cache_dir_with_spaces(self):
        hc = HFLocalConfig(model="gpt2", cache_dir="/path with spaces/cache")
        assert hc.cache_dir == "/path with spaces/cache"

    def test_both_8bit_and_4bit(self):
        hc = HFLocalConfig(model="gpt2", load_in_8bit=True, load_in_4bit=True)
        assert hc.load_in_8bit is True
        assert hc.load_in_4bit is True

    def test_copy_semantics(self):
        import dataclasses
        a = HFLocalConfig(model="gpt2", device="cuda", temperature=0.7)
        b = dataclasses.replace(a, device="cpu")
        assert a.device == "cuda"
        assert b.device == "cpu"
        assert a.model == b.model

    def test_copy_preserves_all_fields(self):
        import dataclasses
        a = HFLocalConfig(
            model="llama", device="cuda", dtype="float16",
            load_in_4bit=True, cache_dir="/cache",
            temperature=0.1, max_new_tokens=512,
        )
        b = dataclasses.replace(a, model="mistral")
        assert b.model == "mistral"
        assert b.device == "cuda"
        assert b.dtype == "float16"
        assert b.load_in_4bit is True
        assert b.cache_dir == "/cache"
        assert b.temperature == 0.1
        assert b.max_new_tokens == 512


class TestHFLocalConfigDeviceResolution:
    def test_auto_resolves_to_cpu(self):
        cfg = HFLocalConfig(model="gpt2", device="auto")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.config.device == "cpu"

    def test_explicit_cpu_preserved(self):
        cfg = HFLocalConfig(model="gpt2", device="cpu")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.config.device == "cpu"

    def test_explicit_cuda_preserved(self):
        cfg = HFLocalConfig(model="gpt2", device="cuda")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.config.device == "cuda"

    def test_explicit_mps_preserved(self):
        cfg = HFLocalConfig(model="gpt2", device="mps")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.config.device == "mps"

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

    def test_get_dtype_auto(self):
        cfg = HFLocalConfig(model="gpt2", dtype="auto")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader._get_dtype() == "auto"

    def test_get_dtype_unknown_fallback(self):
        cfg = HFLocalConfig(model="gpt2", dtype="invalid")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader._get_dtype() == "float32"


class TestHFLocalLoaderInteraction:
    def test_loader_stores_config(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.config is cfg

    def test_loader_model_initially_none(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.model is None

    def test_loader_tokenizer_initially_none(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        assert loader.tokenizer is None

    def test_generate_raises_before_load(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        try:
            loader.generate("hello")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not loaded" in str(e)

    def test_chat_prompt_format(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = loader._format_chat_prompt(messages)
        assert result == "User: hi\nAssistant: hello\nAssistant:"

    def test_unload_sets_none(self):
        cfg = HFLocalConfig(model="gpt2")
        loader = HuggingFaceLocalLoader(cfg)
        loader.model = "mock"
        loader.tokenizer = "mock"
        loader.unload()
        assert loader.model is None
        assert loader.tokenizer is None

    def test_local_client_is_subclass(self):
        assert issubclass(HuggingFaceLocalClient, HuggingFaceLocalLoader)
