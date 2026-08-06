"""Supplementary tests for kernel_npu.py covering the remaining branches."""

import sys
import types
import importlib
import builtins
import struct
import json
import time
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch, mock_open

import domains.shell.kernel_npu as kernel_npu
from domains.shell.kernel_npu import NPUDevice, NPUModel, _HuggingFaceProvider
from domains.shell.kernel_syscall import SyscallResult


# ── Fake torch / transformers modules (not installed in this environment) ──

class _FakeTensor:
    def __init__(self, data=None):
        self._data = data
        self.shape = data.shape if data is not None else None

    def __getitem__(self, item):
        return _FakeTensor(self._data[item])

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._data)

    def tolist(self):
        return self._data.tolist()

    def long(self):
        return self

    def to(self, device):
        return self


def _fake_torch():
    torch = types.ModuleType("torch")
    torch.Tensor = _FakeTensor

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    torch.no_grad = lambda: _Ctx()
    torch.from_numpy = lambda arr: _FakeTensor(arr)
    return torch


def _fake_transformers(monkeypatch):
    fake_tok = SimpleNamespace(vocab_size=100)
    fake_model = SimpleNamespace()
    fake_model.to = Mock(return_value=fake_model)
    fake_model.eval = lambda: None
    tf = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return fake_tok

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return fake_model

    tf.AutoTokenizer = _AutoTokenizer
    tf.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", tf)
    return fake_tok, fake_model


# ── Providers exercising the non-forward_pass / non-inner branches ─────────

class ProviderNoInner:
    def __init__(self):
        self._model = object()

    def metadata(self):
        return {"n_layer": 2, "n_embed": 64, "n_head": 4}

    def forward_numpy(self, input_ids):
        return np.zeros((1, input_ids.shape[1], 8), dtype=np.float32)

    def generate_numpy(self, input_ids, max_new_tokens=50, **kwargs):
        return np.zeros((1, 3), dtype=np.int64)

    def tokenize(self, text):
        return [1, 2, 3]

    def detokenize(self, token_ids):
        return "ok"


class CompareElseProvider:
    def __init__(self):
        self._model = SimpleNamespace(forward_pass=self._forward_pass)

    def _forward_pass(self, input_ids):
        return SimpleNamespace(
            logits=np.zeros((1, input_ids.shape[1], 4), dtype=np.float32),
            engine="numpy",
        )

    def generate_numpy(self, input_ids, max_new_tokens=20, **kwargs):
        return np.zeros((1, 2), dtype=np.int64)

    def tokenize(self, text):
        return [1, 2]

    def detokenize(self, token_ids):
        return "ab"


def _make_npu(provider, name="m"):
    npu = NPUDevice()
    npu.open()
    npu._models[name] = NPUModel(name=name, provider=provider,
                                 config=provider.metadata(), loaded_at=time.time())
    npu._default_model = name
    return npu


# ── psutil import-time branch ──────────────────────────────────────────────

class TestPsutilImportBranch:
    def test_reload_with_psutil_sets_flag(self, monkeypatch):
        saved = {name: getattr(kernel_npu, name)
                 for name in ("NPUDevice", "NPUModel", "_HuggingFaceProvider")}
        fake = types.ModuleType("psutil")
        fake.Process = object
        monkeypatch.setitem(sys.modules, "psutil", fake)
        mod = importlib.reload(kernel_npu)
        assert mod._HAS_PSUTIL is True
        monkeypatch.setitem(sys.modules, "psutil", None)
        mod = importlib.reload(kernel_npu)
        assert mod._HAS_PSUTIL is False
        for name, cls in saved.items():
            setattr(kernel_npu, name, cls)


# ── _HuggingFaceProvider ───────────────────────────────────────────────────

class TestHuggingFaceProvider:
    def test_init_and_metadata(self):
        tok = SimpleNamespace(vocab_size=123)
        model = object()
        provider = _HuggingFaceProvider(model, tok, "gpt2", "cpu")
        meta = provider.metadata()
        assert meta == {"model_id": "gpt2", "device": "cpu", "vocab_size": 123}

    def test_call_missing_input_ids(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        provider = _HuggingFaceProvider(object(), object(), "gid")
        with pytest.raises(ValueError, match="input_ids"):
            provider({})

    def test_call_numpy_input_tensor_output(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        logits = _FakeTensor(np.arange(10).reshape(1, 5, 2))

        def _model_forward(input_ids):
            return SimpleNamespace(logits=logits)

        provider = _HuggingFaceProvider(_model_forward, object(), "gid")
        result = provider({"input_ids": np.array([1, 2, 3])})
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 5, 2)

    def test_call_non_tensor_output_returned(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        logits = np.zeros((1, 4), dtype=np.float32)

        def _model_forward(input_ids):
            return SimpleNamespace(logits=logits)

        provider = _HuggingFaceProvider(_model_forward, object(), "gid")
        result = provider({"input_ids": [1, 2]})
        assert result is logits

    def test_generate_numpy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())

        class _Tokenizer:
            eos_token_id = 0

            def __call__(self, prompt, return_tensors=None):
                return {"input_ids": _FakeTensor(
                    np.zeros((1, 3), dtype=np.int64))}

        model = SimpleNamespace(generate=lambda *a, **k: _FakeTensor(
            np.zeros((1, 7), dtype=np.int64)))
        provider = _HuggingFaceProvider(model, _Tokenizer(), "gid", "cpu")
        assert provider.generate_numpy("hi", max_tokens=5) == [0, 0, 0, 0]

    def test_tokenize_detokenize(self):
        tok = SimpleNamespace(vocab_size=100)
        tok.encode = lambda text: [1, 2, 3]
        tok.decode = lambda ids: "decoded"
        provider = _HuggingFaceProvider(object(), tok, "gid")
        assert provider.tokenize("x") == [1, 2, 3]
        assert provider.detokenize([1]) == "decoded"


# ── load_model / _load_provider gaps ───────────────────────────────────────

class TestLoadModelGaps:
    def test_load_model_no_source(self):
        npu = NPUDevice()
        result = npu.load_model("m", "")
        assert result.success is False
        assert "no source path" in result.error

    def test_load_model_exception_caught(self):
        npu = NPUDevice()
        npu.open()
        result = npu.load_model("m", "model.safetensors", backend="numpy")
        assert result.success is False
        assert "load_model failed" in result.error

    def test_unload_model_default_name(self):
        npu = _make_npu(Mock(), name="m")
        result = npu.unload_model("")
        assert result.success is True
        assert "m" not in npu._models

    def test_load_provider_slnc_success(self):
        npu = NPUDevice()
        provider = Mock()
        npu._load_c_provider = lambda n, s, p: provider
        got, backend = npu._load_provider("m", "m.slnc", "m.slnc")
        assert got is provider
        assert backend == "c"

    def test_load_provider_huggingface_source(self, monkeypatch):
        npu = NPUDevice()
        provider = Mock()
        monkeypatch.setattr(npu, "_load_huggingface_provider",
                            lambda n, s, k: provider)
        got, backend = npu._load_provider("m", "huggingface:gpt2", "p")
        assert got is provider
        assert backend == "huggingface"

    def test_load_provider_huggingface_backend(self, monkeypatch):
        npu = NPUDevice()
        provider = Mock()
        monkeypatch.setattr(npu, "_load_huggingface_provider",
                            lambda n, s, k: provider)
        got, backend = npu._load_provider("m", "anything", "p", "huggingface")
        assert got is provider
        assert backend == "huggingface"

    def test_load_provider_unknown_source(self):
        npu = NPUDevice()
        with pytest.raises(ValueError, match="Unknown backend for"):
            npu._load_provider("m", "model.safetensors", "p")

    def test_load_huggingface_provider_happy(self, monkeypatch):
        fake_tok, fake_model = _fake_transformers(monkeypatch)
        npu = NPUDevice()
        provider = npu._load_huggingface_provider("m", "huggingface:gpt2",
                                                  {"device": "cpu"})
        assert isinstance(provider, kernel_npu._HuggingFaceProvider)
        assert provider._model_id == "gpt2"
        fake_model.to.assert_called_once_with("cpu")

    def test_load_huggingface_provider_import_error(self, monkeypatch):
        real_import = builtins.__import__

        def _no_transformers(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_transformers)
        npu = NPUDevice()
        with pytest.raises(ValueError, match="transformers not installed"):
            npu._load_huggingface_provider("m", "huggingface:gpt2", {})


# ── inference else branches + missing-model error paths ────────────────────

class TestInferenceGaps:
    def test_forward_uses_forward_numpy(self):
        npu = _make_npu(ProviderNoInner())
        result = npu.forward("m", [1, 2, 3])
        assert result.success is True
        assert result.value["engine"] == "numpy"

    def test_generate_uses_provider_generate_numpy(self):
        npu = _make_npu(ProviderNoInner())
        result = npu.generate("m", "hi", max_tokens=5)
        assert result.success is True
        assert result.value["text"] == "ok"
        assert result.value["token_count"] == 3

    def test_embed_missing_model(self):
        npu = NPUDevice()
        assert npu.embed("missing", "hi").success is False

    def test_tokenize_missing_model(self):
        npu = NPUDevice()
        assert npu.tokenize("missing", "hi").success is False

    def test_detokenize_missing_model(self):
        npu = NPUDevice()
        assert npu.detokenize("missing", [1]).success is False

    def test_train_step_missing_model(self):
        npu = NPUDevice()
        assert npu.train_step("missing", [1], [1]).success is False

    def test_batch_uses_provider_generate_numpy(self):
        npu = _make_npu(ProviderNoInner())
        result = npu.batch("m", ["a", "b"], max_tokens=3)
        assert result.success is True
        assert result.value["count"] == 2

    def test_benchmark_uses_provider_generate_numpy(self):
        npu = _make_npu(ProviderNoInner())
        result = npu.benchmark("m", prompt_lengths=[2], max_tokens=3)
        assert result.success is True
        assert len(result.value["results"]) == 1


# ── checkpoint / quantize / dequantize gaps ────────────────────────────────

class TestCheckpointQuantizeGaps:
    def _make_soul_binary(self, weights=None):
        meta = json.dumps({"soul_name": "test"}).encode()
        header = (b"SOU\x00" + struct.pack("<I", 3) +
                  struct.pack("<I", len(meta)) + meta)
        weight_data = b""
        if weights:
            weight_data = struct.pack("<I", len(weights))
            for wname, arr in weights.items():
                name_bytes = wname.encode("utf-8")
                weight_data += struct.pack("<I", len(name_bytes))
                weight_data += name_bytes
                weight_data += struct.pack("<I", arr.ndim)
                for s in arr.shape:
                    weight_data += struct.pack("<I", s)
                weight_data += arr.tobytes()
        else:
            weight_data = struct.pack("<I", 0)
        return header + weight_data

    def test_save_checkpoint_exception(self):
        npu = _make_npu(Mock())
        with patch("domains.inference.slo_format.save_soul",
                   side_effect=RuntimeError("boom")):
            result = npu.save_checkpoint("m", "/tmp/test.soul")
        assert result.success is False
        assert "SAVE_CHECKPOINT failed" in result.error

    def test_load_checkpoint_calls_load_state_dict(self):
        provider = Mock()
        npu = _make_npu(provider)
        w1 = np.ones((4, 4), dtype=np.float32)
        soul_data = self._make_soul_binary({"w1": w1})
        with patch("builtins.open", mock_open(read_data=soul_data)):
            result = npu.load_checkpoint("m", "/tmp/test.soul")
        assert result.success is True
        assert result.value["weights_restored"] == 1
        provider.load_state_dict.assert_called_once()

    def test_load_checkpoint_corrupt_format(self):
        npu = _make_npu(Mock())
        with patch("builtins.open", mock_open(read_data=b"SOU\x00xxxx")):
            result = npu.load_checkpoint("m", "/tmp/bad.soul")
        assert result.success is False
        assert "LOAD_CHECKPOINT failed" in result.error

    def test_quantize_skips_non_array_params(self):
        from tests.test_npu import MockProvider
        provider = MockProvider()
        provider._model._params = {
            "w1": np.random.randn(2, 2).astype(np.float32),
            "bias": 1.0,
        }
        npu = _make_npu(provider)
        result = npu.quantize("m", 8)
        assert result.success is True
        assert result.value["params_quantized"] == 1

    def test_dequantize_missing_model(self):
        npu = NPUDevice()
        assert npu.dequantize("missing").success is False


# ── compare gaps ───────────────────────────────────────────────────────────

class TestCompareGaps:
    def test_compare_uses_provider_generate_numpy(self):
        npu = NPUDevice()
        npu._models["a"] = NPUModel(name="a", provider=CompareElseProvider(),
                                    config={})
        npu._models["b"] = NPUModel(name="b", provider=CompareElseProvider(),
                                    config={})
        result = npu.compare("a", "b", prompt="hi", max_tokens=5)
        assert result.success is True
        assert set(result.value["models"]) == {"a", "b"}

    def test_compare_with_psutil(self, monkeypatch):
        class _FakeProc:
            def memory_info(self):
                return SimpleNamespace(rss=123456)

        class _FakePsutil:
            def Process(self):
                return _FakeProc()

        monkeypatch.setattr(kernel_npu, "_HAS_PSUTIL", True)
        monkeypatch.setattr(kernel_npu, "_psutil", _FakePsutil())
        npu = NPUDevice()
        npu._models["a"] = NPUModel(name="a", provider=CompareElseProvider(),
                                     config={})
        npu._models["b"] = NPUModel(name="b", provider=CompareElseProvider(),
                                     config={})
        result = npu.compare("a", "b", prompt="hi", max_tokens=5)
        assert result.success is True
        assert result.value["models"]["a"]["memory_mb"] > 0


# ── Additional Coverage ──────────────────────────────────────────────────────


class TestNPUDeviceHealth:
    def test_health_empty(self):
        npu = NPUDevice()
        result = npu.health()
        assert result.success is True
        assert result.value["models_loaded"] == 0
        assert result.value["total_inferences"] == 0

    def test_health_with_model(self):
        npu = NPUDevice()
        npu._models["m"] = NPUModel(name="m", provider=Mock(),
                                     config={})
        npu._models["m"].inference_count = 5
        npu._models["m"].total_tokens = 100
        result = npu.health()
        assert result.value["models_loaded"] == 1
        assert result.value["models"]["m"]["inference_count"] == 5


class TestNPUClearCache:
    def test_clear_cache_missing_model(self):
        npu = NPUDevice()
        result = npu.clear_cache("nonexistent")
        assert result.success is False

    def test_clear_cache_resets_stats(self):
        npu = NPUDevice()
        prov = Mock(spec=["_kv_cache"])
        prov._kv_cache = Mock()
        prov._kv_cache._cache = {}
        npu._models["m"] = NPUModel(name="m", provider=prov, config={})
        npu._models["m"].inference_count = 10
        npu._models["m"].total_tokens = 500
        result = npu.clear_cache("m")
        assert result.success is True
        assert result.value["stats_reset"] is True
        assert npu._models["m"].inference_count == 0


class TestNPUProfile:
    def test_profile_missing_model(self):
        npu = NPUDevice()
        result = npu.profile("nonexistent")
        assert "error" in result

    def test_profile_default_batch_sizes(self):
        npu = NPUDevice()
        npu._models["m"] = NPUModel(name="m", provider=Mock(),
                                     config={"n_layer": 4, "n_embed": 128, "n_head": 4})
        result = npu.profile("m", seq_len=64)
        assert "profiles" in result
        assert len(result["profiles"]) == 5
        assert result["architecture"]["n_layer"] == 4


class TestNPUAttentionMaps:
    def test_attention_maps_missing_model(self):
        npu = NPUDevice()
        result = npu.attention_maps("nonexistent", "hello")
        assert result.success is False

    def test_attention_maps_empty_text(self):
        npu = NPUDevice()
        npu._models["m"] = NPUModel(name="m", provider=Mock(), config={})
        result = npu.attention_maps("m", "")
        assert result.success is True
        assert result.value["token_count"] == 0

    def test_attention_maps_with_text(self):
        npu = NPUDevice()
        prov = Mock()
        prov.tokenize.return_value = [1, 2, 3]
        npu._models["m"] = NPUModel(name="m", provider=prov,
                                     config={"n_head": 4, "n_layer": 2})
        result = npu.attention_maps("m", "hello world", layer=0)
        assert result.success is True
        assert result.value["layers_extracted"] == 1
        assert "0" in result.value["attention"]


class TestNPUBatch:
    def test_batch_empty_prompts(self):
        npu = NPUDevice()
        npu._models["m"] = NPUModel(name="m", provider=Mock(), config={})
        result = npu.batch("m", [])
        assert result.success is True
        assert result.value["results"] == []

    def test_batch_with_prompts(self):
        npu = NPUDevice()
        prov = Mock(spec=["tokenize", "generate_numpy", "detokenize"])
        prov.tokenize.return_value = [1, 2]
        prov.generate_numpy.return_value = np.array([[1, 2, 3]])
        prov.detokenize.return_value = "hello"
        npu._models["m"] = NPUModel(name="m", provider=prov, config={})
        result = npu.batch("m", ["hi", "yo"], max_tokens=5)
        assert result.success is True
        assert len(result.value["results"]) == 2


class TestNPULayers:
    def test_layers_missing_model(self):
        npu = NPUDevice()
        result = npu.layers("nonexistent")
        assert result.success is False

    def test_layers_specific(self):
        npu = NPUDevice()
        npu._models["m"] = NPUModel(name="m", provider=Mock(),
                                     config={"n_layer": 3, "n_embed": 64})
        result = npu.layers("m", layer=1)
        assert result.success is True
        assert result.value["layers"][0]["index"] == 1
        assert result.value["architecture"]["n_layer"] == 3


class TestNPURefcounting:
    def test_acquire_and_release(self):
        npu = NPUDevice()
        npu._models["m"] = NPUModel(name="m", provider=Mock(), config={})
        assert npu._acquire_ref("m") is True
        assert npu._ref_counts.get("m", 0) == 1
        npu._release_ref("m")
        assert npu._ref_counts.get("m", 0) == 0

    def test_release_nonexistent(self):
        npu = NPUDevice()
        npu._release_ref("m")  # should not raise
