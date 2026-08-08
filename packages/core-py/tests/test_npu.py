"""Tests for the NPU (Neural Processing Unit) device."""

import pytest
import time
import importlib
import sys
import types
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock, mock_open
from dataclasses import dataclass

from domains.shell.kernel_npu import NPUDevice, NPUModel
from domains.shell.kernel_devices import DeviceType, DeviceState
from domains.shell.kernel_syscall import SyscallResult
from domains.inference.forward_pass import ForwardPassResult


# ── Mock Provider ─────────────────────────────────────────────────────────

class MockSloTransformer:
    """Mock SloTransformer for testing without real weights."""

    def __init__(self, vocab_size=256, n_embed=64, n_layer=2):
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.max_seq_len = 128
        self._config = {
            "n_embd": n_embed,
            "n_head": 4,
            "n_layer": n_layer,
            "vocab_size": vocab_size,
        }
        self._layers = []
        self._params = {}
        self._original_weights = {}
        self._quant_scales = {}
        self._quant_bits = {}
        self._is_quantized = False

    def forward_numpy(self, input_ids):
        batch, seq = input_ids.shape
        return np.random.randn(batch, seq, self.vocab_size).astype(np.float32)

    def forward_pass(self, input_ids):
        """Unified forward pass — returns ForwardPassResult."""
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        logits = np.random.randn(input_ids.shape[0], input_ids.shape[1],
                                 self.vocab_size).astype(np.float32)
        return ForwardPassResult(logits=logits, engine="numpy")

    def generate_numpy(self, input_ids, max_new_tokens=10, temperature=1.0,
                       top_k=None, top_p=None, repetition_penalty=1.0,
                       eos_token=0):
        batch = input_ids.shape[0]
        gen_len = min(max_new_tokens, 5)
        return np.random.randint(0, self.vocab_size, (batch, gen_len), dtype=np.int64)

    def generate_numpy_stream(self, input_ids, max_new_tokens=10, eos_token=0,
                              temperature=1.0, top_k=None, top_p=None,
                              repetition_penalty=1.0):
        gen_len = min(max_new_tokens, 5)
        for _ in range(gen_len):
            yield int(np.random.randint(0, self.vocab_size))

    def parameters(self):
        return self._params

    def _named_parameters(self):
        """Return (name, mock_param) pairs where mock_param has .data attribute."""
        class _MockParam:
            def __init__(self, data):
                self.data = data
        result = []
        for name, arr in self._params.items():
            if isinstance(arr, np.ndarray):
                result.append((name, _MockParam(arr)))
        return result

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(state_dict, dict):
            for key, val in state_dict.items():
                self._params[key] = val


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.eos_token_id = 0

    def encode(self, text):
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, token_ids):
        return "".join(chr(t % 128) for t in token_ids)

    def apply_chat_template(self, messages):
        if messages and isinstance(messages[-1], dict):
            return messages[-1].get("content", "")
        return ""


class MockProvider:
    """Mock SlonetChatProvider for testing."""

    def __init__(self, vocab_size=256, n_embed=64):
        self._model = MockSloTransformer(vocab_size, n_embed)
        self._tokenizer = MockTokenizer(vocab_size)
        self._model_id = "mock-model"
        self._hf_model_id = "mock-model"
        self._device = "cpu"
        self._model_lock = None  # NPUDevice checks for this

    def metadata(self):
        return {
            "model_id": self._model_id,
            "architecture": "SloTransformer",
            "total_params": 1000,
            "n_layer": 2,
            "n_embed": 64,
            "n_head": 4,
            "vocab_size": 256,
            "max_seq_len": 128,
            "device": "cpu",
            "quantized": False,
            "has_tokenizer": True,
        }

    def generate(self, prompt, max_tokens=50, temperature=1.0,
                 top_k=None, top_p=None, repetition_penalty=1.0):
        tokens = self._tokenizer.encode(prompt)
        input_ids = np.array([tokens], dtype=np.int64)
        result = self._model.generate_numpy(input_ids, max_new_tokens=max_tokens,
                                            temperature=temperature)
        return self._tokenizer.decode(result[0].tolist())

    def embed(self, text, layer=-1):
        tokens = self._tokenizer.encode(text)
        input_ids = np.array([tokens], dtype=np.int64)
        output = self._model.forward_numpy(input_ids)
        return output[0, -1, :]

    def tokenize(self, text):
        return self._tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self._tokenizer.decode(token_ids)


# ── NPUDevice Tests ──────────────────────────────────────────────────────

class TestNPUDeviceInit:
    def test_create(self):
        npu = NPUDevice()
        assert npu.name == "npu"
        assert npu.device_type == DeviceType.INFERENCE
        assert npu.state == DeviceState.CLOSED
        assert npu._models == {}
        assert npu._default_model == ""

    def test_open(self):
        npu = NPUDevice()
        assert npu.open() is True
        assert npu.state == DeviceState.OPEN
        assert npu._open_count == 1

    def test_close(self):
        npu = NPUDevice()
        npu.open()
        npu.close()
        assert npu.state == DeviceState.CLOSED

    def test_info_empty(self):
        npu = NPUDevice()
        info = npu.info()
        assert info["device"] == "npu"
        assert info["models_loaded"] == 0
        assert info["models"] == {}
        assert info["total_inferences"] == 0


class TestNPUModelManagement:
    def test_load_model_mock(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()

        model = NPUModel(
            name="test-model",
            provider=provider,
            config=provider.metadata(),
            loaded_at=time.time(),
        )
        npu._models["test-model"] = model
        npu._default_model = "test-model"

        assert "test-model" in npu._models
        assert npu._default_model == "test-model"

    def test_unload_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="test-model", provider=provider)
        npu._models["test-model"] = model
        npu._default_model = "test-model"

        result = npu.unload_model("test-model")
        assert result.success is True
        assert "test-model" not in npu._models
        assert npu._default_model == ""

    def test_unload_nonexistent(self):
        npu = NPUDevice()
        result = npu.unload_model("nonexistent")
        assert result.success is False
        assert "not loaded" in result.error

    def test_get_model_default(self):
        npu = NPUDevice()
        provider = MockProvider()
        model = NPUModel(name="default", provider=provider)
        npu._models["default"] = model
        npu._default_model = "default"

        got, err = npu._get_model("")
        assert err is None
        assert got.name == "default"

    def test_get_model_by_name(self):
        npu = NPUDevice()
        provider = MockProvider()
        model = NPUModel(name="specific", provider=provider)
        npu._models["specific"] = model

        got, err = npu._get_model("specific")
        assert err is None
        assert got.name == "specific"

    def test_get_model_missing(self):
        npu = NPUDevice()
        got, err = npu._get_model("missing")
        assert got is None
        assert err is not None
        assert err.success is False


class TestNPUInference:
    def _setup_npu(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="test", provider=provider, config=provider.metadata())
        npu._models["test"] = model
        npu._default_model = "test"
        return npu

    def test_forward(self):
        npu = self._setup_npu()
        result = npu.forward("test", [1, 2, 3, 4])
        assert result.success is True
        assert "logits" in result.value
        assert "shape" in result.value
        assert result.value["forward_time_ms"] >= 0

    def test_forward_missing_model(self):
        npu = self._setup_npu()
        result = npu.forward("nonexistent", [1, 2, 3])
        assert result.success is False

    def test_generate(self):
        npu = self._setup_npu()
        result = npu.generate("test", "Hello world", max_tokens=10)
        assert result.success is True
        assert "text" in result.value
        assert "token_count" in result.value
        assert result.value["token_count"] >= 0

    def test_generate_missing_model(self):
        npu = self._setup_npu()
        result = npu.generate("nonexistent", "Hello")
        assert result.success is False

    def test_embed(self):
        npu = self._setup_npu()
        result = npu.embed("test", "Hello world")
        assert result.success is True
        assert "embedding" in result.value
        assert "shape" in result.value

    def test_tokenize(self):
        npu = self._setup_npu()
        result = npu.tokenize("test", "Hello")
        assert result.success is True
        assert "token_ids" in result.value
        assert "token_count" in result.value
        assert result.value["token_count"] == 5  # len("Hello")

    def test_detokenize(self):
        npu = self._setup_npu()
        result = npu.detokenize("test", [72, 101, 108, 108, 111])
        assert result.success is True
        assert "text" in result.value
        assert result.value["text"] == "Hello"


class TestNPUTraining:
    def _setup_npu(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="test", provider=provider, config=provider.metadata())
        npu._models["test"] = model
        npu._default_model = "test"
        return npu

    def test_train_step(self):
        npu = self._setup_npu()
        input_ids = [1, 2, 3, 4, 5]
        targets = [2, 3, 4, 5, 6]
        result = npu.train_step("test", input_ids, targets, lr=0.001)
        assert result.success is True
        assert "loss" in result.value
        assert "train_step_time_ms" in result.value


class TestNPUIoctl:
    def _setup_npu(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="test", provider=provider, config=provider.metadata())
        npu._models["test"] = model
        npu._default_model = "test"
        return npu

    def test_ioctl_info(self):
        npu = self._setup_npu()
        result = npu.ioctl("INFO")
        assert result.success is True
        assert result.value["device"] == "npu"

    def test_ioctl_set_default(self):
        npu = self._setup_npu()
        provider = MockProvider()
        model2 = NPUModel(name="other", provider=provider)
        npu._models["other"] = model2

        result = npu.ioctl("SET_DEFAULT", "other")
        assert result.success is True
        assert npu._default_model == "other"

    def test_ioctl_set_default_missing(self):
        npu = self._setup_npu()
        result = npu.ioctl("SET_DEFAULT", "nonexistent")
        assert result.success is False

    def test_ioctl_unknown(self):
        npu = self._setup_npu()
        with pytest.raises(ValueError, match="unknown ioctl"):
            npu.ioctl("DO_SOMETHING_WEIRD")

    def test_ioctl_generate(self):
        npu = self._setup_npu()
        result = npu.ioctl("GENERATE", "test", "Hello")
        assert result.success is True
        assert "text" in result.value

    def test_ioctl_forward(self):
        npu = self._setup_npu()
        result = npu.ioctl("FORWARD", "test", [1, 2, 3])
        assert result.success is True
        assert "logits" in result.value

    def test_ioctl_tokenize(self):
        npu = self._setup_npu()
        result = npu.ioctl("TOKENIZE", "test", "Hello")
        assert result.success is True
        assert "token_ids" in result.value

    def test_ioctl_embed(self):
        npu = self._setup_npu()
        result = npu.ioctl("EMBED", "test", "Hello")
        assert result.success is True
        assert "embedding" in result.value


class TestNPUStats:
    def test_inference_count(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="test", provider=provider)
        npu._models["test"] = model
        npu._default_model = "test"

        npu.forward("test", [1, 2, 3])
        npu.forward("test", [4, 5, 6])

        assert npu._total_inferences == 2
        assert model.inference_count == 2

    def test_token_count(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="test", provider=provider)
        npu._models["test"] = model
        npu._default_model = "test"

        npu.generate("test", "Hi", max_tokens=5)
        assert npu._total_tokens_generated > 0


class TestNPUModel:
    def test_npu_model_defaults(self):
        model = NPUModel(name="test", provider=MockProvider())
        assert model.name == "test"
        assert model.config == {}
        assert model.loaded_at == 0.0
        assert model.inference_count == 0
        assert model.total_tokens == 0

    def test_npu_model_with_config(self):
        provider = MockProvider()
        model = NPUModel(
            name="test",
            provider=provider,
            config=provider.metadata(),
            loaded_at=time.time(),
        )
        assert model.config["total_params"] == 1000
        assert model.config["vocab_size"] == 256


class TestNPUProfile:
    """Test NPU throughput profiling."""

    def _make_npu_with_model(self):
        npu = NPUDevice()
        provider = MockProvider()
        model = NPUModel(
            name="test_model",
            provider=provider,
            config={
                "total_params": 500_000_000,
                "n_layer": 24,
                "n_embed": 896,
                "n_head": 14,
                "n_kv_heads": 2,
                "vocab_size": 151936,
                "block_size": 4096,
                "device": "cpu",
            },
        )
        npu._models["test_model"] = model
        npu._default_model = "test_model"
        return npu

    def test_profile_returns_dict(self):
        npu = self._make_npu_with_model()
        result = npu.profile("test_model")
        assert isinstance(result, dict)
        assert "profiles" in result
        assert "architecture" in result

    def test_profile_architecture(self):
        npu = self._make_npu_with_model()
        result = npu.profile("test_model")
        arch = result["architecture"]
        assert arch["n_layer"] == 24
        assert arch["n_embed"] == 896
        assert arch["n_head"] == 14
        assert arch["total_params"] == 500_000_000

    def test_profile_batch_sizes(self):
        npu = self._make_npu_with_model()
        result = npu.profile("test_model", batch_sizes=[1, 8, 32])
        profiles = result["profiles"]
        assert len(profiles) == 3
        assert profiles[0]["batch_size"] == 1
        assert profiles[1]["batch_size"] == 8
        assert profiles[2]["batch_size"] == 32

    def test_profile_latency_increases_with_batch(self):
        npu = self._make_npu_with_model()
        result = npu.profile("test_model", batch_sizes=[1, 32])
        p1, p2 = result["profiles"]
        assert p2["latency_ms"] >= p1["latency_ms"]

    def test_profile_throughput_increases_with_batch(self):
        npu = self._make_npu_with_model()
        result = npu.profile("test_model", batch_sizes=[1, 8])
        p1, p8 = result["profiles"]
        assert p8["tokens_per_sec"] >= p1["tokens_per_sec"]

    def test_profile_memory_grows_with_batch(self):
        npu = self._make_npu_with_model()
        result = npu.profile("test_model", batch_sizes=[1, 64])
        p1, p64 = result["profiles"]
        assert p64["memory_mb"] > p1["memory_mb"]

    def test_profile_bottleneck_detected(self):
        npu = self._make_npu_with_model()
        result = npu.profile("test_model")
        for p in result["profiles"]:
            assert p["bottleneck"] in ("compute", "memory")

    def test_profile_no_model(self):
        npu = self._make_npu_with_model()
        result = npu.profile("nonexistent")
        assert "error" in result

    def test_profile_flops_per_token(self):
        npu = self._make_npu_with_model()
        result = npu.profile("test_model")
        assert result["flops_per_token"] > 0

    def test_profile_seq_len_affects_flops(self):
        npu = self._make_npu_with_model()
        r1 = npu.profile("test_model", seq_len=128)
        r2 = npu.profile("test_model", seq_len=512)
        assert r2["flops_per_token"] > r1["flops_per_token"]

    def test_profile_vm_device(self):
        from domains.shell.vm_devices import NPUVMDevice
        npu = self._make_npu_with_model()
        dev = NPUVMDevice(npu)
        result = dev.call("profile", "test_model", 256, "1,8")
        assert isinstance(result, dict)
        assert "profiles" in result
        assert len(result["profiles"]) == 2


class TestNPURefCounting:
    """Test unload-during-inference race condition protection."""

    def _make_npu_with_model(self):
        npu = NPUDevice()
        provider = MockProvider()
        model = NPUModel(name="m", provider=provider, config=provider.metadata())
        npu._models["m"] = model
        return npu

    def test_unload_refuses_during_active_inference(self):
        npu = self._make_npu_with_model()
        npu._ref_counts["m"] = 1
        result = npu.unload_model("m")
        assert not result.success
        assert "active operation" in result.error
        assert "m" in npu._models

    def test_unload_succeeds_when_no_refs(self):
        npu = self._make_npu_with_model()
        result = npu.unload_model("m")
        assert result.success
        assert "m" not in npu._models

    def test_ref_count_increments_decrements(self):
        npu = self._make_npu_with_model()
        assert npu._acquire_ref("m") is True
        assert npu._ref_counts["m"] == 1
        assert npu._acquire_ref("m") is True
        assert npu._ref_counts["m"] == 2
        npu._release_ref("m")
        assert npu._ref_counts["m"] == 1
        npu._release_ref("m")
        assert "m" not in npu._ref_counts

    def test_acquire_ref_nonexistent_model(self):
        npu = NPUDevice()
        assert npu._acquire_ref("nonexistent") is False

    def test_info_shows_active_refs(self):
        npu = self._make_npu_with_model()
        npu._ref_counts["m"] = 2
        info = npu.info()
        assert "active_refs" in info
        assert info["active_refs"]["m"] == 2

    def test_forward_acquires_and_releases_ref(self):
        """forward() should increment then decrement ref count."""
        npu = self._make_npu_with_model()
        assert npu._ref_counts.get("m", 0) == 0
        result = npu.forward("m", [1, 2, 3])
        assert result.success
        assert npu._ref_counts.get("m", 0) == 0  # released after completion

    def test_generate_acquires_and_releases_ref(self):
        npu = self._make_npu_with_model()
        assert npu._ref_counts.get("m", 0) == 0
        result = npu.generate("m", "hello", max_tokens=3)
        assert result.success
        assert npu._ref_counts.get("m", 0) == 0

    def test_embed_acquires_and_releases_ref(self):
        npu = self._make_npu_with_model()
        assert npu._ref_counts.get("m", 0) == 0
        result = npu.embed("m", "hello")
        assert result.success
        assert npu._ref_counts.get("m", 0) == 0

    def test_concurrent_forward_blocks_unload(self):
        """Two concurrent forwards each hold a ref — unload fails."""
        import threading
        npu = self._make_npu_with_model()
        barrier = threading.Barrier(3)
        results = []

        def slow_forward():
            npu._acquire_ref("m")
            barrier.wait()  # all 3 parties sync: threads + main
            barrier.wait()  # wait for unload attempt to complete
            npu._release_ref("m")
            results.append("done")

        t1 = threading.Thread(target=slow_forward)
        t2 = threading.Thread(target=slow_forward)
        t1.start()
        t2.start()
        barrier.wait()  # sync with both threads

        result = npu.unload_model("m")
        assert not result.success
        assert "2 active" in result.error

        barrier.wait()  # release threads
        t1.join()
        t2.join()
        assert len(results) == 2
        assert npu._ref_counts.get("m", 0) == 0


# ── Save/Load Checkpoint (Disk Persistence) Tests ────────────────────────

class TestNPUSaveLoadCheckpoint:
    def _make_npu_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="m", provider=provider, config=provider.metadata(), loaded_at=time.time())
        npu._models["m"] = model
        npu._default_model = "m"
        return npu

    @patch("domains.inference.slo_format.save_soul")
    def test_save_checkpoint_default_path(self, mock_save_soul):
        mock_save_soul.return_value = "/data/checkpoints/m.soul"
        npu = self._make_npu_with_model()
        with patch("os.makedirs"), patch("os.path.getsize", return_value=1000):
            result = npu.save_checkpoint("m", "")
        assert result.success
        assert result.value["saved"] == "m"
        assert "size_bytes" in result.value

    @patch("domains.inference.slo_format.save_soul")
    def test_save_checkpoint_custom_path(self, mock_save_soul):
        mock_save_soul.return_value = "/tmp/test.soul"
        npu = self._make_npu_with_model()
        with patch("os.path.getsize", return_value=2000):
            result = npu.save_checkpoint("m", "/tmp/test.soul")
        assert result.success
        assert result.value["path"] == "/tmp/test.soul"

    def test_save_checkpoint_no_model(self):
        npu = NPUDevice()
        npu.open()
        result = npu.save_checkpoint("nonexistent", "")
        assert not result.success
        assert "not loaded" in result.error

    def _make_soul_binary(self, weights=None):
        """Create a minimal valid .soul binary for testing."""
        import struct, json
        meta = json.dumps({"soul_name": "test"}).encode()
        json_len = len(meta)
        header = b"SOU\x00" + struct.pack("<I", 3) + struct.pack("<I", json_len) + meta
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

    def test_load_checkpoint(self):
        npu = self._make_npu_with_model()
        w1 = np.ones((8, 8), dtype=np.float32)
        w2 = np.zeros((4,), dtype=np.float32)
        soul_data = self._make_soul_binary({"w1": w1, "w2": w2})
        with patch("builtins.open", mock_open(read_data=soul_data)):
            result = npu.load_checkpoint("m", "/tmp/test.soul")
        assert result.success
        assert result.value["weights_restored"] == 2
        assert result.value["model"] == "m"

    def test_load_checkpoint_no_model(self):
        npu = NPUDevice()
        npu.open()
        result = npu.load_checkpoint("nonexistent", "/tmp/test.soul")
        assert not result.success
        assert "not loaded" in result.error

    def test_load_checkpoint_bad_magic(self):
        npu = self._make_npu_with_model()
        with patch("builtins.open", mock_open(read_data=b"BADDATA")):
            result = npu.load_checkpoint("m", "/tmp/bad.soul")
        assert not result.success
        assert "bad magic" in result.error

    def test_load_checkpoint_file_not_found(self):
        npu = self._make_npu_with_model()
        with patch("builtins.open", side_effect=FileNotFoundError("no file")):
            result = npu.load_checkpoint("m", "/tmp/missing.soul")
        assert not result.success
        assert "LOAD_CHECKPOINT failed" in result.error


# ── Quantize Tests ────────────────────────────────────────────────────────

class TestNPUQuantize:
    def _make_npu_with_params(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        m = provider._model
        m._params = {
            "w1": np.random.randn(32, 32).astype(np.float32),
            "w2": np.random.randn(16, 32).astype(np.float32),
        }
        model = NPUModel(name="m", provider=provider, config=provider.metadata(), loaded_at=time.time())
        npu._models["m"] = model
        npu._default_model = "m"
        return npu

    def test_quantize_int8(self):
        npu = self._make_npu_with_params()
        result = npu.quantize("m", 8)
        assert result.success
        assert result.value["bits"] == 8
        assert result.value["params_quantized"] >= 2
        assert result.value["memory_saved_mb"] >= 0

    def test_quantize_int4(self):
        npu = self._make_npu_with_params()
        result = npu.quantize("m", 4)
        assert result.success
        assert result.value["bits"] == 4
        assert result.value["compression_ratio"] > 1

    def test_quantize_invalid_bits(self):
        npu = self._make_npu_with_params()
        result = npu.quantize("m", 16)
        assert not result.success
        assert "4 or 8 bits" in result.error

    def test_quantize_no_model(self):
        npu = NPUDevice()
        npu.open()
        result = npu.quantize("nonexistent", 8)
        assert not result.success

    def test_quantize_preserves_shape(self):
        npu = self._make_npu_with_params()
        original_shapes = {k: v.shape for k, v in npu._models["m"].provider._model._params.items()}
        npu.quantize("m", 8)
        for key, shape in original_shapes.items():
            assert npu._models["m"].provider._model._params[key].shape == shape


# ── Clear Cache Tests ─────────────────────────────────────────────────────

class TestNPUClearCache:
    def _make_npu_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        provider._kv_cache = MagicMock()
        provider._kv_cache._cache = {"0": [np.zeros((1, 10, 64)), np.zeros((1, 10, 64))]}
        model = NPUModel(name="m", provider=provider, config=provider.metadata(),
                         loaded_at=time.time(), inference_count=5, total_tokens=100, total_forward_ms=50.0)
        npu._models["m"] = model
        npu._default_model = "m"
        return npu

    def test_clear_cache_resets_stats(self):
        npu = self._make_npu_with_model()
        result = npu.clear_cache("m")
        assert result.success
        assert result.value["stats_reset"] is True
        assert npu._models["m"].inference_count == 0
        assert npu._models["m"].total_tokens == 0

    def test_clear_cache_frees_memory(self):
        npu = self._make_npu_with_model()
        result = npu.clear_cache("m")
        assert result.success
        assert result.value["cache_freed_mb"] >= 0

    def test_clear_cache_no_model(self):
        npu = NPUDevice()
        npu.open()
        result = npu.clear_cache("nonexistent")
        assert not result.success


# ── Health Tests ──────────────────────────────────────────────────────────

class TestNPUHealth:
    def test_health_empty(self):
        npu = NPUDevice()
        npu.open()
        result = npu.health()
        assert result.success
        info = result.value
        assert info["device"] == "npu"
        assert info["models_loaded"] == 0
        assert info["total_inferences"] == 0
        assert "thread_count" in info

    def test_health_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="m", provider=provider, config=provider.metadata(),
                         loaded_at=time.time(), inference_count=3, total_tokens=50, total_forward_ms=30.0)
        npu._models["m"] = model
        npu._default_model = "m"
        result = npu.health()
        assert result.success
        info = result.value
        assert info["models_loaded"] == 1
        assert "m" in info["models"]
        assert info["models"]["m"]["inference_count"] == 3
        assert info["models"]["m"]["total_tokens"] == 50

    def test_health_process_memory(self):
        npu = NPUDevice()
        npu.open()
        result = npu.health()
        assert result.success
        assert result.value["process_memory_mb"] >= 0

    def test_health_ioctl(self):
        npu = NPUDevice()
        npu.open()
        result = npu.ioctl("HEALTH")
        assert result.success
        assert result.value["device"] == "npu"


# ── Batch Processing Tests ────────────────────────────────────────────────

class TestNPUBatch:
    def _make_npu_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="m", provider=provider, config=provider.metadata(), loaded_at=time.time())
        npu._models["m"] = model
        npu._default_model = "m"
        return npu

    def test_batch_multiple_prompts(self):
        npu = self._make_npu_with_model()
        result = npu.batch("m", ["Hello", "World", "Test"], max_tokens=5)
        assert result.success
        assert result.value["count"] == 3
        assert result.value["total_tokens"] > 0
        assert len(result.value["results"]) == 3
        for r in result.value["results"]:
            assert "text" in r
            assert "token_count" in r
            assert "latency_ms" in r

    def test_batch_single_prompt(self):
        npu = self._make_npu_with_model()
        result = npu.batch("m", ["Hello"], max_tokens=5)
        assert result.success
        assert result.value["count"] == 1

    def test_batch_empty_prompts(self):
        npu = self._make_npu_with_model()
        result = npu.batch("m", [], max_tokens=5)
        assert result.success
        assert result.value["count"] == 0

    def test_batch_no_model(self):
        npu = NPUDevice()
        npu.open()
        result = npu.batch("nonexistent", ["Hello"], 5)
        assert not result.success

    def test_batch_ref_count_cleanup(self):
        npu = self._make_npu_with_model()
        npu.batch("m", ["Hello", "World"], 5)
        assert npu._ref_counts.get("m", 0) == 0

    def test_batch_throughput(self):
        npu = self._make_npu_with_model()
        result = npu.batch("m", ["a", "b", "c", "d"], 5)
        assert result.success
        assert result.value["avg_tokens_per_sec"] >= 0


# ── Ioctl Dispatch Tests for New Ops ─────────────────────────────────────

class TestNPUNewIoctl:
    def _make_npu_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        provider._model._params["tok_emb.weight"] = np.random.randn(256, 64).astype(np.float32)
        model = NPUModel(name="m", provider=provider, config=provider.metadata(), loaded_at=time.time())
        npu._models["m"] = model
        npu._default_model = "m"
        return npu

    def _make_soul_binary(self, weights=None):
        import struct, json
        meta = json.dumps({"soul_name": "test"}).encode()
        json_len = len(meta)
        header = b"SOU\x00" + struct.pack("<I", 3) + struct.pack("<I", json_len) + meta
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

    @patch("domains.inference.slo_format.save_soul")
    def test_ioctl_save_checkpoint(self, mock_save_soul):
        mock_save_soul.return_value = "/tmp/test.soul"
        npu = self._make_npu_with_model()
        with patch("os.path.getsize", return_value=100):
            result = npu.ioctl("SAVE_CHECKPOINT", "m", "/tmp/test.soul")
        assert result.success

    def test_ioctl_load_checkpoint(self):
        npu = self._make_npu_with_model()
        w = np.ones((8, 8), dtype=np.float32)
        soul_data = self._make_soul_binary({"w": w})
        with patch("builtins.open", mock_open(read_data=soul_data)):
            result = npu.ioctl("LOAD_CHECKPOINT", "m", "/tmp/test.soul")
        assert result.success

    def test_ioctl_quantize(self):
        npu = self._make_npu_with_model()
        result = npu.ioctl("QUANTIZE", "m", 8)
        assert result.success

    def test_ioctl_clear_cache(self):
        npu = self._make_npu_with_model()
        result = npu.ioctl("CLEAR_CACHE", "m")
        assert result.success

    def test_ioctl_health(self):
        npu = NPUDevice()
        npu.open()
        result = npu.ioctl("HEALTH")
        assert result.success

    def test_ioctl_batch(self):
        npu = self._make_npu_with_model()
        result = npu.ioctl("BATCH", "m", ["Hello", "World"], 5)
        assert result.success

    def test_ioctl_attention_maps(self):
        npu = self._make_npu_with_model()
        result = npu.ioctl("ATTENTION_MAPS", "m", "Hello world", -1)
        assert result.success

    def test_ioctl_compare(self):
        npu = self._make_npu_with_model()
        # Need a second model
        provider2 = MockProvider()
        model2 = NPUModel(name="m2", provider=provider2, config=provider2.metadata(), loaded_at=time.time())
        npu._models["m2"] = model2
        result = npu.ioctl("COMPARE", "m", "m2", "Hello", 5)
        assert result.success

    def test_ioctl_layers(self):
        npu = self._make_npu_with_model()
        result = npu.ioctl("LAYERS", "m", -1)
        assert result.success

    def test_ioctl_benchmark(self):
        npu = self._make_npu_with_model()
        result = npu.ioctl("BENCHMARK", "m", None, 5)
        assert result.success


# ── Attention Maps Tests ─────────────────────────────────────────────────

class TestNPUAttentionMaps:
    def _make_npu_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        provider._model._params["tok_emb.weight"] = np.random.randn(256, 64).astype(np.float32)
        model = NPUModel(name="m", provider=provider, config=provider.metadata(), loaded_at=time.time())
        npu._models["m"] = model
        npu._default_model = "m"
        return npu

    def test_attention_maps_basic(self):
        npu = self._make_npu_with_model()
        result = npu.attention_maps("m", "Hello world")
        assert result.success
        info = result.value
        assert info["token_count"] > 0
        assert info["layers_extracted"] >= 0
        assert info["n_head"] > 0
        assert "attention" in info

    def test_attention_maps_specific_layer(self):
        npu = self._make_npu_with_model()
        result = npu.attention_maps("m", "Hello", layer=0)
        assert result.success

    def test_attention_maps_no_model(self):
        npu = NPUDevice()
        npu.open()
        result = npu.attention_maps("nonexistent", "Hello")
        assert not result.success

    def test_attention_maps_empty_text(self):
        npu = self._make_npu_with_model()
        result = npu.attention_maps("m", "")
        assert result.success
        assert result.value["token_count"] == 0

    def test_attention_maps_returns_per_head(self):
        npu = self._make_npu_with_model()
        result = npu.attention_maps("m", "Test")
        assert result.success
        for li, info in result.value.get("attention", {}).items():
            assert "per_head_avg" in info
            assert "shape" in info


# ── Model Comparison Tests ───────────────────────────────────────────────

class TestNPUCompare:
    def _make_npu_with_two_models(self):
        npu = NPUDevice()
        npu.open()
        p1 = MockProvider()
        p2 = MockProvider()
        m1 = NPUModel(name="a", provider=p1, config=p1.metadata(), loaded_at=time.time())
        m2 = NPUModel(name="b", provider=p2, config=p2.metadata(), loaded_at=time.time())
        npu._models["a"] = m1
        npu._models["b"] = m2
        npu._default_model = "a"
        return npu

    def test_compare_basic(self):
        npu = self._make_npu_with_two_models()
        result = npu.compare("a", "b", "Hello", 5)
        assert result.success
        info = result.value
        assert "models" in info
        assert "a" in info["models"]
        assert "b" in info["models"]
        assert "comparison" in info
        comp = info["comparison"]
        assert "speed_ratio" in comp
        assert "faster" in comp

    def test_compare_same_prompt(self):
        npu = self._make_npu_with_two_models()
        result = npu.compare("a", "b", "Test prompt")
        assert result.success
        assert result.value["prompt"] == "Test prompt"

    def test_compare_no_model(self):
        npu = NPUDevice()
        npu.open()
        result = npu.compare("x", "y", "Hello")
        assert not result.success

    def test_compare_model_stats(self):
        npu = self._make_npu_with_two_models()
        result = npu.compare("a", "b", "Hello", 5)
        for label in ["a", "b"]:
            m = result.value["models"][label]
            assert "forward_ms" in m
            assert "generate_ms" in m
            assert "tokens_generated" in m
            assert "memory_mb" in m
            assert "generated_text" in m


# ── Layer Introspection Tests ────────────────────────────────────────────

class TestNPULayers:
    def _make_npu_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="m", provider=provider, config=provider.metadata(), loaded_at=time.time())
        npu._models["m"] = model
        npu._default_model = "m"
        return npu

    def test_layers_basic(self):
        npu = self._make_npu_with_model()
        result = npu.layers("m", -1)
        assert result.success
        info = result.value
        assert "architecture" in info
        assert "layers" in info
        assert "total_params" in info

    def test_layers_specific_layer(self):
        npu = self._make_npu_with_model()
        result = npu.layers("m", 0)
        assert result.success

    def test_layers_no_model(self):
        npu = NPUDevice()
        npu.open()
        result = npu.layers("nonexistent")
        assert not result.success

    def test_layers_architecture_info(self):
        npu = self._make_npu_with_model()
        result = npu.layers("m")
        arch = result.value["architecture"]
        assert "n_layer" in arch
        assert "n_embed" in arch
        assert "n_head" in arch
        assert "ff_dim" in arch


# ── Benchmark Tests ──────────────────────────────────────────────────────

class TestNPUBenchmark:
    def _make_npu_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        model = NPUModel(name="m", provider=provider, config=provider.metadata(), loaded_at=time.time())
        npu._models["m"] = model
        npu._default_model = "m"
        return npu

    def test_benchmark_basic(self):
        npu = self._make_npu_with_model()
        result = npu.benchmark("m", prompt_lengths=[1, 5], max_tokens=3)
        assert result.success
        info = result.value
        assert "results" in info
        assert "summary" in info
        assert len(info["results"]) == 2

    def test_benchmark_default_lengths(self):
        npu = self._make_npu_with_model()
        result = npu.benchmark("m")
        assert result.success
        assert len(result.value["results"]) == 5  # default: [1, 10, 50, 100, 200]

    def test_benchmark_no_model(self):
        npu = NPUDevice()
        npu.open()
        result = npu.benchmark("nonexistent")
        assert not result.success

    def test_benchmark_summary(self):
        npu = self._make_npu_with_model()
        result = npu.benchmark("m", prompt_lengths=[1, 5], max_tokens=3)
        summary = result.value["summary"]
        assert "avg_tokens_per_sec" in summary
        assert "min_latency_ms" in summary
        assert "max_latency_ms" in summary

    def test_benchmark_ref_count_cleanup(self):
        npu = self._make_npu_with_model()
        npu.benchmark("m", prompt_lengths=[1], max_tokens=2)
        assert npu._ref_counts.get("m", 0) == 0

    def test_benchmark_per_result_fields(self):
        npu = self._make_npu_with_model()
        result = npu.benchmark("m", prompt_lengths=[1], max_tokens=3)
        r = result.value["results"][0]
        assert "prompt_tokens" in r
        assert "generated_tokens" in r
        assert "avg_latency_ms" in r
        assert "tokens_per_sec" in r
        assert "generated_sample" in r


# ── Dequantize Tests ─────────────────────────────────────────────────────

class TestNPUDequantize:
    def _make_npu_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        provider._model._params["tok_emb.weight"] = np.random.randn(256, 64).astype(np.float32)
        model = NPUModel(name="m", provider=provider, config=provider.metadata(), loaded_at=time.time())
        npu._models["m"] = model
        npu._default_model = "m"
        return npu

    def test_dequantize_not_quantized(self):
        npu = self._make_npu_with_model()
        result = npu.dequantize("m")
        assert result.success
        assert result.value["restored"] is False

    def test_quantize_then_dequantize(self):
        npu = self._make_npu_with_model()
        q_result = npu.quantize("m", 8)
        assert q_result.success
        assert q_result.value["params_quantized"] > 0

        d_result = npu.dequantize("m")
        assert d_result.success
        assert d_result.value["restored"] is True
        assert d_result.value["params_restored"] > 0

    def test_quantize_dequantize_roundtrip_weights_restored(self):
        npu = self._make_npu_with_model()
        m = npu._models["m"].provider._model
        original = m._params["tok_emb.weight"].copy()

        npu.quantize("m", 8)
        npu.dequantize("m")

        restored = m._params["tok_emb.weight"]
        np.testing.assert_array_almost_equal(restored, original, decimal=5)

    def test_dequantize_ioctl(self):
        npu = self._make_npu_with_model()
        npu.quantize("m", 8)
        result = npu.ioctl("DEQUANTIZE", "m")
        assert result.success

    def test_dequantize_no_original_weights(self):
        npu = self._make_npu_with_model()
        m = npu._models["m"].provider._model
        m._is_quantized = True
        m._original_weights = {}
        result = npu.dequantize("m")
        assert not result.success

    def test_quantize_sets_quantized_flag(self):
        npu = self._make_npu_with_model()
        m = npu._models["m"].provider._model
        assert not getattr(m, '_is_quantized', False)
        npu.quantize("m", 8)
        assert m._is_quantized

    def test_dequantize_clears_quantized_flag(self):
        npu = self._make_npu_with_model()
        npu.quantize("m", 8)
        m = npu._models["m"].provider._model
        assert m._is_quantized
        npu.dequantize("m")
        assert not m._is_quantized


# ── Quantized Inference Tests ────────────────────────────────────────────

class TestNPUQuantizedInference:
    def _make_npu_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        provider._model._params["tok_emb.weight"] = np.random.randn(256, 64).astype(np.float32)
        model = NPUModel(name="m", provider=provider, config=provider.metadata(), loaded_at=time.time())
        npu._models["m"] = model
        npu._default_model = "m"
        return npu

    def test_forward_after_quantize(self):
        npu = self._make_npu_with_model()
        npu.quantize("m", 8)
        result = npu.forward("m", [1, 2, 3, 4])
        assert result.success
        assert result.value["shape"] is not None

    def test_generate_after_quantize(self):
        npu = self._make_npu_with_model()
        npu.quantize("m", 8)
        result = npu.generate("m", "Hello", max_tokens=3)
        assert result.success
        assert result.value["token_count"] > 0

    def test_forward_restores_quantized_after(self):
        npu = self._make_npu_with_model()
        npu.quantize("m", 8)
        m = npu._models["m"].provider._model
        assert m._is_quantized
        npu.forward("m", [1, 2])
        assert m._is_quantized

    def test_generate_restores_quantized_after(self):
        npu = self._make_npu_with_model()
        npu.quantize("m", 8)
        m = npu._models["m"].provider._model
        assert m._is_quantized
        npu.generate("m", "Hi", max_tokens=2)
        assert m._is_quantized

    def test_ref_count_clean_after_quantized_forward(self):
        npu = self._make_npu_with_model()
        npu.quantize("m", 8)
        npu.forward("m", [1, 2, 3])
        assert npu._ref_counts.get("m", 0) == 0


# ── C Backend Routing ──────────────────────────────────────────────────────

class MockCTransformProvider:
    """Mock CTransformProvider for testing C backend routing."""

    def __init__(self):
        self._model = MockSloTransformer()
        self._model_id = "c-model"
        self._device = "cpu"
        self._tokenizer = MockTokenizer()

    @classmethod
    def from_slnc(cls, path, model_id="c-model"):
        p = cls()
        p._model_id = model_id
        return p

    def metadata(self):
        return {
            "model_id": self._model_id,
            "architecture": "NativeEngine",
            "total_params": 0,
            "n_layer": 2,
            "n_embed": 64,
            "n_head": 4,
            "vocab_size": 256,
            "max_seq_len": 2048,
            "device": "cpu",
            "quantized": False,
            "engine": "c",
        }

    def generate(self, prompt, max_tokens=50, **kwargs):
        return "c-generated"

    def tokenize(self, text):
        return [ord(c) % 256 for c in text]

    def detokenize(self, token_ids):
        return "".join(chr(t % 128) for t in token_ids)

    def embed(self, text, layer=-1):
        return np.random.randn(64).astype(np.float32)


class TestCBackendRouting:
    """Tests for the C backend routing in NPUDevice.load_model()."""

    @patch("domains.shell.kernel_npu.NPUDevice._load_c_provider")
    def test_load_model_backend_c_calls_c_provider(self, mock_load_c):
        mock_load_c.return_value = MockCTransformProvider()
        npu = NPUDevice()
        npu.open()

        result = npu.load_model("c-model", "model.slnc", backend="c")

        assert result.success is True
        assert result.value["backend"] == "c"
        assert result.value["model"] == "c-model"
        mock_load_c.assert_called_once_with("c-model", "model.slnc", "model.slnc")

    @patch("domains.shell.kernel_npu.NPUDevice._load_numpy_provider")
    def test_load_model_default_c_falls_back_to_numpy(self, mock_load_numpy):
        mock_load_numpy.return_value = MockProvider()
        npu = NPUDevice()
        npu.open()

        # C fails (no mock for _load_c_provider), falls back to numpy
        result = npu.load_model("np-model", "model.slnc")

        assert result.success is True
        assert result.value["backend"] == "numpy"
        mock_load_numpy.assert_called_once()

    @patch("domains.shell.kernel_npu.NPUDevice._load_numpy_provider")
    def test_load_model_backend_numpy_explicit(self, mock_load_numpy):
        mock_load_numpy.return_value = MockProvider()
        npu = NPUDevice()
        npu.open()

        result = npu.load_model("np-model", "model.slnc", backend="numpy")

        assert result.success is True
        assert result.value["backend"] == "numpy"
        mock_load_numpy.assert_called_once()

    def test_load_c_provider_rejects_non_slnc(self):
        npu = NPUDevice()
        npu.open()

        with pytest.raises(ValueError, match="C backend only supports .slnc"):
            npu._load_c_provider("m", "model.safetensors", "model.safetensors")

    def test_load_c_provider_accepts_slnc(self):
        npu = NPUDevice()
        with patch("domains.inference.ct_provider.CTransformProvider.from_slnc") as mock_from:
            mock_from.return_value = MockCTransformProvider()
            provider = npu._load_c_provider("m", "model.slnc", "model.slnc")
            mock_from.assert_called_once_with("model.slnc", model_id="m")
            assert provider._model_id == "c-model"

    def test_load_numpy_provider_routes_slnc(self):
        npu = NPUDevice()
        with patch("domains.inference.slonet_provider.SlonetChatProvider.from_slnc") as mock_from:
            mock_from.return_value = MockProvider()
            provider = npu._load_numpy_provider("m", "model.slnc", "model.slnc", {})
            mock_from.assert_called_once()

    def test_load_numpy_provider_rejects_non_slnc(self):
        npu = NPUDevice()
        with pytest.raises(AttributeError):
            npu._load_numpy_provider("m", "model.safetensors", "model.safetensors", {})

    def test_c_backend_model_stored_in_models_dict(self):
        npu = NPUDevice()
        npu.open()
        with patch("domains.shell.kernel_npu.NPUDevice._load_c_provider") as mock_load:
            mock_load.return_value = MockCTransformProvider()
            npu.load_model("c-test", "model.slnc", backend="c")
            assert "c-test" in npu._models
            assert npu._models["c-test"].provider._model_id == "c-model"

    def test_c_backend_becomes_default_if_first(self):
        npu = NPUDevice()
        npu.open()
        with patch("domains.shell.kernel_npu.NPUDevice._load_c_provider") as mock_load:
            mock_load.return_value = MockCTransformProvider()
            npu.load_model("c-first", "model.slnc", backend="c")
            assert npu._default_model == "c-first"

    @patch("domains.shell.kernel_npu.NPUDevice._load_numpy_provider")
    @patch("domains.shell.kernel_npu.NPUDevice._load_c_provider")
    def test_c_backend_falls_back_to_numpy_on_failure(self, mock_c, mock_numpy):
        mock_c.side_effect = RuntimeError("C engine not available")
        mock_numpy.return_value = MockProvider()
        npu = NPUDevice()
        npu.open()

        result = npu.load_model("fallback", "model.slnc", backend="c")

        assert result.success is True
        assert result.value["backend"] == "numpy"
        mock_c.assert_called_once()
        mock_numpy.assert_called_once()

    def test_unknown_backend_raises(self):
        npu = NPUDevice()
        npu.open()
        with pytest.raises(ValueError, match="Unknown backend"):
            npu._load_provider("m", "model.slnc", "model.slnc", "invalid", {})


# ── HuggingFace Provider Tests ────────────────────────────────────────────


class _NullContext:
    """Minimal context manager standing in for torch.no_grad()."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeTorchTensor(np.ndarray):
    """Minimal torch.Tensor stand-in for _HuggingFaceProvider tests."""

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def long(self):
        return self

    def to(self, *args, **kwargs):
        return self


def _make_fake_torch():
    fake = types.ModuleType("torch")
    fake.Tensor = FakeTorchTensor
    fake.from_numpy = staticmethod(lambda arr: np.asarray(arr).view(FakeTorchTensor))
    fake.no_grad = staticmethod(lambda: _NullContext())
    return fake


class TestHuggingFaceProvider:
    def test_metadata(self):
        from domains.shell.kernel_npu import _HuggingFaceProvider
        tok = types.SimpleNamespace(vocab_size=1000)
        prov = _HuggingFaceProvider(model=object(), tokenizer=tok,
                                    model_id="m1", device="cpu")
        meta = prov.metadata()
        assert meta["model_id"] == "m1"
        assert meta["device"] == "cpu"
        assert meta["vocab_size"] == 1000

    def test_metadata_default_vocab(self):
        from domains.shell.kernel_npu import _HuggingFaceProvider
        prov = _HuggingFaceProvider(model=object(), tokenizer=object(),
                                    model_id="m1", device="cpu")
        assert prov.metadata()["vocab_size"] == 0

    def test_call_missing_input_ids(self):
        from domains.shell.kernel_npu import _HuggingFaceProvider
        torch = _make_fake_torch()
        prov = _HuggingFaceProvider(model=object(), tokenizer=object(),
                                    model_id="m1", device="cpu")
        with patch.dict(sys.modules, {"torch": torch}):
            with pytest.raises(ValueError, match="input_ids"):
                prov({"x": 1})

    def test_call_ndarray_tensor_logits(self):
        from domains.shell.kernel_npu import _HuggingFaceProvider
        torch = _make_fake_torch()
        output = types.SimpleNamespace()

        def _model(ids):
            output.logits = np.random.randn(1, 3).astype(np.float32).view(FakeTorchTensor)
            return output

        prov = _HuggingFaceProvider(model=_model, tokenizer=object(),
                                    model_id="m1", device="cpu")
        with patch.dict(sys.modules, {"torch": torch}):
            out = prov({"input_ids": np.array([[1, 2, 3]], dtype=np.int64)})
        assert isinstance(out, np.ndarray)
        assert out.shape == (1, 3)

    def test_call_ndarray_numpy_logits(self):
        from domains.shell.kernel_npu import _HuggingFaceProvider
        torch = _make_fake_torch()
        output = types.SimpleNamespace()

        def _model(ids):
            output.logits = np.random.randn(1, 3).astype(np.float32)
            return output

        prov = _HuggingFaceProvider(model=_model, tokenizer=object(),
                                    model_id="m1", device="cpu")
        with patch.dict(sys.modules, {"torch": torch}):
            out = prov({"input_ids": np.array([[1, 2, 3]], dtype=np.int64)})
        assert out.shape == (1, 3)

    def test_generate_numpy(self):
        from domains.shell.kernel_npu import _HuggingFaceProvider
        torch = _make_fake_torch()

        class _Tok:
            eos_token_id = 50256

            def __call__(self, prompt, **kwargs):
                return {"input_ids": np.asarray([1, 2, 3], dtype=np.int64).reshape(1, -1).view(FakeTorchTensor)}

        class _Gen:
            def generate(self, input_ids, **kwargs):
                return np.asarray([1, 2, 3, 9, 9], dtype=np.int64).reshape(1, -1).view(FakeTorchTensor)

        prov = _HuggingFaceProvider(model=_Gen(), tokenizer=_Tok(),
                                    model_id="m1", device="cpu")
        with patch.dict(sys.modules, {"torch": torch}):
            out = prov.generate_numpy("hi", max_tokens=2, temperature=0.7)
        assert out == [9, 9]

    def test_tokenize_detokenize(self):
        from domains.shell.kernel_npu import _HuggingFaceProvider
        tok = types.SimpleNamespace(encode=lambda t: [1, 2],
                                    decode=lambda ids: "decoded")
        prov = _HuggingFaceProvider(model=object(), tokenizer=tok,
                                    model_id="m1", device="cpu")
        assert prov.tokenize("hi") == [1, 2]
        assert prov.detokenize([1, 2]) == "decoded"


class _FakeHFAutoTokenizer:
    """Minimal AutoTokenizer stand-in for _load_huggingface_provider."""

    vocab_size = 1000
    eos_token_id = 0

    @classmethod
    def from_pretrained(cls, model_id):
        return cls()


class _FakeHFAutoModel:
    """Minimal AutoModelForCausalLM stand-in for _load_huggingface_provider."""

    def __init__(self):
        self._device = None

    def to(self, device):
        self._device = device
        return self

    def eval(self):
        return self

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls()


class TestHuggingFaceLoad:
    def test_load_success(self):
        fake = types.ModuleType("transformers")
        fake.AutoModelForCausalLM = _FakeHFAutoModel
        fake.AutoTokenizer = _FakeHFAutoTokenizer
        npu = NPUDevice()
        with patch.dict(sys.modules, {"transformers": fake}):
            prov = npu._load_huggingface_provider("m", "huggingface:gpt2", {"device": "cpu"})
        assert prov._model_id == "gpt2"
        assert prov._device == "cpu"
        assert prov._model._device == "cpu"

    def test_load_transformers_missing(self):
        npu = NPUDevice()
        with patch.dict(sys.modules, {"transformers": None}):
            with pytest.raises(ValueError, match="transformers not installed"):
                npu._load_huggingface_provider("m", "huggingface:gpt2", {})


class TestLoadProviderRouting:
    def test_provider_routes_slnc_to_c(self):
        npu = NPUDevice()
        with patch.object(npu, "_load_c_provider", return_value=MockCTransformProvider()):
            prov, backend = npu._load_provider("m", "model.slnc", "model.slnc")
        assert backend == "c"

    def test_provider_routes_huggingface(self):
        npu = NPUDevice()
        with patch.object(npu, "_load_huggingface_provider", return_value=object()):
            prov, backend = npu._load_provider("m", "huggingface:gpt2", "hf")
        assert backend == "huggingface"

    def test_provider_rejects_unknown_source(self):
        npu = NPUDevice()
        with pytest.raises(ValueError, match="Unknown backend for"):
            npu._load_provider("m", "model.bin", "model.bin")


class TestNPULoadErrors:
    def test_load_model_no_source(self):
        npu = NPUDevice()
        result = npu.load_model("m", "")
        assert not result.success
        assert "no source path provided" in result.error

    def test_load_model_provider_error(self):
        npu = NPUDevice()
        with patch.object(npu, "_load_provider", side_effect=RuntimeError("boom")):
            result = npu.load_model("m", "model.slnc")
        assert not result.success
        assert "load_model failed" in result.error

    def test_unload_default_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        npu._models["m"] = NPUModel(name="m", provider=provider)
        npu._default_model = "m"
        result = npu.unload_model("")
        assert result.success
        assert "m" not in npu._models
        assert npu._default_model == ""


class _RawInner:
    """Bare model object — no forward_pass or generate_numpy."""


class RawModelProvider:
    """Provider whose _model lacks forward_pass/generate_numpy — exercises
    the else branches in forward/generate/batch/compare/benchmark."""

    def __init__(self, inner=None):
        self._model = inner if inner is not None else _RawInner()
        self._tokenizer = MockTokenizer()
        self._model_id = "raw"
        self._device = "cpu"

    def forward_numpy(self, input_ids):
        return np.random.randn(input_ids.shape[0], input_ids.shape[1], 256).astype(np.float32)

    def generate_numpy(self, input_ids, max_new_tokens=10, **kwargs):
        return np.random.randint(0, 256, (input_ids.shape[0], 2), dtype=np.int64)

    def tokenize(self, text):
        return self._tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self._tokenizer.decode(token_ids)

    def embed(self, text, layer=-1):
        return np.random.randn(64).astype(np.float32)

    def metadata(self):
        return {
            "model_id": self._model_id,
            "total_params": 100,
            "n_layer": 2,
            "n_embed": 64,
            "n_head": 4,
            "vocab_size": 256,
        }


class TestNPUProviderFallbacks:
    def _make_npu(self):
        npu = NPUDevice()
        npu.open()
        provider = RawModelProvider()
        npu._models["raw"] = NPUModel(name="raw", provider=provider,
                                      config=provider.metadata())
        npu._default_model = "raw"
        return npu

    def test_forward_uses_forward_numpy(self):
        npu = self._make_npu()
        result = npu.forward("raw", [1, 2, 3])
        assert result.success
        assert result.value["engine"] == "numpy"

    def test_generate_uses_provider_generate_numpy(self):
        npu = self._make_npu()
        result = npu.generate("raw", "hello", max_tokens=4)
        assert result.success
        assert "text" in result.value

    def test_batch_uses_provider_generate_numpy(self):
        npu = self._make_npu()
        result = npu.batch("raw", ["a", "b"], max_tokens=4)
        assert result.success
        assert result.value["count"] == 2

    def test_compare_uses_provider_generate_numpy(self):
        npu = NPUDevice()
        npu.open()
        inner = types.SimpleNamespace(
            forward_pass=lambda ids: ForwardPassResult(
                logits=np.random.randn(1, 3, 256).astype(np.float32), engine="numpy"))
        p1 = RawModelProvider(inner=inner)
        p2 = RawModelProvider(inner=inner)
        npu._models["a"] = NPUModel(name="a", provider=p1, config=p1.metadata())
        npu._models["b"] = NPUModel(name="b", provider=p2, config=p2.metadata())
        result = npu.compare("a", "b", "hi", 3)
        assert result.success

    def test_benchmark_uses_provider_generate_numpy(self):
        npu = self._make_npu()
        result = npu.benchmark("raw", [1, 2])
        assert result.success
        assert len(result.value["results"]) == 2


class TestNPUMissingModelErrors:
    def _setup(self):
        npu = NPUDevice()
        npu.open()
        return npu

    def test_embed_missing_model(self):
        result = self._setup().embed("nope", "hi")
        assert not result.success

    def test_tokenize_missing_model(self):
        result = self._setup().tokenize("nope", "hi")
        assert not result.success

    def test_detokenize_missing_model(self):
        result = self._setup().detokenize("nope", [1, 2])
        assert not result.success

    def test_train_step_missing_model(self):
        result = self._setup().train_step("nope", [1], [2])
        assert not result.success

    def test_dequantize_missing_model(self):
        result = self._setup().dequantize("nope")
        assert not result.success


def _build_soul_binary(weights=None):
    """Create a minimal valid .soul binary for testing."""
    import struct
    import json
    meta = json.dumps({"soul_name": "test"}).encode()
    json_len = len(meta)
    header = b"SOU\x00" + struct.pack("<I", 3) + struct.pack("<I", json_len) + meta
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


class TestNPUCheckpointErrors:
    def _make_npu_with_model(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        npu._models["m"] = NPUModel(name="m", provider=provider,
                                    config=provider.metadata(), loaded_at=time.time())
        npu._default_model = "m"
        return npu

    @patch("domains.inference.slo_format.save_soul", side_effect=RuntimeError("disk full"))
    def test_save_checkpoint_error(self, mock_save):
        npu = self._make_npu_with_model()
        result = npu.save_checkpoint("m", "/tmp/x.soul")
        assert not result.success
        assert "SAVE_CHECKPOINT failed" in result.error

    def test_load_checkpoint_applies_state_dict(self):
        npu = self._make_npu_with_model()
        provider = npu._models["m"].provider
        provider.load_state_dict = lambda weights: setattr(provider, "loaded", weights)
        soul_data = _build_soul_binary({"w1": np.ones((8, 8), dtype=np.float32)})
        with patch("builtins.open", mock_open(read_data=soul_data)):
            result = npu.load_checkpoint("m", "/tmp/x.soul")
        assert result.success
        assert result.value["weights_restored"] == 1
        assert provider.loaded["w1"].shape == (8, 8)

    def test_load_checkpoint_generic_error(self):
        npu = self._make_npu_with_model()
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = npu.load_checkpoint("m", "/tmp/x.soul")
        assert not result.success
        assert "LOAD_CHECKPOINT failed" in result.error


class TestNPUQuantizeSkips:
    def test_quantize_skips_non_array_params(self):
        npu = NPUDevice()
        npu.open()
        provider = MockProvider()
        m = provider._model
        m._params = {"w1": np.random.randn(32, 32).astype(np.float32)}
        m._params["meta"] = "not-an-array"
        npu._models["m"] = NPUModel(name="m", provider=provider,
                                    config=provider.metadata())
        npu._default_model = "m"
        result = npu.quantize("m", 8)
        assert result.success
        assert result.value["params_quantized"] == 1


class _FakePsutil:
    class Process:
        @staticmethod
        def memory_info():
            return types.SimpleNamespace(rss=10 * 1024 * 1024)


class TestNPUPsutil:
    def test_psutil_available_branch(self):
        kp = importlib.import_module("domains.shell.kernel_npu")
        fake = _FakePsutil()
        with patch.dict(sys.modules, {"psutil": fake}):
            reloaded = importlib.reload(kp)
            assert reloaded._HAS_PSUTIL is True
        try:
            npu = NPUDevice()
            npu.open()
            inner = types.SimpleNamespace(
                forward_pass=lambda ids: ForwardPassResult(
                    logits=np.random.randn(1, 3, 256).astype(np.float32), engine="numpy"))
            p1 = RawModelProvider(inner=inner)
            p2 = RawModelProvider(inner=inner)
            npu._models["a"] = NPUModel(name="a", provider=p1, config=p1.metadata())
            npu._models["b"] = NPUModel(name="b", provider=p2, config=p2.metadata())
            result = npu.compare("a", "b", "hi", 3)
            assert result.success
            assert result.value["models"]["a"]["memory_mb"] > 0
        finally:
            with patch.dict(sys.modules, {"psutil": None}):
                importlib.reload(kp)
            assert kp._HAS_PSUTIL is False
