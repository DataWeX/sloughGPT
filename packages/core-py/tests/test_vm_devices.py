"""
Comprehensive tests for domains.shell.vm_devices.

Pure-logic tests for all device abstractions: TensorDevice, PythonExecDevice,
SlonetDevice, MultimodalDevice, EngineDevice, SlonetTrainingDevice, NPUVMDevice.
No mocks for external APIs — only real numpy and Python operations.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from domains.shell.vm_devices import (
    TensorDevice,
    PythonExecDevice,
    SlonetDevice,
    MultimodalDevice,
    EngineDevice,
    SlonetTrainingDevice,
    NPUVMDevice,
)
from domains.shell.vm import DeviceFault


# =============================================================================
# TensorDevice
# =============================================================================

class TestTensorDevice:
    def _dev(self, weights=None):
        return TensorDevice(weights=weights)

    def test_info_returns_type_and_ops(self):
        d = self._dev()
        info = d.info()
        assert info["type"] == "tensor"
        assert "matmul" in info["ops"]
        assert "relu" in info["ops"]
        assert isinstance(info["weight_names"], list)

    def test_info_lists_weight_names(self):
        d = self._dev(weights={"w": np.array([1, 2])})
        assert "w" in d.info()["weight_names"]

    def test_call_unknown_op_raises(self):
        d = self._dev()
        with pytest.raises(DeviceFault, match="unknown op"):
            d.call("nonexistent")

    # -- to_arr conversions --
    def test_to_arr_ndarray_passthrough(self):
        d = self._dev()
        arr = np.array([1.0, 2.0])
        assert np.array_equal(d._to_arr(arr), arr)

    def test_to_arr_int_scalar(self):
        d = self._dev()
        result = d._to_arr(5)
        assert isinstance(result, np.float64)
        assert result == 5.0

    def test_to_arr_float_scalar(self):
        d = self._dev()
        result = d._to_arr(3.14)
        assert result == pytest.approx(3.14)

    def test_to_arr_list(self):
        d = self._dev()
        result = d._to_arr([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert list(result) == [1.0, 2.0, 3.0]

    def test_to_arr_json_string_list(self):
        d = self._dev()
        result = d._to_arr("[1, 2, 3]")
        assert isinstance(result, np.ndarray)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_to_arr_invalid_json_string_falls_back(self):
        d = self._dev()
        with pytest.raises(ValueError):
            d._to_arr("not json")

    def test_to_arr_nested_list(self):
        d = self._dev()
        result = d._to_arr([[1, 2], [3, 4]])
        assert result.shape == (2, 2)

    # -- matmul --
    def test_matmul_vectors(self):
        d = self._dev()
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = d.call("matmul", a, b)
        assert result == pytest.approx(32.0)  # 1*4 + 2*5 + 3*6

    def test_matmul_matrices(self):
        d = self._dev()
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = d.call("matmul", a, b)
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(result, expected)

    def test_matmul_with_lists(self):
        d = self._dev()
        result = d.call("matmul", [[1, 2]], [[3], [4]])
        assert result == pytest.approx(11.0)

    # -- relu --
    def test_relu_positive(self):
        d = self._dev()
        result = d.call("relu", np.array([1, 2, 3]))
        assert np.array_equal(result, [1, 2, 3])

    def test_relu_negative(self):
        d = self._dev()
        result = d.call("relu", np.array([-1, -2, 0]))
        assert np.array_equal(result, [0, 0, 0])

    def test_relu_mixed(self):
        d = self._dev()
        result = d.call("relu", np.array([-3, 1, -1, 5]))
        assert np.array_equal(result, [0, 1, 0, 5])

    # -- softmax --
    def test_softmax_sums_to_one(self):
        d = self._dev()
        x = np.array([1.0, 2.0, 3.0])
        result = d.call("softmax", x)
        assert result.sum() == pytest.approx(1.0)

    def test_softmax_all_equal(self):
        d = self._dev()
        x = np.array([5.0, 5.0, 5.0])
        result = d.call("softmax", x)
        assert np.allclose(result, [1 / 3, 1 / 3, 1 / 3])

    def test_softmax_large_values_stable(self):
        d = self._dev()
        x = np.array([1000.0, 1001.0, 1002.0])
        result = d.call("softmax", x)
        assert result.sum() == pytest.approx(1.0)
        assert all(np.isfinite(result))

    # -- sigmoid --
    def test_sigmoid_zero(self):
        d = self._dev()
        assert d.call("sigmoid", 0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        d = self._dev()
        assert d.call("sigmoid", 100.0) == pytest.approx(1.0)

    def test_sigmoid_large_negative(self):
        d = self._dev()
        assert d.call("sigmoid", -100.0) == pytest.approx(0.0)

    def test_sigmoid_array(self):
        d = self._dev()
        result = d.call("sigmoid", np.array([0, 0, 0]))
        assert np.allclose(result, [0.5, 0.5, 0.5])

    # -- tanh --
    def test_tanh_zero(self):
        d = self._dev()
        assert d.call("tanh", 0.0) == pytest.approx(0.0)

    def test_tanh_positive(self):
        d = self._dev()
        assert d.call("tanh", 100.0) == pytest.approx(1.0)

    def test_tanh_negative(self):
        d = self._dev()
        assert d.call("tanh", -100.0) == pytest.approx(-1.0)

    # -- add --
    def test_add(self):
        d = self._dev()
        result = d.call("add", np.array([1, 2]), np.array([3, 4]))
        assert np.array_equal(result, [4, 6])

    def test_add_scalar(self):
        d = self._dev()
        result = d.call("add", 5, 3)
        assert result == pytest.approx(8.0)

    # -- mul --
    def test_mul(self):
        d = self._dev()
        result = d.call("mul", np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert np.array_equal(result, [4, 10, 18])

    # -- sub --
    def test_sub(self):
        d = self._dev()
        result = d.call("sub", np.array([10, 20]), np.array([3, 7]))
        assert np.array_equal(result, [7, 13])

    # -- neg --
    def test_neg(self):
        d = self._dev()
        result = d.call("neg", np.array([1, -2, 3]))
        assert np.array_equal(result, [-1, 2, -3])

    # -- abs --
    def test_abs(self):
        d = self._dev()
        result = d.call("abs", np.array([-5, 3, -1]))
        assert np.array_equal(result, [5, 3, 1])

    # -- sum --
    def test_sum(self):
        d = self._dev()
        assert d.call("sum", np.array([1, 2, 3, 4])) == pytest.approx(10.0)

    # -- mean --
    def test_mean(self):
        d = self._dev()
        assert d.call("mean", np.array([1, 2, 3, 4])) == pytest.approx(2.5)

    # -- max --
    def test_max(self):
        d = self._dev()
        assert d.call("max", np.array([1, 5, 3])) == pytest.approx(5.0)

    # -- argmax --
    def test_argmax(self):
        d = self._dev()
        assert d.call("argmax", np.array([1, 5, 3])) == 1

    # -- norm --
    def test_norm(self):
        d = self._dev()
        result = d.call("norm", np.array([3, 4]))
        assert result == pytest.approx(5.0)

    # -- load / store weights --
    def test_store_and_load_weight(self):
        d = self._dev()
        d.call("store", "myweight", np.array([1, 2, 3]))
        result = d.call("load", "myweight")
        assert np.array_equal(result, [1, 2, 3])

    def test_store_weight_converts(self):
        d = self._dev()
        d.call("store", "w", [10, 20])
        result = d.call("load", "w")
        assert result.dtype == np.float64

    def test_load_missing_weight_raises(self):
        d = self._dev()
        with pytest.raises(DeviceFault, match="no weight"):
            d.call("load", "missing")

    def test_weights_independent_instances(self):
        d1 = self._dev()
        d2 = self._dev()
        d1.call("store", "w", [1])
        with pytest.raises(DeviceFault):
            d2.call("load", "w")

    # -- shape --
    def test_shape(self):
        d = self._dev()
        result = d.call("shape", np.zeros((3, 4)))
        assert result == [3, 4]

    def test_shape_vector(self):
        d = self._dev()
        result = d.call("shape", np.zeros(5))
        assert result == [5]

    # -- zeros --
    def test_zeros(self):
        d = self._dev()
        result = d.call("zeros", 3, 4)
        assert result.shape == (3, 4)
        assert result.dtype == np.float64
        assert np.allclose(result, 0.0)

    def test_zeros_non_numeric_fallback(self):
        d = self._dev()
        result = d.call("zeros", "a", "b")
        assert result.shape == (1, 1)

    # -- randn --
    def test_randn_shape(self):
        d = self._dev()
        result = d.call("randn", 5, 6)
        assert result.shape == (5, 6)
        assert result.dtype == np.float64

    def test_randn_different_each_call(self):
        d = self._dev()
        r1 = d.call("randn", 1, 10)
        r2 = d.call("randn", 1, 10)
        assert not np.array_equal(r1, r2)

    # -- forward (MLP) --
    def test_forward_requires_w1_w2(self):
        d = self._dev()
        with pytest.raises(DeviceFault, match="requires w1, w2"):
            d.call("forward", np.array([1, 2]))

    def test_forward_without_biases(self):
        d = self._dev(weights={
            "w1": np.array([[0.5, 0.5], [0.5, 0.5]]),
            "w2": np.array([[0.5, 0.5]]),
        })
        result = d.call("forward", np.array([1.0, 2.0]))
        assert result.ndim == 1
        assert result.sum() == pytest.approx(1.0)  # softmax output

    def test_forward_with_biases(self):
        d = self._dev(weights={
            "w1": np.array([[0.5, 0.5], [0.5, 0.5]]),
            "b1": np.array([0.1, 0.1]),
            "w2": np.array([[0.5, 0.5]]),
            "b2": np.array([0.05]),
        })
        result = d.call("forward", np.array([1.0, 2.0]))
        assert result.ndim == 1
        assert result.sum() == pytest.approx(1.0)

    def test_forward_stable_with_large_input(self):
        d = self._dev(weights={
            "w1": np.ones((2, 2)),
            "w2": np.ones((1, 2)),
        })
        result = d.call("forward", np.array([1000.0, 2000.0]))
        assert result.sum() == pytest.approx(1.0)
        assert all(np.isfinite(result))


# =============================================================================
# PythonExecDevice
# =============================================================================

class TestPythonExecDevice:
    def _dev(self):
        return PythonExecDevice()

    def test_info(self):
        d = self._dev()
        info = d.info()
        assert info["type"] == "python_exec"
        assert "eval" in info["ops"]
        assert "call" in info["ops"]

    def test_eval_arithmetic(self):
        d = self._dev()
        assert d.call("eval", "2 + 3") == 5

    def test_eval_accesses_scope(self):
        d = self._dev()
        d.call("set", "x", 10)
        assert d.call("eval", "x + 5") == 15

    def test_eval_no_dangerous_builtins(self):
        d = self._dev()
        with pytest.raises(Exception):
            d.call("eval", "__import__('os')")

    def test_call_builtin_len(self):
        d = self._dev()
        assert d.call("call", "len", [1, 2, 3]) == 3

    def test_call_builtin_min_max_sum(self):
        d = self._dev()
        assert d.call("call", "min", [5, 3, 8]) == 3
        assert d.call("call", "max", [5, 3, 8]) == 8
        assert d.call("call", "sum", [1, 2, 3]) == 6

    def test_call_scope_function(self):
        d = self._dev()
        d.call("set", "double", lambda x: x * 2)
        assert d.call("call", "double", 7) == 14

    def test_call_unknown_raises(self):
        d = self._dev()
        with pytest.raises(DeviceFault, match="unknown callable"):
            d.call("call", "nonexistent_func")

    def test_import_numpy(self):
        d = self._dev()
        mod = d.call("import", "math")
        assert mod.pi == pytest.approx(math.pi)
        assert "math" in d.call("scope")

    def test_exec_side_effect(self):
        d = self._dev()
        d.call("exec", "result = 21 * 2")
        assert d.call("get", "result") == 42

    def test_exec_returns_none(self):
        d = self._dev()
        assert d.call("exec", "x = 1") is None

    def test_set_and_get(self):
        d = self._dev()
        d.call("set", "name", "alice")
        assert d.call("get", "name") == "alice"

    def test_get_undefined_raises(self):
        d = self._dev()
        with pytest.raises(DeviceFault, match="undefined"):
            d.call("get", "nope")

    def test_scope_returns_copy(self):
        d = self._dev()
        d.call("set", "a", 1)
        s1 = d.call("scope")
        s1["b"] = 999
        s2 = d.call("scope")
        assert "b" not in s2

    def test_unknown_op_raises(self):
        d = self._dev()
        with pytest.raises(DeviceFault, match="unknown op"):
            d.call("delete")

    def test_np_in_scope(self):
        d = self._dev()
        assert d.call("eval", "np.array([1,2]).sum()") == 3


# =============================================================================
# EngineDevice (no real engine needed — use simple callable)
# =============================================================================

class TestEngineDevice:
    def _dev(self, engine_name="fake"):
        class FakeEngine:
            def generate(self, prompt, max_tokens=50, temperature=1.0, **kw):
                return f"response:{prompt}"
        return EngineDevice(FakeEngine(), engine_name=engine_name)

    def test_generate(self):
        d = self._dev()
        result = d.call("generate", "hello")
        assert result == "response:hello"

    def test_generate_with_kwargs(self):
        d = self._dev()
        result = d.call("generate", "hi", 10)
        assert result == "response:hi"

    def test_info(self):
        d = self._dev(engine_name="myengine")
        info = d.call("info")
        assert info["engine"] == "myengine"
        assert info["type"] == "FakeEngine"

    def test_unknown_op_raises(self):
        d = self._dev()
        with pytest.raises(DeviceFault, match="unknown op"):
            d.call("encode", "text")


# =============================================================================
# SlonetDevice (needs provider with tokenizer + model)
# =============================================================================

class TestSlonetDevice:
    def _make_provider(self):
        class FakeTokenizer:
            eos_token_id = 0
            def encode(self, text):
                return list(range(len(text)))
            def decode(self, ids):
                return "".join(chr(i + 65) for i in ids)

        class FakeModel:
            vocab_size = 100
            n_embed = 32
            block_size = 128
            layers = [None, None, None]

            def generate_numpy(self, input_ids, max_new_tokens=10, **kw):
                return np.arange(max_new_tokens).reshape(1, -1)

            def generate_numpy_stream(self, input_ids, max_new_tokens=10, **kw):
                for i in range(max_new_tokens):
                    yield i

            def forward(self, tensor):
                class Logits:
                    def __init__(self, data):
                        self.data = data
                return Logits(np.ones((1, 10, 100))), None

        class FakeProvider:
            _tokenizer = FakeTokenizer()
            _model = FakeModel()
            _model_id = "test-model"

        return FakeProvider()

    def _dev(self):
        return SlonetDevice(self._make_provider())

    def test_tokenize(self):
        d = self._dev()
        result = d.call("tokenize", "abc")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert len(result) == 3

    def test_detokenize(self):
        d = self._dev()
        result = d.call("detokenize", [0, 1, 2])
        assert isinstance(result, str)
        assert len(result) == 3

    def test_generate(self):
        d = self._dev()
        result = d.call("generate", [0, 1, 2], 5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_generate_stream(self):
        d = self._dev()
        tokens = list(d.call("generate_stream", [0, 1], 4))
        assert tokens == [0, 1, 2, 3]

    def test_forward(self):
        d = self._dev()
        result = d.call("forward", [0, 1, 2])
        assert isinstance(result, np.ndarray)

    def test_info(self):
        d = self._dev()
        info = d.call("info")
        assert info["model_id"] == "test-model"
        assert info["vocab_size"] == 100
        assert info["n_embed"] == 32

    def test_unknown_op_raises(self):
        d = self._dev()
        with pytest.raises(DeviceFault, match="unknown op"):
            d.call("unknown_op")


# =============================================================================
# MultimodalDevice
# =============================================================================

class TestMultimodalDevice:
    def _make_engine(self):
        class FakeVision:
            embed_dim = 64

        class FakeEngine:
            _trained = True
            vision = FakeVision()

            def generate(self, image_np=None, max_len=20, temperature=1.0):
                class Output:
                    text = "caption:cat"
                return Output()

            def _concat_modalities(self, img, audio, text):
                class Embed:
                    def __init__(self):
                        self.data = np.zeros(64)
                return Embed(), None, None

        return FakeEngine()

    def _dev(self):
        return MultimodalDevice(self._make_engine())

    def test_generate(self):
        d = self._dev()
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = d.call("generate", img)
        assert result == "caption:cat"

    def test_generate_no_image(self):
        d = self._dev()
        result = d.call("generate")
        assert result == "caption:cat"

    def test_embed(self):
        d = self._dev()
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = d.call("embed", img)
        assert isinstance(result, np.ndarray)

    def test_info(self):
        d = self._dev()
        info = d.call("info")
        assert info["trained"] is True
        assert info["embed_dim"] == 64

    def test_unknown_op_raises(self):
        d = self._dev()
        with pytest.raises(DeviceFault, match="unknown op"):
            d.call("predict")


# =============================================================================
# NPUVMDevice
# =============================================================================

class TestNPUVMDevice:
    def _make_npu(self):
        class FakeResult:
            def __init__(self, value=None, error=None, success=True):
                self.value = value
                self.error = error
                self.success = success

        class FakeNPU:
            def load_model(self, name, source, **kw):
                return FakeResult(value={"loaded": name})

            def unload_model(self, name):
                return FakeResult(value={"unloaded": name})

            def tokenize(self, name, text):
                return FakeResult(value={"token_ids": [1, 2, 3]})

            def detokenize(self, name, ids):
                return FakeResult(value={"text": "hello"})

            def generate(self, name, prompt, max_tokens, **kw):
                return FakeResult(value={"text": "generated"})

            def forward(self, name, ids):
                return FakeResult(value={"logits": np.ones((1, 5, 100))})

            def embed(self, name, text, layer=-1):
                return FakeResult(value={"embedding": np.ones(64)})

            def train_step(self, name, ids, targets, lr=0.001, **kw):
                return FakeResult(value={"loss": 2.5})

            def info(self):
                return {"loaded_models": []}

            def profile(self, name, seq_len, batch_sizes):
                return {"profile": "done"}

            def checkpoint(self, name, ckpt_name):
                return FakeResult(value={"checkpointed": True})

            def restore(self, name, ckpt_name):
                return FakeResult(value={"restored": True})

            def list_checkpoints(self):
                return FakeResult(value=["ckpt1", "ckpt2"])

            def delete_checkpoint(self, name):
                return FakeResult(value={"deleted": name})

            def save_checkpoint(self, name, path):
                return FakeResult(value={"saved": True})

            def load_checkpoint(self, name, path):
                return FakeResult(value={"loaded": True})

            def quantize(self, name, bits):
                return FakeResult(value={"quantized": bits})

            def dequantize(self, name):
                return FakeResult(value={"dequantized": True})

            def clear_cache(self, name):
                return FakeResult(value={"cleared": True})

            def health(self):
                return FakeResult(value={"healthy": True})

            def batch(self, name, prompts, max_tokens):
                return FakeResult(value={"results": ["a", "b"]})

            def attention_maps(self, name, text, layer):
                return FakeResult(value={"maps": []})

            def compare(self, ma, mb, prompt, max_tokens):
                return FakeResult(value={"winner": "a"})

            def layers(self, name, layer):
                return FakeResult(value={"layers": []})

            def benchmark(self, name, prompt_lengths, max_tokens):
                return FakeResult(value={"throughput": 100})

        return FakeNPU()

    def _dev(self):
        return NPUVMDevice(self._make_npu())

    def test_load_model(self):
        d = self._dev()
        result = d.call("load_model", "qwen", "hf://model")
        assert result == {"loaded": "qwen"}

    def test_unload_model(self):
        d = self._dev()
        result = d.call("unload_model", "qwen")
        assert result == {"unloaded": "qwen"}

    def test_tokenize(self):
        d = self._dev()
        result = d.call("tokenize", "qwen", "hello")
        assert result == [1, 2, 3]

    def test_detokenize(self):
        d = self._dev()
        result = d.call("detokenize", "qwen", [1, 2, 3])
        assert result == "hello"

    def test_generate(self):
        d = self._dev()
        result = d.call("generate", "qwen", "prompt", 50)
        assert result == "generated"

    def test_forward(self):
        d = self._dev()
        result = d.call("forward", "qwen", [1, 2, 3])
        assert isinstance(result, np.ndarray)

    def test_embed(self):
        d = self._dev()
        result = d.call("embed", "qwen", "hello world")
        assert isinstance(result, np.ndarray)

    def test_train_step(self):
        d = self._dev()
        result = d.call("train_step", "qwen", [1, 2], [2, 3], 0.001)
        assert result == {"loss": 2.5}

    def test_info(self):
        d = self._dev()
        result = d.call("info")
        assert "loaded_models" in result

    def test_profile(self):
        d = self._dev()
        result = d.call("profile", "qwen", 512)
        assert result["profile"] == "done"

    def test_checkpoint(self):
        d = self._dev()
        result = d.call("checkpoint", "qwen", "ckpt1")
        assert result["checkpointed"] is True

    def test_restore(self):
        d = self._dev()
        result = d.call("restore", "qwen", "ckpt1")
        assert result["restored"] is True

    def test_list_checkpoints(self):
        d = self._dev()
        result = d.call("list_checkpoints")
        assert result == ["ckpt1", "ckpt2"]

    def test_delete_checkpoint(self):
        d = self._dev()
        result = d.call("delete_checkpoint", "ckpt1")
        assert result == {"deleted": "ckpt1"}

    def test_save_checkpoint(self):
        d = self._dev()
        result = d.call("save_checkpoint", "qwen", "/tmp/ckpt")
        assert result == {"saved": True}

    def test_load_checkpoint(self):
        d = self._dev()
        result = d.call("load_checkpoint", "qwen", "/tmp/ckpt")
        assert result == {"loaded": True}

    def test_quantize(self):
        d = self._dev()
        result = d.call("quantize", "qwen", 8)
        assert result == {"quantized": 8}

    def test_dequantize(self):
        d = self._dev()
        result = d.call("dequantize", "qwen")
        assert result == {"dequantized": True}

    def test_clear_cache(self):
        d = self._dev()
        result = d.call("clear_cache", "qwen")
        assert result == {"cleared": True}

    def test_health(self):
        d = self._dev()
        result = d.call("health")
        assert result == {"healthy": True}

    def test_batch(self):
        d = self._dev()
        result = d.call("batch", "qwen", ["p1", "p2"], 50)
        assert result == {"results": ["a", "b"]}

    def test_attention_maps(self):
        d = self._dev()
        result = d.call("attention_maps", "qwen", "text", -1)
        assert "maps" in result

    def test_compare(self):
        d = self._dev()
        result = d.call("compare", "qwen", "llama", "prompt", 20)
        assert result == {"winner": "a"}

    def test_layers(self):
        d = self._dev()
        result = d.call("layers", "qwen", -1)
        assert "layers" in result

    def test_benchmark(self):
        d = self._dev()
        result = d.call("benchmark", "qwen", [10, 20], 50)
        assert result == {"throughput": 100}

    def test_unknown_op_raises(self):
        d = self._dev()
        with pytest.raises(DeviceFault, match="unknown op"):
            d.call("nonexistent")

    def test_load_model_failure_raises(self):
        class FailNPU:
            def load_model(self, *a, **kw):
                class R:
                    success = False
                    error = "disk full"
                return R()
        d = NPUVMDevice(FailNPU())
        with pytest.raises(DeviceFault, match="load_model failed"):
            d.call("load_model", "m", "src")

    def test_batch_with_string_prompts(self):
        d = self._dev()
        result = d.call("batch", "qwen", "p1|p2", 50)
        assert result == {"results": ["a", "b"]}

    def test_detokenize_numpy_input(self):
        d = self._dev()
        result = d.call("detokenize", "qwen", np.array([1, 2, 3]))
        assert result == "hello"

    def test_forward_numpy_input(self):
        d = self._dev()
        result = d.call("forward", "qwen", np.array([1, 2, 3]))
        assert isinstance(result, np.ndarray)

    def test_profile_with_batch_sizes_string(self):
        d = self._dev()
        result = d.call("profile", "qwen", 256, "1,2,4")
        assert result["profile"] == "done"

    def test_benchmark_with_string_lengths(self):
        d = self._dev()
        result = d.call("benchmark", "qwen", "10,20,30", 50)
        assert result == {"throughput": 100}
