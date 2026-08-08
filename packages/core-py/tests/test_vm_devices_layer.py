"""
Coverage-completion tests for domains/shell/vm_devices.py.

Exercises every layer-4 device driver (TensorDevice, PythonExecDevice,
SlonetDevice, MultimodalDevice, EngineDevice, SlonetTrainingDevice,
NPUVMDevice) including dispatch, error branches, and type-normalisation
paths.  The Slonet* devices use real SloNet primitives with tiny models so
no heavyweight dependencies are pulled in.
"""

import sys
import types

import numpy as np
import pytest

from domains.shell.vm import DeviceFault
from domains.shell.vm_devices import (
    TensorDevice,
    PythonExecDevice,
    SlonetDevice,
    MultimodalDevice,
    EngineDevice,
    SlonetTrainingDevice,
    NPUVMDevice,
)
from domains.training.slonet import SloTransformer


# ── Shared fakes ──────────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, success, value=None, error=None):
        self.success = success
        self.value = value
        self.error = error or "boom"


class _FakeLogits:
    def __init__(self, data):
        self.data = data


class _FakeTokenizer:
    eos_token_id = 2

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _FakeSlonetModel:
    vocab_size = 100
    n_embed = 32
    block_size = 64
    layers = [object(), object(), object()]

    def generate_numpy(self, input_ids, max_new_tokens, temperature,
                       top_k, top_p, repetition_penalty, eos_token):
        return np.array([[5, 6, 7]])

    def generate_numpy_stream(self, input_ids, max_new_tokens, eos_token,
                              temperature, top_k, top_p, repetition_penalty):
        yield 8
        yield 9

    def forward(self, inp):
        return _FakeLogits(np.array([[0.1, 0.2, 0.3]])), None


class _FakeMinimalModel:
    vocab_size = 12
    n_embed = 8
    n_layer = 0

    def forward(self, inp):
        return _FakeLogits(np.zeros((1, 3))), None


class _FakeSlonetProvider:
    def __init__(self, minimal=False):
        self._tokenizer = _FakeTokenizer()
        self._model = _FakeMinimalModel() if minimal else _FakeSlonetModel()
        self._model_id = "fake-model"


class _FakeEmbed:
    data = np.zeros((1, 64))


class _FakeVision:
    embed_dim = 64


class _FakeBareVision:
    pass


class _FakeMultimodalEngine:
    def __init__(self, trained=True, bare_vision=False):
        self._trained = trained
        self.vision = _FakeBareVision() if bare_vision else _FakeVision()

    def generate(self, image_np, max_len, temperature):
        return _FakeOutput("a caption")

    def _concat_modalities(self, img, text, extra):
        return _FakeEmbed(), np.zeros((1, 7, 16)), None


class _FakeOutput:
    def __init__(self, text):
        self.text = text


class _FakeGenerateEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_tokens, temperature, **kwargs):
        self.calls.append((prompt, max_tokens, temperature, kwargs))
        return f"resp:{prompt}"


def _tiny_model():
    return SloTransformer(vocab_size=32, n_embed=16, n_layer=1, n_head=2,
                          block_size=32, dropout=0.0)


def _slonet_shim_without_scheduler():
    """Module shim that shadows WarmupCosineScheduler to force the
    has_scheduler=False branch in SlonetTrainingDevice._train."""
    import domains.training.slonet as real
    shim = types.ModuleType("domains.training.slonet")
    for name in ("cross_entropy", "SloAdam", "clip_grad_norm_",
                 "Tensor", "export_to_sou"):
        setattr(shim, name, getattr(real, name))
    return shim


# ── TensorDevice ──────────────────────────────────────────────────────────


class TestTensorDevice:
    def test_init_copies_weights(self):
        w = {"a": np.array([1.0])}
        dev = TensorDevice(w)
        assert "a" in dev._weights
        assert TensorDevice()._weights == {}

    def test_info(self):
        info = TensorDevice({"w": np.array([1.0])}).info()
        assert info["type"] == "tensor"
        assert "matmul" in info["ops"]
        assert info["weight_names"] == ["w"]

    def test_call_unknown_op_raises(self):
        with pytest.raises(DeviceFault, match="TensorDevice"):
            TensorDevice().call("bogus")

    def test_to_arr_branches(self):
        dev = TensorDevice()
        arr = np.array([1.0, 2.0])
        assert dev._to_arr(arr) is arr
        assert np.isscalar(dev._to_arr(3)) or dev._to_arr(3).ndim == 0
        assert np.allclose(dev._to_arr([1, 2]), [1, 2])
        assert np.allclose(dev._to_arr("[1, 2]"), [1, 2])
        assert np.allclose(dev._to_arr("42"), 42.0)

    def test_to_arr_invalid_json_raises(self):
        with pytest.raises(ValueError):
            TensorDevice()._to_arr("not json")

    def test_matmul(self):
        out = TensorDevice().call("matmul", [[1.0, 2.0]], [[3.0], [4.0]])
        assert np.allclose(out, [[11.0]])

    def test_pointwise_ops(self):
        dev = TensorDevice()
        assert np.allclose(dev.call("relu", [-1, 2]), [0, 2])
        sm = dev.call("softmax", [1.0, 1.0])
        assert np.allclose(sm.sum(), 1.0)
        assert np.allclose(dev.call("sigmoid", 0.0), 0.5)
        assert np.allclose(dev.call("tanh", 0.0), 0.0)
        assert np.allclose(dev.call("add", [1, 2], [3, 4]), [4, 6])
        assert np.allclose(dev.call("mul", [2, 3], [4, 5]), [8, 15])
        assert np.allclose(dev.call("sub", [5, 5], [1, 2]), [4, 3])
        assert np.allclose(dev.call("neg", [1, -2]), [-1, 2])
        assert np.allclose(dev.call("abs", [-1, 2]), [1, 2])

    def test_reduction_ops(self):
        dev = TensorDevice()
        assert dev.call("sum", [1, 2, 3]) == 6.0
        assert dev.call("mean", [1, 2, 3]) == 2.0
        assert dev.call("max", [1, 2, 3]) == 3.0
        assert dev.call("argmax", [1, 3, 2]) == 1
        assert dev.call("norm", [3, 4]) == 5.0

    def test_weight_ops(self):
        dev = TensorDevice({"w": np.array([1.0])})
        assert np.allclose(dev.call("load", "w"), [1.0])
        dev.call("store", "v", [2.0, 3.0])
        assert np.allclose(dev._weights["v"], [2.0, 3.0])
        with pytest.raises(DeviceFault, match="no weight"):
            dev.call("load", "missing")

    def test_shape_zeros_randn(self):
        dev = TensorDevice()
        assert dev.call("shape", [1, 2, 3]) == [3]
        z = dev.call("zeros", 2, 3)
        assert z.shape == (2, 3)
        assert z.sum() == 0.0
        assert dev.call("zeros", "x", "y").shape == (1, 1)
        r = dev.call("randn", 2, 2)
        assert r.shape == (2, 2)
        assert dev.call("randn", "x", "y").shape == (1, 1)

    def test_forward_requires_weights(self):
        with pytest.raises(DeviceFault, match="requires w1, w2"):
            TensorDevice().call("forward", [1, 2])

    def test_forward_full(self):
        dev = TensorDevice({
            "w1": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "b1": np.array([0.1, 0.2]),
            "w2": np.array([[1.0, 1.0]]),
            "b2": np.array([0.0]),
        })
        probs = dev.call("forward", [1.0, 2.0])
        assert np.allclose(probs.sum(), 1.0)
        assert probs.ndim == 1

    def test_forward_no_biases(self):
        dev = TensorDevice({
            "w1": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "w2": np.array([[1.0, 1.0]]),
        })
        probs = dev.call("forward", [1.0, 2.0])
        assert np.allclose(probs.sum(), 1.0)


# ── PythonExecDevice ──────────────────────────────────────────────────────


class TestPythonExecDevice:
    def test_info(self):
        info = PythonExecDevice().info()
        assert info["type"] == "python_exec"

    def test_call_unknown_op_raises(self):
        with pytest.raises(DeviceFault, match="PythonExecDevice"):
            PythonExecDevice().call("bogus")

    def test_eval_safe_builtin(self):
        dev = PythonExecDevice()
        assert dev.call("eval", "len([1, 2, 3])") == 3

    def test_eval_scope_access(self):
        dev = PythonExecDevice()
        dev.call("set", "x", 5)
        assert dev.call("eval", "x * 2") == 10

    def test_call_scope_and_builtin(self):
        dev = PythonExecDevice()
        dev.call("set", "double", lambda n: n * 2)
        assert dev.call("call", "double", 4) == 8
        assert dev.call("call", "len", [1, 2, 3]) == 3

    def test_call_unknown_raises(self):
        with pytest.raises(DeviceFault, match="unknown callable"):
            PythonExecDevice().call("call", "nope_zzz", 1)

    def test_import(self):
        dev = PythonExecDevice()
        mod = dev.call("import", "math")
        assert mod.sqrt(16) == 4.0
        assert "math" in dev.call("scope")
        dev.call("set", "sqrt", mod.sqrt)
        assert dev.call("call", "sqrt", 16) == 4.0

    def test_exec_side_effect(self):
        dev = PythonExecDevice()
        assert dev.call("exec", "y = 21") is None
        assert dev.call("get", "y") == 21

    def test_get_undefined_raises(self):
        with pytest.raises(DeviceFault, match="undefined"):
            PythonExecDevice().call("get", "zzz")

    def test_scope_snapshot(self):
        dev = PythonExecDevice()
        assert "np" in dev.call("scope")


# ── SlonetDevice ──────────────────────────────────────────────────────────


class TestSlonetDevice:
    def test_call_unknown_op_raises(self):
        dev = SlonetDevice(_FakeSlonetProvider())
        with pytest.raises(DeviceFault, match="SlonetDevice"):
            dev.call("bogus")

    def test_tokenize_and_detokenize(self):
        dev = SlonetDevice(_FakeSlonetProvider())
        ids = dev.call("tokenize", "Hi")
        assert ids.dtype == np.int64
        assert ids.tolist() == [72, 105]
        assert dev.call("detokenize", np.array([72, 105])) == "Hi"

    def test_generate_1d_input(self):
        dev = SlonetDevice(_FakeSlonetProvider())
        out = dev.call("generate", [1, 2, 3], 50)
        assert np.allclose(out, [5, 6, 7])

    def test_generate_2d_input_with_sampling(self):
        dev = SlonetDevice(_FakeSlonetProvider())
        out = dev.call("generate", np.array([[1, 2]]), 10, 0.9, 7, 0.95)
        assert np.allclose(out, [5, 6, 7])

    def test_generate_stream(self):
        dev = SlonetDevice(_FakeSlonetProvider())
        assert list(dev.call("generate_stream", [1, 2, 3], 5)) == [8, 9]
        assert list(dev.call("generate_stream",
                             np.array([[1, 2]]), 5, 0)) == [8, 9]

    def test_forward(self):
        dev = SlonetDevice(_FakeSlonetProvider())
        logits = dev.call("forward", [1, 2])
        assert logits.shape == (1, 3)

    def test_info(self):
        info = SlonetDevice(_FakeSlonetProvider()).call("info")
        assert info["model_id"] == "fake-model"
        assert info["n_layer"] == 3
        assert info["block_size"] == 64

    def test_info_minimal_model(self):
        info = SlonetDevice(_FakeSlonetProvider(minimal=True)).call("info")
        assert info["n_layer"] == 0
        assert info["block_size"] == 0


# ── MultimodalDevice ──────────────────────────────────────────────────────


class TestMultimodalDevice:
    def test_call_unknown_op_raises(self):
        dev = MultimodalDevice(_FakeMultimodalEngine())
        with pytest.raises(DeviceFault, match="MultimodalDevice"):
            dev.call("bogus")

    def test_generate_no_image(self):
        dev = MultimodalDevice(_FakeMultimodalEngine())
        assert dev.call("generate") == "a caption"

    def test_generate_2d_and_3d_images(self):
        dev = MultimodalDevice(_FakeMultimodalEngine())
        assert dev.call("generate", np.zeros((8, 8))) == "a caption"
        assert dev.call("generate", np.zeros((8, 8, 3)), 30, 0.7) == "a caption"

    def test_embed_2d_and_3d(self):
        dev = MultimodalDevice(_FakeMultimodalEngine())
        assert dev.call("embed", np.zeros((8, 8))).shape == (1, 64)
        assert dev.call("embed", np.zeros((8, 8, 3))).shape == (1, 64)

    def test_info_with_and_without_embed_dim(self):
        dev = MultimodalDevice(_FakeMultimodalEngine(trained=True))
        assert dev.call("info")["trained"] is True
        assert dev.call("info")["embed_dim"] == 64
        bare = MultimodalDevice(_FakeMultimodalEngine(trained=False,
                                                      bare_vision=True))
        assert bare.call("info")["trained"] is False
        assert bare.call("info")["embed_dim"] == 0


# ── EngineDevice ──────────────────────────────────────────────────────────


class TestEngineDevice:
    def test_call_unknown_op_raises(self):
        dev = EngineDevice(_FakeGenerateEngine(), "m")
        with pytest.raises(DeviceFault, match=r"EngineDevice\(m\)"):
            dev.call("bogus")

    def test_generate_with_kwargs(self):
        engine = _FakeGenerateEngine()
        dev = EngineDevice(engine, "m")
        assert dev.call("generate", "hi", 10, 0.9) == "resp:hi"
        assert dev._generate("hi", 10, 0.9, stop=".") == "resp:hi"
        assert engine.calls == [("hi", 10, 0.9, {}), ("hi", 10, 0.9, {"stop": "."})]

    def test_generate_defaults(self):
        engine = _FakeGenerateEngine()
        dev = EngineDevice(engine, "m")
        dev.call("generate", "hey")
        assert engine.calls == [("hey", 50, 1.0, {})]

    def test_info(self):
        dev = EngineDevice(_FakeGenerateEngine(), "gpt2")
        assert dev.call("info")["engine"] == "gpt2"
        assert dev.call("info")["type"] == "_FakeGenerateEngine"


# ── SlonetTrainingDevice ──────────────────────────────────────────────────


class TestSlonetTrainingDevice:
    def test_init_with_model(self):
        dev = SlonetTrainingDevice(model=_tiny_model())
        assert dev._created_model is False

    def test_init_from_config(self):
        dev = SlonetTrainingDevice(vocab_size=32, n_embed=16, n_layer=1,
                                   n_head=2, block_size=32)
        assert dev._created_model is True
        assert dev._model is None
        assert dev._model_config["vocab_size"] == 32

    def test_call_unknown_op_raises(self):
        dev = SlonetTrainingDevice(model=_tiny_model())
        with pytest.raises(DeviceFault, match="SlonetTrainingDevice"):
            dev.call("bogus")

    def test_info(self):
        dev = SlonetTrainingDevice(model=_tiny_model())
        info = dev.call("info")
        assert info["type"] == "slonet_train"
        assert info["model_created"] is False

    def test_config_get_and_set(self):
        dev = SlonetTrainingDevice(model=_tiny_model())
        assert dev.call("config")["lr"] == 3e-4
        assert dev.call("config", "lr", "0.001") == {"lr": 0.001}
        assert dev.call("config", "batch_size", "4") == {"batch_size": 4}
        assert dev.call("config", "scheduler", "linear") == {"scheduler": "linear"}
        assert dev.call("config", "lr") == {"lr": 0.001}
        assert dev.call("config", "nope", "x") == {"error": "unknown config key: nope"}

    def test_config_bad_value_falls_back(self):
        dev = SlonetTrainingDevice(model=_tiny_model())
        dev.call("config", "lr", "not-a-number")
        assert dev._train_config["lr"] == "not-a-number"

    def test_tokenize_detokenize(self):
        dev = SlonetTrainingDevice(model=_tiny_model())
        ids = dev.call("tokenize", "abc")
        assert ids.dtype == np.int64
        assert ids.tolist() == [0, 1, 2]
        assert dev.call("detokenize", np.array([72, 105])) == "Hi"
        assert dev.call("detokenize", 65) == "A"
        assert dev.call("detokenize", [97, 98]) == "ab"

    def test_generate_empty_prompt(self):
        dev = SlonetTrainingDevice(vocab_size=32, n_embed=16, n_layer=1,
                                   n_head=2, block_size=32)
        assert dev.call("generate", "", 5, 1.0) == ""

    def test_generate_temp_paths(self):
        dev = SlonetTrainingDevice(vocab_size=32, n_embed=16, n_layer=1,
                                   n_head=2, block_size=32)
        out0 = dev.call("generate", "hi", 5, 0.0)
        out1 = dev.call("generate", "hi", 5, 1.2)
        assert isinstance(out0, str) and out0
        assert isinstance(out1, str) and out1

    def test_forward_input_shapes(self):
        dev = SlonetTrainingDevice(vocab_size=32, n_embed=16, n_layer=1,
                                   n_head=2, block_size=32)
        a = dev.call("forward", [1, 2, 3])
        b = dev.call("forward", np.array([1, 2, 3]))
        c = dev.call("forward", np.array([[1, 2, 3]]))
        assert a.shape[0] == 1
        assert b.shape[0] == 1
        assert c.shape[0] == 1

    def test_train_missing_dataset(self):
        dev = SlonetTrainingDevice(model=_tiny_model())
        res = dev.call("train", 1, "", 0.001, 8, 16)
        assert "error" in res

    def test_eval_missing_dataset(self):
        dev = SlonetTrainingDevice(model=_tiny_model())
        res = dev.call("eval", "", 16)
        assert "error" in res

    def _write_dataset(self, tmp_path):
        text = ("hello world this is a tiny dataset for training. "
                "the quick brown fox jumps over the lazy dog. ") * 6
        p = tmp_path / "data.txt"
        p.write_text(text)
        return p

    def test_train_and_eval(self, tmp_path):
        dev = SlonetTrainingDevice(model=_tiny_model())
        dev.call("config", "checkpoint_dir", str(tmp_path / "ckpt"))
        p = self._write_dataset(tmp_path)
        res = dev.call("train", 1, str(p), 0.001, 32, 16, 1)
        assert "final_loss" in res
        assert res["epochs_completed"] == 1
        assert res["steps"] > 0
        assert (tmp_path / "ckpt").is_dir()

        ev = dev.call("eval", str(p), 16, 2)
        assert "loss" in ev
        assert ev["batches_evaluated"] == 2

    def test_train_without_scheduler(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "domains.training.slonet",
                            _slonet_shim_without_scheduler())
        dev = SlonetTrainingDevice(model=_tiny_model())
        dev.call("config", "checkpoint_dir", str(tmp_path / "ckpt"))
        p = self._write_dataset(tmp_path)
        res = dev.call("train", 1, str(p), 0.001, 32, 16, 0)
        assert "final_loss" in res
        assert res["epochs_completed"] == 1

    def test_save_and_load(self, tmp_path):
        dev = SlonetTrainingDevice(model=_tiny_model())
        dev.call("config", "checkpoint_dir", str(tmp_path / "ckpt"))
        path = dev.call("save", str(tmp_path / "m.soul"))
        assert path == str(tmp_path / "m.soul")
        assert (tmp_path / "m.soul").exists()
        msg = dev.call("load", str(tmp_path / "m.soul"))
        assert msg.startswith("loaded from")
        assert dev._created_model is False

    def test_save_default_path(self, tmp_path):
        dev = SlonetTrainingDevice(model=_tiny_model())
        dev.call("config", "checkpoint_dir", str(tmp_path / "ckpt"))
        path = dev.call("save", "")
        assert path.startswith(str(tmp_path / "ckpt"))
        assert path.endswith(".soul")

    def test_load_error(self, tmp_path):
        dev = SlonetTrainingDevice(model=_tiny_model())
        msg = dev.call("load", str(tmp_path / "missing.soul"))
        assert msg.startswith("load error:")

    def test_load_dataset_fallback_to_fs(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "domains.shell.file_manager", None)
        dev = SlonetTrainingDevice(model=_tiny_model())
        p = tmp_path / "d.txt"
        p.write_text("abcde")
        res = dev._load_dataset(str(p), 8)
        assert res is not None
        assert res[1] == 5

    def test_load_dataset_missing_fs(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "domains.shell.file_manager", None)
        dev = SlonetTrainingDevice(model=_tiny_model())
        assert dev._load_dataset(str(tmp_path / "nope.txt"), 8) is None


# ── NPUVMDevice ───────────────────────────────────────────────────────────


class _FakeNPU:
    def __init__(self, ok=True):
        self.ok = ok

    def _r(self, value=None):
        return _FakeResult(self.ok, value=value)

    def load_model(self, name, source, **kw):
        return self._r("loaded")

    def unload_model(self, name):
        return self._r("unloaded")

    def tokenize(self, name, text):
        return self._r({"token_ids": [1, 2]})

    def detokenize(self, name, ids):
        return self._r({"text": "hi"})

    def generate(self, name, prompt, max_tokens, **kw):
        return self._r({"text": "gen"})

    def forward(self, name, ids):
        return self._r({"logits": [0.5]})

    def embed(self, name, text, layer):
        return self._r({"embedding": [0.1]})

    def train_step(self, name, ids, targets, lr, **kw):
        return self._r({"loss": 1.0})

    def info(self):
        return {"npu": True}

    def profile(self, name, seq_len, batch_sizes):
        return {"profile": batch_sizes}

    def checkpoint(self, name, ckpt):
        return self._r("ckpt")

    def restore(self, name, ckpt):
        return self._r("restored")

    def list_checkpoints(self):
        return self._r(["a"])

    def delete_checkpoint(self, name):
        return self._r("deleted")

    def save_checkpoint(self, name, path):
        return self._r("saved")

    def load_checkpoint(self, name, path):
        return self._r("loaded")

    def quantize(self, name, bits):
        return self._r("q")

    def dequantize(self, name):
        return self._r("dq")

    def clear_cache(self, name):
        return self._r("cache")

    def health(self):
        return self._r({"ok": True})

    def batch(self, name, prompts, max_tokens):
        return self._r(["b1"])

    def attention_maps(self, name, text, layer):
        return self._r({"map": []})

    def compare(self, a, b, prompt, max_tokens):
        return self._r({"delta": 1})

    def layers(self, name, layer):
        return self._r({"l": []})

    def benchmark(self, name, prompt_lengths, max_tokens):
        return self._r({"tok": 1})


class TestNPUVMDevice:
    def test_call_unknown_op_raises(self):
        dev = NPUVMDevice(_FakeNPU())
        with pytest.raises(DeviceFault, match="NPUVMDevice"):
            dev.call("bogus")

    def test_success_paths(self):
        dev = NPUVMDevice(_FakeNPU())
        assert dev.call("load_model", "q", "hf://x", force=1) == "loaded"
        assert dev.call("unload_model", "q") == "unloaded"
        assert dev.call("tokenize", "q", "hi") == [1, 2]
        assert dev.call("detokenize", "q", np.array([1, 2])) == "hi"
        assert dev.call("detokenize", "q", (1, 2)) == "hi"
        assert dev.call("generate", "q", "hi", 20) == "gen"
        assert dev.call("forward", "q", np.array([1, 2])) == [0.5]
        assert dev.call("forward", "q", (1, 2)) == [0.5]
        assert dev.call("embed", "q", "hi", 2) == [0.1]
        assert dev.call("train_step", "q", np.array([1]), np.array([2]),
                        0.001) == {"loss": 1.0}
        assert dev.call("train_step", "q", [1], [2], 0.001) == {"loss": 1.0}
        assert dev.call("info") == {"npu": True}
        assert dev.info() == {"npu": True}
        assert dev.call("profile") == {"profile": None}
        assert dev.call("profile", "q", 32, [1, 2]) == {"profile": [1, 2]}
        assert dev.call("profile", "q", 32, "1,2") == {"profile": [1, 2]}
        assert dev.call("checkpoint", "q", "c") == "ckpt"
        assert dev.call("restore", "q", "c") == "restored"
        assert dev.call("list_checkpoints") == ["a"]
        assert dev.call("delete_checkpoint", "c") == "deleted"
        assert dev.call("save_checkpoint", "q", "p") == "saved"
        assert dev.call("load_checkpoint", "q", "p") == "loaded"
        assert dev.call("quantize", "q", 8) == "q"
        assert dev.call("dequantize", "q") == "dq"
        assert dev.call("clear_cache", "q") == "cache"
        assert dev.call("health") == {"ok": True}
        assert dev.call("batch", "q") == ["b1"]
        assert dev.call("batch", "q", "a|b", 5) == ["b1"]
        assert dev.call("batch", "q", ["x"], 5) == ["b1"]
        assert dev.call("attention_maps", "q", "t", 0) == {"map": []}
        assert dev.call("compare", "a", "b") == {"delta": 1}
        assert dev.call("layers", "q", 1) == {"l": []}
        assert dev.call("benchmark") == {"tok": 1}
        assert dev.call("benchmark", "q", "1,2", 5) == {"tok": 1}
        assert dev.call("benchmark", "q", [1, 2], 5) == {"tok": 1}

    def test_error_paths(self):
        dev = NPUVMDevice(_FakeNPU(ok=False))
        assert dev.call("load_model", "q", "s") == "boom"
        assert dev.call("unload_model", "q") == "boom"
        assert dev.call("tokenize", "q", "hi") == "boom"
        assert dev.call("detokenize", "q", [1]) == "boom"
        assert dev.call("generate", "q", "hi", 5) == "boom"
        assert dev.call("forward", "q", [1]) == "boom"
        assert dev.call("embed", "q", "hi", 1) == "boom"
        assert dev.call("train_step", "q", [1], [2], 0.001) == "boom"
        assert dev.call("checkpoint", "q", "c") == "boom"
        assert dev.call("restore", "q", "c") == "boom"
        assert dev.call("list_checkpoints") == "boom"
        assert dev.call("delete_checkpoint", "c") == "boom"
        assert dev.call("save_checkpoint", "q", "p") == "boom"
        assert dev.call("load_checkpoint", "q", "p") == "boom"
        assert dev.call("quantize", "q", 8) == "boom"
        assert dev.call("dequantize", "q") == "boom"
        assert dev.call("clear_cache", "q") == "boom"
        assert dev.call("health") == "boom"
        assert dev.call("batch", "q", "a|b", 5) == "boom"
        assert dev.call("attention_maps", "q", "t", 0) == "boom"
        assert dev.call("compare", "a", "b") == "boom"
        assert dev.call("layers", "q", 1) == "boom"
        assert dev.call("benchmark", "q", [1], 5) == "boom"
