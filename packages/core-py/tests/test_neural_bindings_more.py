"""Coverage-completing tests for the neural facade (domains.shell.addons.neural_bindings).

Run: PYTHONPATH=packages/core-py python -m pytest tests/test_neural_bindings_more.py -q
"""

import numpy as np
import pytest

from domains.shell.addons.neural import NeuralEmbeddingStore
from domains.shell.addons.neural_bindings import Property, engine
from domains.shell.kernel import Kernel


def _booted() -> Kernel:
    k = Kernel()
    k.boot()
    return k


class _Model:
    def forward(self, inputs):
        return {"out": inputs["x"] * 2}

    def backward(self, grad_output):
        return {"grad": grad_output["out"]}


class TestProperty:
    def test_set_name(self):
        class Owner:
            prop = engine

        assert engine._name == "prop"

    def test_get_with_none_obj(self):
        assert engine.__get__(None, object) is engine

    def test_get_requires_addon_on_missing(self):
        k = Kernel()
        with pytest.raises(RuntimeError):
            engine.__get__(k, Kernel)

    def test_get_after_setup(self):
        k = _booted()
        assert engine.__get__(k, Kernel) is k._engine


class TestEmbeddingStoreFacade:
    def test_embedding_store_empty_returns_default(self):
        k = _booted()
        store = k.embedding_store()
        assert isinstance(store, NeuralEmbeddingStore)
        assert store.embed_dim == 64

    def test_embedding_store_returns_first(self):
        k = _booted()
        s1 = k.create_embedding_store("first", 10, 4)
        k.create_embedding_store("second", 20, 8)
        assert k.embedding_store() is s1


class TestProcessFacade:
    def test_get_neural_process(self):
        k = _booted()
        p = k.create_neural_process("worker")
        assert k.get_neural_process(p.pid) is p
        assert k.get_neural_process(9999) is None

    def test_list_neural_processes(self):
        k = _booted()
        k.create_neural_process("a")
        k.create_neural_process("b")
        assert len(k.list_neural_processes()) == 2

    def test_cleanup_pid(self):
        k = _booted()
        p = k.create_neural_process("worker")
        k.cleanup_pid(p.pid)
        assert k.get_neural_process(p.pid) is None


class TestTokenizationFacade:
    def test_tokenize_fallback_when_ioctl_none(self, monkeypatch):
        k = _booted()
        monkeypatch.setattr(k._tokenizer_device, "ioctl", lambda *a, **kw: None)
        assert k.tokenize("hi") == [104, 105]

    def test_detokenize_fallback_when_ioctl_none(self, monkeypatch):
        k = _booted()
        monkeypatch.setattr(k._tokenizer_device, "ioctl", lambda *a, **kw: None)
        assert k.detokenize([104, 105]) == "hi"


    def test_tokenize_success(self):
        k = _booted()
        tokens = k.tokenize("hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_detokenize_success(self):
        k = _booted()
        tokens = k.tokenize("hello")
        assert k.detokenize(tokens) == "hello"


class TestEmbedFacade:
    def test_embed_missing_store_returns_none(self):
        k = _booted()
        assert k.embed(np.array([1, 2, 3]), "missing") is None

    def test_embed_existing_store(self):
        k = _booted()
        k.create_embedding_store("words", 100, 4)
        vec = k.embed(np.array([1, 2]), "words")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (2, 4)

    def test_embed_text_with_existing_store(self):
        k = _booted()
        k.create_embedding_store("words", 100, 4)
        vec = k.embed_text("hello")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (4,)

    def test_embed_text_without_store(self):
        k = _booted()
        vec = k.embed_text("hello")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (64,)


class TestKVCacheFacade:
    def test_get_kv_cache(self):
        k = _booted()
        cache = k.create_kv_cache("c1", 4, 16)
        assert k.get_kv_cache("c1") is cache
        assert k.get_kv_cache("missing") is None

    def test_remove_kv_cache(self):
        k = _booted()
        k.create_kv_cache("c1", 4, 16)
        k.remove_kv_cache("c1")
        assert k.get_kv_cache("c1") is None


class TestGenerationFacade:
    def test_generate_missing_model_returns_none(self):
        k = _booted()
        assert k.generate("ghost", "hello") is None

    def test_generate_with_model(self):
        k = _booted()

        class MockGen:
            def generate_numpy(self, prompt, max_tokens=10, temperature=1.0):
                return [1, 2, 3]

        k.engine.load_model("mock", MockGen())
        result = k.generate("mock", "hello", max_tokens=3)
        assert result == {"token_count": 3, "tokens": [1, 2, 3]}


class TestComputeFacade:
    def test_forward(self):
        k = _booted()
        p = k.create_neural_process("infer")
        p.model_ref = _Model()
        out = k.forward(p, {"x": np.array([1.0])})
        np.testing.assert_array_equal(out["out"], np.array([2.0]))

    def test_backward(self):
        k = _booted()
        p = k.create_neural_process("train")
        p.model_ref = _Model()
        grads = k.backward(p, {"out": np.array([1.0])})
        assert "grad" in grads

    def test_attention(self):
        k = _booted()
        q = np.random.randn(1, 2, 4)
        kv = np.random.randn(1, 2, 4)
        out = k.attention(q, kv, kv)
        assert out.shape == (1, 2, 4)


class TestNeuralSyscallFacade:
    def test_syscall_forward(self):
        k = _booted()
        p = k.create_neural_process("infer")
        p.model_ref = _Model()
        out = k.neural_syscall(p, "forward", {"x": np.array([1.0])})
        np.testing.assert_array_equal(out["out"], np.array([2.0]))

    def test_syscall_backward(self):
        k = _booted()
        p = k.create_neural_process("train")
        p.model_ref = _Model()
        grads = k.neural_syscall(p, "backward", {"out": np.array([1.0])})
        assert "grad" in grads

    def test_syscall_embed(self):
        k = _booted()
        vec = k.neural_syscall(None, "embed", "hello")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (64,)

    def test_syscall_unknown_op(self):
        k = _booted()
        p = k.create_neural_process("infer")
        assert k.neural_syscall(p, "bogus") is None


class TestRegistrationStats:
    def test_register_devices(self):
        k = _booted()
        k.register_devices()
        assert k._devices.table._devices

    def test_neural_stats(self):
        k = _booted()
        k.create_neural_process("infer")
        k.create_kv_cache("c1", 4, 16)
        k.create_embedding_store("words", 100, 4)
        stats = k.neural_stats()
        assert stats["neural_processes"] == 1
        assert stats["kv_caches"] == 1
        assert stats["embedding_stores"] == 1
        assert "gradient_accumulator" in stats
        assert "batch_processor" in stats
        assert "attention_device" in stats
        assert "engine" in stats
