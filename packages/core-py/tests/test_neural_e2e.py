"""End-to-end test: NeuralKernel boots inside DaitRuntime, tokenize → generate → detokenize."""

import numpy as np
from domains.shell.runtime import DaitRuntime
from domains.shell.kernel import Kernel
from domains.shell.kernel_neural import (
    NeuralKernel, NeuralProcess, NeuralProcessType,
    NeuralKVCache, NeuralEngineDevice,
)
from domains.shell.kernel_process import ProcessState


def test_boot_creates_neural_kernel():
    rt = DaitRuntime()
    rt.boot()
    assert isinstance(rt.kernel, NeuralKernel)
    rt.shutdown()


def test_boot_registers_neural_devices():
    rt = DaitRuntime()
    rt.boot()
    nk = rt.kernel
    assert nk.engine is not None
    assert nk.tokenizer_device is not None
    assert nk.embedding_device is not None
    rt.shutdown()


def test_tokenize_through_kernel():
    rt = DaitRuntime()
    rt.boot()
    nk = rt.kernel
    tokens = nk.tokenize("hello world")
    assert tokens == list(b"hello world")
    rt.shutdown()


def test_detokenize_through_kernel():
    rt = DaitRuntime()
    rt.boot()
    nk = rt.kernel
    text = nk.detokenize([104, 101, 108, 108, 111])
    assert text == "hello"
    rt.shutdown()


def test_generate_with_mock_model():
    rt = DaitRuntime()
    rt.boot()
    nk = rt.kernel

    class MockModel:
        def generate_numpy(self, prompt, max_tokens=10, temperature=1.0):
            return [10, 20, 30, 40, 50]

    nk.engine.load_model("mock", MockModel())
    result = nk.generate("mock", "test prompt", max_tokens=5)
    assert result is not None
    assert result["token_count"] == 5
    rt.shutdown()


def test_embed_through_kernel():
    rt = DaitRuntime()
    rt.boot()
    nk = rt.kernel
    store = nk.create_embedding_store("test_store", 1000, 64)
    vecs = nk.embed(np.array([1, 2, 3]), "test_store")
    assert vecs is not None
    assert vecs.shape == (3, 64)
    rt.shutdown()


def test_kv_cache_through_kernel():
    rt = DaitRuntime()
    rt.boot()
    nk = rt.kernel
    cache = nk.create_kv_cache("test_cache", num_layers=4, head_dim=32)
    cache.initialize(num_heads=8)
    k0 = np.random.randn(8, 32)
    v0 = np.random.randn(8, 32)
    cache.update(0, k0, v0)
    cache.advance(1)
    kr, vr = cache.get(0, 0, 1)
    np.testing.assert_array_almost_equal(kr[:, 0, :], k0)
    rt.shutdown()


def test_neural_process_through_kernel():
    rt = DaitRuntime()
    rt.boot()
    nk = rt.kernel
    proc = nk.create_neural_process("e2e_infer", NeuralProcessType.INFERENCE, model_name="test")
    assert proc.pid > 0
    assert proc.neural_type == NeuralProcessType.INFERENCE
    assert proc.model_name == "test"
    proc.start_timing()
    assert proc.state == ProcessState.RUNNING
    proc.record_tokens([1, 2, 3], "generated text")
    proc.stop_timing(result={"tokens": [1, 2, 3]})
    assert proc.state == ProcessState.ZOMBIE
    assert proc.token_count == 3
    rt.shutdown()


def test_neural_syscall_tokenize():
    rt = DaitRuntime()
    rt.boot()
    from domains.shell.kernel_neural import NeuralSyscall
    result = rt.kernel.syscall(NeuralSyscall.TOKENIZE, "hello")
    assert result.success
    assert result.value["tokens"] == list(b"hello")
    rt.shutdown()


def test_neural_stats():
    rt = DaitRuntime()
    rt.boot()
    nk = rt.kernel
    stats = nk.neural_stats()
    assert "kv_caches" in stats
    assert "embedding_stores" in stats
    assert "engine" in stats
    assert stats["engine"]["models_loaded"] == 0
    rt.shutdown()
