"""Tests for Neural Interface Layer — kernel primitives for neural computation."""

import pytest
import numpy as np
from domains.shell.kernel_core import Kernel, reset_kernel
from domains.shell.kernel_process import Process, ProcessState, Priority
from domains.shell.kernel_neural import (
    NeuralKernel, NeuralProcess, NeuralProcessType,
    NeuralKVCache, NeuralEmbeddingStore, NeuralMemoryType,
    NeuralEngineDevice, TokenizerDevice, EmbeddingStoreDevice,
    NeuralInterrupt, NeuralSyscall,
)
from domains.shell.kernel_syscall import SyscallNumber


# ── Neural Process tests ─────────────────────────────────────────────────────

class TestNeuralProcess:
    def setup_method(self):
        reset_kernel()

    def teardown_method(self):
        reset_kernel()

    def test_create_inference_process(self):
        nk = NeuralKernel()
        nk.boot()
        proc = nk.create_neural_process("infer", NeuralProcessType.INFERENCE, model_name="gpt2")
        assert proc.neural_type == NeuralProcessType.INFERENCE
        assert proc.model_name == "gpt2"
        assert proc.pid > 0
        nk.shutdown()

    def test_create_training_process(self):
        nk = NeuralKernel()
        nk.boot()
        proc = nk.create_neural_process("train", NeuralProcessType.TRAINING, model_name="gpt2")
        assert proc.neural_type == NeuralProcessType.TRAINING
        nk.shutdown()

    def test_record_tokens(self):
        nk = NeuralKernel()
        nk.boot()
        proc = nk.create_neural_process("gen", NeuralProcessType.GENERATION)
        proc.forward_time_ms = 100.0
        proc.record_tokens([1, 2, 3, 4, 5], "hello world")
        assert proc.token_count == 5
        assert proc.generated_text == "hello world"
        assert proc.tokens_per_second == 50.0  # 5 tokens / 0.1s
        nk.shutdown()

    def test_record_loss(self):
        nk = NeuralKernel()
        nk.boot()
        proc = nk.create_neural_process("train", NeuralProcessType.TRAINING)
        proc.set_loss(0.5)
        assert proc.loss == 0.5
        assert proc.process.metadata["loss"] == 0.5
        nk.shutdown()

    def test_record_attention(self):
        nk = NeuralKernel()
        nk.boot()
        proc = nk.create_neural_process("attn", NeuralProcessType.ATTENTION)
        # Low entropy attention = converged
        pattern = np.zeros((8, 10, 10))
        pattern[:, :, 0] = 1.0  # all attention on first token
        proc.record_attention([pattern])
        assert proc.attention_converged
        nk.shutdown()

    def test_record_gradients(self):
        nk = NeuralKernel()
        nk.boot()
        proc = nk.create_neural_process("grad", NeuralProcessType.TRAINING)
        grads = {"weight": np.array([0.1, 0.2, 0.3])}
        proc.record_gradients(grads)
        assert proc.gradient_norm > 0
        assert "weight" in proc.gradients
        nk.shutdown()


# ── KV Cache tests ───────────────────────────────────────────────────────────

class TestNeuralKVCache:
    def test_create_and_initialize(self):
        cache = NeuralKVCache(num_layers=12, head_dim=64, max_positions=512)
        cache.initialize(num_heads=8)
        assert cache.get_position() == 0
        assert cache.memory_bytes() > 0

    def test_update_and_read(self):
        cache = NeuralKVCache(num_layers=2, head_dim=32, max_positions=100)
        cache.initialize(num_heads=4)
        k = np.random.randn(4, 32)
        v = np.random.randn(4, 32)
        pos = cache.update(0, k, v)
        assert pos == 0
        cache.advance(1)
        assert cache.get_position() == 1
        k_read, v_read = cache.get(0, 0, 1)
        np.testing.assert_array_almost_equal(k_read[:, 0, :], k)
        np.testing.assert_array_almost_equal(v_read[:, 0, :], v)

    def test_multi_layer(self):
        cache = NeuralKVCache(num_layers=4, head_dim=16, max_positions=50)
        cache.initialize(num_heads=2)
        for layer in range(4):
            k = np.random.randn(2, 16)
            v = np.random.randn(2, 16)
            cache.update(layer, k, v)
        cache.advance(1)
        for layer in range(4):
            k_read, v_read = cache.get(layer, 0, 1)
            assert k_read.shape == (2, 1, 16)

    def test_reset(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        cache.initialize(num_heads=4)
        cache.update(0, np.random.randn(4, 16), np.random.randn(4, 16))
        cache.advance(5)
        assert cache.get_position() == 5
        cache.reset()
        assert cache.get_position() == 0

    def test_layer_out_of_range(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        cache.initialize(num_heads=4)
        with pytest.raises(ValueError, match="out of range"):
            cache.update(5, np.random.randn(4, 16), np.random.randn(4, 16))


# ── Embedding Store tests ────────────────────────────────────────────────────

class TestNeuralEmbeddingStore:
    def test_create(self):
        store = NeuralEmbeddingStore(vocab_size=1000, embed_dim=64)
        assert store.vocab_size == 1000
        assert store.embed_dim == 64

    def test_lookup(self):
        store = NeuralEmbeddingStore(vocab_size=100, embed_dim=32)
        ids = np.array([1, 5, 10])
        vecs = store.lookup(ids)
        assert vecs.shape == (3, 32)

    def test_update(self):
        store = NeuralEmbeddingStore(vocab_size=100, embed_dim=32)
        ids = np.array([0, 1, 2])
        new_vecs = np.ones((3, 32))
        count = store.update(ids, new_vecs)
        assert count == 3
        vecs = store.lookup(ids)
        np.testing.assert_array_almost_equal(vecs, new_vecs)

    def test_similarity(self):
        store = NeuralEmbeddingStore(vocab_size=100, embed_dim=32)
        a = np.ones(32)
        b = np.ones(32)
        assert store.similarity(a, b) > 0.99
        c = -np.ones(32)
        assert store.similarity(a, c) < -0.99

    def test_nearest(self):
        store = NeuralEmbeddingStore(vocab_size=100, embed_dim=32)
        # Set known embeddings
        store._embeddings = np.zeros((100, 32))
        store._embeddings[0] = np.ones(32)
        store._embeddings[1] = np.ones(32) * 0.5
        query = np.ones(32)
        results = store.nearest(query, k=2)
        assert len(results) == 2
        assert results[0][0] == 0  # most similar

    def test_stats(self):
        store = NeuralEmbeddingStore(vocab_size=500, embed_dim=64)
        stats = store.stats()
        assert stats["vocab_size"] == 500
        assert stats["embed_dim"] == 64


# ── Neural Engine Device tests ───────────────────────────────────────────────

class TestNeuralEngineDevice:
    def test_register_and_info(self):
        dev = NeuralEngineDevice()
        assert dev.name == "neural_engine"
        assert dev.device_type.name == "INFERENCE"

    def test_load_unload_model(self):
        dev = NeuralEngineDevice()
        dev.open()
        dev.load_model("test", lambda x: x)
        info = dev.info()
        assert "test" in info["model_names"]
        dev.unload_model("test")
        assert "test" not in dev.info()["model_names"]

    def test_forward_pass(self):
        dev = NeuralEngineDevice()
        dev.open()
        model = lambda x: x * 2
        dev.load_model("double", model)
        result = dev.ioctl("forward", "double", np.array([1, 2, 3]))
        assert result.success
        np.testing.assert_array_equal(result.value["output"], np.array([2, 4, 6]))

    def test_forward_no_model(self):
        dev = NeuralEngineDevice()
        dev.open()
        result = dev.ioctl("forward", "nonexistent", np.array([1]))
        assert not result.success

    def test_generate(self):
        dev = NeuralEngineDevice()
        dev.open()
        # Simple model that returns token IDs
        class MockGen:
            def generate_numpy(self, prompt, max_tokens=10, temperature=1.0):
                return [1, 2, 3]
        dev.load_model("gen", MockGen())
        result = dev.ioctl("generate", "gen", "hello", max_tokens=3)
        assert result.success
        assert result.value["token_count"] == 3

    def test_attention(self):
        dev = NeuralEngineDevice()
        dev.open()
        q = np.random.randn(1, 4, 8)
        k = np.random.randn(1, 4, 8)
        v = np.random.randn(1, 4, 8)
        result = dev.ioctl("attention", q, k, v)
        assert result.success
        assert "output" in result.value
        assert "attention_weights" in result.value

    def test_loss_cross_entropy(self):
        dev = NeuralEngineDevice()
        dev.open()
        pred = np.array([[0.7, 0.3], [0.2, 0.8]])
        tgt = np.array([[1, 0], [0, 1]])
        result = dev.ioctl("loss", pred, tgt, loss_fn="cross_entropy")
        assert result.success
        assert result.value["loss"] > 0

    def test_loss_mse(self):
        dev = NeuralEngineDevice()
        dev.open()
        pred = np.array([1.0, 2.0, 3.0])
        tgt = np.array([1.1, 2.1, 3.1])
        result = dev.ioctl("loss", pred, tgt, loss_fn="mse")
        assert result.success
        assert result.value["loss"] < 0.1


# ── Tokenizer Device tests ───────────────────────────────────────────────────

class TestTokenizerDevice:
    def test_byte_level_fallback(self):
        dev = TokenizerDevice()
        dev.open()
        result = dev.ioctl("encode", "hello world")
        assert result.success
        assert result.value["tokens"] == list(b"hello world")
        assert result.value["encoding"] == "byte-level"

    def test_byte_level_decode(self):
        dev = TokenizerDevice()
        dev.open()
        result = dev.ioctl("decode", [104, 101, 108, 108, 111])
        assert result.success
        assert result.value["text"] == "hello"

    def test_with_custom_tokenizer(self):
        class MockTokenizer:
            def encode(self, text):
                return [ord(c) for c in text]
            def decode(self, tokens):
                return "".join(chr(t) for t in tokens)
        dev = TokenizerDevice(MockTokenizer())
        dev.open()
        enc = dev.ioctl("encode", "abc")
        assert enc.success
        assert enc.value["tokens"] == [97, 98, 99]
        dec = dev.ioctl("decode", [97, 98, 99])
        assert dec.success
        assert dec.value["text"] == "abc"


# ── Embedding Store Device tests ─────────────────────────────────────────────

class TestEmbeddingStoreDevice:
    def test_create_store(self):
        dev = EmbeddingStoreDevice()
        dev.open()
        result = dev.ioctl("create", store_name="test", vocab_size=100, embed_dim=32)
        assert result.success

    def test_lookup(self):
        dev = EmbeddingStoreDevice()
        dev.open()
        dev.create_store("test", 100, 32)
        result = dev.ioctl("lookup", [1, 2, 3], store_name="test")
        assert result.success
        assert result.value["shape"] == (3, 32)

    def test_update(self):
        dev = EmbeddingStoreDevice()
        dev.open()
        dev.create_store("test", 100, 32)
        new_vecs = np.ones((3, 32))
        result = dev.ioctl("update", [0, 1, 2], new_vecs, store_name="test")
        assert result.success
        assert result.value["updated"] == 3

    def test_nearest(self):
        dev = EmbeddingStoreDevice()
        dev.open()
        dev.create_store("test", 100, 32)
        store = dev.get_store("test")
        store._embeddings = np.zeros((100, 32))
        store._embeddings[0] = np.ones(32)
        result = dev.ioctl("nearest", np.ones(32), k=1, store_name="test")
        assert result.success
        assert len(result.value["nearest"]) == 1


# ── Neural Kernel integration tests ──────────────────────────────────────────

class TestNeuralKernel:
    def setup_method(self):
        reset_kernel()

    def teardown_method(self):
        reset_kernel()

    def test_create_and_tokenize(self):
        nk = NeuralKernel()
        nk.boot()
        tokens = nk.tokenize("hello world")
        assert tokens == list(b"hello world")
        nk.shutdown()

    def test_create_and_detokenize(self):
        nk = NeuralKernel()
        nk.boot()
        text = nk.detokenize([104, 101, 108, 108, 111])
        assert text == "hello"
        nk.shutdown()

    def test_create_embedding_store(self):
        nk = NeuralKernel()
        nk.boot()
        store = nk.create_embedding_store("words", 1000, 64)
        assert store.vocab_size == 1000
        vecs = nk.embed(np.array([1, 2, 3]), "words")
        assert vecs is not None
        assert vecs.shape == (3, 64)
        nk.shutdown()

    def test_kv_cache_create_and_use(self):
        nk = NeuralKernel()
        nk.boot()
        cache = nk.create_kv_cache("turn1", num_layers=6, head_dim=32)
        cache.initialize(num_heads=4)
        k0 = np.random.randn(4, 32)
        v0 = np.random.randn(4, 32)
        cache.update(0, k0, v0)
        cache.advance(1)
        kr, vr = cache.get(0, 0, 1)
        np.testing.assert_array_almost_equal(kr[:, 0, :], k0)
        nk.shutdown()

    def test_neural_stats(self):
        nk = NeuralKernel()
        nk.boot()
        nk.create_kv_cache("c1", 4, 32)
        nk.create_embedding_store("e1", 500, 128)
        stats = nk.neural_stats()
        assert stats["kv_caches"] == 1
        assert stats["embedding_stores"] == 1
        assert stats["engine"]["models_loaded"] == 0
        nk.shutdown()

    def test_neural_syscall_tokenize(self):
        nk = NeuralKernel()
        nk.boot()
        result = nk.syscall(NeuralSyscall.TOKENIZE, "test text")
        assert result.success
        assert result.value["tokens"] == list(b"test text")
        nk.shutdown()

    def test_neural_syscall_generate(self):
        nk = NeuralKernel()
        nk.boot()
        # Register a mock model
        class MockModel:
            def generate_numpy(self, prompt, max_tokens=10, temperature=1.0):
                return [10, 20, 30]
        nk.engine.load_model("mock", MockModel())
        result = nk.syscall(NeuralSyscall.GENERATE, "hello", "mock", max_tokens=3)
        assert result.success
        assert result.value["token_count"] == 3
        nk.shutdown()

    def test_neural_process_lifecycle(self):
        nk = NeuralKernel()
        nk.boot()
        proc = nk.create_neural_process("infer", NeuralProcessType.INFERENCE, model_name="test")
        assert proc.state == ProcessState.READY
        proc.start_timing()
        assert proc.state == ProcessState.RUNNING
        proc.record_tokens([1, 2, 3], "generated")
        proc.stop_timing(result={"tokens": [1, 2, 3]})
        assert proc.state == ProcessState.ZOMBIE
        assert proc.token_count == 3
        nk.shutdown()
