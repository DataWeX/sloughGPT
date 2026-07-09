"""Tests for NumpyEngine — pure NumPy inference engine with compression and KV cache."""

import numpy as np
import pytest
from domains.infrastructure.numpy_engine import NumpyEngine, KVCache, _CompressedWeight, _LRUCache
from domains.infrastructure.model_server import NumpyBackend, ModelServer, GuardBackend


@pytest.fixture(scope="session")
def gpt2():
    """Load GPT-2 once for entire test session."""
    return NumpyEngine.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def gpt2_uncompressed():
    """Load GPT-2 without compression for comparison."""
    return NumpyEngine.from_pretrained("gpt2", compress=False)


@pytest.fixture(scope="session")
def qwen2():
    """Load Qwen2 once for entire test session."""
    return NumpyEngine.from_pretrained("Qwen/Qwen2-0.5B-Instruct")


@pytest.fixture(scope="session")
def gpt2_backend(gpt2):
    """NumpyBackend wrapping GPT-2, once per session."""
    return NumpyBackend(gpt2)


class TestNumpyEngineGPT2:
    """GPT-2 inference tests."""

    def test_arch(self, gpt2):
        assert "gpt2" in gpt2.arch.name.lower()

    def test_vocab_size(self, gpt2):
        assert gpt2.vocab_size == 50257

    def test_has_tokenizer(self, gpt2):
        assert gpt2.tokenizer is not None

    def test_forward_returns_logits(self, gpt2):
        ids = gpt2.tokenizer.encode("Hello")
        logits = gpt2._forward(ids)
        assert isinstance(logits, np.ndarray)
        assert logits.shape == (gpt2.vocab_size,)

    def test_generate_returns_string(self, gpt2):
        result = gpt2.generate("The capital of France is", max_new_tokens=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_deterministic(self, gpt2):
        r1 = gpt2.generate("Hello", max_new_tokens=3, temperature=0.0)
        r2 = gpt2.generate("Hello", max_new_tokens=3, temperature=0.0)
        assert r1 == r2

    def test_info(self, gpt2):
        info = gpt2.info()
        assert "gpt2" in info["arch"].lower()
        assert info["vocab_size"] == 50257
        assert info["has_tokenizer"] is True
        assert info["num_params"] > 0


class TestNumpyEngineCompression:
    """Compression tests for NumpyEngine."""

    def test_compression_enabled(self, gpt2):
        assert gpt2._compress is True
        assert len(gpt2._compressed_weights) > 0

    def test_compression_disabled(self, gpt2_uncompressed):
        assert gpt2_uncompressed._compress is False
        assert len(gpt2_uncompressed._raw_weights) > 0

    def test_compression_ratio(self, gpt2):
        info = gpt2.info()
        assert info["compressed"] is True
        assert info["compression_ratio"] > 1.0  # At least some compression
        assert info["raw_bytes"] > info["compressed_bytes"]

    def test_get_weight_decompresses(self, gpt2):
        weight_name = list(gpt2._compressed_weights.keys())[0]
        weight = gpt2._get_weight(weight_name)
        assert isinstance(weight, np.ndarray)
        assert np.issubdtype(weight.dtype, np.floating)

    def test_lru_cache(self, gpt2):
        weight_name = list(gpt2._compressed_weights.keys())[0]
        w1 = gpt2._get_weight(weight_name)
        w2 = gpt2._get_weight(weight_name)
        assert np.array_equal(w1, w2)

    def test_compressed_weight_decompress(self):
        centroids = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
        compressed = _CompressedWeight(centroids, assignments, (8,), np.float32)
        decompressed = compressed.decompress()
        assert np.array_equal(decompressed, centroids[assignments])


class TestNumpyEngineKVCache:
    """KV cache tests for incremental decoding."""

    def test_kv_cache_init(self):
        cache = KVCache(n_layers=12)
        assert cache.seq_len == 0
        assert cache.get(0) is None

    def test_kv_cache_update(self):
        cache = KVCache(n_layers=2)
        k = np.random.randn(8, 10, 64).astype(np.float32)
        v = np.random.randn(8, 10, 64).astype(np.float32)

        k_cat, v_cat = cache.update(0, k, v)
        assert k_cat.shape == (8, 10, 64)
        assert cache.seq_len == 10

    def test_kv_cache_concatenation(self):
        cache = KVCache(n_layers=2)
        k1 = np.random.randn(8, 5, 64).astype(np.float32)
        v1 = np.random.randn(8, 5, 64).astype(np.float32)
        k2 = np.random.randn(8, 3, 64).astype(np.float32)
        v2 = np.random.randn(8, 3, 64).astype(np.float32)

        cache.update(0, k1, v1)
        k_cat, v_cat = cache.update(0, k2, v2)
        assert k_cat.shape == (8, 8, 64)  # 5 + 3 = 8
        assert cache.seq_len == 8

    def test_kv_cache_reset(self):
        cache = KVCache(n_layers=2)
        k = np.random.randn(8, 10, 64).astype(np.float32)
        v = np.random.randn(8, 10, 64).astype(np.float32)
        cache.update(0, k, v)

        cache.reset()
        assert cache.seq_len == 0
        assert cache.get(0) is None

    def test_generate_with_kv_cache(self, gpt2):
        result = gpt2.generate("Hello", max_new_tokens=5, use_kv_cache=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_without_kv_cache(self, gpt2):
        result = gpt2.generate("Hello", max_new_tokens=5, use_kv_cache=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_kv_cache_matches_full_forward(self, gpt2):
        # Both should produce same output for greedy decoding
        r1 = gpt2.generate("Hello world", max_new_tokens=3, temperature=0.0, use_kv_cache=False)
        r2 = gpt2.generate("Hello world", max_new_tokens=3, temperature=0.0, use_kv_cache=True)
        # Greedy should be identical (modulo floating-point accumulation)
        assert r1[:20] == r2[:20]  # First 20 chars should match


class TestNumpyEngineQwen2:
    """Qwen2 inference tests (bfloat16 loading)."""

    def test_arch(self, qwen2):
        assert "qwen" in qwen2.arch.name.lower()

    def test_vocab_size(self, qwen2):
        assert qwen2.vocab_size > 100000

    def test_forward_returns_logits(self, qwen2):
        ids = qwen2.tokenizer.encode("Hello")
        logits = qwen2._forward(ids)
        assert isinstance(logits, np.ndarray)
        assert logits.shape == (qwen2.vocab_size,)

    def test_generate_returns_string(self, qwen2):
        result = qwen2.generate("The capital of France is", max_new_tokens=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_produces_english(self, qwen2):
        result = qwen2.generate("The capital of France is", max_new_tokens=10)
        assert any(c.isalpha() for c in result)

    def test_info(self, qwen2):
        info = qwen2.info()
        assert "qwen" in info["arch"].lower()
        assert info["vocab_size"] > 100000


class TestNumpyEngineStreaming:
    """Streaming generation tests."""

    @pytest.mark.asyncio
    async def test_generate_stream(self, gpt2):
        tokens = []
        async for token in gpt2.generate_stream("Hello", max_new_tokens=5, temperature=0.0):
            tokens.append(token)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.asyncio
    async def test_generate_stream_yields_tokens(self, gpt2):
        count = 0
        async for _ in gpt2.generate_stream("Hello", max_new_tokens=5, temperature=0.0):
            count += 1
            if count >= 3:
                break
        assert count > 0


class TestNumpyEngineErrors:
    """Error handling tests."""

    def test_missing_model_raises(self):
        with pytest.raises(FileNotFoundError):
            NumpyEngine.from_pretrained("nonexistent/model-xyz")


class TestNumpyBackend:
    """NumpyBackend wrapping NumpyEngine into ModelServer backend interface."""

    def test_alive(self, gpt2_backend):
        assert gpt2_backend.alive is True

    def test_generate(self, gpt2_backend):
        result = gpt2_backend.generate(
            "Hello", max_new_tokens=10, temperature=0.0,
            top_p=1.0, top_k=0, repetition_penalty=1.0,
        )
        assert isinstance(result, dict)
        assert "text" in result
        assert "tokens_generated" in result
        assert len(result["text"]) > 0
        assert result["tokens_generated"] >= 0

    def test_generate_stream(self, gpt2_backend):
        tokens = list(gpt2_backend.generate_stream(
            "The capital of France is", max_new_tokens=5, temperature=0.0,
            top_p=1.0, top_k=0, repetition_penalty=1.0,
        ))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_generate_stream_cancel(self, gpt2_backend):
        from threading import Event
        cancel = Event()
        cancel.set()  # cancel immediately
        tokens = list(gpt2_backend.generate_stream(
            "Hello", max_new_tokens=50, temperature=0.0,
            top_p=1.0, top_k=0, repetition_penalty=1.0,
            cancel_event=cancel,
        ))
        assert len(tokens) == 0

    def test_dead_engine(self):
        backend = NumpyBackend(None)
        assert backend.alive is False


class TestModelServerNumpy:
    """ModelServer with numpy_engine (no PyTorch model)."""

    def test_select_backend_numpy(self, gpt2):
        server = ModelServer(
            model=None, tokenizer=None, model_id="test-numpy",
            numpy_engine=gpt2, enable_warmup=False,
        )
        backend = server._select_backend()
        assert isinstance(backend, NumpyBackend)

    def test_select_backend_guard_preferred(self, gpt2):
        mock_guard = type("MockGuard", (), {"alive": True, "generate": lambda *a, **k: {}})()
        server = ModelServer(
            model=None, tokenizer=None, model_id="test-priority",
            numpy_engine=gpt2, enable_warmup=False,
        )
        server._guard_backend = GuardBackend(mock_guard)
        backend = server._select_backend()
        assert isinstance(backend, GuardBackend)
