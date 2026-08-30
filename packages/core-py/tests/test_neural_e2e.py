"""End-to-end test: NeuralKernel boots inside DaitRuntime, tokenize → generate → detokenize."""

import math
import numpy as np
import pytest

from domains.shell.runtime import DaitRuntime
from domains.shell.kernel import Kernel
from domains.shell.kernel_neural import (
    NeuralKernel, NeuralProcess, NeuralProcessType,
    NeuralKVCache, NeuralEngineDevice,
    NeuralSyscall, NeuralState, NeuralOp, NeuralMemoryType,
    CacheStrategy, KVCacheEntry, NeuralEmbeddingStore,
    EmbeddingEntry, TokenizerDevice, EmbeddingStoreDevice,
    MultiHeadAttentionDevice, NeuralInterrupt,
    GradientAccumulator, BatchRequest, BatchResult, BatchProcessor,
)
from domains.shell.kernel_process import ProcessState


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

def test_boot_creates_neural_kernel():
    rt = DaitRuntime()
    rt.boot()
    assert isinstance(rt.kernel, Kernel)
    assert rt.kernel.has_addon("neural")
    assert hasattr(rt.kernel, "engine")
    rt.shutdown()


def test_boot_registers_neural_devices():
    rt = DaitRuntime()
    rt.boot()
    nk = rt.kernel
    assert nk.engine is not None
    assert nk.tokenizer_device is not None
    assert nk.embedding_device is not None
    rt.shutdown()


def test_boot_twice_is_idempotent():
    rt = DaitRuntime()
    rt.boot()
    msg = rt.kernel.boot()
    assert msg == "Already booted"
    rt.shutdown()


def test_shutdown_twice_is_idempotent():
    rt = DaitRuntime()
    rt.boot()
    rt.shutdown()
    msg = rt.kernel.shutdown()
    assert msg == "Already shut down"


# ---------------------------------------------------------------------------
# Tokenize / Detokenize
# ---------------------------------------------------------------------------

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


def test_tokenize_empty_string():
    rt = DaitRuntime()
    rt.boot()
    tokens = rt.kernel.tokenize("")
    assert tokens == []
    rt.shutdown()


def test_tokenize_unicode():
    rt = DaitRuntime()
    rt.boot()
    tokens = rt.kernel.tokenize("cafe")
    assert tokens == list(b"cafe")
    rt.shutdown()


def test_detokenize_empty_list():
    rt = DaitRuntime()
    rt.boot()
    text = rt.kernel.detokenize([])
    assert text == ""
    rt.shutdown()


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

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


def test_generate_empty_model_name():
    rt = DaitRuntime()
    rt.boot()
    result = rt.kernel.generate("", "prompt")
    assert result is None or result.get("token_count", -1) == 0
    rt.shutdown()


def test_generate_unknown_model():
    rt = DaitRuntime()
    rt.boot()
    result = rt.kernel.generate("nonexistent", "prompt")
    assert result is None or result.get("token_count", -1) == 0
    rt.shutdown()


def test_generate_returns_token_list():
    rt = DaitRuntime()
    rt.boot()

    class MockModel:
        def generate_numpy(self, prompt, max_tokens=10, temperature=1.0):
            return [1, 2, 3]

    rt.kernel.engine.load_model("m", MockModel())
    result = rt.kernel.generate("m", "p", max_tokens=3)
    assert "tokens" in result
    assert result["tokens"] == [1, 2, 3]
    rt.shutdown()


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

def test_embed_through_kernel():
    rt = DaitRuntime()
    rt.boot()
    nk = rt.kernel
    store = nk.create_embedding_store("test_store", 1000, 64)
    vecs = nk.embed(np.array([1, 2, 3]), "test_store")
    assert vecs is not None
    assert vecs.shape == (3, 64)
    rt.shutdown()


def test_embed_returns_array():
    rt = DaitRuntime()
    rt.boot()
    store = rt.kernel.create_embedding_store("norm_store", 100, 32)
    vecs = rt.kernel.embed(np.array([0, 1]), "norm_store")
    assert vecs is not None
    assert vecs.shape == (2, 32)
    rt.shutdown()


def test_embed_text():
    rt = DaitRuntime()
    rt.boot()
    store = rt.kernel.create_embedding_store("txt_store", 100, 32)
    vec = rt.kernel.embed_text("hello world")
    assert vec is not None
    assert vec.ndim == 1
    rt.shutdown()


def test_embed_different_stores():
    rt = DaitRuntime()
    rt.boot()
    rt.kernel.create_embedding_store("s1", 100, 32)
    rt.kernel.create_embedding_store("s2", 100, 32)
    v1 = rt.kernel.embed(np.array([0]), "s1")
    v2 = rt.kernel.embed(np.array([0]), "s2")
    assert v1.shape == v2.shape
    rt.shutdown()


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

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


def test_kv_cache_advance_multiple():
    rt = DaitRuntime()
    rt.boot()
    cache = rt.kernel.create_kv_cache("mc", num_layers=2, head_dim=16)
    cache.initialize(num_heads=4)
    k = np.random.randn(4, 16)
    v = np.random.randn(4, 16)
    cache.update(0, k, v)
    cache.advance(1)
    cache.update(0, k * 2, v * 2)
    cache.advance(1)
    assert cache.get_position() == 2
    rt.shutdown()


def test_kv_cache_reset():
    rt = DaitRuntime()
    rt.boot()
    cache = rt.kernel.create_kv_cache("rc", num_layers=2, head_dim=16)
    cache.initialize(num_heads=4)
    k = np.random.randn(4, 16)
    cache.update(0, k, k)
    cache.advance(1)
    cache.reset()
    assert cache.get_position() == 0
    rt.shutdown()


def test_kv_cache_stats():
    rt = DaitRuntime()
    rt.boot()
    cache = rt.kernel.create_kv_cache("sc", num_layers=2, head_dim=16)
    cache.initialize(num_heads=4)
    stats = cache.stats()
    assert "layers_cached" in stats
    assert "memory_bytes" in stats
    assert "position" in stats
    rt.shutdown()


def test_kv_cache_memory_bytes():
    rt = DaitRuntime()
    rt.boot()
    cache = rt.kernel.create_kv_cache("mb", num_layers=2, head_dim=16)
    cache.initialize(num_heads=4)
    mb = cache.memory_bytes()
    assert mb > 0
    rt.shutdown()


def test_kv_cache_get_position():
    rt = DaitRuntime()
    rt.boot()
    cache = rt.kernel.create_kv_cache("pos", num_layers=1, head_dim=8)
    cache.initialize(num_heads=2)
    assert cache.get_position() == 0
    cache.advance(5)
    assert cache.get_position() == 5
    rt.shutdown()


# ---------------------------------------------------------------------------
# Neural Process
# ---------------------------------------------------------------------------

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


def test_neural_process_training_type():
    rt = DaitRuntime()
    rt.boot()
    proc = rt.kernel.create_neural_process("train", NeuralProcessType.TRAINING, model_name="m")
    assert proc.neural_type == NeuralProcessType.TRAINING
    rt.shutdown()


def test_neural_process_list():
    rt = DaitRuntime()
    rt.boot()
    rt.kernel.create_neural_process("p1", NeuralProcessType.INFERENCE)
    rt.kernel.create_neural_process("p2", NeuralProcessType.TRAINING)
    procs = rt.kernel.list_neural_processes()
    assert len(procs) >= 2
    rt.shutdown()


def test_neural_process_get_by_pid():
    rt = DaitRuntime()
    rt.boot()
    proc = rt.kernel.create_neural_process("lookup", NeuralProcessType.INFERENCE)
    found = rt.kernel.get_neural_process(proc.pid)
    assert found is not None
    assert found.pid == proc.pid
    rt.shutdown()


def test_neural_process_transition_neural():
    rt = DaitRuntime()
    rt.boot()
    proc = rt.kernel.create_neural_process("trans", NeuralProcessType.INFERENCE)
    proc.transition_neural(NeuralState.LOADING_WEIGHTS)
    assert proc.neural_state == NeuralState.LOADING_WEIGHTS
    proc.transition_neural(NeuralState.COMPUTING)
    assert proc.neural_state == NeuralState.COMPUTING
    rt.shutdown()


def test_neural_process_is_computing():
    rt = DaitRuntime()
    rt.boot()
    proc = rt.kernel.create_neural_process("comp", NeuralProcessType.INFERENCE)
    assert not proc.is_computing
    proc.transition_neural(NeuralState.COMPUTING)
    assert proc.is_computing
    proc.transition_neural(NeuralState.COMPLETE)
    assert not proc.is_computing
    rt.shutdown()


def test_neural_process_record_gradients():
    rt = DaitRuntime()
    rt.boot()
    proc = rt.kernel.create_neural_process("grad", NeuralProcessType.TRAINING)
    grads = {"w": np.array([1.0, 2.0]), "b": np.array([0.5])}
    proc.record_gradients(grads)
    assert "w" in proc.gradients
    assert proc.gradient_norm > 0
    rt.shutdown()


def test_neural_process_clear_gradients():
    rt = DaitRuntime()
    rt.boot()
    proc = rt.kernel.create_neural_process("clr", NeuralProcessType.TRAINING)
    proc.record_gradients({"w": np.array([1.0])})
    proc.clear_gradients()
    assert len(proc.gradients) == 0
    assert proc.gradient_norm == 0.0
    rt.shutdown()


def test_neural_process_set_loss():
    rt = DaitRuntime()
    rt.boot()
    proc = rt.kernel.create_neural_process("loss", NeuralProcessType.TRAINING)
    proc.set_loss(0.42)
    assert proc.loss == 0.42
    assert proc.process.metadata["loss"] == 0.42
    rt.shutdown()


def test_neural_process_status_line():
    rt = DaitRuntime()
    rt.boot()
    proc = rt.kernel.create_neural_process("sl", NeuralProcessType.INFERENCE, model_name="m")
    line = proc.status_line()
    assert "IDLE" in line or "idle" in line.lower()
    assert "m" in line
    rt.shutdown()


# ---------------------------------------------------------------------------
# Syscalls
# ---------------------------------------------------------------------------

def test_neural_syscall_tokenize():
    rt = DaitRuntime()
    rt.boot()
    result = rt.kernel.syscall(NeuralSyscall.TOKENIZE, "hello")
    assert result.success
    assert result.value["tokens"] == list(b"hello")
    rt.shutdown()


def test_neural_syscall_tokenize_empty():
    rt = DaitRuntime()
    rt.boot()
    result = rt.kernel.syscall(NeuralSyscall.TOKENIZE, "")
    assert result.success
    assert result.value["tokens"] == []
    rt.shutdown()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NeuralEngineDevice
# ---------------------------------------------------------------------------

class TestNeuralEngineDevice:
    def test_load_and_info(self):
        dev = NeuralEngineDevice()
        dev.load_model("m1", lambda x: x)
        info = dev.info()
        assert "m1" in info["model_names"]
        assert info["models_loaded"] == 1

    def test_unload_model(self):
        dev = NeuralEngineDevice()
        dev.load_model("m1", lambda x: x)
        dev.unload_model("m1")
        assert "m1" not in dev.info()["model_names"]

    def test_unload_nonexistent(self):
        dev = NeuralEngineDevice()
        dev.unload_model("ghost")
        assert dev.info()["models_loaded"] == 0

    def test_write_increments_request_count(self):
        dev = NeuralEngineDevice()
        dev.write("data")
        assert dev.info()["request_count"] == 1

    def test_ioctl_forward(self):
        dev = NeuralEngineDevice()
        dev.load_model("m", lambda inputs: inputs)
        result = dev.ioctl("forward", "m", {"x": 1})
        assert result.success
        assert result.value["output"] == {"x": 1}

    def test_ioctl_forward_missing_model(self):
        dev = NeuralEngineDevice()
        result = dev.ioctl("forward", "missing", {})
        assert not result.success

    def test_ioctl_generate(self):
        dev = NeuralEngineDevice()

        class M:
            def generate_numpy(self, prompt, max_tokens=10):
                return [1, 2, 3]

        dev.load_model("g", M())
        result = dev.ioctl("generate", "g", "hi", max_tokens=3)
        assert result.success
        assert result.value["token_count"] == 3

    def test_ioctl_generate_missing_model(self):
        dev = NeuralEngineDevice()
        result = dev.ioctl("generate", "no", "hi")
        assert not result.success

    def test_ioctl_attention(self):
        dev = NeuralEngineDevice()
        q = np.random.randn(2, 4, 8)
        k = np.random.randn(2, 4, 8)
        v = np.random.randn(2, 4, 8)
        result = dev.ioctl("attention", q, k, v)
        assert result is not None
        assert "output" in result.value
        assert "attention_weights" in result.value

    def test_ioctl_loss_mse(self):
        dev = NeuralEngineDevice()
        pred = np.array([1.0, 2.0, 3.0])
        tgt = np.array([1.5, 2.5, 3.5])
        result = dev.ioctl("loss", pred, tgt)
        assert result.success
        assert result.value["loss"] > 0

    def test_ioctl_loss_cross_entropy(self):
        dev = NeuralEngineDevice()
        pred = np.array([[0.1, 0.9], [0.8, 0.2]])
        tgt = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = dev.ioctl("loss", pred, tgt, loss_fn="cross_entropy")
        assert result.success
        assert result.value["loss"] > 0

    def test_ioctl_unknown_command(self):
        dev = NeuralEngineDevice()
        result = dev.ioctl("unknown_cmd")
        assert result is None


# ---------------------------------------------------------------------------
# TokenizerDevice
# ---------------------------------------------------------------------------

class TestTokenizerDevice:
    def test_tokenize_encode(self):
        dev = TokenizerDevice()
        result = dev.ioctl("encode", "hello")
        assert result.success
        assert result.value["tokens"] == list(b"hello")
        assert result.value["encoding"] == "byte-level"

    def test_tokenize_decode(self):
        dev = TokenizerDevice()
        result = dev.ioctl("decode", [104, 101, 108, 108, 111])
        assert result.success
        assert result.value["text"] == "hello"

    def test_encode_empty(self):
        dev = TokenizerDevice()
        result = dev.ioctl("encode", "")
        assert result.success
        assert result.value["tokens"] == []

    def test_decode_empty(self):
        dev = TokenizerDevice()
        result = dev.ioctl("decode", [])
        assert result.success
        assert result.value["text"] == ""

    def test_info(self):
        dev = TokenizerDevice()
        info = dev.info()
        assert "tokenize_count" in info

    def test_read_no_tokenizer(self):
        dev = TokenizerDevice()
        assert dev.read() is None


# ---------------------------------------------------------------------------
# EmbeddingStoreDevice
# ---------------------------------------------------------------------------

class TestEmbeddingStoreDevice:
    def test_create_store(self):
        dev = EmbeddingStoreDevice()
        result = dev.ioctl("create", store_name="s1", vocab_size=100, embed_dim=32)
        assert result.success
        assert dev.get_store("s1") is not None

    def test_lookup(self):
        dev = EmbeddingStoreDevice()
        dev.create_store("s", 100, 32)
        result = dev.ioctl("lookup", [0, 1, 2], store_name="s")
        assert result.success
        assert result.value["shape"] == (3, 32)

    def test_nearest(self):
        dev = EmbeddingStoreDevice()
        dev.create_store("s", 100, 32)
        q = np.random.randn(32)
        result = dev.ioctl("nearest", q, store_name="s", k=3)
        assert result.success
        assert "nearest" in result.value

    def test_missing_store(self):
        dev = EmbeddingStoreDevice()
        result = dev.ioctl("lookup", [0], store_name="nope")
        assert not result.success

    def test_info(self):
        dev = EmbeddingStoreDevice()
        dev.create_store("s", 100, 32)
        info = dev.info()
        assert "s" in info["stores"]


# ---------------------------------------------------------------------------
# MultiHeadAttentionDevice
# ---------------------------------------------------------------------------

class TestMultiHeadAttentionDevice:
    def test_attention(self):
        dev = MultiHeadAttentionDevice(num_heads=2, head_dim=8)
        q = np.random.randn(1, 2, 8)
        k = np.random.randn(1, 2, 8)
        v = np.random.randn(1, 2, 8)
        out = dev.ioctl("attention", q, k, v)
        assert out is not None
        assert out.shape == q.shape

    def test_attention_with_mask(self):
        dev = MultiHeadAttentionDevice(num_heads=2, head_dim=8)
        q = np.random.randn(1, 2, 8)
        k = np.random.randn(1, 2, 8)
        v = np.random.randn(1, 2, 8)
        mask = np.array([[True, False], [True, True]])
        out = dev.ioctl("attention", q, k, v, mask)
        assert out is not None

    def test_attention_compute_count(self):
        dev = MultiHeadAttentionDevice()
        q = np.random.randn(1, 3, 8)
        k = np.random.randn(1, 3, 8)
        v = np.random.randn(1, 3, 8)
        dev.ioctl("attention", q, k, v)
        dev.ioctl("attention", q, k, v)
        info = dev.info()
        assert info["compute_count"] == 2

    def test_unknown_command(self):
        dev = MultiHeadAttentionDevice()
        assert dev.ioctl("unknown") is None

    def test_info_keys(self):
        dev = MultiHeadAttentionDevice(num_heads=4, head_dim=16)
        info = dev.info()
        assert info["num_heads"] == 4
        assert info["head_dim"] == 16


# ---------------------------------------------------------------------------
# GradientAccumulator
# ---------------------------------------------------------------------------

class TestGradientAccumulator:
    def test_accumulate(self):
        ga = GradientAccumulator(accumulation_steps=2)
        ready = ga.accumulate({"w": np.array([1.0])})
        assert not ready
        ready = ga.accumulate({"w": np.array([2.0])})
        assert ready

    def test_get_clipped_gradients(self):
        ga = GradientAccumulator(max_grad_norm=1.0, accumulation_steps=1)
        ga.accumulate({"w": np.array([100.0])})
        grads = ga.get_clipped_gradients()
        norm = np.linalg.norm(grads["w"])
        assert norm <= 1.0 + 1e-5

    def test_reset(self):
        ga = GradientAccumulator(accumulation_steps=1)
        ga.accumulate({"w": np.array([1.0])})
        ga.reset()
        assert ga.step_count == 0
        assert not ga.ready

    def test_stats(self):
        ga = GradientAccumulator(max_grad_norm=2.0, accumulation_steps=3)
        stats = ga.stats()
        assert stats["max_grad_norm"] == 2.0
        assert stats["accumulation_steps"] == 3

    def test_ready_initially_false(self):
        ga = GradientAccumulator(accumulation_steps=5)
        assert not ga.ready


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------

class TestBatchProcessor:
    def test_submit(self):
        bp = BatchProcessor(max_batch_size=10)
        req = BatchRequest(id="r1", inputs={"x": np.array([1.0])})
        assert bp.submit(req) is True
        assert bp.queue_size == 1

    def test_process_batch(self):
        bp = BatchProcessor(max_batch_size=5)
        for i in range(3):
            bp.submit(BatchRequest(id=f"r{i}", inputs={"x": np.array([float(i)])}))
        results = bp.process_batch()
        assert len(results) == 3
        assert all(r.error is None for r in results)

    def test_flush(self):
        bp = BatchProcessor(max_batch_size=5)
        for i in range(8):
            bp.submit(BatchRequest(id=f"r{i}", inputs={"x": np.array([float(i)])}))
        all_results = bp.flush()
        assert len(all_results) == 8

    def test_process_empty_returns_empty(self):
        bp = BatchProcessor()
        results = bp.process_batch()
        assert results == []

    def test_stats(self):
        bp = BatchProcessor(max_batch_size=10)
        bp.submit(BatchRequest(id="r1", inputs={}))
        stats = bp.stats()
        assert stats["total_requests"] == 1

    def test_queue_overflow_rejects(self):
        bp = BatchProcessor(max_batch_size=2)
        for i in range(5):
            bp.submit(BatchRequest(id=f"r{i}", inputs={}))
        assert bp.queue_size <= 4  # max_batch_size * 2

    def test_callback_invoked(self):
        called = []
        bp = BatchProcessor(max_batch_size=5)
        req = BatchRequest(id="c1", inputs={}, callback=lambda r: called.append(r))
        bp.submit(req)
        bp.process_batch()
        assert len(called) == 1

    def test_process_fn_error(self):
        def bad_fn(inputs):
            raise ValueError("bad")
        bp = BatchProcessor(max_batch_size=5, process_fn=bad_fn)
        bp.submit(BatchRequest(id="e1", inputs={}))
        results = bp.process_batch()
        assert results[0].error is not None


# ---------------------------------------------------------------------------
# NeuralInterrupt
# ---------------------------------------------------------------------------

class TestNeuralInterrupt:
    def test_inference_done(self):
        intr = NeuralInterrupt.inference_done(42, result={"out": 1})
        assert intr.source_pid == 42
        assert intr.data == {"out": 1}

    def test_training_step(self):
        intr = NeuralInterrupt.training_step(1, loss=0.5, step=10)
        assert intr.data["loss"] == 0.5
        assert intr.data["step"] == 10

    def test_gradient_update(self):
        intr = NeuralInterrupt.gradient_update(1, grad_norm=0.1)
        assert intr.data["grad_norm"] == 0.1

    def test_data_ready(self):
        intr = NeuralInterrupt.data_ready(1, batch_size=32)
        assert intr.data["batch_size"] == 32


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_neural_op_values(self):
        assert NeuralOp.EMBEDDING == 1
        assert NeuralOp.ATTENTION == 2
        assert NeuralOp.LINEAR == 3

    def test_neural_state_values(self):
        assert NeuralState.IDLE == 0
        assert NeuralState.COMPUTING == 2
        assert NeuralState.FAILED == 5

    def test_neural_process_type_values(self):
        assert NeuralProcessType.INFERENCE == 0
        assert NeuralProcessType.TRAINING == 1

    def test_neural_memory_type_values(self):
        assert NeuralMemoryType.KV_CACHE == 0
        assert NeuralMemoryType.EMBEDDING == 1

    def test_cache_strategy_values(self):
        assert CacheStrategy.LRU == 0
        assert CacheStrategy.LFU == 1
        assert CacheStrategy.FIFO == 2


# ---------------------------------------------------------------------------
# EmbeddingEntry
# ---------------------------------------------------------------------------

class TestEmbeddingEntry:
    def test_create(self):
        e = EmbeddingEntry(id="e1", vector=np.array([1.0]), text="hello")
        assert e.id == "e1"
        assert e.text == "hello"
        assert len(e.metadata) == 0

    def test_metadata(self):
        e = EmbeddingEntry(id="e1", vector=np.array([1.0]), text="t", metadata={"k": "v"})
        assert e.metadata["k"] == "v"


# ---------------------------------------------------------------------------
# KVCacheEntry
# ---------------------------------------------------------------------------

class TestKVCacheEntry:
    def test_create(self):
        e = KVCacheEntry(layer_idx=0)
        assert e.layer_idx == 0
        assert e.keys is None
        assert e.seq_len == 0


# ---------------------------------------------------------------------------
# NeuralEmbeddingStore
# ---------------------------------------------------------------------------

class TestNeuralEmbeddingStore:
    def test_create(self):
        store = NeuralEmbeddingStore(vocab_size=100, embed_dim=32)
        assert store.vocab_size == 100
        assert store.embed_dim == 32

    def test_add_and_nearest(self):
        store = NeuralEmbeddingStore(vocab_size=100, embed_dim=32)
        vec = np.random.randn(32).astype(np.float32)
        store.add("e1", vec, "hello")
        nearest = store.nearest(vec, k=1)
        assert len(nearest) >= 1

    def test_similarity(self):
        store = NeuralEmbeddingStore()
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        assert store.similarity(a, b) == pytest.approx(1.0)

    def test_similarity_zero_vector(self):
        store = NeuralEmbeddingStore()
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert store.similarity(a, b) == 0.0

    def test_nearest(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=8)
        for i in range(5):
            store.add(f"e{i}", np.random.randn(8), f"text{i}")
        nearest = store.nearest(np.random.randn(8), k=3)
        assert len(nearest) <= 3

    def test_stats(self):
        store = NeuralEmbeddingStore(vocab_size=50, embed_dim=16)
        stats = store.stats()
        assert stats["vocab_size"] == 50
        assert stats["embed_dim"] == 16

    def test_size(self):
        store = NeuralEmbeddingStore()
        assert store.size == 0
        store.add("a", np.random.randn(8), "a")
        assert store.size == 1
