"""Tests for neural.py — dataclasses, enums, config, pure logic."""

from __future__ import annotations

import time
import math

import numpy as np
import pytest

from domains.shell.kernel_process import Process, ProcessState
from domains.shell.kernel_devices import DeviceDriver, DeviceType
from domains.shell.kernel_interrupts import Interrupt, InterruptType
from domains.shell.kernel_syscall import SyscallResult

from domains.shell.addons.neural import (
    NeuralOp,
    NeuralState,
    NeuralProcessType,
    NeuralMemoryType,
    CacheStrategy,
    NeuralProcess,
    KVCacheEntry,
    NeuralKVCache,
    EmbeddingEntry,
    NeuralEmbeddingStore,
    GradientAccumulator,
    BatchRequest,
    BatchResult,
    BatchProcessor,
    NeuralInterrupt,
    NeuralSyscall,
    NeuralEngineDevice,
    TokenizerDevice,
    EmbeddingStoreDevice,
    MultiHeadAttentionDevice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_process(pid: int = 1, name: str = "test") -> Process:
    return Process(pid=pid, name=name)


def _make_neural_proc(**kwargs) -> NeuralProcess:
    return NeuralProcess(process=_make_process(**kwargs))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_neural_op_members(self):
        assert NeuralOp.NONE == 0
        assert NeuralOp.EMBEDDING == 1
        assert NeuralOp.OPTIMIZER_STEP == 9

    def test_neural_state_members(self):
        assert NeuralState.IDLE == 0
        assert NeuralState.COMPUTING == 2
        assert NeuralState.FAILED == 5

    def test_neural_process_type_members(self):
        assert NeuralProcessType.INFERENCE == 0
        assert NeuralProcessType.TRAINING == 1

    def test_neural_memory_type_members(self):
        assert NeuralMemoryType.KV_CACHE == 0
        assert NeuralMemoryType.COMPLETE == 7

    def test_cache_strategy_members(self):
        assert CacheStrategy.LRU == 0
        assert CacheStrategy.FIFO == 2

    def test_all_enums_are_intenum(self):
        for cls in (NeuralOp, NeuralState, NeuralProcessType, NeuralMemoryType, CacheStrategy):
            assert issubclass(cls, int)


# ---------------------------------------------------------------------------
# NeuralProcess dataclass
# ---------------------------------------------------------------------------

class TestNeuralProcess:
    def test_defaults(self):
        proc = _make_neural_proc()
        assert proc.neural_state == NeuralState.IDLE
        assert proc.neural_type == NeuralProcessType.INFERENCE
        assert proc.model_name == ""
        assert proc.loss == 0.0
        assert proc.max_retries == 3

    def test_pid_property(self):
        proc = _make_neural_proc(pid=42)
        assert proc.pid == 42

    def test_name_property(self):
        proc = _make_neural_proc(name="my_proc")
        assert proc.name == "my_proc"

    def test_state_property(self):
        proc = _make_neural_proc()
        assert proc.state == ProcessState.CREATED

    def test_is_computing(self):
        proc = _make_neural_proc()
        assert not proc.is_computing
        proc.neural_state = NeuralState.COMPUTING
        assert proc.is_computing
        proc.neural_state = NeuralState.BACKPROPAGATING
        assert proc.is_computing
        proc.neural_state = NeuralState.OPTIMIZING
        assert proc.is_computing

    def test_is_not_computing_idle(self):
        proc = _make_neural_proc()
        proc.neural_state = NeuralState.IDLE
        assert not proc.is_computing

    def test_compute_time_ms_no_start(self):
        proc = _make_neural_proc()
        assert proc.compute_time_ms == 0.0

    def test_compute_time_ms_with_times(self):
        proc = _make_neural_proc()
        proc.compute_start = 100.0
        proc.compute_end = 100.5
        assert proc.compute_time_ms == pytest.approx(500.0)

    def test_compute_time_ms_ongoing(self):
        proc = _make_neural_proc()
        proc.compute_start = time.time() - 1.0
        ms = proc.compute_time_ms
        assert ms >= 900.0

    def test_transition_neural_computing(self):
        proc = _make_neural_proc()
        proc.transition_neural(NeuralState.COMPUTING)
        assert proc.neural_state == NeuralState.COMPUTING
        assert proc.compute_start is not None
        assert proc.compute_end is None

    def test_transition_neural_complete(self):
        proc = _make_neural_proc()
        proc.transition_neural(NeuralState.COMPUTING)
        proc.transition_neural(NeuralState.COMPLETE)
        assert proc.neural_state == NeuralState.COMPLETE
        assert proc.compute_end is not None

    def test_transition_neural_failed(self):
        proc = _make_neural_proc()
        proc.transition_neural(NeuralState.FAILED)
        assert proc.compute_end is not None

    def test_record_tokens(self):
        proc = _make_neural_proc()
        proc.forward_time_ms = 1000.0
        proc.record_tokens([1, 2, 3], "hello")
        assert proc.token_count == 3
        assert proc.generated_text == "hello"
        assert proc.tokens_per_second == pytest.approx(3.0)

    def test_record_tokens_zero_time(self):
        proc = _make_neural_proc()
        proc.record_tokens([1], "a")
        assert proc.tokens_per_second == 0.0

    def test_set_loss(self):
        proc = _make_neural_proc()
        proc.set_loss(0.42)
        assert proc.loss == 0.42
        assert proc.process.metadata["loss"] == 0.42

    def test_record_gradients(self):
        proc = _make_neural_proc()
        grads = {"w": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
        proc.record_gradients(grads)
        assert "w" in proc.gradients
        assert "b" in proc.gradients
        assert proc.gradient_norm == pytest.approx(math.sqrt(2.0))

    def test_store_gradient(self):
        proc = _make_neural_proc()
        proc.store_gradient("w", np.array([3.0, 4.0]))
        assert "w" in proc.gradients
        assert proc.gradient_norm == pytest.approx(25.0)  # 3^2 + 4^2

    def test_clear_gradients(self):
        proc = _make_neural_proc()
        proc.store_gradient("w", np.array([1.0]))
        proc.clear_gradients()
        assert proc.gradients == {}
        assert proc.gradient_norm == 0.0

    def test_record_attention_converged(self):
        proc = _make_neural_proc()
        # Low entropy → converged
        pattern = np.array([[0.9, 0.1], [0.05, 0.95]])
        proc.record_attention([pattern])
        assert proc.attention_converged is True

    def test_record_attention_not_converged(self):
        proc = _make_neural_proc()
        # High entropy → not converged
        pattern = np.array([[0.5, 0.5], [0.5, 0.5]])
        proc.record_attention([pattern])
        assert proc.attention_converged is False

    def test_record_attention_3d_pattern(self):
        proc = _make_neural_proc()
        pattern = np.random.rand(2, 3, 4)
        proc.record_attention([pattern])
        assert isinstance(proc.attention_converged, bool)

    def test_record_attention_empty_patterns(self):
        proc = _make_neural_proc()
        proc.record_attention([])
        assert proc.attention_converged is False

    def test_status_line(self):
        proc = _make_neural_proc(pid=7, name="test_proc")
        proc.model_name = "gpt2"
        line = proc.status_line()
        assert "test_proc" in line
        assert "IDLE" in line
        assert "gpt2" in line


# ---------------------------------------------------------------------------
# KVCacheEntry dataclass
# ---------------------------------------------------------------------------

class TestKVCacheEntry:
    def test_defaults(self):
        entry = KVCacheEntry(layer_idx=0)
        assert entry.layer_idx == 0
        assert entry.keys is None
        assert entry.values is None
        assert entry.seq_len == 0
        assert entry.access_count == 0


# ---------------------------------------------------------------------------
# NeuralKVCache
# ---------------------------------------------------------------------------

class TestNeuralKVCache:
    def test_init_defaults(self):
        cache = NeuralKVCache()
        assert cache.total_tokens_cached == 0
        assert cache.evictions == 0

    def test_init_kwargs(self):
        cache = NeuralKVCache(num_heads=4)
        assert cache._num_heads == 4

    def test_initialize_creates_entries(self):
        cache = NeuralKVCache(num_layers=4, head_dim=32, max_positions=64)
        cache.initialize(num_heads=2)
        stats = cache.stats()
        assert stats["layers_cached"] == 4
        assert stats["max_positions"] == 64

    def test_get_position(self):
        cache = NeuralKVCache()
        assert cache.get_position() == 0
        cache.advance(5)
        assert cache.get_position() == 5

    def test_advance(self):
        cache = NeuralKVCache()
        cache.advance(3)
        cache.advance(2)
        assert cache.get_position() == 5

    def test_memory_bytes(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=32)
        cache.initialize(num_heads=4)
        mb = cache.memory_bytes()
        assert mb > 0
        # 2 layers * 4 heads * 32 positions * 16 dims * 8 bytes (float64) * 2 (k+v)
        expected = 2 * 4 * 32 * 16 * 8 * 2
        assert mb == expected

    def test_update_and_get(self):
        cache = NeuralKVCache(num_layers=2, head_dim=8, max_positions=10)
        cache.initialize(num_heads=3)
        k = np.random.rand(3, 8)
        v = np.random.rand(3, 8)
        pos = cache.update(0, k, v)
        assert pos == 0
        cache.advance(1)
        keys, values = cache.get(0, start=0, end=1)
        assert keys is not None
        assert keys.shape == (3, 1, 8)

    def test_update_out_of_range_layer(self):
        cache = NeuralKVCache(num_layers=2)
        with pytest.raises(ValueError, match="out of range"):
            cache.update(10, np.zeros((1, 8)), np.zeros((1, 8)))

    def test_get_nonexistent_layer(self):
        cache = NeuralKVCache()
        keys, values = cache.get(99)
        assert keys is None
        assert values is None

    def test_reset_single_layer(self):
        cache = NeuralKVCache(num_layers=2, head_dim=8, max_positions=10)
        cache.initialize(num_heads=2)
        cache.update(0, np.zeros((2, 8)), np.zeros((2, 8)))
        cache.reset(layer_idx=0)
        keys, values = cache.get(0)
        assert keys is None

    def test_reset_all(self):
        cache = NeuralKVCache(num_layers=2, head_dim=8, max_positions=10)
        cache.initialize(num_heads=2)
        cache.reset()
        assert cache.get_position() == 0
        stats = cache.stats()
        assert stats["layers_cached"] == 0

    def test_stats(self):
        cache = NeuralKVCache(num_layers=1, head_dim=8, max_positions=10)
        cache.initialize(num_heads=2)
        s = cache.stats()
        assert s["layers_cached"] == 1
        assert "memory_bytes" in s
        assert "memory_mb" in s
        assert "position" in s


# ---------------------------------------------------------------------------
# EmbeddingEntry dataclass
# ---------------------------------------------------------------------------

class TestEmbeddingEntry:
    def test_fields(self):
        vec = np.array([1.0, 2.0, 3.0])
        entry = EmbeddingEntry(id="e1", vector=vec, text="hello")
        assert entry.id == "e1"
        assert entry.text == "hello"
        assert np.array_equal(entry.vector, vec)
        assert isinstance(entry.created_at, float)
        assert entry.metadata == {}


# ---------------------------------------------------------------------------
# NeuralEmbeddingStore
# ---------------------------------------------------------------------------

class TestNeuralEmbeddingStore:
    def test_init_defaults(self):
        store = NeuralEmbeddingStore()
        assert store.vocab_size == 1000
        assert store.embed_dim == 64
        assert store.dim == 64
        assert store.size == 0

    def test_init_with_dim(self):
        store = NeuralEmbeddingStore(dim=128)
        assert store.dim == 128

    def test_lookup(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=4)
        ids = np.array([0, 5])
        result = store.lookup(ids)
        assert result.shape == (2, 4)

    def test_update(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=4)
        ids = np.array([0, 1])
        vecs = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        count = store.update(ids, vecs)
        assert count == 2

    def test_update_out_of_range(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=4)
        ids = np.array([5, 15])  # 15 out of range
        vecs = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        count = store.update(ids, vecs)
        assert count == 1  # only first one updated

    def test_similarity_identical(self):
        store = NeuralEmbeddingStore()
        a = np.array([1.0, 0.0, 0.0])
        assert store.similarity(a, a) == pytest.approx(1.0)

    def test_similarity_orthogonal(self):
        store = NeuralEmbeddingStore()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert store.similarity(a, b) == pytest.approx(0.0)

    def test_similarity_zero_vector(self):
        store = NeuralEmbeddingStore()
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert store.similarity(a, b) == 0.0

    def test_nearest(self):
        store = NeuralEmbeddingStore(vocab_size=5, embed_dim=3)
        store._embeddings[0] = np.array([1.0, 0.0, 0.0])
        store._embeddings[1] = np.array([0.0, 1.0, 0.0])
        store._embeddings[2] = np.array([0.0, 0.0, 1.0])
        query = np.array([1.0, 0.0, 0.0])
        results = store.nearest(query, k=2)
        assert len(results) == 2
        assert results[0][0] == 0  # closest

    def test_nearest_zero_query(self):
        store = NeuralEmbeddingStore(vocab_size=5, embed_dim=3)
        query = np.zeros(3)
        results = store.nearest(query, k=5)
        assert results == []

    def test_nearest_all_zero_embeddings(self):
        store = NeuralEmbeddingStore(vocab_size=5, embed_dim=3)
        store._embeddings[:] = 0
        query = np.array([1.0, 0.0, 0.0])
        results = store.nearest(query, k=5)
        assert results == []

    def test_add(self):
        store = NeuralEmbeddingStore()
        vec = np.array([1.0, 2.0, 3.0])
        store.add("e1", vec, "hello", {"key": "val"})
        assert store.size == 1
        entry = list(store._entries.values())[0]
        assert entry.text == "hello"
        assert entry.metadata == {"key": "val"}
        # vector is normalized
        assert abs(np.linalg.norm(entry.vector) - 1.0) < 1e-6

    def test_stats(self):
        store = NeuralEmbeddingStore(vocab_size=500, embed_dim=32)
        s = store.stats()
        assert s["vocab_size"] == 500
        assert s["embed_dim"] == 32
        assert s["entries"] == 0


# ---------------------------------------------------------------------------
# GradientAccumulator
# ---------------------------------------------------------------------------

class TestGradientAccumulator:
    def test_init(self):
        ga = GradientAccumulator(max_grad_norm=2.0, accumulation_steps=4)
        assert ga.step_count == 0
        assert ga.ready is False

    def test_accumulate(self):
        ga = GradientAccumulator(accumulation_steps=2)
        ready = ga.accumulate({"w": np.array([1.0])})
        assert ready is False
        assert ga.step_count == 1
        ready = ga.accumulate({"w": np.array([2.0])})
        assert ready is True
        assert ga.step_count == 2

    def test_get_clipped_gradients_under_threshold(self):
        ga = GradientAccumulator(max_grad_norm=10.0, accumulation_steps=1)
        ga.accumulate({"w": np.array([1.0, 0.0])})
        grads = ga.get_clipped_gradients()
        assert "w" in grads
        assert np.allclose(grads["w"], [1.0, 0.0])

    def test_get_clipped_gradients_over_threshold(self):
        ga = GradientAccumulator(max_grad_norm=1.0, accumulation_steps=1)
        ga.accumulate({"w": np.array([3.0, 4.0])})  # norm = 5.0
        grads = ga.get_clipped_gradients()
        # scale = 1.0 / 5.0 = 0.2
        assert np.allclose(grads["w"], [0.6, 0.8])

    def test_reset(self):
        ga = GradientAccumulator(accumulation_steps=2)
        ga.accumulate({"w": np.array([1.0])})
        ga.reset()
        assert ga.step_count == 0
        assert ga.ready is False
        assert ga.stats()["total_norm"] == 0.0

    def test_stats(self):
        ga = GradientAccumulator(max_grad_norm=1.0, accumulation_steps=3)
        s = ga.stats()
        assert s["accumulation_steps"] == 3
        assert s["max_grad_norm"] == 1.0
        assert s["ready"] is False


# ---------------------------------------------------------------------------
# BatchRequest / BatchResult dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_batch_request_defaults(self):
        req = BatchRequest(id="r1", inputs={"x": np.array([1])})
        assert req.priority == 0
        assert req.callback is None
        assert isinstance(req.created_at, float)

    def test_batch_result_defaults(self):
        res = BatchResult(id="r1", outputs={"y": np.array([2])})
        assert res.elapsed_ms == 0.0
        assert res.error is None


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------

class TestBatchProcessor:
    def test_submit(self):
        bp = BatchProcessor(max_batch_size=4)
        req = BatchRequest(id="r1", inputs={"x": np.array([1])})
        assert bp.submit(req) is True
        assert bp.queue_size == 1

    def test_submit_over_capacity(self):
        bp = BatchProcessor(max_batch_size=2)
        for i in range(5):
            bp.submit(BatchRequest(id=f"r{i}", inputs={}))
        assert bp.submit(BatchRequest(id="overflow", inputs={})) is False

    def test_process_batch(self):
        bp = BatchProcessor(max_batch_size=2)
        bp.submit(BatchRequest(id="r1", inputs={"x": np.array([1])}))
        bp.submit(BatchRequest(id="r2", inputs={"x": np.array([2])}))
        results = bp.process_batch()
        assert len(results) == 2
        assert results[0].id == "r1"
        assert bp.queue_size == 0

    def test_process_batch_empty(self):
        bp = BatchProcessor()
        assert bp.process_batch() == []

    def test_process_batch_with_callback(self):
        collected = []
        bp = BatchProcessor(max_batch_size=1)
        req = BatchRequest(id="r1", inputs={}, callback=lambda r: collected.append(r))
        bp.submit(req)
        bp.process_batch()
        assert len(collected) == 1

    def test_process_batch_with_error(self):
        def bad_fn(inputs):
            raise ValueError("boom")
        bp = BatchProcessor(max_batch_size=1, process_fn=bad_fn)
        bp.submit(BatchRequest(id="r1", inputs={}))
        results = bp.process_batch()
        assert results[0].error == "boom"

    def test_flush(self):
        bp = BatchProcessor(max_batch_size=2)
        bp.submit(BatchRequest(id="r1", inputs={}))
        bp.submit(BatchRequest(id="r2", inputs={}))
        bp.submit(BatchRequest(id="r3", inputs={}))
        all_results = bp.flush()
        assert len(all_results) == 3
        assert bp.queue_size == 0

    def test_stats(self):
        bp = BatchProcessor(max_batch_size=5)
        s = bp.stats()
        assert s["max_batch_size"] == 5
        assert s["total_batches"] == 0


# ---------------------------------------------------------------------------
# NeuralInterrupt static methods
# ---------------------------------------------------------------------------

class TestNeuralInterrupt:
    def test_inference_done(self):
        intr = NeuralInterrupt.inference_done(pid=42, result="ok")
        assert intr.vector == InterruptType.INFERENCE_DONE
        assert intr.source_pid == 42
        assert intr.data == "ok"

    def test_training_step(self):
        intr = NeuralInterrupt.training_step(pid=1, loss=0.5, step=10)
        assert intr.vector == InterruptType.TRAINING_STEP
        assert intr.data["loss"] == 0.5
        assert intr.data["step"] == 10

    def test_gradient_update(self):
        intr = NeuralInterrupt.gradient_update(pid=1, grad_norm=1.23)
        assert intr.vector == InterruptType.GRADIENT_UPDATE
        assert intr.data["grad_norm"] == 1.23

    def test_data_ready(self):
        intr = NeuralInterrupt.data_ready(pid=1, batch_size=32)
        assert intr.vector == InterruptType.DATA_READY
        assert intr.data["batch_size"] == 32


# ---------------------------------------------------------------------------
# NeuralEngineDevice
# ---------------------------------------------------------------------------

class TestNeuralEngineDevice:
    def test_init(self):
        dev = NeuralEngineDevice()
        assert dev.name == "neural_engine"
        assert dev.device_type == DeviceType.INFERENCE

    def test_load_unload_model(self):
        dev = NeuralEngineDevice()
        dev.load_model("m1", "model_obj")
        info = dev.info()
        assert "m1" in info["model_names"]
        assert info["models_loaded"] == 1
        dev.unload_model("m1")
        assert dev.info()["models_loaded"] == 0

    def test_write_increments_count(self):
        dev = NeuralEngineDevice()
        dev.write("data")
        assert dev.info()["request_count"] == 1

    def test_ioctl_forward_missing_model(self):
        dev = NeuralEngineDevice()
        result = dev.ioctl("forward", "nonexistent")
        assert isinstance(result, SyscallResult)
        assert result.success is False

    def test_ioctl_generate_missing_model(self):
        dev = NeuralEngineDevice()
        result = dev.ioctl("generate", "nonexistent", "prompt")
        assert result.success is False

    def test_ioctl_attention(self):
        dev = NeuralEngineDevice()
        q = np.random.rand(1, 4, 8)
        k = np.random.rand(1, 4, 8)
        v = np.random.rand(1, 4, 8)
        result = dev.ioctl("attention", q, k, v)
        assert result is not None
        assert result.success is True
        assert "output" in result.value

    def test_ioctl_loss_mse(self):
        dev = NeuralEngineDevice()
        pred = np.array([1.0, 2.0, 3.0])
        tgt = np.array([1.0, 2.0, 3.0])
        result = dev.ioctl("loss", pred, tgt)
        assert result.success is True
        assert result.value["loss"] == pytest.approx(0.0)

    def test_ioctl_loss_cross_entropy(self):
        dev = NeuralEngineDevice()
        pred = np.array([[0.9, 0.1], [0.1, 0.9]])
        tgt = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = dev.ioctl("loss", pred, tgt, loss_fn="cross_entropy")
        assert result.success is True
        assert result.value["loss"] < 0.5  # good predictions → low loss

    def test_ioctl_unknown_command(self):
        dev = NeuralEngineDevice()
        assert dev.ioctl("unknown_cmd") is None

    def test_read_returns_models(self):
        dev = NeuralEngineDevice()
        dev.load_model("m1", "x")
        models = dev.read()
        assert "m1" in models


# ---------------------------------------------------------------------------
# TokenizerDevice
# ---------------------------------------------------------------------------

class TestTokenizerDevice:
    def test_init(self):
        dev = TokenizerDevice()
        assert dev.name == "tokenizer"
        assert dev.device_type == DeviceType.CUSTOM

    def test_read_no_tokenizer(self):
        dev = TokenizerDevice()
        assert dev.read() is None

    def test_read_with_tokenizer(self):
        tok = type("Tok", (), {"vocab_size": 1000})()
        dev = TokenizerDevice(tokenizer=tok)
        assert dev.read() == 1000

    def test_write_no_tokenizer(self):
        dev = TokenizerDevice()
        assert dev.write("hello") is False

    def test_write_non_string(self):
        tok = type("Tok", (), {})()
        dev = TokenizerDevice(tokenizer=tok)
        assert dev.write(123) is False

    def test_ioctl_encode_no_tokenizer(self):
        dev = TokenizerDevice()
        result = dev.ioctl("encode", "hello")
        assert result.success is True
        assert result.value["encoding"] == "byte-level"

    def test_ioctl_decode_no_tokenizer(self):
        dev = TokenizerDevice()
        result = dev.ioctl("decode", [72, 101])
        assert result.success is True
        assert "He" in result.value["text"]


# ---------------------------------------------------------------------------
# EmbeddingStoreDevice
# ---------------------------------------------------------------------------

class TestEmbeddingStoreDevice:
    def test_init(self):
        dev = EmbeddingStoreDevice()
        assert dev.name == "embedding-store"
        assert dev.device_type == DeviceType.STORAGE

    def test_init_with_store(self):
        store = NeuralEmbeddingStore()
        dev = EmbeddingStoreDevice(store=store)
        assert dev.get_store("default") is store

    def test_create_store(self):
        dev = EmbeddingStoreDevice()
        store = dev.create_store("my_store", vocab_size=500, embed_dim=32)
        assert dev.get_store("my_store") is store
        assert store.vocab_size == 500

    def test_get_nonexistent_store(self):
        dev = EmbeddingStoreDevice()
        assert dev.get_store("nope") is None

    def test_read(self):
        dev = EmbeddingStoreDevice()
        dev.create_store("s1")
        result = dev.read()
        assert "s1" in result

    def test_write_always_false(self):
        dev = EmbeddingStoreDevice()
        assert dev.write("data") is False

    def test_ioctl_create(self):
        dev = EmbeddingStoreDevice()
        result = dev.ioctl("create", store_name="new_store")
        assert result.success is True
        assert dev.get_store("new_store") is not None

    def test_ioctl_lookup(self):
        dev = EmbeddingStoreDevice()
        dev.create_store("s1", vocab_size=10, embed_dim=4)
        result = dev.ioctl("lookup", [0, 1], store_name="s1")
        assert result.success is True
        assert result.value["vectors"].shape == (2, 4)

    def test_ioctl_nonexistent_store(self):
        dev = EmbeddingStoreDevice()
        result = dev.ioctl("lookup", [], store_name="nope")
        assert result.success is False

    def test_info(self):
        dev = EmbeddingStoreDevice()
        dev.create_store("s1")
        info = dev.info()
        assert "s1" in info["stores"]


# ---------------------------------------------------------------------------
# MultiHeadAttentionDevice
# ---------------------------------------------------------------------------

class TestMultiHeadAttentionDevice:
    def test_init(self):
        dev = MultiHeadAttentionDevice(num_heads=8, head_dim=64)
        assert dev.name == "mha-device"
        assert dev._num_heads == 8

    def test_compute_attention(self):
        dev = MultiHeadAttentionDevice()
        q = np.random.rand(1, 4, 16)
        k = np.random.rand(1, 4, 16)
        v = np.random.rand(1, 4, 16)
        out = dev._compute_attention(q, k, v)
        assert out.shape == (1, 4, 16)
        assert dev._compute_count == 1

    def test_compute_attention_with_mask(self):
        dev = MultiHeadAttentionDevice()
        q = np.random.rand(1, 3, 8)
        k = np.random.rand(1, 3, 8)
        v = np.random.rand(1, 3, 8)
        mask = np.array([[[True, True, False], [True, True, True], [True, True, True]]])
        out = dev._compute_attention(q, k, v, mask)
        assert out.shape == (1, 3, 8)

    def test_ioctl_attention(self):
        dev = MultiHeadAttentionDevice()
        q = np.random.rand(1, 2, 8)
        k = np.random.rand(1, 2, 8)
        v = np.random.rand(1, 2, 8)
        result = dev.ioctl("attention", q, k, v)
        assert result is not None
        assert result.shape == (1, 2, 8)

    def test_ioctl_unknown(self):
        dev = MultiHeadAttentionDevice()
        assert dev.ioctl("unknown") is None

    def test_info(self):
        dev = MultiHeadAttentionDevice(num_heads=4, head_dim=32)
        info = dev.info()
        assert info["num_heads"] == 4
        assert info["head_dim"] == 32
        assert info["compute_count"] == 0

    def test_read(self):
        dev = MultiHeadAttentionDevice()
        dev._compute_attention(
            np.random.rand(1, 2, 8),
            np.random.rand(1, 2, 8),
            np.random.rand(1, 2, 8),
        )
        assert dev.read()["compute_count"] == 1


# ---------------------------------------------------------------------------
# NeuralSyscall static methods
# ---------------------------------------------------------------------------

class TestNeuralSyscall:
    def test_constants(self):
        assert NeuralSyscall.TOKENIZE == 20000
        assert NeuralSyscall.GENERATE == 20001

    def test_embed(self):
        store = NeuralEmbeddingStore(dim=64)
        vec = NeuralSyscall.embed(store, "hello")
        assert vec.shape == (64,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    def test_attention_via_device(self):
        dev = MultiHeadAttentionDevice()
        q = np.random.rand(1, 3, 8)
        k = np.random.rand(1, 3, 8)
        v = np.random.rand(1, 3, 8)
        result = NeuralSyscall.attention(dev, q, k, v)
        assert result.shape == q.shape

    def test_attention_no_device(self):
        q = np.random.rand(1, 3, 8)
        result = NeuralSyscall.attention(object(), q, q, q)
        assert result.shape == q.shape  # returns zeros_like(q)
