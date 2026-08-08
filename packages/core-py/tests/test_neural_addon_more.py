"""Coverage-completing tests for the neural addon (domains.shell.addons.neural).

Run: PYTHONPATH=packages/core-py python -m pytest tests/test_neural_addon_more.py -q
"""

from types import SimpleNamespace

import numpy as np
import pytest

from domains.shell.addons import neural
from domains.shell.addons.neural import (
    BatchProcessor,
    BatchRequest,
    CacheStrategy,
    EmbeddingStoreDevice,
    GradientAccumulator,
    MultiHeadAttentionDevice,
    NeuralEmbeddingStore,
    NeuralEngineDevice,
    NeuralInterrupt,
    NeuralKVCache,
    NeuralMemoryType,
    NeuralOp,
    NeuralProcess,
    NeuralProcessType,
    NeuralState,
    NeuralSyscall,
    NeuralKVCache,
    TokenizerDevice,
)
from domains.shell.kernel import Kernel
from domains.shell.kernel_devices import DeviceType
from domains.shell.kernel_interrupts import InterruptType
from domains.shell.kernel_process import Process, ProcessState


def _make_proc(pid: int = 7, name: str = "neural-worker") -> Process:
    return Process(pid=pid, name=name, state=ProcessState.READY)


class TestEnums:
    def test_neural_op_values(self):
        assert NeuralOp.NONE == 0
        assert NeuralOp.EMBEDDING == 1
        assert NeuralOp.OPTIMIZER_STEP == 9

    def test_neural_state_values(self):
        assert NeuralState.IDLE == 0
        assert NeuralState.BACKPROPAGATING == 6

    def test_memory_type_values(self):
        assert NeuralMemoryType.KV_CACHE == 0
        assert NeuralMemoryType.FAILED == 8

    def test_cache_strategy_values(self):
        assert CacheStrategy.LRU == 0
        assert CacheStrategy.PRIORITY == 3


class TestNeuralProcess:
    def test_name_property(self):
        proc = NeuralProcess(process=_make_proc(pid=7, name="worker"))
        assert proc.name == "worker"
        assert proc.pid == 7

    def test_state_property(self):
        proc = NeuralProcess(process=_make_proc())
        assert proc.state == ProcessState.READY

    def test_is_computing_states(self):
        proc = NeuralProcess(process=_make_proc())
        assert proc.is_computing is False
        for state in (NeuralState.COMPUTING, NeuralState.BACKPROPAGATING, NeuralState.OPTIMIZING):
            proc.neural_state = state
            assert proc.is_computing is True
        proc.neural_state = NeuralState.IDLE
        assert proc.is_computing is False

    def test_compute_time_ms_no_start(self):
        proc = NeuralProcess(process=_make_proc())
        assert proc.compute_time_ms == 0.0

    def test_compute_time_ms_with_end(self):
        proc = NeuralProcess(process=_make_proc())
        proc.compute_start = 100.0
        proc.compute_end = 100.5
        assert proc.compute_time_ms == pytest.approx(500.0)

    def test_compute_time_ms_live(self):
        proc = NeuralProcess(process=_make_proc())
        proc.compute_start = 100.0
        proc.compute_end = None
        assert proc.compute_time_ms > 0

    def test_transition_to_computing_starts_timer(self):
        proc = NeuralProcess(process=_make_proc())
        proc.transition_neural(NeuralState.COMPUTING)
        assert proc.compute_start is not None
        assert proc.compute_end is None

    def test_transition_to_complete_stops_timer(self):
        proc = NeuralProcess(process=_make_proc())
        proc.transition_neural(NeuralState.COMPUTING)
        proc.transition_neural(NeuralState.COMPLETE)
        assert proc.compute_end is not None

    def test_transition_to_failed_stops_timer(self):
        proc = NeuralProcess(process=_make_proc())
        proc.transition_neural(NeuralState.COMPUTING)
        proc.transition_neural(NeuralState.FAILED)
        assert proc.compute_end is not None

    def test_transition_other_state_no_timing(self):
        proc = NeuralProcess(process=_make_proc())
        proc.compute_start = None
        proc.transition_neural(NeuralState.IDLE)
        assert proc.compute_start is None

    def test_store_gradient(self):
        proc = NeuralProcess(process=_make_proc())
        proc.store_gradient("w", np.array([1.0, 2.0]))
        assert "w" in proc.gradients
        assert proc.gradient_norm == pytest.approx(5.0)

    def test_clear_gradients(self):
        proc = NeuralProcess(process=_make_proc())
        proc.store_gradient("w", np.array([1.0, 2.0]))
        proc.clear_gradients()
        assert proc.gradients == {}
        assert proc.gradient_norm == 0.0

    def test_status_line(self):
        proc = NeuralProcess(process=_make_proc(), model_name="gpt2")
        line = proc.status_line()
        assert line.endswith("[IDLE] model=gpt2")


class TestNeuralKVCache:
    def test_total_tokens_cached_property(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        assert cache.total_tokens_cached == 0

    def test_evictions_property(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        assert cache.evictions == 0

    def test_update_out_of_range_raises(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        cache.initialize(num_heads=4)
        with pytest.raises(ValueError, match="out of range"):
            cache.update(5, np.zeros((4, 16)), np.zeros((4, 16)))
        with pytest.raises(ValueError, match="out of range"):
            cache.update(-1, np.zeros((4, 16)), np.zeros((4, 16)))

    def test_update_creates_layer_on_demand(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        pos = cache.update(0, np.zeros((4, 16)), np.zeros((4, 16)))
        assert pos == 0
        entry = cache._entries[0]
        assert entry.keys.shape == (4, 50, 16)

    def test_get_missing_layer(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        k, v = cache.get(3)
        assert k is None
        assert v is None

    def test_get_default_end(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        cache.initialize(num_heads=4)
        cache.update(0, np.random.randn(4, 16), np.random.randn(4, 16))
        cache.advance(1)
        cache.update(0, np.random.randn(4, 16), np.random.randn(4, 16))
        cache.advance(1)
        k, v = cache.get(0)
        assert k.shape == (4, 2, 16)
        assert v.shape == (4, 2, 16)

    def test_get_with_explicit_end(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        cache.initialize(num_heads=4)
        cache.update(0, np.random.randn(4, 16), np.random.randn(4, 16))
        cache.advance(3)
        k, v = cache.get(0, start=1, end=2)
        assert k.shape == (4, 1, 16)

    def test_reset_single_layer(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        cache.initialize(num_heads=4)
        cache.update(0, np.zeros((4, 16)), np.zeros((4, 16)))
        cache.reset(layer_idx=0)
        assert 0 not in cache._entries
        assert 1 in cache._entries

    def test_reset_all(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        cache.initialize(num_heads=4)
        cache.advance(7)
        cache.reset()
        assert cache._entries == {}
        assert cache.get_position() == 0

    def test_stats(self):
        cache = NeuralKVCache(num_layers=2, head_dim=16, max_positions=50)
        cache.initialize(num_heads=4)
        stats = cache.stats()
        assert stats["layers_cached"] == 2
        assert stats["max_positions"] == 50
        assert "memory_mb" in stats


class TestNeuralEmbeddingStore:
    def test_dim_override(self):
        store = NeuralEmbeddingStore(vocab_size=100, embed_dim=32, dim=128)
        assert store.dim == 128

    def test_dim_defaults_to_embed_dim(self):
        store = NeuralEmbeddingStore(vocab_size=100, embed_dim=32)
        assert store.dim == 32

    def test_size(self):
        store = NeuralEmbeddingStore(vocab_size=100, embed_dim=32)
        assert store.size == 0
        store.add("a", np.ones(32), "hello")
        assert store.size == 1

    def test_similarity_zero_norm(self):
        store = NeuralEmbeddingStore(vocab_size=100, embed_dim=32)
        assert store.similarity(np.zeros(32), np.ones(32)) == 0.0
        assert store.similarity(np.ones(32), np.zeros(32)) == 0.0

    def test_nearest_no_valid_embeddings(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=4)
        store._embeddings = np.zeros((10, 4))
        assert store.nearest(np.ones(4)) == []

    def test_nearest_zero_query(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=4)
        store._embeddings[0] = np.ones(4)
        assert store.nearest(np.zeros(4)) == []

    def test_add_normalizes_vector(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=4)
        store.add("id", np.full(4, 5.0), "text", metadata={"k": 1})
        entry = store._entries["id"]
        assert np.linalg.norm(entry.vector) > 0.999
        assert entry.metadata == {"k": 1}

    def test_add_default_metadata(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=4)
        store.add("id", np.ones(4), "text")
        assert store._entries["id"].metadata == {}

    def test_search_empty_store(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=4)
        assert store.search(np.ones(4)) == []

    def test_search_with_entries(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=4)
        store._embeddings = np.zeros((10, 4))
        store._embeddings[0] = np.ones(4)
        store._embeddings[1] = np.ones(4) * 0.5
        store.add("first", np.ones(4), "first text")
        store.add("second", np.ones(4), "second text")
        results = store.search(np.ones(4), top_k=2)
        assert len(results) == 2
        assert results[0][0] == "first"
        assert results[0][2] == "first text"


class TestNeuralEngineDevice:
    def test_device_type(self):
        dev = NeuralEngineDevice()
        assert dev.device_type == DeviceType.INFERENCE

    def test_read_returns_models(self):
        dev = NeuralEngineDevice()
        model = lambda x: x
        dev.load_model("m", model)
        assert dev.read() == {"m": model}

    def test_write_increments_request_count(self):
        dev = NeuralEngineDevice()
        assert dev.write("anything") is True
        assert dev._request_count == 1

    def test_generate_missing_model(self):
        dev = NeuralEngineDevice()
        result = dev.ioctl("generate", "ghost", "hello")
        assert result.success is False
        assert "not found" in result.error

    def test_ioctl_unknown_command(self):
        dev = NeuralEngineDevice()
        assert dev.ioctl("bogus") is None


class TestTokenizerDevice:
    def test_read_without_tokenizer(self):
        dev = TokenizerDevice()
        assert dev.read() is None

    def test_read_with_tokenizer(self):
        tok = SimpleNamespace(vocab_size=123)
        dev = TokenizerDevice(tokenizer=tok)
        assert dev.read() == 123

    def test_write_without_tokenizer(self):
        dev = TokenizerDevice()
        assert dev.write("hello") is False

    def test_write_non_string(self):
        tok = SimpleNamespace()
        dev = TokenizerDevice(tokenizer=tok)
        assert dev.write(42) is False

    def test_write_tokenizer_without_encode(self):
        tok = SimpleNamespace()
        dev = TokenizerDevice(tokenizer=tok)
        assert dev.write("hello") is True

    def test_write_success(self):
        tok = SimpleNamespace(encode=lambda text: [ord(c) for c in text])
        dev = TokenizerDevice(tokenizer=tok)
        assert dev.write("abc") is True
        assert dev._tokenize_count == 1
        assert dev._total_tokens == 3

    def test_write_encode_raises(self):
        class Boom:
            def encode(self, text):
                raise RuntimeError("boom")

        dev = TokenizerDevice(tokenizer=Boom())
        assert dev.write("abc") is False

    def test_encode_with_custom_tokenizer(self):
        tok = SimpleNamespace(encode=lambda text: [1, 2, 3])
        dev = TokenizerDevice(tokenizer=tok)
        result = dev.ioctl("encode", "hello")
        assert result.success
        assert result.value["tokens"] == [1, 2, 3]
        assert result.value["encoding"] == "custom"

    def test_decode_with_custom_tokenizer(self):
        tok = SimpleNamespace(decode=lambda tokens: "".join(chr(t) for t in tokens))
        dev = TokenizerDevice(tokenizer=tok)
        result = dev.ioctl("decode", [97, 98])
        assert result.success
        assert result.value["text"] == "ab"

    def test_decode_byte_level(self):
        dev = TokenizerDevice()
        result = dev.ioctl("decode", [104, 105])
        assert result.value["text"] == "hi"

    def test_decode_byte_level_invalid_bytes(self):
        dev = TokenizerDevice()
        result = dev.ioctl("decode", [0xFF, 0xFE])
        assert result.success
        assert "\uFFFD" in result.value["text"]

    def test_ioctl_unknown_command(self):
        dev = TokenizerDevice()
        assert dev.ioctl("bogus") is None


class TestEmbeddingStoreDevice:
    def test_constructor_with_store(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=4)
        dev = EmbeddingStoreDevice(store=store)
        assert dev.get_store("default") is store

    def test_create_store_returns_store(self):
        dev = EmbeddingStoreDevice()
        store = dev.create_store("s", 20, 8)
        assert store.vocab_size == 20
        assert store.embed_dim == 8
        assert dev.get_store("s") is store

    def test_read(self):
        dev = EmbeddingStoreDevice()
        dev.create_store("s", 20, 8)
        info = dev.read()
        assert "s" in info

    def test_write(self):
        dev = EmbeddingStoreDevice()
        assert dev.write("anything") is False

    def test_ioctl_missing_store(self):
        dev = EmbeddingStoreDevice()
        result = dev.ioctl("lookup", [1, 2], store_name="nope")
        assert result.success is False
        assert "not found" in result.error

    def test_ioctl_create(self):
        dev = EmbeddingStoreDevice()
        result = dev.ioctl("create", store_name="new", vocab_size=10, embed_dim=4)
        assert result.success
        assert dev.get_store("new") is not None

    def test_ioctl_unknown_command_with_store(self):
        dev = EmbeddingStoreDevice()
        dev.create_store("s", 10, 4)
        assert dev.ioctl("bogus", store_name="s") is None


class TestMultiHeadAttentionDevice:
    def test_device_type(self):
        dev = MultiHeadAttentionDevice(num_heads=4, head_dim=8)
        assert dev.device_type == DeviceType.CUSTOM

    def test_compute_attention_without_mask(self):
        dev = MultiHeadAttentionDevice(num_heads=1, head_dim=4)
        q = np.random.randn(1, 2, 4)
        k = np.random.randn(1, 2, 4)
        v = np.random.randn(1, 2, 4)
        out = dev._compute_attention(q, k, v)
        assert out.shape == (1, 2, 4)
        assert dev._compute_count == 1

    def test_compute_attention_with_mask(self):
        dev = MultiHeadAttentionDevice(num_heads=1, head_dim=4)
        q = np.random.randn(1, 2, 4)
        k = np.random.randn(1, 2, 4)
        v = np.random.randn(1, 2, 4)
        mask = np.array([[[True, False], [True, True]]])
        out = dev._compute_attention(q, k, v, mask)
        assert out.shape == (1, 2, 4)

    def test_read(self):
        dev = MultiHeadAttentionDevice()
        assert dev.read() == {"compute_count": 0}

    def test_write(self):
        dev = MultiHeadAttentionDevice()
        assert dev.write("data") is False

    def test_ioctl_attention(self):
        dev = MultiHeadAttentionDevice(num_heads=1, head_dim=4)
        q = np.random.randn(1, 2, 4)
        k = np.random.randn(1, 2, 4)
        v = np.random.randn(1, 2, 4)
        out = dev.ioctl("attention", q, k, v)
        assert out.shape == (1, 2, 4)

    def test_ioctl_unknown_command(self):
        dev = MultiHeadAttentionDevice()
        assert dev.ioctl("bogus") is None


class TestNeuralInterrupt:
    def test_inference_done(self):
        it = NeuralInterrupt.inference_done(5, result={"ok": True})
        assert it.vector == InterruptType.INFERENCE_DONE
        assert it.source_pid == 5
        assert it.data == {"ok": True}

    def test_training_step(self):
        it = NeuralInterrupt.training_step(5, loss=0.5, step=3)
        assert it.vector == InterruptType.TRAINING_STEP
        assert it.data == {"loss": 0.5, "step": 3}

    def test_gradient_update(self):
        it = NeuralInterrupt.gradient_update(5, grad_norm=2.5)
        assert it.vector == InterruptType.GRADIENT_UPDATE
        assert it.data == {"grad_norm": 2.5}

    def test_data_ready(self):
        it = NeuralInterrupt.data_ready(5, batch_size=32)
        assert it.vector == InterruptType.DATA_READY
        assert it.data == {"batch_size": 32}


class _Model:
    def forward(self, inputs):
        return {"out": inputs["x"] * 2}

    def backward(self, grad_output):
        return {"grad": grad_output["out"]}


class _BadModel:
    def forward(self, inputs):
        raise RuntimeError("forward boom")

    def backward(self, grad_output):
        raise RuntimeError("backward boom")


class TestNeuralSyscall:
    def test_forward_success(self):
        proc = NeuralProcess(process=_make_proc(), model_ref=_Model())
        inputs = {"x": np.array([1.0, 2.0])}
        out = NeuralSyscall.forward(proc, inputs)
        np.testing.assert_array_equal(out["out"], np.array([2.0, 4.0]))
        assert proc.neural_state == NeuralState.COMPLETE
        assert proc.output_tensors is out

    def test_forward_no_model_ref(self):
        proc = NeuralProcess(process=_make_proc())
        inputs = {"x": np.array([1.0])}
        out = NeuralSyscall.forward(proc, inputs)
        assert out is inputs
        assert proc.neural_state == NeuralState.COMPLETE

    def test_forward_raises(self):
        proc = NeuralProcess(process=_make_proc(), model_ref=_BadModel())
        with pytest.raises(RuntimeError, match="forward boom"):
            NeuralSyscall.forward(proc, {"x": np.array([1.0])})
        assert proc.neural_state == NeuralState.FAILED
        assert proc.last_error == "forward boom"

    def test_backward_success(self):
        proc = NeuralProcess(process=_make_proc(), model_ref=_Model())
        grads = NeuralSyscall.backward(proc, {"out": np.array([1.0])})
        assert "grad" in grads
        assert "grad" in proc.gradients
        assert proc.neural_state == NeuralState.COMPLETE

    def test_backward_no_model_ref(self):
        proc = NeuralProcess(process=_make_proc())
        grads = NeuralSyscall.backward(proc, {"out": np.array([1.0])})
        assert grads == {}
        assert proc.neural_state == NeuralState.COMPLETE

    def test_backward_raises(self):
        proc = NeuralProcess(process=_make_proc(), model_ref=_BadModel())
        with pytest.raises(RuntimeError, match="backward boom"):
            NeuralSyscall.backward(proc, {"out": np.array([1.0])})
        assert proc.neural_state == NeuralState.FAILED
        assert proc.last_error == "backward boom"

    def test_embed_with_dim(self):
        store = NeuralEmbeddingStore(vocab_size=10, embed_dim=8)
        vec = NeuralSyscall.embed(store, "hello")
        assert vec.shape == (8,)
        assert vec.dtype == np.float32

    def test_embed_without_dim(self):
        plain = SimpleNamespace()
        vec = NeuralSyscall.embed(plain, "hello")
        assert vec.shape == (384,)
        assert np.all(vec == 0)

    def test_attention_with_compute(self):
        dev = MultiHeadAttentionDevice(num_heads=1, head_dim=4)
        q = np.random.randn(1, 2, 4)
        k = np.random.randn(1, 2, 4)
        v = np.random.randn(1, 2, 4)
        out = NeuralSyscall.attention(dev, q, k, v)
        assert out.shape == (1, 2, 4)

    def test_attention_without_compute(self):
        dev = SimpleNamespace()
        q = np.ones((2, 3))
        out = NeuralSyscall.attention(dev, q, np.ones((2, 3)), np.ones((2, 3)))
        np.testing.assert_array_equal(out, np.zeros_like(q))


class TestGradientAccumulator:
    def test_step_count(self):
        acc = GradientAccumulator()
        acc.accumulate({"w": np.array([1.0])})
        assert acc.step_count == 1

    def test_ready_after_accumulation_steps(self):
        acc = GradientAccumulator(accumulation_steps=2)
        assert acc.accumulate({"w": np.array([2.0])}) is False
        assert acc.ready is False
        assert acc.accumulate({"w": np.array([4.0])}) is True
        assert acc.ready is True

    def test_accumulate_averages_over_steps(self):
        acc = GradientAccumulator(accumulation_steps=2)
        acc.accumulate({"w": np.array([2.0])})
        acc.accumulate({"w": np.array([4.0])})
        np.testing.assert_array_equal(acc._accumulated["w"], np.array([3.0]))

    def test_get_clipped_gradients_when_under_limit(self):
        acc = GradientAccumulator(max_grad_norm=10.0)
        acc.accumulate({"w": np.array([1.0, 2.0])})
        grads = acc.get_clipped_gradients()
        np.testing.assert_array_equal(grads["w"], np.array([1.0, 2.0]))

    def test_get_clipped_gradients_when_over_limit(self):
        acc = GradientAccumulator(max_grad_norm=1.0)
        acc.accumulate({"w": np.array([3.0, 4.0])})
        grads = acc.get_clipped_gradients()
        norm = np.linalg.norm(grads["w"])
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_reset(self):
        acc = GradientAccumulator(accumulation_steps=2)
        acc.accumulate({"w": np.array([1.0])})
        acc.reset()
        assert acc._accumulated == {}
        assert acc.step_count == 0
        assert acc._total_norm == 0.0


class TestBatchProcessor:
    def test_queue_size_empty(self):
        bp = BatchProcessor()
        assert bp.queue_size == 0

    def test_queue_size_after_submit(self):
        bp = BatchProcessor()
        req = BatchRequest(id="r1", inputs={"x": np.array([1.0])})
        assert bp.submit(req) is True
        assert bp.queue_size == 1

    def test_submit_when_queue_full(self):
        bp = BatchProcessor(max_batch_size=1)
        for i in range(4):
            bp._queue.append(BatchRequest(id=f"q{i}", inputs={"x": np.array([1.0])}))
        assert bp.submit(BatchRequest(id="full", inputs={"x": np.array([1.0])})) is False

    def test_process_batch_empty(self):
        bp = BatchProcessor()
        assert bp.process_batch() == []

    def test_process_batch_with_process_fn(self):
        bp = BatchProcessor(process_fn=lambda inputs: {"out": inputs["x"] * 2})
        bp.submit(BatchRequest(id="r1", inputs={"x": np.array([1.0])}))
        results = bp.process_batch()
        assert len(results) == 1
        np.testing.assert_array_equal(results[0].outputs["out"], np.array([2.0]))
        assert bp.stats()["total_batches"] == 1

    def test_process_batch_without_process_fn(self):
        bp = BatchProcessor()
        inputs = {"x": np.array([1.0])}
        bp.submit(BatchRequest(id="r1", inputs=inputs))
        results = bp.process_batch()
        assert results[0].outputs is inputs

    def test_process_batch_handles_error(self):
        def _boom(inputs):
            raise ValueError("batch boom")

        bp = BatchProcessor(process_fn=_boom)
        bp.submit(BatchRequest(id="r1", inputs={"x": np.array([1.0])}))
        results = bp.process_batch()
        assert results[0].error == "batch boom"
        assert results[0].outputs == {}
        assert bp._total_errors == 1

    def test_process_batch_fires_callback(self):
        received = []
        bp = BatchProcessor(process_fn=lambda inputs: {"out": inputs["x"]})
        bp.submit(BatchRequest(
            id="r1", inputs={"x": np.array([1.0])},
            callback=lambda result: received.append(result),
        ))
        bp.process_batch()
        assert len(received) == 1
        assert received[0].id == "r1"

    def test_process_batch_callback_raises_is_swallowed(self):
        bp = BatchProcessor(process_fn=lambda inputs: {"out": inputs["x"]})
        bp.submit(BatchRequest(
            id="r1", inputs={"x": np.array([1.0])},
            callback=lambda result: (_ for _ in ()).throw(RuntimeError("cb boom")),
        ))
        results = bp.process_batch()
        assert len(results) == 1

    def test_process_batch_splits_large_queue(self):
        bp = BatchProcessor(max_batch_size=2)
        for i in range(4):
            bp.submit(BatchRequest(id=f"r{i}", inputs={"x": np.array([1.0])}))
        results = bp.process_batch()
        assert len(results) == 2
        assert bp.queue_size == 2

    def test_flush(self):
        bp = BatchProcessor(process_fn=lambda inputs: {"out": inputs["x"]})
        for i in range(3):
            bp.submit(BatchRequest(id=f"r{i}", inputs={"x": np.array([1.0])}))
        results = bp.flush()
        assert len(results) == 3
        assert bp.queue_size == 0


class TestNeuralKernelReExport:
    def test_module_getattr_neural_kernel(self):
        from domains.shell.kernel import NeuralKernel as NK
        assert neural.NeuralKernel is NK

    def test_module_getattr_unknown(self):
        with pytest.raises(AttributeError):
            neural.definitely_not_here


class TestSetupGenerateHandlerEmpty:
    def test_generate_missing_model_returns_empty(self):
        k = Kernel()
        k.boot()
        result = k.syscall(NeuralSyscall.GENERATE, "hello", "missing_model")
        assert result.success is True
        assert result.value == {"token_count": 0, "tokens": []}
        k.shutdown()

    def test_generate_with_model_returns_tokens(self):
        k = Kernel()
        k.boot()

        class MockModel:
            def generate_numpy(self, prompt, max_tokens=10, temperature=1.0):
                return [10, 20, 30]

        k.engine.load_model("mock", MockModel())
        result = k.syscall(NeuralSyscall.GENERATE, "hello", "mock", max_tokens=3)
        assert result.success is True
        assert result.value["token_count"] == 3
        k.shutdown()
