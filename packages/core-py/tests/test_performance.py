"""Tests for domains/training/performance.py."""

import numpy as np
import pytest

from domains.training.performance import (
    CUDAGraphManager,
    FastInferenceSampler,
    InferenceOptimizations,
    OptimizedBatchCache,
    OptimizedDataLoader,
    OptimizedInferenceEngine,
    PerformanceConfig,
    PerformanceMonitor,
    PreallocatedBatchDataset,
    TrainingOptimizations,
    _as_array,
    _clip_grad_norm_,
    _collate,
    _pad_last,
    _softmax,
    _NumpyBatchIterator,
    benchmark_inference,
    benchmark_training,
    effective_dataloader_workers,
    effective_prefetch_factor,
    get_device_name,
    get_optimal_device,
    optimize_model_for_inference,
    setup_device_environment,
)
from domains.training.slonet import SloTransformer, tensor


def _tiny_model():
    return SloTransformer(
        vocab_size=64,
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=16,
        dropout=0.0,
        max_seq_len=32,
        tie_weights=True,
        norm_type="rms_norm",
    )


class TestConfigs:
    def test_training_defaults(self):
        c = TrainingOptimizations()
        assert c.use_compile is True
        assert c.compile_mode == "reduce-overhead"
        assert c.use_cuda_graphs is False
        assert c.channel_last is True
        assert c.dataloader_workers == 4
        assert c.gradient_checkpointing is True

    def test_inference_defaults(self):
        c = InferenceOptimizations()
        assert c.use_compile is True
        assert c.compile_mode == "default"
        assert c.max_batch_size == 32
        assert c.use_kv_cache is True

    def test_performance_config_device_auto(self):
        c = PerformanceConfig()
        assert c.device in ("cpu", "cuda", "mps")
        assert isinstance(c.training, TrainingOptimizations)
        assert isinstance(c.inference, InferenceOptimizations)

    def test_performance_config_explicit_device(self):
        c = PerformanceConfig(device="cuda")
        assert c.device == "cuda"

    def test_override(self):
        c = PerformanceConfig(device="mps", training=TrainingOptimizations(use_compile=False))
        assert c.device == "mps"
        assert c.training.use_compile is False


class TestDeviceDetection:
    def test_get_optimal_device_no_torch(self):
        device = get_optimal_device()
        assert device == "cpu"

    def test_get_device_name_no_torch(self):
        assert get_device_name() == "CPU"

    def test_setup_device_environment_no_torch(self):
        assert setup_device_environment() is None


class TestCUDAGraphManager:
    def test_disabled_without_torch(self):
        model = _tiny_model()
        mgr = CUDAGraphManager(model, InferenceOptimizations(use_cuda_graphs=True))
        assert mgr._enabled is False

    def test_capture_returns_false(self):
        mgr = CUDAGraphManager(_tiny_model(), InferenceOptimizations())
        assert mgr.capture(2, 8, 64) is False

    def test_replay_falls_back_to_forward(self):
        model = _tiny_model()
        mgr = CUDAGraphManager(model, InferenceOptimizations())
        x = np.zeros((1, 8), dtype=np.int64)
        out = mgr.replay(x)
        assert out is not None


class TestNumpyBatchIterator:
    def test_iteration_order_shuffled(self):
        ds = list(range(10))
        it = _NumpyBatchIterator(ds, 4, shuffle=True)
        seen = []
        for batch in it:
            for item in batch:
                seen.append(item)
        assert sorted(seen) == list(range(10))

    def test_no_shuffle_preserves_order(self):
        it = _NumpyBatchIterator(list(range(10)), 4, shuffle=False)
        first = next(iter(it))
        assert list(first) == [0, 1, 2, 3]

    def test_len(self):
        it = _NumpyBatchIterator(list(range(10)), 4, shuffle=False)
        assert len(it) == 3

    def test_len_empty_dataset(self):
        it = _NumpyBatchIterator([], 4, shuffle=False)
        assert len(it) == 1

    def test_stop_iteration_at_end(self):
        it = _NumpyBatchIterator(list(range(3)), 4, shuffle=False)
        batch = next(it)
        assert len(batch) == 3
        with pytest.raises(StopIteration):
            next(it)


class TestCollate:
    def test_empty_batch(self):
        assert _collate([]) is None

    def test_pair_list_padded(self):
        x, y = _collate([
            (np.array([1, 2]), np.array([3])),
            (np.array([4, 5, 6]), np.array([7, 8])),
        ])
        assert x.shape == (2, 3)
        assert y.shape == (2, 2)
        assert x[1, 2] == 6
        assert y[1, 1] == 8

    def test_plain_arrays_stacked(self):
        out = _collate([np.array([1, 2]), np.array([3, 4])])
        assert out.shape == (2, 2)

    def test_pad_last_noop(self):
        a = np.array([[1, 2], [3, 4]])
        out = _pad_last(a, 2)
        assert out is a

    def test_pad_last_pads(self):
        a = np.array([[1, 2]])
        out = _pad_last(a, 4)
        assert out.shape == (1, 4)
        assert out[0, 2] == 0


class TestOptimizedDataLoader:
    def test_worker_helpers(self):
        assert effective_dataloader_workers(4) == 4
        assert effective_dataloader_workers("bad") == 0
        assert effective_prefetch_factor(0, 2) is None
        assert effective_prefetch_factor(4, "bad") == 2
        assert effective_prefetch_factor(4, 1) == 1

    def test_iteration_never_terminates(self):
        loader = OptimizedDataLoader(list(range(5)), 2)
        it = iter(loader)
        epoch1 = []
        for _ in range(3):
            epoch1.extend(next(it).tolist())
        assert sorted(epoch1) == list(range(5))
        epoch2 = []
        for _ in range(3):
            epoch2.extend(next(it).tolist())
        assert sorted(epoch2) == list(range(5))

    def test_prefetch_and_get_batch(self):
        loader = OptimizedDataLoader(list(range(6)), 3)
        loader.prefetch()
        batch1 = loader.get_batch()
        assert len(batch1) == 3
        loader.prefetch()
        batch2 = loader.get_batch()
        assert len(batch2) == 3

    def test_get_batch_wraps_at_end(self):
        loader = OptimizedDataLoader(list(range(3)), 4)
        b1 = loader.get_batch()
        assert len(b1) == 3
        b2 = loader.get_batch()
        assert len(b2) == 3

    def test_default_workers_uses_resource_manager(self):
        loader = OptimizedDataLoader(list(range(4)), 2)
        assert loader.dataloader is not None

    def test_next_restarts(self):
        loader = OptimizedDataLoader(list(range(3)), 4)
        b = next(iter(loader))
        assert len(b) == 3
        b2 = next(iter(loader))
        assert len(b2) == 3


class TestPreallocatedDataset:
    def test_len(self):
        ds = PreallocatedBatchDataset(np.arange(20), 4, 2)
        assert len(ds) == 20 - 5

    def test_getitem(self):
        ds = PreallocatedBatchDataset(np.arange(20), 4, 2)
        x, y = ds[0]
        assert list(x) == [0, 1, 2, 3]
        assert list(y) == [1, 2, 3, 4]

    def test_getitem_out_of_range(self):
        ds = PreallocatedBatchDataset(np.arange(20), 4, 2)
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_getitem_negative(self):
        ds = PreallocatedBatchDataset(np.arange(20), 4, 2)
        with pytest.raises(IndexError):
            ds[-1]


class TestOptimizedBatchCache:
    def test_allocate_reuses_cache(self):
        cache = OptimizedBatchCache("cpu")
        x1, y1 = cache.allocate(4, 8)
        x2, y2 = cache.allocate(4, 8)
        assert x1 is x2
        assert y1 is y2

    def test_allocate_reallocates_on_shape_change(self):
        cache = OptimizedBatchCache("cpu")
        x1, _ = cache.allocate(4, 8)
        x2, _ = cache.allocate(4, 16)
        assert x1 is not x2
        assert x2.shape == (4, 16)

    def test_fill(self):
        cache = OptimizedBatchCache("cpu")
        data = np.arange(20)
        x, y = cache.fill(2, 3, data, np.array([0, 5]))
        assert list(x[0]) == [0, 1, 2]
        assert list(y[0]) == [1, 2, 3]
        assert list(x[1]) == [5, 6, 7]
        assert list(y[1]) == [6, 7, 8]


class TestArrayHelpers:
    def test_as_array_unwraps_tensor(self):
        t = tensor(np.array([1.0, 2.0]), requires_grad=True)
        out = _as_array(t)
        assert isinstance(out, np.ndarray)
        assert out.dtype != object

    def test_as_array_plain(self):
        arr = np.array([1, 2])
        assert _as_array(arr) is not None

    def test_as_array_dtype(self):
        out = _as_array(np.array([1.0]), dtype=np.int64)
        assert out.dtype == np.int64

    def test_softmax_sums_to_one(self):
        out = _softmax(np.array([[1.0, 2.0, 3.0]]))
        assert np.allclose(out.sum(axis=-1), 1.0)

    def test_softmax_stable(self):
        out = _softmax(np.array([[1000.0, 1001.0]]))
        assert np.all(np.isfinite(out))


class TestFastInferenceSampler:
    def test_temperature_zero_argmax(self):
        logits = np.array([[0.1, 0.9, 0.2]])
        out = FastInferenceSampler.sample(logits, temperature=0.0)
        assert out[0, 0] == 1

    def test_sample_shape(self):
        logits = np.random.RandomState(0).randn(2, 8)
        out = FastInferenceSampler.sample(logits, temperature=1.0)
        assert out.shape == (2, 1)

    def test_top_k(self):
        logits = np.random.RandomState(0).randn(2, 10)
        out = FastInferenceSampler.sample(logits, temperature=1.0, top_k=3)
        assert out.shape == (2, 1)

    def test_top_p(self):
        logits = np.random.RandomState(0).randn(2, 10)
        out = FastInferenceSampler.sample(logits, temperature=1.0, top_p=0.9)
        assert out.shape == (2, 1)

    def test_repetition_penalty(self):
        logits = np.random.RandomState(0).randn(2, 10)
        out = FastInferenceSampler.sample(
            logits, temperature=1.0, top_k=0, top_p=1.0,
            repetition_penalty=1.2, prev_tokens=np.array([[1, 2, 3]]),
        )
        assert out.shape == (2, 1)

    def test_repetition_penalty_no_prev(self):
        logits = np.random.RandomState(0).randn(2, 10)
        out = FastInferenceSampler.sample(logits, temperature=1.0, repetition_penalty=1.2)
        assert out.shape == (2, 1)

    def test_repetition_penalty_vectorized_empty(self):
        logits = np.random.RandomState(0).randn(1, 6)
        out = FastInferenceSampler._apply_repetition_penalty_vectorized(logits, np.array([]), 1.2)
        assert np.allclose(out, logits)

    def test_repetition_penalty_vectorized_applies(self):
        logits = np.array([[1.0, -1.0, 2.0, 0.5]])
        out = FastInferenceSampler._apply_repetition_penalty_vectorized(logits, np.array([0, 1]), 2.0)
        assert out[0, 0] == pytest.approx(2.0)
        assert out[0, 1] == pytest.approx(-0.5)
        assert out[0, 2] == pytest.approx(2.0)
        assert out[0, 3] == pytest.approx(0.5)

    def test_apply_top_k_1d(self):
        logits = np.array([0.1, 0.9, 0.5, 0.2])
        out = FastInferenceSampler._apply_top_k(logits, 2)
        assert out.shape == (4,)
        assert np.isinf(out).sum() == 2

    def test_apply_top_k_2d(self):
        logits = np.random.RandomState(0).randn(2, 6)
        out = FastInferenceSampler._apply_top_k(logits, 3)
        assert out.shape == (2, 6)
        assert np.isinf(out).sum() == 6

    def test_apply_top_k_full_vocab(self):
        logits = np.random.RandomState(0).randn(2, 5)
        out = FastInferenceSampler._apply_top_k(logits, 100)
        assert np.allclose(out, logits)

    def test_apply_top_k_zero_or_negative(self):
        logits = np.random.RandomState(0).randn(2, 6)
        out = FastInferenceSampler._apply_top_k(logits, 0)
        assert np.allclose(out, logits)
        out2 = FastInferenceSampler._apply_top_k(logits, -3)
        assert np.allclose(out2, logits)

    def test_apply_top_p_1d(self):
        logits = np.array([0.1, 0.9, 0.5, 0.2])
        out = FastInferenceSampler._apply_top_p(logits, 0.9)
        assert out.shape == (4,)

    def test_apply_top_p_2d(self):
        logits = np.random.RandomState(0).randn(2, 6)
        out = FastInferenceSampler._apply_top_p(logits, 0.9)
        assert out.shape == (2, 6)


class TestOptimizedInferenceEngine:
    def test_init_calls_eval(self):
        model = _tiny_model()
        engine = OptimizedInferenceEngine(model)
        assert engine.cuda_graph_manager is None

    def test_generate_greedy(self):
        model = _tiny_model()
        engine = OptimizedInferenceEngine(model)
        x = np.random.default_rng(0).integers(0, 64, size=(1, 8))
        out = engine.generate(x, max_new_tokens=3, temperature=0.0)
        assert out.shape[0] == 1
        assert out.shape[1] >= 8

    def test_generate_sampling(self):
        model = _tiny_model()
        engine = OptimizedInferenceEngine(model)
        x = np.random.default_rng(0).integers(0, 64, size=(1, 8))
        out = engine.generate(x, max_new_tokens=3, temperature=0.9)
        assert out.shape[1] >= 8

    def test_generate_1d_input(self):
        model = _tiny_model()
        engine = OptimizedInferenceEngine(model)
        x = np.random.default_rng(0).integers(0, 64, size=(8,))
        out = engine.generate(x, max_new_tokens=3, temperature=0.0)
        assert out.shape[1] >= 8


class TestPerformanceMonitor:
    def test_empty_stats(self):
        assert PerformanceMonitor().get_stats() == {}

    def test_record_and_stats(self):
        mon = PerformanceMonitor(window_size=10)
        mon.record_step(1.0, 0.01, 8, 32)
        stats = mon.get_stats()
        assert stats["avg_step_time_ms"] == 10.0
        assert stats["steps_per_sec"] == 100.0
        assert stats["avg_loss"] == 1.0
        assert stats["total_steps"] == 1
        assert stats["tokens_per_sec"] > 0

    def test_window_trimming(self):
        mon = PerformanceMonitor(window_size=3)
        for i in range(10):
            mon.record_step(float(i), 0.01, 1, 1)
        assert len(mon.step_times) == 3
        assert mon.get_stats()["total_steps"] == 3

    def test_avg_loss_zero_when_no_losses(self):
        mon = PerformanceMonitor()
        mon.step_times.append(0.01)
        assert mon.get_stats()["avg_loss"] == 0


class TestOptimizeModel:
    def test_optimize_model_for_inference(self):
        model = _tiny_model()
        out = optimize_model_for_inference(model, device="cpu")
        assert out is model


class TestClipGradNorm:
    def test_no_grads_returns_zero(self):
        model = _tiny_model()
        assert _clip_grad_norm_(model, 1.0) == 0.0

    def test_clips_grads(self):
        model = _tiny_model()
        rng = np.random.default_rng(0)
        x = rng.integers(0, 64, size=(2, 8))
        logits, loss = model(x, x)
        loss.backward()
        norm = _clip_grad_norm_(model, 1e-6)
        assert norm > 0

    def test_clips_to_max_norm(self):
        model = _tiny_model()
        rng = np.random.default_rng(0)
        x = rng.integers(0, 64, size=(2, 8))
        logits, loss = model(x, x)
        loss.backward()
        _clip_grad_norm_(model, 1e-6)
        total_sq = 0.0
        for p in model.parameters():
            g = getattr(p, "grad", None)
            if g is None:
                continue
            arr = _as_array(g).reshape(-1)
            total_sq += float(np.dot(arr, arr))
        assert float(np.sqrt(total_sq)) <= 1e-6 * 1.0001


class TestBenchmarks:
    def test_benchmark_training(self):
        model = _tiny_model()
        result = benchmark_training(model, batch_size=2, seq_len=8, num_steps=2)
        assert set(result) >= {"elapsed_sec", "tokens_per_sec", "steps_per_sec", "device"}
        assert result["elapsed_sec"] > 0

    def test_benchmark_inference(self):
        model = _tiny_model()
        result = benchmark_inference(model, batch_size=1, seq_len=8, gen_len=3, num_runs=2)
        assert set(result) >= {"avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "tokens_per_sec", "device"}
        assert result["avg_latency_ms"] > 0
