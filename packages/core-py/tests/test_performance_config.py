"""Tests for domains.training.performance — TrainingOptimizations, InferenceOptimizations, PerformanceConfig."""

import numpy as np
import pytest
from domains.training.performance import (
    TrainingOptimizations,
    InferenceOptimizations,
    PerformanceConfig,
    FastInferenceSampler,
    PerformanceMonitor,
    CUDAGraphManager,
    _NumpyBatchIterator,
    _collate,
    _pad_last,
    _softmax,
    _as_array,
    effective_dataloader_workers,
    effective_prefetch_factor,
    PreallocatedBatchDataset,
    OptimizedBatchCache,
)


# ── TrainingOptimizations ────────────────────────────────────────────────


class TestTrainingOptimizations:
    def test_defaults(self):
        to = TrainingOptimizations()
        assert to.use_compile is True
        assert to.compile_mode == "reduce-overhead"
        assert to.dataloader_workers == 4
        assert to.use_flash_attention is True
        assert to.gradient_checkpointing is True

    def test_custom(self):
        to = TrainingOptimizations(use_compile=False, dataloader_workers=0)
        assert to.use_compile is False
        assert to.dataloader_workers == 0

    def test_compile_fullgraph_default(self):
        to = TrainingOptimizations()
        assert to.compile_fullgraph is False

    def test_use_cuda_graphs_default(self):
        to = TrainingOptimizations()
        assert to.use_cuda_graphs is False

    def test_channel_last_default(self):
        to = TrainingOptimizations()
        assert to.channel_last is True

    def test_dataloader_prefetch_default(self):
        to = TrainingOptimizations()
        assert to.dataloader_prefetch == 2

    def test_dataloader_persistent_default(self):
        to = TrainingOptimizations()
        assert to.dataloader_persistent is True

    def test_dataloader_pin_memory_default(self):
        to = TrainingOptimizations()
        assert to.dataloader_pin_memory is True

    def test_use_fused_optimizer_default(self):
        to = TrainingOptimizations()
        assert to.use_fused_optimizer is True

    def test_cudnn_benchmark_default(self):
        to = TrainingOptimizations()
        assert to.cudnn_benchmark is True

    def test_cudnn_deterministic_default(self):
        to = TrainingOptimizations()
        assert to.cudnn_deterministic is False

    def test_batch_preallocation_default(self):
        to = TrainingOptimizations()
        assert to.batch_preallocation is True

    def test_all_fields_custom(self):
        to = TrainingOptimizations(
            use_compile=False, compile_mode="max-autotune", compile_fullgraph=True,
            use_cuda_graphs=True, channel_last=False, dataloader_workers=8,
            dataloader_prefetch=4, dataloader_persistent=False, dataloader_pin_memory=False,
            use_fused_optimizer=False, cudnn_benchmark=False, cudnn_deterministic=True,
            use_flash_attention=False, gradient_checkpointing=False, batch_preallocation=False,
        )
        assert to.use_compile is False
        assert to.compile_mode == "max-autotune"
        assert to.compile_fullgraph is True
        assert to.use_cuda_graphs is True
        assert to.channel_last is False
        assert to.dataloader_workers == 8
        assert to.dataloader_prefetch == 4
        assert to.dataloader_persistent is False
        assert to.dataloader_pin_memory is False
        assert to.use_fused_optimizer is False
        assert to.cudnn_benchmark is False
        assert to.cudnn_deterministic is True
        assert to.use_flash_attention is False
        assert to.gradient_checkpointing is False
        assert to.batch_preallocation is False

    def test_repr(self):
        to = TrainingOptimizations()
        r = repr(to)
        assert "TrainingOptimizations" in r

    def test_equality(self):
        to1 = TrainingOptimizations(use_compile=True, dataloader_workers=4)
        to2 = TrainingOptimizations(use_compile=True, dataloader_workers=4)
        assert to1 == to2

    def test_inequality(self):
        to1 = TrainingOptimizations(use_compile=True)
        to2 = TrainingOptimizations(use_compile=False)
        assert to1 != to2


# ── InferenceOptimizations ───────────────────────────────────────────────


class TestInferenceOptimizations:
    def test_defaults(self):
        io = InferenceOptimizations()
        assert io.use_compile is True
        assert io.use_cuda_graphs is True
        assert io.max_batch_size == 32
        assert io.use_kv_cache is True

    def test_custom(self):
        io = InferenceOptimizations(max_batch_size=1, use_kv_cache=False)
        assert io.max_batch_size == 1
        assert io.use_kv_cache is False

    def test_compile_mode_default(self):
        io = InferenceOptimizations()
        assert io.compile_mode == "default"

    def test_channel_last_default(self):
        io = InferenceOptimizations()
        assert io.channel_last is True

    def test_use_flash_attention_default(self):
        io = InferenceOptimizations()
        assert io.use_flash_attention is True

    def test_use_sdpa_default(self):
        io = InferenceOptimizations()
        assert io.use_sdpa is True

    def test_kv_cache_preallocate_default(self):
        io = InferenceOptimizations()
        assert io.kv_cache_preallocate is True

    def test_use_continuous_batching_default(self):
        io = InferenceOptimizations()
        assert io.use_continuous_batching is True

    def test_all_fields_custom(self):
        io = InferenceOptimizations(
            use_compile=False, compile_mode="max-autotune",
            use_cuda_graphs=False, channel_last=False,
            use_flash_attention=False, use_sdpa=False,
            max_batch_size=64, kv_cache_preallocate=False,
            use_kv_cache=False, use_continuous_batching=False,
        )
        assert io.use_compile is False
        assert io.compile_mode == "max-autotune"
        assert io.use_cuda_graphs is False
        assert io.channel_last is False
        assert io.use_flash_attention is False
        assert io.use_sdpa is False
        assert io.max_batch_size == 64
        assert io.kv_cache_preallocate is False
        assert io.use_kv_cache is False
        assert io.use_continuous_batching is False

    def test_repr(self):
        io = InferenceOptimizations()
        r = repr(io)
        assert "InferenceOptimizations" in r

    def test_equality(self):
        io1 = InferenceOptimizations(max_batch_size=32)
        io2 = InferenceOptimizations(max_batch_size=32)
        assert io1 == io2

    def test_inequality(self):
        io1 = InferenceOptimizations(max_batch_size=32)
        io2 = InferenceOptimizations(max_batch_size=64)
        assert io1 != io2


# ── PerformanceConfig ────────────────────────────────────────────────────


class TestPerformanceConfig:
    def test_defaults(self):
        pc = PerformanceConfig()
        assert pc.device in ("cpu", "cuda", "mps")
        assert isinstance(pc.training, TrainingOptimizations)
        assert isinstance(pc.inference, InferenceOptimizations)

    def test_custom_device(self):
        pc = PerformanceConfig(device="cpu")
        assert pc.device == "cpu"

    def test_auto_device(self):
        pc = PerformanceConfig(device="auto")
        assert pc.device in ("cpu", "cuda", "mps")

    def test_training_is_separate_instance(self):
        pc = PerformanceConfig()
        assert isinstance(pc.training, TrainingOptimizations)

    def test_inference_is_separate_instance(self):
        pc = PerformanceConfig()
        assert isinstance(pc.inference, InferenceOptimizations)

    def test_two_configs_independent(self):
        pc1 = PerformanceConfig(device="cpu")
        pc2 = PerformanceConfig(device="cpu")
        assert pc1.training is not pc2.training
        assert pc1.inference is not pc2.inference

    def test_custom_training(self):
        t = TrainingOptimizations(use_compile=False)
        pc = PerformanceConfig(device="cpu", training=t)
        assert pc.training.use_compile is False

    def test_custom_inference(self):
        i = InferenceOptimizations(max_batch_size=1)
        pc = PerformanceConfig(device="cpu", inference=i)
        assert pc.inference.max_batch_size == 1

    def test_repr(self):
        pc = PerformanceConfig(device="cpu")
        r = repr(pc)
        assert "PerformanceConfig" in r

    def test_equality(self):
        pc1 = PerformanceConfig(device="cpu")
        pc2 = PerformanceConfig(device="cpu")
        assert pc1 == pc2

    def test_inequality(self):
        pc1 = PerformanceConfig(device="cpu")
        pc2 = PerformanceConfig(device="cuda")
        assert pc1 != pc2


# ── FastInferenceSampler ─────────────────────────────────────────────────


class TestFastInferenceSampler:
    def test_sample_greedy(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = FastInferenceSampler.sample(logits, temperature=0)
        assert result.shape == (1, 1)
        assert result[0, 0] == 2

    def test_sample_with_temperature(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        np.random.seed(42)
        result = FastInferenceSampler.sample(logits, temperature=1.0)
        assert result.shape == (1, 1)

    def test_sample_top_k(self):
        logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        np.random.seed(42)
        result = FastInferenceSampler.sample(logits, temperature=1.0, top_k=2)
        assert result[0, 0] in [3, 4]

    def test_sample_top_p(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        np.random.seed(42)
        result = FastInferenceSampler.sample(logits, temperature=1.0, top_p=0.5)
        assert result.shape == (1, 1)

    def test_sample_repetition_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        prev = np.array([2])
        result = FastInferenceSampler.sample(
            logits, temperature=1.0, repetition_penalty=1.5, prev_tokens=prev,
        )
        assert result.shape == (1, 1)

    def test_sample_repetition_penalty_no_prev(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = FastInferenceSampler.sample(
            logits, temperature=1.0, repetition_penalty=1.5, prev_tokens=None,
        )
        assert result.shape == (1, 1)

    def test_apply_top_k_small_k(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = FastInferenceSampler._apply_top_k(logits, k=2)
        masked = np.sum(result > -1e9)
        assert masked == 2

    def test_apply_top_k_zero_k(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = FastInferenceSampler._apply_top_k(logits, k=0)
        np.testing.assert_array_equal(result, logits)

    def test_apply_top_p_basic(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = FastInferenceSampler._apply_top_p(logits, p=0.9)
        assert result.shape == (3,)

    def test_apply_repetition_penalty_vectorized(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        prev = np.array([0, 1])
        result = FastInferenceSampler._apply_repetition_penalty_vectorized(
            logits, prev, penalty=2.0,
        )
        assert result.shape == logits.shape


# ── PerformanceMonitor ───────────────────────────────────────────────────


class TestPerformanceMonitor:
    def test_init(self):
        pm = PerformanceMonitor()
        assert pm.window_size == 100
        assert pm.step_times == []
        assert pm.losses == []
        assert pm.tokens_processed == 0

    def test_record_step(self):
        pm = PerformanceMonitor()
        pm.record_step(loss=0.5, step_time=0.01, batch_size=8, seq_len=128)
        assert len(pm.step_times) == 1
        assert pm.losses[0] == 0.5
        assert pm.tokens_processed == 8 * 128

    def test_window_limit(self):
        pm = PerformanceMonitor(window_size=3)
        for i in range(5):
            pm.record_step(loss=float(i), step_time=0.01, batch_size=1, seq_len=1)
        assert len(pm.step_times) == 3

    def test_get_stats_empty(self):
        pm = PerformanceMonitor()
        assert pm.get_stats() == {}

    def test_get_stats_with_data(self):
        pm = PerformanceMonitor()
        pm.record_step(loss=0.5, step_time=0.01, batch_size=8, seq_len=128)
        stats = pm.get_stats()
        assert "avg_step_time_ms" in stats
        assert "tokens_per_sec" in stats
        assert "avg_loss" in stats
        assert "total_steps" in stats

    def test_custom_window(self):
        pm = PerformanceMonitor(window_size=50)
        assert pm.window_size == 50


# ── CUDAGraphManager ────────────────────────────────────────────────────


class TestCUDAGraphManager:
    def test_capture_returns_false(self):
        model = lambda x: x
        config = InferenceOptimizations()
        mgr = CUDAGraphManager(model, config)
        assert mgr.capture(1, 128, 256) is False

    def test_replay_calls_model(self):
        model = lambda x: np.zeros((1, 10))
        config = InferenceOptimizations()
        mgr = CUDAGraphManager(model, config)
        result = mgr.replay(np.array([[1, 2, 3]]))
        assert result.shape == (1, 10)


# ── _NumpyBatchIterator ─────────────────────────────────────────────────


class TestNumpyBatchIterator:
    def test_basic_iteration(self):
        data = [(np.array([1, 2]), np.array([3, 4]))]
        it = _NumpyBatchIterator(data, batch_size=1)
        batch = next(iter(it))
        assert batch is not None

    def test_batch_size(self):
        data = [(np.array([i]), np.array([i])) for i in range(10)]
        it = _NumpyBatchIterator(data, batch_size=5)
        batches = list(it)
        assert len(batches) == 2

    def test_len(self):
        data = [(np.array([i]), np.array([i])) for i in range(10)]
        it = _NumpyBatchIterator(data, batch_size=3)
        assert len(it) == 4

    def test_shuffle(self):
        data = [(np.array([i]), np.array([i])) for i in range(20)]
        it = _NumpyBatchIterator(data, batch_size=5, shuffle=True)
        batches = list(it)
        assert len(batches) == 4

    def test_no_shuffle(self):
        data = [(np.array([i]), np.array([i])) for i in range(10)]
        it = _NumpyBatchIterator(data, batch_size=5, shuffle=False)
        batches = list(it)
        assert len(batches) == 2


# ── _collate & _pad_last ─────────────────────────────────────────────────


class TestCollateAndPad:
    def test_collate_pairs(self):
        batch = [(np.array([1, 2]), np.array([3, 4]))]
        x, y = _collate(batch)
        assert x.shape == (1, 2)
        assert y.shape == (1, 2)

    def test_collate_padding(self):
        batch = [
            (np.array([1, 2]), np.array([3, 4])),
            (np.array([5]), np.array([6])),
        ]
        x, y = _collate(batch)
        assert x.shape == (1, 2) or x.shape == (2, 2)

    def test_pad_last_no_pad(self):
        a = np.array([1, 2, 3])
        result = _pad_last(a, 3)
        np.testing.assert_array_equal(result, a)

    def test_pad_last_with_pad(self):
        a = np.array([1, 2])
        result = _pad_last(a, 4)
        assert result.shape == (4,)
        assert result[2] == 0

    def test_collate_single_arrays(self):
        batch = [np.array([1, 2]), np.array([3, 4])]
        result = _collate(batch)
        assert result.shape == (2, 2)


# ── _softmax ─────────────────────────────────────────────────────────────


class TestSoftmax:
    def test_basic(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = _softmax(logits)
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_equal_logits(self):
        logits = np.array([1.0, 1.0, 1.0])
        probs = _softmax(logits)
        np.testing.assert_array_almost_equal(probs, [1 / 3, 1 / 3, 1 / 3])

    def test_numerical_stability(self):
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = _softmax(logits)
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_batch(self):
        logits = np.array([[1.0, 2.0], [3.0, 4.0]])
        probs = _softmax(logits)
        assert abs(np.sum(probs[0]) - 1.0) < 1e-6
        assert abs(np.sum(probs[1]) - 1.0) < 1e-6


# ── _as_array ────────────────────────────────────────────────────────────


class TestAsArray:
    def test_numpy(self):
        arr = np.array([1.0, 2.0])
        result = _as_array(arr)
        assert isinstance(result, np.ndarray)

    def test_list(self):
        result = _as_array([1.0, 2.0])
        assert isinstance(result, np.ndarray)

    def test_int(self):
        result = _as_array(5)
        assert result == 5


# ── effective_dataloader_workers ─────────────────────────────────────────


class TestEffectiveDataloaderWorkers:
    def test_positive(self):
        assert effective_dataloader_workers(4) >= 0

    def test_zero(self):
        assert effective_dataloader_workers(0) == 0

    def test_negative(self):
        assert effective_dataloader_workers(-1) == 0

    def test_string(self):
        assert effective_dataloader_workers("abc") == 0

    def test_float(self):
        assert effective_dataloader_workers(2.5) >= 0


# ── effective_prefetch_factor ────────────────────────────────────────────


class TestEffectivePrefetchFactor:
    def test_normal(self):
        assert effective_prefetch_factor(4, 2) == 2

    def test_zero_workers(self):
        assert effective_prefetch_factor(0, 2) is None

    def test_negative_workers(self):
        assert effective_prefetch_factor(-1, 2) is None

    def test_zero_prefetch(self):
        result = effective_prefetch_factor(4, 0)
        assert result >= 1

    def test_string_prefetch(self):
        assert effective_prefetch_factor(4, "abc") == 2


# ── PreallocatedBatchDataset ─────────────────────────────────────────────


class TestPreallocatedBatchDataset:
    def test_len(self):
        data = np.arange(100)
        ds = PreallocatedBatchDataset(data, block_size=10, batch_size=8)
        assert len(ds) == 89  # 100 - (10 + 1)

    def test_getitem(self):
        data = np.arange(100)
        ds = PreallocatedBatchDataset(data, block_size=10, batch_size=8)
        x, y = ds[0]
        assert x.shape == (10,)
        assert y.shape == (10,)
        np.testing.assert_array_equal(x, data[:10])
        np.testing.assert_array_equal(y, data[1:11])

    def test_getitem_boundary(self):
        data = np.arange(20)
        ds = PreallocatedBatchDataset(data, block_size=5, batch_size=2)
        x, y = ds[13]  # last valid index is len(data) - seq_len - 1 = 20 - 6 - 1 = 13
        assert x.shape == (5,)

    def test_out_of_range(self):
        data = np.arange(10)
        ds = PreallocatedBatchDataset(data, block_size=5, batch_size=2)
        with pytest.raises(IndexError):
            ds[100]


# ── OptimizedBatchCache ──────────────────────────────────────────────────


class TestOptimizedBatchCache:
    def test_allocate(self):
        cache = OptimizedBatchCache(device="cpu")
        x, y = cache.allocate(batch_size=4, block_size=16)
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)

    def test_allocate_same_shape(self):
        cache = OptimizedBatchCache(device="cpu")
        x1, y1 = cache.allocate(batch_size=4, block_size=16)
        x2, y2 = cache.allocate(batch_size=4, block_size=16)
        assert x1 is x2
        assert y1 is y2

    def test_allocate_different_shape(self):
        cache = OptimizedBatchCache(device="cpu")
        x1, y1 = cache.allocate(batch_size=4, block_size=16)
        x2, y2 = cache.allocate(batch_size=8, block_size=32)
        assert x1 is not x2

    def test_fill(self):
        cache = OptimizedBatchCache(device="cpu")
        data = np.arange(100)
        indices = np.array([0, 10, 20])
        x, y = cache.fill(batch_size=3, block_size=5, data=data, indices=indices)
        assert x.shape == (3, 5)
        np.testing.assert_array_equal(x[0], data[0:5])
        np.testing.assert_array_equal(y[0], data[1:6])
