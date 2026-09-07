"""Tests for training.performance — performance optimization classes."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.performance import (
    get_optimal_device,
    get_device_name,
    setup_device_environment,
    TrainingOptimizations,
    InferenceOptimizations,
    PerformanceConfig,
    CUDAGraphManager,
    FastInferenceSampler,
)


# ── Device detection ────────────────────────────────────────────────────────


class TestDeviceDetection:

    def test_get_optimal_device(self):
        device = get_optimal_device()
        assert device in ("cpu", "cuda", "mps")

    def test_get_device_name(self):
        name = get_device_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_setup_device_environment(self):
        result = setup_device_environment()
        assert result is None


# ── TrainingOptimizations ──────────────────────────────────────────────────


class TestTrainingOptimizations:

    def test_default(self):
        opts = TrainingOptimizations()
        assert opts.use_compile is True
        assert opts.gradient_checkpointing is True
        assert opts.dataloader_workers == 4

    def test_to_dict(self):
        opts = TrainingOptimizations()
        d = opts.to_dict()
        assert "use_compile" in d
        assert "gradient_checkpointing" in d


# ── InferenceOptimizations ─────────────────────────────────────────────────


class TestInferenceOptimizations:

    def test_default(self):
        opts = InferenceOptimizations()
        assert opts.use_kv_cache is True
        assert opts.max_batch_size == 32

    def test_to_dict(self):
        opts = InferenceOptimizations()
        d = opts.to_dict()
        assert "use_kv_cache" in d
        assert "max_batch_size" in d


# ── PerformanceConfig ──────────────────────────────────────────────────────


class TestPerformanceConfig:

    def test_default(self):
        config = PerformanceConfig()
        assert config.device in ("cpu", "cuda", "mps")

    def test_to_dict(self):
        config = PerformanceConfig()
        d = config.to_dict()
        assert "device" in d
        assert "training" in d
        assert "inference" in d


# ── CUDAGraphManager ──────────────────────────────────────────────────────


class TestCUDAGraphManager:

    def test_init(self):
        model = lambda x: x
        manager = CUDAGraphManager(model)
        assert manager.is_captured is False

    def test_capture(self):
        model = lambda x: x
        manager = CUDAGraphManager(model)
        result = manager.capture()
        assert result is False

    def test_replay(self):
        model = lambda x: np.array([1.0, 2.0])
        manager = CUDAGraphManager(model)
        result = manager.replay(np.array([0]))
        assert np.allclose(result, [1.0, 2.0])


# ── FastInferenceSampler ──────────────────────────────────────────────────


class TestFastInferenceSampler:

    def test_sample_basic(self):
        logits = np.random.randn(1, 10)
        result = FastInferenceSampler.sample(logits, temperature=1.0)
        assert result.shape[0] == 1

    def test_sample_top_k(self):
        logits = np.random.randn(1, 10)
        result = FastInferenceSampler.sample(logits, temperature=1.0, top_k=5)
        assert result.shape[0] == 1

    def test_sample_top_p(self):
        logits = np.random.randn(1, 10)
        result = FastInferenceSampler.sample(logits, temperature=1.0, top_p=0.9)
        assert result.shape[0] == 1

    def test_sample_repetition_penalty(self):
        logits = np.random.randn(1, 10)
        result = FastInferenceSampler.sample(logits, temperature=1.0, repetition_penalty=1.2)
        assert result.shape[0] == 1

    def test_sample_with_prev_tokens(self):
        logits = np.random.randn(1, 10)
        prev = np.array([1, 2, 3])
        result = FastInferenceSampler.sample(logits, temperature=1.0, prev_tokens=prev)
        assert result.shape[0] == 1
