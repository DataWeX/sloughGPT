"""Tests for LR scheduler configs."""
from __future__ import annotations

from domains.training.lr_schedulers import (
    BEST_PRACTICES,
    CyclicConfig,
    CosineAnnealingConfig,
    OneCycleConfig,
    SchedulerConfig,
    WarmupConfig,
)


class TestSchedulerConfig:
    def test_defaults(self):
        c = SchedulerConfig()
        assert c.name == "none"
        assert c.initial_lr == 1e-4


class TestCosineAnnealingConfig:
    def test_defaults(self):
        c = CosineAnnealingConfig()
        assert c.name == "cosine"
        assert c.min_lr == 1e-6
        assert c.warmup_steps == 0
        assert c.total_steps == 10000

    def test_override(self):
        c = CosineAnnealingConfig(total_steps=5000, min_lr=1e-5)
        assert c.total_steps == 5000
        assert c.min_lr == 1e-5


class TestWarmupConfig:
    def test_defaults(self):
        c = WarmupConfig()
        assert c.name == "warmup"
        assert c.warmup_steps == 500
        assert c.warmup_start_lr == 1e-7


class TestOneCycleConfig:
    def test_defaults(self):
        c = OneCycleConfig()
        assert c.name == "onecycle"
        assert c.max_lr == 1e-3
        assert c.pct_start == 0.1
        assert c.anneal_strategy == "cos"


class TestCyclicConfig:
    def test_defaults(self):
        c = CyclicConfig()
        assert c.name == "cyclic"
        assert c.base_lr == 1e-5
        assert c.max_lr == 1e-3
        assert c.mode == "triangular2"


class TestBestPractices:
    def test_exists(self):
        assert isinstance(BEST_PRACTICES, str)
        assert "WARMUP" in BEST_PRACTICES
