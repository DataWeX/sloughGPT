"""Tests for LR schedulers — config dataclasses, factory, scheduler creation."""

import pytest
from domains.training.lr_schedulers import (
    SchedulerConfig, CosineAnnealingConfig, WarmupConfig,
    OneCycleConfig, CyclicConfig, WarmupCosineScheduler,
    PolynomialDecayScheduler, LinearWarmupScheduler,
    create_scheduler, BEST_PRACTICES,
)
from domains.training.slonet import SloLRScheduler


# ── Mock Optimizer ─────────────────────────────────────────────────────────

class MockOptimizer:
    """Minimal optimizer mock that provides param_groups for scheduler init."""
    def __init__(self, lr=1e-3):
        self.param_groups = [{"lr": lr}]


# ── Config Dataclasses ─────────────────────────────────────────────────────

class TestSchedulerConfig:

    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.name == "none"
        assert cfg.initial_lr == 1e-4

    def test_custom(self):
        cfg = SchedulerConfig(name="custom", initial_lr=0.01)
        assert cfg.name == "custom"
        assert cfg.initial_lr == 0.01


class TestCosineAnnealingConfig:

    def test_defaults(self):
        cfg = CosineAnnealingConfig()
        assert cfg.name == "cosine"
        assert cfg.min_lr == 1e-6
        assert cfg.warmup_steps == 0
        assert cfg.total_steps == 10000
        assert cfg.num_cycles == 0.5
        assert cfg.initial_lr == 1e-4

    def test_custom(self):
        cfg = CosineAnnealingConfig(total_steps=5000, warmup_steps=100)
        assert cfg.total_steps == 5000
        assert cfg.warmup_steps == 100


class TestWarmupConfig:

    def test_defaults(self):
        cfg = WarmupConfig()
        assert cfg.name == "warmup"
        assert cfg.warmup_steps == 500
        assert cfg.warmup_start_lr == 1e-7

    def test_custom(self):
        cfg = WarmupConfig(warmup_steps=100, warmup_start_lr=1e-6)
        assert cfg.warmup_steps == 100
        assert cfg.warmup_start_lr == 1e-6


class TestOneCycleConfig:

    def test_defaults(self):
        cfg = OneCycleConfig()
        assert cfg.name == "onecycle"
        assert cfg.max_lr == 1e-3
        assert cfg.pct_start == 0.1
        assert cfg.anneal_strategy == "cos"


class TestCyclicConfig:

    def test_defaults(self):
        cfg = CyclicConfig()
        assert cfg.name == "cyclic"
        assert cfg.base_lr == 1e-5
        assert cfg.max_lr == 1e-3
        assert cfg.step_size_up == 1000
        assert cfg.step_size_down is None
        assert cfg.mode == "triangular2"


# ── Scheduler Classes (wrappers) ──────────────────────────────────────────

class TestSchedulerClasses:

    def test_warmup_cosine_is_scheduler(self):
        assert issubclass(WarmupCosineScheduler, SloLRScheduler)

    def test_polynomial_decay_is_scheduler(self):
        assert issubclass(PolynomialDecayScheduler, SloLRScheduler)

    def test_linear_warmup_is_scheduler(self):
        assert issubclass(LinearWarmupScheduler, SloLRScheduler)

    def test_warmup_cosine_instantiate(self):
        sched = WarmupCosineScheduler(MockOptimizer(), warmup_steps=10, total_steps=100)
        assert sched is not None

    def test_polynomial_decay_instantiate(self):
        sched = PolynomialDecayScheduler(MockOptimizer(), total_steps=100, min_lr=1e-6)
        assert sched is not None

    def test_linear_warmup_instantiate(self):
        sched = LinearWarmupScheduler(MockOptimizer(), warmup_steps=10)
        assert sched is not None


# ── create_scheduler factory ───────────────────────────────────────────────

class TestCreateScheduler:

    def test_cosine(self):
        sched = create_scheduler(MockOptimizer(), "cosine", total_steps=1000, warmup_steps=10)
        assert sched is not None
        assert hasattr(sched, "step")

    def test_warmup(self):
        sched = create_scheduler(MockOptimizer(), "warmup", warmup_steps=50)
        assert sched is not None

    def test_polynomial(self):
        sched = create_scheduler(MockOptimizer(), "polynomial", total_steps=500)
        assert sched is not None

    def test_none_scheduler(self):
        sched = create_scheduler(MockOptimizer(), "none")
        assert sched is not None

    def test_scheduler_has_step(self):
        sched = create_scheduler(MockOptimizer(), "cosine", total_steps=100)
        assert callable(getattr(sched, "step", None))

    def test_scheduler_has_get_lr(self):
        sched = create_scheduler(MockOptimizer(), "cosine", total_steps=100)
        lr = sched.get_lr()
        assert isinstance(lr, list)
        assert len(lr) > 0
        assert all(isinstance(v, float) for v in lr)

    def test_cosine_lr_decreases(self):
        sched = create_scheduler(MockOptimizer(lr=1e-3), "cosine", total_steps=100, warmup_steps=0)
        lr_initial = sched.get_lr()[0]
        for _ in range(50):
            sched.step()
        lr_later = sched.get_lr()[0]
        assert lr_later <= lr_initial

    def test_warmup_lr_starts_low(self):
        sched = create_scheduler(MockOptimizer(lr=1e-3), "warmup", warmup_steps=10)
        lr_initial = sched.get_lr()[0]
        for _ in range(5):
            sched.step()
        lr_after = sched.get_lr()[0]
        assert lr_after >= lr_initial


# ── BEST_PRACTICES ─────────────────────────────────────────────────────────

class TestBestPractices:

    def test_is_string(self):
        assert isinstance(BEST_PRACTICES, str)

    def test_mentions_warmup(self):
        assert "WARMUP" in BEST_PRACTICES

    def test_mentions_cosine(self):
        assert "COSINE" in BEST_PRACTICES

    def test_mentions_onecycle(self):
        assert "ONECYCLE" in BEST_PRACTICES


class TestSchedulerConfigInheritance:
    def test_cosine_config_inherits_scheduler_config(self):
        cfg = CosineAnnealingConfig()
        assert hasattr(cfg, "initial_lr")  # inherited from SchedulerConfig

    def test_onecycle_config_inherits(self):
        cfg = OneCycleConfig()
        assert cfg.name == "onecycle"
        assert hasattr(cfg, "initial_lr")

    def test_cyclic_config_inherits(self):
        cfg = CyclicConfig()
        assert cfg.name == "cyclic"
        assert cfg.base_lr == 1e-5

    def test_warmup_config_inherits(self):
        cfg = WarmupConfig()
        assert cfg.name == "warmup"
        assert hasattr(cfg, "initial_lr")


class TestCreateSchedulerEdgeCases:
    def test_unknown_type_raises_value_error(self):
        """Unknown scheduler type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler(MockOptimizer(), "unknown_type_xyz")

    def test_onecycle_type(self):
        """onecycle type should create a valid scheduler."""
        sched = create_scheduler(MockOptimizer(lr=1e-3), "onecycle", total_steps=100)
        assert sched is not None
        lr = sched.get_lr()
        assert isinstance(lr, list)
        assert len(lr) > 0
