"""Tests for training.slonet — LR scheduler classes (comprehensive)."""

from __future__ import annotations

import math
import pytest
import numpy as np
from domains.training.slonet import (
    SloSGD,
    SloAdam,
    SloAdamW,
    SloLRScheduler,
    SloConstantLR,
    SloStepLR,
    SloCosineAnnealingLR,
    SloReduceLROnPlateau,
    WarmupCosineScheduler,
    PolynomialDecayScheduler,
    LinearWarmupScheduler,
    SloOneCycleLR,
    SloCyclicLR,
    create_scheduler,
)


# ── SloLRScheduler base ─────────────────────────────────────────────────────


class TestSloLRSchedulerBase:

    def test_step_updates_lr(self):
        opt = SloSGD(lr=1.0)
        sched = SloConstantLR(opt)
        sched.step()
        assert sched.last_epoch == 1

    def test_get_last_lr(self):
        opt = SloSGD(lr=1.0)
        sched = SloConstantLR(opt)
        lrs = sched.get_last_lr()
        assert lrs == [1.0]

    def test_state_dict(self):
        opt = SloSGD(lr=1.0)
        sched = SloConstantLR(opt)
        sched.step()
        sd = sched.state_dict()
        assert sd["last_epoch"] == 1
        assert "base_lrs" in sd

    def test_load_state_dict(self):
        opt = SloSGD(lr=1.0)
        sched = SloConstantLR(opt)
        sched.load_state_dict({"last_epoch": 5, "base_lrs": [1.0]})
        assert sched.last_epoch == 5


# ── SloConstantLR ───────────────────────────────────────────────────────────


class TestSloConstantLRDetailed:

    def test_constant_after_steps(self):
        opt = SloSGD(lr=0.5)
        sched = SloConstantLR(opt)
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == [0.5]


# ── SloStepLR ───────────────────────────────────────────────────────────────


class TestSloStepLRDetailed:

    def test_step_decay(self):
        opt = SloSGD(lr=1.0)
        sched = SloStepLR(opt, step_size=3, gamma=0.5)
        # Step 0 already happened in __init__, so last_epoch=0
        # lrs[i] corresponds to last_epoch = i+1
        lrs = []
        for _ in range(9):
            sched.step()
            lrs.append(sched.get_lr()[0])
        assert lrs[0] == 1.0   # last_epoch=1, 1//3=0
        assert lrs[1] == 1.0   # last_epoch=2, 2//3=0
        assert lrs[2] == 0.5   # last_epoch=3, 3//3=1
        assert lrs[3] == 0.5   # last_epoch=4, 4//3=1


# ── SloCosineAnnealingLR ────────────────────────────────────────────────────


class TestSloCosineAnnealingLRDetailed:

    def test_cosine_schedule(self):
        opt = SloSGD(lr=1.0)
        sched = SloCosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        lrs = []
        for _ in range(11):
            sched.step()
            lrs.append(sched.get_lr()[0])
        # At T_max, lr should be eta_min
        assert lrs[-1] == pytest.approx(0.0, abs=1e-5)
        # At start, lr should be close to base_lr
        assert lrs[0] == pytest.approx(1.0, abs=0.1)


# ── SloReduceLROnPlateau ───────────────────────────────────────────────────


class TestSloReduceLROnPlateauDetailed:

    def test_reduce_on_plateau(self):
        opt = SloSGD(lr=1.0)
        sched = SloReduceLROnPlateau(opt, patience=3, factor=0.5)
        # Simulate metric not improving
        for i in range(5):
            sched.step(1.0 + i * 0.1)
        assert opt.lr < 1.0

    def test_no_reduce_improving(self):
        opt = SloSGD(lr=1.0)
        sched = SloReduceLROnPlateau(opt, patience=5, factor=0.5)
        for i in range(5):
            sched.step(1.0 - i * 0.1)
        assert opt.lr == 1.0

    def test_state_dict(self):
        opt = SloSGD(lr=1.0)
        sched = SloReduceLROnPlateau(opt)
        sched.step(1.0)
        sd = sched.state_dict()
        assert "best" in sd
        assert "num_bad_epochs" in sd

    def test_load_state_dict(self):
        opt = SloSGD(lr=1.0)
        sched = SloReduceLROnPlateau(opt)
        sched.load_state_dict({"best": 0.5, "num_bad_epochs": 2, "cooldown_counter": 0, "last_lr": 0.5})
        assert sched.best == 0.5
        assert sched.num_bad_epochs == 2


# ── WarmupCosineScheduler ───────────────────────────────────────────────────


class TestWarmupCosineSchedulerDetailed:

    def test_warmup_phase(self):
        opt = SloSGD(lr=1.0)
        sched = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(sched.get_lr()[0])
        # During warmup, lr should increase
        assert lrs[-1] > lrs[0]

    def test_cosine_phase(self):
        opt = SloSGD(lr=1.0)
        sched = WarmupCosineScheduler(opt, warmup_steps=0, total_steps=100)
        lrs = []
        for _ in range(50):
            sched.step()
            lrs.append(sched.get_lr()[0])
        # After halfway, lr should decrease
        assert lrs[-1] < lrs[0]


# ── PolynomialDecayScheduler ────────────────────────────────────────────────


class TestPolynomialDecaySchedulerDetailed:

    def test_polynomial_decay(self):
        opt = SloSGD(lr=1.0)
        sched = PolynomialDecayScheduler(opt, total_steps=100, min_lr=0.0, power=1.0)
        lrs = []
        for _ in range(101):
            sched.step()
            lrs.append(sched.get_lr()[0])
        # At end, lr should be min_lr
        assert lrs[-1] == pytest.approx(0.0, abs=1e-5)


# ── LinearWarmupScheduler ───────────────────────────────────────────────────


class TestLinearWarmupSchedulerDetailed:

    def test_warmup_then_hold(self):
        opt = SloSGD(lr=1.0)
        sched = LinearWarmupScheduler(opt, warmup_steps=5, hold_steps=10, base_lr=1.0)
        lrs = []
        for _ in range(20):
            sched.step()
            lrs.append(sched.get_lr()[0])
        # During hold, lr should be constant
        assert lrs[6] == pytest.approx(1.0, abs=1e-5)
        assert lrs[15] == pytest.approx(1.0, abs=1e-5)

    def test_warmup_then_cosine(self):
        opt = SloSGD(lr=1.0)
        sched = LinearWarmupScheduler(opt, warmup_steps=5, hold_steps=0, base_lr=1.0, decay_type="cosine", total_steps=20)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(sched.get_lr()[0])
        # During warmup, lr increases; after, cosine decay decreases
        assert lrs[2] > lrs[0]
        assert lrs[-1] < lrs[4]


# ── SloOneCycleLR ───────────────────────────────────────────────────────────


class TestSloOneCycleLRDetailed:

    def test_one_cycle_schedule(self):
        opt = SloSGD(lr=0.001)
        sched = SloOneCycleLR(opt, max_lr=1.0, total_steps=100, div_factor=25.0)
        lrs = []
        for _ in range(100):
            sched.step()
            lrs.append(sched.get_lr()[0])
        # Should increase then decrease
        assert max(lrs) > lrs[0]


# ── SloCyclicLR ─────────────────────────────────────────────────────────────


class TestSloCyclicLRDetailed:

    def test_cyclic_schedule(self):
        opt = SloSGD(lr=0.001)
        sched = SloCyclicLR(opt, base_lr=0.001, max_lr=0.1, step_size_up=10)
        lrs = []
        for _ in range(20):
            sched.step()
            lrs.append(sched.get_lr()[0])
        # Should oscillate
        assert max(lrs) > min(lrs)


# ── create_scheduler factory ────────────────────────────────────────────────


class TestCreateSchedulerFactory:

    def test_all_types(self):
        opt = SloSGD(lr=0.1)
        types = ["constant", "cosine", "step", "polynomial", "warmup", "onecycle", "cyclic"]
        for stype in types:
            sched = create_scheduler(opt, stype, total_steps=100, warmup_steps=10)
            assert sched is not None

    def test_plateau_type(self):
        opt = SloSGD(lr=0.1)
        sched = create_scheduler(opt, "plateau", factor=0.5, patience=5)
        assert isinstance(sched, SloReduceLROnPlateau)

    def test_cosine_annealing_type(self):
        opt = SloSGD(lr=0.1)
        sched = create_scheduler(opt, "cosine_annealing", T_max=100)
        assert isinstance(sched, SloCosineAnnealingLR)

    def test_invalid_type(self):
        opt = SloSGD(lr=0.1)
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(opt, "invalid_type")
