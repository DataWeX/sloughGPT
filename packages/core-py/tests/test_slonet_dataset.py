"""Tests for training.slonet — SloDataset, SloDataLoader, and LR schedulers."""

from __future__ import annotations

import math
import pytest
import numpy as np
from domains.training.slonet import (
    SloDataset,
    SloDataLoader,
    SloSGD,
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


# ── Helpers ─────────────────────────────────────────────────────────────────


class DummyDataset(SloDataset):
    def __init__(self, n=10):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"x": np.array([float(idx)]), "y": idx % 2}


# ── SloDataset ──────────────────────────────────────────────────────────────


class TestSloDataset:

    def test_len(self):
        ds = DummyDataset(10)
        assert len(ds) == 10

    def test_getitem(self):
        ds = DummyDataset(10)
        item = ds[0]
        assert "x" in item
        assert "y" in item

    def test_iter(self):
        ds = DummyDataset(5)
        items = list(ds)
        assert len(items) == 5


# ── SloDataLoader ───────────────────────────────────────────────────────────


class TestDataLoader:

    def test_init(self):
        ds = DummyDataset(10)
        loader = SloDataLoader(ds, batch_size=2)
        assert loader.batch_size == 2

    def test_len(self):
        ds = DummyDataset(10)
        loader = SloDataLoader(ds, batch_size=3)
        assert len(loader) == 4  # ceil(10/3)

    def test_len_drop_last(self):
        ds = DummyDataset(10)
        loader = SloDataLoader(ds, batch_size=3, drop_last=True)
        assert len(loader) == 3  # floor(10/3)

    def test_iter(self):
        ds = DummyDataset(10)
        loader = SloDataLoader(ds, batch_size=2)
        batches = list(loader)
        assert len(batches) == 5
        assert len(batches[0]) == 2

    def test_shuffle(self):
        ds = DummyDataset(10)
        loader = SloDataLoader(ds, batch_size=2, shuffle=True)
        batches = list(loader)
        assert len(batches) == 5

    def test_collate_fn(self):
        ds = DummyDataset(10)
        def collate(batch):
            return {"x": np.stack([b["x"] for b in batch]), "y": np.array([b["y"] for b in batch])}
        loader = SloDataLoader(ds, batch_size=2, collate_fn=collate)
        batch = next(iter(loader))
        assert batch["x"].shape == (2, 1)

    def test_reset(self):
        ds = DummyDataset(10)
        loader = SloDataLoader(ds, batch_size=2)
        iter(loader)
        loader.reset()
        assert loader._idx == 0

    def test_min_length(self):
        ds = DummyDataset(1)
        loader = SloDataLoader(ds, batch_size=10)
        assert len(loader) >= 1


# ── SloConstantLR ───────────────────────────────────────────────────────────


class TestSloConstantLR:

    def test_constant(self):
        opt = SloSGD(lr=0.1)
        sched = SloConstantLR(opt)
        lrs = sched.get_lr()
        assert lrs == [0.1]


# ── SloStepLR ───────────────────────────────────────────────────────────────


class TestSloStepLR:

    def test_step(self):
        opt = SloSGD(lr=1.0)
        sched = SloStepLR(opt, step_size=2, gamma=0.5)
        assert sched.get_lr() == [1.0]
        sched.step()
        sched.step()
        assert sched.get_lr() == [0.5]


# ── SloCosineAnnealingLR ────────────────────────────────────────────────────


class TestSloCosineAnnealingLR:

    def test_cosine(self):
        opt = SloSGD(lr=1.0)
        sched = SloCosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        lrs = sched.get_lr()
        assert lrs[0] == pytest.approx(1.0, rel=1e-5)
        sched.step()
        sched.step()
        sched.step()
        lrs = sched.get_lr()
        assert lrs[0] < 1.0


# ── SloReduceLROnPlateau ───────────────────────────────────────────────────


class TestSloReduceLROnPlateau:

    def test_reduce(self):
        opt = SloSGD(lr=1.0)
        sched = SloReduceLROnPlateau(opt, patience=2, factor=0.5)
        sched.step(1.0)
        sched.step(1.1)  # worse
        sched.step(1.2)  # worse
        sched.step(1.3)  # worse
        assert opt.lr < 1.0

    def test_no_reduce(self):
        opt = SloSGD(lr=1.0)
        sched = SloReduceLROnPlateau(opt, patience=5, factor=0.5)
        for i in range(5):
            sched.step(1.0 - i * 0.1)
        assert opt.lr == 1.0


# ── WarmupCosineScheduler ───────────────────────────────────────────────────


class TestWarmupCosineScheduler:

    def test_warmup(self):
        opt = SloSGD(lr=1.0)
        sched = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
        lrs = sched.get_lr()
        assert lrs[0] < 1.0
        sched.step()
        lrs = sched.get_lr()
        assert lrs[0] > 0

    def test_cosine(self):
        opt = SloSGD(lr=1.0)
        sched = WarmupCosineScheduler(opt, warmup_steps=0, total_steps=100)
        lrs = sched.get_lr()
        assert lrs[0] == pytest.approx(1.0, rel=1e-5)


# ── PolynomialDecayScheduler ────────────────────────────────────────────────


class TestPolynomialDecayScheduler:

    def test_decay(self):
        opt = SloSGD(lr=1.0)
        sched = PolynomialDecayScheduler(opt, total_steps=100, min_lr=0.0, power=1.0)
        lrs = sched.get_lr()
        assert lrs[0] == pytest.approx(1.0, rel=1e-5)
        sched.last_epoch = 50
        lrs = sched.get_lr()
        assert lrs[0] == pytest.approx(0.5, rel=1e-2)


# ── LinearWarmupScheduler ───────────────────────────────────────────────────


class TestLinearWarmupScheduler:

    def test_warmup(self):
        opt = SloSGD(lr=1.0)
        sched = LinearWarmupScheduler(opt, warmup_steps=10, base_lr=1.0)
        lrs = sched.get_lr()
        assert lrs[0] < 1.0

    def test_hold(self):
        opt = SloSGD(lr=1.0)
        sched = LinearWarmupScheduler(opt, warmup_steps=5, hold_steps=10, base_lr=1.0)
        sched.last_epoch = 10
        lrs = sched.get_lr()
        assert lrs[0] == pytest.approx(1.0, rel=1e-5)


# ── SloOneCycleLR ───────────────────────────────────────────────────────────


class TestSloOneCycleLR:

    def test_one_cycle(self):
        opt = SloSGD(lr=0.001)
        sched = SloOneCycleLR(opt, max_lr=1.0, total_steps=100, div_factor=25.0)
        lrs = sched.get_lr()
        # At step 0, factor = 0/pct_start, so lr = base_lr * div_factor * 0
        # After stepping, lr should be positive
        sched.step()
        lrs = sched.get_lr()
        assert lrs[0] > 0


# ── SloCyclicLR ─────────────────────────────────────────────────────────────


class TestSloCyclicLR:

    def test_cyclic(self):
        opt = SloSGD(lr=0.001)
        sched = SloCyclicLR(opt, base_lr=0.001, max_lr=0.1, step_size_up=10)
        lrs = sched.get_lr()
        assert lrs[0] >= 0.001


# ── create_scheduler ────────────────────────────────────────────────────────


class TestCreateScheduler:

    def test_create_constant(self):
        opt = SloSGD(lr=0.1)
        sched = create_scheduler(opt, "constant")
        assert isinstance(sched, SloConstantLR)

    def test_create_cosine(self):
        opt = SloSGD(lr=0.1)
        sched = create_scheduler(opt, "cosine", total_steps=100, warmup_steps=10)
        assert isinstance(sched, WarmupCosineScheduler)

    def test_create_step(self):
        opt = SloSGD(lr=0.1)
        sched = create_scheduler(opt, "step", step_size=10)
        assert isinstance(sched, SloStepLR)

    def test_create_unknown(self):
        opt = SloSGD(lr=0.1)
        with pytest.raises(ValueError):
            create_scheduler(opt, "unknown")


# ── Scheduler state_dict ────────────────────────────────────────────────────


class TestSchedulerStateDict:

    def test_state_dict(self):
        opt = SloSGD(lr=0.1)
        sched = SloStepLR(opt, step_size=10)
        sched.step()
        sd = sched.state_dict()
        assert "last_epoch" in sd

    def test_load_state_dict(self):
        opt1 = SloSGD(lr=0.1)
        sched1 = SloStepLR(opt1, step_size=10)
        sched1.step()
        sd = sched1.state_dict()

        opt2 = SloSGD(lr=0.1)
        sched2 = SloStepLR(opt2, step_size=10)
        sched2.load_state_dict(sd)
        assert sched2.last_epoch == 1
