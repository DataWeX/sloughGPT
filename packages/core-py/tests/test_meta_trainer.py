"""Tests for the StepStrategy seam (train_step hook) and the MetaTrainer host."""

from __future__ import annotations

import numpy as np
import pytest

from domain.training._internal.meta_trainer import MetaTrainer
from domain.training._internal.slonet import Tensor
from domain.training._internal.step_strategy import (
    DistillationStrategy,
    EwcStrategy,
    OwnedObjectiveStrategy,
    StepStrategy,
)
from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

DATA_TEXT = (
    "the quick brown fox jumps over the lazy dog and runs across the meadow "
    "again and again while the dog sleeps soundly in the warm sun all day "
    "long. " * 8
)


# ── Fixtures / helpers ──────────────────────────────────────────────────────


@pytest.fixture
def data_path(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(DATA_TEXT, encoding="utf-8")
    return str(p)


def tiny_config(tmp_path, **overrides):
    cfg = TrainerConfig(
        vocab_size=0,
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=8,
        dropout=0.0,
        batch_size=2,
        epochs=1,
        max_steps=None,
        gradient_accumulation_steps=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_interval=1,
        eval_interval=1000,
        checkpoint_interval=1000,
        warmup_steps=1,
        min_lr=1e-5,
        max_checkpoints=5,
        scheduler_type="cosine",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_trainer(data_path, cfg, **kwargs):
    return SloughGPTTrainer(data_path, config=cfg, **kwargs)


class _NoopOptimizer:
    """Freezes weights between steps so two train_steps see identical loss."""

    def step(self, params=None):
        return None


def _trainable(tmp_path, data_path, **kwargs):
    """Trainer whose batch and weights are deterministic across train_steps."""
    trainer = make_trainer(data_path, tiny_config(tmp_path), **kwargs)
    trainer.optimizer = _NoopOptimizer()
    batch = trainer.get_batch("train")
    trainer.get_batch = lambda split="train": batch
    return trainer, batch


class _RecordingStrategy:
    name = "recording"

    def __init__(self, loss_delta: float = 0.0, metrics: dict | None = None):
        self.loss_delta = loss_delta
        self.metrics = metrics or {}
        self.calls: list[dict] = []

    def compute(self, model, x, y, loss, *, logits=None, metadata=None):
        self.calls.append({"logits": logits, "metadata": metadata})
        out = loss + self.loss_delta if self.loss_delta else loss
        return out, dict(self.metrics)


class _FakeEwc:
    def __init__(self, snapshots: dict):
        self.task_snapshots = snapshots

    def ewc_loss(self, task_id=None):
        return Tensor(np.array(2.0)), {"active_tasks": 1 if self.task_snapshots else 0}


class _FakeOwned:
    name = "owned_objective"

    def compute(self, model, x, y, loss, metadata=None):
        return loss, {"primary": float(loss.item())}


# ── StepStrategy seam ───────────────────────────────────────────────────────


class TestStepStrategySeam:
    def test_no_strategy_keeps_legacy_metric_keys(self, tmp_path, data_path):
        trainer = make_trainer(data_path, tiny_config(tmp_path))
        assert trainer.strategies == []
        assert trainer._iter_strategies() == []
        metrics = trainer.train_step()
        assert set(metrics) == {"loss", "raw_loss"}

    def test_strategy_is_structural_and_metrics_merge(self, tmp_path, data_path):
        rec = _RecordingStrategy(metrics={"feedback_score": 7.0})
        assert isinstance(rec, StepStrategy)
        trainer = make_trainer(data_path, tiny_config(tmp_path), strategies=[rec])
        metrics = trainer.train_step()
        assert metrics["feedback_score"] == 7.0
        assert metrics["raw_loss"] > 0
        assert len(rec.calls) == 1
        assert rec.calls[0]["logits"] is not None

    def test_strategy_loss_modification_reaches_raw_loss(self, tmp_path, data_path):
        trainer, _ = _trainable(tmp_path, data_path)
        base = trainer.train_step()["raw_loss"]
        trainer.strategies = [_RecordingStrategy(loss_delta=1.0)]
        bumped = trainer.train_step()["raw_loss"]
        assert bumped == pytest.approx(base + 1.0)

    def test_ewc_strategy_guards_empty_snapshots(self, tmp_path, data_path):
        trainer, _ = _trainable(tmp_path, data_path)
        base = trainer.train_step()["raw_loss"]

        trainer._ewc = _FakeEwc({})
        assert trainer.train_step()["raw_loss"] == pytest.approx(base)

        trainer._ewc = _FakeEwc({"task-a": object()})
        with_penalty = trainer.train_step()
        assert with_penalty["raw_loss"] == pytest.approx(base + 2.0)
        # legacy parity: the EWC path contributed loss only, no metric keys
        assert set(with_penalty) == {"loss", "raw_loss"}

    def test_strategy_order_legacy_first_then_injected(self, tmp_path, data_path):
        trainer = make_trainer(data_path, tiny_config(tmp_path))
        trainer._ewc = _FakeEwc({"task-a": object()})
        trainer._owned_objective = _FakeOwned()
        trainer.strategies = [_RecordingStrategy()]
        names = [s.name for s in trainer._iter_strategies()]
        assert names == ["ewc", "owned_objective", "recording"]

    def test_owned_objective_strategy_keeps_owned_prefix(self, tmp_path, data_path):
        trainer = make_trainer(data_path, tiny_config(tmp_path))
        strategy = OwnedObjectiveStrategy(_FakeOwned())
        _, metrics = strategy.compute(
            trainer.model, np.zeros(1), np.zeros(1), Tensor(np.array(1.0))
        )
        assert set(metrics) == {"owned_primary"}

    def test_distillation_strategy_adds_soft_loss(self, tmp_path, data_path):
        trainer, (x, y) = _trainable(tmp_path, data_path)
        base = trainer.train_step()["raw_loss"]

        student_logits, _ = trainer.training_model(x, y)
        zeros = np.zeros_like(np.asarray(student_logits.data))
        trainer.strategies = [DistillationStrategy(lambda xs, ys: (zeros, None))]

        metrics = trainer.train_step()
        assert metrics["distill_soft"] > 0
        assert metrics["raw_loss"] == pytest.approx(base + metrics["distill_soft"])

    def test_distillation_strategy_weight_zero_is_noop(self, tmp_path, data_path):
        trainer, _ = _trainable(tmp_path, data_path)
        base = trainer.train_step()["raw_loss"]
        teacher = lambda xs, ys: (_ for _ in ()).throw(AssertionError("teacher called"))
        trainer.strategies = [DistillationStrategy(teacher, weight=0.0)]
        metrics = trainer.train_step()
        assert metrics["raw_loss"] == pytest.approx(base)
        assert "distill_soft" not in metrics


# ── MetaTrainer host ────────────────────────────────────────────────────────


class TestMetaTrainer:
    def test_injects_strategies_and_delegates_train(self, tmp_path, data_path):
        rec = _RecordingStrategy(metrics={"feedback_score": 1.0})
        mt = MetaTrainer(data_path, config=tiny_config(tmp_path), strategies=[rec])
        trainer = mt.build_trainer()
        assert trainer.strategies == [rec]
        assert mt.trainer is trainer

        metrics = trainer.train_step()
        assert metrics["feedback_score"] == 1.0

        sentinel = object()
        trainer.train = lambda **kwargs: sentinel
        assert mt.train(epochs=2) is sentinel

    def test_train_builds_lazily_without_config(self, tmp_path, data_path):
        mt = MetaTrainer(
            data_path,
            n_embed=16,
            n_layer=1,
            n_head=2,
            block_size=8,
            vocab_size=0,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        assert mt.trainer is None
        trainer = mt.build_trainer()
        assert isinstance(trainer, SloughGPTTrainer)
        assert trainer.strategies == []
        assert mt.trainer is trainer

    def test_factories_compose_expected_strategies(self, tmp_path, data_path):
        fake_ewc = _FakeEwc({})
        mt = MetaTrainer.with_ewc(data_path, fake_ewc, config=tiny_config(tmp_path))
        assert isinstance(mt.strategies[0], EwcStrategy)
        assert mt.strategies[0].learner is fake_ewc

        def teacher(x, y):
            return np.zeros(1), None

        mt2 = MetaTrainer.with_distillation(data_path, teacher, temperature=1.5)
        assert isinstance(mt2.strategies[0], DistillationStrategy)
        assert mt2.strategies[0].temperature == 1.5
