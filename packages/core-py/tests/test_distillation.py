"""Tests for knowledge distillation — DistillationLoss, DistillationTrainer.

All tests use SloNet natively (no PyTorch dependency).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from domains.training.slonet import Tensor, no_grad
from domains.training.distillation import (
    DistillationConfig, DistillationLoss, DistillationTrainer,
    ProgressiveDistiller, create_distillation_trainer,
    _to_np, _to_tensor,
)


class TestDistillationConfig:
    def test_default_values(self):
        c = DistillationConfig()
        assert c.temperature == 4.0
        assert c.alpha == 0.5
        assert c.beta == 0.5
        assert c.gamma == 0.0
        assert c.distillation_type == "logits"

    def test_temperature_schedule(self):
        c = DistillationConfig(temperature_schedule=[4.0, 3.0, 2.0])
        assert c.temperature_schedule == [4.0, 3.0, 2.0]

    def test_stage_weights(self):
        c = DistillationConfig(progressive=True, stage_weights=[0.3, 0.7])
        assert c.progressive is True


class TestDistillationLoss:
    def test_soft_loss_only(self):
        config = DistillationConfig(alpha=0.0, beta=1.0, gamma=0.0)
        loss_fn = DistillationLoss(config)
        s_logits = np.random.randn(2, 10).astype(np.float32)
        t_logits = np.random.randn(2, 10).astype(np.float32)
        total, losses = loss_fn.forward(s_logits, t_logits)
        assert "soft_loss" in losses
        assert "hard_loss" not in losses
        assert total > 0

    def test_hard_loss_only(self):
        config = DistillationConfig(alpha=1.0, beta=0.0, gamma=0.0)
        loss_fn = DistillationLoss(config)
        s_logits = np.random.randn(2, 10).astype(np.float32)
        t_logits = np.random.randn(2, 10).astype(np.float32)
        labels = np.array([[1, 2]], dtype=np.int64)
        total, losses = loss_fn.forward(s_logits, t_logits, labels)
        assert "hard_loss" in losses
        assert total > 0

    def test_combined_alpha_beta(self):
        config = DistillationConfig(alpha=0.5, beta=0.5, gamma=0.0)
        loss_fn = DistillationLoss(config)
        s_logits = np.random.randn(2, 10).astype(np.float32)
        t_logits = np.random.randn(2, 10).astype(np.float32)
        labels = np.array([[1, 2]], dtype=np.int64)
        total, losses = loss_fn.forward(s_logits, t_logits, labels)
        assert "soft_loss" in losses
        assert "hard_loss" in losses
        assert total > 0

    def test_feature_loss(self):
        config = DistillationConfig(alpha=0.0, beta=0.0, gamma=1.0)
        loss_fn = DistillationLoss(config)
        s_logits = np.random.randn(2, 10).astype(np.float32)
        t_logits = np.random.randn(2, 10).astype(np.float32)
        sh = np.random.randn(2, 8).astype(np.float32)
        th = np.random.randn(2, 8).astype(np.float32)
        total, losses = loss_fn.forward(s_logits, t_logits, student_hidden=sh, teacher_hidden=th)
        assert "feature_loss" in losses
        assert total > 0

    def test_feature_loss_with_projection(self):
        config = DistillationConfig(alpha=0.0, beta=0.0, gamma=1.0)
        loss_fn = DistillationLoss(config)
        s_logits = np.random.randn(2, 10).astype(np.float32)
        t_logits = np.random.randn(2, 10).astype(np.float32)
        sh = np.random.randn(2, 16).astype(np.float32)
        th = np.random.randn(2, 8).astype(np.float32)
        total, losses = loss_fn.forward(s_logits, t_logits, student_hidden=sh, teacher_hidden=th)
        assert "feature_loss" in losses
        assert total > 0
        assert loss_fn.projection is not None

    def test_loss_is_scalar(self):
        config = DistillationConfig(alpha=0.5, beta=0.5)
        loss_fn = DistillationLoss(config)
        s_logits = np.random.randn(2, 10).astype(np.float32)
        t_logits = np.random.randn(2, 10).astype(np.float32)
        labels = np.array([[1, 2]], dtype=np.int64)
        total, losses = loss_fn.forward(s_logits, t_logits, labels)
        assert isinstance(total, float)



    def test_zero_weights_produce_zero_total(self):
        config = DistillationConfig(alpha=0.0, beta=0.0, gamma=0.0)
        loss_fn = DistillationLoss(config)
        s_logits = np.random.randn(2, 10).astype(np.float32)
        t_logits = np.random.randn(2, 10).astype(np.float32)
        total, losses = loss_fn.forward(s_logits, t_logits)
        assert total == 0.0

    def test_call_method(self):
        config = DistillationConfig(alpha=0.0, beta=1.0)
        loss_fn = DistillationLoss(config)
        total, losses = loss_fn(np.random.randn(2, 5), np.random.randn(2, 5))
        assert total > 0

    def test_teacher_logits_allow_none_labels(self):
        config = DistillationConfig(alpha=0.0, beta=1.0)
        loss_fn = DistillationLoss(config)
        total, losses = loss_fn(np.random.randn(2, 5), np.random.randn(2, 5), labels=None)
        assert total > 0


class _MockModel:
    """Minimal SloNet model stub for distillation trainer tests."""
    def __init__(self, vocab_size=10):
        self.vocab_size = vocab_size
        self._params = []

    def __call__(self, x):
        batch, seq = x.shape[0], x.shape[1]
        return np.random.randn(batch, seq, self.vocab_size).astype(np.float32)

    def parameters(self):
        return self._params

    def named_modules(self):
        return []

    def named_children(self):
        return []

    def eval(self):
        pass


class TestDistillationTrainer:
    def test_step_returns_loss_dict(self):
        teacher = _MockModel(vocab_size=10)
        student = _MockModel(vocab_size=10)
        config = DistillationConfig(alpha=0.5, beta=0.5)
        trainer = DistillationTrainer(teacher, student, config)
        inputs = Tensor(np.array([[1, 2, 3]], dtype=np.int64))
        labels = Tensor(np.array([[2, 3, 4]], dtype=np.int64))
        losses = trainer.step(inputs, labels)
        assert isinstance(losses, dict)

    def test_step_teacher_eval_mode(self):
        teacher = _MockModel(vocab_size=10)
        student = _MockModel(vocab_size=10)
        trainer = DistillationTrainer(teacher, student, DistillationConfig())
        assert not any(hasattr(p, 'requires_grad') and p.requires_grad for p in trainer.teacher.parameters())

    def test_step_soft_only(self):
        teacher = _MockModel(vocab_size=10)
        student = _MockModel(vocab_size=10)
        config = DistillationConfig(alpha=0.0, beta=1.0)
        trainer = DistillationTrainer(teacher, student, config)
        inputs = Tensor(np.array([[1, 2, 3]], dtype=np.int64))
        labels = Tensor(np.array([[2, 3, 4]], dtype=np.int64))
        losses = trainer.step(inputs, labels)
        assert "soft_loss" in losses

    def test_step_wipes_student_grads(self):
        teacher = _MockModel(vocab_size=5)
        student = _MockModel(vocab_size=5)
        trainer = DistillationTrainer(teacher, student, DistillationConfig(alpha=0.5, beta=0.5))
        inputs = Tensor(np.array([[1, 2]], dtype=np.int64))
        labels = Tensor(np.array([[2, 3]], dtype=np.int64))
        trainer.step(inputs, labels)
        for p in student.parameters():
            if hasattr(p, 'grad'):
                assert p.grad is None

    def test_distill_logits_returns_loss(self):
        teacher = _MockModel(vocab_size=10)
        student = _MockModel(vocab_size=10)
        trainer = DistillationTrainer(teacher, student, DistillationConfig())
        s = np.random.randn(2, 10).astype(np.float32)
        t = np.random.randn(2, 10).astype(np.float32)
        loss = trainer.distill_logits(s, t)
        assert isinstance(loss, (float, Tensor))

    def test_distill_hidden_states(self):
        teacher = _MockModel(vocab_size=10)
        student = _MockModel(vocab_size=10)
        trainer = DistillationTrainer(teacher, student, DistillationConfig())
        sh = np.random.randn(2, 8).astype(np.float32)
        th = np.random.randn(2, 8).astype(np.float32)
        loss = trainer.distill_hidden_states(sh, th)
        assert loss is not None


class TestProgressiveDistiller:
    def test_init_creates_layer_mapping(self):
        teacher = _MockModel(vocab_size=10)
        student = _MockModel(vocab_size=10)
        distiller = ProgressiveDistiller(teacher, student, DistillationConfig())
        assert distiller.layer_mapping is not None

    def test_distill_intermediate_returns_float(self):
        teacher = _MockModel(vocab_size=10)
        student = _MockModel(vocab_size=10)
        distiller = ProgressiveDistiller(teacher, student, DistillationConfig())
        result = distiller.distill_intermediate(np.array([[1, 2]]))
        assert result == 0.0


class TestHelpers:
    def test_to_np_from_tensor(self):
        t = Tensor(np.array([1.0, 2.0]))
        result = _to_np(t)
        assert isinstance(result, np.ndarray)
        assert result[0] == 1.0

    def test_to_np_from_ndarray(self):
        a = np.array([3.0, 4.0])
        result = _to_np(a)
        assert result is a

    def test_to_np_from_list(self):
        result = _to_np([1.0, 2.0])
        assert isinstance(result, np.ndarray)

    def test_to_tensor_from_array(self):
        arr = np.array([5.0, 6.0])
        t = _to_tensor(arr)
        assert isinstance(t, Tensor)

    def test_to_tensor_requires_grad(self):
        arr = np.array([1.0, 2.0])
        t = _to_tensor(arr, requires_grad=True)
        assert t.requires_grad


def test_create_distillation_trainer():
    teacher = _MockModel(vocab_size=10)
    student = _MockModel(vocab_size=10)
    trainer = create_distillation_trainer(teacher, student, temperature=3.0, alpha=0.3, beta=0.7)
    assert isinstance(trainer, DistillationTrainer)
    assert trainer.config.temperature == 3.0
    assert trainer.config.alpha == 0.3
    assert trainer.config.beta == 0.7
