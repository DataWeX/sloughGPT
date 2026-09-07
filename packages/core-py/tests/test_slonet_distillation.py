"""Tests for training.distillation — knowledge distillation."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.distillation import (
    DistillationConfig,
    DistillationLoss,
    _to_np,
    _to_tensor,
    _size,
)
from domains.training.slonet import Tensor


# ── _to_np / _to_tensor / _size helpers ────────────────────────────────────


class TestHelperFunctions:

    def test_to_np_tensor(self):
        t = Tensor(np.array([1.0, 2.0]))
        result = _to_np(t)
        assert isinstance(result, np.ndarray)

    def test_to_np_array(self):
        arr = np.array([1.0, 2.0])
        result = _to_np(arr)
        assert isinstance(result, np.ndarray)

    def test_to_np_list(self):
        result = _to_np([1.0, 2.0])
        assert isinstance(result, np.ndarray)

    def test_to_tensor_array(self):
        arr = np.array([1.0, 2.0])
        t = _to_tensor(arr)
        assert isinstance(t, Tensor)
        assert t.requires_grad is False

    def test_to_tensor_no_grad(self):
        arr = np.array([1.0, 2.0])
        t = _to_tensor(arr, requires_grad=False)
        assert t.requires_grad is False

    def test_size(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert _size(arr, 0) == 2
        assert _size(arr, 1) == 2


# ── DistillationConfig ─────────────────────────────────────────────────────


class TestDistillationConfig:

    def test_default(self):
        config = DistillationConfig()
        assert config.temperature == 4.0
        assert config.alpha == 0.5
        assert config.beta == 0.5

    def test_custom(self):
        config = DistillationConfig(temperature=2.0, alpha=0.7)
        assert config.temperature == 2.0
        assert config.alpha == 0.7


# ── DistillationLoss ───────────────────────────────────────────────────────


class TestDistillationLoss:

    def test_soft_loss(self):
        config = DistillationConfig(beta=1.0, alpha=0.0)
        loss_fn = DistillationLoss(config)
        student = np.random.randn(1, 10)
        teacher = np.random.randn(1, 10)
        total, losses = loss_fn(student, teacher)
        assert "soft_loss" in losses
        assert total > 0

    def test_hard_loss(self):
        config = DistillationConfig(alpha=1.0, beta=0.0)
        loss_fn = DistillationLoss(config)
        student = np.random.randn(1, 10)
        teacher = np.random.randn(1, 10)
        labels = np.array([3])
        total, losses = loss_fn(student, teacher, labels=labels)
        assert "hard_loss" in losses
        assert total > 0

    def test_combined_loss(self):
        config = DistillationConfig(alpha=0.5, beta=0.5)
        loss_fn = DistillationLoss(config)
        student = np.random.randn(1, 10)
        teacher = np.random.randn(1, 10)
        labels = np.array([3])
        total, losses = loss_fn(student, teacher, labels=labels)
        assert "total_loss" in losses
        assert total > 0

    def test_feature_loss(self):
        config = DistillationConfig(gamma=1.0, alpha=0.0, beta=0.0)
        loss_fn = DistillationLoss(config)
        student = np.random.randn(1, 10)
        teacher = np.random.randn(1, 10)
        student_hidden = np.random.randn(1, 10)
        teacher_hidden = np.random.randn(1, 10)
        total, losses = loss_fn(student, teacher, student_hidden=student_hidden, teacher_hidden=teacher_hidden)
        assert "feature_loss" in losses
        assert total > 0

    def test_feature_loss_projection(self):
        config = DistillationConfig(gamma=1.0, alpha=0.0, beta=0.0)
        loss_fn = DistillationLoss(config)
        student = np.random.randn(1, 10)
        teacher = np.random.randn(1, 10)
        student_hidden = np.random.randn(1, 10)
        teacher_hidden = np.random.randn(1, 15)
        total, losses = loss_fn(student, teacher, student_hidden=student_hidden, teacher_hidden=teacher_hidden)
        assert "feature_loss" in losses
