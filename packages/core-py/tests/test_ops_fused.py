"""Tests for domains.ops — FusedLayerNorm, FusedRMSNorm, FusedCrossEntropyLoss, FusedAttentionBias."""

import numpy as np
from domains.ops import (
    FusedLayerNorm, FusedRMSNorm, FusedCrossEntropyLoss, FusedAttentionBias,
)


class TestFusedLayerNorm:
    def test_shape_preserved(self):
        ln = FusedLayerNorm(64)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        out = ln(x)
        assert out.shape == x.shape

    def test_mean_near_zero(self):
        ln = FusedLayerNorm(64)
        x = np.ones((4, 8, 64), dtype=np.float32) * 5.0
        out = ln(x)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-5)

    def test_callable(self):
        ln = FusedLayerNorm(32)
        x = np.random.randn(1, 5, 32).astype(np.float32)
        out = ln(x)
        assert out.shape == x.shape


class TestFusedRMSNorm:
    def test_shape_preserved(self):
        norm = FusedRMSNorm(64)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        out = norm(x)
        assert out.shape == x.shape

    def test_rms_near_one(self):
        norm = FusedRMSNorm(64)
        x = np.ones((4, 8, 64), dtype=np.float32) * 3.0
        out = norm(x)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-4)


class TestFusedCrossEntropyLoss:
    def test_basic(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.array([[1.0, 2.0, 0.5], [0.1, 0.2, 0.3]], dtype=np.float32)
        targets = np.array([1, 2])
        loss = loss_fn(logits, targets)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_ignore_index(self):
        loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
        logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        targets = np.array([-100])
        loss = loss_fn(logits, targets)
        assert loss == 0.0

    def test_label_smoothing(self):
        loss_fn_smooth = FusedCrossEntropyLoss(label_smoothing=0.1)
        loss_fn_no = FusedCrossEntropyLoss(label_smoothing=0.0)
        logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        targets = np.array([1])
        l_smooth = loss_fn_smooth(logits, targets)
        l_no = loss_fn_no(logits, targets)
        assert l_smooth != l_no


class TestFusedAttentionBias:
    def test_shape(self):
        ab = FusedAttentionBias(num_heads=4)
        q = np.random.randn(1, 4, 8, 16).astype(np.float32)
        k = np.random.randn(1, 4, 8, 16).astype(np.float32)
        v = np.random.randn(1, 4, 8, 16).astype(np.float32)
        out, weights = ab(q, k, v)
        assert out.shape == q.shape
        assert weights.shape[0] == 1
        assert weights.shape[-1] == weights.shape[-2]
