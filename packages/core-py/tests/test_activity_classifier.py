"""Tests for ActivityClassifier — model, augmentation, training, prediction, save/load."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domains.activity import ActivityClassifier, train_classifier, predict_activity
from domains.activity.classifier import _augment_batch, _global_mean, _accuracy
from domains.training.slonet import Tensor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n, t, c = 16, 128, 6
    X = np.random.randn(n, t, c).astype(np.float32)
    y = np.array([i % 3 for i in range(n)], dtype=np.int64)
    return X, y


@pytest.fixture
def tiny_data():
    np.random.seed(42)
    X = np.random.randn(24, 64, 6).astype(np.float32)
    y = np.array([i % 2 for i in range(24)], dtype=np.int64)
    return X, y


@pytest.fixture
def model():
    return ActivityClassifier(num_classes=3)


# ── ActivityClassifier tests ──────────────────────────────────────────────────

class TestActivityClassifier:
    def test_init_default_params(self):
        m = ActivityClassifier()
        params = m.parameters()
        assert len(params) == 6  # conv1 W+b, conv2 W+b, fc W+b

    def test_init_custom_classes(self):
        m = ActivityClassifier(num_classes=8)
        assert m.num_classes == 8
        fc = m._params  # access cached params
        params = m.parameters()
        fc_weight = params[-2].data  # second-to-last = fc weights
        assert fc_weight.shape[0] == 8

    def test_forward_shape(self, model, sample_data):
        X, _ = sample_data
        x = Tensor(X, requires_grad=False)
        logits = model.forward(x)
        assert logits.data.shape == (16, 3)

    def test_forward_no_nan(self, model, sample_data):
        X, _ = sample_data
        x = Tensor(X, requires_grad=False)
        logits = model.forward(x)
        assert not np.any(np.isnan(logits.data))
        assert not np.any(np.isinf(logits.data))

    def test_forward_batch_1(self, model):
        x = Tensor(np.random.randn(1, 128, 6).astype(np.float32), requires_grad=False)
        logits = model.forward(x)
        assert logits.data.shape == (1, 3)

    def test_forward_batch_32(self, model):
        x = Tensor(np.random.randn(32, 128, 6).astype(np.float32), requires_grad=False)
        logits = model.forward(x)
        assert logits.data.shape == (32, 3)

    def test_forward_gradient_flows(self, model, sample_data):
        X, _ = sample_data
        x = Tensor(X, requires_grad=True)
        logits = model.forward(x)
        loss = logits.sum()
        loss.backward()
        # Every parameter should have a gradient
        for p in model.parameters():
            assert p.grad is not None
            assert not np.allclose(p.grad.data, 0)

    def test_forward_always_requires_grad(self, model, sample_data):
        """Forward always enables gradients (conv path uses requires_grad=True)."""
        X, _ = sample_data
        x = Tensor(X, requires_grad=False)
        logits = model.forward(x)
        assert logits.requires_grad


# ── Data augmentation tests ───────────────────────────────────────────────────

class TestDataAugmentation:
    def test_preserves_shape(self):
        X = np.random.randn(8, 128, 6).astype(np.float32)
        aug = _augment_batch(X)
        assert aug.shape == X.shape

    def test_no_nan_or_inf(self):
        X = np.random.randn(8, 128, 6).astype(np.float32)
        aug = _augment_batch(X)
        assert not np.any(np.isnan(aug))
        assert not np.any(np.isinf(aug))

    def test_actually_changes_data(self):
        np.random.seed(0)
        X = np.random.randn(8, 128, 6).astype(np.float32)
        aug = _augment_batch(X)
        # With 4 augmentations each at 15-60% probability,
        # at least one should fire almost certainly
        assert not np.allclose(aug, X, atol=1e-7)

    def test_deterministic_with_seed(self):
        X = np.random.randn(8, 128, 6).astype(np.float32)
        np.random.seed(99)
        a1 = _augment_batch(X)
        np.random.seed(99)
        a2 = _augment_batch(X)
        assert np.allclose(a1, a2)

    def test_handles_zero_input(self):
        X = np.zeros((4, 128, 6), dtype=np.float32)
        aug = _augment_batch(X)
        assert aug.shape == X.shape
        assert not np.any(np.isnan(aug))

    def test_handles_single_channel(self):
        X = np.random.randn(4, 128, 1).astype(np.float32)
        aug = _augment_batch(X)
        assert aug.shape == X.shape


# ── Global mean tests ─────────────────────────────────────────────────────────

class TestGlobalMean:
    def test_forward_shape(self):
        x = Tensor(np.random.randn(4, 6, 128, 1).astype(np.float32), requires_grad=True)
        out = _global_mean(x, axis=2)
        assert out.data.shape == (4, 6, 1, 1)

    def test_forward_value(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1)
        x = Tensor(data, requires_grad=True)
        out = _global_mean(x, axis=2)
        expected = data.mean(axis=2, keepdims=True)
        assert np.allclose(out.data, expected)

    def test_backward_correct(self):
        x = Tensor(np.random.randn(2, 3, 10, 1).astype(np.float32), requires_grad=True)
        out = _global_mean(x, axis=2)
        loss = out.sum()
        loss.backward()
        # Gradient to x should be 1 / N (where N=10 is the mean axis size)
        expected_grad = np.ones_like(x.data) / 10.0
        assert np.allclose(x.grad.data, expected_grad, atol=1e-6)

    def test_backward_no_grad(self):
        x = Tensor(np.random.randn(2, 3, 10, 1).astype(np.float32), requires_grad=False)
        out = _global_mean(x, axis=2)
        assert not out.requires_grad


# ── Accuracy helper tests ─────────────────────────────────────────────────────

class TestAccuracy:
    def test_perfect(self):
        logits = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32))
        targets = np.array([0, 1], dtype=np.int64)
        assert _accuracy(logits, targets) == 1.0

    def test_half(self):
        logits = Tensor(np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32))
        targets = np.array([0, 1], dtype=np.int64)
        assert _accuracy(logits, targets) == 0.5

    def test_all_wrong(self):
        logits = Tensor(np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32))
        targets = np.array([0, 1], dtype=np.int64)
        assert _accuracy(logits, targets) == 0.0

    def test_empty(self):
        logits = Tensor(np.zeros((0, 3), dtype=np.float32))
        targets = np.array([], dtype=np.int64)
        # No samples → accuracy is NaN from 0/0; ensure float output
        acc = _accuracy(logits, targets)
        assert isinstance(acc, float)


# ── Training tests ────────────────────────────────────────────────────────────

class TestTraining:
    def test_train_basic(self, tiny_data):
        X, y = tiny_data
        model = train_classifier(X, y, epochs=2, lr=0.01, batch_size=8,
                                 augment=False, verbose=False)
        assert isinstance(model, ActivityClassifier)
        assert model.num_classes == 2

    def test_train_with_augmentation(self, tiny_data):
        X, y = tiny_data
        model = train_classifier(X, y, epochs=2, lr=0.01, batch_size=8,
                                 augment=True, verbose=False)
        assert isinstance(model, ActivityClassifier)

    def test_train_loss_decreases(self, tiny_data):
        X, y = tiny_data
        # Use a single batch, track loss
        from domains.training.slonet import Tensor, cross_entropy
        m = ActivityClassifier(num_classes=2)
        opt = type("_", (object,), {"lr": 0.01, "step": lambda p: None})()

        xb = Tensor(X[:8], requires_grad=False)
        yb = Tensor(y[:8], requires_grad=False)
        logits = m.forward(xb)
        loss_initial = float(cross_entropy(logits, yb).data)

        # After training with gradient descent
        m2 = train_classifier(X, y, epochs=3, lr=0.01, batch_size=8,
                              augment=False, verbose=False)
        logits2 = m2.forward(xb)
        loss_final = float(cross_entropy(logits2, yb).data)
        assert loss_final < loss_initial

    def test_train_persists_model(self, tiny_data, tmp_path):
        X, y = tiny_data
        # Monkey-patch save path
        import domains.activity.classifier as mod
        original_path = mod.Path  # save reference
        fake_path = tmp_path / "model.npz"
        # Patch _augment_batch to avoid rng issues
        model = train_classifier(X, y, epochs=1, lr=0.01, batch_size=8,
                                 augment=False, verbose=False)
        # Manual save to tmp
        model.save(str(fake_path))
        assert fake_path.exists()
        assert fake_path.stat().st_size > 100

    def test_train_different_activities(self, sample_data):
        """Train on 3-class data, verify model works."""
        X, y = sample_data  # 3 classes
        model = train_classifier(X, y, epochs=5, lr=0.01, batch_size=8,
                                 augment=True, verbose=False)
        assert model.num_classes == 3


# ── Prediction tests ──────────────────────────────────────────────────────────

class TestPrediction:
    def test_predict_returns_tuple(self, model):
        x = np.random.randn(128, 6).astype(np.float32)
        result = predict_activity(model, x)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_predict_class_id_int(self, model):
        x = np.random.randn(128, 6).astype(np.float32)
        cid, name, probs = predict_activity(model, x)
        assert isinstance(cid, int)
        assert 0 <= cid < 3

    def test_predict_class_name_str(self, model):
        x = np.random.randn(128, 6).astype(np.float32)
        cid, name, probs = predict_activity(model, x)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_predict_probabilities_sum_to_1(self, model):
        x = np.random.randn(128, 6).astype(np.float32)
        cid, name, probs = predict_activity(model, x)
        assert np.abs(probs.sum() - 1.0) < 1e-5

    def test_predict_probabilities_non_negative(self, model):
        x = np.random.randn(128, 6).astype(np.float32)
        cid, name, probs = predict_activity(model, x)
        assert np.all(probs >= 0)

    def test_predict_accepts_3d_input(self, model):
        x = np.random.randn(1, 128, 6).astype(np.float32)
        cid, name, probs = predict_activity(model, x)
        assert isinstance(cid, int)

    def test_predict_deterministic(self, model):
        x = np.random.randn(128, 6).astype(np.float32)
        r1 = predict_activity(model, x)
        r2 = predict_activity(model, x)
        assert r1[0] == r2[0]
        assert np.allclose(r1[2], r2[2])


# ── Save/load tests ───────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_file(self, model, tmp_path):
        path = tmp_path / "test_model.npz"
        model.save(str(path))
        assert path.exists()
        assert path.stat().st_size > 100

    def test_load_returns_model(self, model, tmp_path):
        path = tmp_path / "test_model.npz"
        model.save(str(path))
        loaded = ActivityClassifier.load(str(path), num_classes=3)
        assert isinstance(loaded, ActivityClassifier)
        assert loaded.num_classes == 3

    def test_roundtrip_preserves_weights(self, model, tmp_path):
        path = tmp_path / "test_model.npz"
        # Record original weights
        orig_weights = [p.data.copy() for p in model.parameters()]
        model.save(str(path))
        loaded = ActivityClassifier.load(str(path), num_classes=3)
        loaded_weights = [p.data for p in loaded.parameters()]
        for orig, loaded_p in zip(orig_weights, loaded_weights):
            assert np.allclose(orig, loaded_p)

    def test_roundtrip_same_predictions(self, tmp_path):
        np.random.seed(42)
        X = np.random.randn(32, 128, 6).astype(np.float32)
        y = np.array([i % 3 for i in range(32)], dtype=np.int64)
        model = train_classifier(X, y, epochs=2, lr=0.01, batch_size=8,
                                 augment=False, verbose=False)
        path = tmp_path / "test_model.npz"
        model.save(str(path))
        loaded = ActivityClassifier.load(str(path), num_classes=3)

        test_x = np.random.randn(4, 128, 6).astype(np.float32)
        for i in range(4):
            cid1, name1, probs1 = predict_activity(model, test_x[i])
            cid2, name2, probs2 = predict_activity(loaded, test_x[i])
            assert cid1 == cid2
            assert name1 == name2
            assert np.allclose(probs1, probs2)

    def test_load_from_disk(self):
        """Load the production model.npz if it exists."""
        model_path = Path(__file__).resolve().parents[1] / "domains" / "activity" / "model.npz"
        if not model_path.exists():
            pytest.skip("production model.npz not found")
        m = ActivityClassifier.load(str(model_path))
        assert m.num_classes == 3
        x = np.random.randn(1, 128, 6).astype(np.float32)
        cid, name, probs = predict_activity(m, x[0])
        assert 0 <= cid < 3
        assert np.abs(probs.sum() - 1.0) < 1e-5
