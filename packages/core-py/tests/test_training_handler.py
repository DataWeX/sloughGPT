"""Tests for training_strategies — utilities, samplers, gradient handlers, loss trackers, checkpoint savers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from domain.training._internal.training_handler import (
    AccumulationGradientHandler,
    BatchSampler,
    ChatPairSampler,
    CheckpointSaver,
    DirectGradientHandler,
    EMALossTracker,
    GradientHandler,
    LossTracker,
    NpzCheckpointSaver,
    PermutationSampler,
    RandomBlockSampler,
    RawLossTracker,
    SoulCheckpointSaver,
    build_char_vocab,
    clip_gradients,
    zero_grads,
)


# ── Helpers ────────────────────────────────────────────────────────────────


class _FakeParam:
    """Minimal parameter mock with .grad and .data."""

    def __init__(self, data: np.ndarray, grad: np.ndarray | None = None):
        self.data = data
        self.grad = grad


class _FakeModel:
    """Minimal model mock for gradient handler tests."""

    def __init__(self, params: list[_FakeParam] | None = None, names: list[str] | None = None):
        self._params = params or [_FakeParam(np.zeros(4))]
        self._names = names or ["weight"]

    def parameters(self):
        return self._params

    def named_parameters(self):
        return zip(self._names, self._params)

    def forward(self, x, targets=None):
        from domain.training._internal.slonet import tensor as slo_tensor

        x_t = slo_tensor(x, requires_grad=False)
        # simple mean loss
        loss_val = x_t.mean()
        return x_t, loss_val


class _FakeOptimizer:
    def __init__(self):
        self.step_count = 0
        self.last_params = None

    def step(self, params):
        self.step_count += 1
        self.last_params = params


# ── Utilities ──────────────────────────────────────────────────────────────


class TestClipGradients:
    def test_empty_params(self):
        assert clip_gradients([], 1.0) == 0.0

    def test_no_grads(self):
        params = [_FakeParam(np.zeros(4), grad=None)]
        assert clip_gradients(params, 1.0) == 0.0

    def test_no_clipping_needed(self):
        params = [_FakeParam(np.zeros(4), grad=np.array([0.1, 0.1]))]
        norm = clip_gradients(params, 10.0)
        assert norm < 1.0
        np.testing.assert_allclose(params[0].grad, [0.1, 0.1])

    def test_clipping_applied(self):
        params = [_FakeParam(np.zeros(4), grad=np.array([10.0, 10.0]))]
        norm = clip_gradients(params, 1.0)
        assert norm > 1.0
        assert np.max(np.abs(params[0].grad)) < 1.0

    def test_zero_max_norm(self):
        params = [_FakeParam(np.zeros(4), grad=np.array([1.0, 1.0]))]
        norm = clip_gradients(params, 0.0)
        assert norm == 0.0


class TestZeroGrads:
    def test_sets_grad_to_none(self):
        params = [_FakeParam(np.zeros(4), grad=np.ones(4))]
        zero_grads(params)
        assert params[0].grad is None

    def test_empty_params(self):
        zero_grads([])  # should not raise


class TestBuildCharVocab:
    def test_basic(self):
        stoi, itos = build_char_vocab(["hello", "world"])
        assert "h" in stoi
        assert "w" in stoi
        assert len(stoi) == len(itos)

    def test_special_tokens(self):
        stoi, itos = build_char_vocab(
            ["ab"], special_tokens={"<pad>": 0, "<bos>": 1}, offset=2
        )
        assert stoi["<pad>"] == 0
        assert stoi["<bos>"] == 1
        assert stoi["a"] >= 2
        assert itos[0] == "<pad>"
        assert itos[1] == "<bos>"

    def test_sorted_chars(self):
        stoi, _ = build_char_vocab(["ba"])
        assert stoi["a"] < stoi["b"]

    def test_empty_texts(self):
        stoi, itos = build_char_vocab([])
        assert len(stoi) == 0
        assert len(itos) == 0


# ── BatchSampler implementations ──────────────────────────────────────────


class TestRandomBlockSampler:
    def test_len(self):
        data = np.arange(100)
        sampler = RandomBlockSampler(data, block_size=10)
        assert len(sampler) == 90

    def test_get_batch_shapes(self):
        data = np.arange(100)
        sampler = RandomBlockSampler(data, block_size=10)
        x, y = sampler.get_batch(4)
        assert x.shape == (4, 10)
        assert y.shape == (4, 10)

    def test_y_is_x_shifted(self):
        data = np.arange(100)
        sampler = RandomBlockSampler(data, block_size=10)
        x, y = sampler.get_batch(1)
        np.testing.assert_array_equal(y[0, :-1], x[0, 1:])

    def test_too_small_raises(self):
        data = np.arange(5)
        sampler = RandomBlockSampler(data, block_size=10)
        with pytest.raises(ValueError, match="too small"):
            sampler.get_batch(1)


class TestPermutationSampler:
    def test_len(self):
        x = np.zeros((20, 8))
        y = np.zeros((20, 8))
        sampler = PermutationSampler(x, y)
        assert len(sampler) == 20

    def test_get_batch(self):
        x = np.arange(40).reshape(10, 4)
        y = np.arange(40, 80).reshape(10, 4)
        sampler = PermutationSampler(x, y)
        xb, yb = sampler.get_batch(3)
        assert xb.shape == (3, 4)
        assert yb.shape == (3, 4)


class TestChatPairSampler:
    def test_len(self):
        pairs = [{"user_msg": "hi", "assistant_msg": "hello"}]
        stoi = {"h": 1, "i": 2, "e": 3, "l": 4, "o": 5, " ": 6, "\n": 7, "\x00": 0,
                "U": 8, "s": 9, "e": 3, "r": 10, ":": 11, "A": 12, "a": 13}
        sampler = ChatPairSampler(pairs, stoi, block_size=4)
        assert len(sampler) > 0

    def test_get_batch(self):
        pairs = [
            {"user_msg": "hello world", "assistant_msg": "hi there"},
            {"user_msg": "test", "assistant_msg": "ok"},
        ]
        stoi, _ = build_char_vocab(
            [p["user_msg"] + p["assistant_msg"] for p in pairs],
            special_tokens={"\x00": 0},
            offset=1,
        )
        sampler = ChatPairSampler(pairs, stoi, block_size=8, seed=42)
        x, y = sampler.get_batch(3)
        assert x.shape == (3, 8)
        assert y.shape == (3, 8)

    def test_reproducible(self):
        pairs = [{"user_msg": "abc", "assistant_msg": "def"}]
        stoi, _ = build_char_vocab(["abcdef"], special_tokens={"\x00": 0}, offset=1)
        s1 = ChatPairSampler(pairs, stoi, block_size=4, seed=42)
        s2 = ChatPairSampler(pairs, stoi, block_size=4, seed=42)
        x1, _ = s1.get_batch(2)
        x2, _ = s2.get_batch(2)
        np.testing.assert_array_equal(x1, x2)


# ── GradientHandler implementations ────────────────────────────────────────


class TestDirectGradientHandler:
    def test_step(self):
        handler = DirectGradientHandler(grad_clip=1.0)
        model = _FakeModel([_FakeParam(np.zeros(4), grad=np.ones(4))])
        opt = _FakeOptimizer()
        from domain.training._internal.slonet import tensor as slo_tensor

        loss = slo_tensor(1.0)
        metrics = handler.step(model, loss, opt)
        assert "loss" in metrics
        assert "grad_norm" in metrics
        assert opt.step_count == 1

    def test_zero_grads(self):
        handler = DirectGradientHandler()
        model = _FakeModel([_FakeParam(np.zeros(4), grad=np.ones(4))])
        handler.zero_grads(model)
        assert model.parameters()[0].grad is None


class TestAccumulationGradientHandler:
    def test_accumulates_before_step(self):
        handler = AccumulationGradientHandler(grad_clip=1.0, accum_steps=3)
        model = _FakeModel([_FakeParam(np.zeros(4), grad=np.ones(4))])
        opt = _FakeOptimizer()
        from domain.training._internal.slonet import tensor as slo_tensor

        for _ in range(2):
            loss = slo_tensor(1.0)
            handler.step(model, loss, opt)
        assert opt.step_count == 0  # hasn't stepped yet

        loss = slo_tensor(1.0)
        handler.step(model, loss, opt)
        assert opt.step_count == 1  # now it stepped


# ── LossTracker implementations ────────────────────────────────────────────


class TestRawLossTracker:
    def test_empty(self):
        tracker = RawLossTracker()
        assert tracker.current == 0.0

    def test_running_average(self):
        tracker = RawLossTracker()
        tracker.update(1.0)
        assert tracker.current == 1.0
        tracker.update(3.0)
        assert tracker.current == 2.0
        tracker.update(5.0)
        assert tracker.current == 3.0


class TestEMALossTracker:
    def test_empty(self):
        tracker = EMALossTracker()
        assert tracker.current == 0.0

    def test_first_value(self):
        tracker = EMALossTracker(alpha=0.5)
        val = tracker.update(10.0)
        assert val == 10.0

    def test_converges(self):
        tracker = EMALossTracker(alpha=0.1)
        for _ in range(100):
            tracker.update(5.0)
        assert abs(tracker.current - 5.0) < 0.01


# ── CheckpointSaver implementations ────────────────────────────────────────


class TestNpzCheckpointSaver:
    def test_save_and_load(self, tmp_path):
        saver = NpzCheckpointSaver(checkpoint_dir=str(tmp_path))
        model = _FakeModel([_FakeParam(np.array([1.0, 2.0, 3.0, 4.0]))])
        path = saver.save(model, "test")
        assert Path(path).exists()

        model2 = _FakeModel([_FakeParam(np.zeros(4))])
        assert saver.load(path, model2)
        np.testing.assert_allclose(model2.parameters()[0].data, [1.0, 2.0, 3.0, 4.0])

    def test_load_nonexistent(self, tmp_path):
        saver = NpzCheckpointSaver(checkpoint_dir=str(tmp_path))
        model = _FakeModel()
        assert not saver.load(str(tmp_path / "missing.npz"), model)

    def test_latest_path(self, tmp_path):
        saver = NpzCheckpointSaver(checkpoint_dir=str(tmp_path))
        model = _FakeModel()
        saver.save(model, "a")
        saver.save(model, "b")
        assert saver.latest_path() is not None
        assert "b" in saver.latest_path()

    def test_filter_fn(self, tmp_path):
        saver = NpzCheckpointSaver(
            checkpoint_dir=str(tmp_path),
            filter_fn=lambda name: "lora" in name,
        )
        # model with named params
        class _NamedModel:
            def __init__(self):
                self.lora_A = _FakeParam(np.ones(4))
                self.lora_B = _FakeParam(np.zeros(4))
                self.embed = _FakeParam(np.full(4, 99.0))

            def named_parameters(self):
                return [("lora_A", self.lora_A), ("lora_B", self.lora_B), ("embed", self.embed)]

        model = _NamedModel()
        path = saver.save(model, "lora_only")
        data = np.load(path)
        assert "lora_A" in data
        assert "lora_B" in data
        assert "embed" not in data
