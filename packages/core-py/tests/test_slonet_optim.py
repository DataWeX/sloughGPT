"""Tests for training.slonet — optimizers (SloSGD, SloAdam, SloAdamW)."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    SloSGD,
    SloAdam,
    SloAdamW,
    clip_grad_norm_,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_params(shapes=None):
    """Create a list of parameters with gradients."""
    if shapes is None:
        shapes = [(3, 4), (2,), (5, 5)]
    params = []
    for shape in shapes:
        p = Tensor(np.ones(shape), requires_grad=True)
        p.grad = Tensor(np.ones(shape) * 0.1)
        params.append(p)
    return params


# ── SloSGD ──────────────────────────────────────────────────────────────────


class TestSloSGD:

    def test_init(self):
        opt = SloSGD(lr=0.01, momentum=0.9)
        assert opt.lr == 0.01
        assert opt.momentum == 0.9

    def test_step(self):
        opt = SloSGD(lr=0.1)
        params = _make_params()
        old_data = [p.data.copy() for p in params]
        opt.step(params)
        for p, old in zip(params, old_data):
            assert not np.allclose(p.data, old)

    def test_step_no_grad(self):
        opt = SloSGD(lr=0.1)
        params = _make_params()
        params[0].grad = None
        old = params[0].data.copy()
        opt.step(params)
        assert np.allclose(params[0].data, old)

    def test_momentum(self):
        opt = SloSGD(lr=0.1, momentum=0.9)
        params = _make_params()
        opt.step(params)
        opt.step(params)
        # After two steps with momentum, data should have changed
        assert params[0].data.mean() < 1.0

    def test_grad_cleared(self):
        opt = SloSGD(lr=0.1)
        params = _make_params()
        opt.step(params)
        for p in params:
            assert p.grad is None

    def test_max_grad_norm(self):
        opt = SloSGD(lr=0.1, max_grad_norm=1.0)
        params = _make_params()
        opt.step(params)
        assert params[0].data.mean() < 1.0


# ── SloAdam ─────────────────────────────────────────────────────────────────


class TestSloAdam:

    def test_init(self):
        opt = SloAdam(lr=0.001, b1=0.9, b2=0.999)
        assert opt.lr == 0.001
        assert opt.b1 == 0.9
        assert opt.b2 == 0.999

    def test_step(self):
        opt = SloAdam(lr=0.01)
        params = _make_params()
        old_data = [p.data.copy() for p in params]
        opt.step(params)
        for p, old in zip(params, old_data):
            assert not np.allclose(p.data, old)

    def test_weight_decay(self):
        opt = SloAdam(lr=0.01, weight_decay=0.1)
        params = _make_params()
        old_data = [p.data.copy() for p in params]
        opt.step(params)
        for p, old in zip(params, old_data):
            # With weight decay, values should decrease
            assert p.data.mean() < old.mean()

    def test_timestep(self):
        opt = SloAdam(lr=0.01)
        params = _make_params()
        assert opt._t == 0
        opt.step(params)
        assert opt._t == 1
        opt.step(params)
        assert opt._t == 2

    def test_grad_cleared(self):
        opt = SloAdam(lr=0.01)
        params = _make_params()
        opt.step(params)
        for p in params:
            assert p.grad is None

    def test_state_dict(self):
        opt = SloAdam(lr=0.01)
        params = _make_params()
        opt.step(params)
        state = opt.state_dict(params)
        assert "hyperparameters" in state
        assert "state" in state
        assert state["t"] == 1

    def test_load_state_dict(self):
        opt1 = SloAdam(lr=0.01)
        params1 = _make_params()
        opt1.step(params1)
        state = opt1.state_dict(params1)

        opt2 = SloAdam(lr=0.01)
        params2 = _make_params()
        opt2.load_state_dict(state, params2)
        assert opt2._t == 1

    def test_max_grad_norm(self):
        opt = SloAdam(lr=0.01, max_grad_norm=1.0)
        params = _make_params()
        opt.step(params)
        assert params[0].data.mean() < 1.0


# ── SloAdamW ────────────────────────────────────────────────────────────────


class TestSloAdamW:

    def test_init(self):
        opt = SloAdamW(lr=0.001, weight_decay=0.01)
        assert opt.lr == 0.001
        assert opt.weight_decay == 0.01
        assert opt.amsgrad is False
        assert opt.maximize is False

    def test_step(self):
        opt = SloAdamW(lr=0.01, weight_decay=0.01)
        params = _make_params()
        old_data = [p.data.copy() for p in params]
        opt.step(params)
        for p, old in zip(params, old_data):
            assert not np.allclose(p.data, old)

    def test_weight_decay_decoupled(self):
        opt = SloAdamW(lr=0.01, weight_decay=0.5)
        params = _make_params([(3,)])
        old = params[0].data.copy()
        opt.step(params)
        # Decoupled weight decay: p -= lr * wd * p
        expected = old - 0.01 * 0.5 * old
        assert params[0].data.mean() < old.mean()

    def test_amsgrad(self):
        opt = SloAdamW(lr=0.01, amsgrad=True)
        params = _make_params()
        opt.step(params)
        assert len(opt._vmax) == len(params)

    def test_maximize(self):
        opt = SloAdamW(lr=0.01, maximize=True)
        params = _make_params()
        old_data = [p.data.copy() for p in params]
        opt.step(params)
        # With maximize, gradient is inverted, so values should increase
        for p, old in zip(params, old_data):
            assert p.data.mean() > old.mean()

    def test_state_dict_amsgrad(self):
        opt = SloAdamW(lr=0.01, amsgrad=True)
        params = _make_params()
        opt.step(params)
        state = opt.state_dict(params)
        assert state["hyperparameters"]["amsgrad"] is True

    def test_load_state_dict(self):
        opt1 = SloAdamW(lr=0.01, amsgrad=True)
        params1 = _make_params()
        opt1.step(params1)
        state = opt1.state_dict(params1)

        opt2 = SloAdamW(lr=0.01)
        params2 = _make_params()
        opt2.load_state_dict(state, params2)
        assert opt2.amsgrad is True
        assert opt2._t == 1


# ── clip_grad_norm_ ─────────────────────────────────────────────────────────


class TestClipGradNorm:

    def test_no_clipping(self):
        params = _make_params()
        clip_grad_norm_(params, max_norm=100.0)
        for p in params:
            assert p.grad is not None

    def test_clipping(self):
        params = _make_params()
        for p in params:
            p.grad = Tensor(np.ones_like(p.data) * 10.0)
        clip_grad_norm_(params, max_norm=1.0)
        total_norm = sum(np.linalg.norm(p.grad.data) ** 2 for p in params) ** 0.5
        assert total_norm <= 1.0 + 1e-5

    def test_empty_params(self):
        clip_grad_norm_([], max_norm=1.0)
