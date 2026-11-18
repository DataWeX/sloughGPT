"""Tests for training_strategies — utilities, samplers, gradient handlers, loss trackers, checkpoint savers."""

from __future__ import annotations

import numpy as np

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


# ── BatchSampler implementations ──────────────────────────────────────────


# ── GradientHandler implementations ────────────────────────────────────────


# ── LossTracker implementations ────────────────────────────────────────────


# ── CheckpointSaver implementations ────────────────────────────────────────
