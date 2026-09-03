"""Tests for ForwardPassResult and timed_forward."""
from __future__ import annotations

import numpy as np

from domains.inference.forward_pass import ForwardPassResult, timed_forward


class TestForwardPassResult:
    def test_shape(self):
        logits = np.random.randn(1, 10, 100).astype(np.float32)
        r = ForwardPassResult(logits=logits)
        assert r.shape == [1, 10, 100]

    def test_defaults(self):
        r = ForwardPassResult(logits=np.zeros((1, 1, 10), dtype=np.float32))
        assert r.forward_time_ms == 0.0
        assert r.model_name == ""
        assert r.cached_tokens == 0
        assert r.engine == "unknown"


class TestTimedForward:
    def test_sets_time(self):
        class DummyModel:
            def forward_pass(self, input_ids):
                return ForwardPassResult(logits=np.zeros((1, 1, 10), dtype=np.float32))

        r = timed_forward(DummyModel(), np.zeros((1, 1), dtype=np.int64), model_name="test")
        assert r.forward_time_ms >= 0
        assert r.model_name == "test"

    def test_satisfies_protocol(self):
        class GoodModel:
            def forward_pass(self, input_ids):
                return ForwardPassResult(logits=np.zeros((1, 1, 5), dtype=np.float32))

        from domains.inference.forward_pass import ForwardPassable
        assert isinstance(GoodModel(), ForwardPassable)
