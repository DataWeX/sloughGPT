"""Tests for forward_pass.py — unified forward pass interface."""

import numpy as np
from domains.inference.forward_pass import (
    ForwardPassResult,
    ForwardPassable,
    timed_forward,
)


class _FakeModel:
    """Minimal ForwardPassable implementation."""

    def forward_pass(self, input_ids: np.ndarray) -> ForwardPassResult:
        return ForwardPassResult(logits=np.zeros((1, 4, 5)))


class _NotForwardPassable:
    pass


class TestForwardPassResult:

    def test_defaults(self):
        logits = np.zeros((1, 2, 3))
        result = ForwardPassResult(logits=logits)
        assert result.forward_time_ms == 0.0
        assert result.model_name == ""
        assert result.cached_tokens == 0
        assert result.engine == "unknown"

    def test_shape_property(self):
        result = ForwardPassResult(logits=np.zeros((1, 8, 16)))
        assert result.shape == [1, 8, 16]

    def test_shape_matches_logits_ndim(self):
        logits = np.random.randn(3, 5, 7)
        result = ForwardPassResult(logits=logits)
        assert result.shape == list(logits.shape)


class TestForwardPassable:

    def test_runtime_checkable_positive(self):
        assert isinstance(_FakeModel(), ForwardPassable)

    def test_runtime_checkable_negative(self):
        assert not isinstance(_NotForwardPassable(), ForwardPassable)

    def test_protocol_method_signature(self):
        sig = ForwardPassable.forward_pass
        assert callable(sig)


class TestTimedForward:

    def test_sets_model_name(self):
        fake = _FakeModel()
        result = timed_forward(fake, np.zeros((1, 4), dtype=np.int64), model_name="gpt2")
        assert result.model_name == "gpt2"

    def test_measures_elapsed_time(self):
        fake = _FakeModel()
        result = timed_forward(fake, np.zeros((1, 4), dtype=np.int64))
        assert result.forward_time_ms >= 0.0

    def test_returns_underlying_result(self):
        fake = _FakeModel()
        result = timed_forward(fake, np.zeros((1, 4), dtype=np.int64))
        assert isinstance(result, ForwardPassResult)
        assert result.engine == "unknown"

    def test_empty_model_name_default(self):
        fake = _FakeModel()
        result = timed_forward(fake, np.zeros((1, 4), dtype=np.int64))
        assert result.model_name == ""
