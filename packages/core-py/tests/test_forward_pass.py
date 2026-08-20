"""Tests for domains.inference.forward_pass — ForwardPassResult, ForwardPassable, timed_forward.

Covers: dataclass fields, shape property, protocol compliance, timed wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.inference.forward_pass import (
    ForwardPassResult,
    ForwardPassable,
    timed_forward,
)


class TestForwardPassResult:
    def test_creation(self):
        logits = np.zeros((1, 10, 100))
        r = ForwardPassResult(logits=logits)
        assert r.logits.shape == (1, 10, 100)
        assert r.forward_time_ms == 0.0
        assert r.model_name == ""
        assert r.cached_tokens == 0
        assert r.engine == "unknown"

    def test_shape_property(self):
        logits = np.zeros((2, 5, 50))
        r = ForwardPassResult(logits=logits)
        assert r.shape == [2, 5, 50]

    def test_custom_fields(self):
        logits = np.zeros((1, 1, 10))
        r = ForwardPassResult(logits=logits, forward_time_ms=12.5, model_name="test", engine="numpy")
        assert r.forward_time_ms == 12.5
        assert r.model_name == "test"
        assert r.engine == "numpy"


class MockForwardPass:
    def forward_pass(self, input_ids: np.ndarray) -> ForwardPassResult:
        batch, seq = input_ids.shape
        vocab = 100
        logits = np.zeros((batch, seq, vocab))
        return ForwardPassResult(logits=logits, engine="numpy")


class TestForwardPassable:
    def test_protocol_compliance(self):
        mock = MockForwardPass()
        assert isinstance(mock, ForwardPassable)

    def test_protocol_rejects_non_conforming(self):
        class Bad:
            pass
        assert not isinstance(Bad(), ForwardPassable)


class TestTimedForward:
    def test_basic(self):
        model = MockForwardPass()
        input_ids = np.array([[1, 2, 3]])
        result = timed_forward(model, input_ids, model_name="test")
        assert result.forward_time_ms >= 0
        assert result.model_name == "test"
        assert result.logits.shape == (1, 3, 100)

    def test_timing_positive(self):
        model = MockForwardPass()
        input_ids = np.array([[1]])
        result = timed_forward(model, input_ids)
        assert result.forward_time_ms >= 0
