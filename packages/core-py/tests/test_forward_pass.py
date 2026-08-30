"""Tests for domains.inference.forward_pass — ForwardPassResult, ForwardPassable, timed_forward.

Covers: dataclass fields, shape property, protocol compliance, timed wrapper, edge cases.
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

    def test_logits_preserved(self):
        logits = np.ones((1, 3, 50)) * 42.0
        r = ForwardPassResult(logits=logits)
        assert np.allclose(r.logits, 42.0)

    def test_shape_batch_1(self):
        logits = np.zeros((1, 1, 1000))
        r = ForwardPassResult(logits=logits)
        assert r.shape == [1, 1, 1000]

    def test_shape_large_batch(self):
        logits = np.zeros((32, 128, 50000))
        r = ForwardPassResult(logits=logits)
        assert r.shape == [32, 128, 50000]

    def test_shape_single_token(self):
        logits = np.zeros((1, 1, 100))
        r = ForwardPassResult(logits=logits)
        assert r.shape[1] == 1

    def test_cached_tokens_default(self):
        r = ForwardPassResult(logits=np.zeros((1, 1, 10)))
        assert r.cached_tokens == 0

    def test_cached_tokens_custom(self):
        r = ForwardPassResult(logits=np.zeros((1, 1, 10)), cached_tokens=50)
        assert r.cached_tokens == 50

    def test_engine_numpy(self):
        r = ForwardPassResult(logits=np.zeros((1, 1, 10)), engine="numpy")
        assert r.engine == "numpy"

    def test_engine_c(self):
        r = ForwardPassResult(logits=np.zeros((1, 1, 10)), engine="c")
        assert r.engine == "c"

    def test_forward_time_ms_large(self):
        r = ForwardPassResult(logits=np.zeros((1, 1, 10)), forward_time_ms=9999.99)
        assert r.forward_time_ms == 9999.99

    def test_model_name_long(self):
        name = "a" * 1000
        r = ForwardPassResult(logits=np.zeros((1, 1, 10)), model_name=name)
        assert len(r.model_name) == 1000

    def test_shape_3d_required(self):
        logits = np.zeros((1, 10, 100))
        r = ForwardPassResult(logits=logits)
        assert len(r.shape) == 3

    def test_shape_returns_list(self):
        r = ForwardPassResult(logits=np.zeros((1, 2, 3)))
        assert isinstance(r.shape, list)

    def test_logits_is_ndarray(self):
        r = ForwardPassResult(logits=np.zeros((1, 1, 10)))
        assert isinstance(r.logits, np.ndarray)

    def test_default_engine_unknown(self):
        r = ForwardPassResult(logits=np.zeros((1, 1, 10)))
        assert r.engine == "unknown"

    def test_equality(self):
        logits = np.zeros((1, 1, 10))
        a = ForwardPassResult(logits=logits)
        b = ForwardPassResult(logits=logits)
        assert a == b


class MockForwardPass:
    def forward_pass(self, input_ids: np.ndarray) -> ForwardPassResult:
        batch, seq = input_ids.shape
        vocab = 100
        logits = np.zeros((batch, seq, vocab))
        return ForwardPassResult(logits=logits, engine="numpy")


class MockNoForwardPass:
    pass


class MockWrongSignature:
    def forward_pass(self):
        return None


class TestForwardPassable:
    def test_protocol_compliance(self):
        mock = MockForwardPass()
        assert isinstance(mock, ForwardPassable)

    def test_protocol_rejects_non_conforming(self):
        class Bad:
            pass
        assert not isinstance(Bad(), ForwardPassable)

    def test_rejects_no_forward_pass(self):
        mock = MockNoForwardPass()
        assert not isinstance(mock, ForwardPassable)

    def test_wrong_signature_still_passes_runtime_check(self):
        mock = MockWrongSignature()
        assert isinstance(mock, ForwardPassable)

    def test_mock_returns_result(self):
        mock = MockForwardPass()
        input_ids = np.array([[1, 2, 3]])
        result = mock.forward_pass(input_ids)
        assert isinstance(result, ForwardPassResult)

    def test_mock_shape_correct(self):
        mock = MockForwardPass()
        input_ids = np.array([[1, 2, 3]])
        result = mock.forward_pass(input_ids)
        assert result.logits.shape == (1, 3, 100)

    def test_mock_engine(self):
        mock = MockForwardPass()
        result = mock.forward_pass(np.array([[1]]))
        assert result.engine == "numpy"


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

    def test_timing_overwrite(self):
        model = MockForwardPass()
        input_ids = np.array([[1]])
        result = timed_forward(model, input_ids, model_name="m")
        assert result.model_name == "m"

    def test_timing_default_name(self):
        model = MockForwardPass()
        input_ids = np.array([[1]])
        result = timed_forward(model, input_ids)
        assert result.model_name == ""

    def test_batch_timing(self):
        model = MockForwardPass()
        input_ids = np.array([[1, 2], [3, 4]])
        result = timed_forward(model, input_ids)
        assert result.logits.shape == (2, 2, 100)
        assert result.forward_time_ms >= 0

    def test_large_input_timing(self):
        model = MockForwardPass()
        input_ids = np.array([[i for i in range(128)]])
        result = timed_forward(model, input_ids)
        assert result.forward_time_ms >= 0

    def test_single_token_timing(self):
        model = MockForwardPass()
        input_ids = np.array([[1]])
        result = timed_forward(model, input_ids)
        assert result.logits.shape == (1, 1, 100)

    def test_engine_preserved(self):
        model = MockForwardPass()
        input_ids = np.array([[1]])
        result = timed_forward(model, input_ids)
        assert result.engine == "numpy"

    def test_model_name_custom(self):
        model = MockForwardPass()
        input_ids = np.array([[1, 2]])
        result = timed_forward(model, input_ids, model_name="custom")
        assert result.model_name == "custom"
